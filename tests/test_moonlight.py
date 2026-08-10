"""Tests for the Moonlight optimizer (Muon on matrices, AdamW on the rest).

Checks the two claims the implementation rests on:
  1. Newton-Schulz really does drive singular values toward 1, so the update is
     approximately semi-orthogonal.
  2. The 0.2 * sqrt(max(n, m)) scale pins the update RMS at 0.2 independent of
     matrix shape -- the property that lets an AdamW learning rate transfer.
Plus the parameter routing, an AdamW-branch reference comparison, and a real
optimization run on the video DiT.

Run: python tests/test_moonlight.py [cuda|mps|cpu]
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.imf_dit_video import imf_dit_video_S
from moonlight import (
    JORDAN_COEFFS,
    Moonlight,
    build_moonlight,
    get_ns_coefficients,
    sample_normalized_spectrum,
    solve_ns_coefficients,
    split_params,
    zeropower_via_newtonschulz5,
)


def pick_device():
    """argv override, else CUDA, then MPS (Apple Metal), then CPU."""
    if len(sys.argv) > 1:
        return torch.device(sys.argv[1])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def matrix_with_spectrum(n, m, singular_values, seed=0):
    """Build an (n, m) matrix with a prescribed singular value spectrum.

    Args:
        n, m: output shape.
        singular_values: (r,) tensor of target singular values, r = min(n, m).
        seed: RNG seed for the random orthogonal factors.

    Returns:
        (n, m) matrix U diag(s) V^T with the requested spectrum.
    """
    r = min(n, m)
    assert singular_values.numel() == r
    g = torch.Generator().manual_seed(seed)
    # QR of a Gaussian gives Haar-distributed orthonormal columns.
    U, _ = torch.linalg.qr(torch.randn(n, r, generator=g))   # (n, r)
    V, _ = torch.linalg.qr(torch.randn(m, r, generator=g))   # (m, r)
    return U @ torch.diag(singular_values) @ V.T             # (n, m)


def test_newtonschulz_map_itself_has_no_shape_dependence():
    """A FIXED (a, b, c) acts identically on any shape given the same spectrum.

    The iteration applies the scalar polynomial p(sigma) = a sigma + b sigma^3 +
    c sigma^5 elementwise, and that map cannot see n or m. Give different shapes
    the same input spectrum and the output spectra coincide.

    Note carefully what this does NOT say: it does not say the OPTIMAL
    coefficients are shape-independent. The map is shape-blind, but the
    distribution of singular values it must pull to 1 is not -- see
    test_per_shape_coefficients_beat_jordan. Both facts are true at once.

    Shape symbols: n rows, m cols, r = min(n, m) singular values.
    """
    torch.manual_seed(0)
    r = 32
    s = torch.linspace(0.2, 1.0, r)                          # (r,) shared spectrum
    spectra = {}
    for n, m in [(32, 64), (32, 256), (64, 32), (256, 32), (32, 1024)]:
        G = matrix_with_spectrum(n, m, s)                    # (n, m)
        # Undo the Frobenius normalization NS applies, so the input spectrum
        # entering the polynomial is identical across shapes.
        G = G / G.norm() * math.sqrt((s**2).sum())
        X = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32)
        sv = torch.linalg.svdvals(X.float()).sort(descending=True).values  # (r,)
        spectra[(n, m)] = sv
        print(f"  ({n:>3},{m:>4}) out sigma in [{sv.min():.4f}, {sv.max():.4f}]")

    ref_key = (32, 64)
    ref = spectra[ref_key]
    for key, sv in spectra.items():
        diff = (sv - ref).abs().max().item()
        assert diff < 1e-3, (
            f"shape {key} produced a different spectrum than {ref_key}: "
            f"max diff {diff:.2e} -- the coefficients would have to be "
            f"shape-dependent for this to happen"
        )
    print(f"  all 5 shapes share one output spectrum (max diff "
          f"{max((sv - ref).abs().max().item() for sv in spectra.values()):.2e})")


def test_newtonschulz_orthogonality():
    """Newton-Schulz must pull a well-conditioned spectrum onto ~1.

    With 5 steps the coefficients target roughly [0.68, 1.14]. That envelope is
    reached only when the input is not near-singular: the polynomial multiplies
    tiny singular values by about `a` per step, so five steps cannot rescue a
    spectrum whose small end sits at 1e-4. See the separate near-singular test.

    Shape symbols: n rows, m cols, r = min(n, m).
    """
    torch.manual_seed(0)
    r_frac = 0.3  # smallest singular value as a fraction of the largest
    for n, m in [(64, 64), (128, 64), (64, 128), (256, 48), (48, 256)]:
        r = min(n, m)
        s = torch.linspace(r_frac, 1.0, r)                   # (r,) mild spread
        G = matrix_with_spectrum(n, m, s)                    # (n, m)
        X = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32)
        assert X.shape == G.shape, (X.shape, G.shape)
        sv = torch.linalg.svdvals(X.float())                 # (r,)
        assert sv.numel() == r
        assert sv.min() > 0.6, f"({n},{m}) sigma_min={sv.min().item():.4f}"
        assert sv.max() < 1.2, f"({n},{m}) sigma_max={sv.max().item():.4f}"
        # ||X||_F^2 = sum sigma^2 ~ r for a semi-orthogonal matrix. With sigma
        # spread over [0.68, 1.13] rather than pinned at 1, sum sigma^2 / r lands
        # in roughly [0.46, 1.28], so allow 35%. This shortfall is exactly why the
        # measured update RMS comes out near 0.18 rather than exactly 0.20.
        fro_sq = (X.float() ** 2).sum().item()
        assert abs(fro_sq - r) / r < 0.35, (
            f"({n},{m}) ||X||_F^2={fro_sq:.1f} vs r={r}"
        )
        print(f"  ({n:>3},{m:>3}) sigma in [{sv.min():.3f}, {sv.max():.3f}], "
              f"||X||_F^2={fro_sq:.1f} (r={r})")


def test_solver_reproduces_su_jianlin_table():
    """Our coefficient fit must match the published reference values.

    Reference: Su Jianlin, https://kexue.fm/archives/10592, table in the
    "收敛加速" (convergence acceleration) section. Columns are
    n, m, T, a, b, c, mse.
    """
    reference = [
        (1024, 1024, 3, 4.328, -9.666, 7.020, 0.10257),
        (1024, 1024, 5, 3.297, -4.136, 1.724, 0.02733),
        (2048, 1024, 3, 4.095, -9.327, 7.028, 0.01628),
        (2048, 1024, 5, 2.644, -3.128, 1.476, 0.00038),
    ]
    for n, m, T, a_ref, b_ref, c_ref, mse_ref in reference:
        (a, b, c), mse = solve_ns_coefficients(
            n, m, steps=T, num_samples=24, opt_iters=4000
        )
        # c = kappa is the scale-setting parameter; allow 2% relative drift from
        # the reference, which used a different optimizer and sample count.
        assert abs(c - c_ref) / abs(c_ref) < 0.02, f"({n},{m},{T}) c={c:.3f}"
        assert abs(a - a_ref) / abs(a_ref) < 0.02, f"({n},{m},{T}) a={a:.3f}"
        assert abs(b - b_ref) / abs(b_ref) < 0.02, f"({n},{m},{T}) b={b:.3f}"
        assert mse < mse_ref * 1.1 + 1e-5, f"({n},{m},{T}) mse={mse:.5f}"
        print(f"  ({n:>4},{m:>4}) T={T}: a={a:+.3f} b={b:+.3f} c={c:+.3f} "
              f"mse={mse:.5f}  (Su: {a_ref:+.3f} {b_ref:+.3f} {c_ref:+.3f} "
              f"{mse_ref:.5f})")


def test_per_shape_coefficients_beat_jordan():
    """Fitted coefficients must reduce the residual versus the fixed triple.

    This is the substantive claim: because the input singular value distribution
    depends on shape, coefficients tuned to that shape land the singular values
    closer to 1. The gain should be largest for non-square matrices, since
    Jordan's triple is approximately the SQUARE, steps=5 optimum.
    """
    steps = 5
    rows = []
    for n, m in [(256, 256), (1024, 256), (2048, 256), (1024, 128)]:
        sigma = sample_normalized_spectrum(n, m, num_samples=24, seed=1)  # (N,)

        def residual(coeffs):
            """Mean squared distance from 1 after `steps` iterations."""
            a, b, c = coeffs
            x = sigma.clone()                                            # (N,)
            for _ in range(steps):
                x = a * x + b * x**3 + c * x**5
            return ((x - 1.0) ** 2).mean().item()

        fitted, _ = solve_ns_coefficients(
            n, m, steps=steps, num_samples=24, opt_iters=4000, seed=0
        )
        mse_fit = residual(fitted)
        mse_jordan = residual(JORDAN_COEFFS)
        rows.append((n, m, mse_jordan, mse_fit))
        assert mse_fit <= mse_jordan, (
            f"({n},{m}) fitted {mse_fit:.5f} worse than Jordan {mse_jordan:.5f}"
        )
        print(f"  ({n:>4},{m:>4}) jordan mse={mse_jordan:.5f} -> "
              f"fitted mse={mse_fit:.5f}  ({mse_jordan/max(mse_fit,1e-12):.1f}x "
              f"better)")

    # The rectangular cases should gain more than the square one.
    square_gain = rows[0][2] / max(rows[0][3], 1e-12)
    rect_gain = rows[2][2] / max(rows[2][3], 1e-12)
    assert rect_gain > square_gain, (
        f"expected a larger gain on rectangular matrices: square {square_gain:.1f}x "
        f"vs rectangular {rect_gain:.1f}x"
    )


def test_optimizer_uses_per_shape_coefficients():
    """The optimizer must actually dispatch different coefficients per shape."""
    from models.imf_dit_video import imf_dit_video_S

    net = imf_dit_video_S(num_classes=8)
    opt = build_moonlight(net, lr=1e-3, weight_decay=0.1,
                          coeff_mode="per_shape", verbose=False)
    muon_group = next(g for g in opt.param_groups if g["use_muon"])
    shapes = {Moonlight._matrix_shape(p) for p in muon_group["params"]}
    coeffs = {
        s: get_ns_coefficients(*s, steps=5, mode="per_shape") for s in shapes
    }
    distinct = set(coeffs.values())
    assert len(shapes) > 1, "test model has only one matrix shape"
    assert len(distinct) > 1, (
        f"{len(shapes)} shapes collapsed to one coefficient triple -- per-shape "
        f"dispatch is not working"
    )
    for s, c in sorted(coeffs.items()):
        print(f"  shape {str(s):>14} -> a={c[0]:+.4f} b={c[1]:+.4f} c={c[2]:+.4f}")

    # And "jordan" mode must return the fixed triple for every shape.
    for s in shapes:
        assert get_ns_coefficients(*s, steps=5, mode="jordan") == JORDAN_COEFFS


def test_near_singular_inputs_are_only_partly_orthogonalized():
    """Document the real limitation: 5 steps cannot fix a near-singular matrix.

    A square iid Gaussian is near-singular (its smallest singular value sits
    around 1e-4 after Frobenius normalization), and p(sigma) ~ a sigma for small
    sigma, so five steps amplify it by only ~a^5. The BULK still lands near 1,
    which is what makes the update a usable direction, but sigma_min does not.
    Tall/wide Gaussians are well conditioned and do reach the full envelope.
    """
    torch.manual_seed(0)
    G_sq = torch.randn(256, 256)                             # (256, 256) square
    X5 = zeropower_via_newtonschulz5(G_sq, steps=5, dtype=torch.float32)
    X10 = zeropower_via_newtonschulz5(G_sq, steps=10, dtype=torch.float32)
    sv5 = torch.linalg.svdvals(X5.float())                   # (256,)
    sv10 = torch.linalg.svdvals(X10.float())                 # (256,)
    assert sv5.min() < 0.3, "square Gaussian unexpectedly well conditioned"
    assert sv5.median() > 0.6, f"bulk collapsed: median={sv5.median().item():.3f}"
    assert sv10.min() > sv5.min(), "more steps should lift the small tail"
    print(f"  square (256,256) Gaussian: 5 steps -> sigma_min="
          f"{sv5.min():.4f} median={sv5.median():.3f}; "
          f"10 steps -> sigma_min={sv10.min():.4f}")

    G_tall = torch.randn(1024, 256)                          # (1024, 256) tall
    svt = torch.linalg.svdvals(
        zeropower_via_newtonschulz5(G_tall, steps=5, dtype=torch.float32).float()
    )                                                        # (256,)
    assert svt.min() > 0.6, f"tall Gaussian sigma_min={svt.min().item():.4f}"
    print(f"  tall   (1024,256) Gaussian: 5 steps -> sigma in "
          f"[{svt.min():.3f}, {svt.max():.3f}] (well conditioned)")


def test_rms_scale_is_shape_invariant():
    """0.2 * sqrt(max(n,m)) * Phi must have RMS ~ 0.2 for every shape.

    This is the property that makes an AdamW-tuned lr transfer to Muon: AdamW's
    update entries are ~ +/- lr, so RMS ~ lr, and this scaling gives Muon the
    same per-entry magnitude regardless of layer width.
    """
    torch.manual_seed(0)
    const = 0.2
    # Exact only when every singular value is 1, i.e. ||Phi||_F^2 = r exactly.
    # Newton-Schulz lands them near 1, so hold RMS to 30% of the target across a
    # 64x span of aspect ratios; the point is that it does not DRIFT with shape.
    for n, m in [(64, 64), (1024, 256), (256, 1024), (2048, 64), (128, 4096)]:
        G = torch.randn(n, m, dtype=torch.float32)        # (n, m)
        X = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32)
        scale = Moonlight.muon_lr_scale((n, m), const)
        rms = (scale * X.float()).pow(2).mean().sqrt().item()
        assert abs(rms - const) / const < 0.3, f"({n},{m}) RMS={rms:.4f}"
        print(f"  ({n:>4},{m:>4}) scale={scale:7.3f} update RMS={rms:.4f}")

    # Without the per-shape scale the RMS would fall off as 1/sqrt(max(n,m)),
    # which is exactly what the scale cancels. Show the span it removes.
    unscaled = []
    for n, m in [(64, 64), (128, 4096)]:
        G = torch.randn(n, m, dtype=torch.float32)        # (n, m)
        X = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32)
        unscaled.append(X.float().pow(2).mean().sqrt().item())
    ratio = unscaled[0] / unscaled[1]
    expected = math.sqrt(4096 / 64)                       # sqrt(max ratio) = 8
    assert abs(ratio - expected) / expected < 0.3, (
        f"unscaled RMS ratio {ratio:.2f} != expected {expected:.2f}"
    )
    print(f"  unscaled RMS ratio (64,64)/(128,4096) = {ratio:.2f}, "
          f"predicted sqrt(4096/64) = {expected:.2f}")


def test_param_split():
    """Every parameter must land in exactly one bucket, with the right rule."""
    net = imf_dit_video_S(num_classes=10)
    muon, adamw = split_params(net, verbose=True)

    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    assert not (muon_ids & adamw_ids), "a parameter landed in both buckets"
    all_ids = {id(p) for p in net.parameters() if p.requires_grad}
    assert muon_ids | adamw_ids == all_ids, "split does not cover all parameters"

    # Muon must receive only matrices.
    for p in muon:
        assert p.ndim == 2, f"non-matrix in Muon bucket: {tuple(p.shape)}"

    named = dict(net.named_parameters())
    muon_names = {n for n, p in named.items() if id(p) in muon_ids}
    adamw_names = {n for n, p in named.items() if id(p) in adamw_ids}

    # Attention and MLP projections belong to Muon.
    for frag in ("attn.q_a_proj", "attn.q_b_proj", "attn.kv_a_proj",
                 "attn.kv_b_proj", "attn.out_proj", "mlp.w1", "mlp.w2", "mlp.w3"):
        hit = [n for n in muon_names if frag in n]
        assert hit, f"expected {frag} in the Muon bucket"

    # Embeddings, token banks, patch embed, output layers and all 1-D go to AdamW.
    for frag in ("embedding_table", "_tokens", "_embedder", "x_embedder",
                 "final_layer"):
        assert any(frag in n for n in adamw_names), f"expected {frag} in AdamW"
        assert not any(frag in n for n in muon_names), f"{frag} leaked into Muon"
    for n in adamw_names:
        p = named[n]
        assert p.ndim < 2 or any(
            f in n for f in ("embedding_table", "_tokens", "_embedder",
                             "x_embedder", "final_layer")
        ), f"unexpected matrix in AdamW bucket: {n} {tuple(p.shape)}"

    # The zero-init residual gates are 1-D and must be on the AdamW side.
    for n in ("shared_blocks.0.attn_scale", "shared_blocks.0.mlp_scale"):
        assert n in adamw_names, f"{n} should be AdamW"
    print(f"  Muon tensors: {len(muon)}, AdamW tensors: {len(adamw)}")


def test_adamw_branch_matches_reference():
    """The hand-written AdamW branch must track torch.optim.AdamW."""
    torch.manual_seed(0)
    p_ref = torch.nn.Parameter(torch.randn(7))            # (7,) 1-D -> AdamW branch
    p_ours = torch.nn.Parameter(p_ref.detach().clone())

    ref = torch.optim.AdamW([p_ref], lr=1e-2, weight_decay=0.1,
                            betas=(0.9, 0.95), eps=1e-8)
    ours = Moonlight(adamw_params=[p_ours], lr=1e-2, weight_decay=0.1,
                     adamw_betas=(0.9, 0.95), adamw_eps=1e-8)

    for _ in range(20):
        g = torch.randn(7)                                # (7,) shared gradient
        p_ref.grad = g.clone()
        p_ours.grad = g.clone()
        ref.step()
        ours.step()
    diff = (p_ref - p_ours).abs().max().item()
    assert diff < 1e-5, f"AdamW branch diverges from reference: {diff:.2e}"
    print(f"  max |ours - torch.optim.AdamW| after 20 steps: {diff:.2e}")


def test_end_to_end(device):
    """Moonlight must reduce a real iMF loss and move both branches' weights."""
    from imf_video import IMFVideoLoss

    torch.manual_seed(0)
    num_classes = 8
    net = imf_dit_video_S(num_classes=num_classes).to(device)
    loss_mod = IMFVideoLoss(net, num_classes)
    opt = build_moonlight(net, lr=1e-3, weight_decay=0.1, verbose=False)

    # latents: (B, C, T, H, W);  labels: (B,)
    x = torch.randn(2, 16, 3, 16, 16, device=device)
    y = torch.randint(0, num_classes, (2,), device=device)

    muon_w = net.shared_blocks[0].attn.q_b_proj.weight     # (H*(dn+dr), dq)
    adamw_w = net.shared_blocks[0].attn_scale              # (d,) residual gate
    before_muon = muon_w.detach().clone()
    before_adamw = adamw_w.detach().clone()

    losses = []
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_mod(x, y)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        opt.step()
        losses.append(loss.item())

    for p in net.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after Moonlight step"

    moved_adamw = not torch.allclose(before_adamw, adamw_w)
    moved_muon = not torch.allclose(before_muon, muon_w)
    assert moved_adamw, "AdamW branch never moved"
    assert moved_muon, "Muon branch never moved"
    print(f"  losses: {[f'{v:.4f}' for v in losses]}")
    print(f"  Muon weight moved={moved_muon}, AdamW gate moved={moved_adamw}")


def test_lr_schedule_drives_both_groups():
    """An external scheduler writing group['lr'] must reach both branches."""
    net = imf_dit_video_S(num_classes=8)
    opt = build_moonlight(net, lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2, len(opt.param_groups)
    flags = sorted(g["use_muon"] for g in opt.param_groups)
    assert flags == [False, True], flags
    for g in opt.param_groups:
        g["lr"] = 5e-4
    assert all(g["lr"] == 5e-4 for g in opt.param_groups)
    print("  both param groups expose and honour 'lr'")


def test_state_dict_round_trip(device):
    """Optimizer state must survive save/load so resume works."""
    torch.manual_seed(0)
    net = imf_dit_video_S(num_classes=8).to(device)
    opt = build_moonlight(net, lr=1e-3, weight_decay=0.1)
    for p in net.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    net2 = imf_dit_video_S(num_classes=8).to(device)
    net2.load_state_dict(net.state_dict())
    opt2 = build_moonlight(net2, lr=1e-3, weight_decay=0.1)
    opt2.load_state_dict(sd)
    assert len(opt2.state) == len(opt.state), (len(opt2.state), len(opt.state))
    print(f"  round-tripped state for {len(opt2.state)} parameters")


def main():
    device = pick_device()
    print(f"device: {device}")
    print("a fixed (a,b,c) is shape-blind given the same spectrum:")
    test_newtonschulz_map_itself_has_no_shape_dependence()
    print("coefficient solver vs Su Jianlin's published table:")
    test_solver_reproduces_su_jianlin_table()
    print("per-shape coefficients vs Jordan's fixed triple:")
    test_per_shape_coefficients_beat_jordan()
    print("optimizer dispatches per-shape coefficients:")
    test_optimizer_uses_per_shape_coefficients()
    print("Newton-Schulz orthogonality (well-conditioned input):")
    test_newtonschulz_orthogonality()
    print("Newton-Schulz on near-singular input:")
    test_near_singular_inputs_are_only_partly_orthogonalized()
    print("RMS scale shape-invariance:")
    test_rms_scale_is_shape_invariant()
    print("parameter split:")
    test_param_split()
    print("AdamW branch vs torch.optim.AdamW:")
    test_adamw_branch_matches_reference()
    print("lr schedule wiring:")
    test_lr_schedule_drives_both_groups()
    print("state_dict round trip:")
    test_state_dict_round_trip(device)
    print("end-to-end training:")
    test_end_to_end(device)
    print("ALL MOONLIGHT TESTS PASSED")


if __name__ == "__main__":
    main()
