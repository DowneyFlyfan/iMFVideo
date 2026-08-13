"""Moonlight optimizer: Moonshot AI's scalable variant of Muon, plus AdamW.

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024) replaces the
SGD-momentum update of a weight MATRIX by its nearest semi-orthogonal matrix,
computed with a fixed quintic Newton-Schulz iteration. Moonlight ("Muon is
Scalable for LLM Training", arXiv:2502.16982) makes it usable at scale by adding
two things:

  1. decoupled weight decay, and
  2. a per-shape update scale that matches Muon's update RMS to AdamW's,

so that an AdamW-tuned learning rate and weight decay transfer unchanged.

Why the update scale is 0.2 * sqrt(max(n, m)). Newton-Schulz drives the momentum
matrix M(n, m) toward Phi = U V^T with r = min(n, m) orthonormal columns on each
side, so

    ||Phi||_F^2 = tr(Phi^T Phi) = tr(V U^T U V^T) = tr(V^T V) = r,

giving RMS(Phi) = ||Phi||_F / sqrt(n m) = sqrt(r / (n m)) = 1 / sqrt(max(n, m))
once r = min(n, m). That shrinks as a layer gets wider, so a single learning rate
would under-train wide layers. Multiplying by 0.2 * sqrt(max(n, m)) pins the
update RMS at 0.2 for every shape. AdamW's update entries are roughly +/- lr
(the m / sqrt(v) ratio is sign-like), i.e. RMS ~ lr, which is what makes the two
families comparable.

Parameter routing. Muon is only defined for matrices, and Moonlight further
excludes the embedding and output layers even though those are 2-D: their rows
index vocabulary/tokens rather than a hidden subspace, so orthogonalizing across
them is not meaningful. Everything Muon does not take -- all 1-D parameters
(normalization scales, biases, residual gates), lookup tables, learned token
banks, the patch-embedding convolution and the final projections -- goes to
AdamW. See `split_params` for the concrete rule applied to IMFDiTVideo.

Shape symbols used below:
    n : rows of a weight matrix (output features)
    m : columns of a weight matrix (input features)
    r : rank = min(n, m)
"""

import math

import torch

# Keller Jordan's coefficients, used by Moonlight as-is. Su Jianlin's analysis
# (https://kexue.fm/archives/10592) shows these are approximately the optimum for
# a SQUARE matrix at steps=5; see solve_ns_coefficients for the per-shape solve.
JORDAN_COEFFS = (3.4445, -4.7750, 2.0315)

# Cache keyed by (n, m, steps) so each distinct weight shape is solved once.
_COEFF_CACHE = {}


def sample_normalized_spectrum(n, m, num_samples=32, seed=0, device="cpu"):
    """Empirical distribution of Frobenius-normalized singular values.

    Newton-Schulz starts from X_0 = M / ||M||_F, so the singular values it must
    map to 1 are sigma_i / ||M||_F for the momentum matrix M. Modelling M as iid
    Gaussian, that distribution depends on the SHAPE: the squared singular values
    of an (n, m) Gaussian follow the Marchenko-Pastur law with aspect ratio
    min/max, and normalizing by ||M||_F ~ sqrt(n m) makes the individual values
    scale as 1 / sqrt(min(n, m)). Bigger and squarer matrices therefore start
    with smaller, more spread-out singular values and need a more aggressive
    polynomial.

    Args:
        n, m: matrix shape.
        num_samples: how many random matrices to draw.
        seed: RNG seed.
        device: where to run the SVD.

    Returns:
        (num_samples * min(n, m),) tensor of normalized singular values.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(num_samples):
        M = torch.randn(n, m, generator=g, dtype=torch.float32)     # (n, m)
        M = M.to(device)
        s = torch.linalg.svdvals(M)                                 # (min(n,m),)
        out.append(s / s.norm())          # ||M||_F = ||sigma||_2
    return torch.cat(out)                                           # (S*min,)


def solve_ns_coefficients(
    n, m, steps=5, num_samples=32, opt_iters=3000, lr=2e-2, seed=0, device="cpu",
):
    """Fit (a, b, c) so `steps` Newton-Schulz iterations best map sigma -> 1.

    Follows Su Jianlin (https://kexue.fm/archives/10592), after @leloykun. The
    quintic is reparameterized by its fixed points, which keeps the search
    well-behaved:

        g(x) = x + kappa * x * (x^2 - x1^2) * (x^2 - x2^2)

    with fixed points {0, +/-x1, +/-x2}. Expanding gives

        a = 1 + kappa * x1^2 * x2^2,   b = -kappa * (x1^2 + x2^2),   c = kappa.

    We minimize E[(g^steps(sigma) - 1)^2] over the shape's empirical spectrum, so
    the fitted coefficients are specific to (n, m, steps). This is the sense in
    which coefficients DO depend on shape: the polynomial itself never sees n or
    m, but the distribution of inputs it has to handle does.

    Args:
        n, m: matrix shape the coefficients are for.
        steps: number of iterations the coefficients are tuned for.
        num_samples: random matrices used to estimate the spectrum.
        opt_iters, lr: Adam budget for the fit.
        seed: RNG seed.
        device: device for the SVD and the fit.

    Returns:
        ((a, b, c), mse): the fitted coefficients and the achieved mean squared
        error of the final singular values against 1.
    """
    sigma = sample_normalized_spectrum(
        n, m, num_samples=num_samples, seed=seed, device=device
    )                                                               # (N,)

    # Su initializes at kappa=1, x1=0.9, x2=1.1: one fixed point just below the
    # target and one just above, so the iteration lands near 1 from either side.
    params = torch.tensor([1.0, 0.9, 1.1], device=sigma.device, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)

    for _ in range(opt_iters):
        opt.zero_grad()
        kappa, x1, x2 = params[0], params[1], params[2]
        x = sigma                                                   # (N,)
        for _ in range(steps):
            x = x + kappa * x * (x**2 - x1**2) * (x**2 - x2**2)
        loss = ((x - 1.0) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        kappa, x1, x2 = (v.item() for v in params)
        a = 1.0 + kappa * x1**2 * x2**2
        b = -kappa * (x1**2 + x2**2)
        c = kappa
        x = sigma
        for _ in range(steps):
            x = x + kappa * x * (x**2 - x1**2) * (x**2 - x2**2)
        mse = ((x - 1.0) ** 2).mean().item()
    return (a, b, c), mse


def get_ns_coefficients(n, m, steps, mode="per_shape", verbose=False, **solve_kw):
    """Return the (a, b, c) triple for a given shape, caching the per-shape solve.

    Args:
        n, m: matrix shape.
        steps: Newton-Schulz iterations.
        mode: "jordan" for the fixed constants, "per_shape" to fit them.
        verbose: print each solve.
        **solve_kw: forwarded to solve_ns_coefficients.

    Returns:
        (a, b, c) tuple of floats.
    """
    if mode == "jordan":
        return JORDAN_COEFFS
    if mode != "per_shape":
        raise ValueError(f"unknown coefficient mode {mode!r}")

    # sigma(M) == sigma(M^T), so (n, m) and (m, n) share a spectrum and a solve.
    n, m = min(n, m), max(n, m)
    key = (n, m, steps)
    if key not in _COEFF_CACHE:
        coeffs, mse = solve_ns_coefficients(n, m, steps=steps, **solve_kw)
        _COEFF_CACHE[key] = coeffs
        if verbose:
            # Baseline MSE using Jordan's fixed triple, for comparison.
            print(
                f"  NS coeffs ({n:>5},{m:>5}) T={steps}: "
                f"a={coeffs[0]:+.4f} b={coeffs[1]:+.4f} c={coeffs[2]:+.4f} "
                f"mse={mse:.3e}"
            )
    return _COEFF_CACHE[key]


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, dtype=None, coeffs=None):
    """Approximate the semi-orthogonal factor of G by a quintic Newton-Schulz run.

    Iterates X <- a X + b (X X^T) X + c (X X^T)^2 X. Because the iteration acts
    as the scalar map sigma -> a sigma + b sigma^3 + c sigma^5 on each singular
    value, the polynomial itself never sees n or m. The distribution of inputs it
    must handle does, however, so the best (a, b, c) is shape-dependent -- pass
    `coeffs` from get_ns_coefficients to exploit that. The default is Jordan's
    triple, which Su Jianlin identifies as roughly the square-matrix, steps=5
    optimum.

    G is normalized by its Frobenius norm first, which makes the iteration
    scale-invariant, and transposed when n > m so the Gram matrix formed inside
    the loop is the smaller of X X^T and X^T X.

    Args:
        G: (n, m) gradient or momentum matrix.
        steps: number of Newton-Schulz iterations.
        eps: guard added to the Frobenius norm before dividing.
        dtype: compute dtype. Defaults to bfloat16 on CUDA (where the reduced
            precision is harmless because the output is only a direction) and
            float32 elsewhere, since MPS and CPU bfloat16 matmul support is
            patchier.
        coeffs: (a, b, c) triple; defaults to JORDAN_COEFFS.

    Returns:
        (n, m) approximately semi-orthogonal matrix, in `dtype`.
    """
    assert G.ndim == 2, f"Newton-Schulz needs a matrix, got shape {tuple(G.shape)}"
    a, b, c = JORDAN_COEFFS if coeffs is None else coeffs

    if dtype is None:
        dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32

    X = G.to(dtype)                                   # (n, m)
    X = X / (X.norm() + eps)                          # (n, m), ||X||_F ~ 1

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T                                       # (m, n) with m <= n

    for _ in range(steps):
        A = X @ X.T                                   # (min, min) Gram matrix
        B = b * A + c * (A @ A)                       # (min, min)
        X = a * X + B @ X                             # (min, max)

    if transposed:
        X = X.T                                       # back to (n, m)

    return X


class Moonlight(torch.optim.Optimizer):
    """Muon on matrix parameters, AdamW on everything else, in one optimizer.

    Parameter groups carry a `use_muon` flag. Muon groups get momentum ->
    Newton-Schulz -> RMS-matched step; the rest get a standard decoupled-weight-
    decay AdamW step. Both honour the group's `lr`, so an external scheduler that
    writes `group["lr"]` drives both halves consistently.
    """

    def __init__(
        self,
        muon_params=None,
        adamw_params=None,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        lr_scale_constant=0.2,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        ns_dtype=None,
        coeff_mode="per_shape",
        coeff_samples=32,
        coeff_iters=3000,
        coeff_lr=2e-2,
        coeff_seed=0,
        verbose=False,
    ):
        """
        Args:
            muon_params: iterable of matrix parameters to update with Muon.
            adamw_params: iterable of all remaining parameters.
            lr: shared base learning rate.
            weight_decay: decoupled weight decay, applied in both branches.
            momentum: Muon momentum coefficient mu.
            nesterov: use the Nesterov-style lookahead gradient G + mu * M.
            ns_steps: Newton-Schulz iterations per step.
            lr_scale_constant: the 0.2 in 0.2 * sqrt(max(n, m)); the target
                update RMS.
            adamw_betas, adamw_eps: AdamW hyperparameters.
            ns_dtype: override the Newton-Schulz compute dtype.
            coeff_mode: "per_shape" fits (a, b, c) to each distinct weight shape
                at construction time (Su Jianlin's procedure); "jordan" uses the
                single fixed triple, which is roughly the square-matrix steps=5
                optimum and is what stock Muon and Moonlight ship.
            coeff_samples, coeff_iters, coeff_lr, coeff_seed: budget and seed for
                that fit.
            verbose: print the per-shape coefficient solves.
        """
        muon_params = [] if muon_params is None else list(muon_params)
        adamw_params = [] if adamw_params is None else list(adamw_params)
        assert muon_params or adamw_params, "no parameters given"

        for p in muon_params:
            assert p.ndim >= 2, (
                f"Muon needs matrix parameters, got a {p.ndim}-D tensor of shape "
                f"{tuple(p.shape)}; route it to adamw_params instead"
            )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            lr_scale_constant=lr_scale_constant,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            ns_dtype=ns_dtype,
            coeff_mode=coeff_mode,
        )
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "use_muon": False})
        super().__init__(groups, defaults)

        # Solve the per-shape coefficients up front rather than stalling the first
        # training step. Shapes repeat across blocks, so this is a handful of
        # solves however deep the model is.
        if coeff_mode == "per_shape" and muon_params:
            shapes = {self._matrix_shape(p) for p in muon_params}
            if verbose:
                print(
                    f"Moonlight: fitting Newton-Schulz coefficients for "
                    f"{len(shapes)} distinct shapes (steps={ns_steps})"
                )
            for shape in sorted(shapes):
                get_ns_coefficients(
                    *shape, steps=ns_steps, mode=coeff_mode, verbose=verbose,
                    num_samples=coeff_samples, opt_iters=coeff_iters,
                    lr=coeff_lr, seed=coeff_seed,
                )

    @staticmethod
    def _matrix_shape(p):
        """The 2-D shape Newton-Schulz will actually act on.

        Args:
            p: a parameter tensor of any rank >= 2.

        Returns:
            (n, m) with >2-D tensors flattened to (p.shape[0], rest).
        """
        if p.ndim > 2:
            return (p.shape[0], p.numel() // p.shape[0])
        return tuple(p.shape)

    @staticmethod
    def muon_lr_scale(matrix_shape, constant):
        """Per-shape multiplier that fixes the orthogonalized update's RMS.

        Args:
            matrix_shape: (n, m) shape of the matrix handed to Newton-Schulz.
            constant: target update RMS (0.2 in Moonlight).

        Returns:
            float multiplier: constant * sqrt(max(n, m)).
        """
        n, m = matrix_shape
        return constant * math.sqrt(max(n, m))

    @torch.no_grad()
    def step(self, closure=None):
        """Run one optimization step over both branches."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        """Momentum -> Newton-Schulz -> RMS-matched update, for one group."""
        lr = group["lr"]
        wd = group["weight_decay"]
        mu = group["momentum"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            # Conv and other >2-D weights are treated as (out_features, rest);
            # this is the matrix Newton-Schulz and the RMS scale both act on.
            if grad.ndim > 2:
                grad = grad.reshape(grad.size(0), -1)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]              # (n, m)
            buf.mul_(mu).add_(grad)

            # Nesterov lookahead: step along G + mu * M rather than M.
            direction = grad.add(buf, alpha=mu) if group["nesterov"] else buf

            # Coefficients tuned for THIS shape's singular value distribution.
            coeffs = get_ns_coefficients(
                *tuple(direction.shape),
                steps=group["ns_steps"],
                mode=group["coeff_mode"],
            )
            update = zeropower_via_newtonschulz5(
                direction, steps=group["ns_steps"], dtype=group["ns_dtype"],
                coeffs=coeffs,
            )                                            # (n, m), ~semi-orthogonal
            scale = self.muon_lr_scale(
                tuple(update.shape), group["lr_scale_constant"]
            )

            # Decoupled weight decay, then the scaled orthogonalized step.
            p.mul_(1 - lr * wd)
            p.add_(
                update.reshape(p.shape).to(p.dtype), alpha=-lr * scale
            )

    def _adamw_step(self, group):
        """Standard AdamW with decoupled weight decay, for one group."""
        lr = group["lr"]
        wd = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction, as in the AdamW paper.
            bias1 = 1 - beta1**t
            bias2 = 1 - beta2**t
            denom = (exp_avg_sq / bias2).sqrt_().add_(eps)

            p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg / bias1, denom, value=-lr)


def split_params(model, verbose=False):
    """Route an IMFDiTVideo's parameters into the Muon and AdamW buckets.

    Muon gets the 2-D hidden weight matrices: the MLA (multi-head latent
    attention) projections and the SwiGLU MLP projections inside the transformer
    blocks.

    AdamW gets everything else:
      - every 1-D parameter (RMSNorm scales, biases, the zero-init residual
        gates `attn_scale` / `mlp_scale`),
      - `nn.Embedding` lookup tables and the learned token banks
        (`class_tokens`, `time_tokens`, `omega_tokens`, `t_min_tokens`,
        `t_max_tokens`), whose rows index conditioning slots rather than a
        hidden subspace,
      - the timestep / CFG-scale embedder MLPs, which map a fixed sinusoidal
        basis into the model and so belong to the embedding path,
      - the patch-embedding Conv3d, the model's input layer,
      - `u_final_layer` / `v_final_layer`, the output projections.

    Args:
        model: an IMFDiTVideo (or any module following the same naming).
        verbose: print the resulting split.

    Returns:
        (muon_params, adamw_params): two lists of Parameters, together covering
        every parameter of `model` exactly once.
    """
    # Name fragments that force a parameter into the AdamW bucket even when 2-D.
    adamw_name_markers = (
        "embedding_table",   # nn.Embedding lookup table
        "_tokens",           # learned conditioning token banks
        "_embedder",         # timestep / omega / cfg-interval embedder MLPs
        "x_embedder",        # patch-embedding Conv3d (input layer)
        "final_layer",       # u_final_layer / v_final_layer (output layers)
        "alpha_logit",       # SLA2 (H, Mb) sparse/linear mixing gates:
                             # Hadamard-class ratios, not a hidden subspace
    )

    muon_params, adamw_params = [], []
    muon_names, adamw_names = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        to_adamw = (
            p.ndim < 2
            or any(marker in name for marker in adamw_name_markers)
        )
        if to_adamw:
            adamw_params.append(p)
            adamw_names.append(name)
        else:
            muon_params.append(p)
            muon_names.append(name)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    assert n_muon + n_adamw == total, (
        f"parameter split lost tensors: {n_muon} + {n_adamw} != {total}"
    )

    if verbose:
        print(
            f"Moonlight split: Muon {len(muon_params)} tensors / "
            f"{n_muon/1e6:.1f}M params, AdamW {len(adamw_params)} tensors / "
            f"{n_adamw/1e6:.1f}M params"
        )
    return muon_params, adamw_params


def build_moonlight(model, lr, weight_decay, momentum=0.95, nesterov=True,
                    ns_steps=5, lr_scale_constant=0.2, adamw_betas=(0.9, 0.95),
                    adamw_eps=1e-8, coeff_mode="per_shape",
                    coeff_samples=32, coeff_iters=3000, coeff_lr=2e-2,
                    coeff_seed=0, verbose=False):
    """Convenience constructor: split `model`'s parameters, then build Moonlight.

    Args:
        model: the network whose parameters to optimize.
        lr, weight_decay: shared across both branches.
        momentum, nesterov, ns_steps, lr_scale_constant: Muon settings.
        adamw_betas, adamw_eps: AdamW settings.
        coeff_mode: "per_shape" or "jordan"; see Moonlight.__init__.
        coeff_samples, coeff_iters, coeff_lr, coeff_seed: budget and seed for the
            per-shape coefficient fit.
        verbose: print the parameter split and the coefficient solves.

    Returns:
        a Moonlight optimizer with one Muon group and one AdamW group.
    """
    muon_params, adamw_params = split_params(model, verbose=verbose)
    return Moonlight(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        lr_scale_constant=lr_scale_constant,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        coeff_mode=coeff_mode,
        coeff_samples=coeff_samples,
        coeff_iters=coeff_iters,
        coeff_lr=coeff_lr,
        coeff_seed=coeff_seed,
        verbose=verbose,
    )
