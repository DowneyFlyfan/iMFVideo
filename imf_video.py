"""PyTorch improved MeanFlow training loss for video latents.

Faithful port of iMeanFlow.forward() in refs/imf.py (JAX), adapted from images
(B, H, W, C) to video latents (B, C, T, H, W): per-sample scalars broadcast
as (B, 1, 1, 1, 1) and losses sum over the (C, T, H, W) dims.

Model calls that do not need jvp (guidance) run under torch.no_grad();
the mean-field u prediction and its time derivative come from a single
torch.func.jvp (forward-mode AD) call through the network.
"""

import torch
import torch.nn as nn


def adaptive_loss_metrics(
    loss_u,
    loss_v,
    *,
    norm_eps,
    norm_p,
    loss_v_weight,
    elements_per_sample,
):
    """Return the adaptive objective and its informative raw per-element MSE.

    The inputs are per-sample squared-error sums. With ``norm_p=1``, each
    detached-normalized objective branch is bounded below one, whereas the raw
    mean squared error remains comparable between training steps.
    """
    if elements_per_sample <= 0:
        raise ValueError("elements_per_sample must be positive")

    def adaptive(branch_loss):
        denominator = (branch_loss + norm_eps) ** norm_p
        return branch_loss / denominator.detach()

    objective = adaptive(loss_u) + loss_v_weight * adaptive(loss_v)
    raw_metric = (loss_u + loss_v_weight * loss_v).mean() / elements_per_sample
    return objective.mean(), raw_metric


class IMFVideoLoss(nn.Module):
    """improved MeanFlow loss for video latents."""

    def __init__(
        self,
        net,
        num_classes,
        # Noise distribution
        P_mean=-0.4,
        P_std=1.0,
        # Loss
        data_proportion=0.5,
        cfg_beta=1.0,
        class_dropout_prob=0.1,
        s_max=7.0,
        # Training dynamics
        norm_p=1.0,
        norm_eps=0.01,
        loss_v_weight=1.0,
        # "fast": detached hand-rolled Triton forward-mode pass for du/dt
        # (models/mla_jvp_fast); "functorch": torch.func.jvp (previous path)
        jvp_impl="functorch",
        # stratified (jittered-quantile) time sampling: same marginal law,
        # much lower per-step loss/gradient variance at small batch
        stratified_time=True,
        # stratify the guidance-interval bounds t_min / t_max as well. Kept on
        # its own flag rather than folded into stratified_time because the
        # canonical scoring distribution of the tuning harness pins it to
        # False; flipping it under the old flag would silently redefine every
        # previously recorded probe score.
        stratified_interval=False,
        # Size of the group the strata are spread over. 0 stratifies inside the
        # micro-batch only, which is INERT at batch size 1: one stratum spanning
        # (0, 1) is just a uniform draw, so a memory-bound run (full latents on
        # 16 GB force micro-batch 1) gets no variance reduction at all. Setting
        # this to batch_size * grad_accum spreads the strata across the whole
        # accumulation group instead, which is the quantity the optimizer step
        # actually averages over.
        strat_group=0,
        # Run every network forward (guidance no-grad pair, primal, sampler)
        # under bf16 autocast: Linear GEMMs hit tensor cores, master weights
        # and loss arithmetic stay fp32. Incompatible with jvp_impl
        # "functorch" (forward-mode AD through autocast casts); the "fast"
        # path computes du/dt in its own fp16 pipeline and is unaffected.
        autocast_bf16=False,
    ):
        super().__init__()
        self.stratified_time = stratified_time
        self.stratified_interval = stratified_interval
        self.strat_group = strat_group
        # One independent stratum queue per draw site: sharing a queue would
        # lock t, omega and the flow-matching split into a fixed pattern of
        # joint strata, changing the joint law rather than just its variance.
        self._strat_queues = {}
        self.jvp_impl = jvp_impl
        self.autocast_bf16 = autocast_bf16
        assert not (autocast_bf16 and jvp_impl == "functorch"), (
            "autocast_bf16 requires jvp_impl='fast'"
        )
        self.net = net
        self.num_classes = num_classes
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.s_max = s_max
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.loss_v_weight = loss_v_weight

    #######################################################
    #                       Schedule                      #
    #######################################################

    def _take_strata(self, bz, device, site):
        """Pop bz stratum indices out of `site`'s shuffled queue of 0..G-1.

        Drawing without replacement from a reshuffled 0..G-1 queue is what makes
        the group of G consecutive samples cover every stratum exactly once,
        even though those samples arrive over several accumulation micro-batches.
        The marginal law of a single draw stays uniform over the strata.

        Args:
            bz: number of indices to pop (the micro-batch size B).
            device: placement;  site: draw-site key, one queue per key.

        Returns:
            (B,) float32 stratum indices in [0, G).
        """
        g = self.strat_group  # G: strata per group, from config
        # q: (n,) int64 unconsumed stratum indices for this site, n <= G. Source:
        # self._strat_queues, initialized empty in __init__ and refilled below.
        q = self._strat_queues.get(site)
        if q is None or q.numel() < bz:
            fresh = torch.randperm(g, device=device)  # (G,) a fresh 0..G-1 shuffle
            q = fresh if q is None else torch.cat([q, fresh])
        # out: (B,) the popped indices;  remainder is carried to the next call.
        out, self._strat_queues[site] = q[:bz], q[bz:]
        return out.to(torch.float32)

    def _strat_uniform(self, bz, device, site="default"):
        """Jittered-quantile uniform draws in (0, 1).

        Marginally U(0,1) like torch.rand, but the draws are spread one per
        equal-probability stratum, so any batch statistic has far lower
        variance. Falls back to plain uniform when stratified_time is off.

        Args:
            bz: micro-batch size B;  device: placement.
            site: draw-site key. Only used when strat_group > B, where each
                site keeps its own stratum queue across micro-batches.

        Returns:
            (B,) float32 in (0, 1).
        """
        if not self.stratified_time:
            return torch.rand(bz, device=device, dtype=torch.float32)
        if self.strat_group and self.strat_group > bz:
            g = self.strat_group  # G: group size, batch * grad_accum
            # i: (B,) float32 stratum indices in [0, G), popped without
            # replacement from this site's shuffled queue.
            i = self._take_strata(bz, device, site)
            u = (i + torch.rand(bz, device=device)) / g  # (B,) in (0,1)
        else:
            i = torch.randperm(bz, device=device).to(torch.float32)  # (B,)
            u = (i + torch.rand(bz, device=device)) / bz  # (B,)
        return u.clamp(1e-6, 1 - 1e-6)

    def logit_normal_dist(self, bz, device, dtype, site="time"):
        """Draw times in (0, 1) from a logit-normal distribution.

        Args:
            bz: batch size B.
            device, dtype: placement of the returned tensor.
            site: stratum-queue key; sample_tr passes a distinct key for each
                of the two endpoint draws so they stay independent.

        Returns:
            (B, 1, 1, 1, 1) times, broadcastable against (B, C, T, H, W) latents.
        """
        if self.stratified_time:
            # Stratified draws mapped through the normal quantile function:
            # same logit-normal marginal, far lower batch-statistic variance.
            u = self._strat_uniform(bz, device, site)  # (B,)
            rnd_normal = (
                (torch.erfinv(2 * u - 1) * (2**0.5)).reshape(bz, 1, 1, 1, 1).to(dtype)
            )  # (B,1,1,1,1)
        else:
            rnd_normal = torch.randn(bz, 1, 1, 1, 1, device=device, dtype=dtype)
        #   rnd_normal: (B, 1, 1, 1, 1) standard normal samples
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device, dtype):
        """Sample the interval endpoints t > r for the mean-flow objective.

        Args:
            bz: batch size B.
            device, dtype: placement of the returned tensors.

        Returns:
            t: (B, 1, 1, 1, 1) later time.
            r: (B, 1, 1, 1, 1) earlier time, r <= t.
            fm_mask: (B, 1, 1, 1, 1) bool, True on the leading
                floor(B * data_proportion) samples, which are forced to r = t and
                so reduce to plain flow matching.
        """
        t = self.logit_normal_dist(bz, device, dtype, "time_t")  # (B, 1, 1, 1, 1)
        r = self.logit_normal_dist(bz, device, dtype, "time_r")  # (B, 1, 1, 1, 1)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # Stratified flow-matching split. A deterministic prefix rule
        # (int(bz * data_proportion)) silently yields ZERO flow-matching
        # samples for bz < 1/data_proportion -- at micro-batch 1 the branch
        # never fires. The jittered-quantile rule keeps the exact prefix
        # behaviour in expectation and gives the right proportion on
        # average at any batch size.
        fm_u = self._strat_uniform(bz, device, "fm_split")  # (B,)
        fm_mask = (fm_u < self.data_proportion).reshape(bz, 1, 1, 1, 1)
        r = torch.where(fm_mask, t, r)  # (B, 1, 1, 1, 1)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, device):
        """Sample the CFG (classifier-free guidance) scale omega.

        omega(u) = (1 + u * ((1 + s_max)^(1-beta) - 1))^(1/(1-beta)) with
        u ~ U(0, 1), and omega = (1 + s_max)^u in the beta = 1 limit. Both map
        u in [0, 1] onto omega in [1, 1 + s_max].

        Args:
            bz: batch size B.
            device: placement of the returned tensor.

        Returns:
            (B, 1, 1, 1, 1) float32 guidance scales in [1, 1 + s_max].
        """
        u = self._strat_uniform(bz, device, "omega").reshape(bz, 1, 1, 1, 1)
        #   u: (B, 1, 1, 1, 1) uniform samples (stratified when enabled);
        #   omega is the largest single variance source in the raw loss, so
        #   spreading it across strata matters as much as spreading t.

        if self.cfg_beta == 1.0:
            s = torch.exp(
                u * torch.log1p(torch.tensor(self.s_max, dtype=torch.float32))
            )
        else:
            smax = torch.tensor(self.s_max, dtype=torch.float32)
            b = torch.tensor(self.cfg_beta, dtype=torch.float32)

            log_base = (1.0 - b) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))

            s = torch.exp(log_inner / (1.0 - b))

        return s.float()

    def sample_cfg_interval(self, bz, device, dtype, fm_mask):
        """Sample CFG interval [t_min, t_max] from uniform distribution.

        Args:
            bz: batch size B.
            device, dtype: placement of the returned tensors.
            fm_mask: (B, 1, 1, 1, 1) bool, True on flow-matching (r = t) samples.

        Returns:
            t_min, t_max: (B, 1, 1, 1, 1) guidance interval bounds. Flow-matching
            samples get the degenerate interval [0, 1], i.e. no interval gating.
        """
        if self.stratified_interval:
            # u_lo, u_hi: (B, 1, 1, 1, 1) jittered-quantile uniforms from
            # _strat_uniform, same marginal law as torch.rand but one draw per
            # equal-probability stratum.
            u_lo = self._strat_uniform(bz, device, "t_min").reshape(bz, 1, 1, 1, 1)
            u_hi = self._strat_uniform(bz, device, "t_max").reshape(bz, 1, 1, 1, 1)
            t_min = 0.5 * u_lo.to(dtype)
            t_max = 0.5 + 0.5 * u_hi.to(dtype)
        else:
            t_min = 0.5 * torch.rand(bz, 1, 1, 1, 1, device=device, dtype=dtype)
            t_max = 0.5 + 0.5 * torch.rand(bz, 1, 1, 1, 1, device=device, dtype=dtype)

        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: (B, C, T, H, W) noisy video latent at time t.
            t: (B, 1, 1, 1, 1) current time step.
            h: (B, 1, 1, 1, 1) time difference t - r.
            omega: (B, 1, 1, 1, 1) CFG (classifier-free guidance) scale.
            t_min, t_max: (B, 1, 1, 1, 1) guidance interval bounds.
            y: (B,) integer class labels.

        Returns: (u, v)
            u: (B, C, T, H, W) predicted average velocity field.
            v: (B, C, T, H, W) predicted instantaneous velocity field.
        """
        bz = x.shape[0]
        # The network takes the per-sample scalars flattened to (B,).
        if self.autocast_bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                u, v = self.net(
                    x,
                    t.reshape(bz),
                    h.reshape(bz),
                    omega.reshape(bz),
                    t_min.reshape(bz),
                    t_max.reshape(bz),
                    y,
                )
            # loss arithmetic stays fp32
            return u.float(), v.float()
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        """Read the v-head prediction, discarding the u-head output.

        Args:
            x: (B, C, T, H, W) noisy video latent at time t.
            t: (B, 1, 1, 1, 1) current time step.
            omega: (B, 1, 1, 1, 1) CFG scale.
            y: (B,) integer class labels.

        Returns:
            (B, C, T, H, W) predicted instantaneous velocity.
        """
        # h = 0 and the full interval [0, 1] are dummies: only the v-head output
        # is read, and the v-head does not depend on the interval conditioning.
        h = torch.zeros_like(t)  # (B, 1, 1, 1, 1)
        t_min = torch.zeros_like(t)  # (B, 1, 1, 1, 1)
        t_max = torch.ones_like(t)  # (B, 1, 1, 1, 1)

        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]  # (B, C, T, H, W)

        return v

    def v_fn(self, x, t, omega, y):
        """Predict v with and without class conditioning in one batched call.

        Args:
            x: (B, C, T, H, W) noisy video latent at time t.
            t: (B, 1, 1, 1, 1) current time step.
            omega: (B, 1, 1, 1, 1) CFG scale.
            y: (B,) integer class labels.

        Returns:
            v_c: (B, C, T, H, W) v conditioned on the class labels.
            v_u: (B, C, T, H, W) v with labels replaced by the null class.
        """
        bz = x.shape[0]

        # Stack the conditional and unconditional halves into one 2B batch so
        # both run in a single network call.
        x = torch.cat([x, x], dim=0)  # (2B, C, T, H, W)
        y_null = torch.full((bz,), self.num_classes, device=y.device, dtype=y.dtype)
        #   y_null: (B,) the null-class index used for the unconditional branch
        y = torch.cat([y, y_null], dim=0)  # (2B,)
        t = torch.cat([t, t], dim=0)  # (2B, 1, 1, 1, 1)
        w = torch.cat([omega, torch.ones_like(omega)], dim=0)  # (2B, 1, 1, 1, 1)

        out = self.v_cond_fn(x, t, w, y)  # (2B, C, T, H, W)
        v_c, v_u = torch.chunk(out, 2, dim=0)  # each (B, C, T, H, W)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """Replace some labels with the null class so CFG has an unconditional arm.

        Args:
            v_t: (B, C, T, H, W) empirical velocity e - x.
            v_g: (B, C, T, H, W) guided velocity target.
            labels: (B,) integer class labels.

        Returns:
            labels: (B,) labels with the dropped entries set to num_classes.
            v_g: (B, C, T, H, W) target with v_t substituted on dropped entries,
                since an unconditional sample has no guidance to apply.

        Note: following the JAX reference in refs/imf.py, the number of drops is drawn
        randomly but the drops always land on the LEADING num_drop samples. Those
        are the same positions that sample_tr marks as flow-matching, so with
        data_proportion >= class_dropout_prob the dropped samples are a subset of
        the flow-matching ones and the mean-flow samples never see the null class.
        """
        bz = v_t.shape[0]

        rand_mask = (
            torch.rand(bz, device=v_t.device) < self.class_dropout_prob
        )  # (B,) bool
        num_drop = rand_mask.sum().to(torch.int32)  # scalar count of drops
        drop_mask = (
            torch.arange(bz, device=v_t.device)[:, None, None, None, None] < num_drop
        )  # (B, 1, 1, 1, 1) bool, True on the leading num_drop samples

        labels = torch.where(
            drop_mask.reshape(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: (B, C, T, H, W) empirical instantaneous velocity e - x.
            z_t: (B, C, T, H, W) noisy video latent at time t.
            t, r: (B, 1, 1, 1, 1) the two time steps; r is unused here and kept
                only to mirror the JAX reference signature.
            y: (B,) integer class labels.
            fm_mask: (B, 1, 1, 1, 1) bool, True on flow-matching (r = t) samples.
            w: (B, 1, 1, 1, 1) CFG scale.
            t_min, t_max: (B, 1, 1, 1, 1) guidance interval bounds.

        Returns:
            v_g: (B, C, T, H, W) guided velocity, the training target.
            v_c: (B, C, T, H, W) conditioned velocity, used as the jvp tangent
                along z.
        """

        # Ungated guidance: w applies at every t. This is the target used for the
        # flow-matching samples.
        v_c, v_u = self.v_fn(z_t, t, w, y=y)  # each (B, C, T, H, W)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)  # (B, C, T, H, W)

        # Interval-gated guidance: outside [t_min, t_max] fall back to w = 1,
        # i.e. no guidance.
        w = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))
        #   w: (B, 1, 1, 1, 1) gated scale

        v_c = self.v_cond_fn(z_t, t, w, y=y)  # (B, C, T, H, W)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)  # (B, C, T, H, W)

        # For flow matching samples, there is no CFG interval
        v_g = torch.where(fm_mask, v_g_fm, v_g)  # (B, C, T, H, W)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, latents, labels):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            latents: A batch of video latents, shape (B, C, T, H, W).
            labels: Corresponding class labels, shape (B,).

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = latents  # (B, C, T, H, W)
        bz = x.shape[0]
        device, dtype = x.device, x.dtype

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz, device, dtype)
        #   t, r: (B, 1, 1, 1, 1);  fm_mask: (B, 1, 1, 1, 1) bool

        # Linear interpolation path: z_t = (1-t) x + t e, so dz/dt = e - x.
        e = torch.randn_like(x)  # (B, C, T, H, W) Gaussian noise
        z_t = (1 - t) * x + t * e  # (B, C, T, H, W) state at time t
        v_t = e - x  # (B, C, T, H, W) empirical velocity

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, device, dtype, fm_mask)
        #   t_min, t_max: (B, 1, 1, 1, 1)
        omega = self.sample_cfg_scale(bz, device)  # (B, 1, 1, 1, 1) float32

        # Compute guided velocity v_g and conditioned velocity v_c
        # (no jvp / gradient needed through the guidance targets)
        with torch.no_grad():
            v_g, v_c = self.guidance_fn(
                v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
            )
            #   v_g, v_c: (B, C, T, H, W)

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)
        #   labels: (B,);  v_g: (B, C, T, H, W)

        # Warped u-function for jvp computation.
        # torch.func.jvp has_aux semantics: u_fn returns (output, aux),
        # jvp returns (output, output_tangent, aux). The v-head prediction
        # is the aux — its tangent is never formed.
        def u_fn(z_t, t, r):
            u, v = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)
            return u, v

        # Tangents: dt/dt = 1 and dr/dt = 0, so the total derivative below is
        # taken with respect to t at fixed r.
        dtdt = torch.ones_like(t)  # (B, 1, 1, 1, 1)
        dtdr = torch.zeros_like(t)  # (B, 1, 1, 1, 1)

        # Different from original MeanFlow, we use predicted v in the jvp
        if self.jvp_impl == "fast":
            # du/dt is used only inside stop-gradient, so the tangent pass
            # can be fully detached: a plain forward (with autograd graph)
            # supplies u and v for the loss, and the hand-rolled Triton
            # forward-mode pass (models/mla_jvp_fast) supplies du/dt with
            # no backward graph at all.
            from models.mla_jvp_fast import (
                build_fast_jvp_state,
                model_du_dt_fast,
            )

            u, v = u_fn(z_t, t, r)
            #   u, v: (B, C, T, H, W), carry the loss gradient
            with torch.no_grad():
                state = build_fast_jvp_state(self.net)
                flat = lambda s: s.reshape(s.shape[0])  # (B,1,1,1,1) -> (B,)
                du_dt = model_du_dt_fast(
                    self.net,
                    state,
                    z_t,
                    flat(t),
                    flat(t - r),
                    flat(omega),
                    flat(t_min),
                    flat(t_max),
                    labels,
                    v_c,
                )
                #   du_dt: (B, C, T, H, W), detached by construction
        else:
            u, du_dt, v = torch.func.jvp(
                u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True
            )
        #   u: (B, C, T, H, W) average velocity
        #   du_dt: (B, C, T, H, W) total time derivative of u along the path
        #   v: (B, C, T, H, W) v-head prediction (jvp aux, no tangent formed)

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * du_dt.detach()  # (B, C, T, H, W)

        v_g = v_g.detach()  # (B, C, T, H, W) target carries no gradient

        # improved MeanFlow objective is conceptually v-loss
        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2, 3, 4))  # (B,)
        # auxiliary v-head loss
        loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2, 3, 4))  # (B,)
        loss, raw_loss = adaptive_loss_metrics(
            loss_u,
            loss_v,
            norm_eps=self.norm_eps,
            norm_p=self.norm_p,
            loss_v_weight=self.loss_v_weight,
            elements_per_sample=V[0].numel(),
        )

        dict_losses = {
            "loss": loss.detach(),
            "loss_raw": raw_loss.detach(),
            "loss_u": loss_u.mean().div(V[0].numel()).detach(),
            "loss_v": loss_v.mean().div(V[0].numel()).detach(),
        }

        return loss, dict_losses

    #######################################################
    #                       Sampling                      #
    #######################################################

    @torch.no_grad()
    def sample_one_step(self, z_t, labels, t, r, omega, t_min, t_max):
        """One deterministic MeanFlow step from time t down to time r < t.

        The average-velocity field u integrates the probability-flow ODE over
        the whole interval in closed form, so a single network call moves the
        state across [r, t]:  z_r = z_t - (t - r) * u(z_t, t, t - r).

        Args:
            z_t: (B, C, T, H, W) current state at time t.
            labels: (B,) integer class labels.
            t: (B, 1, 1, 1, 1) current time (1 = pure noise).
            r: (B, 1, 1, 1, 1) target time, r < t (0 = data).
            omega: (B, 1, 1, 1, 1) CFG (classifier-free guidance) scale.
            t_min, t_max: (B, 1, 1, 1, 1) guidance interval bounds.

        Returns:
            (B, C, T, H, W) state at time r.
        """
        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]
        return z_t - (t - r) * u

    @torch.no_grad()
    def sample(
        self,
        z_1,
        labels,
        num_steps=None,
        omega=None,
        t_min=None,
        t_max=None,
        t_steps=None,
    ):
        """Generate latents by integrating from t = 1 (noise) to t = 0 (data).

        Args:
            z_1: (B, C, T, H, W) initial Gaussian noise, i.e. the state at t = 1.
            labels: (B,) integer class labels.
            num_steps: number of MeanFlow steps; 1 gives one-step generation.
                None takes config.sample.num_steps.
            omega: scalar CFG scale used at every step. None takes
                config.sample.omega.
            t_min, t_max: scalar guidance interval bounds. None takes
                config.sample.t_min / t_max.
            t_steps: optional (num_steps + 1,) strictly decreasing schedule from
                1.0 to 0.0. Defaults to a uniform grid.

        Returns:
            (B, C, T, H, W) generated latents at t = 0.
        """
        # Per CLAUDE.md every hyperparameter lives in config.py; these arguments
        # are call-site overrides, not a second source of defaults.
        from config import config as _cfg

        num_steps = _cfg.sample.num_steps if num_steps is None else num_steps
        omega = _cfg.sample.omega if omega is None else omega
        t_min = _cfg.sample.t_min if t_min is None else t_min
        t_max = _cfg.sample.t_max if t_max is None else t_max
        device, dtype = z_1.device, z_1.dtype
        bz = z_1.shape[0]

        if t_steps is None:
            t_steps = torch.linspace(
                1.0, 0.0, num_steps + 1, device=device, dtype=dtype
            )  # (num_steps + 1,)
        else:
            t_steps = torch.as_tensor(t_steps, device=device, dtype=dtype)

        def scalar_to_batch(value):
            """Broadcast a python scalar to (B, 1, 1, 1, 1)."""
            return torch.full((bz, 1, 1, 1, 1), value, device=device, dtype=dtype)

        omega_b = scalar_to_batch(float(omega))  # (B, 1, 1, 1, 1)
        t_min_b = scalar_to_batch(float(t_min))  # (B, 1, 1, 1, 1)
        t_max_b = scalar_to_batch(float(t_max))  # (B, 1, 1, 1, 1)

        z_t = z_1
        for i in range(t_steps.shape[0] - 1):
            t = t_steps[i].expand(bz).reshape(bz, 1, 1, 1, 1)
            r = t_steps[i + 1].expand(bz).reshape(bz, 1, 1, 1, 1)
            z_t = self.sample_one_step(z_t, labels, t, r, omega_b, t_min_b, t_max_b)

        return z_t
