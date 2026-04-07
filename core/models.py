"""core/models.py — Surrogate model architectures.

Four models are provided:

NSFModel   — Neural Spline Flow (conditional, 1-D).
             Handles dips; no physics initialisation.

MDNModel   — Skew-t Mixture Density Network with optional Cauchy components.
             Physics-initialised; positive-definite; cannot represent dips.

EBMModel   — Conditional Energy-Based Model (option 2 from the design chat).
             log p(y|θ) = f(y, embed(θ)) − log Z(θ).
             Z(θ) computed exactly via Gauss-Legendre quadrature in 1-D.
             Handles dips; admits physics initialisation; simpler than NSF.

MDNCorrModel — MDN with log-correction (option 3):
             p(y|θ) ∝ MDN(y|θ) · exp(g(y,θ)),  g initialised to 0.
             Backward-compatible with MDN checkpoints; dip-capable;
             least invasive extension.
"""

import math
import numpy as np
import torch
import torch.nn as nn

try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    _NFLOWS_OK = True
except ImportError:
    _NFLOWS_OK = False

import core.config as cfg
from core.config import SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Two-layer residual block with ELU activations."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# NSFModel
# ─────────────────────────────────────────────────────────────────────────────

class NSFModel(nn.Module):
    """Conditional Neural Spline Flow for 1-D mHH.

    Architecture
    ~~~~~~~~~~~~
    Base distribution : N(0,1) in 1-D latent space.
    Transform         : `num_layers` stacked
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform layers.
        For 1-D data, autoregressive == coupling (no masking ambiguity).
    Conditioning      : raw context → 3-layer MLP → embed_dim embedding
                        fed to each flow layer's internal conditioner.
    y-space           : y = (mHH − threshold) / SCALE.
    tail_bound        : spline range [−tb, +tb]; linear tails outside.
                        Set >= max(y) with 20% margin (auto-tuned by
                        auto_nsf_hyperparams).

    The spline learns the full conditional density without any structural
    constraint, so it can represent dips, asymmetric peaks, and multi-modal
    distributions.  The price is that it has no physics initialisation —
    prior samples (from feature tracks or comments) are injected during
    training to compensate.
    """

    def __init__(
        self,
        context_dim:   int   = -1,
        num_bins:      int   = 32,
        num_layers:    int   = 6,
        hidden_dim:    int   = 128,
        num_blocks:    int   = 2,
        tail_bound:    float = 6.0,
        embed_dim:     int   = 64,
        analytic_shift: bool = False,
        threshold_y:   float = 0.0,
        bw_mass_axis:  int   = 0,
    ):
        super().__init__()
        if not _NFLOWS_OK:
            raise ImportError('nflows is required for NSFModel. pip install nflows')

        if context_dim < 0:
            context_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2

        self.analytic_shift = bool(analytic_shift)
        self.threshold_y    = float(threshold_y)
        self.bw_mass_axis   = int(bw_mass_axis)

        # Context encoder: maps normalised coords → learned embedding.
        # This is the key to smooth interpolation — the flow sees a rich
        # learned representation rather than raw coordinate values.
        self._ctx_encoder = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )

        transforms = [
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features        = 1,
                hidden_features = hidden_dim,
                context_features= embed_dim,
                num_bins        = num_bins,
                num_blocks      = num_blocks,
                tails           = 'linear',
                tail_bound      = tail_bound,
                use_residual_blocks = True,
                use_batch_norm      = False,
                activation          = nn.functional.elu,
            )
            for _ in range(num_layers)
        ]

        self._flow = Flow(
            transform    = CompositeTransform(transforms),
            distribution = StandardNormal([1]),
        )
        self.tail_bound = tail_bound

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.analytic_shift:
            # Shift y by the expected BW peak position: z = y - max(0, ctx_mass - threshold_y)
            mu = (context[:, self.bw_mass_axis:self.bw_mass_axis+1] - self.threshold_y).clamp(min=0.0)
            y_in = y - mu
        else:
            y_in = y
        return self._flow.log_prob(y_in, context=self._ctx_encoder(context))

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        z = self._flow.sample(num_samples, context=self._ctx_encoder(context))
        if self.analytic_shift:
            mu = (context[:, self.bw_mass_axis] - self.threshold_y).clamp(min=0.0)
            return z + mu.unsqueeze(1)
        return z

    def forward(self, y, context):
        return self.log_prob(y, context)


def nsf_nll_loss(flow, y, context, weights):
    """Weighted NLL for NSF.  weights are Voronoi / histogram importance weights."""
    log_p  = flow.log_prob(y, context=context)
    log_p  = log_p.nan_to_num(nan=-100.0, posinf=0.0, neginf=-100.0)
    w_norm = weights / (weights.mean() + 1e-12)
    return -(log_p * w_norm).mean()


def _mdn_analytic_bw_kwargs(n_cauchy):
    """Return kwargs for MDNModel enabling analytic BW mu/sigma, or {} if not applicable.

    Active when PARAM_SPACINGS contains at least one (linear, log) pair and n_cauchy >= 1.
      2-D database (mS1, WoMS1): mass_axes=[0], wom_axes=[1]
      4-D database (mS1, WoMS1, mS2, WoMS2): mass_axes=[0, 2], wom_axes=[1, 3]
    """
    if n_cauchy == 0 or not cfg.PARAM_SPACINGS:
        return {}
    pairs = []
    i = 0
    while i + 1 < len(cfg.PARAM_SPACINGS) and len(pairs) < n_cauchy:
        if cfg.PARAM_SPACINGS[i] == 'linear' and cfg.PARAM_SPACINGS[i + 1] == 'log':
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    if not pairs:
        return {}
    while len(pairs) < n_cauchy:
        pairs.append(pairs[-1])
    thr_y = (cfg.MHH_THRESHOLD / SCALE) if cfg.MHH_THRESHOLD is not None else 0.0
    return {
        'analytic_bw':  True,
        'threshold_y':  thr_y,
        'bw_mass_axes': [p[0] for p in pairs[:n_cauchy]],
        'bw_wom_axes':  [p[1] for p in pairs[:n_cauchy]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MDNModel
# ─────────────────────────────────────────────────────────────────────────────

class MDNModel(nn.Module):
    """Skew-t Mixture Density Network.

    n_gaussians skew-t components + n_cauchy pure Cauchy (Lorentzian) components.
    The Cauchy components are initialised from BW peak tracks and have their
    nu pinned to 1 and alpha pinned to 0 in forward() so only mu/sigma learn.

    Positive-definite by construction → cannot represent dips.
    Physics-initialised via _init_mdn_from_tracks() in core/training.py.
    """

    _alpha_max = 2.5
    _nu_min    = 4.0   # higher floor → faster tail decay, prevents far-tail leakage

    def __init__(self, in_dim=-1, n_gaussians=14, n_cauchy=2,
                 hidden_dim=512, n_blocks=4, min_sigma=1e-4,
                 analytic_bw=False, threshold_y=0.0,
                 bw_mass_axes=None, bw_wom_axes=None):
        super().__init__()
        if in_dim < 0:
            in_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2
        self.min_sigma   = min_sigma
        self.n_gaussians = n_gaussians
        self.n_cauchy    = n_cauchy
        _total           = n_gaussians + n_cauchy

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.act        = nn.ELU()
        self.pi    = nn.Linear(hidden_dim, _total)
        self.mu    = nn.Linear(hidden_dim, _total)
        self.sigma = nn.Linear(hidden_dim, _total)
        self.alpha = nn.Linear(hidden_dim, _total)
        self.nu    = nn.Linear(hidden_dim, _total)
        self.softplus = nn.Softplus()

        # Analytic BW mode: mu and sigma for the last n_cauchy (Cauchy) components are
        # overridden in forward() with physics-exact values:
        #   mu_k    = x[:, mass_ax] - threshold_y        (peak at mS in y-space)
        #   sigma_k = 0.5 * x[:, mass_ax] * exp(x[:, wom_ax])   (HWHM = WoMS*mS/2/SCALE)
        # x[:, mass_ax] = mS/SCALE (linear encoding), x[:, wom_ax] = log(WoMS) (log encoding).
        self.analytic_bw  = bool(analytic_bw)
        self.threshold_y  = float(threshold_y)
        self.bw_mass_axes = list(bw_mass_axes) if bw_mass_axes else [0] * n_cauchy
        self.bw_wom_axes  = list(bw_wom_axes)  if bw_wom_axes  else [1] * n_cauchy

    def forward(self, x):
        h = self.act(self.input_proj(x))
        for blk in self.res_blocks:
            h = self.act(blk(h))
        pi    = torch.softmax(self.pi(h), dim=1)
        mu    = self.mu(h)
        sigma = torch.clamp(self.softplus(self.sigma(h)), min=self.min_sigma)
        alpha = self._alpha_max * torch.tanh(self.alpha(h))
        nu    = self._nu_min + self.softplus(self.nu(h))
        if self.n_cauchy > 0:
            nu = torch.cat([
                nu[:, :self.n_gaussians],
                torch.ones(nu.shape[0], self.n_cauchy, dtype=nu.dtype, device=nu.device),
            ], dim=1)
            alpha = torch.cat([
                alpha[:, :self.n_gaussians],
                torch.zeros(alpha.shape[0], self.n_cauchy, dtype=alpha.dtype, device=alpha.device),
            ], dim=1)
            if self.analytic_bw:
                mu_cols    = list(mu.unbind(1))
                sigma_cols = list(sigma.unbind(1))
                for ci in range(self.n_cauchy):
                    col     = self.n_gaussians + ci
                    mass_ax = self.bw_mass_axes[ci]
                    wom_ax  = self.bw_wom_axes[ci]
                    mu_cols[col]    = x[:, mass_ax] - self.threshold_y
                    sigma_cols[col] = (0.5 * x[:, mass_ax] * torch.exp(x[:, wom_ax])).clamp(min=self.min_sigma)
                mu    = torch.stack(mu_cols,    dim=1)
                sigma = torch.stack(sigma_cols, dim=1)
        return pi, mu, sigma, alpha, nu


def _student_t_log_cdf(w, df):
    try:
        x   = (df / (df + w.pow(2) + 1e-8)).clamp(1e-7, 1 - 1e-7)
        hp  = 0.5 * torch.special.betainc(
            (0.5 * df).clamp(min=0.5),
            torch.full_like(df, 0.5), x)
        cdf = torch.where(w >= 0, 1.0 - hp, hp)
    except (AttributeError, NotImplementedError):
        cdf = 0.5 * (1.0 + torch.erf(w / math.sqrt(2)))
    return torch.log(cdf.clamp(min=1e-38))


def mdn_loss(pi, mu, sigma, alpha, nu, y, voronoi_w=None):
    """Weighted negative log-likelihood for the skew-t MDN."""
    y_exp  = y.expand_as(mu)
    z      = (y_exp - mu) / sigma
    log_t  = (torch.lgamma(0.5 * (nu + 1))
              - torch.lgamma(0.5 * nu)
              - 0.5 * torch.log(math.pi * nu)
              - 0.5 * (nu + 1) * torch.log1p(z.pow(2) / (nu + 1e-8))
              - torch.log(sigma))
    w_cdf   = alpha * z * torch.sqrt((nu + 1.0) / (nu + z.pow(2) + 1e-8))
    log_cdf = _student_t_log_cdf(w_cdf, nu + 1.0)
    log_p   = torch.logsumexp(torch.log(pi + 1e-12) + math.log(2) + log_t + log_cdf, dim=1)
    ws = torch.ones_like(log_p)
    if voronoi_w is not None:
        ws = voronoi_w
    ws = ws / (ws.mean() + 1e-12)
    return -(ws * log_p).mean()


# ─────────────────────────────────────────────────────────────────────────────
# EBMModel  (Conditional Energy-Based Model — option 2)
# ─────────────────────────────────────────────────────────────────────────────

class EBMModel(nn.Module):
    """Conditional Energy-Based Model for 1-D mHH.

    Model
    ~~~~~
        log p(y | θ) = f(y, embed(θ)) − log Z(θ)

    where f is an unconstrained MLP and the partition function

        Z(θ) = ∫_{y_lo}^{y_hi} exp(f(y, embed(θ))) dy

    is computed **exactly** via Gauss-Legendre quadrature on n_bins points.
    Because y (= mHH transformed) is 1-D, the quadrature is microseconds on
    CPU and the gradient ∂ log Z / ∂w is obtained by differentiating through
    the quadrature sum — no MCMC, no sampling required during training.

    Advantages over MDN
    ~~~~~~~~~~~~~~~~~~~
    - f is unconstrained → can represent dips (negative interference)
    - Physics initialisation is trivial: init f ≈ log(BW density) by setting
      the output layer bias to a Cauchy log-likelihood at the BW peak position
      (see _init_ebm_from_tracks() in core/training.py)

    Advantages over NSF
    ~~~~~~~~~~~~~~~~~~~
    - No normalising-flow Jacobian: simpler architecture, no spline knots
    - Physics init is natural (init f ≈ log-density, no knot positions)
    - Smoother gradient landscape: NLL = -f.mean() + log_Z is convex in the
      quadrature approximation for fixed embed output

    Disadvantage
    ~~~~~~~~~~~~
    - Sampling requires CDF inversion, but we evaluate p on a bin grid for
      morphing and do not need fast sampling in the critical path.
      sample_model() falls back to inverse-CDF on the GL grid (fast).

    Parameters
    ----------
    context_dim : int   Dimension of the parameter context (auto from PARAM_NAMES)
    hidden_dim  : int   Width of the energy MLP
    n_blocks    : int   Depth of the energy MLP (ResBlocks)
    embed_dim   : int   Width of the context embedding
    n_bins      : int   Number of GL quadrature points (256–1024)
    y_lo, y_hi  : float Integration range in y-space.  Set to [0, tail_bound].
                        Auto-set from tail_bound after construction if 0/0.
    """

    def __init__(
        self,
        context_dim:  int   = -1,
        hidden_dim:   int   = 256,
        n_blocks:     int   = 4,
        embed_dim:    int   = 64,
        n_bins:       int   = 512,
        y_lo:         float = 0.0,
        y_hi:         float = 6.0,
        analytic_bw:  bool  = False,
        threshold_y:  float = 0.0,
        bw_mass_axis: int   = 0,
        bw_wom_axis:  int   = 1,
    ):
        super().__init__()
        if context_dim < 0:
            context_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2

        self.n_bins      = n_bins
        self.y_lo        = y_lo
        self.y_hi        = y_hi
        self.analytic_bw  = bool(analytic_bw)
        self.threshold_y  = float(threshold_y)
        self.bw_mass_axis = int(bw_mass_axis)
        self.bw_wom_axis  = int(bw_wom_axis)

        # Context encoder
        self._ctx_encoder = nn.Sequential(
            nn.Linear(context_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim,   embed_dim), nn.ELU(),
            nn.Linear(embed_dim,   embed_dim), nn.ELU(),
        )

        # Unconstrained energy MLP f(y, ctx_emb).
        # f is pure ML — no physics prior baked in.
        # The Cauchy-CDF change of variables in _log_Z_unique handles
        # accurate integration over narrow peaks without modifying f.
        self._energy_in  = nn.Linear(1 + embed_dim, hidden_dim)
        self._energy_res = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self._energy_out = nn.Linear(hidden_dim, 1)
        self._act        = nn.ELU()

        # Gauss-Legendre nodes.
        # When analytic_bw=False: map to y-space [y_lo, y_hi] (fixed nodes).
        # When analytic_bw=True : store raw t_k ∈ [-1,1] and gl_w_k;
        #   at runtime, map to [y_lo, y_hi] via the truncated Cauchy CDF so
        #   that nodes concentrate near the BW peak regardless of peak width.
        #   This is a change of variables for the INTEGRAL only — f is unchanged.
        gl_t, gl_w = np.polynomial.legendre.leggauss(n_bins)
        if not analytic_bw:
            mid   = 0.5 * (y_hi + y_lo)
            half  = 0.5 * (y_hi - y_lo)
            nodes   = (mid + half * gl_t).astype(np.float32)
            weights = (half * gl_w).astype(np.float32)
            self.register_buffer('_gl_nodes',   torch.from_numpy(nodes))
            self.register_buffer('_gl_weights', torch.from_numpy(weights))
        else:
            # Raw GL nodes in [-1, 1] and their weights (sum to 2).
            # y-positions computed per-context at runtime.
            self.register_buffer('_gl_t', torch.from_numpy(gl_t.astype(np.float32)))
            self.register_buffer('_gl_w', torch.from_numpy(gl_w.astype(np.float32)))
            # Dummy _gl_nodes/_gl_weights so existing callers don't crash.
            self.register_buffer('_gl_nodes',   torch.zeros(n_bins, dtype=torch.float32))
            self.register_buffer('_gl_weights', torch.zeros(n_bins, dtype=torch.float32))

    # ── Cauchy helpers (analytic_bw only) ───────────────────────────────────

    def _bw_mu_sigma(self, ctx_raw: torch.Tensor):
        """Cauchy peak mu and HWHM sigma from raw context (shape (K, ctx_dim) → (K,), (K,))."""
        mu    = ctx_raw[:, self.bw_mass_axis] - self.threshold_y
        sigma = (0.5 * ctx_raw[:, self.bw_mass_axis]
                 * torch.exp(ctx_raw[:, self.bw_wom_axis])).clamp(min=1e-6)
        return mu, sigma

    # ── energy and partition function ────────────────────────────────────────

    def _f(self, y: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        """Unconstrained energy f(y, ctx_emb) — pure ML, no physics prior."""
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if ctx_emb.shape[0] == 1 and y.shape[0] > 1:
            ctx_emb = ctx_emb.expand(y.shape[0], -1)
        h = torch.cat([y, ctx_emb], dim=1)
        h = self._act(self._energy_in(h))
        for blk in self._energy_res:
            h = self._act(blk(h))
        return self._energy_out(h).squeeze(1)

    def _log_Z(self, ctx_emb: torch.Tensor) -> torch.Tensor:
        """log Z via GL quadrature (fixed y-space nodes, non-analytic_bw path only)."""
        nodes  = self._gl_nodes.unsqueeze(1)
        emb    = ctx_emb.expand(self.n_bins, -1)
        f_vals = self._f(nodes, emb)
        log_w  = torch.log(self._gl_weights.clamp(min=1e-38))
        return torch.logsumexp(f_vals + log_w, dim=0)

    # ── public interface ─────────────────────────────────────────────────────

    def _log_Z_unique(self, unique_ctx: torch.Tensor) -> torch.Tensor:
        """GL quadrature for K unique contexts.

        When analytic_bw=False: fixed y-space nodes on [y_lo, y_hi].

        When analytic_bw=True: Cauchy-CDF change of variables, truncated to
        [y_lo, y_hi].  Raw GL nodes t_k ∈ [-1,1] are mapped:
            F_lo = 0.5 + arctan((y_lo−μ)/σ)/π
            F_hi = 0.5 + arctan((y_hi−μ)/σ)/π
            u_k  = (F_lo+F_hi)/2 + (F_hi−F_lo)/2 × t_k      ← u in [F_lo,F_hi]
            y_k  = μ + σ × tan(π(u_k−0.5))
            log Jacobian = log((F_hi−F_lo)/2 × π × σ) − 2 log|cos(π(u_k−0.5))|

        GL weight: the standard (F_hi−F_lo)/2 × gl_w_k factor is absorbed into
        log_jac so it appears inside the logsumexp.

        When f=0 (init): log_Z = log(y_hi−y_lo) exactly, giving uniform density.
        The quadrature concentrates nodes near μ so even σ≈0.0025 y-units is resolved.
        """
        K = unique_ctx.shape[0]
        N = self.n_bins
        unique_emb = self._ctx_encoder(unique_ctx)  # (K, embed_dim)

        if not self.analytic_bw:
            nodes_r = (self._gl_nodes.unsqueeze(1).unsqueeze(0)
                       .expand(K, N, 1).reshape(K * N, 1))
            emb_r   = (unique_emb.unsqueeze(1).expand(K, N, -1)
                       .reshape(K * N, -1))
            f_vals  = self._f(nodes_r, emb_r)
            log_w   = torch.log(self._gl_weights.clamp(min=1e-38))
            log_w_r = log_w.unsqueeze(0).expand(K, N).reshape(K * N)
            return torch.logsumexp((f_vals + log_w_r).reshape(K, N), dim=1)

        # Cauchy-CDF change of variables
        mu_K, sigma_K = self._bw_mu_sigma(unique_ctx)   # (K,)
        t_k = self._gl_t                                 # (N,) in [-1,1]
        gl_w = self._gl_w                                # (N,) sums to 2

        F_lo = 0.5 + torch.atan((self.y_lo - mu_K) / sigma_K) / math.pi   # (K,)
        F_hi = 0.5 + torch.atan((self.y_hi - mu_K) / sigma_K) / math.pi   # (K,)
        mid_F  = (F_lo + F_hi) * 0.5                                        # (K,)
        half_F = (F_hi - F_lo) * 0.5                                        # (K,)

        # u_k(ctx): (K, N)
        u_kn    = mid_F.unsqueeze(1) + half_F.unsqueeze(1) * t_k.unsqueeze(0)
        angle   = math.pi * (u_kn - 0.5)                                    # (K, N)
        cos_sq  = torch.cos(angle).pow(2).clamp(min=1e-12)                  # (K, N)
        y_kn    = mu_K.unsqueeze(1) + sigma_K.unsqueeze(1) * torch.tan(angle)  # (K, N)

        # Log Jacobian: log((F_hi-F_lo)/2 * π * σ / cos²)
        # cos_sq = cos²(angle), so log(sec²) = -log(cos_sq) — NOT -2*log(cos_sq)
        log_jac = (torch.log(half_F.clamp(min=1e-12)).unsqueeze(1)
                   + math.log(math.pi)
                   + torch.log(sigma_K.clamp(min=1e-12)).unsqueeze(1)
                   - torch.log(cos_sq))                                      # (K, N)

        log_w_k = torch.log(gl_w.clamp(min=1e-38))                          # (N,)
        y_r     = y_kn.reshape(K * N, 1)
        emb_r   = unique_emb.unsqueeze(1).expand(K, N, -1).reshape(K * N, -1)
        f_vals  = self._f(y_r, emb_r)                                       # (K*N,)
        integrand = (f_vals.reshape(K, N) + log_jac + log_w_k.unsqueeze(0)) # (K, N)
        return torch.logsumexp(integrand, dim=1)                             # (K,)

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """log p(y | context) = f(y, emb) − log Z(emb).

        y       : (B, 1)
        context : (B, context_dim)
        Returns : (B,) log-probabilities.
        """
        emb   = self._ctx_encoder(context)          # (B, embed_dim)
        f_y   = self._f(y, emb)                     # (B,)
        # Deduplicate: during training at most n_grid_points (~18) unique contexts
        # exist per minibatch.  Run GL quadrature only for unique ones.
        unique_ctx, inv = torch.unique(context, dim=0, return_inverse=True)
        log_z_unique    = self._log_Z_unique(unique_ctx)                     # (K,)
        log_z           = log_z_unique[inv]                                  # (B,)
        return f_y - log_z

    def forward(self, y, context):
        return self.log_prob(y, context)

    def pdf_on_grid(self, context_enc: list, y_grid: np.ndarray) -> np.ndarray:
        """Evaluate normalised PDF on a numpy y-grid for a single context."""
        self.eval()
        with torch.no_grad():
            ctx = torch.tensor([context_enc], dtype=torch.float32,
                               device=self._gl_nodes.device)
            emb = self._ctx_encoder(ctx)
            y_t = torch.tensor(y_grid, dtype=torch.float32,
                               device=ctx.device).unsqueeze(1)
            f_y   = self._f(y_t, emb)
            lz    = self._log_Z_unique(ctx).squeeze(0)
            log_p = f_y - lz
        return log_p.exp().cpu().numpy()


def ebm_nll_loss(model: EBMModel, y, context, weights):
    """Weighted NLL for EBM.

    For efficiency during training, log Z is computed once per unique context
    in the batch using GL quadrature (already implemented in log_prob).
    The gradient flows through log_Z via the quadrature sum.
    """
    log_p  = model.log_prob(y, context)
    w_norm = weights / (weights.mean() + 1e-12)
    return -(log_p * w_norm).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MDNCorrModel  (MDN + exp(g) correction — option 3)
# ─────────────────────────────────────────────────────────────────────────────

class MDNCorrModel(nn.Module):
    """MDN with a signed log-correction network.

    p(y | θ) ∝ MDN(y | θ) · exp(g(y, θ))

    where g is a small MLP initialised to output 0, so at epoch 0 this
    reduces exactly to the base MDN.  The partition function

        Z_g(θ) = ∫ MDN(y|θ) exp(g(y,θ)) dy

    is computed via GL quadrature on n_bins points in y-space.

    The MDN part is physics-initialised (positive-definite BW prior).
    The g-network only needs to learn the deviation from the MDN shape,
    which is a much easier task than learning the full density from scratch.
    Dips appear when g assigns a strong negative value near the resonance
    peak — g is unconstrained so negative values are allowed.

    Training is done in two phases:
        Phase 1: train only MDN weights (g frozen at 0)   → fast convergence
        Phase 2: fine-tune g while the MDN weights are free to adapt

    This is handled by select_and_train() in core/training.py.
    """

    def __init__(self, in_dim=-1, n_gaussians=14, n_cauchy=2,
                 hidden_dim=512, n_blocks_mdn=4,
                 g_hidden=64, g_blocks=2,
                 n_bins=512, y_lo=0.0, y_hi=6.0,
                 analytic_bw=False, threshold_y=0.0,
                 bw_mass_axis=0, bw_wom_axis=1):
        super().__init__()
        if in_dim < 0:
            in_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2

        self.n_cauchy     = n_cauchy
        self.n_gaussians  = n_gaussians
        self.analytic_bw  = bool(analytic_bw)
        self.threshold_y  = float(threshold_y)
        self.bw_mass_axis = int(bw_mass_axis)
        self.bw_wom_axis  = int(bw_wom_axis)
        self.y_lo         = y_lo
        self.y_hi         = y_hi
        self.n_bins       = n_bins

        # Base MDN (analytic skew-t mixture; g corrects residuals)
        mdn_kw = _mdn_analytic_bw_kwargs(n_cauchy) if analytic_bw else {}
        self._mdn = MDNModel(
            in_dim     = in_dim,
            n_gaussians= n_gaussians,
            n_cauchy   = n_cauchy,
            hidden_dim = hidden_dim,
            n_blocks   = n_blocks_mdn,
            **mdn_kw,
        )

        # g(y, context) — log-correction network
        embed_dim = 64
        self._ctx_enc_g = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim), nn.ELU(),
        )
        self._g_in  = nn.Linear(1 + embed_dim, g_hidden)
        self._g_res = nn.ModuleList([ResBlock(g_hidden) for _ in range(g_blocks)])
        self._g_out = nn.Linear(g_hidden, 1)
        self._act   = nn.ELU()

        nn.init.zeros_(self._g_out.weight)
        nn.init.zeros_(self._g_out.bias)

        # GL quadrature for Z_g = ∫ MDN(y|θ) exp(g(y,θ)) dy.
        # analytic_bw=False: fixed y-space nodes on [y_lo, y_hi].
        # analytic_bw=True : raw t_k ∈ [-1,1]; per-context y-nodes via
        #   truncated Cauchy CDF.  The MDN's analytic_bw Cauchy component already
        #   concentrates the MDN density near the peak; the CDF mapping makes the
        #   quadrature accurate for the narrow peak in Z_g as well.
        gl_t, gl_w = np.polynomial.legendre.leggauss(n_bins)
        if not analytic_bw:
            mid, half = 0.5*(y_hi+y_lo), 0.5*(y_hi-y_lo)
            nodes   = (mid + half * gl_t).astype(np.float32)
            weights = (half * gl_w).astype(np.float32)
            self.register_buffer('_gl_nodes',   torch.from_numpy(nodes))
            self.register_buffer('_gl_weights', torch.from_numpy(weights))
        else:
            self.register_buffer('_gl_t', torch.from_numpy(gl_t.astype(np.float32)))
            self.register_buffer('_gl_w', torch.from_numpy(gl_w.astype(np.float32)))
            self.register_buffer('_gl_nodes',   torch.zeros(n_bins, dtype=torch.float32))
            self.register_buffer('_gl_weights', torch.zeros(n_bins, dtype=torch.float32))

    # ── g network ────────────────────────────────────────────────────────────

    def _g(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        emb = self._ctx_enc_g(context)
        if emb.shape[0] == 1 and y.shape[0] > 1:
            emb = emb.expand(y.shape[0], -1)
        h = self._act(self._g_in(torch.cat([y, emb], dim=1)))
        for blk in self._g_res:
            h = self._act(blk(h))
        return self._g_out(h).squeeze(1)

    # ── MDN log-density (analytic) ───────────────────────────────────────────

    def _mdn_log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Evaluate MDN log p(y|θ) without the g correction."""
        pi, mu, sigma, alpha, nu = self._mdn(context)
        y_exp  = y.expand_as(mu)
        z      = (y_exp - mu) / sigma
        log_t  = (torch.lgamma(0.5*(nu+1)) - torch.lgamma(0.5*nu)
                  - 0.5*torch.log(math.pi*nu)
                  - 0.5*(nu+1)*torch.log1p(z.pow(2)/(nu+1e-8))
                  - torch.log(sigma))
        w_cdf  = alpha * z * torch.sqrt((nu+1.)/(nu+z.pow(2)+1e-8))
        lcdf   = _student_t_log_cdf(w_cdf, nu+1.)
        return torch.logsumexp(torch.log(pi+1e-12) + math.log(2) + log_t + lcdf, dim=1)

    # ── partition function Z_g ────────────────────────────────────────────────

    def _log_Z_g_unique(self, unique_ctx: torch.Tensor) -> torch.Tensor:
        """GL quadrature for K unique contexts.

        analytic_bw=True: Cauchy-CDF change of variables — same scheme as
        EBMModel._log_Z_unique.  The MDN's analytic Cauchy component defines
        mu(ctx) and sigma(ctx) for the quadrature mapping.
        """
        K = unique_ctx.shape[0]
        N = self.n_bins

        if not self.analytic_bw:
            nodes_r = (self._gl_nodes.unsqueeze(1).unsqueeze(0)
                       .expand(K, N, 1).reshape(K * N, 1))
            ctx_r   = unique_ctx.unsqueeze(1).expand(K, N, -1).reshape(K * N, -1)
            lp_mdn  = self._mdn_log_prob(nodes_r, ctx_r)
            g_vals  = self._g(nodes_r, ctx_r)
            log_w   = torch.log(self._gl_weights.clamp(min=1e-38))
            log_w_r = log_w.unsqueeze(0).expand(K, N).reshape(K * N)
            integrand = (lp_mdn + g_vals + log_w_r).reshape(K, N)
            return torch.logsumexp(integrand, dim=1)

        # Cauchy-CDF change of variables (same geometry as EBMModel)
        mu_K    = unique_ctx[:, self.bw_mass_axis] - self.threshold_y
        sigma_K = (0.5 * unique_ctx[:, self.bw_mass_axis]
                   * torch.exp(unique_ctx[:, self.bw_wom_axis])).clamp(min=1e-6)
        t_k  = self._gl_t
        gl_w = self._gl_w
        F_lo   = 0.5 + torch.atan((self.y_lo - mu_K) / sigma_K) / math.pi
        F_hi   = 0.5 + torch.atan((self.y_hi - mu_K) / sigma_K) / math.pi
        mid_F  = (F_lo + F_hi) * 0.5
        half_F = (F_hi - F_lo) * 0.5
        u_kn   = mid_F.unsqueeze(1) + half_F.unsqueeze(1) * t_k.unsqueeze(0)
        angle  = math.pi * (u_kn - 0.5)
        cos_sq = torch.cos(angle).pow(2).clamp(min=1e-12)
        y_kn   = mu_K.unsqueeze(1) + sigma_K.unsqueeze(1) * torch.tan(angle)
        log_jac = (torch.log(half_F.clamp(min=1e-12)).unsqueeze(1)
                   + math.log(math.pi)
                   + torch.log(sigma_K.clamp(min=1e-12)).unsqueeze(1)
                   - torch.log(cos_sq))
        log_w_k = torch.log(gl_w.clamp(min=1e-38))
        nodes_r = y_kn.reshape(K * N, 1)
        ctx_r   = unique_ctx.unsqueeze(1).expand(K, N, -1).reshape(K * N, -1)
        lp_mdn  = self._mdn_log_prob(nodes_r, ctx_r)
        g_vals  = self._g(nodes_r, ctx_r)
        integrand = (lp_mdn + g_vals).reshape(K, N) + log_jac + log_w_k.unsqueeze(0)
        return torch.logsumexp(integrand, dim=1)

    # ── public interface ─────────────────────────────────────────────────────

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(1)
        lp_mdn = self._mdn_log_prob(y, context)
        g_y    = self._g(y, context)
        # Deduplicate: run GL quadrature only for unique contexts (~18 per batch)
        unique_ctx, inv = torch.unique(context, dim=0, return_inverse=True)
        log_zg          = self._log_Z_g_unique(unique_ctx)[inv]             # (B,)
        return lp_mdn + g_y - log_zg

    def forward(self, y, context):
        return self.log_prob(y, context)

    def mdn_parameters(self):
        return self._mdn.parameters()

    def correction_parameters(self):
        return list(self._ctx_enc_g.parameters()) + \
               list(self._g_in.parameters()) + \
               list(self._g_res.parameters()) + \
               list(self._g_out.parameters())

    def pdf_on_grid(self, context_enc: list, y_grid: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            ctx = torch.tensor([context_enc], dtype=torch.float32,
                               device=self._gl_nodes.device)
            y_t = torch.tensor(y_grid, dtype=torch.float32,
                               device=ctx.device).unsqueeze(1)
            lp  = self.log_prob(y_t, ctx.expand(len(y_grid), -1))
        return lp.exp().cpu().numpy()


def mdn_corr_loss(model: MDNCorrModel, y, context, weights):
    """Weighted NLL for MDNCorrModel with L2 regularization on g to suppress spurious structure."""
    if y.dim() == 1:
        y = y.unsqueeze(1)
    log_p  = model.log_prob(y, context)
    w_norm = weights / (weights.mean() + 1e-12)
    nll    = -(log_p * w_norm).mean()
    # L2 penalty on g: keeps the correction small and prevents overfitting bumps
    g_vals = model._g(y, context)
    reg    = 5e-2 * g_vals.pow(2).mean()
    return nll + reg
