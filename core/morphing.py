"""core/morphing.py — Quadratic EFT morphing baseline.

Physics motivation:
    σ(mHH | θ) = Σᵢⱼ cᵢ(θ) cⱼ(θ) · σᵢⱼ(mHH)

where cᵢ are coupling/parameter basis coefficients and σᵢⱼ are basis templates
solved by linear algebra from a set of grid points.

Implementation:
    1. Collect bracketing neighbour grid points (get_files_for_target), same as the
       ML training path.  Using ALL grid points was tried and discarded: distant
       grid points have their BW peaks inside the "continuum" region of the design
       matrix for other sources, causing the polynomial to go wildly negative.

    2. Apply ms1_scale (BW peak alignment) to ALL bins of each source histogram.
       This is the original approach.  The alternative (peak-split: raw histograms
       for continuum bins) was also tried and discarded for the same reason as above:
       the raw BW peaks of neighbouring sources fall in the "continuum" bins of the
       design matrix and destroy the polynomial fit.

    3. Weight each grid point by Voronoi cell area × Gaussian kernel in encoded
       parameter space.  This replaces the original 1/(d²+ε) weights, which had
       a singularity and gave poor conditioning.  The Gaussian bandwidth is set to
       1.5× the median nearest-neighbour distance in encoded space.

    4. Tikhonov (ridge) regularisation at λ = 1e-4 × median(σᵢ)² for numerical
       stability in low-statistics bins.

Public API (unchanged):
    eft_morphing(target, mhh_min, mhh_max, bin_width, verbose=False)
        → (pdf, centers, edges)  normalised PDF as numpy arrays
"""

import math
import numpy as np
import torch

import core.config as cfg
from core.config import (
    SCALE, TOL,
    encode_ctx, inverse_transform_mhh, PARAM_NAMES,
)
from core.dataset import preprocess_file


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial feature map  (1 + x1 + x2 + ... + x1² + x1·x2 + x2² + ...)
# ─────────────────────────────────────────────────────────────────────────────

def _poly_features(x, degree=2):
    """Build polynomial feature vector for a single encoded parameter row x (1-D array).

    Returns 1-D float64 array: [1, x0, x1, ..., x0², x0·x1, x1², ...]
    """
    n = len(x)
    feats = [1.0]
    feats.extend(x.tolist())
    if degree >= 2:
        for i in range(n):
            for j in range(i, n):
                feats.append(x[i] * x[j])
    return np.array(feats, dtype=np.float64)


def _poly_matrix(enc_list, degree=2):
    """Stack poly_features for a list of encoded-parameter rows. Shape (M, F)."""
    rows = [_poly_features(np.asarray(e, dtype=np.float64), degree) for e in enc_list]
    return np.vstack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(coords, edges, ms1_scale=1.0):
    """Load raw mHH histogram for one grid point (normalised).

    ms1_scale = (mS1_target - T) / (mS1_grid - T): rescales (mHH-T) around the
    kinematic threshold T so that the BW peak at mS1_grid maps exactly to
    mS1_target while keeping the threshold fixed.  The Jacobian is absorbed by
    renormalisation.
    """
    mhh_min = float(edges[0])
    mhh_max = float(edges[-1])
    all_mhh = []
    for f in cfg.GRID_FILES[coords]:
        cache  = preprocess_file(f, coords)
        tensor = torch.load(cache, weights_only=True)
        if tensor.shape[0] > 0:
            all_mhh.extend(
                inverse_transform_mhh(
                    tensor[:, -1].numpy(), cfg.MHH_THRESHOLD
                ).tolist()
            )
    arr = np.asarray(all_mhh, dtype=np.float64)
    if ms1_scale != 1.0 and cfg.MHH_THRESHOLD is not None:
        T   = float(cfg.MHH_THRESHOLD)
        arr = (arr - T) * ms1_scale + T
    arr = arr[(arr >= mhh_min) & (arr <= mhh_max)]
    counts, _ = np.histogram(arr, bins=edges)
    norm = np.sum(counts) * (edges[1] - edges[0])
    return counts.astype(np.float64) / norm if norm > 0 else counts.astype(np.float64)


def _distance_sq(enc_target, enc_point):
    """Squared Euclidean distance in encoded parameter space."""
    a = np.asarray(enc_target, dtype=np.float64)
    b = np.asarray(enc_point,  dtype=np.float64)
    return float(np.sum((a - b) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# eft_morphing
# ─────────────────────────────────────────────────────────────────────────────

def eft_morphing(target, mhh_min=0.0, mhh_max=5000.0, bin_width=50.0, verbose=False):
    """Quadratic EFT morphing baseline.

    Uses bracketing neighbour grid points with BW peak alignment (ms1_scale applied
    to all bins), weighted by Voronoi cell area × Gaussian kernel, with Tikhonov
    regularisation.

    Parameters
    ----------
    target : sequence of float
        Target BSM parameter values.
    mhh_min, mhh_max, bin_width : float
        Histogram range.
    verbose : bool

    Returns
    -------
    pdf     : np.ndarray  (n_bins,)  normalised PDF at target
    centers : np.ndarray  (n_bins,)  bin centres [GeV]
    edges   : np.ndarray  (n_bins+1,) bin edges  [GeV]
    """
    edges   = np.arange(mhh_min, mhh_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)

    target_enc = encode_ctx(*[float(v) for v in target])

    # ── Use ALL grid points for well-determined polynomial ────────────────
    coords_list = list(cfg.GRID_FILES.keys())
    n_pts = len(coords_list)

    if n_pts < 2:
        raise RuntimeError('EFT morphing requires at least 2 grid points.')

    if verbose:
        print(f'[EFT] {n_pts} neighbour grid points  |  {n_bins} bins  |  '
              f'range [{mhh_min}, {mhh_max}] bin={bin_width} GeV')

    # Encoded coordinates for each grid point
    enc_list       = [encode_ctx(*c) for c in coords_list]
    enc_arr        = np.array([np.asarray(e, dtype=np.float64) for e in enc_list])
    target_enc_arr = np.asarray(target_enc, dtype=np.float64)
    sq_dists       = np.sum((enc_arr - target_enc_arr) ** 2, axis=1)
    n_params       = len(target_enc)

    # Polynomial degree: quadratic if enough points, else fall back
    n_feats_quad = 1 + n_params + n_params * (n_params + 1) // 2
    n_feats_lin  = 1 + n_params
    if n_pts >= n_feats_quad:
        degree  = 2
        n_feats = n_feats_quad
    elif n_pts >= n_feats_lin:
        degree  = 1
        n_feats = n_feats_lin
    else:
        degree  = 0   # weighted average
        n_feats = 1

    if verbose:
        print(f'[EFT] degree={degree}  features={n_feats}  points={n_pts}')

    # ── Check if target is exactly on grid ────────────────────────────────
    on_grid_idx = np.where(sq_dists < TOL ** 2)[0]
    if len(on_grid_idx) > 0:
        idx = int(on_grid_idx[0])
        if verbose:
            print(f'[EFT] Target on grid at index {idx} — exact reproduction.')
        return _load_pdf(coords_list[idx], edges, ms1_scale=1.0), centers, edges

    # ── mS1 axis for BW peak alignment ───────────────────────────────────
    _ms1_idx = next(
        (i for i, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), -1
    )
    _target_ms1 = float(target[_ms1_idx]) if _ms1_idx >= 0 else None

    def _ms1_scale(c):
        if _ms1_idx < 0 or _target_ms1 is None or cfg.MHH_THRESHOLD is None:
            return 1.0
        T        = float(cfg.MHH_THRESHOLD)
        grid_ms1 = float(c[_ms1_idx])
        denom    = grid_ms1 - T
        num      = _target_ms1 - T
        return num / denom if abs(denom) > 1.0 else 1.0

    # ── Load all grid PDFs with peak alignment ────────────────────────────
    if verbose:
        print('[EFT] Loading grid histograms...')
    grid_pdfs = np.stack(
        [_load_pdf(c, edges, ms1_scale=_ms1_scale(c)) for c in coords_list], axis=0
    )   # (n_pts, n_bins)

    # Raw (unscaled) neighbour PDFs — used only for envelope clipping in the continuum
    grid_pdfs_raw = np.stack(
        [_load_pdf(c, edges, ms1_scale=1.0) for c in coords_list], axis=0
    )

    # ── Weights: Voronoi × Gaussian kernel ────────────────────────────────
    try:
        from core.dataset import compute_voronoi_weights
        voronoi_w = compute_voronoi_weights(coords_list, verbose=False)
    except Exception:
        voronoi_w = {c: 1.0 for c in coords_list}

    # Gaussian bandwidth = 1.5 × median nearest-neighbour distance in encoded space
    nn_dists = []
    for i in range(n_pts):
        d2 = np.sum((enc_arr - enc_arr[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn_dists.append(math.sqrt(float(np.min(d2))))
    bandwidth = max(float(np.median(nn_dists)) * 1.5, 1e-6)

    gauss_w    = np.exp(-0.5 * sq_dists / bandwidth ** 2)
    combined_w = np.array([
        float(voronoi_w.get(c, 1.0)) * gauss_w[i]
        for i, c in enumerate(coords_list)
    ])
    combined_w = np.maximum(combined_w, 1e-30)

    if verbose:
        print(f'[EFT] bandwidth={bandwidth:.4f}')
        for i, c in enumerate(coords_list):
            print(f'       {c}  voronoi={voronoi_w.get(c,1.0):.4f}  '
                  f'gauss={gauss_w[i]:.4f}  combined={combined_w[i]:.4e}')

    # ── Design matrix and weighted least-squares ──────────────────────────
    if degree == 0:
        A = np.ones((n_pts, 1), dtype=np.float64)
    else:
        A = _poly_matrix(enc_list, degree=degree)   # (n_pts, n_feats)

    W  = np.diag(np.sqrt(combined_w))
    WA = W @ A                # (n_pts, n_feats)
    Wy = W @ grid_pdfs        # (n_pts, n_bins)

    # Tikhonov regularisation: λ = 1e-6 × median(σᵢ)²  (reduced to avoid over-flattening)
    LAMBDA_FACTOR = 1e-6
    _, sv, _ = np.linalg.svd(WA, full_matrices=False)
    lambda_tik = LAMBDA_FACTOR * float(np.median(sv) ** 2)

    A_reg = np.vstack([WA, math.sqrt(lambda_tik) * np.eye(n_feats)])
    b_reg = np.vstack([Wy, np.zeros((n_feats, n_bins))])

    coeffs, _, rank, sv2 = np.linalg.lstsq(A_reg, b_reg, rcond=None)   # (n_feats, n_bins)

    if verbose:
        print(f'[EFT] lstsq rank={rank}/{n_feats}  λ={lambda_tik:.2e}  '
              f'cond={sv2[0]/max(sv2[-1], 1e-30):.1e}')

    # ── Evaluate polynomial at target ─────────────────────────────────────
    target_feat = (_poly_features(target_enc_arr, degree)
                   if degree > 0 else np.array([1.0]))
    pdf = target_feat @ coeffs   # (n_bins,)

    # ── Sanitise: clip negatives, renormalise ─────────────────────────────
    pdf = np.maximum(pdf, 0.0)

    # Clip to raw-neighbour envelope in the continuum (below peak) region.
    T_clip = float(cfg.MHH_THRESHOLD) if cfg.MHH_THRESHOLD is not None else 0.0
    peak_clip = float(_target_ms1) if _target_ms1 is not None else (T_clip + 500.0)
    clip_mask = centers < (T_clip + 0.90 * (peak_clip - T_clip))
    if clip_mask.any() and grid_pdfs_raw.shape[0] > 0:
        env_max = grid_pdfs_raw[:, clip_mask].max(axis=0)
        pdf[clip_mask] = np.minimum(pdf[clip_mask], 2.0 * env_max)

    norm = np.sum(pdf) * bin_width
    if norm > 0:
        pdf /= norm
    else:
        nearest = int(np.argmin(sq_dists))
        if verbose:
            print('[EFT] WARNING: polynomial gave all-zero PDF; using nearest-neighbour fallback.')
        pdf = grid_pdfs[nearest]

    if verbose:
        peak_bin = int(np.argmax(pdf))
        print(f'[EFT] Done.  Peak at {centers[peak_bin]:.0f} GeV  '
              f'(PDF={pdf[peak_bin]:.4e})')

    return pdf, centers, edges


# ─────────────────────────────────────────────────────────────────────────────
# bdt_morphing — gradient-boosted tree morphing baseline
# ─────────────────────────────────────────────────────────────────────────────

def bdt_morphing(target, mhh_min=0.0, mhh_max=5000.0, bin_width=50.0, verbose=False):
    """BDT morphing: joint GradientBoostingRegressor on (context, y) → log_density.

    For each grid point, computes the histogram PDF.  Stacks all (context, bin_centre_y,
    log_density) triples into a training set and fits a single GBT.  Predicts at the
    target context for all bins, then renormalises.

    Parameters
    ----------
    target       : sequence of float  – target BSM parameter values
    mhh_min, mhh_max, bin_width : float – histogram range
    verbose      : bool

    Returns
    -------
    pdf     : np.ndarray  (n_bins,)
    centers : np.ndarray  (n_bins,)
    edges   : np.ndarray  (n_bins+1,)
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        raise ImportError('scikit-learn is required for BDT morphing: pip install scikit-learn')

    edges   = np.arange(mhh_min, mhh_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)

    target_enc = np.array(encode_ctx(*[float(v) for v in target]), dtype=np.float64)

    # All grid points + encoded distances
    coords_list = list(cfg.GRID_FILES.keys())
    n_pts       = len(coords_list)
    enc_list    = [encode_ctx(*c) for c in coords_list]
    enc_arr     = np.array(enc_list, dtype=np.float64)
    sq_dists    = np.sum((enc_arr - target_enc.reshape(1, -1)) ** 2, axis=1)

    # On-grid passthrough
    on_grid = np.where(sq_dists < TOL ** 2)[0]

    # Use only k nearest neighbors in encoded space.
    # Training on all 18 grid points causes a spurious post-peak floor:
    # a distant grid point (e.g. mS1=1200) at peak-centred y_rel=+0.2
    # corresponds to mHH≈1300 GeV — pre-peak for mS1=1200 (high density).
    # For the target mS1=850 the same y_rel=+0.2 is post-peak (near-zero).
    # The GBT cannot resolve this conflict when 14+ distant points all
    # push toward high density at that y_rel.  Local training (nearest
    # neighbors only) ensures all samples share the same post-peak physics.
    K_NEIGHBORS  = min(6, n_pts)
    neighbor_idx = np.argsort(sq_dists)[:K_NEIGHBORS]
    coords_nn    = [coords_list[i] for i in neighbor_idx]
    enc_nn       = [enc_list[i]    for i in neighbor_idx]
    sq_dists_nn  = sq_dists[neighbor_idx]

    # Raw PDFs (no ms1_scale distortion)
    grid_pdfs = np.stack(
        [_load_pdf(c, edges, ms1_scale=1.0) for c in coords_nn], axis=0
    )  # (K, n_bins)

    if len(on_grid):
        on_nn = np.where(sq_dists_nn < TOL ** 2)[0]
        if len(on_nn):
            if verbose:
                print('[BDT] Target on grid — exact reproduction.')
            return grid_pdfs[on_nn[0]], centers, edges

    # Peak-centred coordinates: y_rel = y - peak_y(context)
    T = float(cfg.MHH_THRESHOLD) if cfg.MHH_THRESHOLD is not None else 0.0
    _ms1_idx_bdt = next(
        (i for i, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), -1
    )

    def _peak_y_bdt(c):
        if _ms1_idx_bdt >= 0:
            return (float(c[_ms1_idx_bdt]) - T) / SCALE
        return 0.0

    target_peak_y = (float(target[_ms1_idx_bdt]) - T) / SCALE if _ms1_idx_bdt >= 0 else 0.0
    centers_y     = (centers - T) / SCALE

    X_rows, y_rows = [], []
    for i, enc in enumerate(enc_nn):
        peak_yi = _peak_y_bdt(coords_nn[i])
        for b in range(n_bins):
            dens = float(grid_pdfs[i, b])
            if dens > 0:
                y_rel = float(centers_y[b]) - peak_yi
                X_rows.append(list(enc) + [y_rel])
                y_rows.append(math.log(dens))

    X_train = np.array(X_rows, dtype=np.float64)
    y_train = np.array(y_rows, dtype=np.float64)

    if verbose:
        print(f'[BDT] Training GBT on {len(X_train)} samples '
              f'({len(coords_nn)} grid pts × {n_bins} bins) ...')

    gbt = GradientBoostingRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=2, random_state=42,
    )
    gbt.fit(X_train, y_train)

    # Predict at target context for all bins
    y_rel_pred = centers_y - target_peak_y
    X_pred = np.column_stack([
        np.tile(target_enc, (n_bins, 1)),
        y_rel_pred.reshape(-1, 1),
    ])
    predicted_log = gbt.predict(X_pred)

    pdf = np.exp(predicted_log)
    pdf = np.maximum(pdf, 0.0)

    # The GBT cannot extrapolate below its minimum training value: for narrow
    # features every bin beyond a few widths from the peak is empty in all
    # training neighbors, so the GBT has no data there and returns a flat floor
    # at its minimum leaf value.
    #
    # Fix: find the anchor bin where the GBT slope has genuinely flattened
    # (deviation from the ms1-scaled neighbor envelope signals the floor), then
    # replace everything beyond it with a distance-weighted average of the
    # ms1-scaled neighbor PDFs.  This is data-driven and works for any feature
    # type (BW, threshold, interference) because the scaling is read from the
    # actual neighbor data — no physics is hardcoded here.

    # Load ms1-scaled neighbor PDFs (peak-aligned to target position)
    _ms1_idx_scale = next(
        (k for k, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), -1
    )
    _target_ms1_scale = float(target[_ms1_idx_scale]) if _ms1_idx_scale >= 0 else None

    def _ms1_scale_nn(c):
        if _ms1_idx_scale < 0 or _target_ms1_scale is None or cfg.MHH_THRESHOLD is None:
            return 1.0
        T_s   = float(cfg.MHH_THRESHOLD)
        denom = float(c[_ms1_idx_scale]) - T_s
        num   = _target_ms1_scale - T_s
        return num / denom if abs(denom) > 1.0 else 1.0

    grid_pdfs_scaled = np.stack(
        [_load_pdf(c, edges, ms1_scale=_ms1_scale_nn(c)) for c in coords_nn], axis=0
    )  # (K, n_bins)

    # Distance-weighted average of scaled neighbors
    weights_nn  = 1.0 / (sq_dists_nn + 1e-12)
    weights_nn /= weights_nn.sum()
    neighbor_avg = np.average(grid_pdfs_scaled, axis=0, weights=weights_nn)  # (n_bins,)

    # Locate anchor: the last post-peak bin where the GBT is still tracking the
    # neighbor envelope (within a factor 3), before the floor sets in.
    peak_bin   = int(np.argmax(pdf))
    anchor_bin = peak_bin
    for b in range(peak_bin + 1, n_bins):
        if neighbor_avg[b] > 0 and pdf[b] > 0:
            ratio = pdf[b] / neighbor_avg[b]
            # GBT floor: pdf stops falling while neighbor_avg keeps falling
            if ratio > 3.0:
                break
            anchor_bin = b
        elif pdf[b] == 0:
            break

    if anchor_bin > peak_bin and anchor_bin < n_bins - 1:
        anchor_val  = float(pdf[anchor_bin])
        anchor_ref  = float(neighbor_avg[anchor_bin])
        if anchor_ref > 0:
            scale = anchor_val / anchor_ref
            for b in range(anchor_bin + 1, n_bins):
                pdf[b] = scale * float(neighbor_avg[b])

    norm = np.sum(pdf) * bin_width
    if norm > 0:
        pdf /= norm
    else:
        nearest = int(np.argmin(sq_dists_nn))
        if verbose:
            print('[BDT] WARNING: all-zero prediction — using nearest-neighbour fallback.')
        pdf = grid_pdfs[nearest]

    if verbose:
        peak_bin = int(np.argmax(pdf))
        print(f'[BDT] Done.  Peak at {centers[peak_bin]:.0f} GeV  '
              f'(PDF={pdf[peak_bin]:.4e})')

    return pdf, centers, edges
