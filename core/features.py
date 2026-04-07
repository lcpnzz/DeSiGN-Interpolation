"""core/features.py — Feature detection, track building, prior generation, comment parser.

Pipeline:
    extract_features_at_coords()   — per grid-point histogram analysis
    build_feature_tracks()         — cross-grid feature tracking
    reclassify_paired_tracks()     — interference-pair detection across grid
    auto_nsf_hyperparams()         — auto-tune NSF tail_bound / num_bins
    build_physics_prior_from_analyses() — physics-informed prior samples
    build_track_prior()            — track-extrapolated prior samples
    parse_comment()                — free-text physics comment → prior dict
    generate_prior_samples()       — materialise a parsed prior dict
"""

import math
import re
import numpy as np
import torch

import core.config as cfg
from core.config import (
    SCALE, TOL, MHH_THRESHOLD,
    encode_ctx, transform_mhh, inverse_transform_mhh,
    PARAM_NAMES, PARAM_PAIR, P2_SPACING,
)
from core.dataset import preprocess_file

PDG_H_MASS = 125.09   # SM Higgs mass [GeV] — used for threshold-edge detection


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_smooth(arr, sigma_bins):
    k = max(int(3 * sigma_bins), 1)
    x = np.arange(-k, k + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode='same')[:len(arr)]


def classify_feature_type(feat, p1, p2, all_features_at_coord, topology=None):
    """Classify a single detected feature into bw / interference_pos / interference_neg / threshold.

    Priority order:
      1. sign < 0                              → interference_neg
      2. sign > 0, nearby dip at same coord    → interference_pos
      3. topology says squarks_only            → threshold  (all positive features)
      4. pos ≈ 2·p_i for any squark parameter  → threshold  (ratio-based, any mass scale)
      5. pos < 2·mH + 100 GeV                 → threshold  (SM threshold region)
      6. strongly asymmetric left-skew         → threshold
      7. default                               → bw
    """
    pos   = feat['mhh_position']
    sgn   = feat['sign']
    w_l   = feat.get('left_width_gev',  feat['width_gev'] / 2.0)
    w_r   = feat.get('right_width_gev', feat['width_gev'] / 2.0)
    asym  = feat.get('asymmetry', 0.0)
    total_w = max(w_l + w_r, 1.0)

    if sgn < 0:
        return 'interference_neg'

    nearby_dips = [f for f in all_features_at_coord
                   if f['sign'] < 0 and abs(f['mhh_position'] - pos) < 2.0 * total_w]
    if nearby_dips:
        return 'interference_pos'

    # Topology override: if only squarks are present, all positive features are thresholds.
    if topology is not None and topology.get('squarks_only'):
        return 'threshold'

    # Ratio-based threshold detection at ARBITRARY mHH scale.
    # A squark threshold sits at mHH ≈ 2·m_squark.  Check both parameters.
    # ±20% tolerance accounts for width effects shifting the apparent peak.
    for p_i in [p1, p2]:
        if p_i > 0 and abs(pos / p_i - 2.0) < 0.30:
            if pos > 2.0 * PDG_H_MASS:
                return 'threshold'

    _thr_edge = 2.0 * PDG_H_MASS + 100.0
    if pos < _thr_edge:
        return 'threshold'
    if asym > 0.55 and w_l < 30.0:
        return 'threshold'

    return 'bw'


# ─────────────────────────────────────────────────────────────────────────────
# Per-grid-point feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_at_coords(coords, files, mhh_min=None, mhh_max=None,
                                baseline_smooth_gev=200.0, feature_smooth_gev=None,
                                topology=None):
    """Detect spectral features (peaks, dips, threshold edges) at one grid point.

    Adaptive feature_smooth_gev
    ---------------------------
    For BW databases (S_, SS_, Ss_), peak width at this grid point = WoMS * mS.
    With WoMS=0.001, mS=1000 GeV, the peak is 1 GeV wide; the old hardcoded 15 GeV
    is 15x too wide and guarantees the peak is never detected.
    Fix: feature_smooth = max(WoMS * mS / 3, 1.0) per grid point.

    bin_width floor removed for BW databases
    -----------------------------------------
    The old bin_width = max(bw, feature_smooth/2) floor forced bins wider than the
    peak for narrow resonances.  For scalar databases the floor is removed so binning
    is statistics-driven only (floor at 0.5 GeV).  For squark-only databases the
    floor is kept (features are wide).

    Returns dict: coords, features, n_events, mhh_lo, mhh_hi,
                  y_min, y_max, y_median, y_q25, y_q75.
    """
    N_MIN_PER_BIN = 25

    all_y = []
    for f in files:
        cache = preprocess_file(f, coords)
        try:
            tensor = torch.load(cache, weights_only=True)
        except Exception:
            continue
        if tensor.shape[0] > 0:
            all_y.append(tensor[:, -1].numpy())

    T = cfg.MHH_THRESHOLD
    base = {
        'coords': coords, 'features': [], 'n_events': 0,
        'mhh_lo': T, 'mhh_hi': 1500.0,
        'y_min': 0.0, 'y_max': 1.0, 'y_median': 0.5, 'y_q25': 0.25, 'y_q75': 0.75,
    }
    if not all_y:
        return base

    y_arr   = np.concatenate(all_y)
    mhh_arr = inverse_transform_mhh(y_arr, T)

    mhh_lo = float(np.percentile(mhh_arr, 0.5))
    mhh_hi = float(np.percentile(mhh_arr, 99.5))
    if mhh_min is not None: mhh_lo = max(mhh_lo, float(mhh_min))
    if mhh_max is not None: mhh_hi = min(mhh_hi, float(mhh_max))
    mhh_lo  = max(mhh_lo, T + 1.0)

    mask    = (mhh_arr >= mhh_lo) & (mhh_arr <= mhh_hi)
    mhh_arr = mhh_arr[mask]; y_arr = y_arr[mask]
    N = len(mhh_arr)
    if N < 200:
        return base

    # --- Adaptive feature_smooth_gev ---
    _scalar_db = (topology is not None and topology.get('has_scalars'))
    if feature_smooth_gev is None:
        if _scalar_db:
            woms_idx = next((i for i, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('WoMS')), -1)
            ms_idx   = next((i for i, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), 0)
            if woms_idx >= 0 and len(coords) > woms_idx:
                woms = float(coords[woms_idx])
                ms   = float(coords[ms_idx])
                if ms > 2.0 * PDG_H_MASS:
                    feature_smooth_gev = max(woms * ms / 3.0, 1.0)
                else:
                    feature_smooth_gev = 15.0
            else:
                feature_smooth_gev = 15.0
        else:
            feature_smooth_gev = 15.0

    # --- Bin width ---
    mhh_range = mhh_hi - mhh_lo
    bin_width  = max(0.5, mhh_range * N_MIN_PER_BIN / N)
    if not _scalar_db:
        # Threshold databases: floor ensures ≥ 2 bins per feature
        bin_width = max(bin_width, feature_smooth_gev / 2.0)
    # else: scalar databases — statistics-driven only (no feature_smooth floor)

    edges   = np.arange(mhh_lo, mhh_hi + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(mhh_arr, bins=edges)
    n_bins = len(centers)

    bl_sigma = baseline_smooth_gev / bin_width
    ft_sigma = max(feature_smooth_gev / bin_width, 1.0)
    sqrtN    = np.sqrt(np.maximum(counts.astype(float), 0.25))
    baseline = _gauss_smooth(sqrtN, bl_sigma)
    residual = _gauss_smooth(sqrtN - baseline, ft_sigma)
    noise    = np.median(np.abs(residual - np.median(residual))) / 0.6745
    noise    = max(noise, 0.5 / max(math.sqrt(float(counts.max())) if counts.max() > 0 else 1.0, 1.0))

    p_per_bin   = 0.05 / (2.0 * n_bins)
    z0          = math.sqrt(-2.0 * math.log(p_per_bin))
    sig_bonf    = math.sqrt(-2.0 * math.log(p_per_bin * math.sqrt(2.0 * math.pi) * z0))
    significance = max(3.0, sig_bonf)
    threshold    = significance * noise

    log_counts = np.log(counts.astype(float) + 0.5)
    log_base   = _gauss_smooth(log_counts, bl_sigma)
    log_res    = _gauss_smooth(log_counts - log_base, ft_sigma)
    lq25, lq75 = np.percentile(log_res, [25, 75])
    log_noise  = max((lq75 - lq25) / 1.349, 1e-3)
    log_threshold = significance * log_noise

    min_dip_bins = max(2, int(feature_smooth_gev / bin_width))
    features = []

    # ── Peak detection (positive residual) ──────────────────────────────────
    i = 0
    while i < n_bins:
        if residual[i] > threshold:
            j = i
            while j < n_bins and residual[j] > threshold:
                j += 1
            region  = np.arange(i, j)
            peak_i  = int(region[np.argmax(residual[region])])
            sig    = float(residual[peak_i] / noise)
            peak_mhh = float(centers[peak_i])
            half_h   = residual[peak_i] / 2.0
            left_bins  = int(np.sum(residual[max(0,i):peak_i+1]   > half_h))
            right_bins = int(np.sum(residual[peak_i:min(n_bins,j)] > half_h))
            left_width_gev  = max(left_bins  * bin_width, bin_width / 2.0)
            right_width_gev = max(right_bins * bin_width, bin_width / 2.0)
            total_w = left_width_gev + right_width_gev
            asym    = (right_width_gev - left_width_gev) / max(total_w, 1.0)
            features.append({
                'mhh_position':   peak_mhh,
                'y_position':     float(transform_mhh(peak_mhh, T)),
                'width_gev':      total_w, 'width_y': total_w / SCALE,
                'left_width_gev': left_width_gev, 'right_width_gev': right_width_gev,
                'asymmetry':      float(asym), 'sign': +1, 'significance': sig,
            })
            i = j
        else:
            i += 1

    # ── Dip detection (negative log residual) ───────────────────────────────
    i = 0
    while i < n_bins:
        if log_res[i] < -log_threshold:
            j = i
            while j < n_bins and log_res[j] < -log_threshold:
                j += 1
            region = np.arange(i, j)
            if len(region) >= min_dip_bins:
                peak_i   = int(region[np.argmin(log_res[region])])
                sig      = float(-log_res[peak_i] / log_noise)
                peak_mhh = float(centers[peak_i])
                half_depth = log_res[peak_i] / 2.0
                left_bins  = int(np.sum(log_res[max(0,i):peak_i+1]   < half_depth))
                right_bins = int(np.sum(log_res[peak_i:min(n_bins,j)] < half_depth))
                left_width_gev  = max(left_bins  * bin_width, bin_width / 2.0)
                right_width_gev = max(right_bins * bin_width, bin_width / 2.0)
                total_w = left_width_gev + right_width_gev
                asym    = (right_width_gev - left_width_gev) / max(total_w, 1.0)
                features.append({
                    'mhh_position':   peak_mhh,
                    'y_position':     float(transform_mhh(peak_mhh, T)),
                    'width_gev':      total_w, 'width_y': total_w / SCALE,
                    'left_width_gev': left_width_gev, 'right_width_gev': right_width_gev,
                    'asymmetry':      float(asym), 'sign': -1, 'significance': sig,
                })
            i = j
        else:
            i += 1

    for feat in features:
        feat['feature_type'] = classify_feature_type(
            feat, float(coords[0]),
            float(coords[1]) if len(coords) > 1 else 0.0,
            features, topology=topology)

    features.sort(key=lambda f: -f['significance'])

    # For BW-only databases, keep only the dominant positive feature (the actual peak)
    # and all negative features.  Low-significance tail fluctuations corrupt the prior.
    if topology is not None and topology.get('scalars_only') and not topology.get('has_squarks'):
        pos_feats = [f for f in features if f['sign'] > 0]
        neg_feats = [f for f in features if f['sign'] < 0]
        features = (pos_feats[:1] if pos_feats else []) + neg_feats

    return {
        'coords':   coords, 'features': features, 'n_events': N,
        'mhh_lo':   mhh_lo, 'mhh_hi':  mhh_hi,
        'y_min':    float(y_arr.min()), 'y_max': float(y_arr.max()),
        'y_median': float(np.median(y_arr)),
        'y_q25':    float(np.percentile(y_arr, 25)),
        'y_q75':    float(np.percentile(y_arr, 75)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-grid feature tracking
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_tracks(point_analyses, verbose=False):
    """Group features across grid points into continuous tracks.

    A track is a set of features (one per grid point, same sign) whose mHH
    positions follow a smooth trend in parameter space — fitted by linear
    regression in (p1/p1_scale, p2_enc/lw_scale) coordinates.

    Algorithm: greedy highest-significance-first seed expansion.  A candidate
    feature joins a track if its predicted position (from the current track
    membership) is within the estimated residual tolerance.

    Tracks with only a single grid point in p1 are rejected (spurious spikes).

    Returns: list of track dicts, sorted by mean_significance descending.
    """
    all_f = []
    for analysis in point_analyses:
        coords = analysis['coords']
        p1     = float(coords[0])
        p2_enc = encode_ctx(*coords)[1] if len(coords) >= 2 else 0.0
        for feat in analysis['features']:
            all_f.append({
                'coords':        coords, 'p1': p1, 'p2_enc': p2_enc,
                'y_pos':         feat['y_position'], 'width_y': feat['width_y'],
                'sign':          feat['sign'],        'sig':     feat['significance'],
                'left_width_y':  feat.get('left_width_gev',  feat['width_gev'] / 2.0) / SCALE,
                'right_width_y': feat.get('right_width_gev', feat['width_gev'] / 2.0) / SCALE,
                'asymmetry':     feat.get('asymmetry', 0.0),
            })
    if not all_f:
        if verbose:
            print('Feature tracks: no features detected at any grid point.')
        return []

    p1_vals  = [f['p1']     for f in all_f]
    lw_vals  = [f['p2_enc'] for f in all_f]
    p1_scale = max(max(p1_vals) - min(p1_vals), 1.0)
    lw_scale = max(max(lw_vals) - min(lw_vals), 1.0)

    def _predict_y(track_fs, candidate_f):
        if len(track_fs) < 2:
            return track_fs[0]['y_pos'], 0.5
        X = np.array([[1.0, tf['p1'] / p1_scale, tf['p2_enc'] / lw_scale] for tf in track_fs])
        y = np.array([tf['y_pos'] for tf in track_fs])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return float(np.mean(y)), 0.5
        xc = np.array([1.0, candidate_f['p1'] / p1_scale, candidate_f['p2_enc'] / lw_scale])
        resid_std = float(np.std(y - X @ coeffs)) if len(y) > 2 else 0.3
        return float(coeffs @ xc), max(resid_std * 3.0, 0.15)

    used = [False] * len(all_f)
    tracks = []
    for seed_idx in sorted(range(len(all_f)), key=lambda i: -all_f[i]['sig']):
        if used[seed_idx]:
            continue
        track = [all_f[seed_idx]]; used[seed_idx] = True
        changed = True
        while changed:
            changed = False
            track_coords = {tf['coords'] for tf in track}
            for j, f in enumerate(all_f):
                if used[j] or f['coords'] in track_coords or f['sign'] != track[0]['sign']:
                    continue
                pred_y, tol = _predict_y(track, f)
                if abs(f['y_pos'] - pred_y) <= tol:
                    track.append(f); used[j] = True; changed = True
        if len(track) < 2:
            continue
        p1_range = max(f['p1'] for f in track) - min(f['p1'] for f in track)
        if p1_range < TOL:
            continue
        track.sort(key=lambda f: f['p1'])
        mean_sig = float(np.mean([f['sig'] for f in track]))
        if track[0]['sign'] < 0:
            boost    = min(math.sqrt(len(track) / 2.0), 4.0)
            mean_sig *= boost
        tracks.append({
            'features': track, 'sign': track[0]['sign'],
            'n_points': len(track), 'mean_significance': mean_sig,
        })
    tracks.sort(key=lambda t: -t['mean_significance'])

    if verbose:
        names = cfg.PARAM_NAMES or ['p1', 'p2']
        print(f'Feature tracks: {len(tracks)} found')
        for i, t in enumerate(tracks):
            print(f'  Track {i+1}: sign={t["sign"]:+d}  n_points={t["n_points"]}'
                  f'  sig={t["mean_significance"]:.1f}')
            for f in t['features']:
                print(f'    {names[0]}={f["p1"]:.0f}  '
                      f'mHH={inverse_transform_mhh(f["y_pos"], cfg.MHH_THRESHOLD):.0f} GeV'
                      f'  width={f["width_y"]*SCALE:.0f} GeV  sig={f["sig"]:.1f}')
    return tracks


def reclassify_paired_tracks(tracks, analyses):
    """Cross-grid interference-pair detection (modifies tracks and analyses in-place).

    A positive track is paired with a negative track if their mHH position
    ranges overlap (within the sum of their typical widths).  Paired positive
    track members are reclassified from 'bw' to 'interference_pos'.

    This is more robust than single-coord pairing because dips are harder to
    detect and may be missed at individual grid points even when the pattern
    is clear across the full grid.
    """
    pos_tracks = [t for t in tracks if t['sign'] > 0]
    neg_tracks = [t for t in tracks if t['sign'] < 0]
    if not pos_tracks or not neg_tracks:
        return

    T = cfg.MHH_THRESHOLD

    def _track_range(t):
        positions = [f['y_pos'] * SCALE + T for f in t['features']]
        widths    = [f['width_y'] * SCALE    for f in t['features']]
        med_w = float(np.median(widths))
        return min(positions) - med_w, max(positions) + med_w

    neg_ranges   = [_track_range(t) for t in neg_tracks]
    analysis_map = {a['coords']: a for a in analyses}

    for pt in pos_tracks:
        lo_p, hi_p = _track_range(pt)
        paired = any(lo_n <= hi_p and lo_p <= hi_n for lo_n, hi_n in neg_ranges)
        if not paired:
            continue
        pt['paired_interference'] = True
        for f in pt['features']:
            f['sign_reclassified'] = 'interference_pos'
            coords = f['coords']
            if coords in analysis_map:
                for af in analysis_map[coords]['features']:
                    if (af['sign'] > 0
                            and abs(af['mhh_position'] - (f['y_pos'] * SCALE + T)) < 1.0):
                        af['feature_type'] = 'interference_pos'


# ─────────────────────────────────────────────────────────────────────────────
# Cross-track position prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_track_position(track, coords):
    """Predict the mHH peak position [GeV] of a feature track at given coords.

    Uses the same (p1, p2_enc) linear fit as build_feature_tracks.
    Falls back to mean position if fitting fails.
    Returns float (GeV) or None if the track has no features.
    """
    features = track['features']
    if not features:
        return None

    T = cfg.MHH_THRESHOLD
    coords_f = tuple(float(c) for c in coords)

    # Exact grid hit
    for f in features:
        if all(abs(f['coords'][i] - coords_f[i]) < cfg.TOL for i in range(len(coords_f))):
            return f['y_pos'] * SCALE + T

    if len(features) == 1:
        return features[0]['y_pos'] * SCALE + T

    p1_vals  = [f['p1']     for f in features]
    p2_vals  = [f['p2_enc'] for f in features]
    p1_scale = max(max(p1_vals) - min(p1_vals), 1.0)
    p2_scale = max(max(p2_vals) - min(p2_vals), 1.0)

    X = np.array([[1.0, f['p1'] / p1_scale, f['p2_enc'] / p2_scale] for f in features])
    y = np.array([f['y_pos'] for f in features])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return float(np.mean(y)) * SCALE + T

    enc      = encode_ctx(*coords_f)
    p2_enc_t = enc[1] if len(enc) > 1 else 0.0
    xc       = np.array([1.0, coords_f[0] / p1_scale, p2_enc_t / p2_scale])
    y_pred   = float(coeffs @ xc)
    return y_pred * SCALE + T


# ─────────────────────────────────────────────────────────────────────────────
# NSF hyperparameter auto-tuning
# ─────────────────────────────────────────────────────────────────────────────

def auto_nsf_hyperparams(source_data, mhh_min, mhh_max, verbose=False):
    """Scan all training grid points and auto-tune NSF tail_bound and num_bins.

    tail_bound : max observed y across grid × 1.2 (20% margin), floored at 1.5.
    num_bins   : resolves the narrowest detected feature with >= 4 spline bins.
                 Capped at 128 to avoid overfitting.

    Returns (num_bins, tail_bound, analyses, tracks).
    """
    analyses = [extract_features_at_coords(c, f, topology=cfg.TOPOLOGY)
                for c, f in source_data]

    all_y_max  = [a['y_max'] for a in analyses if a['n_events'] > 0]
    tail_bound = max(all_y_max) * 1.2 if all_y_max else 6.0
    tail_bound = max(tail_bound, 1.5)

    tracks = build_feature_tracks(analyses, verbose=False)
    reclassify_paired_tracks(tracks, analyses)
    if tracks:
        min_width_y = min(
            f['width_y'] for t in tracks for f in t['features'] if f['width_y'] > 0
        )
        num_bins = max(int(math.ceil(4.0 / min_width_y)), 32)
        num_bins = min(num_bins, 128)
    else:
        num_bins = 64

    if verbose:
        print(f'Auto NSF hyperparams: tail_bound={tail_bound:.3f}  num_bins={num_bins}')
        if tracks:
            print(f'  (narrowest feature: {min_width_y*SCALE:.0f} GeV)')

    return num_bins, tail_bound, analyses, tracks


# ─────────────────────────────────────────────────────────────────────────────
# Ratio-space stacked feature detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_stacked_ratio_features(source_data, param_idx,
                                   u_min=0.3, u_max=5.0,
                                   baseline_smooth_u=0.4, feature_smooth_u=0.04,
                                   verbose=False):
    """Detect features in stacked u = mHH / p_i space, gaining ~sqrt(N_grid) sensitivity.

    For squark thresholds and BW peaks the feature position scales as mHH ≈ N·p_i
    (threshold at u≈2, BW peak at u≈1).  Stacking normalised u-histograms across all
    grid points concentrates the feature at fixed u while noise averages as sqrt(N_pts).

    Includes kink detection (slope-change) to catch shallow threshold edges that do not
    produce a clear peak in the sqrt-residual.

    Returns list of dicts: {ratio, ratio_u_width, significance, sign, param, asymmetry}
    """
    N_MIN_PER_BIN = 10

    bin_width_u  = feature_smooth_u / 4.0
    edges_u      = np.arange(u_min, u_max + bin_width_u, bin_width_u)
    n_bins_u     = len(edges_u) - 1
    centers_u    = 0.5 * (edges_u[:-1] + edges_u[1:])

    stacked          = np.zeros(n_bins_u, dtype=np.float64)
    coverage_per_bin = np.zeros(n_bins_u, dtype=np.float64)
    n_contributing   = 0

    T = cfg.MHH_THRESHOLD

    for coords, files in source_data:
        p_i = float(coords[param_idx]) if param_idx < len(coords) else 0.0
        if p_i <= 0:
            continue

        all_y = []
        for f in files:
            cache = preprocess_file(f, coords)
            try:
                tensor = torch.load(cache, weights_only=True)
            except Exception:
                continue
            if tensor.shape[0] > 0:
                all_y.append(tensor[:, -1].numpy())

        if not all_y:
            continue

        y_arr   = np.concatenate(all_y)
        mhh_arr = inverse_transform_mhh(y_arr, T)
        mhh_arr = mhh_arr[mhh_arr > T + 10.0]
        if len(mhh_arr) < 200:
            continue

        u_arr     = mhh_arr / p_i
        counts_i, _ = np.histogram(u_arr, bins=edges_u)
        total = counts_i.sum()
        if total < 200:
            continue

        pdf_i = counts_i.astype(np.float64) / total
        stacked          += pdf_i
        coverage_per_bin += (counts_i > 0).astype(np.float64)
        n_contributing   += 1

    if n_contributing < 3:
        if verbose:
            plabel = (cfg.PARAM_NAMES[param_idx]
                      if cfg.PARAM_NAMES and param_idx < len(cfg.PARAM_NAMES)
                      else f'p{param_idx+1}')
            print(f'[RATIO STACK] {plabel}: only {n_contributing} grid points — skipping')
        return []

    coverage_per_bin = np.maximum(coverage_per_bin, 1.0)
    eff_counts = stacked * n_contributing / coverage_per_bin

    bl_sigma = baseline_smooth_u / bin_width_u
    ft_sigma = max(feature_smooth_u / bin_width_u, 1.0)

    sqrtN    = np.sqrt(np.maximum(eff_counts, 0.25))
    baseline = _gauss_smooth(sqrtN, bl_sigma)
    residual = _gauss_smooth(sqrtN - baseline, ft_sigma)
    noise    = np.median(np.abs(residual - np.median(residual))) / 0.6745
    noise    = max(noise, 1e-6)

    p_per_bin = 0.05 / (2.0 * n_bins_u)
    z0        = math.sqrt(-2.0 * math.log(p_per_bin))
    sig_bonf  = math.sqrt(-2.0 * math.log(p_per_bin * math.sqrt(2.0 * math.pi) * z0))
    threshold = max(3.0, sig_bonf) * noise

    log_eff = np.log(np.maximum(eff_counts, 0.25))
    log_base = _gauss_smooth(log_eff, bl_sigma)
    log_res  = _gauss_smooth(log_eff - log_base, ft_sigma)
    lq25, lq75 = np.percentile(log_res, [25, 75])
    log_noise  = max((lq75 - lq25) / 1.349, 1e-6)
    log_thresh = max(3.0, sig_bonf) * log_noise
    min_dip_bins = max(2, int(feature_smooth_u / bin_width_u))

    plabel   = cfg.PARAM_NAMES[param_idx] if (cfg.PARAM_NAMES and param_idx < len(cfg.PARAM_NAMES)) else f'p{param_idx+1}'
    features = []

    # Positive features
    i = 0
    while i < n_bins_u:
        if residual[i] > threshold:
            j = i
            while j < n_bins_u and residual[j] > threshold:
                j += 1
            region = np.arange(i, j)
            peak_i = int(region[np.argmax(residual[region])])
            sig    = float(residual[peak_i] / noise)
            u_peak = float(centers_u[peak_i])
            half   = residual[peak_i] / 2.0
            lw = float(np.sum(residual[max(0,i):peak_i+1] > half)) * bin_width_u
            rw = float(np.sum(residual[peak_i:min(n_bins_u,j)] > half)) * bin_width_u
            features.append({'ratio': u_peak, 'ratio_u_width': lw + rw,
                              'left_u_width': lw, 'right_u_width': rw,
                              'asymmetry': (rw - lw) / max(lw + rw, 1e-9),
                              'significance': sig, 'sign': +1,
                              'param': plabel})
            i = j
        else:
            i += 1

    # Negative features
    i = 0
    while i < n_bins_u:
        if log_res[i] < -log_thresh:
            j = i
            while j < n_bins_u and log_res[j] < -log_thresh:
                j += 1
            region = np.arange(i, j)
            if len(region) >= min_dip_bins:
                peak_i = int(region[np.argmin(log_res[region])])
                sig    = float(abs(log_res[peak_i]) / log_noise)
                u_peak = float(centers_u[peak_i])
                half   = log_res[peak_i] / 2.0
                lw = float(np.sum(log_res[max(0,i):peak_i+1] < half)) * bin_width_u
                rw = float(np.sum(log_res[peak_i:min(n_bins_u,j)] < half)) * bin_width_u
                features.append({'ratio': u_peak, 'ratio_u_width': lw + rw,
                                  'left_u_width': lw, 'right_u_width': rw,
                                  'asymmetry': (rw - lw) / max(lw + rw, 1e-9),
                                  'significance': sig, 'sign': -1,
                                  'param': plabel})
            i = j
        else:
            i += 1

    # Kink (slope-change) detection for shallow threshold edges
    _eff_floor  = 0.25
    _valid_bins = eff_counts > _eff_floor
    _hw         = max(int(baseline_smooth_u / bin_width_u * 0.5), 4)
    _log_sm     = _gauss_smooth(np.log(np.maximum(eff_counts, _eff_floor)), ft_sigma)
    _kink       = np.zeros(n_bins_u)
    _ix         = np.arange(_hw, dtype=float)
    _ixc        = _ix.mean()
    _ixv        = float(np.sum((_ix - _ixc) ** 2)) + 1e-30
    for _i in range(_hw, n_bins_u - _hw):
        if not (_valid_bins[_i - _hw:_i].all() and _valid_bins[_i:_i + _hw].all()):
            continue
        _sl = float(np.dot(_ix - _ixc, _log_sm[_i - _hw:_i])) / _ixv
        _sr = float(np.dot(_ix - _ixc, _log_sm[_i:_i + _hw])) / _ixv
        _kink[_i] = _sr - _sl
    _kink_s = _gauss_smooth(_kink, ft_sigma)
    _kink_valid = _kink_s[_valid_bins]
    if len(_kink_valid) > 10:
        _kink_noise = max(np.median(np.abs(_kink_valid - np.median(_kink_valid))) / 0.6745, 1e-6)
    else:
        _kink_noise = float('inf')
    _kink_thr = max(3.0, sig_bonf) * _kink_noise
    ki = 0
    while ki < n_bins_u:
        if abs(_kink_s[ki]) > _kink_thr and _valid_bins[ki]:
            kj = ki
            while kj < n_bins_u and abs(_kink_s[kj]) > _kink_thr and _valid_bins[kj]:
                kj += 1
            region = np.arange(ki, kj)
            pk     = int(region[np.argmax(np.abs(_kink_s[region]))])
            sk     = +1 if _kink_s[pk] > 0 else -1
            sig_k  = float(abs(_kink_s[pk]) / _kink_noise)
            u_pk   = float(centers_u[pk])
            lw_k = float(np.sum(np.abs(_kink_s[max(0,ki):pk+1]) > _kink_thr/2)) * bin_width_u
            rw_k = float(np.sum(np.abs(_kink_s[pk:min(n_bins_u,kj)]) > _kink_thr/2)) * bin_width_u
            features.append({'ratio': u_pk, 'ratio_u_width': lw_k + rw_k,
                              'left_u_width': lw_k, 'right_u_width': rw_k,
                              'asymmetry': (rw_k - lw_k) / max(lw_k + rw_k, 1e-9),
                              'significance': sig_k, 'sign': sk,
                              'feature_kind': 'kink', 'param': plabel})
            ki = kj
        else:
            ki += 1

    if verbose:
        print(f'[RATIO STACK] {plabel}: {n_contributing} grid points stacked'
              f' → {len(features)} ratio feature(s) detected')
        for f in sorted(features, key=lambda x: -x['significance']):
            kind = f.get('feature_kind', 'bump')
            print(f"  {'+' if f['sign']>0 else '-'}@u={f['ratio']:.3f}"
                  f"  sig={f['significance']:.1f}"
                  f"  u_width={f['ratio_u_width']:.3f}"
                  f"  asym={f['asymmetry']:+.2f}  [{kind}]")

    return features


def build_ratio_tracks(tracks, analyses, source_data=None,
                       ratio_tol=0.05, min_coverage=0.25, verbose=False):
    """Cluster pos/p_i ratios across grid points and annotate matching feature tracks.

    For every detected feature at (p1, p2, ...) with position pos_gev, compute
    r1 = pos/p1, r2 = pos/p2.  Ratios are clustered with ±ratio_tol relative
    tolerance.  A cluster is a ratio track only when it spans ≥ min_coverage of
    eligible grid points.

    Primary detection is via detect_stacked_ratio_features (√N_grid sensitivity).
    Per-point clustering provides corroborating coverage statistics.

    Annotates matching feature tracks with 'ratio_tag', 'ratio_multiplier',
    'ratio_param', and 'predict_pos' in-place.

    Returns list of ratio-track dicts with 'predict_pos': callable(p1, p2) → mHH GeV.
    """
    T = cfg.MHH_THRESHOLD

    entries = []
    for analysis in analyses:
        coords = analysis['coords']
        p1 = float(coords[0])
        p2 = float(coords[1]) if len(coords) > 1 else 0.0
        for feat in analysis['features']:
            if feat['sign'] < 0:
                continue
            pos_gev = feat['mhh_position']
            if pos_gev <= 0 or p1 <= 0:
                continue
            e = {'coords': coords, 'p1': p1, 'p2': p2,
                 'pos': pos_gev, 'sig': feat['significance'],
                 'r1': pos_gev / p1}
            if p2 > 0:
                e['r2'] = pos_gev / p2
            entries.append(e)

    if not entries:
        return []

    def _n_eligible(ratio, param_key):
        count = 0
        for a in analyses:
            pval = float(a['coords'][0]) if param_key == 'r1' else (float(a['coords'][1]) if len(a['coords']) > 1 else 0.0)
            if pval > 0 and ratio * pval > T + 50.0:
                count += 1
        return max(count, 1)

    def _cluster_ratios(ratio_key):
        has_key = [e for e in entries if ratio_key in e]
        if not has_key:
            return []
        vals = sorted(has_key, key=lambda e: e[ratio_key])
        clusters = []
        used = [False] * len(vals)
        for i, seed in enumerate(vals):
            if used[i]:
                continue
            r_seed  = seed[ratio_key]
            members = []
            for j, e in enumerate(vals):
                if not used[j] and abs(e[ratio_key] - r_seed) / r_seed < ratio_tol:
                    members.append(e)
                    used[j] = True
            if len(members) < 2:
                continue
            mean_r = float(np.mean([e[ratio_key] for e in members]))
            core   = [e for e in members if abs(e[ratio_key] - mean_r) / mean_r < ratio_tol]
            if not core:
                core = members
            mean_r  = float(np.mean([e[ratio_key] for e in core]))
            std_r   = float(np.std([e[ratio_key] for e in core]) / mean_r) if mean_r > 0 else float('inf')
            unique_pts = len({e['coords'] for e in core})
            n_elig     = _n_eligible(mean_r, ratio_key)
            coverage   = unique_pts / n_elig
            mean_sig   = float(np.mean([e['sig'] for e in core]))
            p_i_key    = 'p1' if ratio_key == 'r1' else 'p2'
            p_i_vals   = [e[p_i_key] for e in core]
            p_i_spread = max(p_i_vals) / min(p_i_vals) if min(p_i_vals) > 0 else 1.0
            p1v = np.array([e['p1'] for e in core], dtype=float)
            p2v = np.array([e['p2'] for e in core], dtype=float)
            p12_corr = float(abs(np.corrcoef(p1v, p2v)[0, 1])) if (len(core) >= 3 and p1v.std() > 0 and p2v.std() > 0) else 0.0
            clusters.append({
                'ratio': mean_r, 'ratio_std': std_r,
                'param': ratio_key[1],
                'members': core,
                'n_points': unique_pts, 'n_eligible': n_elig, 'coverage': coverage,
                'mean_sig': mean_sig, 'p_i_spread': p_i_spread, 'p12_corr': p12_corr,
            })
        return clusters

    # Primary: stacking in u-space
    stacked_feats = []
    if source_data is not None:
        n_params = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else (len(next(iter(analyses))['coords']) if analyses else 2)
        for pidx in range(min(n_params, 2)):
            sf = detect_stacked_ratio_features(source_data, pidx, verbose=verbose)
            stacked_feats.extend(sf)

    # Secondary: per-point clusters
    cluster_map = {}
    for rkey, plabel in [('r1', 'p1'), ('r2', 'p2')]:
        for cl in _cluster_ratios(rkey):
            key = (plabel, round(cl['ratio'], 2))
            cluster_map[key] = cl

    ratio_tracks = []
    for sf in stacked_feats:
        mean_r  = sf['ratio']
        param   = sf['param']
        best_cl = None
        for (cp, cr), cl in cluster_map.items():
            if cp == param and abs(cr - mean_r) / max(mean_r, 1e-9) < ratio_tol * 2:
                if best_cl is None or cl['n_points'] > best_cl['n_points']:
                    best_cl = cl
        if best_cl is None:
            continue
        if best_cl.get('p_i_spread', 1.0) < 1.3:
            continue
        if best_cl.get('p12_corr', 0.0) > 0.85:
            continue

        best_track_idx, best_overlap = None, 0
        cl_coords = {e['coords'] for e in best_cl['members']}
        for ti, t in enumerate(tracks):
            overlap = len(cl_coords & {f['coords'] for f in t['features']})
            if overlap > best_overlap:
                best_overlap, best_track_idx = overlap, ti

        def _predict(p1, p2, r=mean_r, p=param):
            pnames = cfg.PARAM_NAMES or ['p1', 'p2']
            p_val  = p1 if (pnames[0] == p or p == 'p1') else p2
            return r * p_val

        ratio_tracks.append({
            'ratio':         mean_r,
            'param':         param,
            'ratio_u_width': sf['ratio_u_width'],
            'asymmetry':     sf['asymmetry'],
            'significance':  sf['significance'],
            'sign':          sf['sign'],
            'ratio_std':     best_cl['ratio_std'],
            'n_points':      best_cl['n_points'],
            'n_eligible':    best_cl['n_eligible'],
            'coverage':      best_cl['coverage'],
            'mean_sig':      best_cl['mean_sig'],
            'track_idx':     best_track_idx,
            'predict_pos':   _predict,
            'members':       best_cl['members'],
        })

    # Also add per-point clusters not found by stacking
    for rkey, plabel in [('r1', 'p1'), ('r2', 'p2')]:
        for cl in _cluster_ratios(rkey):
            if cl['ratio_std'] > ratio_tol:
                continue
            if cl.get('p_i_spread', 1.0) < 1.3:
                continue
            if cl.get('p12_corr', 0.0) > 0.85:
                continue
            other_key = 'p2' if rkey == 'r1' else 'p1'
            other_vals = {e[other_key] for e in cl['members']}
            if len(other_vals) < 2:
                continue
            if cl['coverage'] < min_coverage and cl['n_points'] < 4:
                continue
            already = any(
                rt['param'] == plabel and
                abs(rt['ratio'] - cl['ratio']) / cl['ratio'] < ratio_tol
                for rt in ratio_tracks
            )
            if already:
                continue
            mean_r = cl['ratio']
            best_track_idx, best_overlap = None, 0
            for ti, t in enumerate(tracks):
                overlap = len({e['coords'] for e in cl['members']} &
                               {f['coords'] for f in t['features']})
                if overlap > best_overlap:
                    best_overlap, best_track_idx = overlap, ti
            def _predict2(p1, p2, r=mean_r, p=plabel):
                pnames = cfg.PARAM_NAMES or ['p1', 'p2']
                p_val  = p1 if (pnames[0] == p or p == 'p1') else p2
                return r * p_val
            ratio_tracks.append({
                'ratio': mean_r, 'param': plabel,
                'ratio_u_width': float('nan'), 'asymmetry': float('nan'),
                'significance': cl['mean_sig'], 'sign': +1,
                'ratio_std': cl['ratio_std'], 'n_points': cl['n_points'],
                'n_eligible': cl['n_eligible'], 'coverage': cl['coverage'],
                'mean_sig': cl['mean_sig'], 'track_idx': best_track_idx,
                'predict_pos': _predict2, 'members': cl['members'],
            })

    # Deduplicate keeping stacked version
    ratio_tracks.sort(key=lambda rt: (-rt['significance'], rt['ratio']))
    deduped = []
    for rt in ratio_tracks:
        duplicate = any(
            rt['param'] == ex['param'] and
            abs(rt['ratio'] - ex['ratio']) / max(ex['ratio'], 1e-6) < ratio_tol
            for ex in deduped
        )
        if not duplicate:
            deduped.append(rt)

    # Annotate matching feature tracks
    pnames = cfg.PARAM_NAMES or ['p1', 'p2']
    for rt in deduped:
        plabel = rt['param']
        p_display = pnames[0] if (plabel == 'p1' or plabel == pnames[0]) else (pnames[1] if len(pnames) > 1 else plabel)
        tag = f"{rt['ratio']:.3g}*{p_display}"
        rt['ratio_tag'] = tag
        if rt['track_idx'] is not None:
            tracks[rt['track_idx']]['ratio_tag']        = tag
            tracks[rt['track_idx']]['ratio_multiplier'] = rt['ratio']
            tracks[rt['track_idx']]['ratio_param']      = rt['param']
            tracks[rt['track_idx']]['predict_pos']      = rt['predict_pos']

    if verbose and deduped:
        print(f'Ratio tracks: {len(deduped)} detected')
        for rt in deduped:
            uw_str = f"{rt['ratio_u_width']:.3f}" if rt['ratio_u_width'] == rt['ratio_u_width'] else 'n/a'
            n_str  = f"N={rt['n_points']}/{rt['n_eligible']} (cov={rt['coverage']:.0%})"
            ti_str = f"→ track {rt['track_idx']}" if rt['track_idx'] is not None else '(no matching track)'
            print(f"  {rt['ratio_tag']:12s}  sig={rt['significance']:.1f}"
                  f"  u_width={uw_str}  {n_str}  {ti_str}")
    elif verbose:
        print('Ratio tracks: none detected')

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Physics prior generation
# ─────────────────────────────────────────────────────────────────────────────

def build_physics_prior_from_analyses(analyses, source_data, n_per_coord=5000,
                                      prior_weight=0.30, anneal_frac=0.10,
                                      credibility=None, verbose=False):
    """Build physics-informed prior samples from per-coord feature classifications.

    credibility: dict with 'neg_weight' and 'pos_weight' from _comment_credibility().
      neg_weight < 0.3 → interference_neg prior suppressed entirely.
      neg_weight > 1.0 → interference_neg prior oversampled.

    Returns (y_t, ctx_t, w_t, meta) tensors on CPU, or (None, None, None, meta).
    """
    T    = cfg.MHH_THRESHOLD
    rng  = np.random.default_rng(0)
    cred = credibility or {'neg_weight': 1.0, 'pos_weight': 1.0}

    analysis_map = {a['coords']: a for a in analyses}
    mhh_lo = min((a['mhh_lo'] for a in analyses if a['n_events'] > 0), default=float(T))
    mhh_hi = max((a['mhh_hi'] for a in analyses if a['n_events'] > 0), default=1500.0)
    meta   = {'prior_weight': prior_weight, 'anneal_frac': anneal_frac}

    all_y, all_ctx, all_w = [], [], []

    for coords, _ in source_data:
        p1, p2  = float(coords[0]), float(coords[1]) if len(coords) > 1 else 0.0
        ctx_enc = encode_ctx(*[float(v) for v in coords])
        analysis = analysis_map.get(coords)
        features = analysis['features'] if analysis and analysis['n_events'] > 0 else []

        if not features:
            raw = rng.exponential(scale=400.0, size=n_per_coord * 3) + mhh_lo
            raw = raw[(raw > mhh_lo) & (raw < mhh_hi)][:n_per_coord]
            if len(raw) < 2: continue
        else:
            raw_parts = []
            n_per_feat = max(n_per_coord // len(features), 50)
            for feat in features:
                ftype = feat.get('feature_type',
                                 classify_feature_type(feat, p1, p2, features))
                pos = feat['mhh_position']
                w   = max(feat['width_gev'], 5.0)

                if ftype == 'bw':
                    half_w = min(w / 2.0, 150.0)
                    r = rng.standard_cauchy(size=n_per_feat * 6) * half_w + pos

                elif ftype == 'interference_pos':
                    sigma = w / 2.355
                    r = rng.normal(pos, sigma, size=n_per_feat * 3)

                elif ftype == 'interference_neg':
                    if cred['neg_weight'] < 0.3:
                        continue
                    r = rng.uniform(mhh_lo, mhh_hi, size=n_per_feat * 6)
                    sigma   = max(w / 2.355, 5.0)
                    suppress = 1.0 - np.exp(-0.5 * ((r - pos) / sigma) ** 2)
                    suppress = np.clip(suppress, 0.02, 1.0)
                    keep    = rng.random(len(r)) < suppress
                    r       = r[keep]
                    n_scaled = max(2, int(n_per_feat * cred['neg_weight']))
                    r        = r[:n_scaled]

                elif ftype == 'threshold':
                    r = rng.exponential(scale=max(w, 30.0), size=n_per_feat * 3) + pos

                else:
                    half_w = min(w / 2.0, 150.0)
                    r = rng.standard_cauchy(size=n_per_feat * 6) * half_w + pos

                r = r[(r > mhh_lo) & (r < mhh_hi)][:n_per_feat]
                if len(r) >= 2:
                    raw_parts.append(r)

            if not raw_parts: continue
            raw = np.concatenate(raw_parts)

        y = transform_mhh(raw, T)
        y = y[y >= 0.0]
        if len(y) < 2: continue
        all_y.append(y)
        all_ctx.append(np.tile(ctx_enc, (len(y), 1)))
        all_w.append(np.ones(len(y)))

    if not all_y:
        if verbose:
            print('[PHYSICS PRIOR] No samples generated.')
        return None, None, None, meta

    y_all   = np.concatenate(all_y)
    ctx_all = np.concatenate(all_ctx)
    w_all   = np.concatenate(all_w)

    if verbose:
        type_counts = {}
        for a in analyses:
            for f in a.get('features', []):
                t = f.get('feature_type', '?')
                type_counts[t] = type_counts.get(t, 0) + 1
        print(f'[PHYSICS PRIOR] {len(y_all)} samples | '
              + ', '.join(f'{t}={n}' for t, n in sorted(type_counts.items())))

    return (
        torch.tensor(y_all,   dtype=torch.float32).unsqueeze(1),
        torch.tensor(ctx_all, dtype=torch.float32),
        torch.tensor(w_all,   dtype=torch.float32),
        meta,
    )


def build_track_prior(tracks, source_data, analyses,
                      n_per_coord=5000, prior_weight=0.30, anneal_frac=0.40):
    """Generate prior (y, ctx, w) tensors by extrapolating detected feature tracks.

    sign +1 (BW peak):        Cauchy centred at predicted mHH, HWHM = width/2.
    sign -1 (dip):            Uniform draw, then rejection-suppressed near dip.

    Also sweeps 200 dense synthetic contexts uniformly filling parameter space,
    so the prior covers the full interior of the grid and not just the grid points.

    Returns (y_t, ctx_t, w_t, meta) or (None, None, None, meta).
    """
    T   = cfg.MHH_THRESHOLD
    rng = np.random.default_rng()

    mhh_lo = min((a['mhh_lo'] for a in analyses if a['n_events'] > 0), default=float(T))
    mhh_hi = max((a['mhh_hi'] for a in analyses if a['n_events'] > 0), default=1500.0)
    meta   = {'prior_weight': prior_weight, 'anneal_frac': anneal_frac}

    all_fs   = [f for t in tracks for f in t['features']]
    p1_vals  = [f['p1']     for f in all_fs]
    lw_vals  = [f['p2_enc'] for f in all_fs]
    p1_scale = max(max(p1_vals) - min(p1_vals), 1.0)
    lw_scale = max(max(lw_vals) - min(lw_vals), 1.0)

    def _predict_pos_width(track_fs, p1, *rest_params):
        enc    = encode_ctx(p1, *rest_params) if rest_params else encode_ctx(p1)
        p2_enc = enc[1] if len(enc) > 1 else 0.0
        pos_gev    = [inverse_transform_mhh(f['y_pos'], T) for f in track_fs]
        widths_gev = [f['width_y'] * SCALE for f in track_fs]
        if len(track_fs) == 1:
            return pos_gev[0], widths_gev[0]
        X  = np.array([[1.0, f['p1']/p1_scale, f['p2_enc']/lw_scale] for f in track_fs])
        xc = np.array([1.0, p1/p1_scale, p2_enc/lw_scale])
        try:
            c_pos, _, _, _ = np.linalg.lstsq(X, np.array(pos_gev),    rcond=None)
            c_wid, _, _, _ = np.linalg.lstsq(X, np.array(widths_gev), rcond=None)
            return float(c_pos @ xc), float(max(c_wid @ xc, 10.0))
        except Exception:
            return float(np.mean(pos_gev)), float(np.mean(widths_gev))

    all_y, all_ctx_list, all_w_list = [], [], []

    # Dense synthetic sweep for interior coverage
    p1_lo, p1_hi = min(p1_vals), max(p1_vals)
    lw_lo, lw_hi = min(lw_vals), max(lw_vals)
    rng_d = np.random.default_rng(0)
    n_dense = 200
    if p1_hi > p1_lo and lw_hi > lw_lo:
        synth_p1 = rng_d.uniform(p1_lo, p1_hi, n_dense).tolist()
        if cfg.P2_SPACING == 'log':
            synth_p2 = np.exp(rng_d.uniform(lw_lo, lw_hi, n_dense)).tolist()
        else:
            synth_p2 = (rng_d.uniform(lw_lo, lw_hi, n_dense) * SCALE).tolist()
    else:
        synth_p1, synth_p2 = [], []

    n_per_track = max(n_per_coord // max(len(tracks), 1), 50)
    n_per_synth = max(n_per_track * len(source_data) // max(n_dense, 1), 10)

    grid_coords  = [tuple(float(v) for v in c) + (n_per_track,) for c, _ in source_data]
    synth_coords = [(p1, p2, n_per_synth) for p1, p2 in zip(synth_p1, synth_p2)]
    all_coords   = grid_coords + synth_coords

    for *coord_vals, n_this in all_coords:
        p1 = float(coord_vals[0])
        rest = [float(v) for v in coord_vals[1:]]
        ctx_enc = encode_ctx(*coord_vals)

        for track in tracks:
            peak_mhh, width_gev = _predict_pos_width(track['features'], p1, *rest)
            peak_mhh = float(np.clip(peak_mhh, mhh_lo + 1.0, mhh_hi - 1.0))
            half_w   = float(np.clip(width_gev / 2.0, 5.0, 150.0))

            if track['sign'] > 0:
                raw = rng.standard_cauchy(size=n_this * 6) * half_w + peak_mhh
            else:
                raw   = rng.uniform(mhh_lo, mhh_hi, size=n_this * 6)
                sigma = max(half_w / 0.6745, 5.0)
                supp  = 1.0 - np.exp(-0.5 * ((raw - peak_mhh) / sigma) ** 2)
                supp  = np.clip(supp, 0.02, 1.0)
                keep  = rng.random(n_this * 6) < supp
                raw   = raw[keep]

            raw = raw[(raw > mhh_lo) & (raw < mhh_hi)][:n_this]
            if len(raw) < 5:
                continue
            y = transform_mhh(raw, T)
            y = y[y >= 0.0]
            if len(y) < 2:
                continue
            all_y.append(y)
            all_ctx_list.append(np.tile(ctx_enc, (len(y), 1)))
            all_w_list.append(np.ones(len(y)))

    if not all_y:
        return None, None, None, meta

    y_all   = np.concatenate(all_y)
    ctx_all = np.concatenate(all_ctx_list)
    w_all   = np.concatenate(all_w_list)
    return (
        torch.tensor(y_all,   dtype=torch.float32).unsqueeze(1),
        torch.tensor(ctx_all, dtype=torch.float32),
        torch.tensor(w_all,   dtype=torch.float32),
        meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Comment parser + prior materialiser
# ─────────────────────────────────────────────────────────────────────────────

def parse_comment(comment, target, verbose=False):
    """Parse free-text physics description into a structured prior dict.

    Recognised keywords: bw/resonance/narrow/bump, threshold/kinematic/edge,
    positive/constructive interference, negative/destructive interference/dip,
    continuum/smooth/falling.

    Returns dict with keys 'components', 'prior_weight', 'anneal_frac', or None.
    """
    mS  = float(target[0])
    WoMS = float(target[1]) if len(target) > 1 else 0.0
    txt = comment.lower()
    txt = re.sub(r'woms\s*[\*x]\s*ms', str(WoMS * mS), txt)
    txt = re.sub(r'ms\s*[\*x]\s*woms', str(WoMS * mS), txt)
    txt = re.sub(r'\bwoms\b', str(WoMS), txt)
    txt = re.sub(r'\bms\b',   str(mS),   txt)

    def _near_num(pattern, fallback=None):
        m = re.search(pattern + r'[\s\w]*?([\d.]+)', txt)
        if m:
            try: return float(m.group(1))
            except ValueError: pass
        return fallback

    components = []

    if re.search(r'\b(b\.?w\.?|breit.?w\w*|resso?nan\w*|peaks?|signals?|narrow|bump|resonant)\b', txt):
        peak  = _near_num(r'centered?(\s+at)?', fallback=mS)
        width = _near_num(r'width', fallback=WoMS * mS if WoMS > 0 else 50.0)
        components.append({'type': 'bw', 'peak': max(peak, 250.0),
                           'width': max(width, 10.0), 'sign': +1})

    if re.search(r'\b(threshold|kinematic|edges?|onset|opening|turn.?on)\b', txt):
        pos = _near_num(r'(threshold|edge|onset|opening)\s+(at|around|near)', fallback=2*PDG_H_MASS)
        components.append({'type': 'threshold', 'peak': pos, 'width': 30.0, 'sign': +1})

    if re.search(r'\b(positive\s+interf\w*|constructive\s+interf\w*|interf\w*.*posit|enhanc\w*)\b', txt):
        peak  = _near_num(r'(interf\w*)\s+(at|around|near)', fallback=mS)
        width = _near_num(r'width', fallback=WoMS * mS * 2 if WoMS > 0 else 100.0)
        components.append({'type': 'interference_pos', 'peak': peak,
                           'width': max(width, 20.0), 'sign': +1})

    if re.search(r'\b(negative\s+interf\w*|destructive\s+interf\w*|interf\w*.*neg|dips?|notch|suppress\w*)\b', txt):
        peak  = _near_num(r'(interf\w*|dip|notch)\s+(at|around|near)', fallback=mS * 0.8)
        width = _near_num(r'width', fallback=WoMS * mS * 2 if WoMS > 0 else 100.0)
        components.append({'type': 'interference_neg', 'peak': peak,
                           'width': max(width, 20.0), 'sign': -1})

    if re.search(r'\b(continuum|smooth|background|falling|flat|power.?law|expo\w*|slowly)\b', txt):
        components.append({'type': 'continuum', 'peak': None, 'width': None, 'sign': +1})

    if not components:
        if verbose:
            print('[COMMENT] No physics keywords found — comment ignored.')
        return None

    parsed = {'components': components, 'prior_weight': 0.30, 'anneal_frac': 0.10}
    if verbose:
        print('[COMMENT] Parsed:')
        for c in components:
            if c['type'] == 'bw':
                print(f"  BW: centre={c['peak']:.1f} GeV, width={c['width']:.1f} GeV")
            elif c['type'] == 'threshold':
                print(f"  Threshold at {c['peak']:.1f} GeV")
            elif 'interference' in c['type']:
                print(f"  {c['type']}: centre={c['peak']:.1f} GeV, width={c['width']:.1f} GeV")
            elif c['type'] == 'continuum':
                print('  Continuum/smooth')
    return parsed


def generate_prior_samples(parsed, source_data, n_per_coord, mhh_min, mhh_max):
    """Materialise a parsed comment prior into (y_t, ctx_t, w_t) tensors on CPU."""
    T   = cfg.MHH_THRESHOLD
    rng = np.random.default_rng()
    all_y, all_ctx, all_w = [], [], []

    for coords, _ in source_data:
        p1  = float(coords[0])
        p2  = float(coords[1]) if len(coords) > 1 else 0.0
        ctx_enc = encode_ctx(*[float(v) for v in coords])

        for c in parsed['components']:
            n    = max(n_per_coord // len(parsed['components']), 10)
            peak = c['peak']  if c['peak']  is not None else p1
            wid  = c['width'] if c['width'] is not None else p2 * p1

            if c['type'] == 'bw':
                raw = rng.standard_cauchy(size=n * 6) * wid + peak
                raw = raw[(raw > mhh_min) & (raw < mhh_max)][:n]
                if len(raw) < 5: continue

            elif c['type'] == 'threshold':
                raw = rng.exponential(scale=150.0, size=n * 3) + peak
                raw = raw[(raw > mhh_min) & (raw < mhh_max)][:n]
                if len(raw) < 5: continue

            elif c['type'] == 'interference_pos':
                raw = rng.normal(loc=peak, scale=max(wid / 2.355, 1.0), size=n * 3)
                raw = raw[(raw > mhh_min) & (raw < mhh_max)][:n]
                if len(raw) < 5: continue

            elif c['type'] == 'interference_neg':
                raw = rng.uniform(mhh_min, mhh_max, size=n * 3)
                ww  = 1.0 - np.exp(-0.5 * ((raw - peak) / max(wid / 2.355, 1.0)) ** 2)
                ww  = np.clip(ww, 0.05, 1.0)
                raw = raw[(raw > mhh_min) & (raw < mhh_max)]
                ww  = ww[:len(raw)]
                idx = rng.choice(len(raw), size=min(n, len(raw)), replace=False, p=ww / ww.sum())
                raw = raw[idx]

            elif c['type'] == 'continuum':
                raw = rng.exponential(scale=500.0, size=n * 3) + mhh_min
                raw = raw[(raw > mhh_min) & (raw < mhh_max)][:n]
                if len(raw) < 5: continue

            else:
                continue

            y = transform_mhh(raw, T)
            y = y[y >= 0.0]
            if len(y) < 2: continue
            all_y.append(y)
            all_ctx.append(np.tile(ctx_enc, (len(y), 1)))
            all_w.append(np.ones(len(y)))

    if not all_y:
        return None, None, None
    return (
        torch.tensor(np.concatenate(all_y),   dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.concatenate(all_ctx), dtype=torch.float32),
        torch.tensor(np.concatenate(all_w),   dtype=torch.float32),
    )
