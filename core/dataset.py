"""core/dataset.py — LHE file preprocessing, dataset, Voronoi weights.

Tensor schema per cache file:
    columns 0..n_params-1 : encoded context (encode_ctx output)
    column  n_params       : transformed mHH  y = (mHH - threshold) / SCALE

All events with mHH <= threshold are discarded at preprocessing time.
"""

import os
import gzip
import math
import numpy as np
import torch
from torch.utils.data import Dataset

import core.config as cfg
from core.config import (
    SCALE, TOL,
    encode_ctx, transform_mhh,
)

PDG_H = 25   # SM Higgs PDG id

# ─────────────────────────────────────────────────────────────────────────────
# LHE → tensor cache
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_file(lhe_file, coords):
    """Parse one .lhe.gz file and cache the result as a float32 tensor.

    The cache file encodes both the number of parameters and the threshold value
    so that different runs with different settings never share stale caches.

    Returns:
        str: path to the cached .pt file (already exists or freshly written).
    """
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)

    n_params_tag = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2
    cache_file = os.path.join(
        cfg.CACHE_DIR,
        os.path.basename(lhe_file) + f'.p{n_params_tag}.pt',
    )
    if os.path.exists(cache_file):
        return cache_file

    ctx_enc  = encode_ctx(*coords)
    n_params = len(ctx_enc)
    rows     = []

    with gzip.open(lhe_file, 'rt') as fh:
        in_event = False
        H = []
        for line in fh:
            l = line.strip()
            if not l:
                continue
            if l.startswith('<event>'):
                in_event = True; H = []; continue
            if l.startswith('</event>'):
                if len(H) >= 2:
                    p  = np.array(H[0]) + np.array(H[1])
                    m2 = p[3]**2 - (p[0]**2 + p[1]**2 + p[2]**2)
                    if m2 > 0:
                        mhh = math.sqrt(m2)
                        if mhh > cfg.MHH_THRESHOLD:
                            rows.append(ctx_enc + [transform_mhh(mhh, cfg.MHH_THRESHOLD)])
                in_event = False; continue
            if not in_event or l.startswith('#') or l.startswith('<'):
                continue
            parts = l.split()
            if len(parts) < 10:
                continue
            try:
                if int(parts[0]) == PDG_H and int(parts[1]) == 1:
                    px, py, pz, E = map(float, parts[6:10])
                    if len(H) < 2:
                        H.append([px, py, pz, E])
            except ValueError:
                continue

    expected_cols = n_params + 1
    if rows:
        tensor = torch.tensor(rows, dtype=torch.float32).contiguous()
    else:
        tensor = torch.empty((0, expected_cols), dtype=torch.float32)

    if tensor.ndim != 2 or tensor.shape[1] != expected_cols:
        raise RuntimeError(
            f'Schema error in {lhe_file}: expected (N,{expected_cols}), got {tensor.shape}')

    torch.save(tensor, cache_file)
    return cache_file


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_voronoi_weights(grid_coords, verbose=False):
    """Compute Voronoi cell area per grid point in encoded parameter space.

    The area is the product of per-axis Voronoi widths, computed in the
    encoded (normalised / log) coordinate system.  Areas are left un-normalised;
    LheDataset divides each by the per-point event count before training.

    Returns:
        dict: {coords_tuple: area_float}
    """
    coords_list = list(grid_coords)
    if not coords_list:
        return {}
    n_params = len(coords_list[0])
    spacings = cfg.PARAM_SPACINGS or ['linear'] * n_params

    def _enc_axis(vals, sp):
        return [math.log(v) if v > 0 else -10.0 for v in vals] if sp == 'log' \
               else [v / SCALE for v in vals]

    def _cell_width(enc_vals, idx):
        n = len(enc_vals)
        left  = (enc_vals[idx] - enc_vals[idx-1]) / 2.0 if idx > 0   else (enc_vals[1]  - enc_vals[0])  / 2.0
        right = (enc_vals[idx+1] - enc_vals[idx]) / 2.0 if idx < n-1 else (enc_vals[-1] - enc_vals[-2]) / 2.0
        return left + right

    all_axis_raw = [sorted(set(c[i] for c in coords_list)) for i in range(n_params)]
    all_axis_enc = [_enc_axis(all_axis_raw[i], spacings[i] if i < len(spacings) else 'linear')
                    for i in range(n_params)]

    weights = {}
    for c in coords_list:
        area = 1.0
        for i in range(n_params):
            raw_vals = all_axis_raw[i]
            enc_vals = all_axis_enc[i]
            try:
                idx = raw_vals.index(c[i])
            except ValueError:
                # Nearest grid point by value
                idx = min(range(len(raw_vals)), key=lambda k: abs(raw_vals[k] - c[i]))
            area *= _cell_width(enc_vals, idx)
        weights[c] = area

    if verbose:
        print(f'Voronoi weights computed for {len(weights)} grid points.')
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LheDataset(Dataset):
    """Weighted dataset from a list of (coords, [lhe.gz files]) pairs.

    Weighting strategy
    ------------------
    Each grid point's total gradient contribution equals its Voronoi cell area
    in encoded parameter space, divided by its event count.  This corrects for
    non-uniform grid density and unequal CPU time spent per point.

    Within a grid point all events are weighted equally — the MC sample is an
    unbiased draw from the true distribution and must not be re-weighted.
    Reweighting within a point (e.g. flat-density correction) would actively
    down-weight the BW peak and train the model to fit noise.

    Parameters
    ----------
    source_data    : list of (coords_tuple, [lhe.gz paths])
    voronoi_weights: dict from compute_voronoi_weights() or None (uniform)
    verbose        : print per-point statistics
    """

    def __init__(self, source_data, voronoi_weights=None, verbose=False):
        all_features, all_targets, all_weights = [], [], []

        for coords, files in source_data:
            tensors = []
            for f in files:
                cache = preprocess_file(f, coords)
                try:
                    t = torch.load(cache, weights_only=True)
                except Exception as e:
                    raise RuntimeError(f'Failed to load cache {cache}: {e}')
                if not isinstance(t, torch.Tensor):
                    raise RuntimeError(f'Cache {cache} does not contain a tensor.')
                if t.shape[0] > 0:
                    tensors.append(t)

            if not tensors:
                continue

            pt = torch.cat(tensors, dim=0)   # (N, n_params+1)
            n  = pt.shape[0]

            voronoi_area = float(voronoi_weights[coords]) \
                           if voronoi_weights and coords in voronoi_weights else 1.0

            # For full-grid training: points where mS1 < 2*mH have no BW resonance.
            # Their Voronoi cells can be huge (edge of grid) and dominate continuum
            # learning. Cap their weight to the median of BW-capable points.
            if voronoi_weights and cfg.PARAM_NAMES:
                _ms_idx = next((k for k, nm in enumerate(cfg.PARAM_NAMES) if nm.startswith('mS') or nm.startswith('m')), -1)
                if _ms_idx >= 0:
                    _ms_val = float(coords[_ms_idx])
                    _PDG_H  = 125.09
                    if _ms_val < 2.0 * _PDG_H:
                        _bw_areas = [float(voronoi_weights[c])
                                     for c in voronoi_weights
                                     if float(c[_ms_idx]) >= 2.0 * _PDG_H]
                        if _bw_areas:
                            _cap = float(np.median(_bw_areas))
                            voronoi_area = min(voronoi_area, _cap)

            w = voronoi_area / n

            all_features.append(pt[:, :-1])
            all_targets.append(pt[:, -1:])
            all_weights.append(torch.full((n,), w, dtype=torch.float32))

            if verbose:
                cs = '  '.join(f'{v:.4g}' for v in coords)
                print(f'  ({cs}):  n={n:,}  voronoi={voronoi_area:.4f}  w={w:.2e}')

        if not all_features:
            raise RuntimeError('LheDataset is empty — no events loaded.')

        self.features = torch.cat(all_features, dim=0)
        self.targets  = torch.cat(all_targets,  dim=0)
        self.weights  = torch.cat(all_weights,  dim=0)
        self.total    = self.features.shape[0]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.weights[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Re-export get_files_for_target for callers that import from dataset
# ─────────────────────────────────────────────────────────────────────────────
from core.config import get_files_for_target   # noqa: F401 (re-export)
