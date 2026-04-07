"""core/config.py — Global state, coordinate transforms, grid discovery.

All mutable globals live here and are initialised once by init_globals().
Every other module imports from this module; none of them define globals.
"""

import os
import re
import glob
import math
import bisect
import gzip
import itertools
import concurrent.futures
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Module-level globals (set by init_globals)
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_TYPE  = None   # str, e.g. 'S_13t'
CACHE_DIR      = None   # e.g. './S_13t/LHE_pt_cache_thr250'
MODELS_DIR     = None   # e.g. './S_13t/Models'
DEVICE         = None   # torch.device

MHH_THRESHOLD  = None   # float GeV — kinematic threshold for y-transform
SCALE          = 1000.0  # GeV — denominator in linear-shift transform

TOL = 1e-6

# Parameter-space metadata — set by build_grid_index()
PARAM_NAMES    = None   # list[str], e.g. ['mS1', 'WoMS1']
PARAM_SPACINGS = None   # list[str], 'linear' or 'log' per axis
PARAM_PAIR     = None   # legacy 2-D alias
P1_SPACING     = 'linear'
P2_SPACING     = 'linear'

# Full grid index: {coords_tuple: [lhe.gz files]}
GRID_FILES     = None

# Data base path — overridden by LAPTOP_MODE in morph_ml.py
DB_BASE = '/cfs/data/pg/SHIFT/diHiggs_Full/LHE'

# Topology: set once in init_globals after build_grid_index.
TOPOLOGY = None


# ─────────────────────────────────────────────────────────────────────────────
# init_globals
# ─────────────────────────────────────────────────────────────────────────────

def init_globals(database_type, device, mhh_threshold=None,
                 laptop_mode=False, verbose=False):
    """Initialise all module-level globals. Must be called once before use."""
    global DATABASE_TYPE, CACHE_DIR, MODELS_DIR, DEVICE
    global MHH_THRESHOLD, GRID_FILES, DB_BASE
    global PARAM_NAMES, PARAM_SPACINGS, PARAM_PAIR, P1_SPACING, P2_SPACING
    global TOPOLOGY

    if laptop_mode:
        DB_BASE = '../SUData_Full/LHE'

    DATABASE_TYPE = database_type
    DEVICE        = device

    _lhe_available = os.path.isdir(DB_BASE)

    if _lhe_available:
        GRID_FILES = build_grid_index(database_type)

    if mhh_threshold is not None:
        MHH_THRESHOLD = mhh_threshold
    elif _lhe_available:
        MHH_THRESHOLD = detect_mhh_threshold(verbose=verbose)
    else:
        MHH_THRESHOLD = _detect_threshold_from_cache(database_type, verbose=verbose)

    # Cache dir encodes threshold so caches from different runs don't mix.
    CACHE_DIR  = f'./{DATABASE_TYPE}/LHE_pt_cache_thr{int(MHH_THRESHOLD)}'
    MODELS_DIR = f'./{DATABASE_TYPE}/Models'
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not _lhe_available:
        GRID_FILES = build_grid_index_from_cache(database_type, CACHE_DIR)

    TOPOLOGY = parse_database_topology(database_type)


# ─────────────────────────────────────────────────────────────────────────────
# Cache-only helpers (no LHE files present)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_threshold_from_cache(database_type, verbose=False):
    """Return threshold from threshold.txt or cache-dir name when LHE is unavailable."""
    thr_file = f'./{database_type}/threshold.txt'
    if os.path.exists(thr_file):
        with open(thr_file) as fh:
            t = float(fh.read().strip())
        if verbose:
            print(f'[Threshold] From file: {t:.2f} GeV')
        return t
    for d in sorted(glob.glob(f'./{database_type}/LHE_pt_cache_thr*')):
        m = re.search(r'thr(\d+)', os.path.basename(d))
        if m:
            t = float(m.group(1))
            if verbose:
                print(f'[Threshold] From cache dir name: {t:.2f} GeV')
            return t
    if verbose:
        print('[Threshold] Not found — using 245.0 GeV')
    return 245.0


def build_grid_index_from_cache(database_type, cache_dir):
    """Build GRID_FILES from pre-computed .pt cache files (no LHE needed).

    Virtual lhe.gz paths are constructed so that preprocess_file() resolves
    the correct cache entry via os.path.basename(lhe_file) + '.p<N>.pt'.
    """
    global PARAM_NAMES, PARAM_SPACINGS, PARAM_PAIR, P1_SPACING, P2_SPACING

    if not os.path.isdir(cache_dir):
        raise RuntimeError(
            f"LHE data not found at '{DB_BASE}' and no cache at '{cache_dir}'.\n"
            "Run on a machine with LHE access first to populate the cache.")

    db_prefix      = f'hh_{database_type}_'
    detected_names = None
    grid           = {}

    for fname in sorted(os.listdir(cache_dir)):
        m = re.search(r'\.p(\d+)\.pt$', fname)
        if not m:
            continue
        lhe_gz_name = fname[: -len(f'.p{m.group(1)}.pt')]   # strip .p<N>.pt, keeps .lhe.gz
        parsed = _parse_dir(lhe_gz_name, db_prefix)
        if not parsed:
            continue
        names_here = tuple(p[0] for p in parsed)
        if detected_names is None:
            detected_names = names_here
        elif names_here != detected_names:
            continue
        coords       = tuple(p[1] for p in parsed)
        virtual_path = os.path.join(cache_dir, lhe_gz_name)  # need not exist on disk
        grid.setdefault(coords, []).append(virtual_path)

    if not grid:
        raise RuntimeError(f"No usable .pt cache files found in '{cache_dir}'.")

    n_params = len(detected_names)
    spacings = []
    for i in range(n_params):
        name = detected_names[i]
        if name.startswith('WoM') or name.lower().startswith('w'):
            sp = 'log'
        elif name.startswith('m') or name.startswith('M'):
            sp = 'linear'
        else:
            axis_vals = sorted(set(c[i] for c in grid))
            sp, _ = _detect_spacing(axis_vals)
        spacings.append(sp)

    PARAM_NAMES    = list(detected_names)
    PARAM_SPACINGS = spacings
    PARAM_PAIR     = (PARAM_NAMES[0], PARAM_NAMES[1]) if n_params >= 2 else (PARAM_NAMES[0], PARAM_NAMES[0])
    P1_SPACING     = PARAM_SPACINGS[0]
    P2_SPACING     = PARAM_SPACINGS[1] if n_params >= 2 else 'linear'
    return {c: sorted(v) for c, v in grid.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

def transform_mhh(mhh, threshold=None):
    """y = (mHH − threshold) / SCALE  (linear-shift; no log singularity at threshold).

    Linear spacing preserves the proportionality between BW peak positions and mS,
    which makes interpolation across the mS axis more natural and numerically stable.
    SCALE=1000 GeV keeps y ∈ [0, ~5] for typical 250–5000 GeV ranges.
    """
    import torch
    T = threshold if threshold is not None else MHH_THRESHOLD
    if isinstance(mhh, torch.Tensor):
        return (mhh - T) / SCALE
    return (np.asarray(mhh, dtype=np.float64) - T) / SCALE


def inverse_transform_mhh(y, threshold=None):
    """mHH = y × SCALE + threshold  (invert transform_mhh)."""
    import torch
    T = threshold if threshold is not None else MHH_THRESHOLD
    if isinstance(y, torch.Tensor):
        return y * SCALE + T
    return np.asarray(y, dtype=np.float64) * SCALE + T


def encode_ctx(*params):
    """Map N physical parameter values → normalised context vector (list of floats).

    Log-spaced axes: x = log(p).  Linear-spaced axes: x = p / SCALE.
    Accepts either encode_ctx(p1, p2, ...) or encode_ctx((p1, p2, ...)).
    """
    if len(params) == 1 and hasattr(params[0], '__len__'):
        params = params[0]
    spacings = PARAM_SPACINGS or ['linear'] * len(params)
    result = []
    for i, p in enumerate(params):
        sp = spacings[i] if i < len(spacings) else 'linear'
        if sp == 'log':
            result.append(math.log(float(p)) if float(p) > 0 else -10.0)
        else:
            result.append(float(p) / SCALE)
    return result


def decode_param_enc(xi, axis_idx):
    """Invert encode_ctx for one axis."""
    sp = PARAM_SPACINGS[axis_idx] if PARAM_SPACINGS and axis_idx < len(PARAM_SPACINGS) else 'linear'
    return math.exp(xi) if sp == 'log' else xi * SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_str(target, sep=', '):
    names = PARAM_NAMES or [f'p{i+1}' for i in range(len(target))]
    return sep.join(f'{names[i]}={target[i]}' for i in range(len(target)))


def _target_tag(target):
    names = PARAM_NAMES or [f'p{i+1}' for i in range(len(target))]
    return DATABASE_TYPE + '_' + '_'.join(f'{names[i]}_{target[i]}' for i in range(len(target)))


def _mode_suffix(target):
    names = PARAM_NAMES or [f'p{i+1}' for i in range(len(target))]
    return '_'.join(f'{names[i]}_{target[i]}' for i in range(len(target)))


# ─────────────────────────────────────────────────────────────────────────────
# Grid discovery
# ─────────────────────────────────────────────────────────────────────────────

# Recognised parameter tokens in directory names.
_KNOWN_TOKENS = ['m1', 'm2', 'm3', 'm4', 'mS1', 'WoMS1', 'mS2', 'WoMS2']


def _parse_dir(d, db_prefix):
    """Parse a directory name into [(token, value), ...] or None."""
    if not d.startswith(db_prefix):
        return None
    parts  = d[len(db_prefix):].split('_')
    result = []
    i = 0
    while i < len(parts):
        for tok in _KNOWN_TOKENS:
            tok_parts = tok.split('_')
            if parts[i:i+len(tok_parts)] == tok_parts:
                j = i + len(tok_parts)
                if j < len(parts):
                    try:
                        val = float(parts[j])
                        result.append((tok, val))
                        i = j + 1
                        break
                    except ValueError:
                        pass
        else:
            i += 1
    return result if result else None


def _detect_spacing(sorted_vals, target=None, tol=TOL):
    """Detect grid spacing ('linear' or 'log') on a single axis.

    Log spacing is only declared when values span ≥ one order of magnitude
    (max/min ≥ 10) — this prevents false-positive log detection on compact grids.
    """
    n = len(sorted_vals)
    if n < 2:
        return 'linear', 1
    if all(v > 0 for v in sorted_vals) and sorted_vals[-1] / sorted_vals[0] >= 10.0:
        spacing = 'log'
    else:
        spacing = 'linear'
    n_per_side = 1
    if spacing == 'linear' and target is not None:
        diffs   = [sorted_vals[i+1] - sorted_vals[i] for i in range(n-1)]
        avg_step = sum(diffs) / len(diffs)
        ref = abs(target) if abs(target) > tol else abs(sorted_vals[0])
        n_per_side = 2 if (ref > 0 and avg_step / ref < 0.30) else 1
    return spacing, n_per_side


def build_grid_index(database_type):
    """Scan DB_BASE for LHE grid directories and build {coords: [files]} index.

    Sets PARAM_NAMES, PARAM_SPACINGS and legacy 2-D aliases as side effects.
    Supports 1–4 parameter dimensions auto-detected from directory names.

    Returns:
        dict: {(p1, p2, ...): [lhe.gz paths]}
    """
    global PARAM_NAMES, PARAM_SPACINGS, PARAM_PAIR, P1_SPACING, P2_SPACING

    db_prefix      = f'hh_{database_type}_'
    detected_names = None
    grid           = {}

    for d in sorted(os.listdir(DB_BASE)):
        parsed = _parse_dir(d, db_prefix)
        if not parsed:
            continue
        names_here = tuple(p[0] for p in parsed)
        if detected_names is None:
            detected_names = names_here
        elif names_here != detected_names:
            continue  # skip dirs with unexpected token structure
        coords = tuple(p[1] for p in parsed)
        files  = glob.glob(os.path.join(DB_BASE, d, '*.lhe.gz'))
        if files:
            grid[coords] = sorted(files)

    if not grid:
        raise RuntimeError(
            f"No LHE grid directories found for database '{database_type}' in {DB_BASE}.")

    n_params = len(detected_names)
    spacings = []
    for i in range(n_params):
        name = detected_names[i]
        # Name-based convention: mass params always linear, WoM* always log,
        # regardless of grid range (irregular grids break the ratio heuristic).
        if name.startswith('WoM') or name.lower().startswith('w'):
            sp = 'log'
        elif name.startswith('m') or name.startswith('M'):
            sp = 'linear'
        else:
            axis_vals = sorted(set(c[i] for c in grid))
            sp, _ = _detect_spacing(axis_vals)
        spacings.append(sp)

    PARAM_NAMES    = list(detected_names)
    PARAM_SPACINGS = spacings
    PARAM_PAIR     = (PARAM_NAMES[0], PARAM_NAMES[1]) if n_params >= 2 else (PARAM_NAMES[0], PARAM_NAMES[0])
    P1_SPACING     = PARAM_SPACINGS[0]
    P2_SPACING     = PARAM_SPACINGS[1] if n_params >= 2 else 'linear'
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Neighbour selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_neighbors_1d(sorted_vals, target, spacing, n_per_side, tol=TOL):
    """Return up to n_per_side grid values below and above target.

    Boundary fallback: if one side is exhausted, the shortfall is drawn from
    the other side so the total neighbour count stays close to 2×n_per_side.
    The exact target value is never included (excluded by caller if needed).
    """
    if spacing == 'log':
        log_t      = math.log(target) if target > 0 else -1e30
        candidates = sorted(
            (v for v in sorted_vals if abs(v - target) > tol and v > 0),
            key=lambda v: abs(math.log(v) - log_t),
        )
    else:
        candidates = sorted(
            (v for v in sorted_vals if abs(v - target) > tol),
            key=lambda v: abs(v - target),
        )

    below = [v for v in candidates if v < target - tol]
    above = [v for v in candidates if v > target + tol]

    sel_below = below[-n_per_side:]
    sel_above = above[:n_per_side]

    shortage_above = n_per_side - len(sel_above)
    shortage_below = n_per_side - len(sel_below)
    if shortage_above > 0 and below:
        sel_below = below[-(n_per_side + shortage_above):]
    if shortage_below > 0 and above:
        sel_above = above[:n_per_side + shortage_below]

    return sorted(set(sel_below + sel_above))


def get_files_for_target(target, verbose=True):
    """N-dimensional neighbour selection for a target parameter tuple.

    For each axis independently selects bracketing grid values; the total
    candidate set is the Cartesian product, capped at ~3^N points.
    The exact target is excluded (handled by on-grid passthrough in main).

    Returns:
        list of (coords_tuple, [lhe.gz files])
    """
    target_vals = [float(v) for v in target]
    n_params    = len(target_vals)
    spacings    = PARAM_SPACINGS or ['linear'] * n_params

    all_axis = [sorted(set(c[i] for c in GRID_FILES)) for i in range(n_params)]
    on_grid  = [any(abs(v - target_vals[i]) < TOL for v in all_axis[i]) for i in range(n_params)]

    axis_vals = []
    for i in range(n_params):
        sp, n_ps = _detect_spacing(all_axis[i], target_vals[i])
        neighbours = _select_neighbors_1d(all_axis[i], target_vals[i], sp, n_ps)
        if on_grid[i]:
            axis_vals.append([target_vals[i]] + neighbours)
        else:
            axis_vals.append(neighbours)

    # Cap at 3^N total to avoid O(N^4) explosion for 4-D grids
    max_per_axis = max(1, int(round(27 ** (1.0 / max(n_params, 1)))))
    axis_vals    = [ax[:max_per_axis] for ax in axis_vals]

    result_coords = {
        combo
        for combo in itertools.product(*axis_vals)
        if not all(abs(combo[i] - target_vals[i]) < TOL for i in range(n_params))
    }

    result = [
        (c, GRID_FILES[c])
        for c in sorted(result_coords)
        if c in GRID_FILES
    ]

    if not result:
        raise RuntimeError(f'No neighbours found for target {target}')

    if verbose:
        names   = PARAM_NAMES or [f'p{i+1}' for i in range(n_params)]
        on_off  = ['on' if on_grid[i] else 'off' for i in range(n_params)]
        print('Neighbour selection: '
              + '  '.join(f'{names[i]}={on_off[i]}-grid' for i in range(n_params)))
        print(f'Training grid points ({len(result)}):')
        for coords, files in result:
            cs = '  '.join(f'{names[i]}={coords[i]:.6g}' for i in range(n_params))
            print(f'  {cs}  ({len(files)} file(s))')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Database topology
# ─────────────────────────────────────────────────────────────────────────────

def parse_database_topology(database_type):
    """Parse database type string to infer expected physics features.

    Parsing rules for the topology prefix (first '_'-delimited token):
      's'  (lowercase) = one squark in the loop  → threshold at 2·m_squark
      'ss'             = squark-squark           → thresholds ± interference
      'S'  (uppercase) = one neutral scalar      → BW resonance at mS
      'SS'             = two scalars             → double resonance ± dips
      'Ss' or 'sS'     = squark + scalar         → threshold + BW
      'B'              = SM background included  → adds dip from SM interference

    Returns dict with topology flags.
    """
    topo_str = database_type.split('_')[0]

    n_squarks = topo_str.count('s')   # lowercase s = squark
    n_scalars  = topo_str.count('S')  # uppercase S = neutral scalar
    has_sm_bg  = 'B' in topo_str

    has_squarks     = n_squarks > 0
    has_scalars     = n_scalars > 0
    has_interference = (n_squarks + n_scalars + int(has_sm_bg)) >= 2

    expected_positive = []
    if has_squarks:
        expected_positive.append('threshold')
    if has_scalars:
        expected_positive.append('bw')
    if not expected_positive:
        expected_positive.append('bw')

    expected_negative = []
    if has_interference:
        expected_negative.append('interference_neg')

    return {
        'has_squarks':       has_squarks,
        'has_scalars':       has_scalars,
        'has_sm_bg':         has_sm_bg,
        'has_interference':  has_interference,
        'n_squarks':         n_squarks,
        'n_scalars':         n_scalars,
        'squarks_only':      has_squarks and not has_scalars and not has_sm_bg,
        'scalars_only':      has_scalars and not has_squarks and not has_sm_bg,
        'single_bw':         has_scalars and not has_squarks and not has_sm_bg and n_scalars == 1,
        'expected_positive': expected_positive,
        'expected_negative': expected_negative,
        'topo_str':          topo_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic threshold detection
# ─────────────────────────────────────────────────────────────────────────────

def _scan_file_min_mhh(lhe_file):
    """Return minimum mHH value found in lhe_file (no filtering)."""
    min_val = float('inf')
    try:
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
                            if mhh < min_val:
                                min_val = mhh
                    in_event = False; continue
                if not in_event:
                    continue
                toks = l.split()
                if len(toks) < 6:
                    continue
                try:
                    pdg = int(toks[0])
                except ValueError:
                    continue
                if abs(pdg) == 25:  # SM Higgs (PDG id 25)
                    try:
                        px, py, pz, e = float(toks[6]), float(toks[7]), float(toks[8]), float(toks[9])
                        if len(H) < 2:
                            H.append([px, py, pz, e])
                    except (IndexError, ValueError):
                        pass
    except Exception:
        pass
    return min_val


def detect_mhh_threshold(n_files=5, verbose=False):
    """Scan all grid files in parallel to find the global min mHH.

    Result is cached in ./<DATABASE_TYPE>/threshold.txt so the CFS scan runs
    only once per database.  Returns floor(min_mhh) - 1 GeV.
    Falls back to 245.0 GeV if no files are readable.
    """
    os.makedirs(f'./{DATABASE_TYPE}', exist_ok=True)
    threshold_file = f'./{DATABASE_TYPE}/threshold.txt'
    if os.path.exists(threshold_file):
        with open(threshold_file) as fh:
            t = float(fh.read().strip())
        if verbose:
            print(f'[Threshold] Cached: {t:.2f} GeV')
        return t

    all_files = [f for files in GRID_FILES.values() for f in files]
    if not all_files:
        if verbose:
            print('[WARNING] No grid files found — using 245.0 GeV')
        return 245.0

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    n_workers  = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)
    if verbose:
        print(f'[Threshold] Scanning {len(all_files)} files ({n_workers} workers)...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        mins = list(ex.map(_scan_file_min_mhh, all_files))

    finite = [m for m in mins if m < float('inf')]
    if not finite:
        if verbose:
            print('[WARNING] Could not detect mHH threshold — using 245.0 GeV')
        return 245.0

    global_min = min(finite)
    threshold  = global_min - 1.0

    with open(threshold_file, 'w') as fh:
        fh.write(str(threshold))

    if verbose:
        print(f'[Threshold] Auto-detected: {threshold:.2f} GeV  '
              f'(min observed: {global_min:.2f} GeV)')
    return threshold
