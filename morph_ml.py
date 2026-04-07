#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
#
# morph_ml.py  —  Universal BSM mHH morphing via surrogate models.
#
# Usage:
#   python3 -u morph_ml.py dbname p1 [p2 p3 p4] [OPTIONS]
#
# Supported models (auto-selected by feature detection or forced via --model):
#   mdn   — Skew-t Mixture Density Network (physics-initialised; best for pure BW peaks)
#   nsf   — Neural Spline Flow (handles interference dips; no physics init)
#   ebm   — Conditional Energy-Based Model (unconstrained log-density + GL quadrature;
#            handles dips AND admits physics init; recommended for mixed cases)
#   mdn+g — MDN with signed log-correction exp(g); backward-compatible, dip-capable
#   eft   — Quadratic EFT morphing baseline (no training; exact within perturbation theory)
#
# All models share a common CLI, cache structure, and plotting pipeline.

import sys
import os

# CUDA must be disabled before importing torch if --cpu is requested.
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

LAPTOP_MODE = '--laptop' in sys.argv

import argparse
try:
    import argcomplete
    _ARGCOMPLETE = True
except ImportError:
    _ARGCOMPLETE = False

import math
import time
import numpy as np
import torch

# ── Core modules ────────────────────────────────────────────────────────────
from core.config import (
    init_globals, DB_BASE, SCALE,
    build_grid_index, detect_mhh_threshold,
    _target_str, _target_tag, _mode_suffix,
)
from core.dataset  import preprocess_file, LheDataset, compute_voronoi_weights
from core.features import (
    extract_features_at_coords, build_feature_tracks,
    reclassify_paired_tracks, auto_nsf_hyperparams,
)
from core.training import select_and_train, monitor_grid_reproduction
from core.morphing import eft_morphing, bdt_morphing
from core.plotting import (
    plot_target_only, plot_with_neighbours, plot_partial_distribution,
    get_plot_neighbours,
)

# ── Model registry ───────────────────────────────────────────────────────────
from core.models import NSFModel, MDNModel, EBMModel, MDNCorrModel
from core.training import sample_model

import core.config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description='BSM mHH surrogate morphing (MDN / NSF / EBM / MDN+g / EFT)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('database_type', type=str,
                   help='Database label (e.g. S_13t). Directories are hh_<db>_*.')
    p.add_argument('params', nargs='+', type=float,
                   help='Target BSM parameter values (1–4 floats, order = grid axis order).')

    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--fulltraining', action='store_true',
                      help='Train a single model on all grid points.')
    mode.add_argument('--targettraining', action='store_true',
                      help='Train on neighbours of target only (default).')

    dev = p.add_mutually_exclusive_group()
    dev.add_argument('--gpu', action='store_true')
    dev.add_argument('--cpu', action='store_true')

    p.add_argument('--model', choices=['mdn', 'nsf', 'ebm', 'mdn+g', 'eft', 'bdt'], default=None,
                   help='Force model type (default: auto-select by feature detection).')
    p.add_argument('--trainallmodels', action='store_true',
                   help='Train MDN, NSF, EBM, MDN+g and pick winner by χ²/ndf.')
    p.add_argument('--noEFT', action='store_true',
                   help='Skip EFT quadratic morphing overlay (run by default alongside ML model).')
    p.add_argument('--probe-epochs', type=int, default=20,
                   help='Epochs for probe race.')

    p.add_argument('--mhh-range', nargs=3, type=float, metavar=('MIN', 'MAX', 'BIN'),
                   help='mHH histogram range and bin width in GeV.')
    p.add_argument('--mhh-threshold', type=float, default=None,
                   help='Override auto-detected kinematic threshold (y=(mHH-T)/SCALE).')
    p.add_argument('--epochs', type=int, default=None,
                   help='Training epochs (default: 100 laptop / 500 HPC).')
    p.add_argument('--patience-p1', type=int, default=30)
    p.add_argument('--retrain', action='store_true',
                   help='Delete cached model and retrain.')
    p.add_argument('--comment', type=str, default=None,
                   help='Free-text physics description used as soft training prior.')

    # Thermal / performance
    p.add_argument('--amp', action='store_true')
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--gpu-duty-cycle', type=float, default=1.0)
    p.add_argument('--gpu-safety', type=int, choices=range(6), default=0,
                   help='Thermal safety preset 0–5 (0=off).')
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--lambda-smooth', type=float, default=None)

    p.add_argument('--laptop', action='store_true')
    p.add_argument('--verbose', action='store_true')
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_CLS = {'mdn': MDNModel, 'nsf': NSFModel, 'ebm': EBMModel, 'mdn+g': MDNCorrModel}

def _save_model(model, model_type, path):
    _raw = getattr(model, '_orig_mod', model)
    torch.save({
        'model_type':     model_type,
        'state_dict':     model.state_dict(),
        'n_cauchy':       getattr(model, 'n_cauchy', 0),
        'tail_bound':     getattr(_raw, 'tail_bound', 6.0),
        'n_bins_ebm':     getattr(_raw, 'n_bins', 512),
        # NSF analytic_shift fields
        'analytic_shift': getattr(_raw, 'analytic_shift', False),
        'threshold_y':    getattr(_raw, 'threshold_y', 0.0),
        'bw_mass_axis':   getattr(_raw, 'bw_mass_axis', 0),
        # EBM / MDN+g analytic_bw fields
        'analytic_bw':    getattr(_raw, 'analytic_bw', False),
        'bw_wom_axis':    getattr(_raw, 'bw_wom_axis', 1),
        'y_hi':           getattr(_raw, 'y_hi', 6.0),
        'y_lo':           getattr(_raw, 'y_lo', 0.0),
    }, path)


def _load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or 'model_type' not in ckpt:
        # Legacy NSF-only checkpoint
        model = NSFModel().to(device)
        model.load_state_dict(ckpt)
        return model, 'nsf'
    mt = ckpt['model_type']
    if mt == 'mdn':
        model = MDNModel(n_gaussians=ckpt.get('n_gaussians', 14),
                         n_cauchy=ckpt.get('n_cauchy', 0)).to(device)
    elif mt == 'nsf':
        model = NSFModel(tail_bound=ckpt.get('tail_bound', 6.0),
                         analytic_shift=ckpt.get('analytic_shift', False),
                         threshold_y=ckpt.get('threshold_y', 0.0),
                         bw_mass_axis=ckpt.get('bw_mass_axis', 0)).to(device)
    elif mt == 'ebm':
        model = EBMModel(
            n_bins      = ckpt.get('n_bins_ebm', 512),
            analytic_bw = ckpt.get('analytic_bw', False),
            threshold_y = ckpt.get('threshold_y', 0.0),
            bw_mass_axis= ckpt.get('bw_mass_axis', 0),
            bw_wom_axis = ckpt.get('bw_wom_axis', 1),
            y_hi        = ckpt.get('y_hi', 6.0),
            y_lo        = ckpt.get('y_lo', 0.0),
        ).to(device)
    elif mt == 'mdn+g':
        model = MDNCorrModel(
            n_gaussians = ckpt.get('n_gaussians', 14),
            n_cauchy    = ckpt.get('n_cauchy', 0),
            analytic_bw = ckpt.get('analytic_bw', False),
            threshold_y = ckpt.get('threshold_y', 0.0),
            bw_mass_axis= ckpt.get('bw_mass_axis', 0),
            bw_wom_axis = ckpt.get('bw_wom_axis', 1),
            y_hi        = ckpt.get('y_hi', 6.0),
            y_lo        = ckpt.get('y_lo', 0.0),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type '{mt}' in checkpoint {path}")
    model.load_state_dict(ckpt['state_dict'])
    return model, mt


# ─────────────────────────────────────────────────────────────────────────────
# GPU thermal safety presets
# ─────────────────────────────────────────────────────────────────────────────

_SAFETY = {
    1: dict(amp=True, accum_steps=1, gpu_duty_cycle=1.0, thermal_monitor=False,
            desc='Moderate'),
    2: dict(amp=True, accum_steps=2, gpu_duty_cycle=1.0, thermal_monitor=False,
            desc='Balanced'),
    3: dict(amp=True, accum_steps=2, gpu_duty_cycle=0.85, thermal_monitor=False,
            desc='Aggressive'),
    4: dict(amp=True, accum_steps=4, gpu_duty_cycle=0.75, thermal_monitor=False,
            desc='Maximum'),
    5: dict(amp=True, accum_steps=4, gpu_duty_cycle=0.75, thermal_monitor=True,
            thermal_target=88, thermal_cooldown=83,
            desc='Extreme (active governor)'),
}


def _apply_safety_preset(args):
    if args.gpu_safety == 0:
        args.thermal_monitor = False
        args.thermal_target   = 85
        args.thermal_cooldown = 70
        return
    cfg = _SAFETY[args.gpu_safety]
    if '--amp'          not in sys.argv: args.amp            = cfg['amp']
    if '--accum-steps'  not in sys.argv: args.accum_steps    = cfg['accum_steps']
    if '--gpu-duty-cycle' not in sys.argv: args.gpu_duty_cycle = cfg['gpu_duty_cycle']
    args.thermal_monitor  = cfg.get('thermal_monitor', False)
    args.thermal_target   = cfg.get('thermal_target',  85)
    args.thermal_cooldown = cfg.get('thermal_cooldown', 70)
    if args.verbose:
        print(f"[GPU Safety] Level {args.gpu_safety}: {cfg['desc']}")


def _subprocess_worker(mt, gpu_id, db_type, tgt, mhh_lo, mhh_hi, bw,
                       n_ep, full_tr, probe_ep, model_out_path,
                       mhh_thr, verbose_sub, laptop,
                       patience, use_amp, accum_steps, gpu_duty_cycle,
                       thermal_monitor, thermal_target, thermal_cooldown, lambda_smooth,
                       precomputed_features=None):
    """Subprocess entry: re-init cfg, build dataset, train, save."""
    import sys; sys.path.insert(0, '')
    # spawn gives a non-TTY child: Python defaults to block-buffered stdout.
    # Switch to line-buffering so every print() reaches the SLURM log immediately.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    import torch as _torch
    _dev = _torch.device(f'cuda:{gpu_id}')
    from core.config import init_globals as _ig
    _ig(db_type, _dev, mhh_threshold=mhh_thr,
        laptop_mode=laptop, verbose=False)
    from core.dataset import get_files_for_target, LheDataset, compute_voronoi_weights
    import core.config as _cfg
    if full_tr:
        _src = list(_cfg.GRID_FILES.items())
        _vw  = compute_voronoi_weights([c for c, _ in _src], verbose=False)
        _ds  = LheDataset(_src, voronoi_weights=_vw, verbose=False)
    else:
        _src = get_files_for_target(tgt, verbose=False)
        _ds  = LheDataset(_src, verbose=False)
    from core.training import select_and_train as _sat
    m, _ = _sat(
        dataset=_ds, source_data=_src, target=tgt,
        mhh_min=mhh_lo, mhh_max=mhh_hi, bin_width=bw,
        epochs=n_ep, full_training=full_tr,
        probe_epochs=probe_ep, force_model=mt,
        train_all_models=False,
        verbose=verbose_sub,
        patience=patience,
        use_amp=use_amp,
        accum_steps=accum_steps,
        gpu_duty_cycle=gpu_duty_cycle,
        thermal_monitor=thermal_monitor,
        thermal_target=thermal_target,
        thermal_cooldown=thermal_cooldown,
        lambda_smooth=lambda_smooth,
        precomputed_features=precomputed_features,
    )
    import torch as _t2
    _raw = getattr(m, '_orig_mod', m)
    _mdn_part = getattr(_raw, '_mdn', _raw)   # MDNCorrModel wraps MDN as ._mdn
    _t2.save({
        'model_type':     mt,
        'state_dict':     m.state_dict(),
        'n_gaussians':    getattr(_mdn_part, 'n_gaussians', 14),
        'n_cauchy':       getattr(m, 'n_cauchy', getattr(_mdn_part, 'n_cauchy', 0)),
        'tail_bound':     getattr(_raw, 'tail_bound', 6.0),
        'n_bins_ebm':     getattr(_raw, 'n_bins', 512),
        'analytic_shift': getattr(_raw, 'analytic_shift', False),
        'threshold_y':    getattr(_raw, 'threshold_y', 0.0),
        'bw_mass_axis':   getattr(_raw, 'bw_mass_axis', 0),
    }, model_out_path)


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    if _ARGCOMPLETE:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    _apply_safety_preset(args)

    # ── Device selection ────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device('cpu')
    elif args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('--gpu: CUDA not available')
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Globals ─────────────────────────────────────────────────────────────
    #   init_globals sets DATABASE_TYPE, DEVICE, CACHE_DIR, MODELS_DIR,
    #   MHH_THRESHOLD, PARAM_NAMES, PARAM_SPACINGS, GRID_FILES inside cfg.
    init_globals(
        database_type = args.database_type,
        device        = device,
        mhh_threshold = args.mhh_threshold,
        laptop_mode   = LAPTOP_MODE,
        verbose       = args.verbose,
    )

    target       = tuple(args.params)
    verbose      = args.verbose
    full_training = args.fulltraining or (not args.targettraining and not args.fulltraining and False)
    # default to target-neighbour mode
    if not args.fulltraining and not args.targettraining:
        full_training = False

    # ── mHH range ───────────────────────────────────────────────────────────
    mhh_min, mhh_max, bin_width = 0.0, 5000.0, 50.0
    if args.mhh_range:
        mhh_min, mhh_max, bin_width = args.mhh_range

    n_epochs = args.epochs if args.epochs else (100 if LAPTOP_MODE else 500)

    if verbose:
        print(f"Database : {cfg.DATABASE_TYPE}")
        print(f"Device   : {device}")
        print(f"Target   : {_target_str(target)}")
        print(f"Mode     : {'Full-grid' if full_training else 'Target-neighbour'}")
        print(f"mHH      : [{mhh_min}, {mhh_max}] bin={bin_width} GeV")
        print(f"Threshold: {cfg.MHH_THRESHOLD:.2f} GeV")
        print()

    # ── EFT baseline (no training) ──────────────────────────────────────────
    if args.model == 'eft':
        if verbose:
            print('[EFT] Running quadratic morphing baseline...')
        pdf, centers, edges = eft_morphing(
            target, mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width, verbose=verbose)
        plots_dir = os.path.join(cfg.DATABASE_TYPE, 'Plots')
        os.makedirs(plots_dir, exist_ok=True)
        from core.plotting import plot_eft_result
        plot_eft_result(pdf, centers, edges, target,
                        plots_dir=plots_dir, model_type='eft')
        print('Done.')
        return

    # ── BDT baseline (no training) ──────────────────────────────────────────
    if args.model == 'bdt':
        if verbose:
            print('[BDT] Running gradient-boosted tree morphing...')
        pdf, centers, edges = bdt_morphing(
            target, mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width, verbose=verbose)
        plots_dir = os.path.join(cfg.DATABASE_TYPE, 'Plots')
        os.makedirs(plots_dir, exist_ok=True)
        from core.plotting import plot_eft_result
        plot_eft_result(pdf, centers, edges, target,
                        plots_dir=plots_dir, model_type='bdt')
        print('Done.')
        return

    # ── Model file path ─────────────────────────────────────────────────────
    if full_training:
        model_file = os.path.join(cfg.MODELS_DIR, 'model_full.pt')
        mode_suffix = 'full'
    else:
        mode_suffix = _mode_suffix(target)
        model_file  = os.path.join(cfg.MODELS_DIR, f'model_{mode_suffix}.pt')

    if args.retrain and os.path.exists(model_file):
        os.remove(model_file)
        if verbose:
            print(f'--retrain: deleted {model_file}')

    # ── Load or train ────────────────────────────────────────────────────────
    model = model_type = None

    if os.path.exists(model_file):
        if verbose:
            print(f'Loading existing model: {model_file}')
        try:
            model, model_type = _load_model(model_file, device)
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                if verbose:
                    print('Architecture mismatch — retraining.')
                os.remove(model_file)
            else:
                raise

    if model is None:
        # Source data for training
        from core.dataset import get_files_for_target
        if full_training:
            source_data      = list(cfg.GRID_FILES.items())
            voronoi_weights  = compute_voronoi_weights([c for c,_ in source_data], verbose=verbose)
            dataset          = LheDataset(source_data, voronoi_weights=voronoi_weights, verbose=verbose)
        else:
            source_data     = get_files_for_target(target, verbose=verbose)
            voronoi_weights = compute_voronoi_weights([c for c, _ in source_data], verbose=verbose)
            dataset         = LheDataset(source_data, voronoi_weights=voronoi_weights, verbose=verbose)

        if verbose:
            print(f'Dataset: {len(dataset):,} events over {len(source_data)} grid points')

        # Parse physics comment prior
        comment_prior = None
        if args.comment:
            from core.features import parse_comment
            comment_prior = parse_comment(args.comment, target, verbose=verbose)

        # ── EFT early (before ML training so result is visible immediately) ─
        plots_dir = os.path.join(cfg.DATABASE_TYPE, 'Plots')
        os.makedirs(plots_dir, exist_ok=True)
        if not args.noEFT:
            try:
                if verbose:
                    print('[EFT] Running quadratic morphing baseline...')
                eft_pdf, eft_centers, eft_edges = eft_morphing(
                    target, mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width, verbose=False)
                from core.plotting import plot_eft_result
                plot_eft_result(eft_pdf, eft_centers, eft_edges, target,
                                plots_dir=plots_dir, model_type='eft')
                if verbose:
                    print('[EFT] Done.')
            except Exception as e:
                if verbose:
                    print(f'[EFT] Skipped: {e}')

        _train_kw = dict(
            dataset        = dataset,
            source_data    = source_data,
            target         = target,
            mhh_min        = mhh_min,
            mhh_max        = mhh_max,
            bin_width      = bin_width,
            epochs         = n_epochs,
            full_training  = full_training,
            comment_parsed = comment_prior,
            probe_epochs   = args.probe_epochs,
            verbose        = verbose,
            num_workers    = args.num_workers,
            use_amp        = args.amp,
            accum_steps    = args.accum_steps,
            gpu_duty_cycle = args.gpu_duty_cycle,
            thermal_monitor= args.thermal_monitor,
            thermal_target = args.thermal_target,
            thermal_cooldown=args.thermal_cooldown,
            patience       = args.patience_p1,
            lambda_smooth  = args.lambda_smooth,
        )

        # ── Pre-compute features once (shared across all models) ────────────
        from core.features import auto_nsf_hyperparams as _ansh, reclassify_paired_tracks as _rpt
        _pre_num_bins, _pre_tail_bound, _pre_analyses, _pre_tracks = _ansh(
            source_data, mhh_min, mhh_max, verbose=verbose)
        _rpt(_pre_tracks, _pre_analyses)
        _precomputed_features = dict(
            num_bins=_pre_num_bins, tail_bound=_pre_tail_bound,
            analyses=_pre_analyses, tracks=_pre_tracks,
        )
        _train_kw['precomputed_features'] = _precomputed_features

        # Print feature summary once here (subprocesses will not repeat this)
        if verbose:
            _names = cfg.PARAM_NAMES or [f'p{i+1}' for i in range(len(source_data[0][0]))]
            print('\n[FEATURES] Grid points analysed: ' + str(len(_pre_analyses)))
            for _a in _pre_analyses:
                _cs = '  '.join(f'{_names[i]}={_a["coords"][i]:.6g}' for i in range(len(_a["coords"])))
                print(f'  [{_cs}]  N={_a["n_events"]}  features={len(_a["features"])}')
                for _f in _a['features']:
                    print(f'    {_f["feature_type"]:20s}  pos={_f["mhh_position"]:.1f} GeV  '
                          f'width={_f["width_gev"]:.1f} GeV  sign={"+1" if _f["sign"]>0 else "-1"}  '
                          f'sig={_f["significance"]:.1f}σ')
            print(f'[FEATURES] Tracks: {len(_pre_tracks)}')
            from core.config import inverse_transform_mhh as _itm
            for _i, _t in enumerate(_pre_tracks):
                _pvs = [float(_itm(_f["y_pos"], cfg.MHH_THRESHOLD)) for _f in _t["features"]]
                print(f'  Track {_i}: sign={"+1" if _t["sign"]>0 else "-1"}  '
                      f'n_pts={len(_t["features"])}  '
                      f'pos range=[{min(_pvs):.1f}, {max(_pvs):.1f}] GeV'
                      + ('  [paired_interference]' if _t.get('paired_interference') else ''))
            print()

        # ── trainallmodels: train each model independently ───────────────────
        if args.trainallmodels:
            n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
            all_mts = ('mdn', 'ebm', 'mdn+g', 'nsf')

            # ── BDT: runs synchronously (CPU, no training loop) ──────────────
            bdt_pdfs = {}   # mt → (pdf, centers, edges)  for non-neural methods
            try:
                if verbose:
                    print('[BDT] Running gradient-boosted tree morphing...')
                _bdt_pdf, _bdt_cen, _bdt_edg = bdt_morphing(
                    target, mhh_min=mhh_min, mhh_max=mhh_max,
                    bin_width=bin_width, verbose=verbose)
                bdt_pdfs['bdt'] = (_bdt_pdf, _bdt_cen, _bdt_edg)
                from core.plotting import plot_eft_result as _peft
                _peft(_bdt_pdf, _bdt_cen, _bdt_edg, target,
                      plots_dir=plots_dir, model_type='bdt')
                np.savez(
                    os.path.join(cfg.MODELS_DIR, f'mhh_samples_{mode_suffix}_bdt.npz'),
                    events=np.repeat(_bdt_cen, np.maximum(
                        (_bdt_pdf * 500_000 * bin_width).astype(int), 0)),
                )
                if verbose:
                    print('[BDT] Done.')
            except Exception as _bdt_e:
                print(f'[BDT] Skipped: {_bdt_e}')

            # Determine which neural models still need training
            to_train = []
            loaded   = {}
            for mt in all_mts:
                mt_file = os.path.join(cfg.MODELS_DIR, f'model_{mode_suffix}_{mt}.pt')
                if args.retrain and os.path.exists(mt_file):
                    os.remove(mt_file)
                if os.path.exists(mt_file):
                    if verbose:
                        print(f'[{mt.upper()}] Loading existing: {mt_file}')
                    try:
                        loaded[mt] = _load_model(mt_file, device)[0]
                    except RuntimeError as e:
                        if 'size mismatch' in str(e):
                            os.remove(mt_file)
                            to_train.append(mt)
                        else:
                            raise
                else:
                    to_train.append(mt)

            def _train_one_sequential(mt, dev_i):
                """Train one model on dev_i (called in main process, device patched)."""
                orig_dev = cfg.DEVICE
                cfg.DEVICE = dev_i
                try:
                    m, _ = select_and_train(force_model=mt, train_all_models=False,
                                            **_train_kw)
                finally:
                    cfg.DEVICE = orig_dev
                return m

            if n_gpus >= 2 and len(to_train) > 1:
                import multiprocessing as _mp
                _mp_ctx = _mp.get_context('spawn')
                import importlib



                if verbose:
                    print(f'[TRAINALL] {len(to_train)} models on {n_gpus} GPUs '
                          f'(max {min(len(to_train), n_gpus)} parallel)')

                _common = dict(
                    db_type=cfg.DATABASE_TYPE, tgt=target,
                    mhh_lo=mhh_min, mhh_hi=mhh_max, bw=bin_width,
                    n_ep=n_epochs, full_tr=full_training,
                    probe_ep=args.probe_epochs,
                    mhh_thr=cfg.MHH_THRESHOLD,
                    verbose_sub=verbose, laptop=LAPTOP_MODE,
                    patience=args.patience_p1,
                    use_amp=args.amp,
                    accum_steps=args.accum_steps,
                    gpu_duty_cycle=args.gpu_duty_cycle,
                    thermal_monitor=args.thermal_monitor,
                    thermal_target=args.thermal_target,
                    thermal_cooldown=args.thermal_cooldown,
                    lambda_smooth=args.lambda_smooth,
                    precomputed_features=_precomputed_features,
                )

                # Launch up to n_gpus processes at a time
                running = {}  # gpu_id → (Process, mt, out_path)
                queue   = list(enumerate(to_train))  # [(idx, mt), ...]
                gpu_free = list(range(n_gpus))

                while queue or running:
                    # Fill free GPUs
                    while queue and gpu_free:
                        idx, mt = queue.pop(0)
                        gid = gpu_free.pop(0)
                        out_path = os.path.join(cfg.MODELS_DIR, f'model_{mode_suffix}_{mt}.pt')
                        p = _mp_ctx.Process(
                            target=_subprocess_worker,
                            args=(mt, gid, *[_common[k] for k in
                                  ('db_type','tgt','mhh_lo','mhh_hi','bw','n_ep','full_tr',
                                   'probe_ep')],
                                  out_path,
                                  _common['mhh_thr'], _common['verbose_sub'], _common['laptop'],
                                  _common['patience'], _common['use_amp'],
                                  _common['accum_steps'], _common['gpu_duty_cycle'],
                                  _common['thermal_monitor'], _common['thermal_target'],
                                  _common['thermal_cooldown'], _common['lambda_smooth'],
                                  _common['precomputed_features']),
                            daemon=True,
                        )
                        p.start()
                        running[gid] = (p, mt, out_path)
                        if verbose:
                            print(f'[TRAINALL] GPU {gid}: {mt.upper()} started (pid={p.pid})')

                    # Wait for any to finish
                    import time as _time
                    finished_gpus = []
                    while not finished_gpus:
                        for gid, (p, mt, out_path) in list(running.items()):
                            if not p.is_alive():
                                p.join()
                                if p.exitcode != 0:
                                    print(f'[TRAINALL] WARNING: {mt.upper()} on GPU {gid} '
                                          f'exited with code {p.exitcode}', flush=True)
                                else:
                                    if verbose:
                                        print(f'[TRAINALL] GPU {gid}: {mt.upper()} finished',
                                              flush=True)
                                finished_gpus.append(gid)
                        if not finished_gpus:
                            _time.sleep(2)
                    for gid in finished_gpus:
                        _, mt_done, out_path = running.pop(gid)
                        gpu_free.append(gid)
                        if os.path.exists(out_path):
                            loaded[mt_done] = _load_model(out_path, device)[0]

            else:
                for mt in to_train:
                    if verbose:
                        print(f'\n[TRAINALL] Training {mt.upper()} ...')
                    gid = 0 if n_gpus > 0 else None
                    dev_i = torch.device(f'cuda:{gid}') if gid is not None else torch.device('cpu')
                    m = _train_one_sequential(mt, dev_i)
                    loaded[mt] = m
                    mt_file = os.path.join(cfg.MODELS_DIR, f'model_{mode_suffix}_{mt}.pt')
                    _save_model(m, mt, mt_file)
                    if verbose:
                        print(f'[{mt.upper()}] Saved: {mt_file}')

            for mt in all_mts:
                mt_model = loaded.get(mt)
                if mt_model is None:
                    continue
                mt_samples = sample_model(mt_model, mt, target, n_samples=500_000)
                mt_samples = mt_samples[(mt_samples >= mhh_min) & (mt_samples <= mhh_max)]
                edges_t   = np.arange(mhh_min, mhh_max + bin_width, bin_width)
                centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
                counts_t, _ = np.histogram(mt_samples, bins=edges_t)
                norm_t = np.sum(counts_t) * (edges_t[1] - edges_t[0])
                pdf_t  = counts_t.astype(float) / norm_t if norm_t > 0 else counts_t.astype(float)

                plot_target_only(pdf=pdf_t, centers=centers_t, edges=edges_t,
                                 target=target, model_type=mt, plots_dir=plots_dir)
                plot_with_neighbours(pdf=pdf_t, centers=centers_t, edges=edges_t,
                                     target=target, model_type=mt, plots_dir=plots_dir,
                                     mhh_min=mhh_min, mhh_max=mhh_max)
                np.savez(
                    os.path.join(cfg.MODELS_DIR, f'mhh_samples_{mode_suffix}_{mt}.npz'),
                    events=mt_samples,
                )

            print('Done.')
            return

        model, model_type = select_and_train(force_model=args.model, train_all_models=False,
                                             **_train_kw)
        _save_model(model, model_type, model_file)
        if verbose:
            print(f'Model saved: {model_file}')

    # ── Sampling ─────────────────────────────────────────────────────────────
    if verbose:
        print('Sampling 500 000 events...')

    # On-grid passthrough: if target is exactly a grid point, load raw data.
    TOL = cfg.TOL
    target_vals = tuple(float(v) for v in target)
    n_params    = len(target_vals)
    _grid_key   = next(
        (c for c in cfg.GRID_FILES
         if all(abs(c[i] - target_vals[i]) < TOL for i in range(n_params))),
        None,
    )
    if _grid_key is not None:
        if verbose:
            print('Target on grid — using raw data (exact reproduction).')
        from core.config import inverse_transform_mhh
        raw = []
        for f in cfg.GRID_FILES[_grid_key]:
            cache = preprocess_file(f, _grid_key)
            t = torch.load(cache, weights_only=True)
            if t.shape[0] > 0:
                raw.extend(inverse_transform_mhh(t[:, -1].numpy(), cfg.MHH_THRESHOLD).tolist())
        samples = np.array(raw)
    else:
        samples = sample_model(model, model_type, target, n_samples=500_000)

    samples = samples[(samples >= mhh_min) & (samples <= mhh_max)]

    # ── Histogram ────────────────────────────────────────────────────────────
    edges   = np.arange(mhh_min, mhh_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(samples, bins=edges)
    norm = np.sum(counts) * (edges[1] - edges[0])
    pdf  = counts.astype(float) / norm if norm > 0 else counts.astype(float)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(cfg.DATABASE_TYPE, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_target_only(
        pdf=pdf, centers=centers, edges=edges,
        target=target, model_type=model_type,
        plots_dir=plots_dir,
    )
    plot_with_neighbours(
        pdf=pdf, centers=centers, edges=edges,
        target=target, model_type=model_type,
        plots_dir=plots_dir,
        mhh_min=mhh_min, mhh_max=mhh_max,
    )

    # Save raw samples
    np.savez(
        os.path.join(cfg.MODELS_DIR, f'mhh_samples_{mode_suffix}.npz'),
        events=samples,
    )

    print('Done.')


if __name__ == '__main__':
    main()
