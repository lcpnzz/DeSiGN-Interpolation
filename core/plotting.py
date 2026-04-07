"""core/plotting.py — All matplotlib output for morph_ml.py.

Public API:
    get_plot_neighbours()     — find bracketing grid points for reference curves
    plot_target_only()        — single plot: model PDF for target
    plot_with_neighbours()    — model PDF + raw neighbour reference curves
    plot_partial_distribution() — mid-training snapshot (called from training loop)
    plot_eft_result()         — EFT-only plot (no model, pure morphing result)
"""

import os
import itertools
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import core.config as cfg
from core.config import (
    TOL, PARAM_NAMES,
    inverse_transform_mhh,
    _target_str, _target_tag, _mode_suffix,
)
from core.dataset import preprocess_file


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _symlog_axes(ax):
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_ylim(1e-9, 0.5)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which='both', alpha=0.3)


def _load_neighbour_pdf(coords, edges, mhh_min, mhh_max):
    """Load raw mHH data for one grid point, histogram, normalise."""
    all_mhh = []
    for f in cfg.GRID_FILES[coords]:
        cache  = preprocess_file(f, coords)
        tensor = torch.load(cache, weights_only=True)
        if tensor.shape[0] > 0:
            all_mhh.extend(
                inverse_transform_mhh(tensor[:, -1].numpy(), cfg.MHH_THRESHOLD).tolist())
    arr = np.asarray(all_mhh)
    arr = arr[(arr >= mhh_min) & (arr <= mhh_max)]
    counts, _ = np.histogram(arr, bins=edges)
    norm = np.sum(counts) * (edges[1] - edges[0])
    return counts.astype(float) / norm if norm > 0 else counts.astype(float)


def _coords_label(coords):
    names = cfg.PARAM_NAMES or [f'p{i+1}' for i in range(len(coords))]
    return ', '.join(f'{names[i]}={coords[i]}' for i in range(len(coords)))


# ─────────────────────────────────────────────────────────────────────────────
# get_plot_neighbours
# ─────────────────────────────────────────────────────────────────────────────

def get_plot_neighbours(target):
    """Return bracketing grid points for plot reference.

    For each axis:
      - ON-grid: fix to the target value (no off-grid variation shown on that axis)
      - OFF-grid: show the two bracketing grid values

    This means for target (1190, 0.001) where 0.001 is on-grid for WoMS1:
      axis 0 (mS1, off-grid): [800, 1200]
      axis 1 (WoMS1, on-grid): [0.001]
      → neighbours: (800, 0.001) and (1200, 0.001) only.
    """
    target_vals = [float(v) for v in target]
    n_params    = len(target_vals)

    def _bracket(vals, t):
        below = [v for v in vals if v < t - TOL]
        above = [v for v in vals if v > t + TOL]
        sel = []
        if below: sel.append(below[-1])
        if above: sel.append(above[0])
        return sel

    all_axis = [sorted(set(c[i] for c in cfg.GRID_FILES)) for i in range(n_params)]
    on_grid  = [any(abs(v - target_vals[i]) < TOL for v in all_axis[i]) for i in range(n_params)]

    axis_cands = []
    for i in range(n_params):
        if on_grid[i]:
            axis_cands.append([target_vals[i]])          # fixed — no variation on this axis
        else:
            axis_cands.append(_bracket(all_axis[i], target_vals[i]))

    result = []
    for combo in itertools.product(*axis_cands):
        if all(abs(combo[i] - target_vals[i]) < TOL for i in range(n_params)):
            continue  # exclude exact target
        if combo in cfg.GRID_FILES:
            result.append((combo, cfg.GRID_FILES[combo]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# plot_target_only
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_only(pdf, centers, edges, target, model_type, plots_dir):
    """Single plot: normalised mHH PDF predicted by the model for the target point."""
    param_str = _target_str(target)
    title     = f'{cfg.DATABASE_TYPE}  –  {param_str}  [{model_type.upper()}]'
    tag       = _mode_suffix(target)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(centers, pdf, lw=2, color='black', drawstyle='steps-mid',
            label=param_str)
    ax.set_xlabel(r'$m_{HH}$ [GeV]')
    ax.set_ylabel('PDF')
    ax.set_title(title)
    _symlog_axes(ax)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plots_dir, f'{tag}_{model_type}.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# plot_with_neighbours
# ─────────────────────────────────────────────────────────────────────────────

def plot_with_neighbours(pdf, centers, edges, target, model_type, plots_dir,
                         mhh_min=0.0, mhh_max=5000.0):
    """Model PDF overlaid on raw bracketing grid-point histograms."""
    param_str = _target_str(target)
    title     = f'{cfg.DATABASE_TYPE}  –  {param_str}  [{model_type.upper()} + neighbours]'
    tag       = _mode_suffix(target)

    fig, ax = plt.subplots(figsize=(7, 6))

    for coords, _ in get_plot_neighbours(target):
        pdf_nb = _load_neighbour_pdf(coords, edges, mhh_min, mhh_max)
        ax.plot(centers, pdf_nb, lw=1.2, linestyle='--', drawstyle='steps-mid',
                label=_coords_label(coords))

    ax.plot(centers, pdf, lw=2.2, color='black', drawstyle='steps-mid',
            label=param_str)

    ax.set_xlabel(r'$m_{HH}$ [GeV]')
    ax.set_ylabel('PDF')
    ax.set_title(title)
    _symlog_axes(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(plots_dir, f'{tag}_{model_type}_neighbours.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# plot_partial_distribution  (mid-training snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_partial_distribution(
    model, target, epoch,
    n_samples=500_000, mhh_min=300.0, mhh_max=1500.0, bin_width=10.0,
    verbose=False, model_type='nsf', full_training=False,
):
    from core.training import sample_model

    # On-grid passthrough
    target_vals = tuple(float(v) for v in target)
    n_p = len(target_vals)
    _gk = next(
        (c for c in cfg.GRID_FILES
         if all(abs(c[i] - target_vals[i]) < TOL for i in range(n_p))),
        None,
    )
    if _gk is not None:
        raw = []
        for f in cfg.GRID_FILES[_gk]:
            cache = preprocess_file(f, _gk)
            t = torch.load(cache, weights_only=True)
            if t.shape[0] > 0:
                raw.extend(inverse_transform_mhh(t[:, -1].numpy(), cfg.MHH_THRESHOLD).tolist())
        samples = np.asarray(raw)
    else:
        samples = sample_model(model, model_type, target, n_samples=n_samples)

    samples = samples[(samples >= mhh_min) & (samples <= mhh_max)]
    edges   = np.arange(mhh_min, mhh_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(samples, bins=edges)
    norm = np.sum(counts) * (edges[1] - edges[0])
    pdf  = counts.astype(float) / norm if norm > 0 else counts.astype(float)

    param_str = _target_str(target)
    title     = f'{cfg.DATABASE_TYPE}  –  {param_str}  [{model_type.upper()} epoch {epoch}]'

    plots_dir = os.path.join(cfg.DATABASE_TYPE, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for coords, _ in get_plot_neighbours(target):
        pdf_nb = _load_neighbour_pdf(coords, edges, mhh_min, mhh_max)
        ax.plot(centers, pdf_nb, lw=1.2, linestyle='--', drawstyle='steps-mid',
                label=_coords_label(coords))
    ax.plot(centers, pdf, lw=2, color='black', drawstyle='steps-mid',
            label=param_str)
    ax.set_xlabel(r'$m_{HH}$ [GeV]')
    ax.set_ylabel('PDF')
    ax.set_title(title)
    _symlog_axes(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()

    if full_training:
        fname = os.path.join(plots_dir, f'{model_type}_partial_training.pdf')
    else:
        fname = os.path.join(plots_dir, f'{_mode_suffix(target)}_{model_type}_partial_training.pdf')
    fig.savefig(fname)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# plot_eft_result
# ─────────────────────────────────────────────────────────────────────────────

def plot_eft_result(pdf, centers, edges, target, plots_dir, model_type='eft',
                   chi2_ndf=None):
    """Plot the result of EFT quadratic morphing, optionally with chi2/ndf annotation."""
    param_str = _target_str(target)
    method_str = {'eft': 'EFT quadratic morphing', 'bdt': 'BDT morphing'}.get(
        model_type, f'{model_type.upper()} morphing')
    title = f'{cfg.DATABASE_TYPE}  –  {param_str}  [{method_str}]'
    tag   = _mode_suffix(target)

    mhh_min = float(edges[0])
    mhh_max = float(edges[-1])

    fig, ax = plt.subplots(figsize=(7, 6))

    for coords, _ in get_plot_neighbours(target):
        pdf_nb = _load_neighbour_pdf(coords, edges, mhh_min, mhh_max)
        ax.plot(centers, pdf_nb, lw=1.2, linestyle='--', drawstyle='steps-mid',
                label=_coords_label(coords))

    ax.plot(centers, pdf, lw=2.2, color='black', drawstyle='steps-mid',
            label=param_str)

    if chi2_ndf is not None:
        ax.text(0.97, 0.97, f'χ²/ndf = {chi2_ndf:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    ax.set_xlabel(r'$m_{HH}$ [GeV]')
    ax.set_ylabel('PDF')
    ax.set_title(title)
    _symlog_axes(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(plots_dir, f'{tag}_{model_type}.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path
