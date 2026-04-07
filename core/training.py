"""core/training.py — Training loops, model selection, chi2 monitoring, sampling.

Public API:
    select_and_train()       — unified routing + training for all model types
    monitor_grid_reproduction() — adaptive-bin chi2 between data and model PDF
    sample_model()           — draw mHH samples from any trained model
"""

import os
import math
import time
import concurrent.futures
import numpy as np
import torch
import torch.optim as optim

import core.config as cfg
from core.config import (
    SCALE, TOL,
    encode_ctx, transform_mhh, inverse_transform_mhh,
    PARAM_NAMES,
)
from core.dataset import preprocess_file
from core.models import (
    NSFModel, MDNModel, EBMModel, MDNCorrModel,
    nsf_nll_loss, mdn_loss, ebm_nll_loss, mdn_corr_loss,
    _student_t_log_cdf, _mdn_analytic_bw_kwargs,
)
from core.features import (
    extract_features_at_coords, build_feature_tracks, reclassify_paired_tracks,
    auto_nsf_hyperparams, build_physics_prior_from_analyses, build_track_prior,
    generate_prior_samples, parse_comment, build_ratio_tracks,
)

# Optional thermal monitoring — vendor-agnostic (NVIDIA / AMD / Intel)
import subprocess
import json as _json

try:
    import pynvml as _pynvml
    _PYNVML = True
except ImportError:
    _PYNVML = False

# Cached detection result
_GPU_VENDOR      = None   # 'nvidia' | 'amd' | 'intel' | None
_GPU_TEMP_METHOD = None   # 'pynvml' | 'nvidia-smi' | 'rocm-smi' | 'xpu-smi' | None


def _detect_gpu_vendor():
    """Detect GPU vendor and best temperature-reading method (cached)."""
    global _GPU_VENDOR, _GPU_TEMP_METHOD
    if _GPU_VENDOR is not None:
        return _GPU_VENDOR, _GPU_TEMP_METHOD

    # NVIDIA via pynvml
    if _PYNVML:
        try:
            _pynvml.nvmlInit()
            if _pynvml.nvmlDeviceGetCount() > 0:
                _GPU_VENDOR, _GPU_TEMP_METHOD = 'nvidia', 'pynvml'
                return _GPU_VENDOR, _GPU_TEMP_METHOD
        except Exception:
            pass

    # NVIDIA via nvidia-smi
    try:
        r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            _GPU_VENDOR, _GPU_TEMP_METHOD = 'nvidia', 'nvidia-smi'
            return _GPU_VENDOR, _GPU_TEMP_METHOD
    except Exception:
        pass

    # AMD via rocm-smi
    try:
        r = subprocess.run(['rocm-smi', '--showtemp'], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            _GPU_VENDOR, _GPU_TEMP_METHOD = 'amd', 'rocm-smi'
            return _GPU_VENDOR, _GPU_TEMP_METHOD
    except Exception:
        pass

    # Intel via xpu-smi
    try:
        r = subprocess.run(['xpu-smi', 'discovery'], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            _GPU_VENDOR, _GPU_TEMP_METHOD = 'intel', 'xpu-smi'
            return _GPU_VENDOR, _GPU_TEMP_METHOD
    except Exception:
        pass

    # CUDA present but no tool found — assume NVIDIA, try nvidia-smi anyway
    import torch as _torch
    if _torch.cuda.is_available():
        _GPU_VENDOR, _GPU_TEMP_METHOD = 'nvidia', 'nvidia-smi'
        return _GPU_VENDOR, _GPU_TEMP_METHOD

    _GPU_VENDOR, _GPU_TEMP_METHOD = None, None
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Thermal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_temp(device_id=0):
    """Return GPU temperature in °C, or None if unavailable. Vendor-agnostic."""
    import re as _re
    vendor, method = _detect_gpu_vendor()
    if vendor is None:
        return None

    # NVIDIA — pynvml
    if vendor == 'nvidia' and method == 'pynvml':
        try:
            h = _pynvml.nvmlDeviceGetHandleByIndex(device_id)
            return _pynvml.nvmlDeviceGetTemperature(h, _pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass   # fall through to nvidia-smi

    # NVIDIA — nvidia-smi
    if vendor == 'nvidia':
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu',
                 '--format=csv,noheader,nounits', f'--id={device_id}'],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except Exception:
            pass

    # AMD — rocm-smi JSON
    if vendor == 'amd':
        try:
            r = subprocess.run(['rocm-smi', '--showtemp', '--json'],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                data = _json.loads(r.stdout)
                key  = f'card{device_id}'
                if key in data:
                    val = data[key].get('Temperature (Sensor edge) (C)', '')
                    if val:
                        return float(val)
        except Exception:
            pass
        # AMD — rocm-smi plain text fallback
        try:
            r = subprocess.run(['rocm-smi', '--showtemp'],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                m = _re.search(rf'GPU\[{device_id}\].*?Temperature.*?(\d+(?:\.\d+)?)\s*C',
                               r.stdout)
                if m:
                    return float(m.group(1))
        except Exception:
            pass

    # Intel — xpu-smi (metric 18 = temperature)
    if vendor == 'intel':
        try:
            r = subprocess.run(['xpu-smi', 'dump', '-d', str(device_id), '-m', '18'],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                m = _re.search(r'Temperature.*?(\d+)', r.stdout)
                if m:
                    return int(m.group(1))
        except Exception:
            pass

    return None


def _thermal_governor(target=85, cooldown=70, interval=5.0, verbose=False):
    if cfg.DEVICE.type not in ('cuda', 'xpu'):
        return
    temp = _gpu_temp()
    if temp is None:
        return
    if temp >= target:
        if verbose:
            print(f'\r[THERMAL] {temp}°C ≥ {target}°C — pausing...', end='', flush=True)
        while True:
            time.sleep(interval)
            temp = _gpu_temp()
            if temp is None or temp <= cooldown:
                break
        if verbose:
            print('\r' + ' '*60 + '\r', end='', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: train/val split, NLL evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _make_val_split(dataset, val_frac=0.1):
    n     = len(dataset)
    idx   = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def _fmt_epoch(model_type, epoch, epochs, nll, val_nll, chi2, elapsed, best=False):
    """Return a fixed-width epoch log line, aligned across all model types."""
    tag   = f'[{model_type.upper():<6}]'
    ep_s  = f'Ep {epoch:>4}/{epochs:<4}'
    nll_s = f'NLL {nll:>11.6f}'
    val_s = f'val {val_nll:>11.6f}'
    chi_s = f'chi2 {chi2:>10.3f}'
    t_s   = f'{elapsed:>5.1f}s'
    b_s   = '  *** BEST ***' if best else ''
    return f'{tag} {ep_s}  {nll_s}  {val_s}  {chi_s}  {t_s}{b_s}'


def _eval_nll_nsf(model, feats, tgts, wgts, device=None, batch=8192):
    dev   = device or cfg.DEVICE
    eager = getattr(model, '_orig_mod', model)
    eager.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            xb = feats[i:i+batch].to(dev)
            yb = tgts[i:i+batch].to(dev)
            wb = wgts[i:i+batch].to(dev)
            total += nsf_nll_loss(eager, yb, xb, wb).item() * xb.size(0)
            n     += xb.size(0)
    return total / max(n, 1)


def _eval_nll_mdn(model, feats, tgts, wgts, device=None, batch=8192):
    dev = device or cfg.DEVICE
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            xb = feats[i:i+batch].to(dev)
            yb = tgts[i:i+batch].to(dev)
            wb = wgts[i:i+batch].to(dev)
            pi, mu, sigma, alpha, nu = model(xb)
            y_exp = yb.expand_as(mu)
            z     = (y_exp - mu) / sigma
            log_t = (torch.lgamma(0.5*(nu+1)) - torch.lgamma(0.5*nu)
                     - 0.5*torch.log(math.pi*nu)
                     - 0.5*(nu+1)*torch.log1p(z.pow(2)/(nu+1e-8))
                     - torch.log(sigma))
            w_cdf = alpha * z * torch.sqrt((nu+1.)/(nu+z.pow(2)+1e-8))
            lcdf  = _student_t_log_cdf(w_cdf, nu+1.)
            log_p = torch.logsumexp(torch.log(pi+1e-12)+math.log(2)+log_t+lcdf, dim=1)
            ws    = wb / (wb.mean() + 1e-12)
            total += -(ws * log_p).mean().item() * xb.size(0)
            n     += xb.size(0)
    return total / max(n, 1)


def _eval_nll_generic(model, model_type, feats, tgts, wgts, device=None, batch=8192):
    if model_type == 'mdn':
        return _eval_nll_mdn(model, feats, tgts, wgts, device=device, batch=batch)
    dev   = device or cfg.DEVICE
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            xb = feats[i:i+batch].to(dev)
            yb = tgts[i:i+batch].to(dev)
            wb = wgts[i:i+batch].to(dev)
            lp = model.log_prob(yb, xb)
            w_norm = wb / (wb.mean() + 1e-12)
            total += -(w_norm * lp).mean().item() * xb.size(0)
            n     += xb.size(0)
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MDN physics initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _sp_inv(s):
    return float(np.log(np.exp(float(np.clip(s, 1e-3, 20.0))) - 1.0 + 1e-8))


def _init_nu_per_component(model, sigmas):
    n_g   = len(sigmas)
    nu_lo = model._nu_min + 0.5
    nu_hi = 20.0
    ranks = np.argsort(np.argsort(sigmas))
    log_nus = np.log(nu_lo) + (np.log(nu_hi) - np.log(nu_lo)) * ranks / max(n_g - 1, 1)
    nus = np.exp(log_nus).astype(np.float32)
    return torch.tensor(
        [float(np.log(np.exp(np.clip(v - model._nu_min, 1e-3, 20.0)) - 1.0 + 1e-8)) for v in nus],
        dtype=torch.float32)


def _init_mdn_from_tracks(model, tracks, point_analyses, target=None,
                           ratio_tracks=None, verbose=False):
    """Physics-informed initialisation of MDN weights from feature tracks.

    Components:
      - n_tracks (up to n_gaussians-2) skew-t components seeded from BW tracks.
      - n_cont continuum skew-t components linearly spaced in y.
      - n_cauchy Cauchy components seeded from BW tracks (nu/alpha pinned in forward()).

    When target and ratio_tracks are provided, BW track positions are snapped to
    the physics prediction at the target (ratio * p_i) rather than the grid average,
    and sigma is set to the exact WoMS_target * mS_target / (2 * SCALE) formula
    instead of the grid-averaged width.
    """
    n_g      = model.n_gaussians
    n_c      = model.n_cauchy
    n_total  = n_g + n_c
    n_tracks = min(len(tracks), max(n_g - 2, 1))
    n_cont   = n_g - n_tracks
    valid    = [a for a in point_analyses if a['n_events'] > 0]
    y_min    = min(a['y_min'] for a in valid) if valid else 0.0
    y_max    = max(a['y_max'] for a in valid) if valid else 1.0

    init_mus   = np.empty(n_total, dtype=np.float32)
    init_sig   = np.empty(n_total, dtype=np.float32)
    init_alpha = np.zeros(n_total, dtype=np.float32)

    # Physics-predicted BW sigma at target: WoMS_target * mS_target / (2 * SCALE)
    _target_bw_sigma_y = None
    if target is not None:
        woms_idx = next((k for k, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('WoMS')), -1)
        ms_idx   = next((k for k, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), 0)
        if woms_idx >= 0 and woms_idx < len(target):
            t_woms = float(target[woms_idx])
            t_ms   = float(target[ms_idx])
            from core.features import PDG_H_MASS
            if t_ms > 2.0 * PDG_H_MASS:
                _target_bw_sigma_y = max(t_woms * t_ms / (2.0 * SCALE), 1e-3)

    ratio_tracks = ratio_tracks or []

    for i, track in enumerate(tracks[:n_tracks]):
        mean_y    = float(np.mean([f['y_pos'] for f in track['features']]))
        _has_ratio = False
        if target is not None and ratio_tracks:
            for rt in ratio_tracks:
                if rt.get('track_idx') == i:
                    try:
                        rt_ratio = rt.get('ratio', 0.0)
                        ms_idx_l = next((k for k, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), 0)
                        if ms_idx_l >= 0 and abs(rt_ratio - 1.0) < 0.05:
                            mean_y = transform_mhh(float(target[ms_idx_l]), cfg.MHH_THRESHOLD)
                        else:
                            mean_y = transform_mhh(
                                rt['predict_pos'](float(target[0]),
                                                  float(target[1]) if len(target) > 1 else 0.0),
                                cfg.MHH_THRESHOLD)
                        _has_ratio = True
                    except Exception:
                        pass
                    break

        mean_asym = float(np.mean([f.get('asymmetry', 0.0) for f in track['features']]))
        mean_lw   = float(np.mean([f.get('left_width_y',  f['width_y']/2.0) for f in track['features']]))
        mean_rw   = float(np.mean([f.get('right_width_y', f['width_y']/2.0) for f in track['features']]))
        init_mus[i] = mean_y

        # For BW tracks where target + ratio track are known, use physics-exact sigma
        if _target_bw_sigma_y is not None and track.get('sign', +1) > 0 and _has_ratio:
            init_sig[i] = _target_bw_sigma_y
        else:
            init_sig[i] = float(max(max(mean_lw, mean_rw), 1e-3))

        ta = float(np.clip(mean_asym * model._alpha_max, -model._alpha_max+0.1, model._alpha_max-0.1))
        init_alpha[i] = float(np.arctanh(ta / model._alpha_max)) if abs(ta) > 0.05 else 0.0

    cont_mus = np.linspace(y_min, y_max, n_cont).astype(np.float32)
    init_mus[n_tracks:n_g] = cont_mus
    init_sig[n_tracks:n_g] = float(max((y_max - y_min) / max(n_cont, 1), 0.05))

    bw_tracks = [t for t in tracks if t['sign'] > 0]
    for ci in range(n_c):
        if ci < len(bw_tracks):
            t     = bw_tracks[ci]
            t_idx = tracks.index(t)
            cy    = float(np.mean([f['y_pos'] for f in t['features']]))
            if target is not None and ratio_tracks:
                for rt in ratio_tracks:
                    if rt.get('track_idx') == t_idx:
                        try:
                            rt_ratio = rt.get('ratio', 0.0)
                            ms_idx_l = next((k for k, n in enumerate(cfg.PARAM_NAMES or []) if n.startswith('mS')), 0)
                            if ms_idx_l >= 0 and abs(rt_ratio - 1.0) < 0.05:
                                cy = transform_mhh(float(target[ms_idx_l]), cfg.MHH_THRESHOLD)
                            else:
                                cy = transform_mhh(
                                    rt['predict_pos'](float(target[0]),
                                                      float(target[1]) if len(target) > 1 else 0.0),
                                    cfg.MHH_THRESHOLD)
                        except Exception:
                            pass
                        break
            init_mus[n_g + ci] = cy
            # Cauchy scale = Lorentzian HWHM. Prefer physics formula when target is known.
            init_sig[n_g + ci] = (_target_bw_sigma_y if _target_bw_sigma_y is not None
                                   else float(max(np.mean([f['width_y'] for f in t['features']]) / 2.0, 1e-3)))
        else:
            init_mus[n_g + ci] = float((y_min + y_max) / 2.0)
            init_sig[n_g + ci] = float(max((y_max - y_min) / 4.0, 0.05))

    nu_biases = _init_nu_per_component(model, init_sig[:n_g])
    with torch.no_grad():
        model.mu.bias.copy_(torch.tensor(init_mus, dtype=torch.float32))
        model.mu.weight.zero_()   # start context-independent; learned gradually
        model.sigma.bias.copy_(torch.tensor([_sp_inv(s) for s in init_sig], dtype=torch.float32))
        model.sigma.weight.zero_()
        model.alpha.bias.copy_(torch.tensor(init_alpha, dtype=torch.float32))
        model.alpha.weight.zero_()
        nu_full = torch.zeros(n_total, dtype=torch.float32)
        nu_full[:n_g] = nu_biases
        model.nu.bias.copy_(nu_full)
        model.nu.weight.zero_()
        # pi: zero context weights so they start flat and context-dependence is
        # learned gradually. Bias: give Cauchy components a prior boost so the
        # BW peak starts with ~50% total weight, preventing continuum Gaussians
        # from collapsing onto the peak during early training.
        # Continuum Gaussians start suppressed (large negative bias) so the
        # model is essentially a pure Cauchy BW at epoch 0.  Training gradually
        # promotes continuum components as the fit requires.
        # pi.weight: keep kaiming default (do NOT zero) so the network has
        # context-dependent pi from epoch 1.  With pi.weight=0, all contexts
        # get identical pi logits throughout early training; the continuum
        # components trained on mS1=100 (pure spectrum) grow their pi.bias
        # context-independently, bleeding into the post-peak tail at mS1=850.
        # Non-zero pi.weight lets Adam learn context-dependent mixing immediately.
        pi_bias = torch.full((n_total,), -5.0, dtype=torch.float32)
        pi_bias[n_g:] = float(math.log(max(n_total, 1)))   # Cauchy: boosted above continuum
        # Track components: intermediate boost (below Cauchy, above continuum)
        for i in range(n_tracks):
            pi_bias[i] = float(math.log(max(n_total, 1))) - 1.0
        model.pi.bias.copy_(pi_bias)
        # Store for training loop sparsity penalty
        model._n_bw_tracks = n_tracks

    if verbose:
        print(f'MDN init: {n_tracks} track + {n_cont} continuum + {n_c} Cauchy components'
              + (f' [physics sigma={_target_bw_sigma_y:.4f}]' if _target_bw_sigma_y else ''))


def _init_ebm_from_tracks(model, tracks, point_analyses, verbose=False):
    """Physics-informed init for EBMModel: set energy output bias to log(Cauchy sum).

    The energy function f(y, ctx) is initialised so that exp(f) ≈ sum of Cauchy
    peaks detected by the feature finder.  This gives the EBM a head-start in the
    same way _init_mdn_from_tracks does for the MDN.

    Implementation: we set the _energy_out bias to log(peak_density) at the mean
    of all track positions, which is a crude but non-trivial initialisation.
    The network quickly refines this during training.
    """
    if not tracks:
        return
    y_min = min(a['y_min'] for a in point_analyses if a['n_events'] > 0)
    y_max = max(a['y_max'] for a in point_analyses if a['n_events'] > 0)
    peak_y = float(np.mean([
        float(np.mean([f['y_pos'] for f in t['features']]))
        for t in tracks if t['sign'] > 0
    ] or [(y_min + y_max) / 2.0]))
    # Zero all MLP weights so f_ML starts near 0, but keep output weight
    # as small random (std=0.01) so gradients can actually flow.
    # With _energy_out.weight=0, ∂f/∂any_param = 0 everywhere — model never learns.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'energy' in name and '_out' not in name:
                if 'weight' in name or 'bias' in name:
                    p.zero_()
        # Output weight: small random to break symmetry and allow gradient flow
        torch.nn.init.normal_(model._energy_out.weight, std=0.01)
        # Output bias: target log(1/(y_range)) so exp(f) integrates ~1 initially
        model._energy_out.bias.fill_(float(math.log(max(1.0 / max(y_max - y_min, 0.1), 1e-6))))
    if verbose:
        print(f'EBM init: energy MLP zeroed, bias={float(math.log(max(1.0 / max(y_max - y_min, 0.1), 1e-6))):.3f} '
              f'peak_y={peak_y:.3f}')


# ─────────────────────────────────────────────────────────────────────────────
# chi2 / grid reproduction monitor
# ─────────────────────────────────────────────────────────────────────────────

def monitor_grid_reproduction(model, source_data, mhh_min, mhh_max, bin_width,
                               model_type='nsf', verbose=False, device=None):
    """Adaptive-bin Pearson chi2/ndf between data histogram and model PDF.

    Bin merging strategy:
      1. Fine reference bins (1 GeV) over [mhh_min, mhh_max].
      2. Greedily accumulate fine bins until cumulative data count >= N_MIN_PER_BIN.
         This ensures sqrt-Poisson approximation is valid (20% relative uncertainty).
      3. chi2 = Σ (n_obs − n_exp)² / max(n_exp,1)  /  (k−1)
         where n_exp = pdf_mhh × n_events × merged_width.

    PDF evaluation uses analytic expressions (no sampling):
      MDN:    skew-t mixture evaluated at bin centres.
      NSF:    flow log_prob at bin centres.
      EBM:    energy function at bin centres (via pdf_on_grid).
      MDN+g:  log_prob at bin centres (via pdf_on_grid).

    Returns (scores_dict, overall_float)  where scores_dict = {coords: chi2/ndf}.
    """
    T = cfg.MHH_THRESHOLD
    dev = device or cfg.DEVICE
    N_MIN_PER_BIN = 25
    FINE_BW       = 1.0

    _normal = torch.distributions.Normal(0.0, 1.0)

    def _eval_pdf(y_centres_np, x_enc):
        n_pts = len(y_centres_np)
        x_t   = torch.tensor([x_enc], dtype=torch.float32, device=dev)
        y_t   = torch.tensor(y_centres_np, dtype=torch.float32, device=dev).unsqueeze(1)

        if model_type == 'mdn':
            pi, mu, sigma, alpha, nu = model(x_t)
            pi_b = pi.expand(n_pts, -1);  mu_b = mu.expand(n_pts, -1)
            sg_b = sigma.expand(n_pts, -1); al_b = alpha.expand(n_pts, -1)
            nu_b = nu.expand(n_pts, -1)
            z  = (y_t - mu_b) / sg_b
            lt = (torch.lgamma((nu_b+1)/2) - torch.lgamma(nu_b/2)
                  - 0.5*torch.log(nu_b*math.pi)
                  - 0.5*torch.log(sg_b**2)
                  - ((nu_b+1)/2)*torch.log(1+z**2/nu_b))
            ta = al_b * z * torch.sqrt((nu_b+1)/(nu_b+z**2+1e-8))
            lp = torch.logsumexp(torch.log(pi_b+1e-12)+math.log(2)+lt
                                 +torch.log(_normal.cdf(ta).clamp(1e-38)), dim=1)
            return lp.exp().cpu().numpy()

        elif model_type == 'nsf':
            ctx = x_t.expand(n_pts, -1)
            eager = getattr(model, '_orig_mod', model)
            lp = eager._flow.log_prob(y_t, context=eager._ctx_encoder(ctx))
            return lp.exp().cpu().numpy()

        elif model_type in ('ebm', 'mdn+g'):
            return model.pdf_on_grid(x_enc, y_centres_np)

        else:
            raise ValueError(f'Unknown model_type: {model_type}')

    def _merge(n_obs_fine, fine_edges):
        centres, widths, counts = [], [], []
        acc, lo = 0, fine_edges[0]
        for i, cnt in enumerate(n_obs_fine):
            acc += cnt
            if acc >= N_MIN_PER_BIN:
                hi = fine_edges[i+1]
                centres.append(0.5*(lo+hi)); widths.append(hi-lo); counts.append(int(acc))
                acc = 0; lo = hi
        if acc > 0:
            centres.append(0.5*(lo+fine_edges[-1]))
            widths.append(fine_edges[-1]-lo); counts.append(int(acc))
        return centres, widths, counts

    model.eval()
    scores = {}
    with torch.no_grad():
        for coords, files in source_data:
            all_mhh = []
            for f in files:
                cache = preprocess_file(f, coords)
                try:
                    t = torch.load(cache, weights_only=True)
                except Exception:
                    continue
                if t.shape[0] > 0:
                    all_mhh.extend(inverse_transform_mhh(t[:, -1].numpy(), T).tolist())
            if not all_mhh: continue
            all_mhh  = np.array(all_mhh)
            all_mhh  = all_mhh[(all_mhh >= mhh_min) & (all_mhh <= mhh_max)]
            n_events = len(all_mhh)
            if n_events < N_MIN_PER_BIN: continue

            fine_edges = np.arange(mhh_min, mhh_max + FINE_BW, FINE_BW)
            n_obs_fine, _ = np.histogram(all_mhh, bins=fine_edges)
            m_c, m_w, m_n = _merge(n_obs_fine, fine_edges)
            if len(m_c) < 2: continue

            x_enc = encode_ctx(*[float(v) for v in coords])
            y_c   = transform_mhh(np.array(m_c), T)
            pdf_y = _eval_pdf(y_c, x_enc)

            pdf_mhh = pdf_y / SCALE
            n_exp   = pdf_mhh * n_events * np.array(m_w)
            n_obs_m = np.array(m_n, dtype=float)
            chi2    = float(np.sum((n_obs_m - n_exp)**2 / np.maximum(n_exp, 1.0))) \
                      / max(len(m_c) - 1, 1)
            scores[coords] = chi2

            if verbose:
                names = cfg.PARAM_NAMES or [f'p{i+1}' for i in range(len(coords))]
                cs = '  '.join(f'{names[i]}={coords[i]:.4g}' for i in range(len(coords)))
                print(f'  [REPRO] {cs}  χ²/ndf={chi2:.3f}  k={len(m_c)}  N={n_events}')

    if not scores:
        return {}, float('nan')
    overall = float(np.median(list(scores.values())))
    if verbose:
        print(f'  [REPRO] median χ²/ndf={overall:.3f}  '
              f'(n={len(scores)}, best={min(scores.values()):.3f}, worst={max(scores.values()):.3f})')
    return scores, overall


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop (NSF / EBM / MDN+g share this)
# ─────────────────────────────────────────────────────────────────────────────

def _make_scheduler(opt, epochs, lr):
    n_warmup = min(5, epochs // 10)
    warmup   = optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, n_warmup)
    cosine   = optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs - n_warmup, 1), lr * 1e-2)
    return optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[n_warmup])


def _prior_pool(sources, epochs):
    """Concatenate multiple prior (y, ctx, w) pools; return (y, ctx, w, anneal_epochs)."""
    ys, ctxs, ws, ae = [], [], [], 0
    for y, ctx, w, meta in sources:
        if y is not None:
            ys.append(y); ctxs.append(ctx); ws.append(w)
            ae = max(ae, max(1, int(epochs * meta['anneal_frac'])))
    if not ys:
        return None, None, None, 0
    return torch.cat(ys), torch.cat(ctxs), torch.cat(ws), ae


def _train_generic(
    model, model_type, loss_fn,
    dataset, source_data,
    epochs, batch_size, patience, lr, wd,
    target, mhh_min, mhh_max, full_training,
    prior_y, prior_ctx, prior_w, anneal_epochs,
    chi2_target, verbose, device,
    bin_width=10.0,
    use_amp=False, accum_steps=1, gpu_duty_cycle=1.0,
    thermal_monitor=False, thermal_target=85, thermal_cooldown=70,
):
    """Generic epoch loop shared by NSF, EBM, MDN+g."""
    # Memory layout
    feat_bytes = dataset.features.element_size() * dataset.features.nelement()
    tgt_bytes  = dataset.targets.element_size()  * dataset.targets.nelement()
    use_gpu_res = False
    if device.type == 'cuda':
        avail = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_reserved(0))
        use_gpu_res = (feat_bytes + tgt_bytes) < 0.7 * avail

    if use_gpu_res:
        feats_g = dataset.features.to(device)
        tgts_g  = dataset.targets.to(device)
        wgts_g  = dataset.weights.to(device)

    if prior_y is not None:
        prior_y   = prior_y.to(device)
        prior_ctx = prior_ctx.to(device)
        prior_w   = prior_w.to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = _make_scheduler(opt, epochs, lr)
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    best_chi2  = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = restore_count = 0
    _rp = max(patience // 3, 5)
    _mr = 8
    global_epoch = 0

    # Validation subset for monitoring (not excluded from training)
    _n_val  = max(200, len(dataset) // 10)
    _val_pi = torch.randperm(len(dataset))[:_n_val]
    _val_f  = dataset.features[_val_pi]
    _val_t  = dataset.targets[_val_pi]
    _val_w  = dataset.weights[_val_pi]

    # Model-type-specific gradient clipping
    _GRAD_CLIPS = {'nsf': 0.1, 'ebm': 0.5, 'mdn+g': 0.5}
    _grad_clip  = _GRAD_CLIPS.get(model_type, 0.5)

    if model_type != 'nsf' and device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            cc = torch.cuda.get_device_capability(0)
            if float(f'{cc[0]}.{cc[1]}') >= 7.0:
                model = torch.compile(model)
                if verbose:
                    print(f'torch.compile() enabled (sm {cc[0]}.{cc[1]})')
        except Exception:
            pass

    # ── NSF prior warm-up ─────────────────────────────────────────────────
    # Rational-quadratic spline knots start uniformly spaced and need many
    # gradient steps to concentrate near the BW peak.  Run a warm-up phase on
    # prior samples mixed with real data before the main training loop begins.
    # Mixing real data (50%) ensures context-dependence is learned from epoch 1.
    NSF_WARMUP = 40 if full_training else 20
    if model_type == 'nsf' and prior_y is not None and NSF_WARMUP > 0:
        model.train()
        _wu_perm_p = torch.randperm(len(prior_y))
        _n_data    = len(dataset)
        _wu_perm_d = torch.randperm(_n_data)
        _wu_ratio  = 0.5  # fraction of each mini-batch from prior
        for _wu_ep in range(NSF_WARMUP):
            for _wi in range(0, len(prior_y), batch_size):
                _idx_p = _wu_perm_p[_wi:_wi + batch_size]
                _yb = prior_y[_idx_p]
                _xb = prior_ctx[_idx_p]
                _wb = prior_w[_idx_p]
                # Mix in real data
                _nd = max(1, int(len(_idx_p) * (1.0 - _wu_ratio)))
                _di = _wu_perm_d[(_wi // batch_size * _nd) % max(_n_data - _nd, 1):
                                  (_wi // batch_size * _nd) % max(_n_data - _nd, 1) + _nd]
                if len(_di) > 0:
                    _yb = torch.cat([_yb, dataset.targets[_di].to(device)])
                    _xb = torch.cat([_xb, dataset.features[_di].to(device)])
                    _wb = torch.cat([_wb, dataset.weights[_di].to(device)])
                _loss = loss_fn(model, _yb, _xb, _wb) / accum_steps
                _loss.backward()
                if (_wi // batch_size + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    opt.step()
                    opt.zero_grad()
                    _wu_perm_p = torch.randperm(len(prior_y))
            sch.step()
        if verbose:
            print(f'[NSF   ] Prior warm-up: {NSF_WARMUP} epochs done (50% prior + 50% data).')
    # ── end NSF warm-up ───────────────────────────────────────────────────

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        n_data = len(dataset)
        perm   = torch.randperm(n_data)
        total_l, total_n, step = 0.0, 0, 0

        for i in range(0, n_data, batch_size):
            idx = perm[i:i+batch_size]
            if use_gpu_res:
                xb, yb, wb = feats_g[idx], tgts_g[idx], wgts_g[idx]
            else:
                xb = dataset.features[idx].to(device)
                yb = dataset.targets[idx].to(device)
                wb = dataset.weights[idx].to(device)

            if prior_y is not None and epoch < anneal_epochs:
                ann   = 1.0 - epoch / anneal_epochs
                n_inj = max(1, min(xb.size(0), int(len(prior_y) * ann)))
                pidx  = torch.randint(0, len(prior_y), (n_inj,), device=device)
                yb = torch.cat([yb, prior_y[pidx]])
                xb = torch.cat([xb, prior_ctx[pidx]])
                wb = torch.cat([wb, prior_w[pidx] * ann])

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model, yb, xb, wb) / accum_steps
                scaler.scale(loss).backward()
            else:
                loss = loss_fn(model, yb, xb, wb) / accum_steps
                loss.backward()

            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

                if gpu_duty_cycle < 1.0 and device.type == 'cuda':
                    time.sleep((time.time() - t0) * (1.0/gpu_duty_cycle - 1.0))
                if thermal_monitor:
                    _thermal_governor(thermal_target, thermal_cooldown, verbose=verbose)

            total_l += loss.item() * accum_steps * xb.size(0)
            total_n += xb.size(0)

        sch.step()
        avg     = total_l / total_n
        elapsed = time.time() - t0

        _, chi2 = monitor_grid_reproduction(
            getattr(model, '_orig_mod', model), source_data, mhh_min, mhh_max,
            bin_width=bin_width, model_type=model_type, verbose=False, device=device)

        val_nll = _eval_nll_generic(model, model_type, _val_f, _val_t, _val_w, device=device)

        improved = not math.isnan(chi2) and chi2 < best_chi2 - 1e-3
        if improved:
            best_chi2  = chi2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = restore_count = 0
            try:
                from core.plotting import plot_partial_distribution
                plot_partial_distribution(
                    getattr(model, '_orig_mod', model), target, epoch+1,
                    mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width,
                    verbose=False, model_type=model_type, full_training=full_training,
                )
            except Exception:
                pass
            if verbose:
                print(_fmt_epoch(model_type, epoch+1, epochs, avg, val_nll, chi2, elapsed, best=True))
            if chi2 < chi2_target:
                if verbose:
                    print(f'[{model_type.upper():<6}] Converged at epoch {epoch+1}')
                break
        else:
            no_improve += 1
            if verbose or (epoch+1) % 10 == 0:
                print(_fmt_epoch(model_type, epoch+1, epochs, avg, val_nll, chi2, elapsed, best=False))
            if no_improve >= _rp and restore_count < _mr:
                restore_count += 1; no_improve = 0
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
                if verbose:
                    print(f'[{model_type.upper():<6}] Restore #{restore_count}  lr→{opt.param_groups[0]["lr"]:.2e}')

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    try:
        from core.plotting import plot_partial_distribution
        plot_partial_distribution(
            getattr(model, '_orig_mod', model), target, epoch + 1,
            mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width,
            verbose=False, model_type=model_type, full_training=full_training,
        )
    except Exception:
        pass
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MDN full training (uses its own epoch loop with pi_sparse term)
# ─────────────────────────────────────────────────────────────────────────────

def _train_full_mdn(dataset, source_data, target,
                    mhh_min, mhh_max, bin_width,
                    epochs, full_training,
                    n_cauchy, n_gaussians, physics_prior_pool,
                    comment_parsed, tracks, analyses,
                    chi2_target, verbose, device, kwargs):
    if analyses is None:
        analyses = [extract_features_at_coords(c, f, topology=cfg.TOPOLOGY)
                    for c, f in source_data]
    if tracks is None:
        tracks = build_feature_tracks(analyses, verbose=verbose)
    ratio_tracks = build_ratio_tracks(tracks, analyses,
                                      source_data=source_data, verbose=verbose)

    analytic_bw_kw  = _mdn_analytic_bw_kwargs(n_cauchy)
    analytic_enabled = analytic_bw_kw.get('analytic_bw', False)
    model = MDNModel(n_gaussians=n_gaussians, n_cauchy=n_cauchy,
                     **analytic_bw_kw).to(device)
    if verbose and analytic_enabled:
        print(f'[ANALYTIC-BW] Cauchy mu/sigma pinned to physics formula '
              f'(mass_axes={analytic_bw_kw["bw_mass_axes"]}, '
              f'wom_axes={analytic_bw_kw["bw_wom_axes"]}, '
              f'threshold_y={analytic_bw_kw["threshold_y"]:.4f})')
    _init_mdn_from_tracks(model, tracks, analyses, target=target,
                          ratio_tracks=ratio_tracks, verbose=verbose)

    # Prior floor: small residual prior weight after annealing counters Voronoi bias,
    # unless analytic_bw is active (peak is already pinned exactly).
    _MDN_PRIOR_FLOOR = 0.0 if analytic_enabled else (
        0.05 if (cfg.TOPOLOGY and cfg.TOPOLOGY.get('single_bw')) else 0.0
    )

    if device.type == 'cuda':
        feats_g = dataset.features.to(device)
        tgts_g  = dataset.targets.to(device)
        wgts_g  = dataset.weights.to(device)

    train_idx, val_idx = _make_val_split(dataset, 0.1)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)
    n_train   = len(train_idx)

    # Prior injection
    mdn_prior_y = mdn_prior_ctx = mdn_prior_w = None
    mdn_anneal  = 0
    total_data_w = float(dataset.weights[train_idx.cpu()].sum())

    def _inject(y, ctx, w_raw, meta):
        nonlocal mdn_prior_y, mdn_prior_ctx, mdn_prior_w, mdn_anneal
        w = (w_raw / w_raw.sum()) * total_data_w * meta['prior_weight']
        mdn_prior_y   = y.to(device)
        mdn_prior_ctx = ctx.to(device)
        mdn_prior_w   = w.to(device)
        mdn_anneal    = max(1, int(epochs * meta['anneal_frac']))

    if physics_prior_pool is not None:
        _inject(*physics_prior_pool)
    elif comment_parsed is not None:
        c_y, c_ctx, c_w = generate_prior_samples(comment_parsed, source_data, 5000, mhh_min, mhh_max)
        if c_y is not None:
            _inject(c_y, c_ctx, c_w, comment_parsed)
    elif tracks:
        t_y, t_ctx, t_w, t_meta = build_track_prior(tracks, source_data, analyses)
        if t_y is not None:
            _inject(t_y, t_ctx, t_w, t_meta)

    lr = 2e-4
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = _make_scheduler(opt, epochs, lr)

    best_chi2  = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = restore_count = 0
    _rp = max(kwargs.get('patience', 30) // 3, 5)
    _mr = 8

    # Per-grid on-grid NLL index
    _per_grid_idx = {}
    for _coords, _ in source_data:
        _ctx_enc = torch.tensor(encode_ctx(*[float(v) for v in _coords]),
                                dtype=torch.float32)
        _mask = (dataset.features == _ctx_enc.unsqueeze(0)).all(dim=1)
        _idxs = _mask.nonzero(as_tuple=True)[0]
        if len(_idxs) > 0:
            _per_grid_idx[_coords] = _idxs.to(device)

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_l, total_n = 0.0, 0
        for i in range(0, n_train, 8192):
            local_b = perm[i:i+8192]
            idx_b   = train_idx[local_b]
            if device.type == 'cuda':
                xb, yb, wb = feats_g[idx_b], tgts_g[idx_b], wgts_g[idx_b]
            else:
                xb = dataset.features[idx_b.cpu()].to(device)
                yb = dataset.targets[idx_b.cpu()].to(device)
                wb = dataset.weights[idx_b.cpu()].to(device)

            if mdn_prior_y is not None:
                ann = max(_MDN_PRIOR_FLOOR, 1.0 - ep / mdn_anneal) if mdn_anneal > 0 else _MDN_PRIOR_FLOOR
                if ann > 0:
                    n_inj = max(1, min(xb.size(0), int(len(mdn_prior_y) * ann * 0.5)))
                    pidx  = torch.randint(0, len(mdn_prior_y), (n_inj,), device=device)
                    prior_w_batch = wb.detach().mean().expand(n_inj)
                    yb = torch.cat([yb, mdn_prior_y[pidx]])
                    xb = torch.cat([xb, mdn_prior_ctx[pidx]])
                    wb = torch.cat([wb, prior_w_batch])

            opt.zero_grad()
            pi, mu, sigma, alpha, nu = model(xb)
            loss = mdn_loss(pi, mu, sigma, alpha, nu, yb, voronoi_w=wb)
            # Sparsity on continuum pi: penalise the mean weight of the continuum
            # skew-t components (indices n_bw_tracks..n_gaussians-1).  Without this,
            # mS1=100 data (pure falling spectrum, voronoi-weighted) drives continuum
            # pi.bias up context-independently, producing a floor in the post-peak
            # tail at high-mS1 targets.  lambda=0.03 is ~2% of typical NLL.
            _nt = getattr(model, '_n_bw_tracks', 1)
            if model.n_gaussians > _nt + 1:
                loss = loss + 0.03 * pi[:, _nt:model.n_gaussians].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_l += loss.item() * xb.size(0)
            total_n += xb.size(0)
        sch.step()

        avg = total_l / total_n
        elapsed = time.time() - t0

        _val_f = dataset.features[val_idx.cpu()]
        _val_t = dataset.targets[val_idx.cpu()]
        _val_w = dataset.weights[val_idx.cpu()]
        val_nll = _eval_nll_mdn(model, _val_f, _val_t, _val_w, device=device)

        _, chi2 = monitor_grid_reproduction(
            model, source_data, mhh_min, mhh_max,
            bin_width=bin_width, model_type='mdn', verbose=False, device=device)

        if chi2 < best_chi2 - 1e-3:
            best_chi2 = chi2; no_improve = restore_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            try:
                from core.plotting import plot_partial_distribution
                plot_partial_distribution(
                    model, target, ep+1,
                    mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width,
                    verbose=False, model_type='mdn', full_training=full_training,
                )
            except Exception:
                pass
            if verbose:
                print(_fmt_epoch('mdn', ep+1, epochs, avg, val_nll, chi2, elapsed, best=True))
                if _per_grid_idx:
                    _grid_nlls = {}
                    model.eval()
                    for _gc, _gi in _per_grid_idx.items():
                        gf = dataset.features[_gi.cpu()]
                        gt = dataset.targets[_gi.cpu()]
                        gw = dataset.weights[_gi.cpu()]
                        _grid_nlls[_gc] = _eval_nll_mdn(model, gf, gt, gw, device=device)
                    model.train()
                    _worst = max(_grid_nlls, key=_grid_nlls.get)
                    _best  = min(_grid_nlls, key=_grid_nlls.get)
                    _wn    = cfg.PARAM_NAMES or [f'p{i}' for i in range(len(_worst))]
                    def _fmt(c):
                        return ', '.join(f'{_wn[i]}={c[i]:.4g}' for i in range(len(c)))
                    print(f'    On-grid NLL: worst [{_fmt(_worst)}]={_grid_nlls[_worst]:.4f}  '
                          f'best [{_fmt(_best)}]={_grid_nlls[_best]:.4f}')
            if chi2 < chi2_target:
                if verbose: print(f'[MDN   ] Converged at epoch {ep+1}')
                break
        else:
            no_improve += 1
            if verbose or (ep+1) % 10 == 0:
                print(_fmt_epoch('mdn', ep+1, epochs, avg, val_nll, chi2, elapsed, best=False))
            if no_improve >= _rp and restore_count < _mr:
                restore_count += 1; no_improve = 0
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
                if verbose:
                    print(f'[MDN   ] Restore #{restore_count}  lr→{opt.param_groups[0]["lr"]:.2e}')

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    try:
        from core.plotting import plot_partial_distribution
        plot_partial_distribution(
            model, target, epochs,
            mhh_min=mhh_min, mhh_max=mhh_max, bin_width=bin_width,
            verbose=False, model_type='mdn', full_training=full_training,
        )
    except Exception:
        pass
    return model, best_chi2


# ─────────────────────────────────────────────────────────────────────────────
# Probe trainers (lightweight, for model-race)
# ─────────────────────────────────────────────────────────────────────────────

def _train_probe(model, model_type, loss_fn, dataset, source_data, train_idx,
                 probe_epochs, prior_y, prior_ctx, prior_w, anneal_epochs, device, verbose):
    train_set = set(train_idx.tolist())
    val_idx   = torch.tensor([i for i in range(len(dataset)) if i not in train_set])
    feats     = dataset.features[train_idx]
    tgts      = dataset.targets[train_idx]
    wgts      = dataset.weights[train_idx]

    if prior_y is not None:
        prior_y   = prior_y.to(device)
        prior_ctx = prior_ctx.to(device)
        prior_w   = prior_w.to(device)

    lr  = 2e-4
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = _make_scheduler(opt, probe_epochs, lr)
    n   = len(train_idx)
    best_val, best_state = float('inf'), None

    for ep in range(probe_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            idx = perm[i:i+8192]
            xb  = feats[idx].to(device)
            yb  = tgts[idx].to(device)
            wb  = wgts[idx].to(device)
            if prior_y is not None and ep < anneal_epochs:
                ann   = 1.0 - ep / anneal_epochs
                n_inj = max(1, min(xb.size(0), int(len(prior_y) * ann * 0.5)))
                pidx  = torch.randint(0, len(prior_y), (n_inj,), device=device)
                yb = torch.cat([yb, prior_y[pidx]]); xb = torch.cat([xb, prior_ctx[pidx]])
                wb = torch.cat([wb, prior_w[pidx] * ann])
            opt.zero_grad()
            if model_type == 'mdn':
                pi, mu, sigma, alpha, nu = model(xb)
                loss = mdn_loss(pi, mu, sigma, alpha, nu, yb, voronoi_w=wb)
            else:
                loss = loss_fn(model, yb, xb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sch.step()
        val_nll = _eval_nll_generic(model, model_type,
                                    dataset.features[val_idx],
                                    dataset.targets[val_idx],
                                    dataset.weights[val_idx], device=device)
        if val_nll < best_val:
            best_val   = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f'  [{model_type.upper()} probe] Epoch {ep+1}/{probe_epochs} '
                  f'| val {val_nll:.4f} | {time.time()-t0:.1f}s')

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val


# ─────────────────────────────────────────────────────────────────────────────
# Comment → routing
# ─────────────────────────────────────────────────────────────────────────────

def _comment_credibility(comment_parsed):
    if comment_parsed is None:
        return {'neg_weight': 1.0, 'pos_weight': 1.0}
    types = {c['type'] for c in comment_parsed['components']}
    if 'interference_neg' in types:
        return {'neg_weight': 2.0, 'pos_weight': 1.0}
    if types & {'bw', 'threshold', 'interference_pos'}:
        return {'neg_weight': 0.15, 'pos_weight': 1.0}
    if types == {'continuum'}:
        return {'neg_weight': 0.5, 'pos_weight': 0.5}
    return {'neg_weight': 1.0, 'pos_weight': 1.0}


def _comment_routing(comment_parsed):
    if comment_parsed is None:
        return 'race'
    types = {c['type'] for c in comment_parsed['components']}
    if ('interference_neg' in types or 'interference_pos' in types) and not (types & {'bw', 'threshold'}):
        return 'nsf'
    if types & {'bw', 'threshold'} and not types & {'interference_neg', 'interference_pos'}:
        return 'mdn'
    return 'race'


def _track_routing(tracks, credibility=None):
    if not tracks:
        return 'race'
    cred  = credibility or {'neg_weight': 1.0}
    signs = set()
    for t in tracks:
        if t['sign'] > 0 and not t.get('paired_interference'):
            signs.add(+1)
        elif t['sign'] < 0 and cred['neg_weight'] >= 0.3:
            signs.add(-1)
        elif t['sign'] > 0 and t.get('paired_interference'):
            signs.add(-1)
    if not signs:
        return 'race'
    return 'mdn' if signs == {+1} else 'race'


def _topology_routing(topology):
    """Route model selection from database topology — fast path skipping probe race.

    squarks_only : only asymmetric threshold features, no dips → MDN.
    single_bw    : one clean resonance on falling continuum → MDN.
    anything else: may have interference → race.
    """
    if topology is None:
        return 'race'
    if topology.get('squarks_only'):
        return 'mdn'
    if topology.get('single_bw'):
        return 'mdn'
    return 'race'


def _extract_mdn_prior(mdn_model, source_data):
    mdn_model.eval()
    components = []
    seen = []
    with torch.no_grad():
        for coords, _ in source_data:
            x_enc = encode_ctx(*[float(v) for v in coords])
            x = torch.tensor([x_enc], dtype=torch.float32, device=cfg.DEVICE)
            pi, mu, sigma, alpha, nu = mdn_model(x)
            pi_np = pi.cpu().numpy().flatten()
            mu_np = mu.cpu().numpy().flatten()
            sg_np = sigma.cpu().numpy().flatten()
            narrow = sg_np < 0.15
            if narrow.any():
                best = int(np.argmax(pi_np * narrow))
                peak_mhh = float(inverse_transform_mhh(
                    np.array([mu_np[best]]), cfg.MHH_THRESHOLD)[0])
                if not any(abs(peak_mhh - p) < 100.0 for p in seen):
                    seen.append(peak_mhh)
                    components.append({'type': 'bw', 'peak': None, 'width': None, 'sign': +1})
                    break
    if not components:
        return None
    components.append({'type': 'continuum', 'peak': None, 'width': None, 'sign': +1})
    return {'components': components, 'prior_weight': 0.30, 'anneal_frac': 0.10}


# ─────────────────────────────────────────────────────────────────────────────
# select_and_train  — unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def select_and_train(
    dataset, source_data, target,
    mhh_min, mhh_max, bin_width,
    epochs, full_training,
    comment_parsed=None,
    probe_epochs=20,
    force_model=None,        # 'mdn'|'nsf'|'ebm'|'mdn+g'|None
    train_all_models=False,
    chi2_target=2.0,
    verbose=False,
    num_workers=None,
    use_amp=False,
    accum_steps=1,
    gpu_duty_cycle=1.0,
    thermal_monitor=False,
    thermal_target=85,
    thermal_cooldown=70,
    patience=30,
    lambda_smooth=None,
    precomputed_features=None,  # dict(num_bins, tail_bound, analyses, tracks) — skip re-scan
):
    """Unified model selection and training.

    Steps:
      1. Feature extraction + track building (always, feeds hyperparams/priors/routing).
      2. Routing decision (comment > tracks > race).
      3. Optional probe race (lightweight training of candidates).
      4. Full training of winner.

    force_model overrides routing: 'mdn', 'nsf', 'ebm', or 'mdn+g'.
    train_all_models: train all four models, pick by chi2/ndf.
    """
    dev = cfg.DEVICE
    kwargs = dict(patience=patience)

    # ── Feature analysis ─────────────────────────────────────────────────────
    if precomputed_features is not None:
        num_bins   = precomputed_features['num_bins']
        tail_bound = precomputed_features['tail_bound']
        _analyses  = precomputed_features['analyses']
        _tracks    = precomputed_features['tracks']
    else:
        num_bins, tail_bound, _analyses, _tracks = auto_nsf_hyperparams(
            source_data, mhh_min, mhh_max, verbose=verbose)
        reclassify_paired_tracks(_tracks, _analyses)

    if verbose and precomputed_features is None:
        names = cfg.PARAM_NAMES or [f'p{i+1}' for i in range(len(source_data[0][0]))]
        print(f'\n[FEATURES] Grid points analysed: {len(_analyses)}')
        for a in _analyses:
            coord_str = '  '.join(f'{names[i]}={a["coords"][i]:.6g}' for i in range(len(a["coords"])))
            n_feat = len(a['features'])
            print(f'  [{coord_str}]  N={a["n_events"]}  features={n_feat}')
            for f in a['features']:
                print(f'    {f["feature_type"]:20s}  pos={f["mhh_position"]:.1f} GeV  '
                      f'width={f["width_gev"]:.1f} GeV  sign={"+1" if f["sign"]>0 else "-1"}  '
                      f'sig={f["significance"]:.1f}σ')
        print(f'[FEATURES] Tracks: {len(_tracks)}')
        for i, t in enumerate(_tracks):
            sign_str = '+1' if t['sign'] > 0 else '-1'
            n_pts_tr = len(t['features'])
            from core.config import inverse_transform_mhh as _itm
            pos_vals = [float(_itm(f['y_pos'], cfg.MHH_THRESHOLD)) for f in t['features']]
            print(f'  Track {i}: sign={sign_str}  n_pts={n_pts_tr}  '
                  f'pos range=[{min(pos_vals):.1f}, {max(pos_vals):.1f}] GeV'
                  + (f'  [paired_interference]' if t.get('paired_interference') else ''))
        print()

    _cred     = _comment_credibility(comment_parsed)
    _n_bw     = sum(1 for t in _tracks if t['sign'] > 0 and not t.get('paired_interference'))
    _n_cauchy = min(_n_bw, 3)

    # For single_bw: database name guarantees exactly one BW resonance.
    # Force n_cauchy=1 (analytic_bw handles it exactly) and keep n_gaussians small.
    if cfg.TOPOLOGY and cfg.TOPOLOGY.get('single_bw') and _n_cauchy == 0:
        _n_cauchy = 1
        if verbose:
            print('[CAUCHY] single_bw: enforcing n_cauchy=1 (feature finder found 0 BW tracks)')
    # single_bw optimisation (fewer components) only for target-neighbour mode;
    # full training spans many peak positions and needs full capacity.
    _n_gaussians = 8 if (cfg.TOPOLOGY and cfg.TOPOLOGY.get('single_bw') and not full_training) else 14

    # Build physics prior pool once
    _phys_prior = None
    if comment_parsed is None:
        pp_y, pp_ctx, pp_w, pp_meta = build_physics_prior_from_analyses(
            _analyses, source_data, n_per_coord=5000,
            prior_weight=0.30, anneal_frac=0.10,
            credibility=_cred, verbose=verbose)
        if pp_y is not None:
            total_w  = float(dataset.weights.sum())
            pp_w_sc  = (pp_w / pp_w.sum()) * total_w * pp_meta['prior_weight']
            _phys_prior = (pp_y, pp_ctx, pp_w_sc, pp_meta)

    # NSF probe prior (from tracks + comment)
    comps = []
    for t in _tracks:
        if t['sign'] > 0:
            comps.append({'type': 'bw' if not t.get('paired_interference') else 'interference_pos',
                          'peak': None, 'width': None, 'sign': +1})
        elif t['sign'] < 0 and _cred['neg_weight'] >= 0.3:
            comps.append({'type': 'interference_neg', 'peak': None, 'width': None, 'sign': -1})
    if comment_parsed:
        existing = {c['type'] for c in comps}
        comps += [c for c in comment_parsed['components'] if c['type'] not in existing]
    comps.append({'type': 'continuum', 'peak': None, 'width': None, 'sign': +1})
    _nsf_prior = {'components': comps, 'prior_weight': 0.50, 'anneal_frac': 0.30}

    train_kw = dict(
        source_data=source_data, target=target,
        mhh_min=mhh_min, mhh_max=mhh_max, full_training=full_training,
        chi2_target=chi2_target, verbose=verbose, device=dev,
        use_amp=use_amp, accum_steps=accum_steps, gpu_duty_cycle=gpu_duty_cycle,
        thermal_monitor=thermal_monitor, thermal_target=thermal_target,
        thermal_cooldown=thermal_cooldown,
    )

    # Helper: build and train one generic model
    _raw_abw = _mdn_analytic_bw_kwargs(1)
    _abw_kw_ebm = ({
        'analytic_bw': True,
        'threshold_y': _raw_abw['threshold_y'],
        'bw_mass_axis': _raw_abw['bw_mass_axes'][0],
        'bw_wom_axis':  _raw_abw['bw_wom_axes'][0],
    } if _raw_abw else {})

    def _run(mt, ep):
        ctx_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2
        if mt == 'nsf':
            _embed_dim = 128 if full_training else 64
            mdl = NSFModel(context_dim=ctx_dim, num_bins=num_bins,
                           tail_bound=tail_bound, embed_dim=_embed_dim).to(dev)
            lf  = nsf_nll_loss
        elif mt == 'ebm':
            # analytic_bw=True: Cauchy-CDF change of variables concentrates GL
            # quadrature nodes near the BW peak at runtime.  Without this, the
            # 2048 fixed nodes over [0, y_hi] have spacing ~0.003 y-units and
            # completely miss peaks of width ~0.0004 (WoMS1=0.001), so
            # d(log_Z)/dw ~ 0 near the peak and the EBM cannot converge.
            mdl = EBMModel(context_dim=ctx_dim, y_hi=tail_bound, n_bins=2048,
                           **_abw_kw_ebm).to(dev)
            _init_ebm_from_tracks(mdl, _tracks, _analyses, verbose=verbose)
            lf  = ebm_nll_loss
        elif mt == 'mdn+g':
            mdl = MDNCorrModel(in_dim=ctx_dim, y_hi=tail_bound,
                               **_abw_kw_ebm).to(dev)
            ratio_tracks_mdng = build_ratio_tracks(_tracks, _analyses,
                                                   source_data=source_data, verbose=False)
            _init_mdn_from_tracks(mdl._mdn, _tracks, _analyses,
                                  target=target, ratio_tracks=ratio_tracks_mdng,
                                  verbose=verbose)
            lf  = mdn_corr_loss
        else:
            raise ValueError(mt)

        py, pc, pw, ae = _prior_pool(
            [p for p in [_phys_prior,
                         (*generate_prior_samples(_nsf_prior, source_data, 5000, mhh_min, mhh_max),
                          _nsf_prior) if _nsf_prior else None]
             if p is not None], ep)

        _lrs = {'nsf': 1e-4, 'ebm': 3e-4, 'mdn+g': 3e-4}
        _bs  = 1024 if not full_training else 8192

        if mt == 'mdn+g':
            # Two-phase training:
            # Phase 1: freeze g-correction, train MDN base only.
            #   Allowing g to update from epoch 1 destabilises the MDN before
            #   it can converge: the joint NLL gradient grows g away from 0
            #   while the L2 term fights back, causing oscillation.
            # Phase 2: unfreeze g for residual fine-tuning with reduced LR.
            phase1_ep = max(ep // 3, min(ep, 100))
            phase2_ep = ep - phase1_ep
            for p in mdl.correction_parameters():
                p.requires_grad_(False)
            if verbose:
                print(f'[MDN+G ] Phase 1: {phase1_ep} epochs (g frozen)')
            mdl = _train_generic(
                mdl, mt, lf, dataset, source_data,
                phase1_ep, _bs, patience, _lrs[mt], 1e-5,
                target, mhh_min, mhh_max, full_training,
                py, pc, pw, ae, chi2_target, verbose, dev,
                bin_width=bin_width,
                use_amp=use_amp, accum_steps=accum_steps, gpu_duty_cycle=gpu_duty_cycle,
                thermal_monitor=thermal_monitor, thermal_target=thermal_target,
                thermal_cooldown=thermal_cooldown,
            )
            for p in mdl.correction_parameters():
                p.requires_grad_(True)
            if verbose:
                print(f'[MDN+G ] Phase 2: {phase2_ep} epochs (g unfrozen, lr×0.3)')
            return _train_generic(
                mdl, mt, lf, dataset, source_data,
                phase2_ep, _bs, patience, _lrs[mt] * 0.3, 1e-5,
                target, mhh_min, mhh_max, full_training,
                py, pc, pw, ae, chi2_target, verbose, dev,
                bin_width=bin_width,
                use_amp=use_amp, accum_steps=accum_steps, gpu_duty_cycle=gpu_duty_cycle,
                thermal_monitor=thermal_monitor, thermal_target=thermal_target,
                thermal_cooldown=thermal_cooldown,
            ), mt

        return _train_generic(
            mdl, mt, lf, dataset, source_data,
            ep, _bs, patience, _lrs.get(mt, 2e-4), 1e-5,
            target, mhh_min, mhh_max, full_training,
            py, pc, pw, ae, chi2_target, verbose, dev,
            bin_width=bin_width,
            use_amp=use_amp, accum_steps=accum_steps, gpu_duty_cycle=gpu_duty_cycle,
            thermal_monitor=thermal_monitor, thermal_target=thermal_target,
            thermal_cooldown=thermal_cooldown,
        ), mt

    def _call_full_mdn(t_analyses=None, t_tracks=None):
        return _train_full_mdn(
            dataset, source_data, target, mhh_min, mhh_max, bin_width,
            epochs, full_training, _n_cauchy, _n_gaussians, _phys_prior,
            comment_parsed, t_tracks or _tracks, t_analyses or _analyses,
            chi2_target, verbose, dev, kwargs)

    # ── force single model ───────────────────────────────────────────────────
    if force_model == 'mdn':
        m, _ = _call_full_mdn()
        return m, 'mdn'

    if force_model in ('nsf', 'ebm', 'mdn+g'):
        m, mt = _run(force_model, epochs)
        return m, mt

    # ── train all models ─────────────────────────────────────────────────────
    if train_all_models:
        results = {}
        mdn_m, mdn_chi2 = _call_full_mdn()
        results['mdn'] = (mdn_m, mdn_chi2)
        for mt in ('ebm', 'mdn+g', 'nsf'):
            m, _ = _run(mt, epochs)
            _, c = monitor_grid_reproduction(m, source_data, mhh_min, mhh_max,
                                             bin_width=bin_width, model_type=mt, verbose=False)
            results[mt] = (m, c)
        winner = min(results, key=lambda k: results[k][1])
        if verbose:
            for mt, (_, c) in results.items():
                print(f'[SELECT] {mt.upper()} χ²/ndf={c:.3f}{"  ← winner" if mt==winner else ""}')
        return results[winner][0], winner

    # ── routing + probe race ─────────────────────────────────────────────────
    c_route    = _comment_routing(comment_parsed)
    t_route    = _track_routing(_tracks, _cred)
    topo_route = _topology_routing(cfg.TOPOLOGY)

    # Priority: comment > topology > tracks > race
    if c_route != 'race':
        routing = c_route
    elif topo_route != 'race':
        routing = topo_route
        if verbose:
            topo_str = cfg.TOPOLOGY['topo_str'] if cfg.TOPOLOGY else '?'
            print(f'[SELECT] Topology routing ({topo_str}) → {routing}')
    elif t_route != 'race':
        routing = t_route
    else:
        routing = 'race'

    if verbose:
        print(f'[SELECT] Routing: {routing.upper()}  '
              f'(comment={c_route}, topo={topo_route}, tracks={t_route})')

    n_grid       = len(source_data)
    neg_pts      = len({f['coords'] for t in _tracks if t['sign'] < 0 for f in t['features']})
    neg_frac     = neg_pts / max(n_grid, 1)
    comment_neg  = bool({c['type'] for c in comment_parsed['components']} & {'interference_neg'}) \
                   if comment_parsed else False
    substantial_neg = neg_frac >= 0.20 or comment_neg

    train_idx, _ = _make_val_split(dataset, 0.1)

    # Build probe prior
    py_pr, pc_pr, pw_pr = generate_prior_samples(_nsf_prior, source_data, 3000, mhh_min, mhh_max)
    ae_pr = max(1, int(probe_epochs * _nsf_prior['anneal_frac'])) if py_pr is not None else 0
    if py_pr is not None:
        total_w = float(dataset.weights.sum())
        pw_pr   = (pw_pr / pw_pr.sum()) * total_w * _nsf_prior['prior_weight']

    probe_results = {}
    ctx_dim = len(cfg.PARAM_NAMES) if cfg.PARAM_NAMES else 2

    # MDN probe (skip only if comment/topology hard-routes to NSF)
    if routing != 'nsf':
        _abw_kw = _mdn_analytic_bw_kwargs(_n_cauchy)
        mdl = MDNModel(n_gaussians=_n_gaussians, n_cauchy=_n_cauchy,
                       **_abw_kw).to(dev)
        _init_mdn_from_tracks(mdl, _tracks, _analyses, target=target, verbose=False)
        m_p, nll_p = _train_probe(mdl, 'mdn', mdn_loss, dataset, source_data,
                                   train_idx, probe_epochs,
                                   py_pr, pc_pr, pw_pr, ae_pr, dev, verbose)
        probe_results['mdn'] = (m_p, nll_p)

    # NSF probe (skip only if comment/topology hard-routes to MDN via single_bw/squarks)
    if routing != 'mdn' or topo_route == 'race':
        mdl = NSFModel(context_dim=ctx_dim, num_bins=num_bins, tail_bound=tail_bound).to(dev)
        m_p, nll_p = _train_probe(mdl, 'nsf', nsf_nll_loss, dataset, source_data,
                                   train_idx, probe_epochs,
                                   py_pr, pc_pr, pw_pr, ae_pr, dev, verbose)
        probe_results['nsf'] = (m_p, nll_p)

    # EBM probe — always run (analytic_bw required for narrow-peak resolution)
    mdl = EBMModel(context_dim=ctx_dim, y_hi=tail_bound, **_abw_kw_ebm).to(dev)
    _init_ebm_from_tracks(mdl, _tracks, _analyses, verbose=False)
    m_p, nll_p = _train_probe(mdl, 'ebm', ebm_nll_loss, dataset, source_data,
                               train_idx, probe_epochs,
                               py_pr, pc_pr, pw_pr, ae_pr, dev, verbose)
    probe_results['ebm'] = (m_p, nll_p)

    # MDN+g probe — always run
    mdl = MDNCorrModel(in_dim=ctx_dim, y_hi=tail_bound,
                       **_abw_kw_ebm).to(dev)
    _ratio_tracks_probe = build_ratio_tracks(_tracks, _analyses,
                                             source_data=source_data, verbose=False)
    _init_mdn_from_tracks(mdl._mdn, _tracks, _analyses,
                          target=target, ratio_tracks=_ratio_tracks_probe,
                          verbose=False)
    m_p, nll_p = _train_probe(mdl, 'mdn+g', mdn_corr_loss, dataset, source_data,
                               train_idx, probe_epochs,
                               py_pr, pc_pr, pw_pr, ae_pr, dev, verbose)
    probe_results['mdn+g'] = (m_p, nll_p)

    # Winner from probe (best validation NLL across all candidates)
    winner = min(probe_results, key=lambda k: probe_results[k][1])
    if verbose:
        for k, (_, nll) in sorted(probe_results.items(), key=lambda x: x[1]):
            print(f'[SELECT] {k.upper()} probe NLL={nll:.4f}{" ← winner" if k == winner else ""}')

    mdn_nll = probe_results.get('mdn', (None, float('inf')))[1]
    nsf_nll = probe_results.get('nsf', (None, float('inf')))[1]

    # Override: substantial negative interference → prefer flow-based model if it's competitive
    probe_ran      = routing == 'race' and nsf_nll < float('inf')
    nsf_clearly_better = (not probe_ran) or (nsf_nll < mdn_nll - 0.01)
    if winner == 'mdn' and substantial_neg and nsf_clearly_better:
        if verbose:
            print(f'[SELECT] Override: substantial neg interference '
                  f'({neg_pts}/{n_grid} = {neg_frac:.0%}) → NSF')
        winner = 'nsf'
    elif winner == 'mdn' and substantial_neg and not nsf_clearly_better:
        if verbose:
            print(f'[SELECT] Neg interference substantial but NSF probe not better '
                  f'(NSF={nsf_nll:.4f} vs MDN={mdn_nll:.4f}) — keeping MDN')

    # Enrich NSF prior with MDN-extracted knowledge
    mdn_probe_model = probe_results.get('mdn', (None,))[0]
    if winner == 'nsf' and mdn_probe_model is not None:
        mdn_seed = _extract_mdn_prior(mdn_probe_model, source_data)
        if mdn_seed is not None:
            existing = {c['type'] for c in _nsf_prior['components']}
            extra = [c for c in mdn_seed['components'] if c['type'] not in existing]
            if extra:
                _nsf_prior = dict(_nsf_prior, components=_nsf_prior['components'] + extra)
                if verbose:
                    print('[SELECT] NSF prior enriched with MDN-extracted structure')

    if verbose:
        print(f'[SELECT] Full training: {winner.upper()}')

    if winner == 'mdn':
        m, _ = _call_full_mdn()
        return m, 'mdn'
    else:
        m, mt = _run(winner, epochs)
        return m, mt


# ─────────────────────────────────────────────────────────────────────────────
# sample_model — unified sampler for all model types
# ─────────────────────────────────────────────────────────────────────────────

def sample_model(model, model_type, target, n_samples=500_000):
    """Draw mHH [GeV] samples from any trained model.

    NSF   : batch sampling from flow, y < 0 rejected.
    MDN   : skew-t mixture inverse CDF via Rosenblatt transform.
    EBM   : inverse CDF on GL grid (fast 1-D integration, no MCMC).
    MDN+g : inverse CDF on dense y-grid using pdf_on_grid.
    """
    T = cfg.MHH_THRESHOLD
    model.eval()
    x_enc = encode_ctx(*[float(v) for v in target])

    if model_type == 'nsf':
        eager = getattr(model, '_orig_mod', model)
        ctx   = torch.tensor([x_enc], dtype=torch.float32, device=torch.device('cpu'))
        chunks = []
        eager.cpu()
        with torch.no_grad():
            drawn = 0
            while drawn < n_samples:
                n_this = min(50_000, n_samples - drawn)
                y_raw  = eager.sample(n_this, context=ctx)
                chunks.append(y_raw)
                drawn += n_this
        eager.to(cfg.DEVICE)
        y_all = torch.cat(chunks).squeeze(1).numpy()
        y_all = y_all[(y_all >= 0.0) & (y_all <= eager.tail_bound)]
        return inverse_transform_mhh(y_all, T)

    elif model_type == 'mdn':
        with torch.no_grad():
            x = torch.tensor([x_enc], dtype=torch.float32, device=cfg.DEVICE)
            pi, mu, sigma, alpha, nu = model(x)
            pi = pi.cpu().numpy().flatten()
            mu = mu.cpu().numpy().flatten()
            sg = sigma.cpu().numpy().flatten()
            al = alpha.cpu().numpy().flatten()
            nv = nu.cpu().numpy().flatten()
        comps = np.random.choice(len(pi), size=n_samples, p=pi)
        mu_s, sg_s, al_s, nu_s = mu[comps], sg[comps], al[comps], nv[comps]
        delta = al_s / np.sqrt(1.0 + al_s**2)
        U = np.abs(np.random.normal(size=n_samples))
        V = np.random.normal(size=n_samples)
        z = delta * U + np.sqrt(np.maximum(1.0 - delta**2, 0.0)) * V
        G = np.random.gamma(shape=nu_s / 2.0, scale=2.0 / nu_s)
        s = mu_s + sg_s * z / np.sqrt(np.maximum(G, 1e-8))
        return inverse_transform_mhh(s, T)

    elif model_type in ('ebm', 'mdn+g'):
        # Inverse-CDF via fine grid: compute PDF → cumulative → interpolate
        eager = getattr(model, '_orig_mod', model)
        if hasattr(eager, 'y_hi'):
            y_hi = eager.y_hi
        elif hasattr(eager, 'tail_bound'):
            y_hi = eager.tail_bound
        else:
            y_hi = 6.0
        y_grid  = np.linspace(0.0, y_hi, 4096)
        pdf_y   = eager.pdf_on_grid(x_enc, y_grid)
        pdf_y   = np.maximum(pdf_y, 0.0)
        dy      = y_grid[1] - y_grid[0]
        cdf     = np.cumsum(pdf_y) * dy
        cdf    /= max(cdf[-1], 1e-12)
        u       = np.random.uniform(0.0, 1.0, n_samples)
        y_samp  = np.interp(u, cdf, y_grid)
        return inverse_transform_mhh(y_samp, T)

    else:
        raise ValueError(f'Unknown model_type: {model_type}')
