# Interpolation of deconstructed MC events for Di-Higgs studies

Surrogate-model interpolation of BSM di-Higgs $m_{HH}$ distributions from deconstructed LHE Monte Carlo samples.

> S. Moretti, L. Panizzi, J. Sjölin, H. Waltari, *Deconstructing squark contributions to di-Higgs production at the LHC*, Phys. Rev. D **107** (2023) no.11, 115010, arXiv:2302.03401.  
> S. Moretti, L. Panizzi, J. Sjölin, H. Waltari, *Deconstructing resonant Higgs pair production at the LHC: effects of coloured and neutral scalars in the NMSSM test case*, Phys. Rev. D **112** (2025) no.5, 055005, arXiv:2506.09006.

---

## Installation

```bash
pip install torch nflows numpy matplotlib scikit-learn
```

Optional (GPU temperature monitoring):
```bash
pip install pynvml
```

---

## Quick start (no LHE access required)

The repository includes pre-computed cache files for the `S_13t` database. No LHE files or HPC access are needed:

```bash
git clone https://github.com/lcpnzz/InterpolationDeconstructedMC.git
cd InterpolationDeconstructedMC
python3 morph_ml.py S_13t 850 0.001
```

---

## Usage

```bash
python3 morph_ml.py <database> <p1> [p2 p3 p4] [OPTIONS]
```

### Examples

```bash
# Auto model selection, target-neighbour training
python3 morph_ml.py S_13t 850 0.001

# Force NSF, full-grid training
python3 morph_ml.py S_13t 850 0.001 --model nsf --fulltraining

# EFT baseline only (no training)
python3 morph_ml.py S_13t 850 0.001 --model eft

# Train all models and pick winner by chi2/ndf
python3 morph_ml.py S_13t 850 0.001 --trainallmodels

# Custom mHH range, verbose
python3 morph_ml.py S_13t 850 0.001 --mhh-range 250 2000 10 --verbose
```

### Key options

| Flag | Default | cription |
|------|---------|-------------|
| `--model` | auto | Force model: `mdn`, `nsf`, `ebm`, `mdn+g`, `eft`, `bdt` |
| `--fulltraining` | off | Train on all grid points |
| `--targettraining` | on | Train on neighbours of target only |
| `--mhh-range MIN MAX BIN` | `0 5000 50` | Histogram range and bin width [GeV] |
| `--epochs N` | 100/500 | Training epochs (`--laptop` mode / default) |
| `--retrain` | off | Delete cached model and retrain |
| `--trainallmodels` | off | Train all four models, pick by $\chi^2$/ndf |
| `--comment TEXT` | — | Free-text physics prior |
| `--gpu` / `--cpu` | auto | Device selection |
| `--laptop` | off | Use local data path |
| `--gpu-safety 0–5` | 0 | Thermal safety preset |
| `--verbose` | off | Verbose output |

### Models

| Model | cription | Handles dips | Physics init |
|-------|-------------|:---:|:---:|
| `mdn` | Skew-$t$ Mixture Density Network | ✗ | ✓ |
| `nsf` | Neural Spline Flow | ✓ | ✗ |
| `ebm` | Conditional Energy-Based Model (GL quadrature) | ✓ | ✓ |
| `mdn+g` | MDN with signed log-correction $\exp(g)$ | ✓ | ✓ |
| `eft` | Quadratic EFT morphing baseline (no training) | — | — |
| `bdt` | Gradient-boosted tree morphing baseline (no training) | — | — |

---

## Data format

LHE event files (`.lhe.gz`) must be organised as:

```
<DB_BASE>/hh_<database>_<param1>_<val1>_<param2>_<val2>/events.lhe.gz
```

`DB_BASE` is set in `core/config.py` (default: `/cfs/data/pg/SHIFT/diHiggs_Full/LHE`). Use `--laptop` to switch to a local path. If LHE files are absent the code reads automatically from `./<database>/LHE_pt_cache_thr<N>/`.

---

## Output

- `<db>/Models/model_<params>.pt` — saved model checkpoint
- `<db>/Models/mhh_samples_<params>.npz` — sampled $m_{HH}$ events
- `<db>/Plots/` — PDF plots of the interpolated distribution

---

## Code structure

```
core/
  config.py     — globals, coordinate transforms, grid discovery
  dataset.py    — LHE parsing, caching, Voronoi-weighted dataset
  features.py   — peak/dip detection, feature tracking, physics priors
  models.py     — MDN, NSF, EBM, MDN+g architectures
  training.py   — training loops, model selection, chi2 monitoring
  morphing.py   — EFT quadratic and BDT morphing baselines
  plotting.py   — all matplotlib output
morph_ml.py     — CLI entry point
```
