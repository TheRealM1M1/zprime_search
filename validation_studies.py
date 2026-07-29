"""
Post-hoc validation studies for the Z' search — runs on saved scores only.

Reads search_region_scores.csv (written by DNN.py and BDT.py) and produces:
  PART A — quantitative score-vs-mass correlation + profile plots
  PART B — toy signal-injection study


Usage:
    python validation_studies.py                # DNN
    python validation_studies.py --score-col bdt_score --label BDT
"""

# USAGE:
#   python validation_studies.py --csv /path/to/search_region_scores.csv
# REQUIRES: pip install numpy pandas matplotlib scipy

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, ks_2samp
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# CONFIG — must match DNN.py/BDT.py exactly
SEARCH_MIN       = 250.0
SEARCH_BIN_WIDTH = 20
SEARCH_MASS_MAX  = 3500
SQRT_S           = 13000.0
MIN_BIN_EVENTS   = 10
VERY_HIGH_LO     = 0.95

SCORE_CATEGORIES = [
    ('low',       0.0,  0.5),
    ('medium',    0.5,  0.8),
    ('high',      0.8,  0.95),
    ('very_high', 0.95, 1.0),
]

INJECT_MASSES   = [500.0, 1000.0, 1500.0]
RESOLUTION_FRAC = 0.02
STRENGTH_GRID   = np.arange(0, 201, 10)
RNG_SEED        = 42

parser = argparse.ArgumentParser()
parser.add_argument('--csv',       default=None)
parser.add_argument('--score-col', default='dnn_score')
parser.add_argument('--label',     default='DNN')
parser.add_argument('--toys',      type=int, default=200,
                    help='pseudo-experiments per injected strength')
args = parser.parse_args()


def in_category(scores, lo, hi):
    """Category membership mask.

    Half-open [lo, hi) for every category except the top one (hi >= 1.0),
    which closes on hi (scores <= hi). This keeps float32-saturated sigmoid
    scores of exactly 1.0 inside the very-high category instead of silently
    dropping them. See the module docstring for why this matters.
    """
    top = scores <= hi if hi >= 1.0 else scores < hi
    return (scores >= lo) & top


def locate_csv(explicit=None, filename='search_region_scores.csv'):
    """Find the scores CSV regardless of the current working directory."""
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        raise FileNotFoundError(f"--csv path does not exist: {explicit}")

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filename,
        os.path.join('analysis_dnn', filename),
        os.path.join(here, filename),
        os.path.join(here, 'analysis_dnn', filename),
        os.path.join(here, '..', 'analysis_dnn', filename),
        os.path.join(here, '..', filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    for root_dir in (here, os.path.abspath(os.path.join(here, '..'))):
        for root, _dirs, files in os.walk(root_dir):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        f"  Working directory : {os.getcwd()}\n"
        f"  Script directory  : {here}\n"
        f"Pass the path explicitly, e.g.:\n"
        f"  python validation_studies.py --csv /full/path/to/{filename}"
    )


csv_path = locate_csv(args.csv)
print(f"Reading scores from: {csv_path}")

rng = np.random.default_rng(RNG_SEED)

df    = pd.read_csv(csv_path)
mass  = df['invariant_mass'].to_numpy()
score = df[args.score_col].to_numpy()

# Diagnostic: how many events saturate to exactly 1.0? These are precisely the
n_saturated = int(np.sum(score >= 1.0))
print(f"Loaded {len(score):,} search-region events; "
      f"{n_saturated:,} have score == 1.0 "
      f"({100 * n_saturated / max(len(score), 1):.1f}%) "
      f"and are retained by the closed top edge.")

bins    = np.arange(SEARCH_MIN, SEARCH_MASS_MAX + SEARCH_BIN_WIDTH, SEARCH_BIN_WIDTH)
centers = (bins[:-1] + bins[1:]) / 2


def bg_model(m, p0, p1, p2):
    """Same three-parameter background as DNN.py."""
    x = np.clip(m / SQRT_S, 1e-10, 1 - 1e-10)
    return p0 * ((1 - x) ** p1) * (m ** (-p2))


def asimov(S, B):
    if B <= 0 or S <= 0:
        return 0.0
    return np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))


def fit_background(counts):
    """Fit the 3-param model to binned counts. Returns (popt, chi2/ndf, mask)."""
    fit_mask = counts >= MIN_BIN_EVENTS
    m_fit    = centers[fit_mask]
    n_fit    = counts[fit_mask].astype(float)
    if fit_mask.sum() < 5:
        return None, None, fit_mask
    try:
        popt, _ = curve_fit(
            bg_model, m_fit, n_fit,
            p0=[n_fit[0] * (m_fit[0] ** 3), 5.0, 3.0],
            sigma=np.sqrt(n_fit + 1), absolute_sigma=True,
            maxfev=10000, bounds=([0, 0, 0], [np.inf, 50, 20]),
        )
    except Exception:
        return None, None, fit_mask
    chi2_ndf = np.sum(((n_fit - bg_model(m_fit, *popt)) /
                       np.sqrt(n_fit + 1)) ** 2) / max(len(m_fit) - 3, 1)
    return popt, chi2_ndf, fit_mask


print("=" * 68)
print(f"PART A: Score-vs-mass correlation ({args.label})")
print("=" * 68)

r_all,   p_r_all   = pearsonr(mass, score)
rho_all, p_rho_all = spearmanr(mass, score)
print(f"\nFull search region (n = {len(mass):,}):")
print(f"  Pearson  r   = {r_all:+.4f}   (p = {p_r_all:.3g})")
print(f"  Spearman rho = {rho_all:+.4f}   (p = {p_rho_all:.3g})")

vh = score >= VERY_HIGH_LO
r_vh,   _ = pearsonr(mass[vh], score[vh])
rho_vh, _ = spearmanr(mass[vh], score[vh])
print(f"\nVery-high category only (n = {vh.sum():,})  <- the fitted category:")
print(f"  Pearson  r   = {r_vh:+.4f}")
print(f"  Spearman rho = {rho_vh:+.4f}")

MIN_KS_EVENTS = 20
cat_counts = {name: int(in_category(score, lo, hi).sum())
              for name, lo, hi in SCORE_CATEGORIES}
ref_name = next((name for name, lo, hi in SCORE_CATEGORIES
                 if cat_counts[name] >= MIN_KS_EVENTS), None)

if ref_name is None:
    print("\nKS test: no category has enough events for a meaningful KS test")
else:
    ref_lo, ref_hi = next((lo, hi) for name, lo, hi in SCORE_CATEGORIES
                          if name == ref_name)
    mass_ref = mass[in_category(score, ref_lo, ref_hi)]
    print(f"\nKS test: mass shape in each category vs. '{ref_name}' category "
          f"(reference n = {len(mass_ref):,})")
    for name, lo, hi in SCORE_CATEGORIES:
        if name == ref_name:
            continue
        m_cat = mass[in_category(score, lo, hi)]
        if len(m_cat) < MIN_KS_EVENTS:
            print(f"  {name:10s} (n={len(m_cat):7,})  skipped — too few events")
            continue
        ks, p_ks = ks_2samp(m_cat, mass_ref)
        print(f"  {name:10s} (n={len(m_cat):7,})  KS = {ks:.4f}   p = {p_ks:.3g}")

# --- Profile + category-fraction plots (2 panels; the 2D hist saturates
prof_bins = np.linspace(SEARCH_MIN, 1500, 26)
idx       = np.digitize(mass, prof_bins) - 1

prof_x, prof_y, prof_e = [], [], []
for b in range(len(prof_bins) - 1):
    sel = idx == b
    if sel.sum() >= 10:
        prof_x.append((prof_bins[b] + prof_bins[b + 1]) / 2)
        prof_y.append(score[sel].mean())
        prof_e.append(score[sel].std() / np.sqrt(sel.sum()))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

axes[0].errorbar(prof_x, prof_y, yerr=prof_e, fmt='o-', color='crimson', ms=4)
axes[0].axhline(VERY_HIGH_LO, ls='--', c='k', lw=1, label='very-high threshold')
axes[0].set_xlabel('Invariant mass (GeV)')
axes[0].set_ylabel(f'Mean {args.label} score')
axes[0].set_title('Mean score vs. mass (profile)', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

for name, lo, hi in SCORE_CATEGORIES:
    fracs, xs = [], []
    for b in range(len(prof_bins) - 1):
        sel = idx == b
        if sel.sum() >= 10:
            xs.append((prof_bins[b] + prof_bins[b + 1]) / 2)
            fracs.append(in_category(score[sel], lo, hi).mean())
    axes[1].plot(xs, fracs, 'o-', ms=3, label=name)
axes[1].set_xlabel('Invariant mass (GeV)')
axes[1].set_ylabel('Fraction of events in category')
axes[1].set_title('Category fraction vs. mass', fontweight='bold')
axes[1].set_ylim(-0.03, 1.08)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'score_mass_correlation_{args.label.lower()}.png', dpi=300,
            bbox_inches='tight')
plt.close()
print(f"\n✓ Saved score_mass_correlation_{args.label.lower()}.png")


# PART B — TOY SIGNAL INJECTION (Asimov baseline, many toys per point)
print("\n" + "=" * 68)
print(f"PART B: Toy signal injection ({args.label})")
print("=" * 68)

# saturated 1.0 events are included (consistent with in_category's top edge).
n_obs, _ = np.histogram(mass[vh], bins=bins)
popt0, chi2_0, _ = fit_background(n_obs)
if popt0 is None:
    print(f"\nBaseline very-high fit did not converge: only "
          f"{int((n_obs >= MIN_BIN_EVENTS).sum())} bins have >= {MIN_BIN_EVENTS} "
          f"events (need >= 5). Skipping the injection study.")
    print("\nDone. Part A completed; Part B skipped for lack of statistics.")
    raise SystemExit(0)
print(f"\nBaseline very-high fit to DATA: chi2/ndf = {chi2_0:.2f}  "
      f"p0={popt0[0]:.3e}  p1={popt0[1]:.2f}  p2={popt0[2]:.2f}")
print("(Compare against the published fit to confirm this reproduces the pipeline.)")

b_exp = bg_model(centers, *popt0)
b_exp = np.maximum(b_exp, 0.0)

print(f"\nInjecting Gaussian bumps (sigma = {RESOLUTION_FRAC:.0%} of mass) onto the")
print(f"fitted background expectation; {args.toys} pseudo-experiments per point.")
print("Reporting MEDIAN expected local significance.\n")


def crossing(strengths, med_z, target):
    """First strength where the median curve reaches `target`, interpolated."""
    for i in range(1, len(med_z)):
        if med_z[i - 1] < target <= med_z[i]:
            x0, x1 = strengths[i - 1], strengths[i]
            y0, y1 = med_z[i - 1], med_z[i]
            if y1 == y0:
                return x1
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


results = []
for m_inj in INJECT_MASSES:
    sigma_inj = RESOLUTION_FRAC * m_inj
    shape = np.exp(-0.5 * ((centers - m_inj) / sigma_inj) ** 2)
    shape = shape / shape.sum()
    window = np.abs(centers - m_inj) <= 2 * sigma_inj

    med_z, lo_z, hi_z = [], [], []
    for n_sig in STRENGTH_GRID:
        expected = b_exp + n_sig * shape
        zs = np.empty(args.toys)
        for t in range(args.toys):
            pseudo = rng.poisson(expected)
            popt_i, _, _ = fit_background(pseudo)
            if popt_i is None:
                zs[t] = 0.0
                continue
            b_fit  = bg_model(centers, *popt_i)
            excess = pseudo[window] - b_fit[window]
            zs[t]  = asimov(max(excess.sum(), 0.0),
                            max(b_fit[window].sum(), 1e-9))
        med_z.append(np.median(zs))
        lo_z.append(np.percentile(zs, 16))
        hi_z.append(np.percentile(zs, 84))
        print(f"  m={m_inj:6.0f} GeV  N_sig={n_sig:4d}  "
              f"median z = {med_z[-1]:5.2f}", end='\r')

    med_z = np.array(med_z); lo_z = np.array(lo_z); hi_z = np.array(hi_z)
    z3 = crossing(STRENGTH_GRID, med_z, 3.0)
    z5 = crossing(STRENGTH_GRID, med_z, 5.0)
    b_win = b_exp[window].sum()
    print(f"  m_inj = {m_inj:6.0f} GeV | bkg in +/-2sigma window = {b_win:8.1f} | "
          f"3sigma at N_sig = {f'{z3:.0f}' if z3 else '>max':>6} | "
          f"5sigma at N_sig = {f'{z5:.0f}' if z5 else '>max':>6}")
    results.append((m_inj, med_z, lo_z, hi_z, z3, z5, window))

fig, axes = plt.subplots(1, len(INJECT_MASSES) + 1,
                         figsize=(5.2 * (len(INJECT_MASSES) + 1), 5))

for ax, (m_inj, med_z, lo_z, hi_z, z3, z5, window) in zip(axes[:-1], results):
    sigma_inj = RESOLUTION_FRAC * m_inj
    n_show    = z3 if z3 is not None else STRENGTH_GRID[-1]
    shape     = np.exp(-0.5 * ((centers - m_inj) / sigma_inj) ** 2)
    shape     = shape / shape.sum()
    pseudo    = rng.poisson(b_exp + n_show * shape)
    popt_i, _, _ = fit_background(pseudo)

    ax.errorbar(centers, pseudo, yerr=np.sqrt(np.maximum(pseudo, 1)),
                fmt='o', ms=3, color='crimson', alpha=0.7,
                label=f'Pseudo-data + {n_show:.0f} injected')
    ms = np.linspace(SEARCH_MIN, 2000, 400)
    ax.plot(ms, bg_model(ms, *popt0), 'k--', lw=1.5, label='Background expectation')
    if popt_i is not None:
        ax.plot(ms, bg_model(ms, *popt_i), 'b-', lw=1.5, label='Refit w/ signal')
    ax.axvline(m_inj, color='green', ls=':', lw=1.5)
    ax.set_yscale('log')
    ax.set_xlim(SEARCH_MIN, min(2000, m_inj * 2))
    ax.set_xlabel('Invariant mass (GeV)')
    ax.set_ylabel('Events / 20 GeV')
    ax.set_title(f'Injection at {m_inj:.0f} GeV (3$\\sigma$ yield)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

colors = ['tab:blue', 'tab:orange', 'tab:green']
for (m_inj, med_z, lo_z, hi_z, _, _, _), c in zip(results, colors):
    axes[-1].plot(STRENGTH_GRID, med_z, 'o-', ms=3, color=c,
                  label=f'{m_inj:.0f} GeV')
    axes[-1].fill_between(STRENGTH_GRID, lo_z, hi_z, color=c, alpha=0.18)
axes[-1].axhline(3, ls='--', c='orange', label='3$\\sigma$')
axes[-1].axhline(5, ls='--', c='red',    label='5$\\sigma$')
axes[-1].set_xlabel('Injected signal events')
axes[-1].set_ylabel('Median expected local significance')
axes[-1].set_title(f'Sensitivity vs. injected yield\n({args.toys} toys, 16–84% band)',
                   fontweight='bold')
axes[-1].legend(fontsize=8); axes[-1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'signal_injection_{args.label.lower()}.png', dpi=300,
            bbox_inches='tight')
plt.close()
print(f"\n✓ Saved signal_injection_{args.label.lower()}.png")
print("\nDone.")