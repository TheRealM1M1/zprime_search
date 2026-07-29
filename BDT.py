"""
Z' Boson Search — Collider Analysis Pipeline
=============================================
Mass regions
------------
  Background  : 40  – 71.2  GeV   (training, label=0)
  Signal      : 71.2 – 111.2 GeV  (training, label=1)
  Validation  : 111.2 – 250  GeV  (evaluate only — never trained on)
  Search      : > 250         GeV  (completely blinded during training)

Features
--------
  Raw CSV columns only — no engineered quantities, no invariant mass:
  pt1, eta1, phi1, pt2, eta2, phi2,
  n_jets, leading_jet_pt, leading_jet_eta, leading_jet_phi, met

DELETE hdf5_cache_proper/ before running if you change mass regions or features.
"""

# STEP 2 (BDT). XGBoost counterpart to DNN.py: trains the BDT, scores the
# EDIT BEFORE RUNNING:
#   - data_files list below: point these paths at your main_csv.py output CSVs
#   - OUTPUT_DIR, HDF5_DIR (below): where results / cache are written
# REQUIRES: pip install xgboost h5py scikit-learn matplotlib seaborn scipy pandas numpy
# NOTE: delete the HDF5_DIR folder if you change mass regions or features.
# scores rarely saturate to exactly 1.0, so this is a no-op in practice here.


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import xgboost as xgb
import warnings
import os
import gc

try:
    import h5py
except ImportError:
    raise ImportError("pip install h5py")

warnings.filterwarnings('ignore', category=RuntimeWarning)

# CONFIGURATION

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

OUTPUT_DIR = "bdt_output"        # EDIT: output folder for this run
HDF5_DIR   = "hdf5_cache_bdt"      # EDIT: HDF5 cache folder (delete to rebuild)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HDF5_DIR,   exist_ok=True)

BG_MIN     = 40.0
BG_MAX     = 71.2
SIG_MIN    = 71.2
SIG_MAX    = 111.2
VAL_MIN    = 111.2
VAL_MAX    = 250.0
SEARCH_MIN = 250.0

SQRT_S = 13_000.0

BG_SAMPLE   = None
HDF5_CHUNK  = 50_000
SCORE_CHUNK = 100_000

SCORE_CATEGORIES = [
    ('low',       0.0,  0.5,  'gray'),
    ('medium',    0.5,  0.8,  'steelblue'),
    ('high',      0.8,  0.95, 'darkorange'),
    ('very_high', 0.95, 1.0,  'crimson'),
]

FIT_CATEGORIES = ['very_high']

def in_category(scores, lo, hi):
    # Half-open [lo, hi) for every category EXCEPT the top one (hi >= 1.0), which
    # closes on hi so sigmoid scores that saturate to exactly 1.0 in float32 are
    # kept in the fitted category instead of being silently dropped.
    top = scores <= hi if hi >= 1.0 else scores < hi
    return (scores >= lo) & top


SEARCH_BIN_WIDTH = 20
SEARCH_MASS_MAX  = 3500
MIN_BIN_EVENTS   = 10

print("=" * 80)
print("Z' BOSON SEARCH — COLLIDER ANALYSIS PIPELINE")
print("=" * 80)
print(f"  Background  : {BG_MIN}–{BG_MAX} GeV")
print(f"  Signal      : {SIG_MIN}–{SIG_MAX} GeV")
print(f"  Validation  : {VAL_MIN}–{VAL_MAX} GeV  (evaluate only)")
print(f"  Search      : >{SEARCH_MIN} GeV  (blinded during training)")
print("=" * 80)


dtypes = {
    'file_idx': 'int32',
    'pt1': 'float32', 'eta1': 'float32', 'phi1': 'float32',
    'pt2': 'float32', 'eta2': 'float32', 'phi2': 'float32',
    'invariant_mass': 'float32',
    'n_jets': 'int16',
    'leading_jet_pt': 'float32', 'leading_jet_eta': 'float32',
    'leading_jet_phi': 'float32',
    'met': 'float32'
}

numeric_columns = [
    'pt1', 'eta1', 'phi1', 'pt2', 'eta2', 'phi2',
    'invariant_mass', 'n_jets',
    'leading_jet_pt', 'leading_jet_eta', 'leading_jet_phi', 'met'
]

FEATURE_NAMES = [
    'pt1', 'eta1', 'phi1',
    'pt2', 'eta2', 'phi2',
    'n_jets', 'leading_jet_pt', 'leading_jet_eta', 'leading_jet_phi',
    'met'
]
N_FEATURES = len(FEATURE_NAMES)

# EDIT: point these at your main_csv.py output CSVs (paths relative to this script)
data_files = [
   "../file_name.csv",
]


def quality_cuts(df):
    n_start = len(df)
    df = df[df['invariant_mass'] < 3_500]
    df = df[df['eta1'].abs() <= 2.5]
    df = df[df['eta2'].abs() <= 2.5]
    has_jet  = df['n_jets'] > 0
    bad_jeta = has_jet & (df['leading_jet_eta'].abs() > 2.5)
    df = df[~bad_jeta]
    return df.reset_index(drop=True), n_start - len(df)


def extract_features(df):
    """Extract raw feature matrix and fill missing jet values with 0."""
    out = df.copy()
    out['leading_jet_pt']  = out['leading_jet_pt'].fillna(0)
    out['leading_jet_eta'] = out['leading_jet_eta'].fillna(0)
    out['leading_jet_phi'] = out['leading_jet_phi'].fillna(0)
    return out[FEATURE_NAMES].values.astype(np.float32)

# STEP 1: BUILD HDF5 CACHE

BG_H5  = os.path.join(HDF5_DIR, 'background.h5')
SIG_H5 = os.path.join(HDF5_DIR, 'signal.h5')
VAL_H5 = os.path.join(HDF5_DIR, 'validation.h5')
SR_H5  = os.path.join(HDF5_DIR, 'search.h5')


def init_h5_labeled(path, label):
    if not os.path.exists(path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', shape=(0, N_FEATURES),
                             maxshape=(None, N_FEATURES), dtype='float32',
                             chunks=(HDF5_CHUNK, N_FEATURES))
            f.create_dataset('y', shape=(0,), maxshape=(None,),
                             dtype='float32', chunks=(HDF5_CHUNK,))
            f.attrs['label']           = label
            f.attrs['processed_files'] = []
        print(f"  Created {path}")


def init_h5_mass(path):
    if not os.path.exists(path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', shape=(0, N_FEATURES),
                             maxshape=(None, N_FEATURES), dtype='float32',
                             chunks=(HDF5_CHUNK, N_FEATURES))
            f.create_dataset('mass', shape=(0,), maxshape=(None,),
                             dtype='float32', chunks=(HDF5_CHUNK,))
            f.attrs['processed_files'] = []
        print(f"  Created {path}")


def append_labeled(path, X_arr, label):
    if len(X_arr) == 0:
        return
    with h5py.File(path, 'a') as f:
        ds = f['X']; old = ds.shape[0]; new = old + len(X_arr)
        ds.resize(new, axis=0); ds[old:new] = X_arr
        dy = f['y']; dy.resize(new, axis=0); dy[old:new] = label


def append_mass(path, X_arr, mass_arr):
    if len(X_arr) == 0:
        return
    with h5py.File(path, 'a') as f:
        ds = f['X']; old = ds.shape[0]; new = old + len(X_arr)
        ds.resize(new, axis=0); ds[old:new] = X_arr
        dm = f['mass']; dm.resize(new, axis=0); dm[old:new] = mass_arr


def get_processed(path):
    if not os.path.exists(path):
        return set()
    with h5py.File(path, 'r') as f:
        return set(f.attrs.get('processed_files', []))


def mark_processed(path, fname):
    with h5py.File(path, 'a') as f:
        existing = list(f.attrs.get('processed_files', []))
        existing.append(fname)
        f.attrs['processed_files'] = existing


def h5_len(path, ds='X'):
    if not os.path.exists(path):
        return 0
    with h5py.File(path, 'r') as f:
        return f[ds].shape[0]


print("\nSTEP 1: Building HDF5 cache...")
init_h5_labeled(BG_H5,  label=0)
init_h5_labeled(SIG_H5, label=1)
init_h5_mass(VAL_H5)
init_h5_mass(SR_H5)

all_paths    = [BG_H5, SIG_H5, VAL_H5, SR_H5]
already_done = set.intersection(*[get_processed(p) for p in all_paths])

n_cached = 0
n_built  = 0

for fpath in data_files:
    fname = os.path.basename(fpath)

    if fname in already_done:
        n_cached += 1
        if n_cached <= 3:
            print(f"  [cached] {fname}")
        elif n_cached == 4:
            print(f"  [cached] ... (remaining not shown)")
        continue

    try:
        df = pd.read_csv(fpath, dtype=dtypes)
    except FileNotFoundError:
        print(f"  ✗ NOT FOUND: {fpath}")
        continue

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_columns)

    df, n_removed = quality_cuts(df)

    bg_df  = df[(df['invariant_mass'] >= BG_MIN)  & (df['invariant_mass'] <  BG_MAX)]
    sig_df = df[(df['invariant_mass'] >= SIG_MIN) & (df['invariant_mass'] <= SIG_MAX)]
    val_df = df[(df['invariant_mass'] >  VAL_MIN) & (df['invariant_mass'] <= VAL_MAX)]
    sr_df  = df[df['invariant_mass']  >  SEARCH_MIN]

    append_labeled(BG_H5,  extract_features(bg_df),  0)
    append_labeled(SIG_H5, extract_features(sig_df), 1)
    append_mass(VAL_H5, extract_features(val_df),
                val_df['invariant_mass'].values.astype(np.float32))
    append_mass(SR_H5,  extract_features(sr_df),
                sr_df['invariant_mass'].values.astype(np.float32))

    for p in all_paths:
        mark_processed(p, fname)

    n_built += 1
    print(f"  ✓ {fname}  bg={len(bg_df):,}  sig={len(sig_df):,}  "
          f"val={len(val_df):,}  sr={len(sr_df):,}  "
          f"(removed {n_removed:,})")

    del df, bg_df, sig_df, val_df, sr_df
    gc.collect()

total_bg  = h5_len(BG_H5)
total_sig = h5_len(SIG_H5)
total_val = h5_len(VAL_H5)
total_sr  = h5_len(SR_H5)

print(f"\n  HDF5 summary:")
print(f"    Background : {total_bg:,}  → {BG_H5}")
print(f"    Signal     : {total_sig:,}  → {SIG_H5}")
print(f"    Validation : {total_val:,}  → {VAL_H5}")
print(f"    Search     : {total_sr:,}   → {SR_H5}")
print(f"  {n_cached} files cached, {n_built} newly built")

# STEP 1b: INVARIANT MASS OVERVIEW PLOTS

print("\nSTEP 1b: Generating invariant mass overview plots...")

SAMPLE_MAX = 500_000

with h5py.File(VAL_H5, 'r') as f:
    val_mass_plot = f['mass'][:SAMPLE_MAX]
with h5py.File(SR_H5, 'r') as f:
    sr_mass_plot  = f['mass'][:SAMPLE_MAX]

mass_max_plot = min(3000.0, float(sr_mass_plot.max()) if len(sr_mass_plot) > 0 else 3000.0)
bins_full     = np.linspace(BG_MIN, mass_max_plot, 300)
centers_full  = (bins_full[:-1] + bins_full[1:]) / 2
bin_w         = bins_full[1] - bins_full[0]

n_val_full, _ = np.histogram(val_mass_plot, bins=bins_full)
n_sr_full,  _ = np.histogram(sr_mass_plot,  bins=bins_full)

# ── Full-range plot (log + linear) ──────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

for ax_idx, ax in enumerate(axes):
    ax.bar(centers_full, n_val_full, width=bin_w * 0.9,
           color='#55A868', alpha=0.75, label='Validation (111.2–200 GeV)')
    ax.bar(centers_full, n_sr_full,  width=bin_w * 0.9,
           color='#C44E52', alpha=0.75, label='Search (>200 GeV)')

    ax.axvspan(BG_MIN,  BG_MAX,  alpha=0.15, color='#4C72B0',
               label=f'Background ({BG_MIN}–{BG_MAX} GeV)')
    ax.axvspan(SIG_MIN, SIG_MAX, alpha=0.15, color='#DD8452',
               label=f'Z-peak / Signal ({SIG_MIN}–{SIG_MAX} GeV)')

    # Boundary lines
    boundaries = [
        (BG_MAX,     '#4C72B0', 'BG/Signal'),
        (SIG_MAX,    '#DD8452', 'Signal/Validation'),
        (SEARCH_MIN, '#C44E52', 'Search floor'),
    ]
    ylim_top = ax.get_ylim()[1] if ax_idx == 1 else None
    for bval, bcol, blabel in boundaries:
        ax.axvline(bval, color=bcol, linestyle='--', lw=1.5, alpha=0.85)

    ax.set_xlabel('Invariant Mass (GeV)', fontsize=12)
    ax.set_ylabel('Events / bin', fontsize=12)
    ax.set_xlim(BG_MIN, mass_max_plot)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    if ax_idx == 0:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        ax.set_title('Invariant Mass — Full Range (Log Scale)',
                     fontsize=13, fontweight='bold')
    else:
        ax.set_title('Invariant Mass — Full Range (Linear Scale)',
                     fontsize=13, fontweight='bold')

for ax in axes:
    y_top = ax.get_ylim()[1]
    for bval, bcol, blabel in boundaries:
        ax.text(bval + mass_max_plot * 0.005, y_top * 0.5, blabel,
                rotation=90, fontsize=7, color=bcol, va='top', alpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'invariant_mass_overview.png'),
            dpi=300, bbox_inches='tight')
plt.close()

zoom_max  = 400.0
bins_zoom = np.linspace(BG_MIN, zoom_max, 150)
centers_zoom = (bins_zoom[:-1] + bins_zoom[1:]) / 2
bin_w_zoom   = bins_zoom[1] - bins_zoom[0]

n_val_zoom, _ = np.histogram(val_mass_plot[val_mass_plot <= zoom_max], bins=bins_zoom)
n_sr_zoom,  _ = np.histogram(sr_mass_plot[ sr_mass_plot <= zoom_max],  bins=bins_zoom)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax_idx, ax in enumerate(axes):
    ax.bar(centers_zoom, n_val_zoom, width=bin_w_zoom * 0.9,
           color='#55A868', alpha=0.8, label='Validation (111.2–200 GeV)')
    ax.bar(centers_zoom, n_sr_zoom,  width=bin_w_zoom * 0.9,
           color='#C44E52', alpha=0.8, label='Search (>200 GeV)')
    ax.axvspan(BG_MIN,  BG_MAX,  alpha=0.15, color='#4C72B0',
               label='Background region')
    ax.axvspan(SIG_MIN, SIG_MAX, alpha=0.15, color='#DD8452',
               label='Z-peak / Signal region')
    for bval, bcol in [(BG_MAX, '#4C72B0'), (SIG_MAX, '#DD8452'),
                        (SEARCH_MIN, '#C44E52')]:
        ax.axvline(bval, color=bcol, linestyle='--', lw=1.5, alpha=0.85)
    ax.set_xlabel('Invariant Mass (GeV)', fontsize=12)
    ax.set_ylabel('Events / bin', fontsize=12)
    ax.set_xlim(BG_MIN, zoom_max)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if ax_idx == 0:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        ax.set_title('Z Peak Region — Log Scale', fontsize=12, fontweight='bold')
    else:
        ax.set_title('Z Peak Region — Linear Scale', fontsize=12, fontweight='bold')

plt.suptitle('Invariant Mass — Zoom on Z Peak and Surrounding Regions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'invariant_mass_zpeak_zoom.png'),
            dpi=300, bbox_inches='tight')
plt.close()

del val_mass_plot, sr_mass_plot
gc.collect()
print(f"✓ Invariant mass plots saved")
print(f"  → {OUTPUT_DIR}/invariant_mass_overview.png")
print(f"  → {OUTPUT_DIR}/invariant_mass_zpeak_zoom.png")

# STEP 2: LOAD TRAINING DATA

print("\n" + "=" * 60)
print("STEP 2: Loading training data from HDF5")
print("=" * 60)

bg_limit = min(BG_SAMPLE, total_bg) if BG_SAMPLE else total_bg

with h5py.File(SIG_H5, 'r') as f:
    X_sig = f['X'][:]
    y_sig = f['y'][:]
print(f"  Signal     : {len(X_sig):,}")

with h5py.File(BG_H5, 'r') as f:
    X_bg = f['X'][:bg_limit]
    y_bg = f['y'][:bg_limit]
print(f"  Background : {len(X_bg):,} / {total_bg:,}")

X_all = np.vstack([X_bg, X_sig])
y_all = np.concatenate([y_bg, y_sig])
del X_bg, y_bg, X_sig, y_sig
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
del X_all, y_all
gc.collect()

scale_pos_weight = total_bg / (total_sig + 1e-10)
print(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

# STEP 3: TRAIN BDT

print("\n" + "=" * 60)
print("STEP 3: Training BDT")
print("=" * 60)
print(f"  Features ({N_FEATURES}): {', '.join(FEATURE_NAMES)}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=1.0,
    reg_alpha=0.1,
    reg_lambda=5.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("✓ Training complete")
model.save_model(os.path.join(OUTPUT_DIR, 'zprime_bdt_model.json'))

# STEP 4: EVALUATE ON TEST SET

print("\n" + "=" * 60)
print("STEP 4: Evaluation on held-out test set")
print("=" * 60)

train_scores = model.predict_proba(X_train)[:, 1]
test_scores  = model.predict_proba(X_test)[:,  1]

fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc = auc(fpr, tpr)
print(f"  ROC AUC : {roc_auc:.4f}")

y_pred = (test_scores > 0.5).astype(int)
print(classification_report(y_test, y_pred,
                             target_names=['Background', 'Signal']))

plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1], 'navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=13, fontweight='bold')
plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 1, 51)
for scores_h, labels_h, style, lbl in [
    (train_scores, y_train, '-',  'Train'),
    (test_scores,  y_test,  '--', 'Test'),
]:
    ax.hist(scores_h[labels_h == 0], bins=bins, histtype='step',
            density=True, linestyle=style, color='royalblue',
            lw=2, label=f'BG {lbl}')
    ax.hist(scores_h[labels_h == 1], bins=bins, histtype='step',
            density=True, linestyle=style, color='tomato',
            lw=2, label=f'Sig {lbl}')
ax.set_xlabel('BDT Score'); ax.set_ylabel('Density')
ax.set_title('Overtraining Check — Train vs Test', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overtraining_check.png'),
            dpi=300, bbox_inches='tight')
plt.close()

feat_imp = pd.DataFrame({
    'feature':    FEATURE_NAMES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=feat_imp, x='importance', y='feature', 
            hue='feature', palette='viridis', legend=False)
plt.title('Feature Importance', fontweight='bold')
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("✓ Plots saved")
print("Top features:")
for _, row in feat_imp.head().iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

del X_train, y_train
gc.collect()

# STEP 5: SCORE VALIDATION REGION  (111.2–130 GeV)

print("\n" + "=" * 60)
print(f"STEP 5: Scoring validation region ({VAL_MIN}–{VAL_MAX} GeV)")
print("=" * 60)

val_scores_all = []
val_masses_all = []

with h5py.File(VAL_H5, 'r') as f:
    X_ds    = f['X']
    mass_ds = f['mass']
    n_total = X_ds.shape[0]
    for start in range(0, n_total, SCORE_CHUNK):
        end = min(start + SCORE_CHUNK, n_total)
        val_scores_all.append(model.predict_proba(X_ds[start:end])[:, 1])
        val_masses_all.append(mass_ds[start:end])
        print(f"  Scored {end:,} / {n_total:,}", end='\r')

print()
val_scores = np.concatenate(val_scores_all)
val_masses = np.concatenate(val_masses_all)
del val_scores_all, val_masses_all
gc.collect()

print(f"  Validation events : {len(val_scores):,}")
print(f"  Mean BDT score    : {val_scores.mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(val_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('BDT Score'); axes[0].set_ylabel('Events')
axes[0].set_title(f'BDT Score — Validation Region ({VAL_MIN}–{VAL_MAX} GeV)',
                  fontweight='bold')
axes[0].grid(alpha=0.3)

bins_val = np.linspace(VAL_MIN, VAL_MAX, 20)
for cat_name, lo, hi, color in SCORE_CATEGORIES:
    mask = in_category(val_scores, lo, hi)
    if mask.sum() > 0:
        axes[1].hist(val_masses[mask], bins=bins_val, alpha=0.6,
                     color=color, label=f'{cat_name} ({mask.sum():,})',
                     edgecolor='none')
axes[1].set_xlabel('Invariant Mass (GeV)'); axes[1].set_ylabel('Events')
axes[1].set_title('Validation Region Mass by Score Category', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'validation_region.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Validation region plot saved")

# STEP 6: SCORE SEARCH REGION  (> 130 GeV)

print("\n" + "=" * 60)
print(f"STEP 6: Scoring search region (>{SEARCH_MIN} GeV)  [BLIND OPEN]")
print("=" * 60)

sr_scores_all = []
sr_masses_all = []

with h5py.File(SR_H5, 'r') as f:
    X_ds    = f['X']
    mass_ds = f['mass']
    n_total = X_ds.shape[0]
    for start in range(0, n_total, SCORE_CHUNK):
        end = min(start + SCORE_CHUNK, n_total)
        sr_scores_all.append(model.predict_proba(X_ds[start:end])[:, 1])
        sr_masses_all.append(mass_ds[start:end])
        print(f"  Scored {end:,} / {n_total:,}", end='\r')

print()
sr_scores = np.concatenate(sr_scores_all)
sr_masses = np.concatenate(sr_masses_all)
del sr_scores_all, sr_masses_all, X_test, y_test
gc.collect()

sr_df = pd.DataFrame({'invariant_mass': sr_masses, 'bdt_score': sr_scores})
sr_df.to_csv(os.path.join(OUTPUT_DIR, 'search_region_scores.csv'), index=False)

print(f"  Search region events : {len(sr_df):,}")
print(f"  Mean BDT score       : {sr_scores.mean():.3f}")

# STEP 7: MASS SPECTRA PER SCORE CATEGORY

print("\n" + "=" * 60)
print("STEP 7: Mass spectra per score category")
print("=" * 60)

bins_sr = np.arange(SEARCH_MIN, SEARCH_MASS_MAX + SEARCH_BIN_WIDTH,
                    SEARCH_BIN_WIDTH)
centers = (bins_sr[:-1] + bins_sr[1:]) / 2

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for idx, (cat_name, lo, hi, color) in enumerate(SCORE_CATEGORIES):
    ax   = axes[idx // 2, idx % 2]
    mask = in_category(sr_scores, lo, hi)
    n_cat, _ = np.histogram(sr_masses[mask], bins=bins_sr)
    ax.bar(centers, n_cat, width=SEARCH_BIN_WIDTH * 0.9,
           color=color, alpha=0.7, edgecolor='none',
           label=f'Score {lo}–{hi}  ({mask.sum():,} events)')
    ax.set_yscale('log')
    ax.set_xlabel('Invariant Mass (GeV)', fontsize=11)
    ax.set_ylabel('Events / 20 GeV',      fontsize=11)
    ax.set_title(f'{cat_name.replace("_"," ").title()} purity',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(SEARCH_MIN, min(SEARCH_MASS_MAX, sr_masses.max()))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle(f"Invariant Mass Spectra per BDT Score Category (Search Region >{SEARCH_MIN} GeV)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mass_spectra_categories.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Mass spectra saved")

# STEP 8: BACKGROUND FIT

print("\n" + "=" * 60)
print("STEP 8: Background fitting")
print("=" * 60)

def bg_model(m, p0, p1, p2):
    x = np.clip(m / SQRT_S, 1e-10, 1 - 1e-10)
    return p0 * ((1 - x) ** p1) * (m ** (-p2))

def asimov_significance(S, B):
    if B <= 0 or S <= 0:
        return 0.0
    return np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))

fit_results = {}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for idx, (cat_name, lo, hi, color) in enumerate(SCORE_CATEGORIES):
    ax = axes[idx // 2, idx % 2]
    if cat_name not in FIT_CATEGORIES:
        mask = in_category(sr_scores, lo, hi)
        n_obs, _ = np.histogram(sr_masses[mask], bins=bins_sr)
        ax.errorbar(centers, n_obs, yerr=np.sqrt(np.maximum(n_obs, 1)),
                    fmt='o', color=color, markersize=3, alpha=0.7, label='Data')
        ax.set_yscale('log')
        ax.set_xlabel('Invariant Mass (GeV)', fontsize=11)
        ax.set_ylabel('Events / 20 GeV', fontsize=11)
        ax.set_title(f'{cat_name.replace("_"," ").title()} (score {lo}–{hi}) — not fitted',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(SEARCH_MIN, min(SEARCH_MASS_MAX, sr_masses.max()))
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        print(f"  [{cat_name:10s}]  skipped (not in FIT_CATEGORIES)")
        continue
    mask = in_category(sr_scores, lo, hi)
    n_obs, _ = np.histogram(sr_masses[mask], bins=bins_sr)

    fit_mask = n_obs >= MIN_BIN_EVENTS
    m_fit    = centers[fit_mask]
    n_fit    = n_obs[fit_mask].astype(float)

    ax.errorbar(centers, n_obs, yerr=np.sqrt(np.maximum(n_obs, 1)),
                fmt='o', color=color, markersize=3, alpha=0.7, label='Data')

    if fit_mask.sum() >= 5:
        try:
            p0_init = n_fit[0] * (m_fit[0] ** 3)
            popt, _ = curve_fit(
                bg_model, m_fit, n_fit,
                p0=[p0_init, 5.0, 3.0],
                sigma=np.sqrt(n_fit + 1),
                absolute_sigma=True,
                maxfev=10000,
                bounds=([0, 0, 0], [np.inf, 50, 20])
            )
            m_smooth  = np.linspace(m_fit[0], centers[-1], 500)
            bg_smooth = bg_model(m_smooth, *popt)
            ax.plot(m_smooth, bg_smooth, 'k-', lw=2, label='BG fit')
            ax.fill_between(m_smooth, bg_smooth * 0.9, bg_smooth * 1.1,
                            alpha=0.2, color='black', label='±10% band')

            bg_at_data = bg_model(m_fit, *popt)
            chi2_ndf   = np.sum(((n_fit - bg_at_data) /
                                  np.sqrt(n_fit + 1)) ** 2) / max(len(m_fit)-3, 1)

            ax.text(0.97, 0.97, f'χ²/ndf = {chi2_ndf:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            fit_results[cat_name] = {
                'popt': popt, 'n_counts': n_obs,
                'chi2_ndf': chi2_ndf, 'lo': lo, 'hi': hi, 'color': color
            }
            print(f"  [{cat_name:10s}]  χ²/ndf = {chi2_ndf:.2f}  "
                  f"p0={popt[0]:.2e}  p1={popt[1]:.2f}  p2={popt[2]:.2f}")
        except Exception as e:
            print(f"  [{cat_name:10s}]  Fit failed: {e}")
    else:
        print(f"  [{cat_name:10s}]  Too few bins — skipping")

    ax.set_yscale('log')
    ax.set_xlabel('Invariant Mass (GeV)', fontsize=11)
    ax.set_ylabel('Events / 20 GeV',      fontsize=11)
    ax.set_title(f'{cat_name.replace("_"," ").title()} (score {lo}–{hi})',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(SEARCH_MIN, min(SEARCH_MASS_MAX, sr_masses.max()))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle("Mass Spectra + Background Fit", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'background_fits.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Background fit plot saved")

# STEP 9: SIGNIFICANCE SCAN  (Asimov formula)

print("\n" + "=" * 60)
print("STEP 9: Significance scan (Asimov formula)")
print("=" * 60)

all_candidates = []
for cat_name, result in fit_results.items():
    popt    = result['popt']
    n_obs   = result['n_counts']
    lo, hi  = result['lo'], result['hi']
    bg_pred = bg_model(centers, *popt)

    for c, obs, bg in zip(centers, n_obs, bg_pred):
        if obs < MIN_BIN_EVENTS:
            continue
        excess = obs - bg
        Z_A    = asimov_significance(max(excess, 0), bg)
        sign   = Z_A if excess >= 0 else -asimov_significance(max(-excess, 0), obs)
        all_candidates.append({
            'category':     cat_name,
            'score_lo':     lo,
            'score_hi':     hi,
            'mass_center':  c,
            'observed':     int(obs),
            'expected_bg':  round(bg, 1),
            'excess':       round(excess, 1),
            'significance': round(sign, 3),
        })

candidates_df = pd.DataFrame(all_candidates)

if len(candidates_df) > 0:
    sig_df = candidates_df.sort_values('significance', ascending=False)
    sig_df.to_csv(os.path.join(OUTPUT_DIR, 'zprime_candidates.csv'), index=False)

    print(f"\nTop candidates:")
    print(f"  {'Category':12s}  {'Mass':>8}  {'Obs':>7}  {'Exp BG':>9}  "
          f"{'Excess':>8}  {'Z_A':>6}")
    print("  " + "-" * 60)
    for _, row in sig_df.head(15).iterrows():
        print(f"  {row['category']:12s}  {row['mass_center']:>8.1f}  "
              f"{row['observed']:>7d}  {row['expected_bg']:>9.1f}  "
              f"{row['excess']:>8.1f}  {row['significance']:>6.2f}σ")
else:
    print("  No candidates — check background fits.")
    sig_df = pd.DataFrame()

if len(candidates_df) > 0:
    fig, axes = plt.subplots(len(fit_results), 1,
                             figsize=(16, 4 * len(fit_results)), sharex=True)
    if len(fit_results) == 1:
        axes = [axes]

    for ax, (cat_name, result) in zip(axes, fit_results.items()):
        cat_rows   = candidates_df[candidates_df['category'] == cat_name]
        bar_colors = ['darkred' if s > 3 else 'darkorange' if s > 2
                      else result['color'] if s > 0 else 'steelblue'
                      for s in cat_rows['significance']]
        ax.bar(cat_rows['mass_center'], cat_rows['significance'],
               width=SEARCH_BIN_WIDTH * 0.9, color=bar_colors, alpha=0.8)
        ax.axhline(0,  color='black',  lw=1)
        ax.axhline(2,  color='orange', lw=1.5, linestyle='--',
                   alpha=0.7, label='2σ')
        ax.axhline(3,  color='red',    lw=1.5, linestyle='--',
                   alpha=0.7, label='3σ')
        ax.axhline(-2, color='steelblue', lw=1, linestyle=':', alpha=0.5)
        ax.set_ylabel(f'Z_A  [{cat_name}]', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ymax = max(5, cat_rows['significance'].max() + 1)
        ax.set_ylim(-5, ymax)

    axes[-1].set_xlabel('Invariant Mass (GeV)', fontsize=12)
    plt.suptitle(f"Asimov Significance per Mass Bin (Search Region >{SEARCH_MIN} GeV)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'significance_map.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Significance map saved")

# STEP 10: SUMMARY

with open(os.path.join(OUTPUT_DIR, 'analysis_summary.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Z' BOSON SEARCH — COLLIDER ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Mass regions:\n")
    f.write(f"  Background  : {BG_MIN}–{BG_MAX} GeV\n")
    f.write(f"  Signal      : {SIG_MIN}–{SIG_MAX} GeV\n")
    f.write(f"  Validation  : {VAL_MIN}–{VAL_MAX} GeV\n")
    f.write(f"  Search      : >{SEARCH_MIN} GeV\n\n")
    f.write(f"Features ({N_FEATURES}): {', '.join(FEATURE_NAMES)}\n\n")
    f.write(f"BG events        : {bg_limit:,} / {total_bg:,}\n")
    f.write(f"Signal events    : {total_sig:,}\n")
    f.write(f"Validation events: {total_val:,}\n")
    f.write(f"Search events    : {total_sr:,}\n")
    f.write(f"Test ROC AUC     : {roc_auc:.4f}\n\n")
    f.write("Background fit χ²/ndf:\n")
    for cat_name, res in fit_results.items():
        f.write(f"  {cat_name:12s}: {res['chi2_ndf']:.2f}\n")
    f.write("\nTop Z' candidates (Asimov):\n")
    if len(sig_df) > 0:
        for _, row in sig_df.head(10).iterrows():
            f.write(f"  {row['mass_center']:.1f} GeV | {row['category']:12s} | "
                    f"obs={row['observed']} bg={row['expected_bg']:.1f} | "
                    f"{row['significance']:.2f}σ\n")

print(f"\n✓ Summary → {OUTPUT_DIR}/analysis_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nResults → {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  📊 invariant_mass_overview.png")
print("  📊 invariant_mass_zpeak_zoom.png")
print("  📊 roc_curve.png")
print("  📊 overtraining_check.png")
print("  📊 feature_importance.png")
print("  📊 validation_region.png")
print("  📊 mass_spectra_categories.png")
print("  📊 background_fits.png")
print("  📊 significance_map.png")
print("  📄 zprime_candidates.csv")
print("  📄 search_region_scores.csv")
print("  📄 analysis_summary.txt")
print("  🤖 zprime_bdt_model.json")
print("=" * 80)
