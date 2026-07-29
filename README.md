# Z' Search Using ATLAS Open Data Run 2 2015 proton-proton Collision Data

# ML-Assisted Dilepton Resonance-Search Workflow

Analysis code for *"Machine-Learning-Driven Search for a Z′ Boson in the Dilepton
Channel Using ATLAS Open Data"*. It reproduces the full workflow: data
collection from the CERN Open Data Portal, BDT and DNN training, background
fitting, the significance scan, and the validation studies.

Data: ATLAS Run 2 (2015) open data, DAOD-PHYSLITE format, CERN Open Data Portal
record 80000 (DOI 10.7483/OPENDATA.ATLAS.AOQL.8TT3), 3.2 fb⁻¹.

## Files and run order

1. **`data_collection.py`** — streams the ATLAS Open Data files over XRootD and writes
   one CSV of dilepton events per file range. Run it across the dataset (edit
   `START_FILE` / `END_FILE`, or set `END_FILE = 10049` for all files). Ctrl+C
   saves a checkpoint; rerun to resume.
2. **`DNN.py`** — trains the deep neural network, scores the blinded search
   region, fits the background in the very-high score category, and runs the
   Asimov significance scan. Reads the CSVs from step 1.
3. **`BDT.py`** — the XGBoost counterpart to `DNN.py`. same
   pipeline, same inputs, gradient-boosted decision tree instead of the MLP.
4. **`validation_studies.py`** — post-training/analysis, no retraining. Reads a saved
   `search_region_scores.csv` and produces the score–mass correlation study
   (Pearson / Spearman / KS plus the profile and category-fraction plots) and
   the toy signal-injection study.

Steps 2 and 3 each write their own output folder,
including a `search_region_scores.csv` that step 4 reads.

## What to edit before running

Every user-editable line is marked with an inline `EDIT:` comment.

- `main_csv.py`: `START_FILE`, `END_FILE`, `MAX_EVENTS`.
- `DNN.py` / `BDT.py`: the `data_files` list (paths to your CSVs
  from step 1), `OUTPUT_DIR`, `HDF5_DIR`. Delete the HDF5 cache folder if you
  change mass regions or features.
- `validation_studies.py`: pass options on the command line —
  ```
  python validation_studies.py                                   # DNN (default)
  python validation_studies.py --score-col bdt_score --label BDT # BDT
  python validation_studies.py --csv /path/to/search_region_scores.csv
  ```

## Requirements

```
pip install uproot awkward cernopendata-client          # data collection
pip install torch xgboost h5py scikit-learn scipy       # classifiers + studies
pip install numpy pandas matplotlib seaborn
```

Python 3.10+ recommended. A GPU is optional; `DNN.py` uses CUDA if available,
otherwise CPU.

## Key settings

- Random seed 42 (train/test split, XGBoost, injection RNG).
- Train/test split 80/20, stratified.
- 20 GeV bins; a bin enters the fit/scan only with >= 10 events, which restricts
  the scan to 250-1450 GeV.
- Mass regions: background 40-71.2, signal (Z-like proxy) 71.2-111.2, validation
  111.2-250, search > 250 GeV (hard cut at 3500 GeV).
- Background model: f(m) = p0 (1 - m/sqrt(s))^p1 * m^(-p2), sqrt(s) = 13000 GeV.
- Score categories: low [0, 0.5), medium [0.5, 0.8), high [0.8, 0.95),
  very_high [0.95, 1.0]. 

## Data

ATLAS Open Data: https://opendata.cern.ch/record/80000
