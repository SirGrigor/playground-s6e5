# Playground S6E5 — Predicting F1 Pit Stops

Binary classification, target = `PitNextLap`. Metric = ROC-AUC.
Deadline 2026-05-31. Started 2026-05-11.

## Structure

```
playground-s6e5/
├── data/
│   ├── raw/        # competition files — gitignored
│   ├── external/   # f1_strategy_dataset_v4.csv — gitignored
│   └── splits/     # holdout row IDs — TRACKED (the methodology)
├── src/            # all reusable code
│   ├── config.py        # seeds, paths, COMPOUND constants
│   ├── data.py          # sacred holdout protocol enforced here
│   ├── features.py      # FE blocks (target-free first, TE last)
│   ├── cv.py            # StratifiedKFold setup
│   ├── train.py         # train_variant(algo, ...) for LGB/XGB/CatBoost/HistGB
│   └── evaluate.py      # AUC + holdout evaluation
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_holdout_split.py     # one-time — lock the sacred holdout
│   ├── 03_v1_lgb.py            # baseline
│   ├── 04_v2_xgb.py
│   ├── 05_v3_catboost.py
│   ├── 06_v4_histgb.py
│   └── figs/
├── probs/                       # OOF + holdout + test probs (gitignored)
├── submissions/                 # CSVs (gitignored)
└── docs/
    ├── reconnaissance.md        # Phase 0 findings
    ├── pitfalls.md              # S6E4 lessons captured — DO NOT REPEAT
    └── versions/<vN>.md
```

## Sacred holdout protocol

A random 20% slice (stratified on target, seed locked) is held out at the very start.
The holdout is **never touched** during model development — only at the end of each version to validate the OOF estimate.

`src/data.py::load_data()` returns `(train_pool, holdout, test)`. There is no
function that returns the full train + holdout combined for training purposes.
This is enforced in code, not just discipline.

## Compute strategy

- **Local (RTX 4070 8GB)**: notebook dev, 10K-row sanity passes, quick 1-fold CV, EDA
- **Colab Pro**: full 5-fold CV, Optuna sweeps, multi-seed ensembles
- **Google Drive `s6e5/`**: artifact persistence (OOFs, models, submissions)

Threshold: any training run >10 min → cloud.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e5 -p data/raw/
unzip data/raw/*.zip -d data/raw/
kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction -p data/external/
unzip data/external/*.zip -d data/external/
python notebooks/02_holdout_split.py    # ONE-TIME — locks the sacred holdout
```

## Key references

- `docs/reconnaissance.md` — Phase 0 task spec, group structure, adversarial validation
- `docs/pitfalls.md` — S6E4 mistakes we will NOT repeat
- `notebooks/figs/eda_summary.md` — Phase 2 EDA findings
- `~/knowledge-graph/kaggle/2026-20_s6e5-pit-stops.md` — competition checklist
- `~/knowledge-graph/kaggle/training-configuration-lessons.md` — L1–L18 canonical lessons
