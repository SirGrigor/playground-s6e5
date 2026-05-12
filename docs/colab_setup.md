# Colab setup — one-time

The first Colab run requires manual data upload to Drive. After that, every subsequent run reuses the same data.

## Step 1 — Verify Drive folder structure

Drive folder: <https://drive.google.com/drive/folders/1N6PFShEtMj2KSYWxaQQz-6Kro1CTLilh>

Inside it, create this structure (manually via the Drive UI):

```
<your-drive-folder>/
└── s6e5/
    ├── data/
    │   ├── raw/        ← upload here
    │   └── external/   ← upload here
    ├── probs/          ← created automatically by Colab runs
    └── submissions/    ← created automatically by Colab runs
```

## Step 2 — Upload competition data to Drive

From local machine, the files are at:

```
~/IdeaProjects/kaggle/playground-s6e5/data/raw/train.csv             (54 MB)
~/IdeaProjects/kaggle/playground-s6e5/data/raw/test.csv              (22 MB)
~/IdeaProjects/kaggle/playground-s6e5/data/raw/sample_submission.csv (1.7 MB)
~/IdeaProjects/kaggle/playground-s6e5/data/external/f1_strategy_dataset_v4.csv (13 MB)
```

Total ~91 MB. Upload to:

```
<drive>/s6e5/data/raw/        ← train.csv, test.csv, sample_submission.csv
<drive>/s6e5/data/external/   ← f1_strategy_dataset_v4.csv
```

Easiest path:

1. Drag the four files into the Drive web UI
2. Wait for upload to complete (~2 min on decent connection)

Alternative if you have `gdrive` or `rclone` configured: scriptable upload.

## Step 3 — Verify the Drive path in Colab

Open `notebooks/colab_runner.ipynb` in Colab. The first cell has:

```python
DRIVE_S6E5 = '/content/drive/MyDrive/kaggle/s6e5'   # ← EDIT IF DIFFERENT
```

Adjust this path to match where your s6e5 folder lives in Drive.

## Step 4 — Run the v1 baseline

In the Colab notebook, execute cells 1–6 in order. On cell 5 (Run the target script), wait for output:

```
v1 — LightGBM baseline (raw + Block 1)
pool: (351312, 16)   holdout: (87828, 16)   test: (188165, 15)
...
OOF AUC (mean of folds): 0.93xxx
Holdout AUC:              0.93xxx
...
```

Then cell 6 syncs `probs/v1_lgb/` and `submissions/v1_lgb.csv` back to Drive.

## Step 5 — Submit

Cell 8 downloads the submission CSV to your local machine. From there:

```bash
kaggle competitions submit -c playground-series-s6e5 \
    -f ~/Downloads/v1_lgb.csv \
    -m "v1 LGB baseline + Block 1 features"
```

The leaderboard score appears on your competition page.

## Tips

- Colab Pro: choose **T4** GPU (sufficient for LGB) — saves compute units vs A100/V100.
- If you hit the Colab session timeout, the artifacts on Drive are preserved. Just re-mount and re-clone.
- For multi-model runs (v2 XGB, v3 CatBoost, ...), just edit `SCRIPT` in cell 5 and re-run cells 5–6.
- The experiments.jsonl appends in `/content` are lost on session shutdown unless cell 6 syncs them back. Cell 6 always syncs.
