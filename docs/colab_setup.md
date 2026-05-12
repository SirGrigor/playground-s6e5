# Colab setup — one-time

The first Colab run requires manual data upload to Drive. After that, every subsequent run reuses the same data.

## Step 0 — DISABLE CSV-to-Sheets conversion in Drive (critical, one-time)

Drive's default setting converts uploaded CSV files into Google Sheets documents. This breaks `pd.read_csv()` in Colab because there's no actual `.csv` file on disk — only a Sheets object.

1. Open <https://drive.google.com/drive/settings>
2. Find **"Convert uploads"** under "General"
3. **UNCHECK** "Convert uploaded files to Google Docs editor format"
4. Save

Without this step, the Colab run will fail at cell 4 (Sync data) with FileNotFoundError.

## Step 1 — Verify Drive folder structure

Drive folder: <https://drive.google.com/drive/folders/1N6PFShEtMj2KSYWxaQQz-6Kro1CTLilh>

**Verified location** (Drive MCP, 2026-05-12): `MyDrive/Colab Notebooks/kaggle/s6e5`

Inside `s6e5/`, create this structure (manually via the Drive UI):

```
MyDrive/Colab Notebooks/kaggle/s6e5/
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
DRIVE_S6E5 = '/content/drive/MyDrive/Colab Notebooks/kaggle/s6e5'
```

This matches the verified Drive location. No edit needed unless you reorganize the folder later.

**Sanity check after running cell 1** — the `ls -la` output should show real `.csv` files:

```
-rw------- 1 root root 53714242 May 12 ... train.csv
-rw------- 1 root root 22312035 May 12 ... test.csv
...
```

If you see no `.csv` extension and zero file size, the files were converted to Sheets — go back to Step 0.

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
