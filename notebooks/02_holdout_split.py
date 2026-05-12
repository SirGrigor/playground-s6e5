"""ONE-TIME SCRIPT — generate the sacred holdout split.

Run this once. The resulting split is persisted to data/splits/holdout_v1.parquet
and tracked in git. After that, src/data.py loads it.

Re-running this script with a different seed would create a DIFFERENT holdout,
which would invalidate every prior version's holdout evaluation. So don't.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RAW, SPLITS, HOLDOUT_SEED, HOLDOUT_FRAC, TARGET, ID


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS / "holdout_v1.parquet"

    if out_path.exists():
        print(f"ABORT: {out_path} already exists. Refusing to overwrite the sacred holdout.")
        print("If you genuinely need to regenerate, delete the file by hand and re-run.")
        sys.exit(1)

    train = pd.read_csv(RAW / "train.csv")
    print(f"Loaded train: {train.shape}")
    print(f"Target positive rate (overall): {train[TARGET].mean():.5f}")

    _, holdout_idx = train_test_split(
        np.arange(len(train)),
        test_size=HOLDOUT_FRAC,
        stratify=train[TARGET],
        random_state=HOLDOUT_SEED,
    )
    holdout_rows = train.iloc[holdout_idx]
    holdout_ids = holdout_rows[ID].astype("int64")

    print(f"Holdout size: {len(holdout_ids)} rows ({HOLDOUT_FRAC:.0%})")
    print(f"Holdout positive rate: {holdout_rows[TARGET].mean():.5f}")
    print(f"Train pool size: {len(train) - len(holdout_ids)} rows")

    # Persist only the row IDs — small file, git-trackable, unambiguous
    df = pd.DataFrame({"id": holdout_ids.to_numpy()})
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  size on disk: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
