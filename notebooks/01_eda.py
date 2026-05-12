"""
S6E5 EDA — Phase 2

Goal: verify the recon-doc hypotheses on the actual data and surface
any signal we missed.

Output:
  - stdout: tables and stats
  - notebooks/figs/*.png: plots
  - notebooks/figs/eda_summary.md: written summary

Run with the project venv:
  source .venv/bin/activate && python notebooks/01_eda.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

train = pd.read_csv(DATA / "raw" / "train.csv")
test = pd.read_csv(DATA / "raw" / "test.csv")
orig = pd.read_csv(DATA / "external" / "f1_strategy_dataset_v4.csv")
TARGET = "PitNextLap"

NUM = ["TyreLife", "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change", "LapNumber", "Position", "Stint", "Year", "PitStop"]
CAT = ["Driver", "Compound", "Race"]

print("="*80)
print("§1. Target distribution overall + by group")
print("="*80)
print(f"\nOverall positive rate: {train[TARGET].mean():.4f}")

for c in ["Compound", "Stint", "Position", "PitStop", "Year"]:
    g = train.groupby(c, observed=True)[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(f"\n-- {c} -- (pit rate, count)")
    print(g.to_string())

# Race top + bottom 8
race_rate = train.groupby("Race", observed=True)[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
print(f"\n-- Race TOP 8 (highest pit rate) --")
print(race_rate.head(8).to_string())
print(f"\n-- Race BOTTOM 8 (lowest pit rate) --")
print(race_rate.tail(8).to_string())

print("\n" + "="*80)
print("§2. TyreLife vs pit rate — find the spike threshold")
print("="*80)

# Bin TyreLife and compute pit rate per bin
bins = np.arange(0, 70, 2)
train["TyreLife_bin"] = pd.cut(train["TyreLife"], bins=bins, include_lowest=True)
tl_rate = train.groupby("TyreLife_bin", observed=True)[TARGET].agg(["mean", "count"])
print("\n-- TyreLife binned (2-lap bins) --")
print(tl_rate.head(35).to_string())

# Same but per Compound
print("\n-- TyreLife pit rate × Compound (median TyreLife per compound) --")
for comp in train["Compound"].unique():
    sub = train[train["Compound"] == comp]
    print(f"  {comp}: median TyreLife={sub['TyreLife'].median():.1f}, mean pit rate={sub[TARGET].mean():.4f}, n={len(sub):,}")

# Plot: TyreLife pit rate per Compound
fig, ax = plt.subplots(figsize=(11, 6))
for comp in sorted(train["Compound"].unique()):
    sub = train[train["Compound"] == comp].copy()
    sub["bin"] = pd.cut(sub["TyreLife"], bins=np.arange(0, 70, 2), include_lowest=True)
    rate = sub.groupby("bin", observed=True)[TARGET].mean()
    counts = sub.groupby("bin", observed=True)[TARGET].count()
    # Only plot where count > 100
    rate = rate[counts > 100]
    centers = [iv.mid for iv in rate.index]
    ax.plot(centers, rate.values, marker="o", label=f"{comp} (n={len(sub):,})", linewidth=2)
ax.set_xlabel("TyreLife (laps)")
ax.set_ylabel("PitNextLap rate")
ax.set_title("Pit rate vs TyreLife, per Compound")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "01_tyrelife_per_compound.png", dpi=110)
plt.close()
print(f"  saved fig: {FIGS / '01_tyrelife_per_compound.png'}")

print("\n" + "="*80)
print("§3. RaceProgress vs pit rate — verify Deotte pit windows")
print("="*80)

# Binned overall + per Compound
fig, ax = plt.subplots(figsize=(11, 6))
for comp in sorted(train["Compound"].unique()):
    sub = train[train["Compound"] == comp].copy()
    sub["bin"] = pd.cut(sub["RaceProgress"], bins=np.linspace(0, 1, 41), include_lowest=True)
    rate = sub.groupby("bin", observed=True)[TARGET].mean()
    counts = sub.groupby("bin", observed=True)[TARGET].count()
    rate = rate[counts > 100]
    centers = [iv.mid for iv in rate.index]
    ax.plot(centers, rate.values, marker=".", label=f"{comp}", linewidth=1.8)
ax.set_xlabel("RaceProgress")
ax.set_ylabel("PitNextLap rate")
ax.set_title("Pit rate vs RaceProgress, per Compound — pit windows")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "02_raceprogress_per_compound.png", dpi=110)
plt.close()
print(f"  saved fig: {FIGS / '02_raceprogress_per_compound.png'}")

# Numeric: per-compound pit-rate peaks
print("\n-- RaceProgress bins with HIGHEST pit rate per Compound (n>500) --")
for comp in sorted(train["Compound"].unique()):
    sub = train[train["Compound"] == comp].copy()
    sub["rp_bin"] = pd.cut(sub["RaceProgress"], bins=np.linspace(0, 1, 41), include_lowest=True)
    rate = sub.groupby("rp_bin", observed=True)[TARGET].agg(["mean", "count"])
    rate = rate[rate["count"] > 500].sort_values("mean", ascending=False).head(5)
    print(f"\n  {comp} top-5 bins:")
    print(rate.to_string())

print("\n" + "="*80)
print("§4. Mutual information ranking (numeric features)")
print("="*80)

# Cast categoricals to codes for MI
df = train.copy()
for c in CAT:
    df[c] = pd.Categorical(df[c]).codes

X = df[NUM + CAT].fillna(0)
y = df[TARGET].astype(int)

# Subsample for speed (MI is O(N log N) but with 440K can be slow)
SUBSAMPLE = 100_000
rng = np.random.default_rng(42)
idx = rng.choice(len(X), size=SUBSAMPLE, replace=False)
Xs = X.iloc[idx].reset_index(drop=True)
ys = y.iloc[idx].reset_index(drop=True)

mi = mutual_info_classif(Xs, ys, random_state=42, n_neighbors=3)
mi_df = pd.DataFrame({"feature": Xs.columns, "MI": mi}).sort_values("MI", ascending=False)
print(f"\n-- MI ranking (subsampled to {SUBSAMPLE:,}) --")
print(mi_df.to_string(index=False))

print("\n" + "="*80)
print("§5. Stint structure — laps per (Race, Driver, Year) stint")
print("="*80)

# Combined train+test view of stints
all_df = pd.concat([
    train[["Race", "Driver", "Year", "LapNumber"]].assign(split="train"),
    test[["Race", "Driver", "Year", "LapNumber"]].assign(split="test"),
], ignore_index=True)

stint_sizes = all_df.groupby(["Race", "Driver", "Year"]).size().describe()
print(f"\n-- Stint (lap-counts per Race-Driver-Year combo) — combined train+test --")
print(stint_sizes.to_string())

# Train vs test laps per stint
per_stint = all_df.groupby(["Race", "Driver", "Year", "split"]).size().unstack("split", fill_value=0)
per_stint["total"] = per_stint["train"] + per_stint["test"]
per_stint["train_frac"] = per_stint["train"] / per_stint["total"]
print(f"\n-- train_frac distribution within stints (across all stints) --")
print(per_stint["train_frac"].describe().to_string())

print("\n" + "="*80)
print("§6. Train/Test split pattern by LapNumber")
print("="*80)

# Does test get late laps systematically? Or interleaved?
# For a sample of stints, show which LapNumbers are in train vs test
sample_stints = all_df.groupby(["Race", "Driver", "Year"]).size().sort_values(ascending=False).head(6).index
for (race, driver, year) in sample_stints:
    sub = all_df[(all_df["Race"] == race) & (all_df["Driver"] == driver) & (all_df["Year"] == year)].sort_values("LapNumber")
    train_laps = sorted(sub[sub["split"] == "train"]["LapNumber"].tolist())
    test_laps = sorted(sub[sub["split"] == "test"]["LapNumber"].tolist())
    print(f"\n  {driver} @ {race} {year}:")
    print(f"    train laps ({len(train_laps)}): {train_laps[:15]}{'…' if len(train_laps)>15 else ''}")
    print(f"    test laps  ({len(test_laps)}):  {test_laps[:15]}{'…' if len(test_laps)>15 else ''}")

print("\n" + "="*80)
print("§7. Outlier check — confirm cleaning thresholds from baselines")
print("="*80)
for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"]:
    s = train[c]
    print(f"\n  {c}:")
    print(f"    min={s.min():.2f}, max={s.max():.2f}, "
          f"median={s.median():.2f}, std={s.std():.2f}")
    print(f"    pct > 500: {(s.abs() > 500).mean()*100:.4f}%, "
          f"count > 500: {(s.abs() > 500).sum()}")

print("\n" + "="*80)
print("§8. Key takeaways to write into eda_summary.md")
print("="*80)
print("\n--- DONE --- review figs in notebooks/figs/ ---")
