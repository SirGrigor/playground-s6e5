"""Experiment observer — hypothesis-before-result discipline enforced.

Usage::

    from src.observer import Experiment

    exp = Experiment.start(
        version="v3",
        parent="v2",
        hypothesis="Compound x RaceProgress interaction lifts holdout +0.003-0.005",
        predicted_delta=0.004,
        confidence="medium",
        feature_changes=["+ compound_x_raceprog_bin"],
        config_changes={},
        pipeline_changes=[],
        cloud_or_local="local",
    )

    # ... train + eval ...

    exp.record(
        oof_auc_mean=0.93701,
        oof_auc_per_fold=[0.9362, 0.9381, 0.9358, 0.9369, 0.9381],
        holdout_auc=0.93612,
        runtime_sec=183,
    )
    exp.commit()

`Experiment.start()` enforces non-empty hypothesis + predicted_delta.
`exp.commit()` runs the 7 auto-flag detectors before appending to
`experiments.jsonl`.

`experiments.jsonl` is the source of truth; `docs/diary.md` is rendered
from it by `src.diary` (read-only).
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import ROOT

JSONL_PATH = ROOT / "experiments.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _load_jsonl() -> list[dict]:
    if not JSONL_PATH.exists():
        return []
    out = []
    for line in JSONL_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _find_entry(version: str) -> dict | None:
    for entry in _load_jsonl():
        if entry.get("version") == version:
            return entry
    return None


@dataclass
class Experiment:
    # required pre-run
    version: str
    parent: str | None
    hypothesis: str
    predicted_delta: float
    confidence: str
    feature_changes: list[str]
    config_changes: dict[str, Any]
    pipeline_changes: list[str]
    cloud_or_local: str

    # auto-captured
    created_at: str = field(default_factory=_now_iso)
    git_sha: str | None = field(default_factory=_git_sha)

    # post-run (record())
    completed_at: str | None = None
    oof_auc_mean: float | None = None
    oof_auc_per_fold: list[float] | None = None
    holdout_auc: float | None = None
    runtime_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # post-commit (auto-fill)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    actual_delta: float | None = None
    parent_holdout_auc: float | None = None

    @classmethod
    def start(
        cls,
        *,
        version: str,
        parent: str | None,
        hypothesis: str,
        predicted_delta: float,
        confidence: str = "medium",
        feature_changes: list[str] | None = None,
        config_changes: dict[str, Any] | None = None,
        pipeline_changes: list[str] | None = None,
        cloud_or_local: str = "local",
    ) -> "Experiment":
        if not hypothesis or not hypothesis.strip():
            raise ValueError("Experiment.start() requires a non-empty hypothesis.")
        if predicted_delta is None:
            raise ValueError("Experiment.start() requires predicted_delta (use 0.0 if truly none).")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"confidence must be low/medium/high, got {confidence!r}")
        if _find_entry(version) is not None:
            raise ValueError(
                f"Experiment {version!r} already exists in {JSONL_PATH.name}. "
                "Choose a new version name."
            )
        return cls(
            version=version,
            parent=parent,
            hypothesis=hypothesis.strip(),
            predicted_delta=float(predicted_delta),
            confidence=confidence,
            feature_changes=feature_changes or [],
            config_changes=config_changes or {},
            pipeline_changes=pipeline_changes or [],
            cloud_or_local=cloud_or_local,
        )

    def record(
        self,
        *,
        oof_auc_mean: float,
        oof_auc_per_fold: list[float],
        holdout_auc: float,
        runtime_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.oof_auc_mean = float(oof_auc_mean)
        self.oof_auc_per_fold = [float(x) for x in oof_auc_per_fold]
        self.holdout_auc = float(holdout_auc)
        self.runtime_sec = float(runtime_sec)
        self.completed_at = _now_iso()
        if extra:
            self.extra.update(extra)

    def note(self, text: str) -> None:
        if not text.strip():
            return
        self.notes.append(f"[{_now_iso()}] {text.strip()}")

    def _autoflag(self) -> None:
        f = self.flags

        # Need results for most checks
        if self.oof_auc_per_fold is None or self.holdout_auc is None or self.oof_auc_mean is None:
            return

        # 1. Fold collapse
        fold_min = min(self.oof_auc_per_fold)
        if fold_min < self.oof_auc_mean - 0.01:
            f.append(f"fold_collapse(fold_min={fold_min:.5f}, mean={self.oof_auc_mean:.5f})")

        # 2. Methodology leak
        gap = abs(self.oof_auc_mean - self.holdout_auc)
        if gap > 0.005:
            f.append(f"methodology_leak(|oof-holdout|={gap:.5f})")

        # 3. Silent regression vs parent holdout
        if self.parent:
            parent_entry = _find_entry(self.parent)
            if parent_entry and parent_entry.get("holdout_auc") is not None:
                self.parent_holdout_auc = float(parent_entry["holdout_auc"])
                self.actual_delta = self.holdout_auc - self.parent_holdout_auc
                if self.actual_delta < -0.001:
                    f.append(f"silent_regression(Δhold={self.actual_delta:+.5f} vs {self.parent})")

        # 4. Fold instability
        if len(self.oof_auc_per_fold) >= 2:
            fold_std = stdev(self.oof_auc_per_fold)
            if fold_std > 0.005:
                f.append(f"fold_instability(std={fold_std:.5f})")

        # 5/6. Prediction calibration (only meaningful when predicted_delta != 0 and actual_delta known)
        if self.actual_delta is not None and self.predicted_delta:
            pred = self.predicted_delta
            act = self.actual_delta
            ratio = act / pred if pred != 0 else 0
            # Direction match required for "undershot/overshot" to be meaningful
            if (pred > 0 and act > 0) or (pred < 0 and act < 0):
                abs_ratio = abs(act) / abs(pred)
                if abs_ratio < 0.5:
                    f.append(f"prediction_undershot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
                elif abs_ratio > 2.0:
                    f.append(f"prediction_overshot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
            elif pred != 0 and act != 0:
                # Sign mismatch — bigger deal than ratio miss
                f.append(f"prediction_sign_mismatch(actual={act:+.5f} vs pred={pred:+.5f})")

        # 7. Multiple changes → attribution ambiguous
        n_changes = (
            len(self.feature_changes)
            + len(self.pipeline_changes)
            + len(self.config_changes)
        )
        if n_changes > 1:
            f.append(f"multiple_changes(n={n_changes}) — attribution ambiguous, consider ablation")

    def commit(self) -> None:
        if self.oof_auc_mean is None or self.holdout_auc is None:
            raise RuntimeError(
                "Experiment.commit() requires .record() to have been called first."
            )
        self._autoflag()
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a") as fp:
            fp.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


# Convenience for human notes after the fact (without re-running anything).
def add_note(version: str, text: str) -> None:
    """Append a human note to an existing experiment.

    Re-writes experiments.jsonl with the note appended to the matching entry.
    """
    if not text.strip():
        return
    entries = _load_jsonl()
    found = False
    for entry in entries:
        if entry.get("version") == version:
            entry.setdefault("notes", []).append(f"[{_now_iso()}] {text.strip()}")
            found = True
            break
    if not found:
        raise ValueError(f"No experiment {version!r} in {JSONL_PATH.name}.")
    with JSONL_PATH.open("w") as fp:
        for entry in entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
