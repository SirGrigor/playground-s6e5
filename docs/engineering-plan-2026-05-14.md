# Engineering Plan — Productize the S6E5 Pipeline (2026-05-14)

## Goal

Take what proved its worth on S6E5 (curated public-mix discovery, progressive blend pipeline, harvester, pre-flight validation) and turn it into reusable, tested, dependency-clean tooling that works equally well on local and Colab. Position us for S6E6+ to start from Day 1 with this stack instead of re-deriving it.

## Phases

### Phase 1 — Immediate tools (tonight / tomorrow morning, ~1.5 hours)

**Why first**: tomorrow we have 5+ Kaggle submission slots and 4 v19 candidates queued. The curated explorer should generate the right candidates automatically before then.

**Deliverables**:

1. `notebooks/26_curated_explorer.py` — standalone curated-mix tool
   - Reads `harvest/v18/manifest.json`
   - Filters INCLUDE-TEST candidates by `claimed_lb ≥ threshold` (auto: baseline_holdout - 0.001)
   - For K ∈ {3, 4, 5, 6} and ratio ∈ {0.20, 0.30, 0.50, 0.70, 0.90, 1.00}:
     - Build variant = (1-ratio) × baseline + ratio × mean(top-K curated)
     - Save as `submissions/v19.NNN.csv`
     - Compute ρ vs baseline, public_avg_claimed_lb, predicted_lb_score
   - Output: `curated_audit.md` with ranked recommendations table
   - ~150 lines

2. `25_pipeline.py` — add Phase C (curated mix)
   - After Phase B (all-public mix), run curated phase using the same function
   - Shared module: refactor curated math into reusable function
   - Pipeline ends with Phase A, B, C all having their own release sets
   - ~50 lines new code

**Outcome**: tomorrow you run pipeline OR explorer, get 10-20 curated variants automatically, submit best 3 by predicted_lb.

### Phase 2 — Extract reusable tech to utils repo (~3-4 hours)

**Why**: avoid re-deriving for S6E6+. Currently `25_pipeline.py` knows about: harvesting, blending math, manifest parsing, Drive recovery, validation, etc. — all reusable.

**Deliverables**:

1. New GitHub repo: `kaggle-playground-utils`
   - Layout:
     ```
     kaggle-playground-utils/
     ├── pyproject.toml         # pinned deps, uv-build, semver
     ├── README.md              # quickstart
     ├── LICENSE                # MIT
     ├── CHANGELOG.md
     ├── src/kaggle_playground_utils/
     │   ├── __init__.py
     │   ├── harvesting.py      # kaggle CLI bulk-scan + manifest builder
     │   ├── blending.py        # Nelder-Mead, rank_norm, curated_mix
     │   ├── evaluation.py      # OOF/holdout/test metrics, ρ matrix
     │   ├── pipeline.py        # Phase A/B/C orchestration template
     │   ├── validation.py      # pre-flight check helpers
     │   ├── drive.py           # Colab Drive restore/sync utilities
     │   └── observers.py       # generic experiment logger
     ├── tests/
     │   ├── conftest.py        # synthetic data fixtures
     │   ├── test_blending.py
     │   ├── test_harvesting.py
     │   ├── test_validation.py
     │   └── test_pipeline_e2e.py
     └── examples/
         ├── playground_template/  # minimal end-to-end example
         └── README.md
     ```

2. **Migration**: `playground-s6e5` repo imports from utils
   - `notebooks/25_pipeline.py` becomes a thin wrapper around `kaggle_playground_utils.pipeline.run()`
   - Comp-specific code stays in s6e5: `src/data.py` (sacred holdout), `src/features.py` (yekenot FE)
   - Pin utils version in s6e5's pyproject.toml

3. **Distribution**:
   - First release: `v0.1.0`
   - pip install: `git+https://github.com/{user}/kaggle-playground-utils@v0.1.0`
   - Tag releases via GitHub Releases

**Outcome**: next comp clones a template + pip installs the utils. ~10 minutes to be running a baseline.

### Phase 3 — Dependency audit + lock (~1 hour)

**Why**: today's run hit env issues (Kaggle auth, pytorch_lightning ABI conflicts, etc.). Pinning eliminates these.

**Deliverables**:

1. Audit current `pyproject.toml`:
   - Pin core deps (numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, pytabkit)
   - Use `>=X.Y,<Z` ranges (not exact pins — too brittle)
   - Drop unused

2. Generate `uv.lock` (or `requirements-lock.txt`) for both s6e5 and utils repos

3. Cross-version testing matrix:
   - sklearn 1.3, 1.5, 1.6 (CV-safe TargetEncoder)
   - pytabkit 1.5, 1.7 (RealMLP/TabM api stability)
   - Document min versions in README

4. Colab pre-flight: cell 4 verifies versions match the lock

**Outcome**: no more silent env failures. Reproducible across machines.

### Phase 4 — Test coverage (~2 hours)

**Why**: today's bugs (run_factory key mismatch, harvest restore not merging, etc.) all preventable with tests.

**Deliverables**:

1. `tests/test_blending.py`:
   - `test_rank_norm_uniform_distribution`
   - `test_nelder_mead_blend_recovers_optimal_weights`
   - `test_curated_mix_filter_by_claimed_lb`
   - `test_curated_mix_ratio_grid`

2. `tests/test_harvesting.py`:
   - `test_extract_claimed_lb_from_title` (regex edge cases)
   - `test_extract_oof_auc_from_log` (JSON-stream + plain text)
   - `test_validate_submission_id_alignment`
   - `test_categorize_verdict_paths` (all 8 verdicts)

3. `tests/test_validation.py`:
   - `test_preflight_drive_mounted_missing`
   - `test_preflight_manifest_kernel_subdirs_missing`
   - `test_preflight_baseline_candidate_count`

4. `tests/test_pipeline_e2e.py` (integration):
   - Tiny synthetic dataset (100 rows, 5 features, 3 fake "kernels")
   - Full pipeline run start to finish
   - Asserts: produces ≥1 release, releases.jsonl populated correctly

5. CI: GitHub Actions runs pytest on each push
   - matrix: Python 3.10, 3.11, 3.12
   - reports coverage

**Outcome**: regressions caught at commit time. Confidence to refactor.

### Phase 5 — Local + Colab compatibility (~1 hour)

**Why**: today's friction was Drive sync, path handling, auth — all environment-specific.

**Deliverables**:

1. `kaggle_playground_utils.environment` module
   - `detect()`: returns `'colab' | 'local'`
   - `drive_path() | None`: Drive mount path if Colab, else None
   - `kaggle_auth_from_env_or_userdata()`: handles both Colab Secrets + kaggle.json

2. Test on fresh Colab notebook:
   - Mount Drive, install utils from GitHub
   - Run example pipeline on small synthetic data
   - Verify probs / harvest / submissions all sync correctly

3. Test on fresh local environment:
   - Fresh venv, install utils
   - Same example pipeline (no Drive)
   - Verify works without internet (after deps installed)

4. Document the differences in README

**Outcome**: one script works in both environments without conditional code in the user's notebook.

### Phase 6 — Documentation (~1 hour)

**Why**: utils repo is useless if no one (including future-me) can figure out how to use it.

**Deliverables**:

1. `README.md`:
   - Quickstart (5-line example)
   - Architecture overview (Phase A/B/C diagram)
   - When to use this vs roll-your-own

2. `docs/migration-guide.md` — moving s6e5-style code to utils

3. `docs/api/` — per-module API reference (auto-generated from docstrings)

4. `examples/playground_template/` — clone-and-go skeleton for new comp:
   - notebooks/01_eda.py
   - notebooks/02_holdout_split.py
   - notebooks/10_v1_baseline.py
   - notebooks/20_harvest.py
   - notebooks/25_pipeline.py — uses utils
   - colab_runner.ipynb — pre-configured cells

**Outcome**: S6E6 starts on day 1 with this template, not by re-deriving from S6E5.

## Sequencing

Recommended order:

1. **Phase 1** tonight — get the curated explorer ready for tomorrow's submissions
2. **Phase 3 (dep audit)** before any extraction — clean foundation first
3. **Phase 2 (utils extraction)** — once deps are clean
4. **Phase 4 (tests)** in parallel with Phase 2 — TDD-ish
5. **Phase 5 (env compat)** after extraction is functional
6. **Phase 6 (docs)** last — easier when API is stable

Total estimated effort: **8-10 hours across 3-4 sessions**.

## What we keep s6e5-specific (do NOT extract)

- `src/data.py` — sacred holdout protocol (comp-specific)
- `src/features.py` — yekenot FE recipe (comp-specific)
- `notebooks/03_v1_lgb_baseline.py` through `notebooks/19_v15_stats.py` — comp history
- Hardcoded competition slug `playground-series-s6e5`
- `data/raw/`, `data/external/` — comp data
- `harvest/v13/`, `harvest/v18/` — comp-specific harvested OOFs

## What goes in the utils repo

- Harvesting: list_top_kernels, download_kernel, extract_claimed_lb, validate_submission, categorize
- Blending: rank_norm, nelder_mead_weights, blend_with_weights, curated_mix, ratio_sweep
- Evaluation: pairwise_rho, predicted_lb_score, oof_auc, holdout_auc, triangle_alignment_check
- Pipeline: progressive_blend_phase_a, public_mix_phase_b, curated_mix_phase_c, run()
- Validation: preflight_drive, preflight_data, preflight_probs, preflight_harvest
- Drive: restore_from_drive_merged, sync_to_drive
- Environment: detect, kaggle_auth_setup
- Observer: experiment_logger with 7 auto-flag detectors (from src/observer.py)

## Open questions

1. **Repo name**: `kaggle-playground-utils` or something snappier?
2. **Distribution**: GitHub Releases only, or also PyPI?
3. **License**: MIT (matches ml-variant-factory)? Or Apache 2.0?
4. **Integration with ml-variant-factory**: should those two repos coexist as separate or merge?
   - I'd say keep separate: variant-factory is "model variant generation"; playground-utils is "comp orchestration". Different concerns.

## Pre-conditions before starting

- All today's submissions logged ✓
- v18.002 / v18.003 / v18.004 / v18.005-007 / v19.001-007 probs synced to Drive ✓
- harvest/v18 manifest stable ✓
- post-mortem written (`~/knowledge-graph/kaggle/2026-19_s6e5-postmortem.md`) ✓

## Out of scope (won't do in this engineering effort)

- TabPFN integration (separate compute-heavy work for next session)
- Tomorrow's actual Kaggle submissions (just queue v19.008-011)
- Backporting Phase C to earlier comp data
- Stacking with meta-learner (separate technique, not blend variants)
