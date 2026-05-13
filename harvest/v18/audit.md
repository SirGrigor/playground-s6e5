# Harvest v18 — Audit Report

Total kernels examined: 50

## Verdict counts

- **INCLUDE-TEST**: 34
- **EDA**: 13
- **INCLUDE-OOF**: 1
- **WEAK**: 1
- **EXCEPTION**: 1

## Per-kernel table

| Verdict | Slug | Votes | Claimed LB | Claimed OOF | OOF AUC on our pool | ρ vs v17 | Sub | OOF |
|---|---|---|---|---|---|---|---|---|
| EDA | `sarcasmos/predicting-f1-pit-stops` | 60 | — | — | — | — | — | — |
| EDA | `leonchani/02-nn-predicting-f1-pit-stops` | 58 | — | — | — | — | — | — |
| EDA | `cdeotte/eda-predicting-f1-pit-stops` | 57 | — | — | — | — | — | — |
| EDA | `mpwolke/slowest-kaggle-pit-stop` | 48 | — | — | — | — | — | — |
| EDA | `aryankaisth/s6e5-the-complete-eda-100` | 46 | — | — | — | — | — | — |
| EDA | `leonchani/01-eda-predicting-f1-pit-stops` | 29 | — | — | — | — | — | — |
| EDA | `simarbirsinghsandhu/detailed-eda-on-comp-original-dataset` | 23 | — | — | — | — | — | — |
| EDA | `obosrish/dont-overfit-on-orig-0-9512-public` | 20 | 0.95120 | — | — | — | — | — |
| EDA | `amanatar/3-model-ensemble-realmlp-catboost-xgboost` | 19 | — | — | — | — | — | — |
| EDA | `aidensong123/f1-pit-stops-eda-tyre-life-pit-waves-drift` | 19 | — | — | — | — | — | — |
| EDA | `sidcodegg/tabpfn-is-all-you-need` | 18 | — | — | — | — | — | — |
| EDA | `djenkivanov/s6e5-minimal-starter-xgb-oof` | 18 | — | — | — | — | — | — |
| EDA | `yusufmurtaza01/f1-pit-stop-prediction-comprehensive-eda` | 18 | — | — | — | — | — | — |
| EXCEPTION | `masayakawamata/s6e5-stacking-vibe-coding` | 18 | — | — | — | — | — | — |
| INCLUDE-OOF | `yekenot/ps-s6-e5-realmlp-pytabkit` | 93 | — | — | 0.95407 | 0.9946 | ✓ | ✓ |
| INCLUDE-TEST | `kospintr/pitstop-catb-hgbc-xgb-lgbm-realmlp-baseline` | 67 | — | — | — | 0.9779 | ✓ | — |
| INCLUDE-TEST | `flexonafft/f1-submission-blender-0-954` | 65 | 0.95400 | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `analyticaobscura/pit-or-stay-f1-strategy-1` | 53 | — | 0.96234 | — | 0.9862 | ✓ | — |
| INCLUDE-TEST | `rohit8527kmr7518/ps-s6e5-catboost-10-fold-cv` | 48 | — | 0.95265 | — | 0.9832 | ✓ | — |
| INCLUDE-TEST | `pilkwang/s6e5-driver-s-high-driver-feature-eng` | 47 | — | 0.94682 | — | 0.9913 | ✓ | — |
| INCLUDE-TEST | `lucabasa/f1-strategy-eda-and-base-models` | 45 | — | — | — | 0.9787 | ✓ | — |
| INCLUDE-TEST | `nalisha/end-to-end-ml-pipeline-with-cross-validation` | 42 | — | — | — | 0.8910 | ✓ | — |
| INCLUDE-TEST | `anthonytherrien/predicting-f1-pit-stops-blend` | 42 | — | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `nina2025/ps-s6e5-hb1` | 41 | — | — | — | 0.9926 | ✓ | — |
| INCLUDE-TEST | `flexonafft/f1-best-catboost-solution-0-95259` | 41 | 0.95259 | — | — | 0.9834 | ✓ | — |
| INCLUDE-TEST | `pilkwang/ps-s6e5-eda-pred-step-by-step-guidelines` | 41 | — | — | — | 0.9931 | ✓ | — |
| INCLUDE-TEST | `darwish1337/f1-pitstop-medal` | 38 | — | — | — | 0.9732 | ✓ | — |
| INCLUDE-TEST | `imrancoder786/ps-s6-e5-0-95130` | 35 | 0.95130 | 0.95119 | — | 0.9818 | ✓ | — |
| INCLUDE-TEST | `muhammadusmankhan0/pitstops-lgbm-xgboost-lr-catboost-ens-97-auc` | 34 | — | — | — | 0.9642 | ✓ | — |
| INCLUDE-TEST | `emanuellcs/predicting-f1-pit-stops-xgboost-baseline` | 34 | — | — | — | 0.9695 | ✓ | — |
| INCLUDE-TEST | `ravi20076/playgrounds6e5-public-baseline-v1` | 32 | — | — | — | 0.9925 | ✓ | — |
| INCLUDE-TEST | `berkayzkan/xgboost-baseline` | 31 | — | — | — | 0.9727 | ✓ | — |
| INCLUDE-TEST | `anthonytherrien/predicting-f1-pit-stops-nn-residual-network` | 28 | — | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `sadettinamilverdil/sebastian-vettel-aaaa` | 28 | — | 0.95184 | — | 0.9823 | ✓ | — |
| INCLUDE-TEST | `mohamedsadokaissaoui/beginner-friendly-xgboost-feature-engineering` | 28 | — | 0.95740 | — | 0.9667 | ✓ | — |
| INCLUDE-TEST | `ldausl/predicting-f1-pit-stops` | 28 | — | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `visheshbanna/initial-eda-baseline` | 24 | — | 0.93415 | — | 0.9136 | ✓ | — |
| INCLUDE-TEST | `lancecasimiro/predicting-f1-pit-stops` | 24 | — | — | — | 0.9738 | ✓ | — |
| INCLUDE-TEST | `sohailkhanlml/ps-e5-blending-works-0-95411` | 23 | 0.95411 | — | — | 0.9939 | ✓ | — |
| INCLUDE-TEST | `sohailkhanlml/f1-pit-stop-blending-the-leaderboard-0-95418` | 22 | 0.95418 | — | — | 0.9942 | ✓ | — |
| INCLUDE-TEST | `safar1/lb-score-0-95419` | 22 | 0.95419 | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `evgendvorkin/raw-data-catboost-0-94898` | 22 | 0.94898 | — | — | 0.9687 | ✓ | — |
| INCLUDE-TEST | `simarbirsinghsandhu/xgboost-gpu-fe-encoding` | 21 | — | 0.95708 | — | 0.9684 | ✓ | — |
| INCLUDE-TEST | `gyogyocat/0-95219-r-python` | 21 | — | 0.95263 | — | 0.9818 | ✓ | — |
| INCLUDE-TEST | `sohailkhanlml/f1-0-95419-diversity-is-all-we-need` | 20 | 0.95419 | — | — | 0.9943 | ✓ | — |
| INCLUDE-TEST | `mohamedsadokaissaoui/ps6e5-ensemble-0-95314-best-score` | 20 | 0.95314 | — | — | 0.9903 | ✓ | — |
| INCLUDE-TEST | `nikitakuznetsof/ps6e5-catboost-is-all-you-need` | 19 | — | — | — | 0.9713 | ✓ | — |
| INCLUDE-TEST | `sai10py/du-du-du-du` | 19 | — | 0.96026 | — | 0.9742 | ✓ | — |
| INCLUDE-TEST | `svanikkolli/f1-lap-by-lap-prediction-engine` | 19 | — | 0.95082 | — | 0.9841 | ✓ | — |
| WEAK | `sohailkhanlml/residual-multi-layer-perceptron-resmlp-0-9414` | 21 | 0.94140 | — | — | 0.9511 | ✓ | — |