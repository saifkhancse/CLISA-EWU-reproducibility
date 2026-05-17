# CLISA-EWU: Subject-Independent EEG Emotion Recognition

This repository provides journal-neutral reproducibility materials for the manuscript:

Subject-Independent EEG Emotion Recognition via Contrastive Inter-Subject Alignment and Wearable Channel Compression

## Overview

CLISA-EWU is a contrastive inter-subject alignment framework for subject-independent electroencephalography-based emotion recognition. The study evaluates both the full 62-channel SEED-IV setting and a compact six-channel frontal-temporal wearable setting using FP1, FP2, F7, F8, T7, and T8.

This repository supports peer review, table reproduction, result verification, and independent inspection without redistributing restricted datasets.

## What is included

- Final result tables used for manuscript preparation.
- Reproduced Markdown and CSV table outputs.
- Aggregate result CSVs for provenance.
- Exported figures.
- Split and configuration metadata.
- Verification scripts.
- Dataset-access documentation.
- Checkpoint policy.
- Provenance notebooks for traceability.

## What is not included

The original SEED-IV and FACED datasets are not redistributed in this repository. Derived feature arrays and raw dataset files are also excluded. Users must obtain restricted datasets from official providers and follow the providers' access terms.

Large model checkpoints are not committed directly. If checkpoints are released later, they should be hosted through GitHub Releases, Zenodo, OSF, or another persistent artifact host with checksums.

## Metric definitions

balanced accuracy (AccB) means balanced accuracy, defined as the average of class-wise recall values. It is used as the primary robustness-oriented accuracy metric in the result tables.

Accuracy means standard overall accuracy.

Macro-F1 means the unweighted average of class-wise F1 scores.

## Expected key results

| Model / setting | Metric |
|---|---:|
| CLISA-EWU, SEED-IV 62-channel | balanced accuracy (AccB) about 0.6801 |
| CLISA-EWU, SEED-IV six-channel | balanced accuracy (AccB) about 0.6199 |
| BYOL BYOL, 62-channel | balanced accuracy (AccB) about 0.6089 |
| BYOL BYOL, six-channel | balanced accuracy (AccB) about 0.6124 |

DANCE/DANCE Teacher values require careful reporting because internal sources show a mismatch. DANCE should not be used as the primary contribution claim unless the final leave-one-subject-out value is reconciled.

## Repository structure

configs/
data_splits/
docs/
figures/
notebooks/provenance/
results/
scripts/
src/
tests/

## Reproducibility commands

Run these commands from the repository root:

python scripts/generate_data_splits.py
python scripts/reproduce_tables.py
python scripts/verify_results.py
python scripts/verify_repository.py
python scripts/reproduce_figures.py

Dataset-dependent preprocessing and full leave-one-subject-out evaluation require official dataset access:

python scripts/preprocess_seediv.py
python scripts/evaluate_loso.py

## Current verification status

The repository has passed internal public-release checks for required repository files, JSON split/config validity, final table presence, reproduced table generation, key result value detection, absence of raw data/checkpoint files, and absence of public local absolute paths.

## Supplementary best-fold result

The best single-fold diagnostic result reached 85.35% accuracy. This result is reported only as supplementary evidence and is not used for headline mean leave-one-subject-out comparison or SOTA comparison.

## Scientific cautions

1. The ablation table currently contains 11 verified rows. Do not claim 14 verified ablation rows unless the missing rows are checkpoint-verified.
2. FACED is used through the processed 30-channel differential-entropy feature representation observed in the project audit.
3. DANCE/DANCE Teacher values should be reported cautiously unless the final leave-one-subject-out source is reconciled.
4. Raw SEED-IV and FACED datasets are not redistributed.

## License

This repository uses the MIT License for project code and documentation authored for this work. The license does not grant permission to redistribute SEED-IV, FACED, or any other third-party dataset or derived data files subject to third-party access terms.
