# CLISA-EWU: Subject-Independent EEG Emotion Recognition

This repository contains the reproducibility materials for the manuscript:

**Subject-Independent EEG Emotion Recognition via Contrastive Inter-Subject Alignment and Wearable Channel Compression**

## Overview

CLISA-EWU is a contrastive inter-subject alignment framework for subject-independent EEG emotion recognition. The method is evaluated on SEED-IV using both the full 62-channel setting and a compact six-channel wearable setting with FP1, FP2, F7, F8, T7, and T8.

The repository provides result tables, figures, configuration files, split definitions, and scripts for reproducing the reported tables.

## Repository contents

- `results/final_tables/`: final CSV tables used in the manuscript.
- `results/reproduced_tables/`: tables regenerated from the final CSV files.
- `results/supplementary/`: supplementary result files.
- `figures/`: exported result figures.
- `configs/`: model and processing configuration files.
- `data_splits/`: SEED-IV leave-one-subject-out split definitions and channel mappings.
- `docs/`: dataset access notes, metric definitions, and reproducibility notes.
- `scripts/`: scripts for table reproduction, result verification, and repository checks.
- `notebooks/provenance/`: original analysis notebooks retained for traceability.

## Datasets

The original SEED-IV and FACED datasets are not included in this repository. Users should obtain them from the official dataset providers.

This repository includes the split definitions, channel mapping, and metadata required to reproduce the reported analysis once the datasets are available locally.

## Main results

| Method | Setting | Metric | Result |
|---|---|---:|---:|
| CLISA-EWU | SEED-IV, 62-channel | Balanced accuracy (AccB) | 0.6801 |
| CLISA-EWU | SEED-IV, six-channel | Balanced accuracy (AccB) | 0.6199 |
| BYOL | SEED-IV, 62-channel | Balanced accuracy (AccB) | 0.6089 |
| BYOL | SEED-IV, six-channel | Balanced accuracy (AccB) | 0.6124 |

## Supplementary result

The best single-fold result reached **85.35% accuracy**. This result is provided as a supplementary result, while the main reported performance is based on the full repeated leave-one-subject-out evaluation.

## Metrics

- **Balanced accuracy (AccB)**: the average of class-wise recall values.
- **Accuracy**: the proportion of correctly classified samples.
- **Macro-F1**: the unweighted average of class-wise F1 scores.

## Reproducing the tables

Install the required packages:

    pip install -r requirements.txt

Generate the reproduced tables:

    python scripts/reproduce_tables.py

Verify the reported values:

    python scripts/verify_results.py

Check the repository files:

    python scripts/verify_repository.py

List the available exported figures:

    python scripts/reproduce_figures.py

## Dataset-dependent scripts

The following scripts require local access to the official datasets and locally generated features:

    python scripts/preprocess_seediv.py
    python scripts/evaluate_loso.py

## License

The project code and documentation are released under the MIT License. The license does not cover SEED-IV, FACED, or any other third-party dataset.
