# Reproducibility notes

## Minimum public repository contents

- Source code for CLISA-EWU.
- Preprocessing scripts.
- LOSO split definitions.
- Channel mapping for full 62-channel and six-channel settings.
- Training scripts.
- Evaluation scripts.
- Final CSV result tables.
- Figure reproduction scripts.
- Environment file.
- Dataset access instructions.
- Checkpoint policy.

## Current limitation

This staging folder currently contains provenance notebooks and result artifacts. The next step is to extract reusable code from notebooks into scripts under `src/` and `scripts/`.

## Values requiring caution

- DANCE/DANCE Teacher final LOSO value should be reported carefully because internal audit detected mismatch.
- FACED should be used as domain-gap analysis unless the full FACED benchmark is completed and verified.
- GraphFormer results should be treated as exploratory unless full non-smoke-test runs are verified.
- Ablation table currently has 11 verified rows in the final audit, not 14.
