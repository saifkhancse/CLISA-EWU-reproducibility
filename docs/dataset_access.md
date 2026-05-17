# Dataset access

This repository does not redistribute the original SEED-IV or FACED datasets.

## SEED-IV

Users must obtain SEED-IV from the official dataset provider. This repository should provide:

- expected extracted-feature filenames,
- preprocessing scripts,
- LOSO split files,
- channel mappings,
- expected feature shapes,
- expected label encoding.

Expected processed SEED-IV feature shapes observed in the project audit:

- Full channel DE features: `(37575, 310)` or `(37575, 62, 5)`
- Six-channel DE features: `(37575, 30)`
- Labels: `(37575,)`
- Subjects: `(37575,)`

## FACED

Users must obtain FACED from the official dataset provider.

Important audit finding: the raw dataset may be described as 32-channel, but the processed feature file used in this project was found as:

- `faced_X_32ch.npy`: `(110208, 30, 5)`
- flattened variants: `(110208, 150)`

Therefore, the manuscript should clearly state that this work used the available processed 30-channel DE representation for FACED domain-gap analysis.

## Do not upload

Do not upload raw or derived dataset files to GitHub unless redistribution is explicitly permitted.
