# Checkpoint policy

The original project folder contains many checkpoint/model files. These should not be committed directly to GitHub.

## Recommended policy

- Keep source code, configs, split files, and result CSVs in GitHub.
- Keep large checkpoints outside GitHub.
- Use GitHub Releases, Zenodo, OSF, or institutional storage if model weights must be shared.
- Provide checksums for any externally hosted checkpoints.
- Provide scripts that reproduce the checkpoints when the user has access to the datasets.

## Current staging decision

By default, this staging script excludes:

- `.pt`
- `.pth`
- `.ckpt`
- `.onnx`
- `.joblib`
- `.sav`

Small demonstration checkpoints can be included only after licensing and size review.
