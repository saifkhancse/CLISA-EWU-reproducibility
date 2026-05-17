# Repository staging status

Generated: `2026-05-17_23-33-10`

## Current status

This folder is a clean staging repository, not the final public release.

## Copied material

The staging manifest is stored at:

`results/audit/staging_manifest_cell03.csv`

## Excluded by design

- Raw SEED-IV/FACED files
- Derived `.npy`/`.npz` feature arrays
- Large checkpoints
- Old Word/PDF/PPT drafts
- Downloaded related-paper PDFs
- Webapp files
- Per-fold checkpoint JSON flood

## Next required work

1. Inspect copied final tables and figures.
2. Extract model/evaluation code from notebooks into `src/` and `scripts/`.
3. Add real preprocessing scripts.
4. Add LOSO split files.
5. Add exact dependency versions.
6. Reconcile DANCE/M25 value before final manuscript claim.
7. Decide whether to publish checkpoints externally.
