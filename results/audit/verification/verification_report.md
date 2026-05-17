# CLISA-EWU repository verification report

Generated: `2026-05-18_00-06-51`

## Summary

- Required final tables OK: **9 / 9**
- CSV files scanned: **103**
- Key value hits: **115**
- Key values missing from staged CSVs: **0**
- Disallowed data/checkpoint files found in staged repo: **0**

## Ablation table check

- Status: `OK`
- Rows: `11`
- Warning: 11 rows found. This matches the verified audit but not the earlier expected 14-row ablation plan.

## FACED shape evidence

- `C\features\faced_X_32ch.npy` → shape `(110208, 30, 5)`, dtype `float32`

## DANCE/M25 caution

Use the LOSO verified table for manuscript claims. Treat Phase-B reproduction values as historical or cautionary unless reconciled.

## Missing key values

- None.

## Disallowed files

No disallowed data/checkpoint extensions were found in the staged repository.
