# Cell 6 Table Reproduction Fix Report

Generated: `2026-05-17_23-40-52`

## What was fixed

`scripts/reproduce_tables.py` was rewritten to avoid `pandas.to_markdown()` and therefore no longer requires the optional `tabulate` package.

## Run status

### reproduce_tables.py

- Return code: `0`

Stdout:

```text
reproduce_tables.py completed
Tables found: 11
Tables reproduced OK: 11
Tables with error: 0
Output: C:\Users\Saif\Desktop\CSE400\CLISA-EWU-reproducibility\results\reproduced_tables
```

### verify_results.py

- Return code: `0`

Stdout:

```text
verify_results.py completed
Required final tables OK: 9 / 9
Key value hits: 121
Key values missing: 0
Disallowed files found: 0
Report: C:\Users\Saif\Desktop\CSE400\CLISA-EWU-reproducibility\results\audit\verification\verification_report.md
```

## Reproduced table summary

- Tables found: `11`
- Tables reproduced OK: `11`
- Tables with error: `0`

## Verification summary

- Required tables OK: `9 / 9`
- Key value hits: `121`
- Key values missing: `0`
- Disallowed files found: `0`
