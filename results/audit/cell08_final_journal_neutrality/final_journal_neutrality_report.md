# Final journal-neutral and IEEE Access readiness audit

Generated: `2026-05-17_23-50-59`

## Summary

- Text files scanned: **182**
- Public local-path blockers: **6**
- Disallowed data/checkpoint/archive files: **0**
- Large files over 25 MB: **0**
- IEEE/general reproducibility checklist: **15 / 15**
- Reproducibility scripts passing: **3 / 3**
- GitHub-ready after final manual edits: **False**

## Blocking issues

- Public-facing files contain local absolute paths. These should be removed or moved to internal audit only.
- LICENSE/CITATION still contains placeholder license terms. Choose a license before public release.

## Review items

- CITATION.cff still has GitHub USERNAME placeholder; replace after creating the GitHub repository.
- Public files contain TODO/TBD/placeholder text. Review before final release.

## Required manual edits before push

- Replace USERNAME in CITATION.cff after creating GitHub repository.
- Replace LICENSE placeholder with a real license, preferably MIT for code only, if approved.
- Keep dataset restriction wording for SEED-IV and FACED.
- Avoid putting IEEE Access in the repository name; keep CLISA-EWU-reproducibility.
- Do not upload local audit folders outside the staged repo.

## Scientific cautions to preserve

- Ablation table has 11 verified rows; do not claim 14 verified rows unless checkpoint-verified.
- FACED is used as a processed 30-channel feature representation for domain-gap analysis.
- DANCE/M25 should be reported cautiously unless final LOSO value is reconciled.
- Raw SEED-IV/FACED data are not redistributed.
