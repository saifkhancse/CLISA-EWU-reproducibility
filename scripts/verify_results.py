"""
verify_results.py

Repository verification script for CLISA-EWU reproducibility package.

This script checks:
- Required final table files.
- Key manuscript values.
- BYOL/M22 evidence.
- DANCE/M25 caution markers.
- Ablation row count.
- FACED processed shape evidence.
- Accidental inclusion of large data/checkpoint files.

It uses only staged repository CSV/metadata files.
It does not require raw SEED-IV or FACED access.
"""

from pathlib import Path
from datetime import datetime
import json
import csv
import re
import sys

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas is required. Install with: pip install pandas") from e


ROOT = Path(__file__).resolve().parents[1]
FINAL_TABLE_DIR = ROOT / "results" / "final_tables"
AGG_DIR = ROOT / "results" / "aggregates"
AUDIT_DIR = ROOT / "results" / "audit" / "verification"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REQUIRED_FINAL_TABLES = [
    "table_5_1_classical_main.csv",
    "table_5_2_deep_main.csv",
    "table_5_3_dance_loso_verified.csv",
    "table_5_3b_dance_reproduction.csv",
    "table_5_4_ablations.csv",
    "table_5_5_proto_gain.csv",
    "table_5_6_channel_efficiency.csv",
    "table_5_7_per_class_top_models.csv",
    "table_audit_sources.csv",
]

KEY_VALUES = {
    "CLISA_EWU_62ch_AccB_expected_0.6801": 0.6801,
    "CLISA_EWU_6ch_AccB_expected_0.6199": 0.6199,
    "BYOL_M22_62ch_AccB_expected_0.6089": 0.6089,
    "BYOL_M22_6ch_AccB_expected_0.6124": 0.6124,
    "DANCE_PhaseB_value_caution_0.6170": 0.6170,
    "DANCE_possible_LOSO_value_caution_0.5190": 0.5190,
    "DANCE_M25_mismatch_caution_0.0236": 0.0236,
}

KEY_TERMS = [
    "CLISA",
    "CLISA-EWU",
    "M20",
    "BYOL",
    "M22",
    "DANCE",
    "M25",
    "M26",
    "FACED",
    "SEED-IV",
    "Proto-A",
    "Proto-B",
    "A12",
    "A13",
    "A14",
    "A16",
    "GraphFormer",
    "M27",
    "M28",
]

DISALLOWED_EXTENSIONS_IN_GIT = [
    ".npy", ".npz", ".mat", ".h5", ".hdf5", ".edf", ".set", ".fdt",
    ".pt", ".pth", ".ckpt", ".onnx", ".joblib", ".sav",
]


def read_csv_safe(path: Path):
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return None, repr(e)


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if rows:
            keys = set()
            for r in rows:
                keys.update(r.keys())
            fieldnames = sorted(keys)
        else:
            fieldnames = ["empty"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def dataframe_text(df: pd.DataFrame) -> str:
    return df.astype(str).fillna("").to_csv(index=False)


def value_present_in_df(df: pd.DataFrame, value: float, tolerance: float = 5e-5) -> bool:
    # Numeric scan
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if ((numeric - value).abs() <= tolerance).any():
            return True

    # String scan with rounded forms
    text = dataframe_text(df)
    variants = {
        f"{value:.4f}",
        f"{value:.3f}",
        str(value),
        f"{value * 100:.2f}",
        f"{value * 100:.1f}",
    }
    return any(v in text for v in variants)


def search_terms_in_df(df: pd.DataFrame, file_name: str):
    rows = []
    for idx, row in df.iterrows():
        row_text = " | ".join(str(x) for x in row.values)
        for term in KEY_TERMS:
            if term.lower() in row_text.lower():
                rows.append({
                    "file": file_name,
                    "row_index": idx,
                    "matched_term": term,
                    "row_text": row_text[:2000],
                })
    return rows


def scan_csv_files():
    csv_files = sorted((ROOT / "results").rglob("*.csv"))
    table_summaries = []
    term_hits = []
    value_hits = []

    for path in csv_files:
        rel = path.relative_to(ROOT).as_posix()
        df, err = read_csv_safe(path)

        if df is None:
            table_summaries.append({
                "file": rel,
                "status": "READ_ERROR",
                "rows": "",
                "columns": "",
                "column_names": "",
                "error": err,
            })
            continue

        table_summaries.append({
            "file": rel,
            "status": "OK",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": "; ".join(map(str, df.columns)),
            "error": "",
        })

        term_hits.extend(search_terms_in_df(df, rel))

        for label, value in KEY_VALUES.items():
            if value_present_in_df(df, value):
                value_hits.append({
                    "key_claim": label,
                    "value": value,
                    "found_in": rel,
                })

    return table_summaries, term_hits, value_hits


def check_required_tables():
    rows = []
    for fname in REQUIRED_FINAL_TABLES:
        path = FINAL_TABLE_DIR / fname
        exists = path.exists()
        df = None
        err = ""

        if exists:
            df, err = read_csv_safe(path)

        rows.append({
            "required_file": fname,
            "exists": exists,
            "rows": "" if df is None else len(df),
            "columns": "" if df is None else len(df.columns),
            "status": "OK" if exists and df is not None else "MISSING_OR_READ_ERROR",
            "error": err or "",
        })

    return rows


def check_ablation_table():
    path = FINAL_TABLE_DIR / "table_5_4_ablations.csv"
    if not path.exists():
        return {
            "file": path.relative_to(ROOT).as_posix(),
            "status": "MISSING",
            "rows": "",
            "warning": "Ablation table not found.",
        }

    df, err = read_csv_safe(path)
    if df is None:
        return {
            "file": path.relative_to(ROOT).as_posix(),
            "status": "READ_ERROR",
            "rows": "",
            "warning": err,
        }

    warning = ""
    if len(df) == 11:
        warning = "11 rows found. This matches the verified audit but not the earlier expected 14-row ablation plan."
    elif len(df) < 14:
        warning = f"{len(df)} rows found. Treat missing ablation rows as unverified unless checkpoints are confirmed."
    else:
        warning = f"{len(df)} rows found. Confirm that all rows are checkpoint-verified."

    return {
        "file": path.relative_to(ROOT).as_posix(),
        "status": "OK",
        "rows": len(df),
        "columns": len(df.columns),
        "warning": warning,
    }


def check_dance_tables():
    files = [
        FINAL_TABLE_DIR / "table_5_3_dance_loso_verified.csv",
        FINAL_TABLE_DIR / "table_5_3b_dance_reproduction.csv",
        AGG_DIR / "loso" / "dance_loso.csv",
        AGG_DIR / "phaseB_reproduction" / "phaseB_reproduce_results.csv",
    ]

    rows = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if not path.exists():
            rows.append({
                "file": rel,
                "status": "MISSING",
                "rows": "",
                "columns": "",
                "contains_0_6170": "",
                "contains_0_5190": "",
                "note": "",
            })
            continue

        df, err = read_csv_safe(path)
        if df is None:
            rows.append({
                "file": rel,
                "status": "READ_ERROR",
                "rows": "",
                "columns": "",
                "contains_0_6170": "",
                "contains_0_5190": "",
                "note": err,
            })
            continue

        text = dataframe_text(df)

        rows.append({
            "file": rel,
            "status": "OK",
            "rows": len(df),
            "columns": len(df.columns),
            "contains_0_6170": "0.617" in text or "61.70" in text,
            "contains_0_5190": "0.519" in text or "51.90" in text,
            "note": "Use LOSO verified table for manuscript; reproduction table is historical/cautionary.",
        })

    return rows


def check_faced_shape():
    npy_shape_file = ROOT / "results" / "audit" / "cell02_npy_npz_shapes.csv"
    rows = []

    if not npy_shape_file.exists():
        return [{
            "status": "MISSING",
            "file": "results/audit/cell02_npy_npz_shapes.csv",
            "evidence": "",
            "note": "Cannot verify FACED processed shape evidence.",
        }]

    df, err = read_csv_safe(npy_shape_file)
    if df is None:
        return [{
            "status": "READ_ERROR",
            "file": "results/audit/cell02_npy_npz_shapes.csv",
            "evidence": "",
            "note": err,
        }]

    mask = df.astype(str).apply(
        lambda col: col.str.contains("faced_X_32ch.npy", case=False, regex=False, na=False)
    ).any(axis=1)

    matched = df[mask]

    if matched.empty:
        return [{
            "status": "NOT_FOUND",
            "file": "results/audit/cell02_npy_npz_shapes.csv",
            "evidence": "",
            "note": "No faced_X_32ch.npy row found.",
        }]

    for _, r in matched.iterrows():
        rows.append({
            "status": "OK",
            "file": str(r.get("relative_path", "")),
            "shape": str(r.get("shape", "")),
            "dtype": str(r.get("dtype", "")),
            "note": "Use this as evidence that processed FACED feature representation is 30 channels: (110208, 30, 5).",
        })

    return rows


def scan_disallowed_files():
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        ext = path.suffix.lower()
        if ext in DISALLOWED_EXTENSIONS_IN_GIT:
            rows.append({
                "file": path.relative_to(ROOT).as_posix(),
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "recommendation": "Do not commit to GitHub. Use dataset access instructions or external release if allowed.",
            })

    return rows


def main():
    required_rows = check_required_tables()
    table_summaries, term_hits, value_hits = scan_csv_files()
    ablation_check = check_ablation_table()
    dance_rows = check_dance_tables()
    faced_shape_rows = check_faced_shape()
    disallowed_rows = scan_disallowed_files()

    found_claims = {r["key_claim"] for r in value_hits}
    missing_claims = [
        {"key_claim": label, "value": value, "status": "NOT_FOUND_IN_STAGED_CSVS"}
        for label, value in KEY_VALUES.items()
        if label not in found_claims
    ]

    # Save machine-readable outputs
    write_csv(AUDIT_DIR / "required_final_tables_check.csv", required_rows)
    write_csv(AUDIT_DIR / "csv_table_summaries.csv", table_summaries)
    write_csv(AUDIT_DIR / "key_term_hits_sample.csv", term_hits[:500])
    write_csv(AUDIT_DIR / "key_value_hits.csv", value_hits)
    write_csv(AUDIT_DIR / "key_value_missing.csv", missing_claims)
    write_csv(AUDIT_DIR / "dance_table_check.csv", dance_rows)
    write_csv(AUDIT_DIR / "faced_shape_check.csv", faced_shape_rows)
    write_csv(AUDIT_DIR / "disallowed_large_or_data_files.csv", disallowed_rows)

    summary = {
        "timestamp": timestamp,
        "root": str(ROOT),
        "required_tables_total": len(REQUIRED_FINAL_TABLES),
        "required_tables_ok": sum(1 for r in required_rows if r["status"] == "OK"),
        "csv_files_scanned": len(table_summaries),
        "key_term_hits_total": len(term_hits),
        "key_term_hits_saved_sample": min(len(term_hits), 500),
        "key_value_hits": len(value_hits),
        "key_values_missing": len(missing_claims),
        "ablation_check": ablation_check,
        "dance_tables_checked": len(dance_rows),
        "faced_shape_rows": faced_shape_rows,
        "disallowed_files_found": len(disallowed_rows),
    }

    (AUDIT_DIR / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Markdown report
    md = []
    md.append("# CLISA-EWU repository verification report")
    md.append("")
    md.append(f"Generated: `{timestamp}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Required final tables OK: **{summary['required_tables_ok']} / {summary['required_tables_total']}**")
    md.append(f"- CSV files scanned: **{summary['csv_files_scanned']}**")
    md.append(f"- Key value hits: **{summary['key_value_hits']}**")
    md.append(f"- Key values missing from staged CSVs: **{summary['key_values_missing']}**")
    md.append(f"- Disallowed data/checkpoint files found in staged repo: **{summary['disallowed_files_found']}**")
    md.append("")
    md.append("## Ablation table check")
    md.append("")
    md.append(f"- Status: `{ablation_check.get('status')}`")
    md.append(f"- Rows: `{ablation_check.get('rows')}`")
    md.append(f"- Warning: {ablation_check.get('warning')}")
    md.append("")
    md.append("## FACED shape evidence")
    md.append("")
    for row in faced_shape_rows:
        md.append(f"- `{row.get('file')}` → shape `{row.get('shape')}`, dtype `{row.get('dtype')}`")
    md.append("")
    md.append("## DANCE/M25 caution")
    md.append("")
    md.append("Use the LOSO verified table for manuscript claims. Treat Phase-B reproduction values as historical or cautionary unless reconciled.")
    md.append("")
    md.append("## Missing key values")
    md.append("")
    if missing_claims:
        for row in missing_claims:
            md.append(f"- `{row['key_claim']}` = `{row['value']}` was not found in staged CSVs.")
    else:
        md.append("- None.")
    md.append("")
    md.append("## Disallowed files")
    md.append("")
    if disallowed_rows:
        md.append("The staged repository still contains data/checkpoint-like files. Review before GitHub upload.")
    else:
        md.append("No disallowed data/checkpoint extensions were found in the staged repository.")
    md.append("")

    (AUDIT_DIR / "verification_report.md").write_text("\n".join(md), encoding="utf-8")

    print("verify_results.py completed")
    print(f"Required final tables OK: {summary['required_tables_ok']} / {summary['required_tables_total']}")
    print(f"Key value hits: {summary['key_value_hits']}")
    print(f"Key values missing: {summary['key_values_missing']}")
    print(f"Disallowed files found: {summary['disallowed_files_found']}")
    print(f"Report: {AUDIT_DIR / 'verification_report.md'}")


if __name__ == "__main__":
    main()
