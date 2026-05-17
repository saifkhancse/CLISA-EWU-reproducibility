"""
reproduce_tables.py

Creates Markdown and normalized CSV versions of the final manuscript tables
from staged repository CSV files.

This script does not require tabulate.
It does not recompute model training.
It reproduces paper-ready tables from verified result artifacts.
"""

from pathlib import Path
from datetime import datetime
import json
import csv
import math

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas is required. Install with: pip install pandas") from e


ROOT = Path(__file__).resolve().parents[1]
FINAL_TABLE_DIR = ROOT / "results" / "final_tables"
OUT_DIR = ROOT / "results" / "reproduced_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


TABLE_TITLES = {
    "table_5_1_classical_main.csv": "Table 5.1. Classical machine-learning benchmark",
    "table_5_2_deep_main.csv": "Table 5.2. Deep-learning and adaptation benchmark",
    "table_5_3_dance_loso_verified.csv": "Table 5.3. DANCE LOSO verified results",
    "table_5_3b_dance_reproduction.csv": "Table 5.3b. DANCE reproduction/caution table",
    "table_5_4_ablations.csv": "Table 5.4. Ablation study",
    "table_5_5_proto_gain.csv": "Table 5.5. Proto-A versus Proto-B gain",
    "table_5_6_channel_efficiency.csv": "Table 5.6. Channel-efficiency retention",
    "table_5_7_per_class_top_models.csv": "Table 5.7. Per-class top-model comparison",
    "audit_statistical_tests.csv": "Audit table. Statistical tests",
    "audit_discovered_optional_assets.csv": "Audit table. Optional assets",
    "table_audit_sources.csv": "Audit table. Source mapping",
}


def clean_column_name(name):
    name = str(name).strip()
    name = name.replace("_", " ")
    return name


def format_float(x):
    try:
        if x is None:
            return ""
        val = float(x)
        if math.isnan(val):
            return ""
    except Exception:
        return x

    if abs(val) < 1e-12:
        return "0"
    if abs(val) < 0.001:
        return f"{val:.2e}"
    if abs(val) < 1:
        return f"{val:.4f}"
    if abs(val) < 100:
        return f"{val:.3f}"
    return f"{val:.1f}"


def clean_df(df):
    out = df.copy()

    # Remove unnamed index columns
    out = out[[c for c in out.columns if not str(c).lower().startswith("unnamed")]]

    # Clean column names
    out.columns = [clean_column_name(c) for c in out.columns]

    # Format numeric-looking columns
    for col in out.columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        non_na = numeric.notna().sum()
        if len(out) > 0 and non_na >= max(1, int(0.5 * len(out))):
            out[col] = out[col].apply(format_float)
        else:
            out[col] = out[col].fillna("").astype(str)

    return out


def escape_md(value):
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("|", "\\|")
    return text


def df_to_markdown(df):
    columns = [escape_md(c) for c in df.columns]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = [escape_md(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_csv(path, rows, fieldnames=None):
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


def main():
    table_files = sorted(FINAL_TABLE_DIR.glob("*.csv"))

    index_rows = []
    combined_md = []
    combined_md.append("# Reproduced manuscript tables")
    combined_md.append("")
    combined_md.append(f"Generated: `{timestamp}`")
    combined_md.append("")

    for path in table_files:
        try:
            df = pd.read_csv(path)
            cleaned = clean_df(df)

            stem = path.stem
            title = TABLE_TITLES.get(path.name, stem.replace("_", " ").title())

            out_csv = OUT_DIR / f"{stem}.normalized.csv"
            out_md = OUT_DIR / f"{stem}.md"

            cleaned.to_csv(out_csv, index=False)

            table_md = df_to_markdown(cleaned)

            md = []
            md.append(f"# {title}")
            md.append("")
            md.append(f"Source file: `{path.relative_to(ROOT).as_posix()}`")
            md.append("")
            md.append(table_md)
            md.append("")

            out_md.write_text("\n".join(md), encoding="utf-8")

            combined_md.append(f"## {title}")
            combined_md.append("")
            combined_md.append(f"Source: `{path.relative_to(ROOT).as_posix()}`")
            combined_md.append("")
            combined_md.append(table_md)
            combined_md.append("")

            index_rows.append({
                "source_file": path.relative_to(ROOT).as_posix(),
                "reproduced_csv": out_csv.relative_to(ROOT).as_posix(),
                "reproduced_md": out_md.relative_to(ROOT).as_posix(),
                "rows": len(df),
                "columns": len(df.columns),
                "status": "OK",
                "title": title,
            })

        except Exception as e:
            index_rows.append({
                "source_file": path.relative_to(ROOT).as_posix(),
                "reproduced_csv": "",
                "reproduced_md": "",
                "rows": "",
                "columns": "",
                "status": f"ERROR: {repr(e)}",
                "title": TABLE_TITLES.get(path.name, path.stem),
            })

    write_csv(
        OUT_DIR / "reproduced_tables_index.csv",
        index_rows,
        ["source_file", "reproduced_csv", "reproduced_md", "rows", "columns", "status", "title"]
    )

    (OUT_DIR / "reproduced_tables_all.md").write_text("\n".join(combined_md), encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "final_table_dir": "results/final_tables",
        "output_dir": "results/reproduced_tables",
        "tables_found": len(table_files),
        "tables_reproduced_ok": sum(1 for r in index_rows if r["status"] == "OK"),
        "tables_with_error": sum(1 for r in index_rows if r["status"] != "OK"),
        "tables": index_rows,
    }

    (OUT_DIR / "reproduced_tables_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("reproduce_tables.py completed")
    print(f"Tables found: {summary['tables_found']}")
    print(f"Tables reproduced OK: {summary['tables_reproduced_ok']}")
    print(f"Tables with error: {summary['tables_with_error']}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
