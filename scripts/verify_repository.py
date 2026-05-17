from pathlib import Path
from datetime import datetime
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "audit" / "repository_readiness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "LICENSE",
    "CITATION.cff",
    "docs/dataset_access.md",
    "docs/checkpoint_policy.md",
    "docs/reproducibility_notes.md",
    "data_splits/seediv_loso_folds.json",
    "data_splits/seeds_1_7_21.json",
    "data_splits/channel_indices_seediv_62_to_6.json",
    "data_splits/label_map_seediv_4class.json",
    "configs/faced_processing_contract.json",
    "scripts/verify_results.py",
    "scripts/reproduce_tables.py",
    "scripts/generate_data_splits.py",
]

REQUIRED_FINAL_TABLES = [
    "results/final_tables/table_5_1_classical_main.csv",
    "results/final_tables/table_5_2_deep_main.csv",
    "results/final_tables/table_5_3_dance_loso_verified.csv",
    "results/final_tables/table_5_3b_dance_reproduction.csv",
    "results/final_tables/table_5_4_ablations.csv",
    "results/final_tables/table_5_5_proto_gain.csv",
    "results/final_tables/table_5_6_channel_efficiency.csv",
    "results/final_tables/table_5_7_per_class_top_models.csv",
    "results/final_tables/table_audit_sources.csv",
]

DISALLOWED_EXTENSIONS = {
    ".npy", ".npz", ".mat", ".h5", ".hdf5", ".edf", ".set", ".fdt",
    ".pt", ".pth", ".ckpt", ".onnx", ".joblib", ".sav",
    ".docx", ".pptx", ".zip"
}

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

def check_files(paths, group):
    rows = []
    for rel in paths:
        path = ROOT / rel
        rows.append({
            "group": group,
            "path": rel,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
            "status": "OK" if path.exists() else "MISSING",
        })
    return rows

def scan_disallowed():
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(ROOT).as_posix()

        if ".git/" in rel:
            continue

        ext = path.suffix.lower()

        if ext in DISALLOWED_EXTENSIONS:
            rows.append({
                "path": rel,
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "status": "REVIEW",
                "recommendation": "Do not commit raw data, checkpoints, Office drafts, or zip files."
            })

    return rows

def check_json_validity(paths):
    rows = []
    for rel in paths:
        path = ROOT / rel

        if not path.exists():
            rows.append({"path": rel, "status": "MISSING", "error": ""})
            continue

        try:
            json.loads(path.read_text(encoding="utf-8"))
            rows.append({"path": rel, "status": "OK", "error": ""})
        except Exception as e:
            rows.append({"path": rel, "status": "JSON_ERROR", "error": repr(e)})

    return rows

def read_json_safe(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def main():
    required_rows = []
    required_rows.extend(check_files(REQUIRED_FILES, "required_repo_files"))
    required_rows.extend(check_files(REQUIRED_FINAL_TABLES, "required_final_tables"))

    json_rows = check_json_validity([
        "data_splits/seediv_loso_folds.json",
        "data_splits/seeds_1_7_21.json",
        "data_splits/channel_indices_seediv_62_to_6.json",
        "data_splits/label_map_seediv_4class.json",
        "configs/faced_processing_contract.json",
    ])

    disallowed_rows = scan_disallowed()

    reproduced_summary = read_json_safe(ROOT / "results" / "reproduced_tables" / "reproduced_tables_summary.json")
    verification_summary = read_json_safe(ROOT / "results" / "audit" / "verification" / "verification_summary.json")

    required_ok = sum(1 for r in required_rows if r["status"] == "OK")
    required_total = len(required_rows)
    json_ok = sum(1 for r in json_rows if r["status"] == "OK")
    json_total = len(json_rows)

    blocking_issues = []

    if required_ok != required_total:
        blocking_issues.append("Some required repository files or final tables are missing.")

    if json_ok != json_total:
        blocking_issues.append("Some split/config JSON files are invalid or missing.")

    if disallowed_rows:
        blocking_issues.append("Repository contains disallowed data/checkpoint/draft/archive file extensions.")

    if reproduced_summary.get("tables_reproduced_ok") != reproduced_summary.get("tables_found"):
        blocking_issues.append("Not all final tables reproduce successfully.")

    if verification_summary.get("required_tables_ok") != verification_summary.get("required_tables_total"):
        blocking_issues.append("Final table verification did not pass.")

    if verification_summary.get("key_values_missing", 999) != 0:
        blocking_issues.append("Some expected key manuscript values are missing from staged CSVs.")

    summary = {
        "timestamp": timestamp,
        "repository_root": str(ROOT),
        "required_files_ok": required_ok,
        "required_files_total": required_total,
        "json_files_ok": json_ok,
        "json_files_total": json_total,
        "disallowed_files_found": len(disallowed_rows),
        "tables_found": reproduced_summary.get("tables_found", ""),
        "tables_reproduced_ok": reproduced_summary.get("tables_reproduced_ok", ""),
        "verification_required_tables_ok": verification_summary.get("required_tables_ok", ""),
        "verification_required_tables_total": verification_summary.get("required_tables_total", ""),
        "verification_key_values_missing": verification_summary.get("key_values_missing", ""),
        "blocking_issues": blocking_issues,
        "github_ready_basic": len(blocking_issues) == 0,
        "known_scientific_cautions": [
            "Ablation table has 11 verified rows; do not claim 14 verified rows unless checkpoints are confirmed.",
            "FACED is used as processed 30-channel feature representation for domain-gap analysis.",
            "DANCE/M25 should be reported with caution unless final LOSO value is reconciled.",
            "Raw SEED-IV/FACED data are not redistributed."
        ]
    }

    write_csv(OUT_DIR / "required_files_check.csv", required_rows)
    write_csv(OUT_DIR / "json_validity_check.csv", json_rows)
    write_csv(OUT_DIR / "disallowed_file_check.csv", disallowed_rows)

    (OUT_DIR / "repository_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    md = []
    md.append("# Repository readiness report")
    md.append("")
    md.append(f"Generated: `{timestamp}`")
    md.append("")
    md.append("## Basic GitHub readiness")
    md.append("")
    md.append(f"- Required files OK: **{required_ok} / {required_total}**")
    md.append(f"- JSON files OK: **{json_ok} / {json_total}**")
    md.append(f"- Disallowed files found: **{len(disallowed_rows)}**")
    md.append(f"- Tables reproduced OK: **{reproduced_summary.get('tables_reproduced_ok', '')} / {reproduced_summary.get('tables_found', '')}**")
    md.append(f"- Final table verification: **{verification_summary.get('required_tables_ok', '')} / {verification_summary.get('required_tables_total', '')}**")
    md.append(f"- Key values missing: **{verification_summary.get('key_values_missing', '')}**")
    md.append("")
    md.append(f"**Basic GitHub ready:** `{summary['github_ready_basic']}`")
    md.append("")
    md.append("## Blocking issues")
    md.append("")
    if blocking_issues:
        for issue in blocking_issues:
            md.append(f"- {issue}")
    else:
        md.append("- None.")
    md.append("")
    md.append("## Scientific cautions")
    md.append("")
    for caution in summary["known_scientific_cautions"]:
        md.append(f"- {caution}")
    md.append("")

    (OUT_DIR / "repository_readiness_report.md").write_text("\n".join(md), encoding="utf-8")

    print("verify_repository.py completed")
    print(f"Required files OK: {required_ok} / {required_total}")
    print(f"JSON files OK: {json_ok} / {json_total}")
    print(f"Disallowed files found: {len(disallowed_rows)}")
    print(f"Basic GitHub ready: {summary['github_ready_basic']}")
    print(f"Report: {OUT_DIR / 'repository_readiness_report.md'}")

    return 0 if summary["github_ready_basic"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
