# Auto-exported raw code from notebook: save the pts.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 0
# Categories: preprocessing, model_definition, training, results_tables, audit_verification
# ==============================================================================
# ============================================================
# CHECKUP CELL — find folders/files needed for .pth -> .pt export
# Works even if you are not exactly at project root.
# ============================================================
from pathlib import Path
import os, json, re, glob
from collections import defaultdict
import numpy as np
import torch

# -----------------------------
# 0) Settings
# -----------------------------
EXPECTED_RUNS = 45
DEEP_6CH_MODELS = [f"M{i:02d}" for i in range(11, 25)] + ["M26"]   # M11..M24 + M26
CLASSICAL_MODELS = [f"M{i:02d}" for i in range(1, 11)]              # M01..M10
TEACHER_MODEL = "M25"

cwd = Path(".").resolve()

# -----------------------------
# 1) Helpers
# -----------------------------
def find_dir_by_name(root: Path, tail_parts):
    p = root.joinpath(*tail_parts)
    if p.exists():
        return p

    matches = []
    tail_name = tail_parts[-1]
    for q in root.rglob(tail_name):
        try:
            if q.is_dir() and list(q.parts[-len(tail_parts):]) == list(tail_parts):
                matches.append(q)
        except Exception:
            pass
    return sorted(matches, key=lambda x: len(str(x)))[0] if matches else None

def count_files(folder: Path, pattern: str):
    if folder is None or not folder.exists():
        return 0
    return len(list(folder.glob(pattern)))

def recursive_find(root: Path, pattern: str):
    try:
        return sorted(root.rglob(pattern))
    except Exception:
        return []

def try_load_state_dict(p: Path):
    try:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj.keys()):
                if any(torch.is_tensor(v) for v in obj.values()):
                    return obj, "state_dict"
                if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    return obj["state_dict"], "checkpoint[state_dict]"
        return None, f"unsupported object type: {type(obj)}"
    except Exception as e:
        return None, f"load failed: {e}"

def fmt_shape(path):
    try:
        arr = np.load(path)
        return f"{tuple(arr.shape)}  dtype={arr.dtype}"
    except Exception as e:
        return f"load failed: {e}"

# -----------------------------
# 2) Locate project folders
# -----------------------------
FEATURES_DIR = find_dir_by_name(cwd, ["features"])
CKPT_DIR     = find_dir_by_name(cwd, ["checkpoints", "loso_results"])
MDL_DIR      = find_dir_by_name(cwd, ["checkpoints", "model_weights"])
RESULTS_DEEP = find_dir_by_name(cwd, ["results", "deep_models_seediv"])
RESULTS_CLF  = find_dir_by_name(cwd, ["results", "classical_ml"])

print("=" * 88)
print("EXPORT CHECKUP")
print("=" * 88)
print(f"CWD           : {cwd}")
print(f"FEATURES_DIR  : {FEATURES_DIR}")
print(f"CKPT_DIR      : {CKPT_DIR}")
print(f"MDL_DIR       : {MDL_DIR}")
print(f"RESULTS_DEEP  : {RESULTS_DEEP}")
print(f"RESULTS_CLF   : {RESULTS_CLF}")

# -----------------------------
# 3) Required feature files
# -----------------------------
required_features = {
    "seed_iv_X_62ch.npy": None,
    "seed_iv_X_6ch.npy": None,
    "seed_iv_y_4cls.npy": None,
    "seed_iv_subjects.npy": None,
}

print("\n[Feature files]")
if FEATURES_DIR and FEATURES_DIR.exists():
    for name in required_features:
        p = FEATURES_DIR / name
        ok = p.exists()
        print(f"{'OK ' if ok else 'MISS'}  {name:22s} -> {p if ok else 'not found'}")
        if ok:
            print(f"      shape: {fmt_shape(p)}")
else:
    print("MISS  features folder not found")

# -----------------------------
# 4) Checkpoint JSON coverage
# -----------------------------
print("\n[Checkpoint JSON coverage]")
if CKPT_DIR and CKPT_DIR.exists():
    for mid in DEEP_6CH_MODELS:
        n = count_files(CKPT_DIR, f"{mid}_6ch_seed*_fold*.json")
        print(f"{mid} [6ch]  json: {n:2d}/{EXPECTED_RUNS}")
    n25 = count_files(CKPT_DIR, f"{TEACHER_MODEL}_62ch_seed*_fold*.json")
    print(f"{TEACHER_MODEL} [62ch] json: {n25:2d}/{EXPECTED_RUNS}")
else:
    print("MISS  checkpoint folder not found")

# -----------------------------
# 5) Weight file coverage (.pth / .pt)
# -----------------------------
print("\n[Model weight coverage in checkpoints/model_weights]")
pth_counts = {}
pt_counts = {}

if MDL_DIR and MDL_DIR.exists():
    for mid in DEEP_6CH_MODELS:
        pth_n = count_files(MDL_DIR, f"{mid}_6ch_s*_f*_best.pth")
        pt_n  = count_files(MDL_DIR, f"{mid}_6ch_s*_f*_best.pt") + count_files(MDL_DIR, f"{mid}_6ch_s*_f*.pt")
        pth_counts[mid] = pth_n
        pt_counts[mid] = pt_n
        print(f"{mid} [6ch]  pth: {pth_n:2d}/{EXPECTED_RUNS}   pt: {pt_n:2d}")
    pth_n25 = count_files(MDL_DIR, f"{TEACHER_MODEL}_62ch_s*_f*_best.pth")
    pt_n25  = count_files(MDL_DIR, f"{TEACHER_MODEL}_62ch_s*_f*_best.pt") + count_files(MDL_DIR, f"{TEACHER_MODEL}_62ch_s*_f*.pt")
    print(f"{TEACHER_MODEL} [62ch] pth: {pth_n25:2d}/{EXPECTED_RUNS}   pt: {pt_n25:2d}")
else:
    print("MISS  model_weights folder not found")

# -----------------------------
# 6) Recursive fallback scan for stray weights / 'another notebook'
# -----------------------------
print("\n[Recursive scan from current folder for stray weight files]")
interesting = []
patterns = [
    "*M25*62ch*.pth", "*M26*6ch*.pth", "*student*.pth", "*teacher*.pth",
    "*M25*62ch*.pt",  "*M26*6ch*.pt",  "*student*.pt",  "*teacher*.pt",
]
seen = set()
for pat in patterns:
    for p in recursive_find(cwd, pat):
        if p not in seen:
            seen.add(p)
            interesting.append(p)

if interesting:
    for p in interesting[:50]:
        print("FOUND", p)
    if len(interesting) > 50:
        print(f"... and {len(interesting)-50} more")
else:
    print("No stray teacher/student .pth/.pt files found under current folder")

# -----------------------------
# 7) Quick integrity check: can a weight file be loaded?
# -----------------------------
print("\n[Sample load test]")
sample_candidates = []
if MDL_DIR and MDL_DIR.exists():
    sample_candidates.extend(sorted(MDL_DIR.glob("M26_6ch_s*_f*_best.pth")))
    sample_candidates.extend(sorted(MDL_DIR.glob("M25_62ch_s*_f*_best.pth")))
    sample_candidates.extend(sorted(MDL_DIR.glob("M11_6ch_s*_f*_best.pth")))
sample_candidates.extend([p for p in interesting if p.suffix == ".pth"])

sample_seen = set()
sample_candidates = [p for p in sample_candidates if not (p in sample_seen or sample_seen.add(p))]

if sample_candidates:
    p = sample_candidates[0]
    sd, status = try_load_state_dict(p)
    print(f"Testing: {p}")
    print(f"Status : {status}")
    if isinstance(sd, dict):
        keys = list(sd.keys())
        print(f"n_keys : {len(keys)}")
        print("first keys:")
        for k in keys[:12]:
            v = sd[k]
            shp = tuple(v.shape) if torch.is_tensor(v) else type(v)
            print(f"  - {k} -> {shp}")
else:
    print("No .pth file available yet to test-load")

# -----------------------------
# 8) Classical ML note
# -----------------------------
print("\n[Classical ML note: M01-M10]")
joblib_hits = []
for ext in ("*.joblib", "*.pkl", "*.pickle"):
    joblib_hits.extend(recursive_find(cwd, ext))
joblib_hits = sorted(set(joblib_hits))
print(f"Found serialized classical-model files: {len(joblib_hits)}")
for p in joblib_hits[:20]:
    print("  ", p)
if len(joblib_hits) > 20:
    print(f"  ... and {len(joblib_hits)-20} more")
print("M01-M10 cannot be recreated from .pth unless you separately saved sklearn objects.")

# -----------------------------
# 9) Final readiness summary
# -----------------------------
print("\n[Readiness summary]")
exportable_6ch = [m for m in DEEP_6CH_MODELS if pth_counts.get(m, 0) > 0]
missing_6ch    = [m for m in DEEP_6CH_MODELS if pth_counts.get(m, 0) == 0]
teacher_ready  = (MDL_DIR is not None and count_files(MDL_DIR, f"{TEACHER_MODEL}_62ch_s*_f*_best.pth") > 0)

print(f"6ch deep models with at least one .pth : {exportable_6ch if exportable_6ch else 'NONE'}")
print(f"6ch deep models with NO .pth found     : {missing_6ch if missing_6ch else 'NONE'}")
print(f"M25 teacher weights present            : {teacher_ready}")

print("\nDone. Paste the output here, then I’ll give you the exact export cells.")


# ==============================================================================
# Notebook cell 1
# Categories: preprocessing, training
# ==============================================================================
# ============================================================
# PROOF CELL — show where JSON is saved vs where .pth is saved
# ============================================================
import json
from pathlib import Path

nb_path = Path("02_deep_models.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

KEYS = [
    "save_weights=False",
    "if save_weights:",
    "torch.save(",
    "ck_save(",
    "weight_path(",
    "_MID = 'M25'",
    "_MID = 'M26'",
]

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    hits = [k for k in KEYS if k in src]
    if hits:
        print("\n" + "="*80)
        print(f"CELL {i}  | hits: {hits}")
        print("="*80)
        for ln, line in enumerate(src.splitlines(), start=1):
            if any(k in line for k in KEYS):
                print(f"{ln:03d}: {line}")


# ==============================================================================
# Notebook cell 2
# Categories: other
# ==============================================================================
# ============================================================
# EXHAUSTIVE SEARCH CELL — look everywhere for M11–M24 weights
# ============================================================
from pathlib import Path
import os

MODEL_IDS = [f"M{i:02d}" for i in range(11, 25)]
EXTS = [".pth", ".pt", ".ckpt", ".bin", ".pkl"]

roots = []
for drive in ["C:/", "D:/", "E:/", "F:/", "G:/"]:
    p = Path(drive)
    if p.exists():
        roots.append(p)

hits = []

def looks_relevant(name):
    lname = name.lower()
    return (
        any(mid.lower() in lname for mid in MODEL_IDS)
        or ("6ch" in lname and any(ext in lname for ext in EXTS))
        or ("model" in lname and any(ext in lname for ext in EXTS))
        or ("weight" in lname and any(ext in lname for ext in EXTS))
        or ("best" in lname and any(ext in lname for ext in EXTS))
    )

for root in roots:
    print(f"\nSearching {root} ...")
    for dirpath, dirnames, filenames in os.walk(root):
        # skip heavy/system folders
        skip_parts = ["Windows", "Program Files", "Program Files (x86)", "$Recycle.Bin", "AppData\\Local\\Temp"]
        if any(part.lower() in dirpath.lower() for part in [s.lower() for s in skip_parts]):
            continue

        for fn in filenames:
            if looks_relevant(fn):
                full = str(Path(dirpath) / fn)
                hits.append(full)
                print(full)

print("\n" + "="*80)
print("TOTAL HITS:", len(hits))
print("="*80)


# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ============================================================
# CELL 1 — DRY RUN / PREVIEW
# ============================================================
from pathlib import Path
from collections import defaultdict
import hashlib
import os

PROJECT_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")
DEST_DIR = PROJECT_ROOT / "checkpoints" / "all_model_weights"

SEARCH_ROOTS = [
    PROJECT_ROOT / "checkpoints",                              # current project checkpoints root
    PROJECT_ROOT / "checkpoints" / "model_weights",           # M25/M26 current location
    PROJECT_ROOT / "checkpoints" / "graph_former",            # if any already in project
    Path(r"C:\Users\Saif\Downloads\g2results\checkpoints"),   # external graph_former copy
    Path(r"D:\Downloads\archive\checkpoints"),                # archive copy
]

def sha1_first_mb(path, n_bytes=1024*1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()

def is_relevant_pth(p: Path):
    name = p.name.lower()
    if p.suffix.lower() != ".pth":
        return False
    return (
        "model_m" in name
        or name.startswith("m25_")
        or name.startswith("m26_")
        or name.startswith("m27_")
        or "teacher_baseline" in name
        or "teacher_pretrained" in name
        or "student_baseline" in name
    )

found = []
for root in SEARCH_ROOTS:
    if root.exists():
        for p in root.rglob("*.pth"):
            if is_relevant_pth(p):
                found.append(p.resolve())

# de-duplicate exact same path entries
found = sorted(set(found))

DEST_DIR.mkdir(parents=True, exist_ok=True)

# detect duplicate content by filename + size + quick hash
groups = defaultdict(list)
for p in found:
    try:
        key = (p.name, p.stat().st_size, sha1_first_mb(p))
    except Exception:
        key = (p.name, None, None)
    groups[key].append(p)

move_plan = []
seen_dest_names = set(x.name for x in DEST_DIR.glob("*.pth"))

for key, paths in groups.items():
    # prefer files already inside the project root
    paths = sorted(paths, key=lambda x: (0 if str(x).startswith(str(PROJECT_ROOT)) else 1, str(x)))
    keeper = paths[0]
    dest_name = keeper.name

    if dest_name in seen_dest_names:
        stem, suf = keeper.stem, keeper.suffix
        i = 2
        while f"{stem}__dup{i}{suf}" in seen_dest_names:
            i += 1
        dest_name = f"{stem}__dup{i}{suf}"

    seen_dest_names.add(dest_name)
    move_plan.append({
        "source_keep": keeper,
        "dest": DEST_DIR / dest_name,
        "duplicates_same_content": paths[1:],
    })

print("=" * 100)
print("DRY RUN SUMMARY")
print("=" * 100)
print("Destination:", DEST_DIR)
print("Total candidate .pth files found:", len(found))
print("Unique files after content de-dup:", len(move_plan))
print()

for i, item in enumerate(move_plan[:40], start=1):
    print(f"{i:03d}. {item['source_keep']}  -->  {item['dest']}")
    if item["duplicates_same_content"]:
        print("     same-content duplicates:")
        for d in item["duplicates_same_content"][:3]:
            print("       -", d)
        if len(item["duplicates_same_content"]) > 3:
            print(f"       ... +{len(item['duplicates_same_content'])-3} more")

if len(move_plan) > 40:
    print(f"\n... showing first 40 of {len(move_plan)} unique files")


# ==============================================================================
# Notebook cell 4
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ============================================================
# SAFER CELL A — DRY RUN scoped to M11–M26 only
# ============================================================
from pathlib import Path
from collections import defaultdict
import hashlib
import re

PROJECT_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")
DEST_DIR = PROJECT_ROOT / "checkpoints" / "all_model_weights"

SEARCH_ROOTS = [
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "checkpoints" / "model_weights",
    Path(r"C:\Users\Saif\Downloads\g2results\checkpoints"),
    Path(r"D:\Downloads\archive\checkpoints"),
]

DEST_DIR.mkdir(parents=True, exist_ok=True)

ALLOW_MIDS = {f"M{i:02d}" for i in range(11, 27)}
ALLOW_EXTRA = {"teacher_baseline.pth", "teacher_pretrained.pth", "student_baseline.pth"}

def quick_hash(path, n_bytes=1024*1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()

def detect_mid(name):
    m = re.search(r"(M\d{2})", name)
    return m.group(1) if m else None

def is_allowed_file(p: Path):
    if p.suffix.lower() != ".pth":
        return False
    if p.name in ALLOW_EXTRA:
        return True
    mid = detect_mid(p.name)
    return mid in ALLOW_MIDS

found = []
for root in SEARCH_ROOTS:
    if root.exists():
        for p in root.rglob("*.pth"):
            if is_allowed_file(p):
                found.append(p.resolve())

found = sorted(set(found))

groups = defaultdict(list)
for p in found:
    try:
        key = (p.name, p.stat().st_size, quick_hash(p))
    except Exception:
        key = (p.name, None, None)
    groups[key].append(p)

copy_plan = []
for key, paths in groups.items():
    paths = sorted(paths, key=lambda x: (0 if str(x).startswith(str(PROJECT_ROOT)) else 1, str(x)))
    keeper = paths[0]
    dst = DEST_DIR / keeper.name
    copy_plan.append({
        "source_keep": keeper,
        "dest": dst,
        "duplicates_same_content": paths[1:],
    })

print("=" * 100)
print("SCOPED DRY RUN SUMMARY")
print("=" * 100)
print("Destination:", DEST_DIR)
print("Total candidate .pth files found:", len(found))
print("Unique files after content de-dup:", len(copy_plan))
print()

for i, item in enumerate(copy_plan[:60], start=1):
    print(f"{i:03d}. {item['source_keep']}  -->  {item['dest']}")
    if item["duplicates_same_content"]:
        print("     same-content duplicates:")
        for d in item["duplicates_same_content"][:3]:
            print("       -", d)
        if len(item["duplicates_same_content"]) > 3:
            print(f"       ... +{len(item['duplicates_same_content'])-3} more")

if len(copy_plan) > 60:
    print(f"\n... showing first 60 of {len(copy_plan)} unique files")


# ==============================================================================
# Notebook cell 5
# Categories: training
# ==============================================================================
# ============================================================
# SAFER CELL B — COPY into unified folder
# ============================================================
from pathlib import Path
import shutil
import csv

PROJECT_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")
DEST_DIR = PROJECT_ROOT / "checkpoints" / "all_model_weights"
MANIFEST = DEST_DIR / "copy_manifest.csv"

DEST_DIR.mkdir(parents=True, exist_ok=True)

copied = []
skipped = []
errors = []

for item in copy_plan:
    src = item["source_keep"]
    dst = item["dest"]

    try:
        if dst.exists():
            skipped.append((str(src), f"destination already exists: {dst}"))
            continue

        shutil.copy2(str(src), str(dst))
        copied.append((str(src), str(dst)))

    except Exception as e:
        errors.append((str(src), str(dst), str(e)))

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["status", "source", "destination_or_note"])
    for s, d in copied:
        w.writerow(["COPIED", s, d])
    for s, note in skipped:
        w.writerow(["SKIPPED", s, note])
    for s, d, e in errors:
        w.writerow(["ERROR", s, f"{d} | {e}"])

print("=" * 100)
print("COPY COMPLETE")
print("=" * 100)
print("Copied  :", len(copied))
print("Skipped :", len(skipped))
print("Errors  :", len(errors))
print("Manifest:", MANIFEST)

if copied[:20]:
    print("\nSample copied files:")
    for s, d in copied[:20]:
        print(" -", Path(s).name, "->", d)

if errors:
    print("\nSample errors:")
    for s, d, e in errors[:10]:
        print(" -", s)
        print("   ", e)


# ==============================================================================
# Notebook cell 6
# Categories: model_definition, training, audit_verification
# ==============================================================================
# ============================================================
# SAFER CELL C — VERIFY unified folder
# ============================================================
from pathlib import Path
from collections import Counter
import re

DEST_DIR = Path(r"C:\Users\Saif\Desktop\CSE400\C\checkpoints\all_model_weights")
files = sorted(DEST_DIR.glob("*.pth"))

def detect_mid(name):
    m = re.search(r"(M\d{2})", name)
    return m.group(1) if m else "OTHER"

cnt = Counter(detect_mid(p.name) for p in files)

print("DEST_DIR:", DEST_DIR)
print("TOTAL .pth:", len(files))
print()

for k in sorted(cnt):
    print(f"{k}: {cnt[k]}")

extras = [p.name for p in files if p.name in {"teacher_baseline.pth", "teacher_pretrained.pth", "student_baseline.pth"}]
if extras:
    print("\nExtra baseline files:")
    for x in extras:
        print(" -", x)


# ==============================================================================
# Notebook cell 7
# Categories: model_definition, training, results_tables, audit_verification
# ==============================================================================
# ============================================================
# TARGETED FILL-IN CELL — find/copy missing M13 remainder + M14–M24
# ============================================================
from pathlib import Path
import shutil
import re
from collections import Counter

PROJECT_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")
DEST_DIR = PROJECT_ROOT / "checkpoints" / "all_model_weights"
DEST_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    Path(r"C:\Users\Saif\Desktop\CSE400"),
    Path(r"C:\Users\Saif\Downloads"),
    Path(r"D:\Downloads"),
    Path("D:/"),
]

NEEDED = {f"M{i:02d}" for i in range(13, 25)}  # M13..M24
ALLOW_EXTRA = {"teacher_baseline.pth", "teacher_pretrained.pth", "student_baseline.pth"}

def detect_mid(name):
    m = re.search(r"(M\d{2})", name)
    return m.group(1) if m else None

def is_needed(p: Path):
    if p.suffix.lower() != ".pth":
        return False
    if p.name in ALLOW_EXTRA:
        return True
    mid = detect_mid(p.name)
    return mid in NEEDED

existing = {p.name for p in DEST_DIR.glob("*.pth")}
found = []

for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    print(f"\nSearching {root} ...")
    try:
        for p in root.rglob("*.pth"):
            if is_needed(p):
                found.append(p.resolve())
    except Exception as e:
        print("Skipped part of", root, "because:", e)

found = sorted(set(found))

copied, skipped, errors = [], [], []

for src in found:
    dst = DEST_DIR / src.name
    try:
        if dst.name in existing or dst.exists():
            skipped.append(str(src))
            continue
        shutil.copy2(str(src), str(dst))
        copied.append((str(src), str(dst)))
        existing.add(dst.name)
    except Exception as e:
        errors.append((str(src), str(e)))

print("\n" + "="*90)
print("FILL-IN COPY SUMMARY")
print("="*90)
print("Found   :", len(found))
print("Copied  :", len(copied))
print("Skipped :", len(skipped))
print("Errors  :", len(errors))

if copied[:30]:
    print("\nSample copied:")
    for s, d in copied[:30]:
        print(" -", s, "->", d)

if errors[:10]:
    print("\nSample errors:")
    for s, e in errors[:10]:
        print(" -", s)
        print("   ", e)

files = sorted(DEST_DIR.glob("*.pth"))
cnt = Counter(detect_mid(p.name) if detect_mid(p.name) else "OTHER" for p in files)

print("\nFINAL COUNTS IN all_model_weights")
for k in sorted(cnt):
    print(f"{k}: {cnt[k]}")


# ==============================================================================
# Notebook cell 8
# Categories: model_definition, audit_verification
# ==============================================================================
# ============================================================
# DIAGNOSTIC CELL — search missing models by architecture names
# ============================================================
from pathlib import Path
from collections import defaultdict
import re

SEARCH_ROOTS = [
    Path(r"C:\Users\Saif\Desktop\CSE400"),
    Path(r"C:\Users\Saif\Downloads"),
    Path(r"D:\Downloads"),
    Path("D:/"),
]

# likely architecture/name patterns for M14–M24
PATTERNS = {
    "M14_GRU": ["*GRU*.pth", "*gru*.pth"],
    "M15_Conv1D": ["*Conv1D*.pth", "*conv1d*.pth", "*CNN*.pth", "*cnn*.pth"],
    "M16_Transformer": ["*Transformer*.pth", "*transformer*.pth", "*ViT*.pth", "*vit*.pth"],
    "M17_Conformer": ["*Conformer*.pth", "*conformer*.pth"],
    "M18_ChanDropTransformer": ["*ChanDrop*.pth", "*chandrop*.pth", "*DropTransformer*.pth"],
    "M19_DANN": ["*DANN*.pth", "*dann*.pth", "*domain*.pth"],
    "M20_CLISA": ["*CLISA*.pth", "*clisa*.pth"],
    "M21_SimCLR": ["*SimCLR*.pth", "*simclr*.pth"],
    "M22_BYOL": ["*BYOL*.pth", "*byol*.pth"],
    "M23_PseudoLabel": ["*Pseudo*.pth", "*pseudo*.pth"],
    "M24_MixMatch": ["*MixMatch*.pth", "*mixmatch*.pth"],
}

hits = defaultdict(set)

for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    print(f"\nSearching {root} ...")
    for label, pats in PATTERNS.items():
        for pat in pats:
            try:
                for p in root.rglob(pat):
                    hits[label].add(str(p.resolve()))
            except Exception:
                pass

print("\n" + "="*100)
print("ARCHITECTURE-NAME SEARCH RESULTS")
print("="*100)

total = 0
for label in PATTERNS:
    files = sorted(hits[label])
    total += len(files)
    print(f"\n{label}: {len(files)}")
    for x in files[:20]:
        print(" -", x)
    if len(files) > 20:
        print(f" ... +{len(files)-20} more")

print("\nTOTAL HITS:", total)


# ==============================================================================
# Notebook cell 9
# Categories: training
# ==============================================================================
# ============================================================
# STRICT PROOF CELL — only count project-style checkpoint names
# ============================================================
from pathlib import Path
from collections import Counter, defaultdict
import re

ROOTS = [
    Path(r"C:\Users\Saif\Desktop\CSE400\C\checkpoints"),
    Path(r"D:\Downloads\archive\checkpoints"),
    Path(r"C:\Users\Saif\Downloads\g2results\checkpoints"),
]

# only your project naming styles
PATS = {
    "M11": re.compile(r"^model_M11_ShallowMLP_s\d+_f\d+_best\.pth$", re.I),
    "M12": re.compile(r"^model_M12_DeepMLP_s\d+_f\d+_best\.pth$", re.I),
    "M13": re.compile(r"^model_M13_LSTM_s\d+_f\d+_best\.pth$", re.I),
    "M14": re.compile(r"^model_M14_.*_s\d+_f\d+_best\.pth$", re.I),
    "M15": re.compile(r"^model_M15_.*_s\d+_f\d+_best\.pth$", re.I),
    "M16": re.compile(r"^model_M16_.*_s\d+_f\d+_best\.pth$", re.I),
    "M17": re.compile(r"^model_M17_.*_s\d+_f\d+_best\.pth$", re.I),
    "M18": re.compile(r"^model_M18_.*_s\d+_f\d+_best\.pth$", re.I),
    "M19": re.compile(r"^model_M19_.*_s\d+_f\d+_best\.pth$", re.I),
    "M20": re.compile(r"^model_M20_.*_s\d+_f\d+_best\.pth$", re.I),
    "M21": re.compile(r"^model_M21_.*_s\d+_f\d+_best\.pth$", re.I),
    "M22": re.compile(r"^model_M22_.*_s\d+_f\d+_best\.pth$", re.I),
    "M23": re.compile(r"^model_M23_.*_s\d+_f\d+_best\.pth$", re.I),
    "M24": re.compile(r"^model_M24_.*_s\d+_f\d+_best\.pth$", re.I),
    "M25": re.compile(r"^M25_62ch_s\d+_f\d+_best\.pth$", re.I),
    "M26": re.compile(r"^M26_6ch_s\d+_f\d+_best\.pth$", re.I),
}

hits = defaultdict(list)

for root in ROOTS:
    if not root.exists():
        continue
    for p in root.rglob("*.pth"):
        name = p.name
        for mid, pat in PATS.items():
            if pat.match(name):
                hits[mid].append(str(p.resolve()))
                break

print("="*90)
print("STRICT PROJECT CHECKPOINT COUNTS")
print("="*90)

for mid in [f"M{i:02d}" for i in range(11, 27)]:
    files = sorted(set(hits[mid]))
    print(f"{mid}: {len(files)}")
    for x in files[:5]:
        print("   ", x)
    if len(files) > 5:
        print("    ...")


# ==============================================================================
# Notebook cell 10
# Categories: training
# ==============================================================================
# ============================================================
# CANONICAL COUNT CELL — count only what is inside all_model_weights
# ============================================================
from pathlib import Path
from collections import Counter
import re

ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C\checkpoints\all_model_weights")

def detect_mid(name):
    m = re.search(r"(M\d{2})", name)
    return m.group(1) if m else "OTHER"

files = sorted(ROOT.glob("*.pth"))
cnt = Counter(detect_mid(p.name) for p in files)

print("ROOT:", ROOT)
print("TOTAL FILES:", len(files))
print()

for mid in [f"M{i:02d}" for i in range(11, 27)] + ["OTHER"]:
    if cnt.get(mid, 0):
        print(f"{mid}: {cnt[mid]}")


# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, model_definition, training, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC — Inventory existing .pth files before conversion
# Run this cell first. Do NOT convert anything yet.
# ═══════════════════════════════════════════════════════════════
import json
from pathlib import Path
from collections import defaultdict

BASE     = Path('.')
CKPT_DIR = BASE / 'checkpoints'
MDL_DIR  = CKPT_DIR / 'model_weights'
JSON_DIR = CKPT_DIR / 'loso_results'

print("=" * 70)
print("  STEP 1 — Scan all .pth files")
print("=" * 70)

# Find every .pth in the project
all_pth = sorted(CKPT_DIR.rglob('*.pth'))
print(f"\nTotal .pth files found: {len(all_pth)}")

# Group by inferred model ID
by_model = defaultdict(list)
for p in all_pth:
    name = p.stem  # e.g. model_M11_ShallowMLP_s1_f01_best
    # Infer model tag (M11, M12, ...)
    for part in name.replace('model_', '').split('_'):
        if part.startswith('M') and part[1:].isdigit():
            by_model[part].append(p)
            break
    else:
        by_model['UNKNOWN'].append(p)

print(f"\n{'Model':<8} {'Count':>6}  {'Location':<30}  {'Fold range':<20}  {'Seeds'}")
print("-" * 80)
for mid in sorted(by_model.keys()):
    files = by_model[mid]
    locs  = set(p.parent.name for p in files)
    
    folds, seeds = [], []
    for p in files:
        parts = p.stem.split('_')
        for part in parts:
            if part.startswith('f') and part[1:].isdigit():
                folds.append(int(part[1:]))
            if part.startswith('s') and part[1:].isdigit():
                seeds.append(int(part[1:]))
    
    fold_rng = f"f{min(folds):03d}–f{max(folds):03d}" if folds else "?"
    seed_set = sorted(set(seeds))
    print(f"{mid:<8} {len(files):>6}  {str(list(locs)):<30}  {fold_rng:<20}  {seed_set}")

print()
print("=" * 70)
print("  STEP 2 — Check corresponding JSON checkpoints")
print("=" * 70)

EXPECTED_MODELS = {
    'M11': ('ShallowMLP', None),   # ch unknown — might differ
    'M12': ('DeepMLP',   None),
    'M13': ('LSTM',      None),
    'M14': ('GRU',       '62ch'),
    'M15': ('Conv1D',    '62ch'),
    'M16': ('VanTransf', '62ch'),
    'M17': ('Conformer', '62ch'),
    'M18': ('ChanDrop',  '62ch'),
    'M19': ('DANN',      '62ch'),
    'M20': ('CLISA',     '62ch'),
    'M21': ('SimCLR',    '62ch'),
    'M22': ('BYOL',      '62ch'),
    'M23': ('PseudoLbl', '62ch'),
    'M24': ('MixMatch',  '62ch'),
    'M25': ('DANCETch',  '62ch'),
    'M26': ('DANCEStu',  '6ch'),
}

print(f"\n{'Model':<8} {'pth files':>10}  {'JSON results':>12}  {'Best val_f1 (if JSON exists)'}")
print("-" * 65)

for mid, (name, ch) in EXPECTED_MODELS.items():
    n_pth = len(by_model.get(mid, []))
    
    # Count JSONs
    json_files = list(JSON_DIR.glob(f'{mid}_*.json'))
    n_json = len(json_files)
    
    # Find best val_f1 from JSONs
    best_f1, best_file = 0.0, None
    for jf in json_files:
        try:
            d = json.loads(jf.read_text())
            vf1 = d.get('best_val_f1', 0.0)
            if vf1 > best_f1:
                best_f1 = vf1
                best_file = jf.name
        except Exception:
            pass
    
    f1_str = f"{best_f1:.4f}  ← {best_file}" if best_file else "— (no JSON or no val_f1 key)"
    pth_str = f"✓ {n_pth}" if n_pth > 0 else "✗ MISSING"
    json_str = f"✓ {n_json}" if n_json > 0 else "✗ 0"
    print(f"{mid:<8} {pth_str:>10}  {json_str:>12}  {f1_str}")

print()
print("=" * 70)
print("  STEP 3 — Naming convention sanity check")
print("=" * 70)
print("\nSample .pth filenames found:")
for mid in sorted(by_model.keys())[:6]:
    sample = by_model[mid][0]
    print(f"  {mid}: {sample.relative_to(BASE)}")

print("\n⚠  STOP HERE — verify the output above before running the conversion cell.")
print("   Key questions:")
print("   1. Do M11/M12/M13 fold counts match your expectations?")
print("   2. Are M25 teacher weights present (needed for M26)?")
print("   3. Do JSONs contain 'best_val_f1' key? (If not, we use acc_a instead)")


# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL A — Convert existing .pth → .pt for M11/M12/M13/M25/M26
# ═══════════════════════════════════════════════════════════════
import json, re, torch
from pathlib import Path
from collections import defaultdict

BASE    = Path('.')
CKPT    = BASE / 'checkpoints'
MDL_DIR = CKPT / 'model_weights'
JSON_DIR= CKPT / 'loso_results'
OUT_DIR = MDL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Index all .pth files ───────────────────────────────────────
OLD = re.compile(r'model_(M\d+)_\w+_s(\d+)_f(\d+)_best')
NEW = re.compile(r'(M\d+)_(6ch|62ch)_s(\d+)_f(\d+)_best')

pth_index = {}
for p in CKPT.rglob('*.pth'):
    stem = p.stem
    m = NEW.match(stem)
    if m:
        mid, ch, seed, fold = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        key = (mid, seed, fold, ch)
        # prefer model_weights/ over all_model_weights/
        if key not in pth_index or 'model_weights' in str(p.parent) and 'all_' not in str(p.parent):
            pth_index[key] = p
        continue
    m = OLD.match(stem)
    if m:
        mid, seed, fold = m.group(1), int(m.group(2)), int(m.group(3))
        for ch in ['62ch', '6ch']:
            pth_index.setdefault((mid, seed, fold, ch), p)

print(f"Indexed {len(pth_index)} .pth entries")

# ── Pick best fold per (model × ch) using f1_a from JSONs ─────
CONVERT = ['M11', 'M12', 'M13', 'M25', 'M26']
best = {}  # (mid, ch) -> {'f1_a', 'seed', 'fold', 'pth'}

for jf in sorted(JSON_DIR.glob('*.json')):
    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    mid  = d.get('model_id', '')
    if mid not in CONVERT:
        continue
    ch   = d.get('ch', '')
    seed = d.get('seed', 0)
    fold = d.get('fold', 0)
    f1_a = d.get('f1_a', 0.0)
    if not ch:
        continue
    pth = pth_index.get((mid, seed, fold, ch))
    if pth is None:
        continue
    key = (mid, ch)
    if key not in best or f1_a > best[key]['f1_a']:
        best[key] = dict(f1_a=f1_a, seed=seed, fold=fold, pth=pth)

# ── Convert ────────────────────────────────────────────────────
print(f"\n{'Output .pt':<30} {'f1_a':>6}  s  f   Source")
print("-" * 75)
for (mid, ch), rec in sorted(best.items()):
    dst = OUT_DIR / f'{mid}_{ch}_best.pt'
    src = rec['pth']
    try:
        try:
            state = torch.load(src, map_location='cpu', weights_only=True)
        except Exception:
            state = torch.load(src, map_location='cpu')
        torch.save(state, dst)
        print(f"  ✓ {dst.name:<28} {rec['f1_a']:.4f}  {rec['seed']}  {rec['fold']:02d}  {src.name}")
    except Exception as e:
        print(f"  ✗ FAILED {mid}_{ch}: {e}")

print(f"\n✅ Cell A done. Output → {OUT_DIR.resolve()}")


# ==============================================================================
# Notebook cell 13
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL A — Convert existing .pth → .pt for M11/M12/M13/M25/M26
# ═══════════════════════════════════════════════════════════════
import json, re, torch
from pathlib import Path
from collections import defaultdict

BASE    = Path('.')
CKPT    = BASE / 'checkpoints'
MDL_DIR = CKPT / 'model_weights'
JSON_DIR= CKPT / 'loso_results'
OUT_DIR = MDL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Index all .pth files ───────────────────────────────────────
OLD = re.compile(r'model_(M\d+)_\w+_s(\d+)_f(\d+)_best')
NEW = re.compile(r'(M\d+)_(6ch|62ch)_s(\d+)_f(\d+)_best')

pth_index = {}
for p in CKPT.rglob('*.pth'):
    stem = p.stem
    m = NEW.match(stem)
    if m:
        mid, ch, seed, fold = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        key = (mid, seed, fold, ch)
        # prefer model_weights/ over all_model_weights/
        if key not in pth_index or 'model_weights' in str(p.parent) and 'all_' not in str(p.parent):
            pth_index[key] = p
        continue
    m = OLD.match(stem)
    if m:
        mid, seed, fold = m.group(1), int(m.group(2)), int(m.group(3))
        for ch in ['62ch', '6ch']:
            pth_index.setdefault((mid, seed, fold, ch), p)

print(f"Indexed {len(pth_index)} .pth entries")

# ── Pick best fold per (model × ch) using f1_a from JSONs ─────
CONVERT = ['M11', 'M12', 'M13', 'M25', 'M26']
best = {}  # (mid, ch) -> {'f1_a', 'seed', 'fold', 'pth'}

for jf in sorted(JSON_DIR.glob('*.json')):
    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    mid  = d.get('model_id', '')
    if mid not in CONVERT:
        continue
    ch   = d.get('ch', '')
    seed = d.get('seed', 0)
    fold = d.get('fold', 0)
    f1_a = d.get('f1_a', 0.0)
    if not ch:
        continue
    pth = pth_index.get((mid, seed, fold, ch))
    if pth is None:
        continue
    key = (mid, ch)
    if key not in best or f1_a > best[key]['f1_a']:
        best[key] = dict(f1_a=f1_a, seed=seed, fold=fold, pth=pth)

# ── Convert ────────────────────────────────────────────────────
print(f"\n{'Output .pt':<30} {'f1_a':>6}  s  f   Source")
print("-" * 75)
for (mid, ch), rec in sorted(best.items()):
    dst = OUT_DIR / f'{mid}_{ch}_best.pt'
    src = rec['pth']
    try:
        try:
            state = torch.load(src, map_location='cpu', weights_only=True)
        except Exception:
            state = torch.load(src, map_location='cpu')
        torch.save(state, dst)
        print(f"  ✓ {dst.name:<28} {rec['f1_a']:.4f}  {rec['seed']}  {rec['fold']:02d}  {src.name}")
    except Exception as e:
        print(f"  ✗ FAILED {mid}_{ch}: {e}")

print(f"\n✅ Cell A done. Output → {OUT_DIR.resolve()}")


# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, model_definition, training, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
# CLISA only: export/copy best .pth -> .pt into web app checkpoints dir
# Assumption: your project uses 62ch + 6ch (if you meant "64", change the labels below).

import json, re, shutil
from pathlib import Path
import torch

# ------------------------------------------------------------
# 1) Auto-detect project root / source checkpoint dirs / webapp dir
# ------------------------------------------------------------
ROOT_CANDIDATES = [
    Path('.').resolve(),
    (Path('.') / 'neurosync').resolve(),
    Path('..').resolve(),
    (Path('..') / 'neurosync').resolve(),
]

project_root = None
for root in ROOT_CANDIDATES:
    if (root / 'models' / 'checkpoints').exists() or (root / 'backend' / 'app').exists() or (root / 'frontend').exists():
        project_root = root
        break
if project_root is None:
    project_root = Path('.').resolve()

TARGET_DIR = project_root / 'models' / 'checkpoints'
TARGET_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DIRS = [
    project_root / 'checkpoints' / 'model_weights',
    project_root / 'checkpoints' / 'all_model_weights',
    project_root / 'checkpoints',
]
SOURCE_DIRS = [p for p in SOURCE_DIRS if p.exists()]

JSON_DIR = project_root / 'checkpoints' / 'loso_results'

print('=' * 90)
print('CLISA EXPORT → WEB APP CHECKPOINTS')
print('=' * 90)
print('project_root :', project_root)
print('target_dir   :', TARGET_DIR)
print('source_dirs  :', [str(p) for p in SOURCE_DIRS] if SOURCE_DIRS else 'NONE FOUND')
print('json_dir     :', JSON_DIR if JSON_DIR.exists() else 'NOT FOUND')
print()

if not SOURCE_DIRS:
    raise FileNotFoundError(
        'No source checkpoint directory found. Expected one of: '\
        'checkpoints/model_weights, checkpoints/all_model_weights, or checkpoints/'
    )

# ------------------------------------------------------------
# 2) Index all M20/CLISA .pth files
#    Supports both naming styles seen in your notebooks:
#      - M20_62ch_s1_f01_best.pth
#      - model_M20_CLISA_s1_f01_best.pth
# ------------------------------------------------------------
NEW = re.compile(r'^(M20)_(62ch|6ch)_s(\d+)_f(\d+)_best$', re.I)
OLD = re.compile(r'^model_(M20)_[A-Za-z0-9]+_s(\d+)_f(\d+)_best$', re.I)

pth_index = {}   # (mid, seed, fold, ch) -> path
all_hits = []

for src_dir in SOURCE_DIRS:
    for p in src_dir.rglob('*.pth'):
        stem = p.stem
        m = NEW.match(stem)
        if m:
            mid, ch, seed, fold = m.group(1).upper(), m.group(2).lower(), int(m.group(3)), int(m.group(4))
            key = (mid, seed, fold, ch)
            # prefer model_weights over looser roots if duplicates exist
            if key not in pth_index or 'model_weights' in str(p.parent).lower():
                pth_index[key] = p
            all_hits.append(p)
            continue

        m = OLD.match(stem)
        if m:
            mid, seed, fold = m.group(1).upper(), int(m.group(2)), int(m.group(3))
            # old names do not encode channel type, so make them available to both; JSON selection will disambiguate
            for ch in ('62ch', '6ch'):
                pth_index.setdefault((mid, seed, fold, ch), p)
            all_hits.append(p)

print(f'Indexed M20/CLISA .pth files: {len(set(all_hits))}')
if all_hits:
    for p in sorted(set(all_hits))[:12]:
        print('  -', p)
    if len(set(all_hits)) > 12:
        print(f'  ... +{len(set(all_hits)) - 12} more')
print()

# ------------------------------------------------------------
# 3) Pick the best 62ch and best 6ch checkpoint using JSON metrics
#    Priority: f1_a > best_val_f1 > acc_a > acc_b
# ------------------------------------------------------------
def metric_from_json(d):
    for k in ('f1_a', 'best_val_f1', 'acc_a', 'acc_b'):
        v = d.get(k, None)
        if isinstance(v, (int, float)):
            return float(v), k
    return None, None

best = {}  # ch -> record

if JSON_DIR.exists():
    for jf in sorted(JSON_DIR.glob('M20_*.json')):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue

        mid = str(d.get('model_id', '')).upper()
        ch = str(d.get('ch', '')).lower()
        seed = d.get('seed', None)
        fold = d.get('fold', None)
        if mid != 'M20' or ch not in ('62ch', '6ch') or seed is None or fold is None:
            continue

        src = pth_index.get((mid, int(seed), int(fold), ch))
        if src is None:
            continue

        score, score_key = metric_from_json(d)
        if score is None:
            continue

        rec = {
            'src': src,
            'seed': int(seed),
            'fold': int(fold),
            'score': float(score),
            'score_key': score_key,
            'json': jf,
        }
        if ch not in best or rec['score'] > best[ch]['score']:
            best[ch] = rec

# ------------------------------------------------------------
# 4) Fallback if JSONs are missing: pick newest matching .pth by channel token
# ------------------------------------------------------------
if '62ch' not in best:
    cands = [p for p in sorted(set(all_hits)) if '62ch' in p.stem.lower()]
    if cands:
        src = max(cands, key=lambda p: p.stat().st_mtime)
        m = re.search(r'_s(\d+)_f(\d+)_best$', src.stem, re.I)
        best['62ch'] = {
            'src': src,
            'seed': int(m.group(1)) if m else -1,
            'fold': int(m.group(2)) if m else -1,
            'score': float('-inf'),
            'score_key': 'fallback_latest_mtime',
            'json': None,
        }

if '6ch' not in best:
    cands = [p for p in sorted(set(all_hits)) if '6ch' in p.stem.lower()]
    if cands:
        src = max(cands, key=lambda p: p.stat().st_mtime)
        m = re.search(r'_s(\d+)_f(\d+)_best$', src.stem, re.I)
        best['6ch'] = {
            'src': src,
            'seed': int(m.group(1)) if m else -1,
            'fold': int(m.group(2)) if m else -1,
            'score': float('-inf'),
            'score_key': 'fallback_latest_mtime',
            'json': None,
        }

# ------------------------------------------------------------
# 5) Save as web-app-ready .pt files
#    We save the raw state_dict so it matches your earlier export cells.
# ------------------------------------------------------------
print(f"{'OUTPUT':<22} {'SOURCE':<45} {'metric':<20} {'seed':>4} {'fold':>4}")
print('-' * 110)

for ch in ('62ch', '6ch'):
    rec = best.get(ch)
    if rec is None:
        print(f'!! No CLISA checkpoint found for {ch}')
        continue

    src = rec['src']
    dst = TARGET_DIR / f'M20_{ch}_best.pt'

    try:
        try:
            obj = torch.load(src, map_location='cpu', weights_only=True)
        except TypeError:
            obj = torch.load(src, map_location='cpu')
        except Exception:
            obj = torch.load(src, map_location='cpu')

        state = obj.get('model_state_dict') if isinstance(obj, dict) and 'model_state_dict' in obj else obj
        torch.save(state, dst)

        print(f'{dst.name:<22} {src.name:<45} {rec["score_key"] + "=" + (f"{rec["score"]:.4f}" if rec["score"] != float("-inf") else "NA"):<20} {rec["seed"]:>4} {rec["fold"]:>4}')
    except Exception as e:
        print(f'FAILED for {ch}: {src.name} -> {e}')

print('\nDone.')
print('Web app checkpoints dir:', TARGET_DIR.resolve())
print('Expected outputs       :', [str(TARGET_DIR / 'M20_62ch_best.pt'), str(TARGET_DIR / 'M20_6ch_best.pt')])



# ==============================================================================
# Notebook cell 15
# Categories: preprocessing, model_definition, training, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil

# ------------------------------------------------------------
# Auto-patch NeuroSync web app to support CLISA / M20
# Run from project root (the folder that contains backend/, frontend/, models/)
# ------------------------------------------------------------

def find_project_root(start=Path.cwd()):
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for c in candidates:
        if (c / "backend/app/services/model_loader.py").exists() and \
           (c / "backend/app/services/dance_model.py").exists() and \
           (c / "frontend/src/pages/DeviceHub.jsx").exists():
            return c
    # shallow recursive search if notebook is not opened exactly at root
    for c in start.rglob("*"):
        if c.is_dir() and \
           (c / "backend/app/services/model_loader.py").exists() and \
           (c / "backend/app/services/dance_model.py").exists() and \
           (c / "frontend/src/pages/DeviceHub.jsx").exists():
            return c
    raise FileNotFoundError("Could not find project root with backend/frontend files.")

root = find_project_root()
print("=" * 90)
print("PATCHING NEUROSYNC FOR CLISA / M20")
print("=" * 90)
print("project_root:", root)

def backup_file(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {bak.name}")
    return bak

# ============================================================
# 1) Patch backend/app/services/dance_model.py
#    - add CLISA family
#    - add CLISA builder
#    - improve family detection
# ============================================================
dance_path = root / "backend/app/services/dance_model.py"
backup_file(dance_path)
dance_text = dance_path.read_text(encoding="utf-8")

if "class CLISAClassifier" not in dance_text:
    anchor = '    # ── Family detection ──────────────────────────────────────────────────────\n'
    insert = '''    # ── FAMILY C: CLISA ───────────────────────────────────────────────────────────

    class CLISAClassifier(nn.Module):
        """M20 CLISA inference model with encoder + projector + classifier."""
        def __init__(self, input_dim=30, d_model=128, d_proj=64, n_classes=4):
            super().__init__()
            self.input_dim_val = input_dim
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
            )
            self.projector = nn.Sequential(
                nn.Linear(d_model, d_proj),
                nn.ReLU(),
                nn.Linear(d_proj, d_proj),
            )
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x):
            z = self.encoder(x)
            return self.classifier(z)

        @property
        def input_dim(self):
            return self.input_dim_val

'''
    if anchor not in dance_text:
        raise RuntimeError("Could not find CLISA insertion anchor in dance_model.py")
    dance_text = dance_text.replace(anchor, insert + anchor)

old_detect = '''    def detect_family(state_dict: dict) -> str:
        """
        Inspect state dict keys to determine architecture family.
        Returns: 'eeg_band_transformer' | 'dance_student' | 'unknown'
        """
        keys = set(state_dict.keys())
        top  = {k.split(".")[0] for k in keys}

        bt_markers = {"pos_emb", "band_emb", "ch_attn", "classifier"}
        if bt_markers.issubset(top) and any(k.startswith("blocks.") for k in keys):
            return "eeg_band_transformer"

        if "encoder" in top:
            return "dance_student"

        return "unknown"

'''
new_detect = '''    def detect_family(state_dict: dict) -> str:
        """
        Inspect state dict keys to determine architecture family.
        Returns: 'eeg_band_transformer' | 'dance_student' | 'clisa' | 'unknown'
        """
        keys = set(state_dict.keys())
        top  = {k.split(".")[0] for k in keys}

        bt_markers = {"pos_emb", "band_emb", "ch_attn", "classifier"}
        if bt_markers.issubset(top) and any(k.startswith("blocks.") for k in keys):
            return "eeg_band_transformer"

        if "projector" in top and "classifier" in top and any(k.startswith("encoder.0.") for k in keys):
            return "clisa"

        dance_markers = ("encoder.ch_attn", "encoder.input_proj", "encoder.transformer", "encoder.norm")
        if "encoder" in top and any(k.startswith(dance_markers) for k in keys):
            return "dance_student"

        return "unknown"

'''
if old_detect in dance_text:
    dance_text = dance_text.replace(old_detect, new_detect)

old_factory = '''    def build_eeg_band_transformer(input_dim=30, n_channels=6, d_model=32,
                                    n_heads=1, n_blocks=2, d_ff=128,
                                    n_bands=5) -> EEGBandTransformer:
        if input_dim != n_channels * n_bands:
            n_channels = input_dim // n_bands
        return EEGBandTransformer(n_channels=n_channels, n_bands=n_bands,
                                  d_model=d_model, n_heads=n_heads,
                                  n_blocks=n_blocks, d_ff=d_ff)

except ImportError:
'''
new_factory = '''    def build_eeg_band_transformer(input_dim=30, n_channels=6, d_model=32,
                                    n_heads=1, n_blocks=2, d_ff=128,
                                    n_bands=5) -> EEGBandTransformer:
        if input_dim != n_channels * n_bands:
            n_channels = input_dim // n_bands
        return EEGBandTransformer(n_channels=n_channels, n_bands=n_bands,
                                  d_model=d_model, n_heads=n_heads,
                                  n_blocks=n_blocks, d_ff=d_ff)

    def build_clisa(input_dim=30, n_channels=6, d_model=128, d_proj=64,
                    n_classes=4, n_bands=5) -> CLISAClassifier:
        return CLISAClassifier(input_dim=input_dim, d_model=d_model, d_proj=d_proj, n_classes=n_classes)

except ImportError:
'''
if "def build_clisa(" not in dance_text:
    if old_factory not in dance_text:
        raise RuntimeError("Could not find CLISA factory insertion anchor in dance_model.py")
    dance_text = dance_text.replace(old_factory, new_factory)

old_except_stub = '''    def build_eeg_band_transformer(input_dim=30, n_channels=6, d_model=32, n_heads=1, n_blocks=2, d_ff=128, n_bands=5):
        raise ImportError("PyTorch is required to load model checkpoints.")
    def detect_family(state_dict):
        return "unknown"
'''
new_except_stub = '''    def build_eeg_band_transformer(input_dim=30, n_channels=6, d_model=32, n_heads=1, n_blocks=2, d_ff=128, n_bands=5):
        raise ImportError("PyTorch is required to load model checkpoints.")
    def build_clisa(input_dim=30, n_channels=6, d_model=128, d_proj=64, n_classes=4, n_bands=5):
        raise ImportError("PyTorch is required to load model checkpoints.")
    def detect_family(state_dict):
        return "unknown"
'''
if "def build_clisa(" not in dance_text.split("except ImportError:")[-1]:
    if old_except_stub in dance_text:
        dance_text = dance_text.replace(old_except_stub, new_except_stub)

dance_path.write_text(dance_text, encoding="utf-8")
print(f"[patched] {dance_path.relative_to(root)}")

# ============================================================
# 2) Patch backend/app/services/model_loader.py
#    - add M20 configs
#    - add build_clisa import
#    - add family branch in _build_model
# ============================================================
loader_path = root / "backend/app/services/model_loader.py"
backup_file(loader_path)
loader_text = loader_path.read_text(encoding="utf-8")

old_cfg_6 = '''    "M13_6ch":  {"n_channels": 6,  "n_bands": 5, "input_dim": 30,  "d_model": 256, "n_heads": 8, "n_layers": 3, "family": "dance_student"},
    "M26_6ch":  {"n_channels": 6,  "n_bands": 5, "input_dim": 30,  "d_model": 32,  "n_heads": 1, "n_blocks": 2, "d_ff": 128, "family": "eeg_band_transformer"},
'''
new_cfg_6 = '''    "M13_6ch":  {"n_channels": 6,  "n_bands": 5, "input_dim": 30,  "d_model": 256, "n_heads": 8, "n_layers": 3, "family": "dance_student"},
    "M20_6ch":  {"n_channels": 6,  "n_bands": 5, "input_dim": 30,  "d_model": 128, "d_proj": 64, "family": "clisa"},
    "M26_6ch":  {"n_channels": 6,  "n_bands": 5, "input_dim": 30,  "d_model": 32,  "n_heads": 1, "n_blocks": 2, "d_ff": 128, "family": "eeg_band_transformer"},
'''
if '"M20_6ch"' not in loader_text:
    if old_cfg_6 not in loader_text:
        raise RuntimeError("Could not find 6ch config insertion anchor in model_loader.py")
    loader_text = loader_text.replace(old_cfg_6, new_cfg_6)

old_cfg_62 = '''    "M13_62ch": {"n_channels": 62, "n_bands": 5, "input_dim": 310, "d_model": 512, "n_heads": 8, "n_layers": 6, "family": "dance_student"},
    "M25_62ch": {"n_channels": 62, "n_bands": 5, "input_dim": 310, "d_model": 256, "n_heads": 8, "n_layers": 4, "family": "dance_student"},
'''
new_cfg_62 = '''    "M13_62ch": {"n_channels": 62, "n_bands": 5, "input_dim": 310, "d_model": 512, "n_heads": 8, "n_layers": 6, "family": "dance_student"},
    "M20_62ch": {"n_channels": 62, "n_bands": 5, "input_dim": 310, "d_model": 128, "d_proj": 64, "family": "clisa"},
    "M25_62ch": {"n_channels": 62, "n_bands": 5, "input_dim": 310, "d_model": 256, "n_heads": 8, "n_layers": 4, "family": "dance_student"},
'''
if '"M20_62ch"' not in loader_text:
    if old_cfg_62 not in loader_text:
        raise RuntimeError("Could not find 62ch config insertion anchor in model_loader.py")
    loader_text = loader_text.replace(old_cfg_62, new_cfg_62)

old_import = '        from app.services.dance_model import build_dance_student, build_eeg_band_transformer\n'
new_import = '        from app.services.dance_model import build_dance_student, build_eeg_band_transformer, build_clisa\n'
if "build_clisa" not in loader_text:
    if old_import not in loader_text:
        raise RuntimeError("Could not find import anchor in model_loader.py")
    loader_text = loader_text.replace(old_import, new_import)

old_build_branch = '''        if family == "eeg_band_transformer":
            return build_eeg_band_transformer(
                input_dim=cfg["input_dim"],
                n_channels=cfg["n_channels"],
                d_model=cfg.get("d_model", 32),
                n_heads=cfg.get("n_heads", 1),
                n_blocks=cfg.get("n_blocks", 2),
                d_ff=cfg.get("d_ff", 128),
            )
        elif family == "dance_student":
'''
new_build_branch = '''        if family == "eeg_band_transformer":
            return build_eeg_band_transformer(
                input_dim=cfg["input_dim"],
                n_channels=cfg["n_channels"],
                d_model=cfg.get("d_model", 32),
                n_heads=cfg.get("n_heads", 1),
                n_blocks=cfg.get("n_blocks", 2),
                d_ff=cfg.get("d_ff", 128),
            )
        elif family == "clisa":
            return build_clisa(
                input_dim=cfg["input_dim"],
                n_channels=cfg["n_channels"],
                d_model=cfg.get("d_model", 128),
                d_proj=cfg.get("d_proj", 64),
            )
        elif family == "dance_student":
'''
if 'elif family == "clisa":' not in loader_text:
    if old_build_branch not in loader_text:
        raise RuntimeError("Could not find _build_model anchor in model_loader.py")
    loader_text = loader_text.replace(old_build_branch, new_build_branch)

loader_path.write_text(loader_text, encoding="utf-8")
print(f"[patched] {loader_path.relative_to(root)}")

# ============================================================
# 3) Patch frontend/src/pages/DeviceHub.jsx
#    - add M20 entries to visible model selector
# ============================================================
ui_path = root / "frontend/src/pages/DeviceHub.jsx"
backup_file(ui_path)
ui_text = ui_path.read_text(encoding="utf-8")

old_ui_top = """const MODEL_VARIANTS = [
  { file: 'M26_6ch_best.pt',  label: 'M26 · 6-ch',   ch: 6,  family: 'EEGBandTransformer', desc: 'Band-first transformer — confirmed architecture match', default: true },
  { file: 'M11_6ch_best.pt',  label: 'M11 · 6-ch',   ch: 6,  family: 'DANCEStudent',        desc: 'Compact wearable encoder model' },
"""
new_ui_top = """const MODEL_VARIANTS = [
  { file: 'M26_6ch_best.pt',  label: 'M26 · 6-ch',   ch: 6,  family: 'EEGBandTransformer', desc: 'Band-first transformer — confirmed architecture match', default: true },
  { file: 'M20_6ch_best.pt',  label: 'M20 · 6-ch',   ch: 6,  family: 'CLISA',               desc: 'Contrastive encoder + classifier fine-tune (wearable)' },
  { file: 'M11_6ch_best.pt',  label: 'M11 · 6-ch',   ch: 6,  family: 'DANCEStudent',        desc: 'Compact wearable encoder model' },
"""
if "M20_6ch_best.pt" not in ui_text:
    if old_ui_top not in ui_text:
        raise RuntimeError("Could not find top MODEL_VARIANTS anchor in DeviceHub.jsx")
    ui_text = ui_text.replace(old_ui_top, new_ui_top)

old_ui_bottom = """  { file: 'M13_62ch_best.pt', label: 'M13 · 62-ch',  ch: 62, family: 'DANCEStudent',        desc: 'Largest 62-channel model' },
  { file: 'M25_62ch_best.pt', label: 'M25 · 62-ch',  ch: 62, family: 'DANCEStudent',        desc: '62-channel alternative architecture' },
]
"""
new_ui_bottom = """  { file: 'M13_62ch_best.pt', label: 'M13 · 62-ch',  ch: 62, family: 'DANCEStudent',        desc: 'Largest 62-channel model' },
  { file: 'M20_62ch_best.pt', label: 'M20 · 62-ch',  ch: 62, family: 'CLISA',               desc: 'Contrastive encoder + classifier fine-tune (full-cap)' },
  { file: 'M25_62ch_best.pt', label: 'M25 · 62-ch',  ch: 62, family: 'DANCEStudent',        desc: '62-channel alternative architecture' },
]
"""
if "M20_62ch_best.pt" not in ui_text:
    if old_ui_bottom not in ui_text:
        raise RuntimeError("Could not find bottom MODEL_VARIANTS anchor in DeviceHub.jsx")
    ui_text = ui_text.replace(old_ui_bottom, new_ui_bottom)

ui_path.write_text(ui_text, encoding="utf-8")
print(f"[patched] {ui_path.relative_to(root)}")

# ============================================================
# 4) Quick verification
# ============================================================
print("\n" + "=" * 90)
print("VERIFY")
print("=" * 90)

expected = [
    root / "models/checkpoints/M20_6ch_best.pt",
    root / "models/checkpoints/M20_62ch_best.pt",
]

for p in expected:
    print(f"{'[OK]' if p.exists() else '[MISSING]'} {p}")

print("\nPatched model selector entries:")
for name in ["M20_6ch_best.pt", "M20_62ch_best.pt"]:
    print(f" - {name}: {'YES' if name in ui_path.read_text(encoding='utf-8') else 'NO'}")

print("\nBackend config entries:")
loader_text = loader_path.read_text(encoding="utf-8")
for key in ['"M20_6ch"', '"M20_62ch"', 'family == "clisa"', 'build_clisa']:
    print(f" - {key}: {'YES' if key in loader_text or key in dance_path.read_text(encoding='utf-8') else 'NO'}")

print("\nDone.")
print("Now restart backend + frontend (or rebuild docker) and CLISA should appear in model selection.")


# ==============================================================================
# Notebook cell 16
# Categories: model_definition, training, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import subprocess

# ------------------------------------------------------------
# Fix Docker-visible CLISA checkpoints for NeuroSync
# ------------------------------------------------------------

def find_compose_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "docker-compose.yml").exists() and (p / "models").exists():
            return p
    for p in start.rglob("docker-compose.yml"):
        return p.parent
    raise FileNotFoundError("Could not find docker-compose.yml root")

compose_root = find_compose_root()
dst_dir = compose_root / "models" / "checkpoints"
dst_dir.mkdir(parents=True, exist_ok=True)

needed = ["M20_6ch_best.pt", "M20_62ch_best.pt"]

# search likely older export locations
search_roots = [
    Path(r"C:\Users\Saif\Desktop\CSE400\C"),
    compose_root,
]

print("=" * 90)
print("FIXING DOCKER-VISIBLE CLISA CHECKPOINTS")
print("=" * 90)
print("compose_root :", compose_root)
print("docker maps  :", f"{compose_root / 'models'}  ->  /app/models")
print("target_dir   :", dst_dir)
print()

copied = []

for name in needed:
    src = None

    # first try a recursive search, but avoid preferring the target copy itself
    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend([p for p in root.rglob(name) if p.resolve() != (dst_dir / name).resolve()])

    if matches:
        # prefer shortest path outside destination
        matches = sorted(matches, key=lambda p: (str(p).count("\\"), len(str(p))))
        src = matches[0]

    if src is None:
        print(f"[MISSING SOURCE] {name}")
        continue

    out = dst_dir / name
    shutil.copy2(src, out)
    copied.append(out)
    print(f"[COPIED] {src}")
    print(f"         -> {out}")
    print()

print("HOST VERIFY")
for name in needed:
    p = dst_dir / name
    print(f"{'[OK]' if p.exists() else '[MISSING]'} {p}")

print()
print("CONTAINER VERIFY")
try:
    cmd = ["docker", "compose", "exec", "-T", "backend", "sh", "-lc", "ls -l /app/models/checkpoints/M20_*"]
    res = subprocess.run(cmd, cwd=compose_root, capture_output=True, text=True)
    if res.returncode == 0:
        print("[OK] backend container can see CLISA checkpoints:")
        print(res.stdout.strip())
    else:
        print("[WARN] Could not verify inside running container.")
        if res.stderr.strip():
            print(res.stderr.strip())
        print("If backend is running, restart it with: docker compose restart backend")
except Exception as e:
    print(f"[WARN] Docker check skipped: {e}")
    print("If using Docker, restart backend with: docker compose restart backend")

print()
print("DONE")
print("Now try Load & Verify again.")


# ==============================================================================
# Notebook cell 17
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import json, math, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# CLISA / M20 6ch DIAGNOSTIC CELL
# What it checks:
# 1) Is the app's UI accuracy just a 24-sample sanity check?
# 2) Does the web app use a different 6ch channel map than training?
# 3) Does normalization hurt/help?
# 4) How does M20_6ch perform on:
#       A) app map + app norm
#       B) training map + app norm
#       C) app map + no norm
#       D) training map + no norm
#    both on the first UI file and across ALL available files
# 5) Can it find LOSO JSON results for M20_6ch and compare
# ============================================================

# -----------------------------
# helpers
# -----------------------------
def find_project_root(start=Path.cwd()):
    start = start.resolve()
    cands = [start] + list(start.parents)
    for p in cands:
        if (p / "backend/app/services/model_loader.py").exists() and (p / "data/processed").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "backend/app/services/model_loader.py").exists() and (p / "data/processed").exists():
            return p
    raise FileNotFoundError("Could not find project root with backend/app/services/model_loader.py and data/processed")

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
        # already raw state_dict?
        if all(isinstance(v, (np.ndarray,)) for v in obj.values()):
            return obj, "raw_numpy_dict"
        if any("." in str(k) for k in obj.keys()):
            return obj, "raw_state_dict"
    return obj, "unknown"

def macro_f1(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
    return float(np.mean(f1s))

def balanced_acc(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recs = []
    for c in range(n_classes):
        mask = (y_true == c)
        denom = np.sum(mask)
        rec = np.sum(y_pred[mask] == c) / denom if denom > 0 else 0.0
        recs.append(rec)
    return float(np.mean(recs))

def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def label_dist(y, n_classes=4):
    y = np.asarray(y).astype(int)
    out = {}
    for c in range(n_classes):
        out[c] = int(np.sum(y == c))
    return out

# -----------------------------
# locate project + files
# -----------------------------
root = find_project_root()
processed_dir = root / "data" / "processed"
gt_dir        = root / "data" / "ground_truth"
norm_path     = root / "data" / "norm_stats" / "norm_62ch.npz"
map_path      = root / "backend" / "app" / "config" / "channel_map_62_to_6.json"

ckpt_candidates = list(root.rglob("M20_6ch_best.pt")) + list(root.rglob("M20_6ch_best.pth"))
if not ckpt_candidates:
    raise FileNotFoundError("Could not find M20_6ch_best.pt / .pth under project root")
ckpt_path = ckpt_candidates[0]

print("=" * 100)
print("CLISA / M20 6ch DIAGNOSTIC")
print("=" * 100)
print("project_root :", root)
print("checkpoint   :", ckpt_path)
print("processed_dir:", processed_dir)
print("gt_dir       :", gt_dir)
print("norm_path    :", norm_path, "->", "OK" if norm_path.exists() else "MISSING")
print("map_path     :", map_path, "->", "OK" if map_path.exists() else "MISSING")
print()

# -----------------------------
# import torch + define CLISA exactly like training notebook
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLISA(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=128, d_proj=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d_model), nn.BatchNorm1d(d_model), nn.ReLU()
        )
        self.projector  = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj)
        )
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

    def project(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=-1)

# -----------------------------
# load checkpoint
# -----------------------------
obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
state_dict, sd_source = extract_state_dict(obj)

model = CLISA(30)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
model.eval()

print("CHECKPOINT INSPECTION")
print("-" * 100)
print("state_dict source :", sd_source)
print("missing keys      :", len(missing))
print("unexpected keys   :", len(unexpected))
if missing:
    print("  missing sample  :", missing[:5])
if unexpected:
    print("  unexpected samp :", unexpected[:5])

enc0 = state_dict.get("encoder.0.weight", None)
clf  = state_dict.get("classifier.weight", None)
proj = state_dict.get("projector.0.weight", None)
print("encoder.0.weight  :", tuple(enc0.shape) if enc0 is not None else None)
print("projector.0.weight:", tuple(proj.shape) if proj is not None else None)
print("classifier.weight :", tuple(clf.shape) if clf is not None else None)
print()

# -----------------------------
# inspect maps + norm
# -----------------------------
TRAIN_CH6_IDX = [0, 2, 5, 13, 23, 31]  # from your training notebook
if map_path.exists():
    app_map = json.loads(map_path.read_text(encoding="utf-8")).get("indices", None)
else:
    app_map = None

print("CHANNEL MAP CHECK")
print("-" * 100)
print("training notebook CH6_IDX :", TRAIN_CH6_IDX)
print("webapp channel_map json   :", app_map)
print("same?                     :", app_map == TRAIN_CH6_IDX)
print()

if norm_path.exists():
    nd = np.load(str(norm_path))
    mean62 = nd["mean"].astype(np.float32)
    std62  = nd["std"].astype(np.float32)
    print("NORM CHECK")
    print("-" * 100)
    print("mean shape:", mean62.shape, "std shape:", std62.shape)
    print("mean range:", (float(mean62.min()), float(mean62.max())))
    print("std range :", (float(std62.min()),  float(std62.max())))
    print()
else:
    mean62 = np.zeros(310, dtype=np.float32)
    std62  = np.ones(310, dtype=np.float32)

# -----------------------------
# find data pairs
# -----------------------------
pairs = []
for f in sorted(processed_dir.glob("*.npy")):
    g = gt_dir / f.name
    if g.exists():
        pairs.append((f, g))

if not pairs:
    raise RuntimeError("No processed/ground_truth .npy pairs found")

first_feat, first_lab = pairs[0]
X0 = np.load(str(first_feat)).astype(np.float32)
Y0 = np.load(str(first_lab)).astype(np.int64)

print("GROUND-TRUTH VALIDATION FILE USED BY APP")
print("-" * 100)
print("first file used by UI:", first_feat.name)
print("shape                :", X0.shape, Y0.shape)
print("label dist           :", label_dist(Y0))
print("sample count         :", len(Y0), "(this is why UI shows 24 samples)")
print()

# -----------------------------
# preprocessing modes
# -----------------------------
def subset_6ch(x310, idx6):
    feat2d = x310.reshape(62, 5)
    return feat2d[idx6, :].reshape(-1).astype(np.float32)

def norm_from_62_to_6(idx6):
    m2 = mean62.reshape(62, 5)[idx6, :].reshape(-1)
    s2 = std62.reshape(62, 5)[idx6, :].reshape(-1)
    s2 = np.where(s2 > 1e-8, s2, 1.0)
    return m2.astype(np.float32), s2.astype(np.float32)

def preprocess(x310, idx6, use_norm=True):
    x30 = subset_6ch(x310, idx6)
    if use_norm:
        m, s = norm_from_62_to_6(idx6)
        x30 = (x30 - m) / s
    return x30.astype(np.float32)

@torch.no_grad()
def predict_batch(X30):
    x = torch.tensor(X30, dtype=torch.float32)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs

def eval_pair(feat_file, label_file, idx6, use_norm):
    X = np.load(str(feat_file)).astype(np.float32)
    Y = np.load(str(label_file)).astype(np.int64)
    Xp = np.stack([preprocess(x, idx6, use_norm=use_norm) for x in X], axis=0)
    preds, probs = predict_batch(Xp)
    return {
        "n": len(Y),
        "acc": acc(Y, preds),
        "bacc": balanced_acc(Y, preds),
        "mf1": macro_f1(Y, preds),
        "y_true": Y,
        "y_pred": preds,
        "conf": probs.max(axis=1),
    }

def eval_all_pairs(idx6, use_norm):
    ys, ps = [], []
    per_file = []
    for ff, lf in pairs:
        r = eval_pair(ff, lf, idx6, use_norm)
        ys.append(r["y_true"])
        ps.append(r["y_pred"])
        per_file.append((ff.name, r["acc"], r["bacc"], r["mf1"], r["n"]))
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return {
        "overall_acc": acc(ys, ps),
        "overall_bacc": balanced_acc(ys, ps),
        "overall_mf1": macro_f1(ys, ps),
        "n_total": len(ys),
        "per_file": per_file,
    }

modes = [
    ("APP_MAP + APP_NORM",   app_map, True),
    ("TRAIN_MAP + APP_NORM", TRAIN_CH6_IDX, True),
    ("APP_MAP + NO_NORM",    app_map, False),
    ("TRAIN_MAP + NO_NORM",  TRAIN_CH6_IDX, False),
]

print("EVALUATION COMPARISON")
print("-" * 100)

rows = []
for name, idx6, use_norm in modes:
    if idx6 is None:
        continue
    r_first = eval_pair(first_feat, first_lab, idx6, use_norm)
    r_all   = eval_all_pairs(idx6, use_norm)
    rows.append((name, r_first, r_all))
    print(f"{name}")
    print(f"  first-file  : acc={r_first['acc']:.4f}  bacc={r_first['bacc']:.4f}  mf1={r_first['mf1']:.4f}  n={r_first['n']}")
    print(f"  all-files   : acc={r_all['overall_acc']:.4f}  bacc={r_all['overall_bacc']:.4f}  mf1={r_all['overall_mf1']:.4f}  n={r_all['n_total']}")
    print()

# -----------------------------
# show file-level spread for the two key modes
# -----------------------------
def print_top_bottom(title, per_file):
    per_file = sorted(per_file, key=lambda t: t[1], reverse=True)
    print(title)
    print("  best 5 files:")
    for name, a, b, f, n in per_file[:5]:
        print(f"    {name:20s} acc={a:.4f} bacc={b:.4f} mf1={f:.4f} n={n}")
    print("  worst 5 files:")
    for name, a, b, f, n in per_file[-5:]:
        print(f"    {name:20s} acc={a:.4f} bacc={b:.4f} mf1={f:.4f} n={n}")
    print()

for name, r_first, r_all in rows:
    if name in {"APP_MAP + APP_NORM", "TRAIN_MAP + APP_NORM"}:
        print_top_bottom(name, r_all["per_file"])

# -----------------------------
# look for LOSO JSON results
# -----------------------------
print("LOSO JSON SEARCH")
print("-" * 100)

json_hits = []
search_roots = [root, root.parent, root.parent.parent]
seen = set()
for sr in search_roots:
    if sr.exists():
        for p in sr.rglob("*.json"):
            if p in seen:
                continue
            seen.add(p)
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if '"model_id"' in txt and '"M20"' in txt and '"6ch"' in txt:
                try:
                    d = json.loads(txt)
                    if d.get("model_id") == "M20" and d.get("ch") == "6ch":
                        json_hits.append((p, d))
                except Exception:
                    pass

if not json_hits:
    print("No M20 6ch LOSO JSON files found nearby.")
else:
    acc_a = [d["acc_a"] for _, d in json_hits if "acc_a" in d]
    acc_b = [d["acc_b"] for _, d in json_hits if "acc_b" in d]
    f1_a  = [d["f1_a"]  for _, d in json_hits if "f1_a" in d]
    f1_b  = [d["f1_b"]  for _, d in json_hits if "f1_b" in d]
    print(f"found JSON files: {len(json_hits)}")
    if acc_a:
        print(f"mean acc_a = {np.mean(acc_a):.4f}")
    if acc_b:
        print(f"mean acc_b = {np.mean(acc_b):.4f}")
    if f1_a:
        print(f"mean f1_a  = {np.mean(f1_a):.4f}")
    if f1_b:
        print(f"mean f1_b  = {np.mean(f1_b):.4f}")
    best = sorted(json_hits, key=lambda x: x[1].get("f1_a", -1), reverse=True)[0]
    print("\nbest by f1_a:")
    print("  file :", best[0])
    print("  data :", {k: best[1].get(k) for k in ["seed", "fold", "test_sub", "acc_a", "f1_a", "acc_b", "f1_b", "best_val_f1"]})
print()

# -----------------------------
# final interpretation
# -----------------------------
print("=" * 100)
print("INTERPRETATION GUIDE")
print("=" * 100)
print("1) If APP_MAP + APP_NORM is bad, but TRAIN_MAP + APP_NORM jumps clearly higher,")
print("   then the webapp 6ch channel map is wrong.")
print()
print("2) If both APP_NORM modes are bad, but NO_NORM improves a lot,")
print("   then the webapp normalization does not match the training pipeline.")
print()
print("3) If first-file accuracy is bad but all-files accuracy is much better,")
print("   then the UI's 24-sample check is just a noisy sanity check, not your LOSO result.")
print()
print("4) If the found LOSO JSON mean is much higher than the UI number,")
print("   that confirms the UI metric is not comparable to training-table performance.")
print("=" * 100)


# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import json, re, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PROOF CELL:
# 1) patch 6-ch channel map to the training map
# 2) evaluate many LOSO M20 6-ch checkpoints on the whole app dataset
# ------------------------------------------------------------

TRAIN_CH6_IDX = [0, 2, 5, 13, 23, 31]

def find_app_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "backend/app/config/channel_map_62_to_6.json").exists() and (p / "data/processed").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "backend/app/config/channel_map_62_to_6.json").exists() and (p / "data/processed").exists():
            return p
    raise FileNotFoundError("Could not find app root")

app_root = find_app_root()
search_base = app_root.parents[2] if len(app_root.parents) >= 3 else app_root.parent

map_path   = app_root / "backend/app/config/channel_map_62_to_6.json"
proc_dir   = app_root / "data/processed"
gt_dir     = app_root / "data/ground_truth"
norm_path  = app_root / "data/norm_stats/norm_62ch.npz"

print("=" * 100)
print("M20 6-CH LOSO CHECKPOINT DIAGNOSIS")
print("=" * 100)
print("app_root   :", app_root)
print("search_base:", search_base)
print()

# ------------------------------------------------------------
# 1) Patch channel map
# ------------------------------------------------------------
old_map = None
if map_path.exists():
    old_map = json.loads(map_path.read_text(encoding="utf-8"))
new_map = {"indices": TRAIN_CH6_IDX}
map_path.write_text(json.dumps(new_map, indent=2), encoding="utf-8")

print("CHANNEL MAP PATCH")
print("-" * 100)
print("old:", old_map)
print("new:", new_map)
print()

# ------------------------------------------------------------
# model
# ------------------------------------------------------------
class CLISA(nn.Module):
    def __init__(self, in_feat=30, n_cls=4, d_model=128, d_proj=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d_model), nn.BatchNorm1d(d_model), nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj)
        )
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if any("." in str(k) for k in obj.keys()):
            return obj
    return obj

# ------------------------------------------------------------
# data
# ------------------------------------------------------------
pairs = []
for f in sorted(proc_dir.glob("*.npy")):
    g = gt_dir / f.name
    if g.exists():
        pairs.append((f, g))
if not pairs:
    raise RuntimeError("No processed/ground_truth pairs found")

nd = np.load(str(norm_path))
mean62 = nd["mean"].astype(np.float32)
std62  = nd["std"].astype(np.float32)

def preprocess_batch(X310):
    Xout = []
    m = mean62.reshape(62, 5)[TRAIN_CH6_IDX, :].reshape(-1)
    s = std62.reshape(62, 5)[TRAIN_CH6_IDX, :].reshape(-1)
    s = np.where(s > 1e-8, s, 1.0)
    for x in X310:
        x30 = x.reshape(62, 5)[TRAIN_CH6_IDX, :].reshape(-1).astype(np.float32)
        x30 = (x30 - m) / s
        Xout.append(x30)
    return np.stack(Xout, axis=0)

def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

@torch.no_grad()
def eval_checkpoint(ckpt_path):
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = extract_state_dict(obj)
    model = CLISA()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        return None

    model.eval()
    ys, ps = [], []
    for ff, lf in pairs:
        X = np.load(str(ff)).astype(np.float32)
        Y = np.load(str(lf)).astype(np.int64)
        X30 = preprocess_batch(X)
        logits = model(torch.tensor(X30, dtype=torch.float32))
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(Y)
        ps.append(pred)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return acc(ys, ps)

# ------------------------------------------------------------
# 2) Search all M20 6ch LOSO checkpoints
# ------------------------------------------------------------
pat = re.compile(r"M20_6ch_s(\d+)_f(\d+)_best\.pth$", re.I)
hits = []
for p in search_base.rglob("M20_6ch_s*_f*_best.pth"):
    m = pat.search(p.name)
    if m:
        hits.append((p, int(m.group(1)), int(m.group(2))))

hits = sorted(hits, key=lambda t: (t[1], t[2]))

print("CHECKPOINT SCAN")
print("-" * 100)
print("found:", len(hits))
for p, s, f in hits[:10]:
    print(f"  seed={s:02d} fold={f:02d}  {p}")
if len(hits) > 10:
    print(f"  ... +{len(hits)-10} more")
print()

# ------------------------------------------------------------
# 3) Evaluate a subset/all checkpoints on whole app dataset
# ------------------------------------------------------------
results = []
for i, (p, s, f) in enumerate(hits):
    try:
        a = eval_checkpoint(p)
        if a is not None:
            results.append((a, s, f, p))
            print(f"[{i+1:02d}/{len(hits):02d}] seed={s:02d} fold={f:02d}  acc_all={a:.4f}")
    except Exception as e:
        print(f"[skip] seed={s:02d} fold={f:02d}  error={e}")

print()
print("SUMMARY")
print("-" * 100)
if not results:
    print("No checkpoints evaluated successfully.")
else:
    results = sorted(results, key=lambda t: t[0], reverse=True)
    print("top 10 checkpoints on WHOLE app dataset:")
    for a, s, f, p in results[:10]:
        print(f"  acc_all={a:.4f}  seed={s:02d} fold={f:02d}  {p.name}")

    best = results[0]
    print()
    print("best single LOSO checkpoint on whole dataset:")
    print(f"  acc_all={best[0]:.4f}  seed={best[1]:02d} fold={best[2]:02d}  {best[3].name}")

    mean_acc = float(np.mean([r[0] for r in results]))
    print(f"mean acc across all scanned LOSO checkpoints: {mean_acc:.4f}")

print()
print("=" * 100)
print("HOW TO READ THIS")
print("=" * 100)
print("If no single LOSO checkpoint gets anywhere near your expected deployment score,")
print("that confirms the problem: LOSO checkpoints are fold-specific and should not be")
print("used as one universal webapp model.")
print()
print("Next correct fix:")
print("1) keep the channel map patch")
print("2) train/export one FINAL global M20 6-ch model for deployment")
print("3) replace M20_6ch_best.pt with that final model")


# ==============================================================================
# Notebook cell 19
# Categories: preprocessing, evaluation, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# FEATURE PARITY DIAGNOSTIC
# Compares original training tensors vs current webapp tensors
# Goal:
#   1) Find original training feature files
#   2) Compare them to webapp data/processed + ground_truth
#   3) Check whether webapp data is actually the same feature space
# ============================================================

CH6_IDX = [0, 2, 5, 13, 23, 31]

def find_app_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data/processed").exists() and (p / "data/ground_truth").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "data/processed").exists() and (p / "data/ground_truth").exists():
            return p
    raise FileNotFoundError("Could not find app root")

def find_training_features(search_roots):
    needed = ["seed_iv_X_62ch.npy", "seed_iv_X_6ch.npy", "seed_iv_y_4cls.npy", "seed_iv_subjects.npy"]
    hits = {}
    for root in search_roots:
        if not root.exists():
            continue
        for name in needed:
            for p in root.rglob(name):
                hits[name] = p
                break
    return hits

def build_webapp_dataset(app_root):
    proc_dir = app_root / "data" / "processed"
    gt_dir   = app_root / "data" / "ground_truth"

    X62_list, X6_list, y_list, s_list, file_list = [], [], [], [], []
    for ff in sorted(proc_dir.glob("*.npy")):
        lf = gt_dir / ff.name
        if not lf.exists():
            continue

        X62 = np.load(str(ff)).astype(np.float32)
        y   = np.load(str(lf)).astype(np.int64)

        subj = int(ff.stem.split("_")[0].replace("sub", ""))
        X6   = X62.reshape(-1, 62, 5)[:, CH6_IDX, :].reshape(-1, 30).astype(np.float32)

        X62_list.append(X62)
        X6_list.append(X6)
        y_list.append(y)
        s_list.append(np.full(len(y), subj, dtype=np.int64))
        file_list.extend([ff.name] * len(y))

    X62w = np.concatenate(X62_list, axis=0)
    X6w  = np.concatenate(X6_list, axis=0)
    yw   = np.concatenate(y_list, axis=0)
    Sw   = np.concatenate(s_list, axis=0)
    return X62w, X6w, yw, Sw, file_list

def row_hashes(arr, decimals=5):
    arr = np.round(arr.astype(np.float32), decimals)
    return Counter(row.tobytes() for row in arr)

def overlap_ratio(A, B, decimals=5):
    cA = row_hashes(A, decimals=decimals)
    cB = row_hashes(B, decimals=decimals)
    shared = sum((cA & cB).values())
    denomA = max(1, sum(cA.values()))
    denomB = max(1, sum(cB.values()))
    return {
        "shared_rows": shared,
        "A_coverage": shared / denomA,
        "B_coverage": shared / denomB,
    }

def counts_by_subject_label(S, y):
    out = defaultdict(lambda: defaultdict(int))
    for s, yy in zip(S, y):
        out[int(s)][int(yy)] += 1
    return {k: dict(v) for k, v in sorted(out.items())}

def print_dist(name, y):
    vals, cnts = np.unique(y, return_counts=True)
    print(f"{name}: ", {int(v): int(c) for v, c in zip(vals, cnts)})

app_root = find_app_root()
search_roots = [
    app_root,
    app_root.parent,
    app_root.parent.parent,
    app_root.parent.parent.parent,
    Path(r"C:\Users\Saif\Desktop\CSE400\C"),
]

print("=" * 100)
print("FEATURE PARITY DIAGNOSTIC")
print("=" * 100)
print("app_root:", app_root)
print()

# ------------------------------------------------------------
# Webapp tensors
# ------------------------------------------------------------
X62w, X6w, yw, Sw, files = build_webapp_dataset(app_root)

print("WEBAPP DATA")
print("-" * 100)
print("X62w:", X62w.shape)
print("X6w :", X6w.shape)
print("yw  :", yw.shape)
print("Sw  :", Sw.shape)
print_dist("webapp label counts", yw)
print("webapp subject counts:", {int(s): int((Sw == s).sum()) for s in np.unique(Sw)})
print()

# ------------------------------------------------------------
# Original training tensors
# ------------------------------------------------------------
hits = find_training_features(search_roots)

print("TRAINING FEATURE FILE SEARCH")
print("-" * 100)
for k in ["seed_iv_X_62ch.npy", "seed_iv_X_6ch.npy", "seed_iv_y_4cls.npy", "seed_iv_subjects.npy"]:
    print(f"{k}: {hits.get(k, 'NOT FOUND')}")
print()

missing = [k for k in ["seed_iv_X_62ch.npy", "seed_iv_X_6ch.npy", "seed_iv_y_4cls.npy", "seed_iv_subjects.npy"] if k not in hits]
if missing:
    print("RESULT")
    print("-" * 100)
    print("Original training tensors were NOT found.")
    print("That strongly suggests the webapp is not using the same feature source that the training notebook used.")
    print("You need the original /features folder from the training project.")
    raise SystemExit

X62o = np.load(str(hits["seed_iv_X_62ch.npy"])).astype(np.float32)
X6o  = np.load(str(hits["seed_iv_X_6ch.npy"])).astype(np.float32)
Yo   = np.load(str(hits["seed_iv_y_4cls.npy"])).astype(np.int64)
So   = np.load(str(hits["seed_iv_subjects.npy"])).astype(np.int64)

print("ORIGINAL TRAINING DATA")
print("-" * 100)
print("X62o:", X62o.shape)
print("X6o :", X6o.shape)
print("Yo  :", Yo.shape)
print("So  :", So.shape)
print_dist("orig label counts", Yo)
print("orig subject counts:", {int(s): int((So == s).sum()) for s in np.unique(So)})
print()

# ------------------------------------------------------------
# Internal consistency of original training data
# ------------------------------------------------------------
X6o_from_62 = X62o.reshape(-1, 62, 5)[:, CH6_IDX, :].reshape(-1, 30).astype(np.float32)
diff = np.abs(X6o - X6o_from_62)
print("ORIGINAL INTERNAL CONSISTENCY")
print("-" * 100)
print("Does original X6 equal subset(original X62)?")
print("max abs diff :", float(diff.max()))
print("mean abs diff:", float(diff.mean()))
print("allclose     :", bool(np.allclose(X6o, X6o_from_62, atol=1e-5, rtol=1e-5)))
print()

# ------------------------------------------------------------
# Shape / label / subject parity
# ------------------------------------------------------------
print("DATASET PARITY")
print("-" * 100)
print("same X62 shape :", X62w.shape == X62o.shape)
print("same X6 shape  :", X6w.shape  == X6o.shape)
print("same y shape   :", yw.shape   == Yo.shape)
print("same S shape   :", Sw.shape   == So.shape)
print("same labels exact :", bool(np.array_equal(yw, Yo)))
print("same subjects exact:", bool(np.array_equal(Sw, So)))
print()

print("subject-label table equality?")
web_sl = counts_by_subject_label(Sw, yw)
org_sl = counts_by_subject_label(So, Yo)
same_sl = (web_sl == org_sl)
print("same subject-label counts:", same_sl)
if not same_sl:
    print("webapp subject-label counts:")
    print(web_sl)
    print("orig subject-label counts:")
    print(org_sl)
print()

# ------------------------------------------------------------
# Row-level overlap (order-independent)
# ------------------------------------------------------------
print("ROW-LEVEL FEATURE OVERLAP (ORDER-INDEPENDENT)")
print("-" * 100)

ov62 = overlap_ratio(X62w, X62o, decimals=5)
ov6  = overlap_ratio(X6w,  X6o,  decimals=5)

print("62ch overlap:", ov62)
print("6ch  overlap:", ov6)
print()

# ------------------------------------------------------------
# Quick stats comparison
# ------------------------------------------------------------
def stats_block(name, X):
    print(f"{name}:")
    print(f"  mean={float(X.mean()):.6f} std={float(X.std()):.6f}")
    print(f"  min ={float(X.min()):.6f} max={float(X.max()):.6f}")

print("GLOBAL STATS")
print("-" * 100)
stats_block("webapp X62", X62w)
stats_block("orig   X62", X62o)
stats_block("webapp X6 ", X6w)
stats_block("orig   X6 ", X6o)
print()

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
if ov62["A_coverage"] > 0.95 and ov6["A_coverage"] > 0.95:
    print("GOOD: webapp tensors and original training tensors are basically the same rows.")
    print("Then the remaining problem is likely label mapping / evaluation protocol / model implementation.")
elif ov62["A_coverage"] < 0.20 and ov6["A_coverage"] < 0.20:
    print("BAD: webapp tensors are materially different from original training tensors.")
    print("This is the most likely reason your webapp accuracy is far below the notebook LOSO result.")
    print("Fix path: rebuild webapp data from the original /features tensors, or make the app load those tensors directly.")
else:
    print("MIXED: some overlap exists, but webapp tensors are not a clean match to the original training tensors.")
    print("Most likely there is a reorder / partial regeneration / normalization / session-assembly mismatch.")

print("=" * 100)


# ==============================================================================
# Notebook cell 20
# Categories: model_definition, evaluation, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import re

# ------------------------------------------------------------
# SAFE DEMO PATCH:
# - CLISA-only model list
# - hide dataset loader on Device page
# - add explicit DEMO / SIMULATED wording
# ------------------------------------------------------------

def find_project_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "frontend/src/pages/DeviceHub.jsx").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "frontend/src/pages/DeviceHub.jsx").exists():
            return p
    raise FileNotFoundError("Could not find project root")

root = find_project_root()
devicehub = root / "frontend" / "src" / "pages" / "DeviceHub.jsx"

print("=" * 90)
print("PATCHING DEVICE PAGE FOR CLISA-ONLY DEMO MODE")
print("=" * 90)
print("project_root:", root)
print("file        :", devicehub)

backup = devicehub.with_suffix(".jsx.bak_demo")
if not backup.exists():
    shutil.copy2(devicehub, backup)
    print("[backup]", backup.name)

text = devicehub.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1) Force MODEL_VARIANTS to CLISA only
# ------------------------------------------------------------
model_variants_pattern = r"const MODEL_VARIANTS = \[(.*?)\n\]"
clisa_only_block = """const MODEL_VARIANTS = [
  { file: 'M20_6ch_best.pt',  label: 'M20 · 6-ch',  ch: 6,  family: 'CLISA', desc: 'Demo mode model option', default: true },
  { file: 'M20_62ch_best.pt', label: 'M20 · 62-ch', ch: 62, family: 'CLISA', desc: 'Demo mode model option' },
]"""

if re.search(model_variants_pattern, text, flags=re.DOTALL):
    text = re.sub(model_variants_pattern, clisa_only_block, text, flags=re.DOTALL)
    print("[patched] MODEL_VARIANTS -> CLISA only")
else:
    print("[warn] Could not patch MODEL_VARIANTS automatically")

# ------------------------------------------------------------
# 2) Add a DEMO_MODE constant near top if missing
# ------------------------------------------------------------
if "const DEMO_MODE = true;" not in text:
    insert_anchor = "const MODEL_VARIANTS = ["
    if insert_anchor in text:
        text = text.replace(insert_anchor, "const DEMO_MODE = true;\n\n" + insert_anchor, 1)
        print("[patched] DEMO_MODE constant added")

# ------------------------------------------------------------
# 3) Hide dataset loader section by replacing common heading text
#    and force explicit simulated wording
# ------------------------------------------------------------
replacements = {
    "Dataset Loader": "Demo Playback",
    "dataset loader": "demo playback",
    "Ground-Truth Validation": "Demo Validation",
    "ground-truth validation": "demo validation",
    "Load & Verify": "Load Demo Model",
    "Predicted Emotion": "Displayed Emotion (Simulated Demo)",
    "Prediction": "Demo Prediction",
}

for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)

# ------------------------------------------------------------
# 4) Remove obvious dataset-related blocks if present
#    This uses broad patterns and comments them out safely.
# ------------------------------------------------------------
block_patterns = [
    r"\{/\*\s*Dataset Loader\s*\*/.*?\n\s*\}",
    r"\{[^{}]*dataset[^{}]*loader[^{}]*\}",
]

for i, pat in enumerate(block_patterns, 1):
    new_text = re.sub(pat, "{/* Dataset loader removed in demo mode */}", text, flags=re.DOTALL | re.IGNORECASE)
    if new_text != text:
        text = new_text
        print(f"[patched] dataset block pattern {i}")

# ------------------------------------------------------------
# 5) Add explicit demo banner near top-level return if possible
# ------------------------------------------------------------
banner = """
      {DEMO_MODE && (
        <div style={{
          marginBottom: '12px',
          padding: '10px 14px',
          borderRadius: '10px',
          background: '#fff3cd',
          color: '#856404',
          border: '1px solid #ffe69c',
          fontWeight: 600
        }}>
          Demo Mode: emotions and predictions shown here are simulated for UI demonstration.
        </div>
      )}
"""

return_anchor_patterns = [
    r"return\s*\(\s*<",
    r"return\s*\(\s*\n\s*<",
]

inserted_banner = False
for pat in return_anchor_patterns:
    m = re.search(pat, text)
    if m:
        idx = m.end() - 1
        text = text[:idx] + "\n" + banner + text[idx:]
        inserted_banner = True
        print("[patched] demo banner inserted")
        break

if not inserted_banner:
    print("[warn] Could not automatically insert demo banner")

# ------------------------------------------------------------
# 6) Make synthetic emotion wording explicit if common labels exist
# ------------------------------------------------------------
text = text.replace("selectedEmotion", "selectedEmotion")
text = text.replace("currentEmotion", "currentEmotion")

# Add helper comment so you can find it later
if "SIMULATED DEMO MODE PATCH" not in text:
    text = "// SIMULATED DEMO MODE PATCH\n" + text

devicehub.write_text(text, encoding="utf-8")
print("[saved]", devicehub)

print()
print("DONE")
print("Next:")
print("1) restart frontend")
print("2) verify only M20 models appear")
print("3) verify dataset loader UI is gone")
print("4) verify demo banner appears")


# ==============================================================================
# Notebook cell 21
# Categories: preprocessing, model_definition, training, evaluation, results_tables, figures, statistics, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import re
import json

# ============================================================
# CLISA-ONLY HONEST DEMO MODE PATCH
# - Only CLISA visible in UI
# - Remove dataset/replay card from Device page
# - Simulator becomes Demo Mode
# - Synthetic playback follows chosen emotion exactly
# - Live page shows chosen/simulated emotion consistently
# - Backend default model -> M20_6ch_best.pt
# ============================================================

def find_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "frontend/src/pages/DeviceHub.jsx").exists() and (p / "backend/app/services/eeg_simulator.py").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "frontend/src/pages/DeviceHub.jsx").exists() and (p / "backend/app/services/eeg_simulator.py").exists():
            return p
    raise FileNotFoundError("Could not find project root")

ROOT = find_root()

FILES = {
    "devicehub": ROOT / "frontend/src/pages/DeviceHub.jsx",
    "live": ROOT / "frontend/src/pages/Live.jsx",
    "home": ROOT / "frontend/src/pages/Home.jsx",
    "results": ROOT / "frontend/src/pages/Results.jsx",
    "device_router": ROOT / "backend/app/routers/device.py",
    "inference_router": ROOT / "backend/app/routers/inference.py",
    "simulator": ROOT / "backend/app/services/eeg_simulator.py",
}

print("=" * 90)
print("PATCHING CLISA-ONLY DEMO MODE")
print("=" * 90)
print("ROOT:", ROOT)
for k, p in FILES.items():
    print(f"{k:16s} -> {p}")

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_demo")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {bak.name}")

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"[saved] {path.relative_to(ROOT)}")

def must_replace(text, old, new, label):
    if old not in text:
        print(f"[warn] {label}: anchor not found")
        return text
    print(f"[ok] {label}")
    return text.replace(old, new)

def regex_replace(text, pattern, repl, label, flags=re.DOTALL):
    new_text, n = re.subn(pattern, repl, text, flags=flags)
    if n == 0:
        print(f"[warn] {label}: pattern not found")
        return text
    print(f"[ok] {label}: {n} replacement(s)")
    return new_text

# ------------------------------------------------------------
# 1) backend/app/services/eeg_simulator.py
#    Force synthetic playback to the chosen emotion exactly.
# ------------------------------------------------------------
p = FILES["simulator"]
backup(p)
txt = read(p)

sim_pattern = r"""def infer_emotion\(self\) -> dict:\n(?:    .*?\n)+?    def _prototype_fallback"""
sim_repl = """def infer_emotion(self) -> dict:
        \"\"\"Return a deterministic DEMO prediction that matches the configured synthetic emotion.\"\"\"
        if self._current_de is None:
            self.generate_de_features()

        cfg = self.config
        target = cfg.emotion if cfg.emotion in EMOTIONS else "Neutral"

        # Honest demo-mode confidence: plausible but clearly synthetic / controlled
        conf = float(np.clip(
            0.55
            + 0.18 * float(cfg.emotion_intensity)
            + 0.12 * float(cfg.signal_stability)
            + 0.05 * float(cfg.signal_strength)
            - 0.10 * float(cfg.artifact_burden)
            - 0.05 * float(cfg.channel_dropout),
            0.55, 0.90
        ))

        remainder = max(0.0, 1.0 - conf)
        other = remainder / 3.0

        probs = {emo: other for emo in EMOTIONS}
        probs[target] = conf

        # Normalize after float math
        s = sum(probs.values())
        probs = {k: float(v / s) for k, v in probs.items()}

        ordered = sorted(probs.values(), reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0

        result = {
            "emotion": target,
            "confidence": float(probs[target]),
            "probabilities": probs,
            "margin": margin,
            "source": "demo_simulator",
            "input_adapted": False,
            "simulator_mode": True,
            "demo_mode": True,
            "target_emotion": target,
            "mode_label": "Demo Mode - synthetic emotion playback",
        }
        return result

    def _prototype_fallback"""
txt = regex_replace(txt, sim_pattern, sim_repl, "simulator infer_emotion patch")

write(p, txt)

# ------------------------------------------------------------
# 2) backend/app/routers/inference.py
#    Make simulator stable output immediately follow chosen emotion.
# ------------------------------------------------------------
p = FILES["inference_router"]
backup(p)
txt = read(p)

txt = must_replace(
    txt,
    '"synthetic_augmentation": "Synthetic Test Session",',
    '"synthetic_augmentation": "Demo Mode - Synthetic Emotion Playback",',
    "source label"
)

old_stable = """                inference_stable = stabilizer.update(raw_emotion, raw_conf, raw_probs, raw_margin)
"""
new_stable = """                inference_stable = {
                    "emotion": raw_emotion,
                    "confidence": raw_conf,
                    "probabilities": raw_probs,
                    "margin": raw_margin,
                    "status": "accepted",
                    "stable_for_seconds": 0.0,
                    "candidate_emotion": None,
                    "candidate_count": 0,
                    "thresholds": {
                        "confidence_threshold": stabilizer.confidence_threshold,
                        "margin_threshold": stabilizer.margin_threshold,
                        "min_consistent_frames": 1,
                        "min_hold_seconds": 0.0,
                    },
                    "demo_mode": True,
                }
"""
txt = must_replace(txt, old_stable, new_stable, "simulator immediate stable output")

write(p, txt)

# ------------------------------------------------------------
# 3) backend/app/routers/device.py
#    Switch defaults to M20, relabel simulator as Demo Mode.
# ------------------------------------------------------------
p = FILES["device_router"]
backup(p)
txt = read(p)

txt = txt.replace("M26_6ch_best.pt", "M20_6ch_best.pt")
txt = txt.replace("Synthetic Augmentation (Test Only)", "Demo Mode - Synthetic Emotion Playback")
txt = txt.replace("NeuroSim Engine", "NeuroSync Demo Engine")
txt = txt.replace("Synthetic EEG signals for artifact injection and stress testing", "Synthetic emotion playback for clearly labeled UI demos")
txt = txt.replace('"label":       "Synthetic Augmentation (Test Only)",', '"label":       "Demo Mode - Synthetic Emotion Playback",')
txt = txt.replace('"description": "Synthetic EEG signals for artifact injection and stress testing",', '"description": "Synthetic emotion playback for clearly labeled UI demos",')
write(p, txt)

# ------------------------------------------------------------
# 4) frontend/src/pages/DeviceHub.jsx
#    CLISA only, remove dataset card, rename simulator card to Demo Mode,
#    hide GT accuracy block, relabel verification.
# ------------------------------------------------------------
p = FILES["devicehub"]
backup(p)
txt = read(p)

# CLISA-only models
model_pattern = r"const MODEL_VARIANTS = \[(.*?)\n\]"
model_repl = """const MODEL_VARIANTS = [
  { file: 'M20_6ch_best.pt',  label: 'M20 · 6-ch',  ch: 6,  family: 'CLISA', desc: 'Demo Mode synthetic emotion playback', default: true },
  { file: 'M20_62ch_best.pt', label: 'M20 · 62-ch', ch: 62, family: 'CLISA', desc: 'Demo Mode synthetic emotion playback' },
]"""
txt = regex_replace(txt, model_pattern, model_repl, "CLISA-only model list")

# Page header
txt = txt.replace('title="Device & Data Setup"', 'title="Demo Mode Setup"')
txt = txt.replace('subtitle="Configure your data source for live mental state monitoring"', 'subtitle="Configure CLISA demo playback with synthetic emotions"')

# Remove dataset card block
dataset_block_pattern = r"""\n        /\*.*?Card 2: SEED-IV Replay.*?\*/.*?\n        /\*.*?Card 3: Synthetic Test Mode"""
dataset_repl = """
        {/* Demo note card */}
        <div className="card p-5 flex flex-col gap-3 border-ns-primary/15 bg-ns-primary/3">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-lg bg-ns-primary/10 flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-ns-primary" style={{ fontSize: 22 }}>info</span>
            </div>
            <div>
              <h2 className="font-headline font-bold text-ns-text text-sm">Demo Mode</h2>
              <p className="text-[11px] text-ns-outline mt-0.5">CLISA-only synthetic emotion playback</p>
            </div>
          </div>
          <div className="p-2 bg-ns-primary/5 border border-ns-primary/15 rounded text-[10px] text-ns-primary">
            This build is set to Demo Mode. Synthetic playback follows the chosen emotion and is clearly labeled as simulated.
          </div>
        </div>

        {/* Card 3: Demo Mode"""
txt = regex_replace(txt, dataset_block_pattern, dataset_repl, "remove dataset card")

# Rename simulator card text
repl_map = {
    "Synthetic Test Mode": "Demo Mode",
    "TEST ONLY": "SIMULATED",
    "Augmentation · artifact injection · stress testing": "Choose an emotion and play back synthetic CLISA demo output",
    "Synthetic EEG — not real data": "Synthetic emotion playback",
    "Noise · drift · channel dropout controls": "Chosen emotion drives the displayed prediction",
    "Artifact injection presets": "CLISA-only demo source",
    "Results marked Synthetic Augmentation": "All outputs labeled Demo Mode",
    "⚠ Synthetic Augmentation (Test Only) — results are NOT validated against real data": "Demo Mode - clearly labeled simulated playback",
    "Start Synthetic Test Mode": "Start Demo Mode",
    "Stop Simulator": "Stop Demo Mode",
    "Synthetic Test Mode active — results labeled as simulated.": "Demo Mode active - simulated emotion playback enabled.",
    "AI Model · Checkpoint Selection & Verification": "CLISA Demo Model",
    "Model Verification Report": "Demo Model Status",
    "Load & Verify": "Load Demo Model",
    "Ground-Truth Validation (SEED-IV)": "Demo Status",
}
for old, new in repl_map.items():
    txt = txt.replace(old, new)

# Hide GT accuracy block in VerificationPanel
gt_block_pattern = r"""\n\s*\/\* GT accuracy \*\/.*?\n\s*\)\}\n"""
txt = regex_replace(txt, gt_block_pattern, "\n", "hide GT accuracy block")

write(p, txt)

# ------------------------------------------------------------
# 5) frontend/src/pages/Live.jsx
#    Show Demo Mode labeling and chosen emotion context.
# ------------------------------------------------------------
p = FILES["live"]
backup(p)
txt = read(p)

txt = txt.replace(
    "const sourceLabel     = liveFrame?.source_label ?? (isDataset ? 'SEED-IV Trial Replay' : 'Synthetic Test Session')",
    "const sourceLabel     = liveFrame?.source_label ?? (isDataset ? 'SEED-IV Trial Replay' : 'Demo Mode - Synthetic Emotion Playback')"
)

# Add demo banner before status bar
live_anchor = """  return (
    <div className="p-4 lg:p-6 max-w-7xl mx-auto">

      {/* Status Bar */}"""
live_insert = """  return (
    <div className="p-4 lg:p-6 max-w-7xl mx-auto">

      {isSimulator && (
        <div className="mb-4 px-4 py-3 rounded-xl border border-ns-secondary/20 bg-ns-secondary/5">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-ns-secondary">Demo Mode</p>
              <p className="text-[11px] text-ns-outline mt-1">
                Synthetic playback is active. Displayed emotion follows the selected demo emotion.
              </p>
            </div>
            <div className="px-3 py-1 rounded-lg bg-ns-secondary/10 text-ns-secondary text-xs font-bold">
              Target Emotion: {simConfig?.emotion || 'Neutral'}
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}"""
txt = must_replace(txt, live_anchor, live_insert, "live demo banner")

txt = txt.replace("Mental State", "Displayed Emotion")
txt = txt.replace("Synthetic Augmentation (Test Only)", "Demo Mode")
txt = txt.replace(
    "Data is synthetically generated. Results are not validated. Use SEED-IV Replay for verified inference.",
    "This is a clearly labeled demo. Synthetic playback follows the selected emotion and the displayed output is simulated."
)

write(p, txt)

# ------------------------------------------------------------
# 6) frontend/src/pages/Home.jsx and Results.jsx
#    Relabel simulator sessions as Demo Mode.
# ------------------------------------------------------------
for key in ["home", "results"]:
    p = FILES[key]
    backup(p)
    txt = read(p)
    txt = txt.replace("Synthetic Test", "Demo Mode")
    txt = txt.replace("Synthetic Test Session", "Demo Mode")
    txt = txt.replace("Synthetic Augmentation", "Demo Mode")
    write(p, txt)

# ------------------------------------------------------------
# 7) Optional checkpoint cleanup: keep files, but note CLISA-only UI
# ------------------------------------------------------------
ckpt_dir = ROOT / "models" / "checkpoints"
if ckpt_dir.exists():
    keep = {"M20_6ch_best.pt", "M20_62ch_best.pt"}
    others = sorted([p.name for p in ckpt_dir.glob("*.pt") if p.name not in keep])
    print()
    print("Checkpoint files still on disk (hidden from UI, not deleted):")
    for name in others[:20]:
        print(" -", name)
    if len(others) > 20:
        print(f" - ... +{len(others) - 20} more")

print()
print("=" * 90)
print("DONE")
print("=" * 90)
print("Next steps:")
print("1) From repo root: docker compose up --build")
print("   or frontend only: cd frontend && npm install && npm run dev")
print("2) Open Device page")
print("3) You should see only CLISA options")
print("4) Start Demo Mode, choose an emotion in the simulator tray, then start session")
print("5) Live page should show that chosen emotion consistently, labeled as Demo Mode")


# ==============================================================================
# Notebook cell 22
# Categories: webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil

# ------------------------------------------------------------
# Restore DeviceHub.jsx from backup so frontend can build again
# ------------------------------------------------------------

def find_root(start=Path.cwd()):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "frontend/src/pages/DeviceHub.jsx").exists():
            return p
    for p in start.rglob("*"):
        if p.is_dir() and (p / "frontend/src/pages/DeviceHub.jsx").exists():
            return p
    raise FileNotFoundError("Could not find project root")

root = find_root()
devicehub = root / "frontend" / "src" / "pages" / "DeviceHub.jsx"

candidates = [
    devicehub.with_suffix(".jsx.bak_demo"),
    devicehub.with_suffix(".jsx.bak"),
    devicehub.with_suffix(".bak"),
]

print("=" * 90)
print("RESTORING DEVICEHUB.JSX")
print("=" * 90)
print("target:", devicehub)

restored = False
for bak in candidates:
    if bak.exists():
        shutil.copy2(bak, devicehub)
        print(f"[RESTORED] from {bak}")
        restored = True
        break

if not restored:
    raise FileNotFoundError(
        "No backup found. Look for one of: DeviceHub.jsx.bak_demo, DeviceHub.jsx.bak, DeviceHub.bak"
    )

print()
print("Now rebuild:")
print("1) cd frontend && npm install && npm run build")
print("or")
print("2) docker compose up --build")


# ==============================================================================
# Notebook cell 23
# Categories: preprocessing, model_definition, training, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import json
import zipfile
import hashlib
import numpy as np
from datetime import datetime

# ============================================================
# CLISA BACKEND ASSET BUNDLE PREP
# - standalone cell
# - finds original training-compatible features
# - finds CLISA model files
# - copies everything into one clean bundle
# - writes metadata/config/preprocessing contract
# - creates a zip for later backend integration
# ============================================================

# -----------------------------
# config
# -----------------------------
CH6_IDX = [0, 2, 5, 13, 23, 31]
BUNDLE_DIRNAME = "backend_assets_clisa"
ZIP_NAME = "backend_assets_clisa.zip"

FEATURE_FILES = {
    "X_62ch": "seed_iv_X_62ch.npy",
    "X_6ch": "seed_iv_X_6ch.npy",
    "y_4cls": "seed_iv_y_4cls.npy",
    "subjects": "seed_iv_subjects.npy",
}

MODEL_PREFERENCES = {
    "M20_6ch": ["M20_6ch_best.pt", "M20_6ch_best.pth"],
    "M20_62ch": ["M20_62ch_best.pt", "M20_62ch_best.pth"],
}

# -----------------------------
# helpers
# -----------------------------
def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def find_project_root(start=Path.cwd()):
    start = start.resolve()

    # direct upward search
    for p in [start] + list(start.parents):
        if (p / "features" / FEATURE_FILES["X_62ch"]).exists():
            return p

    # wider search
    for p in start.rglob(FEATURE_FILES["X_62ch"]):
        if p.name == FEATURE_FILES["X_62ch"] and p.parent.name == "features":
            return p.parent.parent

    raise FileNotFoundError(
        f"Could not find project root containing features/{FEATURE_FILES['X_62ch']}"
    )

def find_first_by_names(search_root: Path, names):
    hits = []
    for name in names:
        for p in search_root.rglob(name):
            if p.is_file():
                hits.append(p)
    if not hits:
        return None

    # prefer shorter / more deployment-like paths
    def score(p):
        s = str(p).lower()
        return (
            0 if "webapp" in s else 1,
            0 if "models\\checkpoints" in s or "models/checkpoints" in s else 1,
            len(str(p))
        )
    hits = sorted(hits, key=score)
    return hits[0]

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def print_tree(root: Path, max_depth=3):
    root = root.resolve()
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        prefix = "[D]" if p.is_dir() else "[F]"
        print(f"{indent}{prefix} {p.relative_to(root)}")

# -----------------------------
# locate source project
# -----------------------------
project_root = find_project_root()
features_root = project_root / "features"
bundle_root = project_root / BUNDLE_DIRNAME

data_dir = bundle_root / "data" / "features"
models_dir = bundle_root / "models" / "checkpoints"
config_dir = bundle_root / "config"
refs_dir = bundle_root / "references" / "loso_results"
meta_dir = bundle_root / "metadata"

print("=" * 100)
print("CLISA BACKEND ASSET BUNDLE PREP")
print("=" * 100)
print("project_root :", project_root)
print("features_root:", features_root)
print("bundle_root  :", bundle_root)
print()

# clean bundle dir if it exists
if bundle_root.exists():
    shutil.rmtree(bundle_root)
bundle_root.mkdir(parents=True, exist_ok=True)

# -----------------------------
# copy feature tensors
# -----------------------------
resolved_features = {}
for key, fname in FEATURE_FILES.items():
    src = features_root / fname
    if not src.exists():
        raise FileNotFoundError(f"Missing required feature file: {src}")
    dst = data_dir / fname
    safe_copy(src, dst)
    resolved_features[key] = {"src": str(src), "dst": str(dst)}

print("COPIED FEATURES")
for k, v in resolved_features.items():
    print(f" - {k:10s} {v['src']}")

print()

# -----------------------------
# copy preferred CLISA models
# -----------------------------
resolved_models = {}
for model_key, name_list in MODEL_PREFERENCES.items():
    src = find_first_by_names(project_root, name_list)
    if src is None:
        resolved_models[model_key] = None
        print(f"[WARN] Could not find model for {model_key}: searched {name_list}")
        continue

    # keep same filename found
    dst = models_dir / src.name
    safe_copy(src, dst)
    resolved_models[model_key] = {"src": str(src), "dst": str(dst)}
    print(f"[OK] {model_key:8s} -> {src}")

print()

# -----------------------------
# copy M20 LOSO json references if present
# -----------------------------
loso_count = 0
loso_root = project_root / "checkpoints" / "loso_results"
if loso_root.exists():
    for p in sorted(loso_root.glob("*.json")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if '"model_id"' in txt and '"M20"' in txt:
            safe_copy(p, refs_dir / p.name)
            loso_count += 1

print(f"LOSO JSON COPIED: {loso_count}")
print()

# -----------------------------
# inspect dataset
# -----------------------------
X62 = np.load(features_root / FEATURE_FILES["X_62ch"]).astype(np.float32)
X6  = np.load(features_root / FEATURE_FILES["X_6ch"]).astype(np.float32)
y   = np.load(features_root / FEATURE_FILES["y_4cls"]).astype(np.int64)
sub = np.load(features_root / FEATURE_FILES["subjects"]).astype(np.int64)

X6_from_62 = X62.reshape(-1, 62, 5)[:, CH6_IDX, :].reshape(-1, 30).astype(np.float32)

dataset_summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "dataset_type": "training_compatible_standardized_features",
    "X_62ch_shape": list(X62.shape),
    "X_6ch_shape": list(X6.shape),
    "y_shape": list(y.shape),
    "subjects_shape": list(sub.shape),
    "label_counts": {str(int(c)): int((y == c).sum()) for c in np.unique(y)},
    "subject_counts": {str(int(s)): int((sub == s).sum()) for s in np.unique(sub)},
    "global_stats": {
        "X62_mean": float(X62.mean()),
        "X62_std": float(X62.std()),
        "X62_min": float(X62.min()),
        "X62_max": float(X62.max()),
        "X6_mean": float(X6.mean()),
        "X6_std": float(X6.std()),
        "X6_min": float(X6.min()),
        "X6_max": float(X6.max()),
    },
    "consistency_check": {
        "X6_equals_subset_of_X62": bool(np.allclose(X6, X6_from_62, atol=1e-6, rtol=1e-6)),
        "subset_indices": CH6_IDX,
    },
}

(meta_dir / "dataset_summary.json").write_text(
    json.dumps(dataset_summary, indent=2), encoding="utf-8"
)

# -----------------------------
# write config + preprocessing contract
# -----------------------------
channel_map = {
    "_comment": "6-channel subset used by CLISA training and deployment",
    "indices": CH6_IDX,
    "channel_names": ["FP1_like", "AF3_like", "F7_like", "FC5_like", "T7_like", "CP5_like"],
    "n_source_channels": 62,
    "n_target_channels": 6,
    "n_bands": 5,
    "source_input_dim": 310,
    "target_input_dim": 30,
}
(config_dir / "channel_map_62_to_6.json").write_text(
    json.dumps(channel_map, indent=2), encoding="utf-8"
)

preprocessing_contract = {
    "mode": "training_compatible",
    "primary_input_for_M20_6ch": "data/features/seed_iv_X_6ch.npy",
    "primary_input_for_M20_62ch": "data/features/seed_iv_X_62ch.npy",
    "do_not_use": [
        "webapp data/processed for notebook-faithful evaluation",
        "extra normalization on top of seed_iv_X_6ch.npy and seed_iv_X_62ch.npy"
    ],
    "notes": [
        "Original feature tensors are already in the training-compatible standardized feature space.",
        "For 6ch CLISA, X_6ch is exactly the subset of X_62ch using CH6_IDX.",
        "Use these tensors directly for backend playback/evaluation.",
    ],
    "channel_subset_indices_for_6ch": CH6_IDX,
}
(meta_dir / "preprocessing_contract.json").write_text(
    json.dumps(preprocessing_contract, indent=2), encoding="utf-8"
)

# -----------------------------
# write model registry
# -----------------------------
model_registry = {
    "primary_model": "M20_6ch",
    "models": {
        "M20_6ch": {
            "family": "CLISA",
            "channels": 6,
            "input_dim": 30,
            "preferred_checkpoint": (
                Path(resolved_models["M20_6ch"]["dst"]).name if resolved_models["M20_6ch"] else None
            ),
        },
        "M20_62ch": {
            "family": "CLISA",
            "channels": 62,
            "input_dim": 310,
            "preferred_checkpoint": (
                Path(resolved_models["M20_62ch"]["dst"]).name if resolved_models["M20_62ch"] else None
            ),
        },
    },
}
(meta_dir / "model_registry.json").write_text(
    json.dumps(model_registry, indent=2), encoding="utf-8"
)

# -----------------------------
# write manifest
# -----------------------------
manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(project_root),
    "bundle_root": str(bundle_root),
    "features": {
        k: {
            "source": v["src"],
            "bundle_path": str(Path(v["dst"]).relative_to(bundle_root)),
            "sha256": sha256_file(Path(v["dst"]))
        }
        for k, v in resolved_features.items()
    },
    "models": {
        k: (
            {
                "source": v["src"],
                "bundle_path": str(Path(v["dst"]).relative_to(bundle_root)),
                "sha256": sha256_file(Path(v["dst"]))
            } if v is not None else None
        )
        for k, v in resolved_models.items()
    },
    "loso_json_count": loso_count,
}
(meta_dir / "asset_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

# -----------------------------
# write README
# -----------------------------
readme = f"""CLISA backend asset bundle

Contents:
- data/features/
  - seed_iv_X_62ch.npy
  - seed_iv_X_6ch.npy
  - seed_iv_y_4cls.npy
  - seed_iv_subjects.npy
- models/checkpoints/
  - preferred CLISA checkpoint files
- config/
  - channel_map_62_to_6.json
- references/loso_results/
  - M20 LOSO jsons if available
- metadata/
  - dataset_summary.json
  - preprocessing_contract.json
  - model_registry.json
  - asset_manifest.json

Recommended backend usage:
- For notebook-faithful playback/evaluation:
  - use data/features/seed_iv_X_6ch.npy for M20_6ch
  - use data/features/seed_iv_X_62ch.npy for M20_62ch
- Do not apply extra normalization on top of these original feature tensors.
"""
(bundle_root / "README.txt").write_text(readme, encoding="utf-8")

# -----------------------------
# zip bundle
# -----------------------------
zip_path = project_root / ZIP_NAME
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in bundle_root.rglob("*"):
        zf.write(p, arcname=str(p.relative_to(project_root)))

# -----------------------------
# report
# -----------------------------
print("=" * 100)
print("DONE")
print("=" * 100)
print("Bundle folder :", bundle_root)
print("Zip file      :", zip_path)
print()
print("Bundle tree:")
print_tree(bundle_root, max_depth=4)
print()
print("Dataset summary:")
print(f" - X62 shape : {tuple(X62.shape)}")
print(f" - X6 shape  : {tuple(X6.shape)}")
print(f" - y shape   : {tuple(y.shape)}")
print(f" - subjects  : {tuple(sub.shape)}")
print(f" - X6 matches subset(X62): {np.allclose(X6, X6_from_62, atol=1e-6, rtol=1e-6)}")
print()
print("Next backend use:")
print(" - primary data source for CLISA 6ch  -> backend_assets_clisa/data/features/seed_iv_X_6ch.npy")
print(" - primary data source for CLISA 62ch -> backend_assets_clisa/data/features/seed_iv_X_62ch.npy")
print(" - primary config                     -> backend_assets_clisa/config/channel_map_62_to_6.json")
print(" - primary registry                   -> backend_assets_clisa/metadata/model_registry.json")


# ==============================================================================
# Notebook cell 24
# Categories: preprocessing, model_definition, training, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import json
import zipfile
import hashlib
import numpy as np
from datetime import datetime

CH6_IDX = [0, 2, 5, 13, 23, 31]

project_root = Path(r"C:\Users\Saif\Desktop\CSE400\C")
bundle_root = project_root / "backend_assets_clisa"

data_dir = bundle_root / "data" / "features"
models_dir = bundle_root / "models" / "checkpoints"
config_dir = bundle_root / "config"
refs_dir = bundle_root / "references" / "loso_results"
meta_dir = bundle_root / "metadata"

# FIX: create missing folders
for d in [data_dir, models_dir, config_dir, refs_dir, meta_dir]:
    d.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------
# load bundled features
# ------------------------------------------------------------
X62_path = data_dir / "seed_iv_X_62ch.npy"
X6_path  = data_dir / "seed_iv_X_6ch.npy"
y_path   = data_dir / "seed_iv_y_4cls.npy"
s_path   = data_dir / "seed_iv_subjects.npy"

X62 = np.load(X62_path).astype(np.float32)
X6  = np.load(X6_path).astype(np.float32)
y   = np.load(y_path).astype(np.int64)
sub = np.load(s_path).astype(np.int64)

X6_from_62 = X62.reshape(-1, 62, 5)[:, CH6_IDX, :].reshape(-1, 30).astype(np.float32)

# ------------------------------------------------------------
# dataset summary
# ------------------------------------------------------------
dataset_summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "dataset_type": "training_compatible_standardized_features",
    "X_62ch_shape": list(X62.shape),
    "X_6ch_shape": list(X6.shape),
    "y_shape": list(y.shape),
    "subjects_shape": list(sub.shape),
    "label_counts": {str(int(c)): int((y == c).sum()) for c in np.unique(y)},
    "subject_counts": {str(int(s)): int((sub == s).sum()) for s in np.unique(sub)},
    "global_stats": {
        "X62_mean": float(X62.mean()),
        "X62_std": float(X62.std()),
        "X62_min": float(X62.min()),
        "X62_max": float(X62.max()),
        "X6_mean": float(X6.mean()),
        "X6_std": float(X6.std()),
        "X6_min": float(X6.min()),
        "X6_max": float(X6.max()),
    },
    "consistency_check": {
        "X6_equals_subset_of_X62": bool(np.allclose(X6, X6_from_62, atol=1e-6, rtol=1e-6)),
        "subset_indices": CH6_IDX,
    },
}
(meta_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# config
# ------------------------------------------------------------
channel_map = {
    "_comment": "6-channel subset used by CLISA training and deployment",
    "indices": CH6_IDX,
    "n_source_channels": 62,
    "n_target_channels": 6,
    "n_bands": 5,
    "source_input_dim": 310,
    "target_input_dim": 30,
}
(config_dir / "channel_map_62_to_6.json").write_text(json.dumps(channel_map, indent=2), encoding="utf-8")

preprocessing_contract = {
    "mode": "training_compatible",
    "primary_input_for_M20_6ch": "data/features/seed_iv_X_6ch.npy",
    "primary_input_for_M20_62ch": "data/features/seed_iv_X_62ch.npy",
    "do_not_use": [
        "webapp data/processed for notebook-faithful evaluation",
        "extra normalization on top of seed_iv_X_6ch.npy and seed_iv_X_62ch.npy"
    ],
    "notes": [
        "Original feature tensors are already in the training-compatible standardized feature space.",
        "For 6ch CLISA, X_6ch is exactly the subset of X_62ch using CH6_IDX.",
        "Use these tensors directly for backend playback/evaluation."
    ],
    "channel_subset_indices_for_6ch": CH6_IDX
}
(meta_dir / "preprocessing_contract.json").write_text(json.dumps(preprocessing_contract, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# model registry
# ------------------------------------------------------------
m20_6 = next(models_dir.glob("M20_6ch_best.*"), None)
m20_62 = next(models_dir.glob("M20_62ch_best.*"), None)

model_registry = {
    "primary_model": "M20_6ch",
    "models": {
        "M20_6ch": {
            "family": "CLISA",
            "channels": 6,
            "input_dim": 30,
            "preferred_checkpoint": m20_6.name if m20_6 else None,
        },
        "M20_62ch": {
            "family": "CLISA",
            "channels": 62,
            "input_dim": 310,
            "preferred_checkpoint": m20_62.name if m20_62 else None,
        },
    },
}
(meta_dir / "model_registry.json").write_text(json.dumps(model_registry, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# manifest
# ------------------------------------------------------------
manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "bundle_root": str(bundle_root),
    "files": {}
}
for p in sorted(bundle_root.rglob("*")):
    if p.is_file():
        manifest["files"][str(p.relative_to(bundle_root))] = {
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p)
        }

(meta_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# readme
# ------------------------------------------------------------
readme = """CLISA backend asset bundle

Use these as the source of truth for backend playback/evaluation.

Primary files:
- data/features/seed_iv_X_6ch.npy
- data/features/seed_iv_X_62ch.npy
- data/features/seed_iv_y_4cls.npy
- data/features/seed_iv_subjects.npy
- config/channel_map_62_to_6.json
- metadata/preprocessing_contract.json
- metadata/model_registry.json

Important:
- Do not use webapp data/processed for notebook-faithful evaluation.
- Do not apply extra normalization on top of the original feature tensors.
"""
(bundle_root / "README.txt").write_text(readme, encoding="utf-8")

# ------------------------------------------------------------
# zip
# ------------------------------------------------------------
zip_path = project_root / "backend_assets_clisa.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in bundle_root.rglob("*"):
        zf.write(p, arcname=str(p.relative_to(project_root)))

print("=" * 90)
print("BUNDLE REPAIR COMPLETE")
print("=" * 90)
print("bundle_root:", bundle_root)
print("zip_path   :", zip_path)
print("M20_6ch    :", m20_6)
print("M20_62ch   :", m20_62)
print("X62 shape  :", X62.shape)
print("X6 shape   :", X6.shape)
print("y shape    :", y.shape)
print("subjects   :", sub.shape)
print("X6 matches subset(X62):", np.allclose(X6, X6_from_62, atol=1e-6, rtol=1e-6))


# ==============================================================================
# Notebook cell 25
# Categories: preprocessing, model_definition, training, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import json
import torch

NOTEBOOK_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")
WEBAPP_ROOT   = Path(r"D:\CSE400C\neurosync\neurosync")

# Choose one:
#   "simclr_best_single"  -> best single 6ch checkpoint overall (recommended final deployment)
#   "clisa_hotfix"        -> easiest low-risk hotfix using best 6ch CLISA fold
DEPLOY_CHOICE = "simclr_best_single"

DEPLOY_OPTIONS = {
    "simclr_best_single": {
        "model_id": "M21",
        "model_name": "SimCLR",
        "channel_mode": "6ch",
        "seed": 21,
        "fold": 7,
        "test_sub": 7,
        "acc_b": 0.7948,
        "f1_b": 0.7924,
        "src_weight_name": "M21_6ch_s21_f07_best.pth",
        "src_json_name": "M21_6ch_seed21_fold07.json",
        "dst_weight_name": "M21_6ch_best.pt",
        "notes": "Best single 6ch LOSO checkpoint from deep notebook."
    },
    "clisa_hotfix": {
        "model_id": "M20",
        "model_name": "CLISA",
        "channel_mode": "6ch",
        "seed": 1,
        "fold": 7,
        "test_sub": 7,
        "acc_b": 0.7863,
        "f1_b": 0.7852,
        "src_weight_name": "M20_6ch_s1_f07_best.pth",
        "src_json_name": "M20_6ch_seed1_fold07.json",
        "dst_weight_name": "M20_6ch_best.pt",
        "notes": "Best single 6ch CLISA fold. Use this first if you want a minimal-risk hotfix."
    }
}

cfg = DEPLOY_OPTIONS[DEPLOY_CHOICE]

src_weight = NOTEBOOK_ROOT / "checkpoints" / "model_weights" / cfg["src_weight_name"]
src_json   = NOTEBOOK_ROOT / "checkpoints" / "loso_results" / cfg["src_json_name"]

dst_asset_ckpt_dir = WEBAPP_ROOT / "assets" / "models" / "checkpoints"
dst_ref_dir        = WEBAPP_ROOT / "backend" / "app" / "references" / "loso_results"
dst_meta_dir       = WEBAPP_ROOT / "assets" / "metadata"

dst_asset_ckpt_dir.mkdir(parents=True, exist_ok=True)
dst_ref_dir.mkdir(parents=True, exist_ok=True)
dst_meta_dir.mkdir(parents=True, exist_ok=True)

if not src_weight.exists():
    raise FileNotFoundError(f"Missing source weight: {src_weight}")

if not src_json.exists():
    raise FileNotFoundError(f"Missing source json: {src_json}")

# Save runtime artifact as .pt
state = torch.load(src_weight, map_location="cpu")
dst_weight = dst_asset_ckpt_dir / cfg["dst_weight_name"]
torch.save(state, dst_weight)

# Also keep an exact-copy backup with original filename
backup_weight = dst_asset_ckpt_dir / cfg["src_weight_name"]
shutil.copy2(src_weight, backup_weight)

# Copy LOSO reference json
dst_json = dst_ref_dir / cfg["src_json_name"]
shutil.copy2(src_json, dst_json)

# Optional deploy manifest for your own tracking
manifest = {
    "deploy_choice": DEPLOY_CHOICE,
    "model_id": cfg["model_id"],
    "model_name": cfg["model_name"],
    "channel_mode": cfg["channel_mode"],
    "seed": cfg["seed"],
    "fold": cfg["fold"],
    "test_sub": cfg["test_sub"],
    "acc_b": cfg["acc_b"],
    "f1_b": cfg["f1_b"],
    "source_weight": str(src_weight),
    "deployed_weight": str(dst_weight),
    "reference_json": str(dst_json),
    "notes": cfg["notes"],
}

manifest_path = dst_meta_dir / "6ch_deploy_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Done.")
print(f"Deployed weight : {dst_weight}")
print(f"Backup weight   : {backup_weight}")
print(f"Reference json  : {dst_json}")
print(f"Manifest        : {manifest_path}")
print(json.dumps(manifest, indent=2))


# ==============================================================================
# Notebook cell 26
# Categories: preprocessing, model_definition, training, audit_verification, webapp_or_demo
# ==============================================================================
from pathlib import Path
import shutil
import json

WEBAPP_ROOT = Path(r"D:\CSE400C\neurosync\neurosync")

src_ckpt_pt  = WEBAPP_ROOT / "assets" / "models" / "checkpoints" / "M21_6ch_best.pt"
src_ckpt_pth = WEBAPP_ROOT / "assets" / "models" / "checkpoints" / "M21_6ch_s21_f07_best.pth"
src_json     = WEBAPP_ROOT / "backend" / "app" / "references" / "loso_results" / "M21_6ch_seed21_fold07.json"

dst_ckpt_dir = WEBAPP_ROOT / "backend_assets_clisa" / "models" / "checkpoints"
dst_ref_dir  = WEBAPP_ROOT / "backend_assets_clisa" / "references" / "loso_results"
dst_meta_dir = WEBAPP_ROOT / "backend_assets_clisa" / "metadata"

dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
dst_ref_dir.mkdir(parents=True, exist_ok=True)
dst_meta_dir.mkdir(parents=True, exist_ok=True)

if not src_ckpt_pt.exists():
    raise FileNotFoundError(f"Missing source checkpoint: {src_ckpt_pt}")

shutil.copy2(src_ckpt_pt,  dst_ckpt_dir / "M21_6ch_best.pt")

if src_ckpt_pth.exists():
    shutil.copy2(src_ckpt_pth, dst_ckpt_dir / "M21_6ch_s21_f07_best.pth")

if src_json.exists():
    shutil.copy2(src_json, dst_ref_dir / "M21_6ch_seed21_fold07.json")

manifest = {
    "model_id": "M21_6ch",
    "family": "SimCLR",
    "checkpoint": str(dst_ckpt_dir / "M21_6ch_best.pt"),
    "reference_json": str(dst_ref_dir / "M21_6ch_seed21_fold07.json"),
    "note": "Synced into backend_assets_clisa because docker-compose mounts this folder as /app/assets"
}

with open(dst_meta_dir / "m21_sync_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Done.")
print("Checkpoint copied to:", dst_ckpt_dir / "M21_6ch_best.pt")
print("PTH copied to      :", dst_ckpt_dir / "M21_6ch_s21_f07_best.pth")
print("JSON copied to     :", dst_ref_dir / "M21_6ch_seed21_fold07.json")
