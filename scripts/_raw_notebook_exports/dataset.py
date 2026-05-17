# Auto-exported raw code from notebook: dataset.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 0
# Categories: other
# ==============================================================================
import os, shutil
from pathlib import Path

# ── 1. Explore current structure ──────────────────────────────────────
root = Path(r"C:\Users\Saif\Desktop\CSE400\C")

print("=== Current root ===")
for p in sorted(root.rglob("*")):
    if p.is_dir():
        print(f"[DIR]  {p.relative_to(root)}")
    else:
        print(f"       {p.relative_to(root)}")


# ==============================================================================
# Notebook cell 1
# Categories: preprocessing, audit_verification
# ==============================================================================
import os, shutil
from pathlib import Path

root = Path(r"C:\Users\Saif\Desktop\CSE400\C")

# ── Target directories per plan ───────────────────────────────────────
seed_dst = root / "data" / "SEED_IV" / "ExtractedFeatures"
faced_dst = root / "data" / "FACED" / "EEG_Features" / "DE"

seed_dst.mkdir(parents=True, exist_ok=True)
faced_dst.mkdir(parents=True, exist_ok=True)

# ── Move SEED-IV (sessions 1, 2, 3) ──────────────────────────────────
seed_src = root / "SEED_IV_EEG_Features"
for session in ["1", "2", "3"]:
    src = seed_src / session
    dst = seed_dst / session
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✓ SEED-IV session {session} → {dst}")

# ── Move FACED DE files + fix double extension ────────────────────────
faced_src = root / "FACED_EEG_Features" / "DE"
for f in faced_src.glob("*.pkl.pkl"):
    new_name = f.name.replace(".pkl.pkl", ".pkl")
    shutil.copy2(f, faced_dst / new_name)
print(f"✓ FACED DE files → {faced_dst}")

# ── Verify ────────────────────────────────────────────────────────────
print("\n=== Final data/ structure ===")
for p in sorted((root / "data").rglob("*")):
    rel = p.relative_to(root)
    if p.is_dir():
        files = list(p.iterdir())
        print(f"[DIR]  {rel}  ({len(files)} items)")
    
print("\nSEED-IV .mat count:", len(list(seed_dst.rglob("*.mat"))))
print("FACED   .pkl count:", len(list(faced_dst.glob("*.pkl"))))
