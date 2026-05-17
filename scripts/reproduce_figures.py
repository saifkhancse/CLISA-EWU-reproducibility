"""
Figure reproduction entry point.

The repository includes exported figure files and result tables. This script
checks the released figure directories and reports available figures.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = [
    ROOT / "figures" / "paper",
    ROOT / "figures" / "final_archive",
]

def main():
    total = 0
    for folder in FIGURE_DIRS:
        files = sorted(folder.glob("*.png")) if folder.exists() else []
        total += len(files)
        print(f"{folder.relative_to(ROOT)}: {len(files)} PNG files")
    print(f"Total released PNG figures: {total}")

if __name__ == "__main__":
    main()
