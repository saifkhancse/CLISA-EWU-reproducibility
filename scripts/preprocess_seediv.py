"""
SEED-IV preprocessing entry point.

Raw SEED-IV files are not redistributed in this repository. Dataset holders can
connect official SEED-IV files to the released split definitions and channel map.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("SEED-IV preprocessing requires official dataset access.")
    print("See docs/dataset_access.md and data_splits/ for expected metadata.")

if __name__ == "__main__":
    main()
