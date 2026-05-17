"""
Dataset-dependent LOSO evaluation entry point.

This repository provides verified final tables and table-reproduction scripts.
Full leave-one-subject-out evaluation requires official SEED-IV access and
locally generated features, which are not redistributed here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("LOSO evaluation requires official dataset access and local feature generation.")
    print("Use scripts/verify_results.py and scripts/reproduce_tables.py to reproduce released tables.")

if __name__ == "__main__":
    main()
