from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_SPLITS_DIR = ROOT / "data_splits"
CONFIGS_DIR = ROOT / "configs"

DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    seediv_loso_folds = {
        "dataset": "SEED-IV",
        "protocol": "Leave-One-Subject-Out",
        "num_subjects": 15,
        "subject_indexing_note": "Subjects are represented as integer IDs 1-15. Each fold holds out one subject as test subject.",
        "folds": [
            {
                "fold": i,
                "test_subject": i,
                "train_subjects": [j for j in range(1, 16) if j != i]
            }
            for i in range(1, 16)
        ]
    }

    seeds = {
        "random_seeds": [1, 7, 21],
        "note": "These are the three random seeds used for the LOSO benchmark in the manuscript."
    }

    channel_indices = {
        "dataset": "SEED-IV",
        "full_channel_count": 62,
        "feature_bands": ["delta", "theta", "alpha", "beta", "gamma"],
        "six_channel_wearable_subset": {
            "channel_names": ["FP1", "FP2", "F7", "F8", "T7", "T8"],
            "zero_based_indices": [0, 2, 5, 13, 23, 31],
            "feature_dimension": 30,
            "note": "Six channels x five differential-entropy frequency bands = 30 features."
        },
        "full_feature_dimension": 310,
        "note": "Full setting uses 62 channels x five differential-entropy bands = 310 features."
    }

    label_map = {
        "dataset": "SEED-IV",
        "num_classes": 4,
        "label_encoding": {
            "0": "Neutral",
            "1": "Sad",
            "2": "Fear",
            "3": "Happy"
        },
        "note": "Confirm that this label encoding matches the final preprocessing pipeline before public release."
    }

    faced_contract = {
        "dataset": "FACED",
        "usage_in_manuscript": "Domain-gap analysis only unless full FACED benchmark is completed and verified.",
        "raw_channel_description": "FACED may be described as 32-channel in raw dataset documentation.",
        "processed_feature_representation_observed_in_project": {
            "file": "faced_X_32ch.npy",
            "shape": [110208, 30, 5],
            "dtype": "float32",
            "interpretation": "The processed feature representation used in this project has 30 channels and five frequency bands."
        },
        "manuscript_wording": "FACED is described as a 32-channel dataset in the raw dataset documentation, but the processed DE feature file available for this study contains 30 channels; therefore, all FACED domain-gap analyses use the 30-channel processed feature representation."
    }

    write_json(DATA_SPLITS_DIR / "seediv_loso_folds.json", seediv_loso_folds)
    write_json(DATA_SPLITS_DIR / "seeds_1_7_21.json", seeds)
    write_json(DATA_SPLITS_DIR / "channel_indices_seediv_62_to_6.json", channel_indices)
    write_json(DATA_SPLITS_DIR / "label_map_seediv_4class.json", label_map)
    write_json(CONFIGS_DIR / "faced_processing_contract.json", faced_contract)

    print("Data split/config files generated.")
    print(f"Output: {DATA_SPLITS_DIR}")
    print(f"Output: {CONFIGS_DIR}")

if __name__ == "__main__":
    main()
