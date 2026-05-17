# Auto-exported raw code from notebook: neurosync_data_setup.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 0
# Categories: model_definition, training, webapp_or_demo
# ==============================================================================
from pathlib import Path
import torch
import json

CKPT_DIR = Path("neurosync/models/checkpoints")  # change if needed

def pick_state(obj):
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "model_state"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
        return obj, "raw_dict"
    return obj, type(obj).__name__

def detect_family(keys):
    keys = list(keys)
    s = set(keys)
    joined = "\n".join(keys[:200])

    if any(k.startswith("net.") for k in keys):
        return "mlp_net_family"
    if any(k.startswith("blocks.") for k in keys) and any("pos_emb" == k for k in keys):
        return "transformer_blocks_family"
    if any(k.startswith("encoder.") for k in keys) or any(k.startswith("input_proj.") for k in keys):
        return "dance_student_family"
    return "unknown"

rows = []
for p in sorted(CKPT_DIR.glob("*.*")):
    if p.suffix.lower() not in [".pt", ".pth"]:
        continue
    try:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        state, state_src = pick_state(obj)
        keys = list(state.keys()) if isinstance(state, dict) else []
        fam = detect_family(keys)
        rows.append({
            "file": p.name,
            "state_source": state_src,
            "family_guess": fam,
            "num_keys": len(keys),
            "first_12_keys": keys[:12],
        })
    except Exception as e:
        rows.append({
            "file": p.name,
            "state_source": "ERROR",
            "family_guess": "ERROR",
            "num_keys": -1,
            "first_12_keys": [repr(e)],
        })

print(json.dumps(rows, indent=2))


# ==============================================================================
# Notebook cell 2
# Categories: other
# ==============================================================================
# ── CELL 1: Install dependencies ──────────────────────────────────────────────
!pip install scipy numpy --quiet
print('✓ Dependencies ready')


# ==============================================================================
# Notebook cell 3
# Categories: training
# ==============================================================================
import torch
from pathlib import Path

# Find the checkpoint automatically
search_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
found = []
for root in search_roots:
    found += list(root.rglob("*.pt")) + list(root.rglob("*.pth"))

print("Found checkpoints:")
for f in found:
    print(f" {f}")

# Load the first one found (change index if needed)
target = found[0]
print(f"\nLoading: {target}")

ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

print("\n=== TOP-LEVEL KEYS ===")
if isinstance(ckpt, dict):
    print(list(ckpt.keys()))

print("\n=== STATE DICT KEYS (first 30) ===")
for k in list(state.keys())[:30]:
    print(k)

print(f"\n=== TOTAL KEYS: {len(state.keys())} ===")


# ==============================================================================
# Notebook cell 4
# Categories: training, webapp_or_demo
# ==============================================================================
import torch
from pathlib import Path

# Target the specific M26 file in the webapp
target = Path(r"C:\Users\Saif\Desktop\CSE400\C\webapp\neurosync_webapp\neurosync\models\checkpoints\M26_6ch_best.pt")

ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

print("=== TOP-LEVEL KEYS ===")
print(list(ckpt.keys()) if isinstance(ckpt, dict) else "raw state dict")

print("\n=== ALL STATE DICT KEYS + SHAPES ===")
for k, v in state.items():
    shape = v.shape if hasattr(v, 'shape') else v
    print(f"  {k:50s} {shape}")

print(f"\n=== TOTAL KEYS: {len(state)} ===")


# ==============================================================================
# Notebook cell 5
# Categories: preprocessing, training, figures, audit_verification, webapp_or_demo
# ==============================================================================
# ── CELL 2: CONFIGURE YOUR PATHS — edit these two lines only ─────────────────

# Root folder of your SEED-IV ExtractedFeatures (contains subfolders 1, 2, 3)
SEED_IV_ROOT = r"C:\Users\Saif\Desktop\CSE400\C\data\SEED_IV\ExtractedFeatures"

# Root of your neurosync project (where docker-compose.yml lives)
NEUROSYNC_ROOT = r"C:\Users\Saif\Desktop\CSE400\C\webapp\neurosync_webapp\neurosync"

# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

SEED_IV_ROOT   = Path(SEED_IV_ROOT)
NEUROSYNC_ROOT = Path(NEUROSYNC_ROOT)

# Derived paths
PROCESSED_DIR    = NEUROSYNC_ROOT / 'data' / 'processed'
GROUND_TRUTH_DIR = NEUROSYNC_ROOT / 'data' / 'ground_truth'
NORM_DIR         = NEUROSYNC_ROOT / 'data' / 'norm_stats'
MANIFEST_DIR     = NEUROSYNC_ROOT / 'data' / 'manifests'
CHECKPOINTS_DIR  = NEUROSYNC_ROOT / 'models' / 'checkpoints'

# Create dirs if missing
for d in [PROCESSED_DIR, GROUND_TRUTH_DIR, NORM_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print('Paths configured:')
print(f'  SEED-IV root   : {SEED_IV_ROOT}')
print(f'  Processed out  : {PROCESSED_DIR}')
print(f'  Ground truth   : {GROUND_TRUTH_DIR}')
print(f'  Checkpoints    : {CHECKPOINTS_DIR}')
print()

# Check SEED-IV sessions exist
for ses in ['1','2','3']:
    p = SEED_IV_ROOT / ses
    mats = list(p.glob('*.mat')) if p.exists() else []
    print(f'  Session {ses}: {len(mats)} .mat files  {"✓" if mats else "✗ NOT FOUND"}')


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# ── CELL 3: Convert SEED-IV .mat → .npy ──────────────────────────────────────
import numpy as np
import scipy.io as sio

# SEED-IV trial labels per session (from the official README)
# 0=Neutral  1=Sad  2=Fear  3=Happy
SESSION_LABELS = {
    '1': [1,2,3,0, 2,0,0,1, 0,1,2,1, 1,1,2,3, 2,2,3,3, 0,3,0,3],
    '2': [2,1,3,0, 0,2,0,2, 3,3,2,3, 2,0,1,1, 2,1,0,3, 0,1,3,1],
    '3': [1,2,2,1, 3,3,3,1, 1,2,1,2, 2,3,2,0, 3,0,0,3, 0,1,0,1],
}

converted = []
errors    = []

for session_id in ['1', '2', '3']:
    mat_dir = SEED_IV_ROOT / session_id
    if not mat_dir.exists():
        print(f'⚠  Session {session_id} folder not found, skipping')
        continue

    mat_files = sorted(mat_dir.glob('*.mat'))
    labels    = SESSION_LABELS[session_id]
    ses_label = f'ses0{session_id}'

    print(f'\nSession {session_id} — {len(mat_files)} files')

    for mat_path in mat_files:
        try:
            data = sio.loadmat(str(mat_path))
        except Exception as e:
            print(f'  ✗ {mat_path.name}: load error — {e}')
            errors.append(mat_path.name)
            continue

        # Find DE feature keys — SEED-IV uses de_LDS1..de_LDS24
        trial_keys = sorted([k for k in data.keys() if k.startswith('de_LDS')])
        if not trial_keys:
            # Fallback: any key starting with de_
            trial_keys = sorted([k for k in data.keys()
                                  if k.startswith('de_') and not k.startswith('__')])

        if not trial_keys:
            all_keys = [k for k in data.keys() if not k.startswith('_')]
            print(f'  ✗ {mat_path.name}: no DE keys found. Available: {all_keys[:8]}')
            errors.append(mat_path.name)
            continue

        all_features = []
        all_labels   = []

        for trial_idx, key in enumerate(trial_keys):
            de = data[key]  # expected shape: (62, n_timewindows, 5)

            if de.ndim == 3:
                # Average over time windows → (62, 5), then flatten → (310,)
                feat = de.mean(axis=1).flatten().astype(np.float32)
            elif de.ndim == 2:
                feat = de.flatten().astype(np.float32)
            else:
                feat = de.flatten().astype(np.float32)

            label = labels[trial_idx] if trial_idx < len(labels) else 0
            all_features.append(feat)
            all_labels.append(label)

        features_arr = np.array(all_features, dtype=np.float32)  # (n_trials, 310)
        labels_arr   = np.array(all_labels,   dtype=np.int32)    # (n_trials,)

        # Build output filename: sub001_ses01.npy
        subject_num = mat_path.stem.split('_')[0].zfill(3)
        out_name    = f'sub{subject_num}_{ses_label}.npy'

        np.save(str(PROCESSED_DIR    / out_name), features_arr)
        np.save(str(GROUND_TRUTH_DIR / out_name), labels_arr)

        label_counts = {0:0, 1:0, 2:0, 3:0}
        for l in all_labels: label_counts[l] = label_counts.get(l,0) + 1
        print(f'  ✓ {mat_path.name:30s} → {out_name}  '
              f'shape={features_arr.shape}  '
              f'N={label_counts[0]} S={label_counts[1]} F={label_counts[2]} H={label_counts[3]}')
        converted.append(out_name)

print(f'\n━━━ Conversion complete: {len(converted)} files ✓   {len(errors)} errors ✗ ━━━')


# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, results_tables
# ==============================================================================
# ── CELL 4: Compute and save normalization stats ───────────────────────────────
# Loads all processed files and computes z-score mean/std across all trials

all_feats = []
for f in sorted(PROCESSED_DIR.glob('*.npy')):
    arr = np.load(str(f))
    all_feats.append(arr)

if all_feats:
    combined = np.concatenate(all_feats, axis=0)  # (N_total, 310)
    mean     = combined.mean(axis=0).astype(np.float32)
    std      = combined.std(axis=0).astype(np.float32)
    std      = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    np.savez(str(NORM_DIR / 'norm_62ch.npz'), mean=mean, std=std)
    print(f'✓ Norm stats saved: norm_62ch.npz')
    print(f'  Total trials : {combined.shape[0]}')
    print(f'  Feature dim  : {combined.shape[1]}')
    print(f'  Mean range   : [{mean.min():.4f}, {mean.max():.4f}]')
    print(f'  Std  range   : [{std.min():.4f},  {std.max():.4f}]')
else:
    print('⚠  No processed files found — run Cell 3 first')


# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, results_tables, webapp_or_demo
# ==============================================================================
# ── CELL 5: Write manifest.json ────────────────────────────────────────────────
import json

subject_ids = sorted(set(
    f.stem.split('_')[0].replace('sub','') for f in PROCESSED_DIR.glob('*.npy')
))
session_ids = sorted(set(
    '_'.join(f.stem.split('_')[1:]) for f in PROCESSED_DIR.glob('*.npy')
))

manifest = {
    "_comment":      "NeuroSync Dataset Manifest — SEED-IV 62-channel DE features",
    "dataset_name":  "SEED-IV",
    "subject_ids":   subject_ids,
    "session_ids":   session_ids,
    "n_channels":    62,
    "n_bands":       5,
    "input_dim":     310,
    "sampling_rate": 200,
    "channel_names": "standard 62-ch 10-20 system (SEED-IV order)",
    "band_names":    ["delta","theta","alpha","beta","gamma"],
    "label_schema":  {"0":"Neutral","1":"Sad","2":"Fear","3":"Happy"},
    "is_raw":        False,
    "preprocessing": "de_LDS_features_mean_over_timewindows",
    "normalization": "z_score_saved_in_norm_stats/norm_62ch.npz",
    "split_info": {
        "type":           "LOSO",
        "n_subjects":     len(subject_ids),
        "n_sessions":     len(session_ids),
        "trials_per_session": 24
    },
    "extra": {
        "window_type":    "LDS_smoothed",
        "feature_type":   "differential_entropy"
    }
}

with open(MANIFEST_DIR / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('✓ manifest.json written')
print(f'  Subjects  : {len(subject_ids)} → {subject_ids}')
print(f'  Sessions  : {session_ids}')


# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, training, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
# ── CELL 6: Verify everything ──────────────────────────────────────────────────

print('=' * 60)
print('NEUROSYNC DATA & MODEL VERIFICATION REPORT')
print('=' * 60)

# Processed features
proc_files = sorted(PROCESSED_DIR.glob('*.npy'))
print(f'\n📁 Processed features: {len(proc_files)} files')
for f in proc_files[:5]:
    arr = np.load(str(f))
    print(f'   {f.name:30s} shape={arr.shape}  dtype={arr.dtype}')
if len(proc_files) > 5:
    print(f'   ... and {len(proc_files)-5} more')

# Ground truth labels
gt_files = sorted(GROUND_TRUTH_DIR.glob('*.npy'))
print(f'\n📁 Ground truth labels: {len(gt_files)} files')
for f in gt_files[:3]:
    arr = np.load(str(f))
    unique, counts = np.unique(arr, return_counts=True)
    label_names = {0:'Neutral',1:'Sad',2:'Fear',3:'Happy'}
    dist = '  '.join(f'{label_names[int(u)]}={c}' for u,c in zip(unique,counts))
    print(f'   {f.name:30s} {dist}')

# Norm stats
norm_files = sorted(NORM_DIR.glob('*.npz'))
print(f'\n📁 Norm stats: {len(norm_files)} files')
for f in norm_files:
    d = np.load(str(f))
    print(f'   {f.name:30s} mean.shape={d["mean"].shape}  std.shape={d["std"].shape}')

# Manifest
mf = MANIFEST_DIR / 'manifest.json'
print(f'\n📄 Manifest: {"✓ exists" if mf.exists() else "✗ missing"}')

# Model checkpoints
pt_files = list(CHECKPOINTS_DIR.glob('*.pt')) + list(CHECKPOINTS_DIR.glob('*.pth'))
print(f'\n🧠 Model checkpoints: {len(pt_files)} files')
for f in sorted(pt_files):
    size_mb = f.stat().st_size / 1_048_576
    print(f'   {f.name:40s} {size_mb:.1f} MB')

# Final summary
print()
print('=' * 60)
ok = len(proc_files) > 0 and len(gt_files) > 0 and len(pt_files) > 0
print(f'STATUS: {"✓ READY — start docker-compose up --build" if ok else "⚠ INCOMPLETE — check above"}')
print('=' * 60)

if ok:
    print()
    print('In the app UI:')
    print('  1. Go to Device → Load & Verify → select M11_62ch_best.pt or M25_62ch_best.pt')
    print('  2. Go to Device → Dataset → pick any sub*_ses*.npy file → click Validate')
    print('  3. Go to Device → connect → start a Live Session')


# ==============================================================================
# Notebook cell 10
# Categories: other
# ==============================================================================



# ==============================================================================
# Notebook cell 11
# Categories: other
# ==============================================================================

