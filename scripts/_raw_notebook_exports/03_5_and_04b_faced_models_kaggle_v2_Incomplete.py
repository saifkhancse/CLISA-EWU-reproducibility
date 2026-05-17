# Auto-exported raw code from notebook: 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 2
# Categories: preprocessing, training
# ==============================================================================
# ╔════════════════════════════════════════════════════════════════╗
# ║  TEST MODE — set True to smoke-test the full notebook fast ║
# ║  Set False for a real production run                       ║
# ╚════════════════════════════════════════════════════════════════╝
TEST_RUN = False   # ← change to False for full run

if TEST_RUN:
    _N_FOLDS    = 3          # only 3 subjects instead of all
    _SEEDS      = [1]        # 1 seed only
    _EPOCHS_DL  = 1          # 1 epoch for all DL models
    _PRE_EP     = 1          # DANCE pretrain epochs
    _FT_EP      = 1          # DANCE finetune epochs
    _DIST_EP    = 1          # DANCE distil epochs
    _RF_TREES   = 20         # fast RF/ET/GB for classical ML test
    _XGB_EST    = 20
    print("⚠ TEST_RUN=True  — minimal epochs/folds for smoke-testing")
else:
    _N_FOLDS    = None       # use all subjects (set after loading data)
    _SEEDS      = [1, 7, 21]
    _EPOCHS_DL  = 50
    _PRE_EP     = 100
    _FT_EP      = 100
    _DIST_EP    = 50
    _RF_TREES   = 50
    _XGB_EST    = 100
    print("✅ TEST_RUN=False — full production run")



# ==============================================================================
# Notebook cell 4
# Categories: model_definition, training, evaluation, results_tables, figures, webapp_or_demo
# ==============================================================================
import subprocess, sys
_r = subprocess.run([sys.executable,'-m','pip','install','-q',
                     'torch==2.3.0+cu121','torchvision==0.18.0+cu121',
                     '--index-url','https://download.pytorch.org/whl/cu121'],
                    capture_output=True, text=True)
print('torch reinstalled:', _r.returncode == 0)
print(_r.stderr[-300:] if _r.returncode != 0 else 'OK')

import os, sys, json, time, copy, warnings, shutil, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ── GPU check ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', 'GPU NOT FOUND — check CUDA installation'
print(f'GPU : {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')

# Flash Attention / TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, training, figures
# ==============================================================================
IS_KAGGLE = os.path.exists('/kaggle')
if IS_KAGGLE:
    BASE     = Path('/kaggle/working')
    FEAT_SRC = Path('/kaggle/input/datasets/stone369/features')  # ← your dataset path
    CKPT_SRC = Path('/kaggle/input/datasets/saifkhancse/m12-deepmlp-s1-f28123/checkpoints')           # ← checkpoints dataset
else:
    BASE     = Path('.')
    FEAT_SRC = BASE / 'features'
    CKPT_SRC = BASE / 'checkpoints'

# Output dirs — mirror plan Section 8 folder structure exactly
FEAT_OUT    = BASE / 'features'  / 'preprocessed'
RESULTS_DIR = BASE / 'results'   / 'deep_models_faced'
CKPT_DIR    = BASE / 'checkpoints' / 'loso_results'
MODEL_DIR   = BASE / 'checkpoints'
FIG_DIR     = BASE / 'figures'   / 'models'

for d in [FEAT_OUT, RESULTS_DIR, CKPT_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Restore previous session checkpoints if uploaded ─────────────────────────
if IS_KAGGLE and CKPT_SRC.exists():
    for zf_path in glob.glob(str(CKPT_SRC / '*.zip')):
        print(f'Extracting: {zf_path}')
        with zipfile.ZipFile(zf_path, 'r') as z:
            z.extractall(BASE)
    for jf in CKPT_SRC.rglob('*.json'):
        dest = CKPT_DIR / jf.name
        if not dest.exists(): shutil.copy(jf, dest)
    for pf in CKPT_SRC.rglob('*.pth'):
        dest = MODEL_DIR / pf.name
        if not dest.exists(): shutil.copy(pf, dest)
    # Restore prep flags
    for ff in CKPT_SRC.rglob('*.flag'):
        dest = FEAT_OUT / ff.name
        if not dest.exists(): shutil.copy(ff, dest)
    # Restore prep .npy files
    for nf in CKPT_SRC.rglob('*.npy'):
        dest = FEAT_OUT / nf.name
        if not dest.exists(): shutil.copy(nf, dest)
    n_json = len(list(CKPT_DIR.glob('*.json')))
    n_pth  = len(list(MODEL_DIR.glob('*.pth')))
    print(f'Restored: {n_json} fold checkpoints, {n_pth} model weights')

print(f'BASE     : {BASE}')
print(f'FEAT_SRC : {FEAT_SRC}')
print(f'FEAT_OUT : {FEAT_OUT}')



# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, model_definition, training
# ==============================================================================
# ── Fold-level checkpoint helpers ────────────────────────────────────────────
def ckpt_key(model_id, seed, fold):
    return f'{model_id}_seed{seed}_fold{fold:02d}'

def save_ckpt(model_id, seed, fold, result_dict):
    path = CKPT_DIR / f'{ckpt_key(model_id,seed,fold)}.json'
    with open(path, 'w') as f: json.dump(result_dict, f)

def load_ckpt(model_id, seed, fold):
    path = CKPT_DIR / f'{ckpt_key(model_id,seed,fold)}.json'
    return json.load(open(path)) if path.exists() else None

def ckpt_exists(model_id, seed, fold):
    return (CKPT_DIR / f'{ckpt_key(model_id,seed,fold)}.json').exists()

def model_done_count(model_id, n_folds, seeds):
    return sum(1 for s in seeds for f in range(1, n_folds+1)
               if ckpt_exists(model_id, s, f))

# ── Preprocessing flag helpers ────────────────────────────────────────────────
def prep_done(name):   return (FEAT_OUT / f'_prep_{name}.flag').exists()
def mark_prep_done(name): (FEAT_OUT / f'_prep_{name}.flag').touch()

def save_prep(name, arr, readme=None):
    path = FEAT_OUT / f'{name}.npy'
    if path.exists():
        print(f'  SKIP {name} (exists)')
        mark_prep_done(name); return
    np.save(path, arr)
    if readme:
        (FEAT_OUT / f'{name}_readme.json').write_text(json.dumps(readme, indent=2))
    mark_prep_done(name)
    print(f'  SAVED {name}  shape={arr.shape}  dtype={arr.dtype}')

# ── Session status ─────────────────────────────────────────────────────────────
def session_status():
    # n_folds and seeds may not be set yet — use globals if available
    nf = globals().get('N_FOLDS_FACED', '?')
    sd = globals().get('SEEDS_FACED', ['?'])
    total = (nf * len(sd)) if isinstance(nf, int) else '?'
    print(f'\n{"="*62}')
    print(f'SESSION STATUS  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Folds={nf}  Seeds={sd}  Total per model={total}')
    prep_names = [
        'seed_iv_prep_s1','seed_iv_prep_s3','seed_iv_prep_s4','seed_iv_prep_s5','seed_iv_prep_s6',
        'seed_iv_aug_s1','seed_iv_aug_s2','seed_iv_aug_s3','seed_iv_aug_s4',
        'seed_iv_aug_s5','seed_iv_aug_s6','seed_iv_aug_s7',
        'faced_prep_f1','faced_prep_f3','faced_prep_f4','faced_prep_f5','faced_prep_f6',
        'faced_aug_f1','faced_aug_f2','faced_aug_f3','faced_aug_f4',
        'faced_aug_f5','faced_aug_f6','faced_aug_f7','faced_aug_f8',
    ]
    print('\nPreprocessing:')
    for v in prep_names:
        print(f'  {"DONE" if prep_done(v) else "    "} {v}')
    if isinstance(nf, int):
        model_ids = [
            'M01_LDA_faced_30ch','M02_SVM_faced_30ch','M03_RF_faced_30ch',
            'M04_KNN_faced_30ch','M05_LR_faced_30ch','M06_NB_faced_30ch',
            'M07_ET_faced_30ch','M08_GB_faced_30ch','M09_XGB_faced_30ch','M10_MLP_SK_faced_30ch',
            'M01_LDA_faced_6ch','M02_SVM_faced_6ch','M03_RF_faced_6ch',
            'M11_ShallowMLP','M12_DeepMLP','M13_LSTM','M14_GRU',
            'M15_Conv1D','M16_Transformer','M17_Conformer','M18_ChanDrop',
            'M19_DANN','M20_CLISA','M21_SimCLR','M22_BYOL',
            'M23_PseudoLabel','M24_MixMatch','M25_DANCE_Teacher','M26_DANCE_Student',
        ]
        print('\nModel training:')
        for mid in model_ids:
            n = model_done_count(mid, nf, sd)
            pct = int(20*n/total) if total else 0
            bar = '#'*pct + '.'*(20-pct)
            print(f'  [{bar}] {n:4d}/{total}  {mid}')
    print(f'{"="*62}\n')

print('Checkpoint helpers ready.')
session_status()



# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, audit_verification
# ==============================================================================
# ── Channel config — SEED-IV ──────────────────────────────────────────────────
# Corrected 6-ch indices (FP1,FP2,F7,F8,T7,T8) — plan §16 G3 fix
CH6_SEED = [0, 2, 5, 13, 23, 31]

def _load(name):
    for p in [FEAT_OUT / name, FEAT_SRC / name, BASE / 'features' / name]:
        if p.exists():
            return np.load(p, allow_pickle=True)
    raise FileNotFoundError(f'{name} not found in {FEAT_SRC}')

# ── SEED-IV ───────────────────────────────────────────────────────────────────
print('Loading SEED-IV...')
X_s    = _load('seed_iv_X_62ch.npy')    # (37575, 310)
y_s    = _load('seed_iv_y_4cls.npy')
subj_s = _load('seed_iv_subjects.npy')
sess_s = _load('seed_iv_session.npy')
if X_s.ndim == 3: X_s = X_s.reshape(X_s.shape[0], -1)
X_s6   = X_s.reshape(-1, 62, 5)[:, CH6_SEED, :].reshape(-1, 30)
print(f'  X_s={X_s.shape}  y_s={y_s.shape}  subj_s={subj_s.shape}')
assert len(X_s)==len(y_s)==len(subj_s), 'SEED-IV length mismatch!'

# ── FACED ─────────────────────────────────────────────────────────────────────
print('\nLoading FACED...')
X_f    = _load('faced_X_32ch.npy')
y_f    = _load('faced_y_4cls.npy')
subj_f = _load('faced_subjects.npy')

# Flatten if 3D (N, ch, bands) → (N, ch*bands)
if X_f.ndim == 3: X_f = X_f.reshape(X_f.shape[0], -1)
if y_f.ndim > 1:  y_f = y_f.flatten()
if subj_f.ndim > 1: subj_f = subj_f.flatten()

# ── Auto-detect actual channel count ─────────────────────────────────────────
N_BANDS      = 5
N_CH_FACED   = X_f.shape[1] // N_BANDS   # 150//5 = 30  (actual data)
IN_DIM_FACED = N_CH_FACED * N_BANDS       # 150

print(f'  X_f raw shape  : {X_f.shape}  → N_CH_FACED={N_CH_FACED}  IN_DIM={IN_DIM_FACED}')
print(f'  y_f shape      : {y_f.shape}')
print(f'  subj_f shape   : {subj_f.shape}')

# ── Align lengths (truncate to shortest) ──────────────────────────────────────
n_min = min(len(X_f), len(y_f), len(subj_f))
if not (len(X_f) == len(y_f) == len(subj_f)):
    print(f'  ⚠ Length mismatch — truncating to {n_min}')
    X_f    = X_f[:n_min]
    y_f    = y_f[:n_min]
    subj_f = subj_f[:n_min]

assert len(X_f) == len(y_f) == len(subj_f), 'Still mismatched after truncation!'
print(f'  Aligned: {len(X_f)} samples')

# ── FACED 6-ch indices (FP1,FP2,F7,F8,T7,T8 in 30-ch layout) ─────────────────
# 30-ch layout: FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,
#               FZ,CZ,PZ,OZ,FC1,FC2,CP1,CP2,FC5,FC6,CP5,CP6,POZ,FCZ
FACED_CH_NAMES = [
    'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FZ','CZ','PZ','OZ',
    'FC1','FC2','CP1','CP2','FC5','FC6','CP5','CP6','POZ','FCZ'
]
# Trim to actual channel count (handles both 30 and other sizes gracefully)
FACED_CH_NAMES = FACED_CH_NAMES[:N_CH_FACED]
CH6_FACED_NAMES = ['FP1','FP2','F7','F8','T7','T8']
CH6_FACED = [FACED_CH_NAMES.index(c) for c in CH6_FACED_NAMES
             if c in FACED_CH_NAMES]
print(f'  CH6_FACED indices: {CH6_FACED}  names: {[FACED_CH_NAMES[i] for i in CH6_FACED]}')

X_f6 = X_f.reshape(-1, N_CH_FACED, N_BANDS)[:, CH6_FACED, :].reshape(-1, 30)

N_FACED_SUBJECTS = len(np.unique(subj_f))
print(f'  Subjects: {N_FACED_SUBJECTS}')
print(f'  X_f6 shape: {X_f6.shape}')
print(f'  Class counts: { {int(c): int((y_f==c).sum()) for c in range(4)} }')

# ── Brain region indices for 30-ch FACED augmentation ─────────────────────────
# Indices based on FACED_CH_NAMES list above
REGIONS_SEED = {
    'frontal' : [0,1,2,3,4,5,6,7,8,9,10,11],
    'temporal': [23,24,25,26,29,30],
    'parietal': [31,32,33,34,35,36,37,38],
    'occipital': [50,51,52,53,54,55,56,57],
}
REGIONS_FACED = {
    'frontal' : [i for i,n in enumerate(FACED_CH_NAMES) if n in ['FP1','FP2','F3','F4','F7','F8','FZ','FC1','FC2','FC5','FC6']],
    'temporal': [i for i,n in enumerate(FACED_CH_NAMES) if n in ['T7','T8','CP5','CP6']],
    'parietal': [i for i,n in enumerate(FACED_CH_NAMES) if n in ['P3','P4','P7','P8','PZ','CP1','CP2','POZ']],
    'occipital':[i for i,n in enumerate(FACED_CH_NAMES) if n in ['O1','O2','OZ']],
}
print(f'  REGIONS_FACED: { {k:v for k,v in REGIONS_FACED.items()} }')



# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, audit_verification
# ==============================================================================
# ── Set folds/seeds from TEST_RUN config ─────────────────────────────────────
FACED_SUBJECTS = np.sort(np.unique(subj_f))
N_FOLDS_FACED  = _N_FOLDS if TEST_RUN else len(FACED_SUBJECTS)
SEEDS_FACED    = _SEEDS
TOTAL_RUNS     = N_FOLDS_FACED * len(SEEDS_FACED)

print(f'LOSO config: {N_FOLDS_FACED} folds × {len(SEEDS_FACED)} seeds = {TOTAL_RUNS} runs/model')
print(f'TEST_RUN={TEST_RUN}')

def get_faced_fold(fold_idx, X, y, subjects, seed=1):
    """LOSO fold: leave subject[fold_idx] out as test; 10% of rest as val."""
    rng        = np.random.default_rng(seed)
    test_subj  = FACED_SUBJECTS[fold_idx]
    other      = FACED_SUBJECTS[FACED_SUBJECTS != test_subj]
    n_val      = max(1, len(other) // 10)
    val_subj   = rng.choice(other, n_val, replace=False)
    train_subj = other[~np.isin(other, val_subj)]
    tr  = np.isin(subjects, train_subj)
    val = np.isin(subjects, val_subj)
    te  = subjects == test_subj
    return (X[tr], y[tr], subjects[tr],
            X[val], y[val],
            X[te],  y[te])

# Sanity check
Xtr,ytr,sbl,Xvl,yvl,Xte,yte = get_faced_fold(0, X_f, y_f, subj_f, seed=1)
print(f'Fold 0 check: train={len(Xtr)} val={len(Xvl)} test={len(Xte)} (test_subj={FACED_SUBJECTS[0]})')
assert len(Xtr)>0 and len(Xte)>0, 'Fold split failed!'
print('Fold split OK.')
session_status()



# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
def prep_zscore_subject(X, subjects):
    X_out = X.copy().astype(np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        mu = X_out[m].mean(0, keepdims=True)
        sg = X_out[m].std(0,  keepdims=True) + 1e-8
        X_out[m] = (X_out[m] - mu) / sg
    return X_out

def prep_minmax_subject(X, subjects):
    X_out = X.copy().astype(np.float32)
    for s in np.unique(subjects):
        m  = subjects == s
        mn = X_out[m].min(0, keepdims=True)
        mx = X_out[m].max(0, keepdims=True)
        X_out[m] = (X_out[m] - mn) / (mx - mn + 1e-8)
    return X_out

def prep_robust_subject(X, subjects):
    """Robust scaler: median + IQR. Works on 2-D (N, features) arrays."""
    X_out = X.copy().astype(np.float32)
    for s in np.unique(subjects):
        m    = subjects == s
        med  = np.median(X_out[m], axis=0, keepdims=True)   # (1, F)
        q75, q25 = np.percentile(X_out[m], [75, 25], axis=0)  # (F,)
        iqr  = (q75 - q25)[np.newaxis, :] + 1e-8              # (1, F)
        X_out[m] = (X_out[m] - med) / iqr
    return X_out

def prep_zscore_clip_subject(X, subjects, clip=3.0):
    return np.clip(prep_zscore_subject(X, subjects), -clip, clip)

def prep_bandwise_zscore_subject(X, subjects, n_ch, n_bands=5):
    """Z-score independently per band. X must be 2-D (N, n_ch*n_bands)."""
    assert X.ndim == 2, f'Expected 2-D, got {X.shape}'
    X3  = X.copy().astype(np.float32).reshape(-1, n_ch, n_bands)
    sub = subjects[:len(X3)]         # guard against any length diff
    for s in np.unique(sub):
        m = sub == s
        for b in range(n_bands):
            mu = X3[m, :, b].mean(0, keepdims=True)
            sg = X3[m, :, b].std(0,  keepdims=True) + 1e-8
            X3[m, :, b] = (X3[m, :, b] - mu) / sg
    return X3.reshape(-1, n_ch * n_bands)

print('Preprocessing functions defined (all operate on 2-D arrays).')



# ==============================================================================
# Notebook cell 16
# Categories: preprocessing, results_tables
# ==============================================================================
AUG_SEED = 42

def aug_gaussian_noise(X, noise_std=0.1, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    return (X + rng.normal(0, noise_std, X.shape)).astype(np.float32)

def aug_band_masking(X, n_ch, n_bands=5, p=0.3, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    out = X.copy().reshape(-1, n_ch, n_bands)
    for b in range(n_bands):
        mask = rng.random(len(out)) < p
        out[mask, :, b] = 0.0
    return out.reshape(-1, n_ch * n_bands).astype(np.float32)

def aug_channel_masking(X, n_ch, n_bands=5, mask_ratio=0.3, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    out    = X.copy().reshape(-1, n_ch, n_bands)
    n_mask = max(1, int(n_ch * mask_ratio))
    for i in range(len(out)):
        ch = rng.choice(n_ch, n_mask, replace=False)
        out[i, ch, :] = 0.0
    return out.reshape(-1, n_ch * n_bands).astype(np.float32)

def aug_region_masking(X, regions, n_ch, n_bands=5, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    rlist  = list(regions.values())
    out    = X.copy().reshape(-1, n_ch, n_bands)
    for i in range(len(out)):
        reg   = rlist[rng.integers(len(rlist))]
        valid = [c for c in reg if c < n_ch]
        if valid: out[i, valid, :] = 0.0
    return out.reshape(-1, n_ch * n_bands).astype(np.float32)

def aug_subject_mixup(X, y, subjects, alpha=0.4, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    out = X.copy().astype(np.float32)
    for i in range(len(X)):
        same = np.where((y == y[i]) & (subjects != subjects[i]))[0]
        if len(same) == 0: continue
        j   = rng.choice(same)
        lam = float(rng.beta(alpha, alpha))
        out[i] = lam * X[i] + (1 - lam) * X[j]
    return out

def aug_magnitude_warp(X, lo=0.8, hi=1.2, rng=None):
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    return (X * rng.uniform(lo, hi, X.shape)).astype(np.float32)

def aug_combined_seed(X, y, subjects, rng=None):
    """AUG-S7: band_mask + region_mask + mixup (plan §3.6.1)."""
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    out = aug_band_masking(X, 62, rng=rng)
    out = aug_region_masking(out, REGIONS_SEED, 62, rng=rng)
    out = aug_subject_mixup(out, y, subjects, rng=rng)
    return out

def aug_combined_faced(X, y, subjects, rng=None):
    """AUG-F8: band_mask + region_mask + mixup for FACED (plan §3.6.2)."""
    if rng is None: rng = np.random.default_rng(AUG_SEED)
    out = aug_band_masking(X, N_CH_FACED, rng=rng)
    out = aug_region_masking(out, REGIONS_FACED, N_CH_FACED, rng=rng)
    out = aug_subject_mixup(out, y, subjects, rng=rng)
    return out

print('Augmentation functions defined.')



# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, audit_verification
# ==============================================================================
# ── SEED-IV variants ──────────────────────────────────────────────────────────
print('=== SEED-IV Preprocessing ===')
if not prep_done('seed_iv_prep_s1'):
    Xs1 = prep_zscore_subject(X_s, subj_s)
    save_prep('seed_iv_prep_s1_zscore_subject', Xs1,
              {'variant':'PREP-S1','n_ch':62})
    save_prep('seed_iv_prep_s1_zscore_subject_6ch',
              Xs1.reshape(-1,62,5)[:,CH6_SEED,:].reshape(-1,30), {})
    mark_prep_done('seed_iv_prep_s1')
else: print('  SKIP PREP-S1')

if not prep_done('seed_iv_prep_s3'):
    Xs3 = prep_minmax_subject(X_s, subj_s)
    save_prep('seed_iv_prep_s3_minmax_subject', Xs3, {'variant':'PREP-S3'})
    save_prep('seed_iv_prep_s3_minmax_subject_6ch',
              Xs3.reshape(-1,62,5)[:,CH6_SEED,:].reshape(-1,30), {})
    mark_prep_done('seed_iv_prep_s3')
else: print('  SKIP PREP-S3')

if not prep_done('seed_iv_prep_s4'):
    Xs4 = prep_robust_subject(X_s, subj_s)
    save_prep('seed_iv_prep_s4_robust_subject', Xs4, {'variant':'PREP-S4'})
    save_prep('seed_iv_prep_s4_robust_subject_6ch',
              Xs4.reshape(-1,62,5)[:,CH6_SEED,:].reshape(-1,30), {})
    mark_prep_done('seed_iv_prep_s4')
else: print('  SKIP PREP-S4')

if not prep_done('seed_iv_prep_s5'):
    Xs5 = prep_zscore_clip_subject(X_s, subj_s)
    save_prep('seed_iv_prep_s5_zscore_clip3', Xs5, {'variant':'PREP-S5'})
    save_prep('seed_iv_prep_s5_zscore_clip3_6ch',
              Xs5.reshape(-1,62,5)[:,CH6_SEED,:].reshape(-1,30), {})
    mark_prep_done('seed_iv_prep_s5')
else: print('  SKIP PREP-S5')

if not prep_done('seed_iv_prep_s6'):
    save_prep('seed_iv_prep_s6_bandwise_zscore',
              prep_bandwise_zscore_subject(X_s, subj_s, n_ch=62),
              {'variant':'PREP-S6'})
    mark_prep_done('seed_iv_prep_s6')
else: print('  SKIP PREP-S6')

# SEED-IV augmentations
print('\n=== SEED-IV Augmentations ===')
p_s1 = FEAT_OUT / 'seed_iv_prep_s1_zscore_subject.npy'
Xbs  = np.load(p_s1) if p_s1.exists() else prep_zscore_subject(X_s, subj_s)
rng_s = np.random.default_rng(AUG_SEED)
S_AUGS = [
    ('seed_iv_aug_s1', 'seed_iv_aug_s1_noise',
     lambda: aug_gaussian_noise(Xbs, rng=rng_s), 'AUG-S1'),
    ('seed_iv_aug_s2', 'seed_iv_aug_s2_bandmask',
     lambda: aug_band_masking(Xbs, 62, rng=rng_s), 'AUG-S2'),
    ('seed_iv_aug_s3', 'seed_iv_aug_s3_chanmask',
     lambda: aug_channel_masking(Xbs, 62, rng=rng_s), 'AUG-S3'),
    ('seed_iv_aug_s4', 'seed_iv_aug_s4_regionmask',
     lambda: aug_region_masking(Xbs, REGIONS_SEED, 62, rng=rng_s), 'AUG-S4'),
    ('seed_iv_aug_s5', 'seed_iv_aug_s5_subject_mixup',
     lambda: aug_subject_mixup(Xbs, y_s, subj_s, rng=rng_s), 'AUG-S5'),
    ('seed_iv_aug_s6', 'seed_iv_aug_s6_magwarp',
     lambda: aug_magnitude_warp(Xbs, rng=rng_s), 'AUG-S6'),
    ('seed_iv_aug_s7', 'seed_iv_aug_s7_combined',
     lambda: aug_combined_seed(Xbs, y_s, subj_s, rng=rng_s), 'AUG-S7'),
]
for flag, fname, fn, desc in S_AUGS:
    if not prep_done(flag):
        save_prep(fname, fn(), {'variant': desc})
        mark_prep_done(flag)
    else: print(f'  SKIP {flag}')

# ── FACED variants ────────────────────────────────────────────────────────────
print('\n=== FACED Preprocessing ===')
if not prep_done('faced_prep_f1'):
    Xf1 = prep_zscore_subject(X_f, subj_f)
    save_prep('faced_prep_f1_zscore_subject', Xf1,
              {'variant':'PREP-F1','n_ch':N_CH_FACED,'note':'baseline E02'})
    save_prep('faced_prep_f1_zscore_subject_6ch',
              Xf1.reshape(-1,N_CH_FACED,5)[:,CH6_FACED,:].reshape(-1,30),
              {'variant':'PREP-F1-6ch'})
    mark_prep_done('faced_prep_f1')
else: print('  SKIP PREP-F1')

if not prep_done('faced_prep_f3'):
    save_prep('faced_prep_f3_minmax_subject',
              prep_minmax_subject(X_f, subj_f), {'variant':'PREP-F3'})
    mark_prep_done('faced_prep_f3')
else: print('  SKIP PREP-F3')

if not prep_done('faced_prep_f4'):
    save_prep('faced_prep_f4_robust_subject',
              prep_robust_subject(X_f, subj_f), {'variant':'PREP-F4'})
    mark_prep_done('faced_prep_f4')
else: print('  SKIP PREP-F4')

if not prep_done('faced_prep_f5'):
    save_prep('faced_prep_f5_zscore_clip3',
              prep_zscore_clip_subject(X_f, subj_f), {'variant':'PREP-F5'})
    mark_prep_done('faced_prep_f5')
else: print('  SKIP PREP-F5')

if not prep_done('faced_prep_f6'):
    save_prep('faced_prep_f6_bandwise_zscore',
              prep_bandwise_zscore_subject(X_f, subj_f, n_ch=N_CH_FACED),
              {'variant':'PREP-F6'})
    mark_prep_done('faced_prep_f6')
else: print('  SKIP PREP-F6')

print('\n=== FACED Augmentations ===')
p_f1 = FEAT_OUT / 'faced_prep_f1_zscore_subject.npy'
Xbf  = np.load(p_f1) if p_f1.exists() else prep_zscore_subject(X_f, subj_f)
rng_f = np.random.default_rng(AUG_SEED)
F_AUGS = [
    ('faced_aug_f1','faced_aug_f1_noise',
     lambda: aug_gaussian_noise(Xbf, rng=rng_f),'AUG-F1'),
    ('faced_aug_f2','faced_aug_f2_bandmask',
     lambda: aug_band_masking(Xbf, N_CH_FACED, rng=rng_f),'AUG-F2'),
    ('faced_aug_f3','faced_aug_f3_chanmask',
     lambda: aug_channel_masking(Xbf, N_CH_FACED, rng=rng_f),'AUG-F3'),
    ('faced_aug_f4','faced_aug_f4_subject_mixup',
     lambda: aug_subject_mixup(Xbf, y_f, subj_f, rng=rng_f),'AUG-F4'),
    ('faced_aug_f5','faced_aug_f5_combined',
     lambda: aug_band_masking(aug_subject_mixup(Xbf,y_f,subj_f,rng=rng_f),
                              N_CH_FACED,rng=rng_f),'AUG-F5'),
    ('faced_aug_f6','faced_aug_f6_regionmask',
     lambda: aug_region_masking(Xbf, REGIONS_FACED, N_CH_FACED, rng=rng_f),'AUG-F6'),
    ('faced_aug_f7','faced_aug_f7_magwarp',
     lambda: aug_magnitude_warp(Xbf, rng=rng_f),'AUG-F7'),
    ('faced_aug_f8','faced_aug_f8_combined_m27',
     lambda: aug_combined_faced(Xbf, y_f, subj_f, rng=rng_f),'AUG-F8'),
]
for flag, fname, fn, desc in F_AUGS:
    if not prep_done(flag):
        save_prep(fname, fn(), {'variant': desc})
        mark_prep_done(flag)
    else: print(f'  SKIP {flag}')

# ── Load X_f1_base for all subsequent training ────────────────────────────────
X_f1_base = np.load(FEAT_OUT / 'faced_prep_f1_zscore_subject.npy')
X_f1_6ch  = np.load(FEAT_OUT / 'faced_prep_f1_zscore_subject_6ch.npy')
print(f'\nX_f1_base={X_f1_base.shape}  X_f1_6ch={X_f1_6ch.shape}')
assert X_f1_base.shape[1] == IN_DIM_FACED,     f'Shape mismatch: expected {IN_DIM_FACED} but got {X_f1_base.shape[1]}'
print('\n✅ All preprocessing and augmentation variants complete.')



# ==============================================================================
# Notebook cell 20
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Max training samples per fold in TEST_RUN — prevents SVM/GB from hanging
MAX_TRAIN_TEST = 3000
MAX_TRAIN_FULL = 10000

CLASSICAL_MODELS = {
    'M01_LDA'    : LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'M02_SVM'    : SVC(C=10, kernel='rbf', gamma='scale', max_iter=2000),
    'M03_RF'     : RandomForestClassifier(n_estimators=_RF_TREES, n_jobs=-1, random_state=0),
    'M04_KNN'    : KNeighborsClassifier(n_neighbors=7, metric='cosine', n_jobs=-1),
    'M05_LR'     : LogisticRegression(solver='lbfgs', C=1.0, max_iter=500, n_jobs=-1),
    'M06_NB'     : GaussianNB(),
    'M07_ET'     : ExtraTreesClassifier(n_estimators=50, n_jobs=-1, random_state=0),
    # HistGradientBoosting is 10-100x faster than GradientBoosting on large datasets
    'M08_GB'     : HistGradientBoostingClassifier(max_iter=_RF_TREES, learning_rate=0.05,
                                                   max_depth=6, random_state=0),
    'M09_XGB'    : xgb.XGBClassifier(n_estimators=_XGB_EST, learning_rate=0.05, max_depth=6,
                                      eval_metric='mlogloss', tree_method='hist',
                                      n_jobs=-1, random_state=0, verbosity=0),
    'M10_MLP_SK' : MLPClassifier(hidden_layer_sizes=(256,128), learning_rate_init=1e-3,
                                  max_iter=200, random_state=0),
}

SLOW_MODELS = {'M02_SVM', 'M04_KNN'}   # models that don't scale to 100k+ samples

def run_classical_fold(model_id, clf_obj, fold_idx, seed, X_all, y_all, subj_all):
    if ckpt_exists(model_id, seed, fold_idx+1):
        return load_ckpt(model_id, seed, fold_idx+1)

    Xtr,ytr,_,Xvl,yvl,Xte,yte = get_faced_fold(fold_idx, X_all, y_all, subj_all, seed)
    Xtrv = np.vstack([Xtr, Xvl]); ytrv = np.hstack([ytr, yvl])

    # Subsample only slow models in full run, everything in test run
    base_id = model_id.split('_faced_')[0]  # e.g. 'M02_SVM'
    if TEST_RUN:
        max_tr = MAX_TRAIN_TEST
    elif base_id in SLOW_MODELS:
        max_tr = MAX_TRAIN_FULL   # cap SVM/KNN even in full run
    else:
        max_tr = None             # all other models use full training data

    if max_tr is not None and len(Xtrv) > max_tr:
        rng  = np.random.default_rng(seed + fold_idx)
        idx  = rng.choice(len(Xtrv), max_tr, replace=False)
        Xtrv = Xtrv[idx]; ytrv = ytrv[idx]

    clf  = copy.deepcopy(clf_obj)
    clf.fit(Xtrv, ytrv)
    pred = clf.predict(Xte)
    res  = {'model': model_id, 'seed': seed, 'fold': fold_idx+1,
            'acc': float(accuracy_score(yte, pred)),
            'f1':  float(f1_score(yte, pred, average='macro', zero_division=0)),
            'n_test': len(yte)}
    save_ckpt(model_id, seed, fold_idx+1, res)
    return res

for ch_tag, X_all in [('30ch', X_f1_base), ('6ch', X_f1_6ch)]:
    for mid_base, clf_obj in CLASSICAL_MODELS.items():
        mid   = f'{mid_base}_faced_{ch_tag}'
        total = N_FOLDS_FACED * len(SEEDS_FACED)
        done  = model_done_count(mid, N_FOLDS_FACED, SEEDS_FACED)
        if done == total:
            print(f'SKIP {mid}'); continue
        print(f'Running {mid} ({done}/{total})...', flush=True)
        t0 = time.time()
        for seed in SEEDS_FACED:
            for fi in range(N_FOLDS_FACED):
                run_classical_fold(mid, clf_obj, fi, seed, X_all, y_f, subj_f)
        rows = [load_ckpt(mid,s,f+1) for s in SEEDS_FACED for f in range(N_FOLDS_FACED)]
        accs = [r['acc'] for r in rows]; f1s = [r['f1'] for r in rows]
        print(f'  {mid}  Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'F1={np.mean(f1s):.4f}  [{time.time()-t0:.0f}s]')

print('\n✅ Classical ML complete.')


# ==============================================================================
# Notebook cell 22
# Categories: other
# ==============================================================================
# ADD this at the top of Cell 11 (right after imports, before class EEGDataset)
# Disable autocast on GPUs that don't support it (P100 = sm_60, dropped in PyTorch 2.x+cu128)
USE_AMP = torch.cuda.get_device_capability()[0] >= 7   # Volta+ only
print(f'AMP (fp16): {"ENABLED" if USE_AMP else "DISABLED (GPU < sm_70, using fp32)"}')

from contextlib import contextmanager
@contextmanager
def maybe_autocast():
    if USE_AMP:
        with autocast(): yield
    else:
        yield


# ==============================================================================
# Notebook cell 23
# Categories: model_definition, training, evaluation, results_tables
# ==============================================================================
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.y[i]

def make_loader(X, y, batch_size=128, shuffle=True, use_wrs=True):
    """DataLoader with WeightedRandomSampler for class balance (plan §5.3 H14)."""
    ds = EEGDataset(X, y)
    if shuffle and use_wrs:
        counts  = np.bincount(y)
        weights = 1.0 / counts[y]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            with autocast():
                out = model(Xb.to(device)); logits = out[0] if isinstance(out, tuple) else out
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    return (float(accuracy_score(trues, preds)),
            float(f1_score(trues, preds, average='macro', zero_division=0)))

def proto_calibrate(model, Xcal, ycal, n_cls=4):
    """Proto-B: compute per-class embedding centroids from n_cal samples."""
    model.eval()
    protos = []
    with torch.no_grad():
        for c in range(n_cls):
            idx = np.where(ycal == c)[0]
            if len(idx) == 0:
                protos.append(None); continue
            xc = torch.tensor(Xcal[idx], dtype=torch.float32).to(device)
            with autocast():
                emb = model.encode(xc) if hasattr(model,'encode') else model(xc)
            protos.append(emb.mean(0))
    return protos

def proto_predict(model, protos, Xtest, batch=512):
    """Proto-B: predict by nearest centroid (cosine similarity)."""
    model.eval()
    results = []
    with torch.no_grad():
        Xt = torch.tensor(Xtest, dtype=torch.float32)
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i+batch].to(device)
            with autocast():
                emb = model.encode(xb) if hasattr(model,'encode') else model(xb)
            valid = [(j,p) for j,p in enumerate(protos) if p is not None]
            if not valid:
                results.append(np.zeros(len(xb), dtype=int)); continue
            sims  = torch.stack([F.cosine_similarity(emb, p.unsqueeze(0).expand_as(emb))
                                  for _,p in valid], dim=1)
            idxs  = sims.argmax(1).cpu().numpy()
            cls_map = [j for j,_ in valid]
            results.append(np.array([cls_map[i] for i in idxs]))
    return np.concatenate(results)

def save_best(model, model_id, seed, fold, val_acc, best):
    if val_acc > best:
        torch.save({'model_state': model.state_dict(), 'val_acc': val_acc,
                    'seed': seed, 'fold': fold},
                   MODEL_DIR / f'model_{model_id}_s{seed}_f{fold:02d}_best.pth')
        return val_acc
    return best

N_CAL = 20
LABEL_SMOOTHING = 0.1
print('DL utilities ready.')



# ==============================================================================
# Notebook cell 25
# Categories: preprocessing, model_definition, results_tables
# ==============================================================================
# All sequence models use (B, n_bands, n_ch) view — bands as time steps.
# N_CH_FACED and IN_DIM_FACED are set from actual data in Cell 5.

class ShallowMLP(nn.Module):
    def __init__(self, in_dim, n_cls=4, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256,128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, n_cls))
    def forward(self,x): return self.net(x)
    def encode(self,x):  return self.net[:-1](x)

class DeepMLP(nn.Module):
    def __init__(self, in_dim, n_cls=4, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256,128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, n_cls))
    def forward(self,x): return self.net(x)
    def encode(self,x):  return self.net[:-1](x)

class LSTMModel(nn.Module):
    def __init__(self, n_ch, n_bands=5, hidden=128, n_layers=2, n_cls=4, drop=0.3):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands
        self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True, dropout=drop)
        self.head = nn.Linear(hidden, n_cls); self.drop = nn.Dropout(drop)
    def forward(self,x):
        out,_ = self.lstm(x.view(-1,self.nb,self.n_ch))
        return self.head(self.drop(out[:,-1]))
    def encode(self,x):
        out,_ = self.lstm(x.view(-1,self.nb,self.n_ch)); return out[:,-1]

class GRUModel(nn.Module):
    def __init__(self, n_ch, n_bands=5, hidden=128, n_layers=2, n_cls=4, drop=0.3):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands
        self.gru  = nn.GRU(n_ch, hidden, n_layers, batch_first=True, dropout=drop)
        self.head = nn.Linear(hidden, n_cls); self.drop = nn.Dropout(drop)
    def forward(self,x):
        out,_ = self.gru(x.view(-1,self.nb,self.n_ch))
        return self.head(self.drop(out[:,-1]))
    def encode(self,x):
        out,_ = self.gru(x.view(-1,self.nb,self.n_ch)); return out[:,-1]

class Conv1DModel(nn.Module):
    def __init__(self, n_ch, n_bands=5, n_cls=4, drop=0.3):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(128,n_cls))
    def forward(self,x): return self.head(self.conv(x.view(-1,self.n_ch,self.nb)))
    def encode(self,x):  return self.conv(x.view(-1,self.n_ch,self.nb)).squeeze(-1)

class VanillaTransformer(nn.Module):
    def __init__(self, n_ch, n_bands=5, d_model=64, n_heads=4, n_layers=2, n_cls=4, drop=0.2):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands
        self.proj = nn.Linear(n_bands, d_model)
        enc = nn.TransformerEncoderLayer(d_model,n_heads,256,drop,batch_first=True,norm_first=True)
        self.tf   = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model, n_cls); self.drop = nn.Dropout(drop)
    def forward(self,x): return self.head(self.drop(self.tf(self.proj(x.view(-1,self.n_ch,self.nb))).mean(1)))
    def encode(self,x):  return self.tf(self.proj(x.view(-1,self.n_ch,self.nb))).mean(1)

class EEGConformer(nn.Module):
    def __init__(self, n_ch, n_bands=5, d_model=64, n_heads=4, n_cls=4, drop=0.2):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.GELU())
        enc = nn.TransformerEncoderLayer(d_model,n_heads,256,drop,batch_first=True,norm_first=True)
        self.tf   = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(d_model, n_cls); self.drop = nn.Dropout(drop)
    def forward(self,x):
        return self.head(self.drop(self.tf(self.conv(x.view(-1,self.n_ch,self.nb)).transpose(1,2)).mean(1)))
    def encode(self,x):
        return self.tf(self.conv(x.view(-1,self.n_ch,self.nb)).transpose(1,2)).mean(1)

class ChanDropTransformer(nn.Module):
    def __init__(self, n_ch, n_bands=5, d_model=64, n_heads=4, n_layers=2, n_cls=4, drop=0.2, cdrop=0.1):
        super().__init__(); self.n_ch=n_ch; self.nb=n_bands; self.cdrop=cdrop
        self.proj = nn.Linear(n_bands, d_model)
        enc = nn.TransformerEncoderLayer(d_model,n_heads,256,drop,batch_first=True,norm_first=True)
        self.tf   = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model, n_cls); self.drop = nn.Dropout(drop)
    def forward(self,x):
        x = x.view(-1,self.n_ch,self.nb)
        if self.training and self.cdrop>0:
            m = torch.rand(x.size(0),self.n_ch,1,device=x.device)>self.cdrop
            x = x*m
        return self.head(self.drop(self.tf(self.proj(x)).mean(1)))
    def encode(self,x): return self.tf(self.proj(x.view(-1,self.n_ch,self.nb))).mean(1)

print(f'M11–M18 architectures defined (N_CH_FACED={N_CH_FACED}).')



# ==============================================================================
# Notebook cell 27
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha=alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):       return -ctx.alpha*g, None

class DANN(nn.Module):
    def __init__(self, in_dim, n_subj, n_cls=4, d=128, drop=0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256,d),      nn.BatchNorm1d(d),   nn.ReLU())
        self.cls = nn.Linear(d, n_cls)
        self.dom = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128,n_subj))
    def forward(self,x,alpha=1.0):
        h=self.enc(x); return self.cls(h), self.dom(GradientReversalFn.apply(h,alpha))
    def encode(self,x): return self.enc(x)

class CLISA(nn.Module):
    def __init__(self, in_dim, n_cls=4, d=128, proj=64, drop=0.3):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(drop),
                                   nn.Linear(256,d),nn.BatchNorm1d(d),nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(d,proj),nn.ReLU(),nn.Linear(proj,proj))
        self.cls  = nn.Linear(d, n_cls)
    def forward(self,x):          return self.cls(self.enc(x))
    def encode(self,x):           return self.enc(x)
    def project(self,x):          return F.normalize(self.proj(self.enc(x)),dim=1)

class SimCLR(nn.Module):
    def __init__(self, in_dim, n_cls=4, d=128, proj=64, drop=0.3):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(drop),
                                   nn.Linear(256,d),nn.BatchNorm1d(d),nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(d,proj),nn.ReLU(),nn.Linear(proj,proj))
        self.cls  = nn.Linear(d, n_cls)
    def forward(self,x):          return self.cls(self.enc(x))
    def encode(self,x):           return self.enc(x)
    def project(self,x):          return F.normalize(self.proj(self.enc(x)),dim=1)

def nt_xent_loss(z1, z2, temp=0.5):
    z1=F.normalize(z1,dim=1); z2=F.normalize(z2,dim=1)
    N=z1.size(0); z=torch.cat([z1,z2])
    sim=torch.mm(z,z.t())/temp; sim.fill_diagonal_(-1e4)
    labels=torch.cat([torch.arange(N,2*N),torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim,labels)

class _BYOLEnc(nn.Module):
    def __init__(self, in_dim, d=128, proj=64, pred=32, drop=0.3):
        super().__init__()
        self.bb   = nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(drop),
                                   nn.Linear(256,d),nn.BatchNorm1d(d),nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(d,proj),nn.BatchNorm1d(proj),nn.ReLU(),nn.Linear(proj,proj))
        self.pred = nn.Sequential(nn.Linear(proj,pred),nn.BatchNorm1d(pred),nn.ReLU(),nn.Linear(pred,proj))
    def forward(self,x): h=self.bb(x); return self.proj(h),h
    def predict(self,x): p,h=self.forward(x); return self.pred(p),h

class BYOL(nn.Module):
    def __init__(self, in_dim, n_cls=4, d=128, proj=64, drop=0.3, tau=0.996):
        super().__init__(); self.tau=tau
        self.online=_BYOLEnc(in_dim,d,proj,drop=drop)
        self.target=_BYOLEnc(in_dim,d,proj,drop=drop)
        self.cls=nn.Linear(d,n_cls)
        for po,pt in zip(self.online.parameters(),self.target.parameters()):
            pt.data.copy_(po.data); pt.requires_grad_(False)
    def update_target(self):
        for po,pt in zip(self.online.parameters(),self.target.parameters()):
            pt.data=self.tau*pt.data+(1-self.tau)*po.data
    def forward(self,x): _,h=self.online.forward(x); return self.cls(h)
    def encode(self,x):  _,h=self.online.forward(x); return h
    def byol_loss(self,x1,x2):
        p1,_=self.online.predict(x1); p2,_=self.online.predict(x2)
        with torch.no_grad():
            t1,_=self.target.forward(x1); t2,_=self.target.forward(x2)
        l = lambda a,b: 2-2*(F.normalize(a)*F.normalize(b.detach())).sum(1).mean()
        return (l(p1,t2)+l(p2,t1))*0.5

class PseudoLabelModel(nn.Module):
    def __init__(self,in_dim,n_cls=4,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(drop),
                                nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(drop),
                                nn.Linear(128,n_cls))
    def forward(self,x): return self.net(x)
    def encode(self,x):  return self.net[:-1](x)

class MixMatchModel(nn.Module):
    def __init__(self,in_dim,n_cls=4,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(drop),
                                nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(drop),
                                nn.Linear(128,n_cls))
    def forward(self,x): return self.net(x)
    def encode(self,x):  return self.net[:-1](x)

print('M19–M24 architectures defined.')



# ==============================================================================
# Notebook cell 29
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ── Frozen hyperparameters — plan Section 16 ─────────────────────────────────
D_MODEL = 32; N_HEADS = 4; N_LAYERS_T = 4; N_LAYERS_S = 2
D_FF = 128;   D_PROJ = 128; DROPOUT = 0.2
PRETRAIN_LR = 1e-3; FINETUNE_LR_CLS = 3e-4; FINETUNE_LR_ENC = 3e-5
DISTILL_LR = 1e-3; DISTILL_TEMP = 4.0; CONTRASTIVE_TEMP = 0.5
MASK_RATIO = 0.3; NOISE_STD = 0.1; MIXUP_ALPHA = 0.4; SUBJECT_WEIGHT = 0.5

# Epoch schedule from TEST_RUN config (Cell 1)
PRETRAIN_EPOCHS = _PRE_EP
FINETUNE_EPOCHS = _FT_EP
DISTILL_EPOCHS  = _DIST_EP

class DANCEEncoder(nn.Module):
    def __init__(self, n_ch, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS_T,
                 d_ff=D_FF, drop=DROPOUT):
        super().__init__()
        self.n_ch = n_ch
        self.proj = nn.Linear(5, d_model)          # 5 bands → d_model
        self.pos  = nn.Parameter(torch.randn(1,1,d_model)*0.01)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, drop,
                                          batch_first=True, norm_first=True, activation='gelu')
        self.tf   = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, n_ch*5) → reshape → (B, n_ch, 5)
        B = x.shape[0]
        x = x.view(B, self.n_ch, 5)
        x = self.proj(x) + self.pos       # (B, n_ch, d_model)
        return self.norm(self.tf(x).mean(1))  # (B, d_model)

class DANCETeacher(nn.Module):
    def __init__(self, n_ch, n_subj, n_cls=4):
        super().__init__()
        self.n_ch    = n_ch
        self.encoder = DANCEEncoder(n_ch, D_MODEL, N_HEADS, N_LAYERS_T, D_FF, DROPOUT)
        self.proj    = nn.Sequential(nn.Linear(D_MODEL,D_PROJ), nn.ReLU(), nn.Linear(D_PROJ,D_PROJ))
        self.cls     = nn.Linear(D_MODEL, n_cls)
        self.adv     = nn.Sequential(nn.Linear(D_MODEL,128), nn.ReLU(), nn.Linear(128,n_subj))

    def encode(self, x): return self.encoder(x)
    def forward(self, x, alpha=1.0):
        h    = self.encoder(x)
        proj = F.normalize(self.proj(h), dim=1)
        rev  = GradientReversalFn.apply(h, alpha)
        return self.cls(h), self.adv(rev), proj, h

class DANCEStudent(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.encoder = DANCEEncoder(6, D_MODEL, N_HEADS, N_LAYERS_S, D_FF, DROPOUT)
        self.proj    = nn.Sequential(nn.Linear(D_MODEL,D_PROJ), nn.ReLU(), nn.Linear(D_PROJ,D_PROJ))
        self.cls     = nn.Linear(D_MODEL, n_cls)

    def encode(self, x): return self.encoder(x)
    def forward(self, x):
        h    = self.encoder(x)
        proj = F.normalize(self.proj(h), dim=1)
        return self.cls(h), proj, h

print(f'DANCE Teacher (n_ch={N_CH_FACED}) + Student (n_ch=6) defined.')
print(f'Epochs: pretrain={PRETRAIN_EPOCHS} finetune={FINETUNE_EPOCHS} distill={DISTILL_EPOCHS}')



# ==============================================================================
# Notebook cell 31
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
def train_dl_loso(model_id, model_factory, X_all, y_all, subj_all,
                  n_epochs=_EPOCHS_DL, batch_size=128, lr=1e-3,
                  is_ssl=False, is_byol=False, is_dann=False):
    """
    Full LOSO for one DL model. Skips completed folds (resume).
    model_factory: () → fresh nn.Module each fold.
    """
    total = N_FOLDS_FACED * len(SEEDS_FACED)
    done  = model_done_count(model_id, N_FOLDS_FACED, SEEDS_FACED)
    if done == total:
        print(f'SKIP {model_id} — all {total} done'); return

    print(f'\n{model_id}: {done}/{total} remaining...', flush=True)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for seed in SEEDS_FACED:
        torch.manual_seed(seed); np.random.seed(seed)
        for fi in range(N_FOLDS_FACED):
            fold = fi + 1
            if ckpt_exists(model_id, seed, fold): continue

            Xtr,ytr,_,Xvl,yvl,Xte,yte = get_faced_fold(fi, X_all, y_all, subj_all, seed)
            model  = model_factory().to(device)
            opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
            scaler = GradScaler(enabled=(device.type == 'cuda'))
            tr_ld  = make_loader(Xtr, ytr, batch_size, True,  True)
            vl_ld  = make_loader(Xvl, yvl, batch_size, False, False)
            te_ld  = make_loader(Xte, yte,  batch_size, False, False)

            best = 0.0; pat = 0
            for epoch in range(n_epochs):
                model.train()
                for Xb, yb in tr_ld:
                    Xb, yb = Xb.to(device), yb.to(device)
                    opt.zero_grad()
                    with maybe_autocast():
                        if is_dann:
                            logits, dom = model(Xb, alpha=min(1.0, 2*epoch/max(n_epochs,1)))
                            loss = crit(logits, yb)
                        elif is_byol:
                            half = max(1, len(Xb)//2)
                            loss = crit(model(Xb), yb) + 0.1*model.byol_loss(Xb[:half], Xb[half:2*half])
                        elif is_ssl:
                            half = max(1, len(Xb)//2)
                            loss_ssl = nt_xent_loss(model.project(Xb[:half]),
                                                    model.project(Xb[half:2*half]),
                                                    CONTRASTIVE_TEMP) if half>1 else torch.tensor(0.)
                            loss = crit(model(Xb), yb) + 0.1*loss_ssl
                        else:
                            loss = crit(model(Xb), yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    if is_byol: model.update_target()
                sched.step()
                vacc, _ = evaluate(model, vl_ld)
                best = save_best(model, model_id, seed, fold, vacc, best)
                if vacc <= best - 1e-6: pat += 1
                else:                    pat = 0
                if pat >= 15:           break   # patience=15 — plan §9

            # Load best weights
            bp = MODEL_DIR / f'model_{model_id}_s{seed}_f{fold:02d}_best.pth'
            if bp.exists():
                model.load_state_dict(torch.load(bp, map_location=device)['model_state'])

            # Proto-A
            acc_a, f1_a = evaluate(model, te_ld)
            # Proto-B (n_cal=20)
            cal = np.random.choice(len(Xte), min(N_CAL, len(Xte)), replace=False)
            protos = proto_calibrate(model, Xte[cal], yte[cal])
            pb     = proto_predict(model, protos, Xte)
            acc_b  = float(accuracy_score(yte, pb))
            f1_b   = float(f1_score(yte, pb, average='macro', zero_division=0))

            res = dict(model=model_id, seed=seed, fold=fold,
                       acc_a=acc_a, f1_a=f1_a, acc_b=acc_b, f1_b=f1_b, n_test=len(yte))
            save_ckpt(model_id, seed, fold, res)
            del model; torch.cuda.empty_cache()
            print(f'  {model_id} s={seed} f={fold}/{N_FOLDS_FACED} '
                  f'AccA={acc_a:.4f} AccB={acc_b:.4f} F1B={f1_b:.4f}', flush=True)

print('Generic LOSO loop defined.')



# ==============================================================================
# Notebook cell 33
# Categories: preprocessing, model_definition, training
# ==============================================================================
# Auto-detect input dim from actual data
IN_DIM = X_f1_base.shape[1]   # 150 for 30ch × 5bands
IN_6   = 30                    # 6ch × 5bands
print(f'IN_DIM={IN_DIM}  IN_6={IN_6}')

DL_CONFIGS = [
    # id, factory, is_ssl, is_byol, is_dann
    ('M11_ShallowMLP',  lambda: ShallowMLP(IN_DIM),               False, False, False),
    ('M12_DeepMLP',     lambda: DeepMLP(IN_DIM),                  False, False, False),
    ('M13_LSTM',        lambda: LSTMModel(N_CH_FACED),            False, False, False),
    ('M14_GRU',         lambda: GRUModel(N_CH_FACED),             False, False, False),
    ('M15_Conv1D',      lambda: Conv1DModel(N_CH_FACED),          False, False, False),
    ('M16_Transformer', lambda: VanillaTransformer(N_CH_FACED),   False, False, False),
    ('M17_Conformer',   lambda: EEGConformer(N_CH_FACED),         False, False, False),
    ('M18_ChanDrop',    lambda: ChanDropTransformer(N_CH_FACED),  False, False, False),
    ('M20_CLISA',       lambda: CLISA(IN_DIM),                    True,  False, False),
    ('M21_SimCLR',      lambda: SimCLR(IN_DIM),                   True,  False, False),
    ('M22_BYOL',        lambda: BYOL(IN_DIM),                     False, True,  False),
    ('M23_PseudoLabel', lambda: PseudoLabelModel(IN_DIM),         False, False, False),
    ('M24_MixMatch',    lambda: MixMatchModel(IN_DIM),            False, False, False),
]

for mid, factory, is_ssl, is_byol, is_dann in DL_CONFIGS:
    train_dl_loso(mid, factory, X_f1_base, y_f, subj_f,
                  n_epochs=_EPOCHS_DL, is_ssl=is_ssl, is_byol=is_byol, is_dann=is_dann)

print('\n✅ M11-M24 done.')



# ==============================================================================
# Notebook cell 35
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
mid_dann = 'M19_DANN'
total_d  = N_FOLDS_FACED * len(SEEDS_FACED)
done_d   = model_done_count(mid_dann, N_FOLDS_FACED, SEEDS_FACED)

if done_d == total_d:
    print(f'SKIP {mid_dann} — all done')
else:
    print(f'Running {mid_dann} ({done_d}/{total_d})...', flush=True)
    crit_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    for seed in SEEDS_FACED:
        torch.manual_seed(seed); np.random.seed(seed)
        for fi in range(N_FOLDS_FACED):
            fold = fi + 1
            if ckpt_exists(mid_dann, seed, fold): continue

            Xtr,ytr,sbl_tr,Xvl,yvl,Xte,yte = get_faced_fold(fi, X_f1_base, y_f, subj_f, seed)
            n_subj_tr = len(np.unique(sbl_tr))
            model  = DANN(IN_DIM, n_subj_tr).to(device)
            opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, _EPOCHS_DL)
            scaler = GradScaler(enabled=(device.type == 'cuda'))
            tr_ld  = make_loader(Xtr, ytr, 128, True,  True)
            vl_ld  = make_loader(Xvl, yvl, 128, False, False)
            te_ld  = make_loader(Xte, yte,  128, False, False)

            # Subject-encoded loader for adversarial training
            subj_enc    = {s:i for i,s in enumerate(np.unique(sbl_tr))}
            sbl_enc     = np.array([subj_enc[s] for s in sbl_tr], dtype=np.int64)
            subj_loader = DataLoader(
                TensorDataset(torch.tensor(Xtr,dtype=torch.float32),
                              torch.tensor(sbl_enc,dtype=torch.long)),
                batch_size=128, shuffle=True, drop_last=True)

            best = 0.0; pat = 0
            for epoch in range(_EPOCHS_DL):
                model.train()
                alpha = min(1.0, 2*epoch/max(_EPOCHS_DL,1))
                for (Xb,yb),(Xs,sb) in zip(tr_ld, subj_loader):
                    Xb,yb,Xs,sb = Xb.to(device),yb.to(device),Xs.to(device),sb.to(device)
                    opt.zero_grad()
                    with autocast():
                        cls,_   = model(Xb, alpha)
                        _,dom   = model(Xs, alpha)
                        loss    = crit_ce(cls,yb) + 0.1*F.cross_entropy(dom,sb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                sched.step()
                vacc,_ = evaluate(model, vl_ld)
                best   = save_best(model, mid_dann, seed, fold, vacc, best)
                if vacc <= best-1e-6: pat+=1
                else:                  pat=0
                if pat>=15:           break

            bp = MODEL_DIR/f'model_{mid_dann}_s{seed}_f{fold:02d}_best.pth'
            if bp.exists():
                model.load_state_dict(torch.load(bp,map_location=device)['model_state'])

            acc_a,f1_a = evaluate(model, te_ld)
            cal = np.random.choice(len(Xte), min(N_CAL,len(Xte)), replace=False)
            protos = proto_calibrate(model, Xte[cal], yte[cal])
            pb     = proto_predict(model, protos, Xte)
            acc_b  = float(accuracy_score(yte,pb))
            f1_b   = float(f1_score(yte,pb,average='macro',zero_division=0))

            save_ckpt(mid_dann, seed, fold,
                      dict(model=mid_dann,seed=seed,fold=fold,
                           acc_a=acc_a,f1_a=f1_a,acc_b=acc_b,f1_b=f1_b,n_test=len(yte)))
            del model; torch.cuda.empty_cache()
            print(f'  M19 s={seed} f={fold}/{N_FOLDS_FACED} AccA={acc_a:.4f} AccB={acc_b:.4f}',flush=True)

print('\n✅ M19 DANN done.')



# ==============================================================================
# Notebook cell 37
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
def train_dance_teacher_fold(fi, seed):
    fold = fi + 1; mid = 'M25_DANCE_Teacher'
    if ckpt_exists(mid, seed, fold): return load_ckpt(mid, seed, fold)

    Xtr,ytr,sbl_tr,Xvl,yvl,Xte,yte = get_faced_fold(fi, X_f1_base, y_f, subj_f, seed)
    n_subj_tr = len(np.unique(sbl_tr))
    teacher   = DANCETeacher(N_CH_FACED, n_subj_tr).to(device)

    # Stage 1: contrastive pretraining
    opt1   = torch.optim.AdamW(teacher.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, max(PRETRAIN_EPOCHS,1))
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    tr_ld  = make_loader(Xtr, ytr, 128, True, True)

    for epoch in range(PRETRAIN_EPOCHS):
        teacher.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt1.zero_grad()
            with autocast():
                cls, dom, proj, h = teacher(Xb)
                # NT-Xent with noise augmentation as positive pair
                Xb_n = Xb + torch.randn_like(Xb)*NOISE_STD
                _,_,proj_n,_ = teacher(Xb_n)
                loss_ctr = nt_xent_loss(proj, proj_n, CONTRASTIVE_TEMP)
                # Channel masking augmentation
                mask = (torch.rand(Xb.shape[0],N_CH_FACED,1,device=device) > MASK_RATIO)
                Xb_m = (Xb.view(-1,N_CH_FACED,5) * mask).view(-1, IN_DIM_FACED)
                _,_,proj_m,_ = teacher(Xb_m)
                loss_ctr += nt_xent_loss(proj, proj_m, CONTRASTIVE_TEMP)
                # Adversarial loss
                dummy_labels = torch.zeros(len(Xb),dtype=torch.long,device=device)
                loss = loss_ctr + SUBJECT_WEIGHT * F.cross_entropy(dom, dummy_labels % n_subj_tr)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            scaler.step(opt1); scaler.update()
        sched1.step()

    # Stage 2: supervised finetuning
    opt2 = torch.optim.AdamW([
        {'params': teacher.encoder.parameters(), 'lr': FINETUNE_LR_ENC},
        {'params': teacher.cls.parameters(),     'lr': FINETUNE_LR_CLS},
    ], weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, max(FINETUNE_EPOCHS,1))
    crit   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    vl_ld  = make_loader(Xvl, yvl, 128, False, False)
    te_ld  = make_loader(Xte, yte,  128, False, False)
    best   = 0.0; pat = 0

    for epoch in range(FINETUNE_EPOCHS):
        teacher.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt2.zero_grad()
            with autocast():
                cls,_,_,_ = teacher(Xb); loss = crit(cls, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            scaler.step(opt2); scaler.update()
        sched2.step()
        vacc,_ = evaluate(teacher, vl_ld)
        best   = save_best(teacher, mid, seed, fold, vacc, best)
        if vacc<=best-1e-6: pat+=1
        else:                pat=0
        if pat>=15: break

    # Load best + save teacher weights for M26
    bp = MODEL_DIR/f'model_{mid}_s{seed}_f{fold:02d}_best.pth'
    if bp.exists():
        teacher.load_state_dict(torch.load(bp,map_location=device)['model_state'])
    torch.save(teacher.state_dict(), MODEL_DIR/f'teacher_faced_s{seed}_f{fold:02d}.pth')

    acc_a,f1_a = evaluate(teacher, te_ld)
    cal = np.random.choice(len(Xte), min(N_CAL,len(Xte)), replace=False)
    protos = proto_calibrate(teacher, Xte[cal], yte[cal])
    pb     = proto_predict(teacher, protos, Xte)
    acc_b  = float(accuracy_score(yte,pb)); f1_b=float(f1_score(yte,pb,average='macro',zero_division=0))

    res = dict(model=mid,seed=seed,fold=fold,acc_a=acc_a,f1_a=f1_a,acc_b=acc_b,f1_b=f1_b,n_test=len(yte))
    save_ckpt(mid, seed, fold, res)
    del teacher; torch.cuda.empty_cache()
    print(f'  M25 s={seed} f={fold}/{N_FOLDS_FACED} AccA={acc_a:.4f} AccB={acc_b:.4f}',flush=True)
    return res

# Run M25
total25 = N_FOLDS_FACED*len(SEEDS_FACED)
done25  = model_done_count('M25_DANCE_Teacher', N_FOLDS_FACED, SEEDS_FACED)
if done25 == total25:
    print('SKIP M25 — all done')
else:
    print(f'M25 DANCE Teacher: {done25}/{total25} done')
    for seed in SEEDS_FACED:
        for fi in range(N_FOLDS_FACED):
            train_dance_teacher_fold(fi, seed)
print('\n✅ M25 DANCE Teacher complete.')



# ==============================================================================
# Notebook cell 39
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification
# ==============================================================================
def train_dance_student_fold(fi, seed):
    fold = fi+1; mid='M26_DANCE_Student'
    if ckpt_exists(mid, seed, fold): return load_ckpt(mid, seed, fold)

    tp = MODEL_DIR/f'teacher_faced_s{seed}_f{fold:02d}.pth'
    if not tp.exists():
        print(f'  SKIP M26 s={seed} f={fold} — teacher weights missing, run M25 first')
        return None

    _,ytr,sbl_tr,_,yvl,_,yte = get_faced_fold(fi, X_f1_base, y_f, subj_f, seed)
    Xtr6,_,_,Xvl6,_,Xte6,_  = get_faced_fold(fi, X_f1_6ch, y_f, subj_f, seed)
    Xtr32,_,_,_,_,_,_        = get_faced_fold(fi, X_f1_base, y_f, subj_f, seed)

    n_subj_tr = len(np.unique(sbl_tr))

    # Load frozen teacher
    teacher = DANCETeacher(N_CH_FACED, n_subj_tr).to(device)
    teacher.load_state_dict(torch.load(tp, map_location=device))
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()

    student = DANCEStudent().to(device)
    opt     = torch.optim.AdamW(student.parameters(), lr=DISTILL_LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(DISTILL_EPOCHS,1))
    scaler  = GradScaler()
    crit_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    tr6  = make_loader(Xtr6,  ytr, 128, True,  True)
    tr32 = make_loader(Xtr32, ytr, 128, True,  False)
    vl6  = make_loader(Xvl6,  yvl, 128, False, False)
    te6  = make_loader(Xte6,  yte,  128, False, False)

    best=0.0; pat=0
    for epoch in range(DISTILL_EPOCHS):
        student.train()
        for (Xb6,yb),(Xb32,_) in zip(tr6, tr32):
            Xb6,yb,Xb32 = Xb6.to(device),yb.to(device),Xb32.to(device)
            opt.zero_grad()
            with autocast():
                cls_s,proj_s,h_s = student(Xb6)
                with torch.no_grad():
                    cls_t,_,proj_t,h_t = teacher(Xb32)
                # plan §16: 1*MSE + 2*KL + 1*CE
                mse = F.mse_loss(h_s, h_t)
                kl  = F.kl_div(F.log_softmax(cls_s/DISTILL_TEMP,1),
                                F.softmax(cls_t.detach()/DISTILL_TEMP,1),
                                reduction='batchmean') * DISTILL_TEMP**2
                loss = mse + 2.0*kl + crit_ce(cls_s,yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        vacc,_=evaluate(student,vl6)
        best=save_best(student,mid,seed,fold,vacc,best)
        if vacc<=best-1e-6: pat+=1
        else:                pat=0
        if pat>=15: break

    bp=MODEL_DIR/f'model_{mid}_s{seed}_f{fold:02d}_best.pth'
    if bp.exists():
        student.load_state_dict(torch.load(bp,map_location=device)['model_state'])

    acc_a,f1_a = evaluate(student, te6)
    cal = np.random.choice(len(Xte6), min(N_CAL,len(Xte6)), replace=False)
    protos = proto_calibrate(student, Xte6[cal], yte[cal])
    pb     = proto_predict(student, protos, Xte6)
    acc_b  = float(accuracy_score(yte,pb)); f1_b=float(f1_score(yte,pb,average='macro',zero_division=0))

    res = dict(model=mid,seed=seed,fold=fold,acc_a=acc_a,f1_a=f1_a,acc_b=acc_b,f1_b=f1_b,n_test=len(yte))
    save_ckpt(mid,seed,fold,res)
    del student,teacher; torch.cuda.empty_cache()
    print(f'  M26 s={seed} f={fold}/{N_FOLDS_FACED} AccA={acc_a:.4f} AccB={acc_b:.4f}',flush=True)
    return res

total26=N_FOLDS_FACED*len(SEEDS_FACED)
done26=model_done_count('M26_DANCE_Student',N_FOLDS_FACED,SEEDS_FACED)
if done26==total26:
    print('SKIP M26 — all done')
else:
    print(f'M26 DANCE Student: {done26}/{total26} done')
    for seed in SEEDS_FACED:
        for fi in range(N_FOLDS_FACED):
            train_dance_student_fold(fi, seed)
print('\n✅ M26 DANCE Student complete.')



# ==============================================================================
# Notebook cell 41
# Categories: preprocessing, model_definition, results_tables
# ==============================================================================
ALL_IDS = [
    'M01_LDA_faced_30ch','M02_SVM_faced_30ch','M03_RF_faced_30ch',
    'M04_KNN_faced_30ch','M05_LR_faced_30ch','M06_NB_faced_30ch',
    'M07_ET_faced_30ch','M08_GB_faced_30ch','M09_XGB_faced_30ch','M10_MLP_SK_faced_30ch',
    'M01_LDA_faced_6ch','M02_SVM_faced_6ch','M03_RF_faced_6ch',
    'M04_KNN_faced_6ch','M05_LR_faced_6ch','M06_NB_faced_6ch',
    'M07_ET_faced_6ch','M08_GB_faced_6ch','M09_XGB_faced_6ch','M10_MLP_SK_faced_6ch',
    'M11_ShallowMLP','M12_DeepMLP','M13_LSTM','M14_GRU',
    'M15_Conv1D','M16_Transformer','M17_Conformer','M18_ChanDrop',
    'M19_DANN','M20_CLISA','M21_SimCLR','M22_BYOL',
    'M23_PseudoLabel','M24_MixMatch','M25_DANCE_Teacher','M26_DANCE_Student',
]

rows = []
for mid in ALL_IDS:
    for s in SEEDS_FACED:
        for f in range(1, N_FOLDS_FACED+1):
            r = load_ckpt(mid, s, f)
            if r: rows.append(r)

df_all = pd.DataFrame(rows)
df_all.to_csv(RESULTS_DIR/'faced_models_all_folds.csv', index=False)

summary = []
for mid in ALL_IDS:
    sub = df_all[df_all['model']==mid] if len(df_all) else pd.DataFrame()
    if len(sub)==0: continue
    if 'acc_b' in sub.columns:
        summary.append({'model':mid,'n_runs':len(sub),
                        'AccA':sub['acc_a'].mean(),'AccB':sub['acc_b'].mean(),
                        'AccB_std':sub['acc_b'].std(),
                        'F1A':sub['f1_a'].mean(),'F1B':sub['f1_b'].mean()})
    else:
        summary.append({'model':mid,'n_runs':len(sub),
                        'AccA':float('nan'),'AccB':sub['acc'].mean(),
                        'AccB_std':sub['acc'].std(),
                        'F1A':float('nan'),'F1B':sub['f1'].mean()})

df_sum = pd.DataFrame(summary).sort_values('F1B', ascending=False)
df_sum.to_csv(RESULTS_DIR/'faced_models_master_summary.csv', index=False)

print('\n'+'='*80)
print('  FACED MODELS SUMMARY  (mean ± std over LOSO folds × seeds)')
print('='*80)
for _,r in df_sum.iterrows():
    print(f"  {r['model']:30s}  n={int(r['n_runs']):4d}  "
          f"AccB={r['AccB']:.4f}±{r['AccB_std']:.4f}  F1B={r['F1B']:.4f}")
print('='*80)
print(f"Saved: {RESULTS_DIR/'faced_models_master_summary.csv'}")
print(f"       {RESULTS_DIR/'faced_models_all_folds.csv'}")



# ==============================================================================
# Notebook cell 43
# Categories: preprocessing, evaluation, results_tables, figures
# ==============================================================================
try:
    plot_df = df_sum[df_sum['F1B'].notna()].head(20)
    fig, ax = plt.subplots(figsize=(12,7))
    ax.barh(plot_df['model'], plot_df['F1B'], xerr=plot_df['AccB_std'],
            color='steelblue', alpha=0.8, capsize=3)
    ax.axvline(0.25, color='red', linestyle='--', lw=1, label='Chance')
    ax.set_xlabel('Macro-F1 (FACED LOSO)'); ax.legend()
    ax.set_title('FACED standalone E02 — AccB/F1B mean ± std')
    plt.tight_layout()
    out = FIG_DIR/'faced_models_f1b_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Plot saved: {out}')
except Exception as e:
    print(f'Plot skipped: {e}')



# ==============================================================================
# Notebook cell 45
# Categories: preprocessing, training, figures
# ==============================================================================
ts = datetime.now().strftime('%Y%m%d_%H%M')

# 1. Checkpoints zip (most critical — enables resume next session)
ckpt_zip = BASE/f'checkpoints_{ts}.zip'
with zipfile.ZipFile(ckpt_zip,'w',zipfile.ZIP_DEFLATED) as zf:
    for jf in CKPT_DIR.glob('*.json'):
        zf.write(jf, f'checkpoints/loso_results/{jf.name}')
    for pf in MODEL_DIR.glob('*.pth'):
        zf.write(pf, f'checkpoints/{pf.name}')
    # Include prep flags so preprocessing is skipped next session
    for ff in FEAT_OUT.glob('*.flag'):
        zf.write(ff, f'features/preprocessed/{ff.name}')
print(f'checkpoints zip : {ckpt_zip.name}  ({ckpt_zip.stat().st_size/1e6:.1f} MB)')

# 2. Preprocessed features (only needed from first session)
prep_zip = BASE/f'features_preprocessed_{ts}.zip'
npy_list = list(FEAT_OUT.glob('*.npy'))
if npy_list:
    with zipfile.ZipFile(prep_zip,'w',zipfile.ZIP_DEFLATED) as zf:
        for nf in npy_list:
            zf.write(nf, f'features/preprocessed/{nf.name}')
        for ff in FEAT_OUT.glob('*.flag'):
            zf.write(ff, f'features/preprocessed/{ff.name}')
    print(f'features zip    : {prep_zip.name}  ({prep_zip.stat().st_size/1e6:.1f} MB)')

# 3. Results + figures
res_zip = BASE/f'results_faced_{ts}.zip'
with zipfile.ZipFile(res_zip,'w',zipfile.ZIP_DEFLATED) as zf:
    for cf in RESULTS_DIR.glob('*.csv'):
        zf.write(cf, f'results/deep_models_faced/{cf.name}')
    for pf in FIG_DIR.glob('*.png'):
        zf.write(pf, f'figures/models/{pf.name}')
print(f'results zip     : {res_zip.name}  ({res_zip.stat().st_size/1e6:.1f} MB)')

session_status()

print('\n'+'='*62)
print('KAGGLE SESSION END — DOWNLOAD THESE FILES:')
print('='*62)
print(f'  1. {ckpt_zip.name}')
print(f'     → next session: upload as cse400c-checkpoints dataset')
print(f'  2. {prep_zip.name}')
print(f'     → local: extract to features/preprocessed/ (one-time)')
print(f'  3. {res_zip.name}')
print(f'     → local: extract to results/deep_models_faced/ + figures/models/')
print()
print('Local root: C:/Users/Saif/Desktop/CSE400/C/')
print('='*62)



# ==============================================================================
# Notebook cell 46
# Categories: other
# ==============================================================================

