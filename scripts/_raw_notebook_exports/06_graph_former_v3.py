# Auto-exported raw code from notebook: 06-graph-former-v3.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: training, audit_verification
# ==============================================================================
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SMOKE TEST FLAG  —  set True to do a quick end-to-end sanity run           ║
# ║  Smoke-test: 1 seed · 2 folds · 2 pretrain · 4 finetune · 3 distill eps    ║
# ║  Full run  : 3 seeds · 15 folds · 100 pretrain · 100 finetune · 50 distill ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
SMOKE_TEST = True   # ← flip to True for a quick pipeline check

print(f'🔬 SMOKE_TEST = {SMOKE_TEST}')
if SMOKE_TEST:
    print('   → 1 seed, 2 folds, tiny epochs — pipeline sanity check only')
else:
    print('   → Full training run (3 seeds × 15 folds)')


# ==============================================================================
# Notebook cell 2
# Categories: preprocessing, results_tables
# ==============================================================================
# ── Cell 0: Select a PyTorch build compatible with the Kaggle GPU ───────────
# Why this exists:
# - Kaggle may assign either a T4 (Turing, sm_75) or a P100 (Pascal, sm_60).
# - PyTorch CUDA 12.8+ wheels dropped Pascal/P100 support, which causes:
#     "CUDA error: no kernel image is available for execution on the device"
# - For P100, install a CUDA 12.6 wheel before importing torch.
#
# Official references:
# - PyTorch previous versions: cu126 wheels are available for torch 2.10.0
# - PyTorch packaging note: Pascal removed from CUDA 12.8+ builds

import os, sys, subprocess, shlex

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def sh(cmd):
    return run(['bash', '-lc', cmd])

def get_gpu_name():
    res = sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1")
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    return 'NO_GPU'

def pip_show_version(pkg='torch'):
    res = run([sys.executable, '-m', 'pip', 'show', pkg])
    if res.returncode != 0:
        return 'not-installed'
    for line in res.stdout.splitlines():
        if line.startswith('Version:'):
            return line.split(':', 1)[1].strip()
    return 'unknown'

gpu_name = get_gpu_name()
torch_ver_before = pip_show_version('torch')

print(f'GPU detected      : {gpu_name}')
print(f'torch before      : {torch_ver_before}')
print(f'Python            : {sys.version.split()[0]}')

IS_P100 = 'P100' in gpu_name.upper()
IS_T4   = 'T4' in gpu_name.upper()

if IS_P100:
    print('P100 detected -> installing Pascal-compatible PyTorch wheel (CUDA 12.6).')
    cmd = [
        sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
        '--force-reinstall',
        'torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0',
        '--index-url', 'https://download.pytorch.org/whl/cu126'
    ]
    ret = run(cmd)
    if ret.returncode != 0:
        print('⚠ cu126 install failed. Last stderr lines:')
        print('\n'.join(ret.stderr.strip().splitlines()[-20:]))
        raise RuntimeError('Failed to install torch cu126 required for P100 / Pascal GPUs.')
    else:
        print('✅ Installed Pascal-compatible torch build for P100.')
elif IS_T4:
    print('T4 detected -> keeping Kaggle preinstalled torch (CUDA 12.8/13.x is OK on T4).')
else:
    print('No known Kaggle GPU detected -> keeping current torch build.')

print(f'torch after       : {pip_show_version("torch")}')



# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, model_definition, training
# ==============================================================================
# ── Cell 1: Environment Setup (Kaggle only) ────────────────────────────────────
import os

assert os.path.exists('/kaggle/input'), (
    'This notebook is Kaggle-only. '
    'If running locally, mount /kaggle/input or adapt paths manually.')

BASE     = '/kaggle/working'
FEAT_SRC = '/kaggle/input/datasets/stone369/features'
print('🔧 Kaggle environment')

# Ensure output directories exist
for d in ['checkpoints/graph_former', 'results/graph_former', 'weights/graph_former']:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

print(f'✅ BASE     : {BASE}')
print(f'   FEAT_SRC : {FEAT_SRC}')


# ==============================================================================
# Notebook cell 4
# Categories: model_definition, training, evaluation, webapp_or_demo
# ==============================================================================
# ── Cell 2: Imports ────────────────────────────────────────────────────────────
import glob, json, time, random, traceback, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stability-first defaults for Kaggle:
# - AMP disabled by default to avoid fp16/bfloat16 corner-case crashes
# - TF32 allowed on supported GPUs for some speed without AMP instability
USE_AMP = False
AMP_ENABLED = bool(device.type == 'cuda' and USE_AMP)

if device.type == 'cuda':
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def amp_ctx():
    return autocast(device_type='cuda', enabled=AMP_ENABLED) if device.type == 'cuda' else nullcontext()

def make_scaler():
    return GradScaler('cuda', enabled=AMP_ENABLED) if device.type == 'cuda' else GradScaler('cuda', enabled=False)

def seed_cpu_only(seed: int):
    """Seed Python / NumPy / CPU torch RNG without touching CUDA RNG state."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))
    torch.set_rng_state(g.get_state())

print(f'✅ PyTorch {torch.__version__}')
print(f'   Device : {device}')
if torch.cuda.is_available():
    print(f'   GPU    : {torch.cuda.get_device_name(0)}')
    print(f'   CUDA   : {torch.version.cuda}')
    print(f'   CC     : {torch.cuda.get_device_capability(0)}')
    print(f'   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'   AMP    : {AMP_ENABLED} (disabled by default for stability)')
else:
    print('⚠ No GPU detected — training will be very slow')



# ==============================================================================
# Notebook cell 5
# Categories: model_definition, training, results_tables
# ==============================================================================
# ── Cell 3: Checkpoint Helpers ─────────────────────────────────────────────────
CKPT_DIR    = os.path.join(BASE, 'checkpoints/graph_former')
WEIGHTS_DIR = os.path.join(BASE, 'weights/graph_former')

def ckpt_key(model_id, seed, fold):
    return f'{model_id}_seed{seed}_fold{fold:02d}'

def ckpt_path(key):
    return os.path.join(CKPT_DIR, f'{key}.json')

def teacher_pth(seed, fold):
    return os.path.join(WEIGHTS_DIR, f'M27_teacher_s{seed}_f{fold:02d}.pth')

def save_ckpt(key, result_dict):
    """Atomically write checkpoint JSON."""
    tmp = ckpt_path(key) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result_dict, f, indent=2)
    os.replace(tmp, ckpt_path(key))   # atomic on POSIX

def load_ckpt(key):
    """Return checkpoint dict or None."""
    p = ckpt_path(key)
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f'  ⚠ Corrupt checkpoint ignored: {p}')
    return None

def ckpt_complete(model_id, seed, fold):
    """True only if both the JSON result and (for M27) the .pth weights exist."""
    key  = ckpt_key(model_id, seed, fold)
    data = load_ckpt(key)
    if data is None:
        return False
    if model_id == 'M27_teacher':
        return os.path.exists(teacher_pth(seed, fold))
    return True

def print_resume_banner(model_id, seeds, n_folds):
    """Print a concise resume status table."""
    total = n_folds * len(seeds)
    done  = sum(1 for s in seeds for f in range(n_folds)
                if ckpt_complete(model_id, s, f))
    print(f'  {model_id}: {done}/{total} folds complete', end='')
    if done == total:
        print('  ← nothing to do')
    elif done == 0:
        print('  ← starting fresh')
    else:
        remaining = total - done
        print(f'  ← resuming ({remaining} remaining)')
    return done, total

print('✅ Checkpoint helpers defined')


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, model_definition, training, results_tables, audit_verification
# ==============================================================================
# ── Cell 4: Data Loading ───────────────────────────────────────────────────────
# SEED-IV: 15 subjects, 62ch × 5 bands = 310 features, 4 classes
# Student channels: FP1,FP2,F7,F8,T7,T8 → indices [0,2,5,13,23,31] in 62-ch ordering
# (6ch × 5 bands = 30 features for M28)

CH6_SEED = [0, 2, 5, 13, 23, 31]   # FP1,FP2,F7,F8,T7,T8
N_CH     = 62
N_BANDS  = 5
N_CLS    = 4

# Build 6-ch feature index once — used in Cell 11 distillation too
CH6_IDX = []
for _ch in CH6_SEED:
    CH6_IDX.extend([_ch * N_BANDS + b for b in range(N_BANDS)])
CH6_IDX = np.array(CH6_IDX, dtype=np.int64)   # shape (30,)
print(f'  CH6_IDX (30 features): {CH6_IDX}')

print('\nLoading SEED-IV features...')

raw_paths = [
    f'{FEAT_SRC}/seed_iv_prep_s1_zscore_subject.npy',
    f'{FEAT_SRC}/seed_iv_X_62ch.npy',
    f'/kaggle/input/prev-checkpoints/features/preprocessed/seed_iv_prep_s1_zscore_subject.npy',
]
label_paths = [f'{FEAT_SRC}/seed_iv_y_4cls.npy',  f'{FEAT_SRC}/seed_iv_labels.npy']
subj_paths  = [f'{FEAT_SRC}/seed_iv_subjects.npy', f'{FEAT_SRC}/seed_iv_subj.npy']

X_raw = None
for p in raw_paths:
    if os.path.exists(p):
        X_raw = np.load(p).astype(np.float32)
        print(f'  Loaded X from: {p}  shape={X_raw.shape}')
        break
assert X_raw is not None, f'Could not find SEED-IV X features. Checked:\n' + '\n'.join(raw_paths)

y_s, subj_s = None, None
for p in label_paths:
    if os.path.exists(p):
        y_s = np.load(p)
        print(f'  Loaded y from: {p}  shape={y_s.shape}')
        break
for p in subj_paths:
    if os.path.exists(p):
        subj_s = np.load(p)
        print(f'  Loaded subj from: {p}  shape={subj_s.shape}')
        break

assert y_s is not None, f'Labels not found. Checked: {label_paths}'
assert subj_s is not None, f'Subjects not found. Checked: {subj_paths}'

# Force integer labels on CPU before any training step.
y_s = np.asarray(y_s, dtype=np.int64)
subj_s = np.asarray(subj_s, dtype=np.int64)

# Remap subject IDs to 0..N_SUBJ-1 for the subject-adversarial CE head.
subj_unique_original = np.unique(subj_s)
subj_map = {int(s): i for i, s in enumerate(subj_unique_original.tolist())}
subj_s = np.array([subj_map[int(s)] for s in subj_s], dtype=np.int64)

print('  Subject ID remap:')
print(f'    original unique: {subj_unique_original}')
print(f'    remapped unique: {np.unique(subj_s)}')

# Per-subject z-score normalisation (only if not already done)
def zscore_per_subject(X, subj):
    Xn = X.copy()
    for s in np.unique(subj):
        idx = subj == s
        Xn[idx] = StandardScaler().fit_transform(X[idx])
    return Xn.astype(np.float32)

if float(X_raw.std()) > 2.0:
    print('  Applying z-score per subject (features appear un-normalised)...')
    X_raw = zscore_per_subject(X_raw, subj_s)

# Build 6-channel variant
X6_raw   = X_raw[:, CH6_IDX].astype(np.float32)
subjects = np.unique(subj_s)
N_SUBJ   = len(subjects)

assert int(y_s.min()) >= 0 and int(y_s.max()) < N_CLS,         f'y_s out of range for N_CLS={N_CLS}: min={y_s.min()} max={y_s.max()}'
assert int(subj_s.min()) >= 0 and int(subj_s.max()) < N_SUBJ,         f'subj_s out of range for N_SUBJ={N_SUBJ}: min={subj_s.min()} max={subj_s.max()}'
assert X_raw.shape[1] == N_CH * N_BANDS,         f'Expected {N_CH*N_BANDS} input feats, got {X_raw.shape[1]}'
assert X6_raw.shape[1] == len(CH6_IDX),         f'Expected {len(CH6_IDX)} 6-ch feats, got {X6_raw.shape[1]}'

print(f'  X_62ch: {X_raw.shape} | X_6ch: {X6_raw.shape}')
print(f'  Labels: {y_s.shape}  Subjects: {N_SUBJ}')
print(f'  Label range: {int(y_s.min())}..{int(y_s.max())}')
print(f'  Subject range: {int(subj_s.min())}..{int(subj_s.max())}')
print(f'  Class counts: {dict(zip(*np.unique(y_s, return_counts=True)))}')



# ==============================================================================
# Notebook cell 7
# Categories: preprocessing
# ==============================================================================
# ── Cell 5: LOSO Config (with smoke-test overrides) ────────────────────────────
if SMOKE_TEST:
    SEEDS       = [1]
    N_FOLDS     = 2
    N_CAL       = 10
    BATCH_SIZE  = 64
    PRETRAIN_EP = 2
    FINETUNE_EP = 4
    DISTILL_EP  = 3
    PATIENCE    = 4
    print('🔬 SMOKE TEST config active')
else:
    SEEDS       = [1, 7, 21]
    N_FOLDS     = 15
    N_CAL       = 20
    BATCH_SIZE  = 128
    PRETRAIN_EP = 100
    FINETUNE_EP = 100
    DISTILL_EP  = 50
    PATIENCE    = 25

TOTAL_RUNS = N_FOLDS * len(SEEDS)

print(f'✅ Config loaded')
print(f'   SEEDS={SEEDS}  N_FOLDS={N_FOLDS}  TOTAL_RUNS={TOTAL_RUNS}')
print(f'   Pretrain={PRETRAIN_EP}ep  Finetune={FINETUNE_EP}ep  Distill={DISTILL_EP}ep')
print(f'   N_CAL={N_CAL}  Batch={BATCH_SIZE}  Patience={PATIENCE}')


# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ── Cell 6: EEG-GraphFormer Architecture (M27) ─────────────────────────────────
# (B,C,F) → 2-layer band-wise GCN (adaptive adjacency) → DiagMasked Transformer
#           → Band Attention Pooling → Channel Mean Pool → z (B,d)
# + Subject Adversarial Head (GRL) + Contrastive Projector

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, grad): return -ctx.alpha * grad, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0): super().__init__(); self.alpha = alpha
    def forward(self, x): return GradientReversalFn.apply(x, self.alpha)

class AdaptiveAdjacency(nn.Module):
    """Learnable blend of fixed anatomical + data-driven adjacency."""
    def __init__(self, n_ch, embed_dim=8):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_ch, embed_dim) * 0.1)
        self.alpha  = nn.Parameter(torch.tensor(0.5))

    def forward(self, A_fixed):
        e = self.embed
        sim = torch.mm(e, e.T) / (e.shape[1] ** 0.5)
        A_learn = torch.softmax(sim, dim=-1)
        a = torch.sigmoid(self.alpha)
        A = a * A_fixed + (1.0 - a) * A_learn
        D = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return A / D

class BandGCN(nn.Module):
    """2-layer GCN applied per frequency band."""
    def __init__(self, in_f, d_model, dropout=0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_f, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.bn1  = nn.BatchNorm1d(d_model)
        self.bn2  = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def _gcn(self, x, A, lin, bn):
        h = lin(x)                             # (B,C,d)
        h = torch.einsum('ij,bjd->bid', A, h)
        B, C, d = h.shape
        h = bn(h.reshape(B*C, d)).reshape(B, C, d)
        return F.gelu(h)

    def forward(self, x, A):
        h = self._gcn(x, A, self.lin1, self.bn1)
        h = self.drop(h)
        h = self._gcn(h, A, self.lin2, self.bn2)
        return h

class DiagMaskedTransformer(nn.Module):
    """Transformer encoder with local diagonal attention mask over channels."""
    def __init__(self, d_model, n_heads, n_layers, dropout=0.2, bandwidth=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.bandwidth = bandwidth

    def _make_mask(self, n, device):
        mask = torch.ones(n, n, device=device, dtype=torch.bool)
        for i in range(n):
            lo = max(0, i - self.bandwidth)
            hi = min(n, i + self.bandwidth + 1)
            mask[i, lo:hi] = False
        return mask

    def forward(self, x):
        mask = self._make_mask(x.size(1), x.device)
        return self.encoder(x, mask=mask)

class EEGGraphFormer(nn.Module):
    """M27 EEG-GraphFormer Teacher (62-ch)."""
    def __init__(self, n_ch=62, n_bands=5, d_model=64, n_heads=4, n_layers=4,
                 n_classes=4, n_subjects=15, dropout=0.2, proj_dim=64):
        super().__init__()
        self.n_ch    = n_ch
        self.n_bands = n_bands
        self.d_model = d_model

        self.adj_module = AdaptiveAdjacency(n_ch)
        self.gcn        = BandGCN(1, d_model, dropout)
        self.band_embed = nn.Parameter(torch.randn(n_bands, d_model) * 0.1)
        self.transformer= DiagMaskedTransformer(d_model, n_heads, n_layers, dropout)
        self.band_attn  = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        self.projector  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
        self.grl        = GRL(alpha=1.0)
        self.subj_head  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_subjects))
        self.register_buffer('A_fixed', torch.eye(n_ch))

    def _encode(self, x):
        B   = x.size(0)
        x3  = x.view(B, self.n_ch, self.n_bands)
        A   = self.adj_module(self.A_fixed)
        band_outs = []
        for b in range(self.n_bands):
            xb = x3[:, :, b:b+1]
            hb = self.gcn(xb, A) + self.band_embed[b]
            band_outs.append(hb)
        tf_outs = [self.transformer(hb) for hb in band_outs]
        h_tf    = torch.stack(tf_outs, dim=1)              # (B,F,C,d)
        attn_w  = torch.softmax(self.band_attn(h_tf.mean(dim=2)), dim=1)  # (B,F,1)
        h_pool  = (h_tf * attn_w.unsqueeze(-1)).sum(dim=1) # (B,C,d)
        z       = h_pool.mean(dim=1)                       # (B,d)
        return z, h_pool

    def forward(self, x, return_feats=False):
        z, h_pool   = self._encode(x)
        logits      = self.classifier(z)
        subj_logits = self.subj_head(self.grl(z))
        proj        = F.normalize(self.projector(z), dim=-1)
        if return_feats:
            return logits, subj_logits, proj, z, h_pool
        return logits, subj_logits, proj

print('✅ EEGGraphFormer (M27) defined  d_model=64')


# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, model_definition, training, results_tables, audit_verification
# ==============================================================================
# ── Cell 7: GraphStudent Architecture (M28) ────────────────────────────────────
# Distilled 6-channel student with graph sub-structure alignment.
#
# v3 FIX: Added feat_align layer (d_model=32 → teacher_feat_dim=64) so that
# feature-level MSE distillation loss computes on matching dimensions.
# Previously s_z (32-d) vs t_z (64-d) caused a silent shape mismatch / crash.

class GraphStudent(nn.Module):
    """M28 Graph-Distilled Student (6-ch)."""
    def __init__(self, n_ch=6, n_bands=5, d_model=32, n_heads=2, n_layers=2,
                 n_classes=4, dropout=0.2, teacher_feat_dim=64):
        super().__init__()
        self.n_ch    = n_ch
        self.n_bands = n_bands
        self.d_model = d_model

        self.adj_module = AdaptiveAdjacency(n_ch, embed_dim=4)
        self.gcn        = BandGCN(1, d_model, dropout)
        self.band_embed = nn.Parameter(torch.randn(n_bands, d_model) * 0.1)
        self.transformer= DiagMaskedTransformer(d_model, n_heads, n_layers, dropout, bandwidth=2)
        self.band_attn  = nn.Sequential(nn.Linear(d_model, 16), nn.GELU(), nn.Linear(16, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        # ── v3 FIX: alignment projection for feature distillation ─────────────
        # Projects student latent (32-d) → teacher latent dim (64-d) for MSE loss.
        # Only used during distillation training, not at inference time.
        self.feat_align = nn.Linear(d_model, teacher_feat_dim)

        self.register_buffer('A_fixed', torch.eye(n_ch))

    def _encode(self, x):
        B   = x.size(0)
        x3  = x.view(B, self.n_ch, self.n_bands)
        A   = self.adj_module(self.A_fixed)
        band_outs = []
        for b in range(self.n_bands):
            xb = x3[:, :, b:b+1]
            hb = self.gcn(xb, A) + self.band_embed[b]
            band_outs.append(hb)
        tf_outs = [self.transformer(hb) for hb in band_outs]
        h_tf    = torch.stack(tf_outs, dim=1)
        attn_w  = torch.softmax(self.band_attn(h_tf.mean(dim=2)), dim=1)
        h_pool  = (h_tf * attn_w.unsqueeze(-1)).sum(dim=1)
        z       = h_pool.mean(dim=1)
        return z, h_pool

    def forward(self, x, return_feats=False):
        z, h_pool = self._encode(x)
        logits    = self.classifier(z)
        if return_feats:
            return logits, z, h_pool
        return logits

print('✅ GraphStudent (M28) defined  d_model=32  feat_align=32→64')


# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, training, evaluation, results_tables, audit_verification
# ==============================================================================
# ── Cell 8: Training Functions ─────────────────────────────────────────────────

def augment_sample(X, mask_ratio=0.3, noise_std=0.1):
    """Channel masking + Gaussian noise augmentation."""
    Xa = X.clone()
    if mask_ratio > 0:
        Xa[torch.rand_like(Xa) < mask_ratio] = 0.0
    if noise_std > 0:
        Xa = Xa + torch.randn_like(Xa) * noise_std
    return Xa

def nt_xent_augmented(z, temp=0.5):
    """NT-Xent for augmented pairs.
    z: (2B, d) — first B and second B are augmented twins of the same samples.
    Runs in float32 to avoid half-precision overflow on Kaggle GPUs.
    """
    B = z.size(0) // 2
    if B < 1:
        return torch.tensor(0.0, device=z.device, dtype=torch.float32)

    z = F.normalize(z.float(), dim=-1)
    sim = torch.mm(z, z.T) / temp
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

def validate_class_targets_cpu(y_cpu, n_classes, name):
    """Raise on CPU before any CUDA kernel if labels are invalid."""
    if isinstance(y_cpu, torch.Tensor):
        y_np = y_cpu.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_cpu)
    if y_np.size == 0:
        raise ValueError(f'{name}: empty target array')
    if not np.issubdtype(y_np.dtype, np.integer):
        raise TypeError(f'{name}: expected integer targets, got {y_np.dtype}')
    lo, hi = int(y_np.min()), int(y_np.max())
    if lo < 0 or hi >= int(n_classes):
        raise ValueError(f'{name}: target range [{lo}, {hi}] invalid for n_classes={n_classes}')

def pretrain_epoch(model, loader, opt, scaler,
                   contrastive_temp=0.5, mask_ratio=0.3, noise_std=0.1, adv_weight=1.0):
    """Contrastive pretraining with subject adversarial head."""
    model.train()
    total_loss = n = 0
    n_subjects = int(model.subj_head[-1].out_features)

    for X, y, subj_ids in loader:
        validate_class_targets_cpu(y, N_CLS, 'pretrain y')
        validate_class_targets_cpu(subj_ids, n_subjects, 'pretrain subj_ids')

        X = X.to(device)
        y = y.to(device)
        subj_ids = subj_ids.to(device)

        Xa1 = augment_sample(X, mask_ratio, noise_std)
        Xa2 = augment_sample(X, mask_ratio, noise_std)

        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            _, subj1, proj1 = model(Xa1)
            _, subj2, proj2 = model(Xa2)
            loss_cont = nt_xent_augmented(torch.cat([proj1, proj2], dim=0), contrastive_temp)
            loss_adv  = 0.5 * (F.cross_entropy(subj1, subj_ids) +
                               F.cross_entropy(subj2, subj_ids)) * adv_weight
            loss = loss_cont + loss_adv

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += float(loss.item()) * len(y)
        n += len(y)
    return total_loss / max(n, 1)

def finetune_epoch(model, loader, opt, scaler, label_smoothing=0.1):
    """Supervised finetuning (classification head only)."""
    model.train()
    total_loss = n = 0

    for X, y, _ in loader:
        validate_class_targets_cpu(y, N_CLS, 'finetune y')

        X = X.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            logits, _, _ = model(X)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += float(loss.item()) * len(y)
        n += len(y)
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader):
    """Proto-A evaluation (no calibration)."""
    model.eval()
    preds, trues = [], []
    for X, y, _ in loader:
        validate_class_targets_cpu(y, N_CLS, 'eval y')
        with amp_ctx():
            logits, _, _ = model(X.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return (accuracy_score(trues, preds),
            f1_score(trues, preds, average='macro', zero_division=0))

@torch.no_grad()
def calibrate_and_eval(model, X_cal, y_cal, X_ev, y_ev):
    """Proto-B: apply prior-correction using N_CAL samples from the test subject."""
    validate_class_targets_cpu(y_cal, N_CLS, 'calibration y_cal')
    validate_class_targets_cpu(y_ev, N_CLS, 'calibration y_ev')
    model.eval()
    with amp_ctx():
        logits_cal, _, _ = model(torch.from_numpy(X_cal).float().to(device))
    prior = F.softmax(logits_cal.float(), dim=-1).mean(0, keepdim=True)  # (1, n_cls)
    with amp_ctx():
        logits_ev, _, _ = model(torch.from_numpy(X_ev).float().to(device))
    adj = logits_ev.cpu().float() - prior.cpu().log()
    pred = adj.argmax(1).numpy()
    return (accuracy_score(y_ev, pred),
            f1_score(y_ev, pred, average='macro', zero_division=0))

print('✅ Training functions defined')
print(f'   AMP_ENABLED={AMP_ENABLED}')



# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, model_definition, training
# ==============================================================================
# ── Cell 9: Data Loaders ───────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """Returns (X, y, subj_id) tuples — subject IDs used for adversarial loss."""
    def __init__(self, X, y, subj):
        self.X    = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y    = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()
        self.subj = torch.from_numpy(np.asarray(subj, dtype=np.int64)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subj[idx]

def make_loader(X, y, subj, batch_size, shuffle=False, weighted=False, seed=0):
    dataset = EEGDataset(X, y, subj)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    if weighted and shuffle:
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLS).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = (1.0 / counts)[np.asarray(y, dtype=np.int64)]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=(device.type=='cuda'))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=(device.type=='cuda')
    )

def get_loso_split(X, y, subj, test_subj, seed):
    """LOSO split → (X_tr, y_tr, subj_tr, X_cal, y_cal, X_ev, y_ev)."""
    tr_mask = subj != test_subj
    te_mask = subj == test_subj
    X_tr, y_tr, subj_tr = X[tr_mask], y[tr_mask], subj[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    validate_class_targets_cpu(y_tr, N_CLS, 'split y_tr')
    validate_class_targets_cpu(y_te, N_CLS, 'split y_te')
    validate_class_targets_cpu(subj_tr, N_SUBJ, 'split subj_tr')

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(X_te))
    X_cal, y_cal = X_te[idx[:N_CAL]], y_te[idx[:N_CAL]]
    X_ev,  y_ev  = X_te[idx[N_CAL:]], y_te[idx[N_CAL:]]

    validate_class_targets_cpu(y_cal, N_CLS, 'split y_cal')
    validate_class_targets_cpu(y_ev, N_CLS, 'split y_ev')
    return X_tr, y_tr, subj_tr, X_cal, y_cal, X_ev, y_ev

def log_fold_error(model_id, seed, fold, stage, exc):
    os.makedirs(os.path.join(BASE, 'results/graph_former'), exist_ok=True)
    err = {
        'model_id': model_id,
        'seed': int(seed),
        'fold': int(fold),
        'stage': stage,
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    p = os.path.join(BASE, 'results/graph_former', f'{model_id}_seed{seed}_fold{fold:02d}_ERROR.json')
    with open(p, 'w') as f:
        json.dump(err, f, indent=2)
    print(f'  ❌ Logged error → {p}')

print('✅ Data loaders defined')



# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
# ── Cell 10: M27 EEGGraphFormer LOSO Training ──────────────────────────────────
# Saves .pth weights per fold (required by M28 distillation).
# Skips any fold whose JSON checkpoint AND .pth weight file both exist.

m27_results = {}
abort_m27 = False

print('\n── M27 EEGGraphFormer — resume check ──')
done_m27, total_m27 = print_resume_banner('M27_teacher', SEEDS, N_FOLDS)

if done_m27 < total_m27:
    t0 = time.time()
    for seed in SEEDS:
        seed_cpu_only(seed)

        for fi, test_subj in enumerate(subjects[:N_FOLDS]):
            key = ckpt_key('M27_teacher', seed, fi)

            # Resume: skip if JSON result + .pth both present
            if ckpt_complete('M27_teacher', seed, fi):
                r = load_ckpt(key)
                if r and 'acc_b' in r:
                    m27_results[key] = r
                    print(f'  SKIP {key} (AccB={r["acc_b"]:.4f})')
                else:
                    print(f'  ⚠ Skip marker present but result incomplete for {key}')
                continue

            t_fold = time.time()
            try:
                X_tr, y_tr, subj_tr, X_cal, y_cal, X_ev, y_ev = get_loso_split(
                    X_raw, y_s, subj_s, test_subj, seed)

                tr_loader = make_loader(X_tr, y_tr, subj_tr, BATCH_SIZE, shuffle=True, weighted=True, seed=seed * 100 + fi)
                ev_loader = make_loader(X_ev, y_ev, np.zeros(len(y_ev), dtype=np.int64), 512, seed=seed * 100 + fi)

                model  = EEGGraphFormer(n_ch=N_CH, n_bands=N_BANDS, n_subjects=N_SUBJ).to(device)
                scaler = make_scaler()

                # Stage 1: Contrastive Pretraining
                opt_pre = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sch_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, PRETRAIN_EP)
                for ep in range(PRETRAIN_EP):
                    loss = pretrain_epoch(model, tr_loader, opt_pre, scaler)
                    sch_pre.step()
                    if (ep + 1) % max(1, PRETRAIN_EP // 5) == 0:
                        print(f'  [{key}] Pretrain {ep+1}/{PRETRAIN_EP} | loss={loss:.4f}')

                # Stage 2: Supervised Finetuning
                for p in model.projector.parameters():
                    p.requires_grad_(False)

                opt_ft = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=3e-4, weight_decay=1e-4)

                sch_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, FINETUNE_EP)
                best_f1, patience_cnt, best_state = 0.0, 0, None
                eval_every = max(1, FINETUNE_EP // 20)   # ~20 eval checkpoints total

                for ep in range(FINETUNE_EP):
                    finetune_epoch(model, tr_loader, opt_ft, scaler)
                    sch_ft.step()

                    if (ep + 1) % eval_every == 0 or ep == FINETUNE_EP - 1:
                        _, vf1 = evaluate(model, ev_loader)
                        if vf1 > best_f1:
                            best_f1 = vf1
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1

                        if (ep + 1) % max(1, FINETUNE_EP // 5) == 0:
                            print(f'  [{key}] Finetune {ep+1}/{FINETUNE_EP}'
                                  f' | val_f1={vf1:.4f} | best={best_f1:.4f}')

                        # Early stopping: patience counted in eval-check units
                        if patience_cnt >= max(1, PATIENCE // eval_every):
                            print(f'  [{key}] Early stop at ep {ep+1}')
                            break

                # Save best weights (required for M28 distillation)
                if best_state:
                    model.load_state_dict(best_state)

                torch.save(
                    {'state_dict': model.state_dict(),
                     'seed': seed, 'fold': fi, 'test_subj': int(test_subj)},
                    teacher_pth(seed, fi)
                )

                # Evaluate
                acc_a, f1_a = evaluate(model, ev_loader)
                acc_b, f1_b = calibrate_and_eval(model, X_cal, y_cal, X_ev, y_ev)

                result = {'acc_a': acc_a, 'f1_a': f1_a,
                          'acc_b': acc_b, 'f1_b': f1_b,
                          'seed': seed, 'fold': fi, 'test_subj': int(test_subj),
                          'elapsed': time.time() - t_fold}
                save_ckpt(key, result)
                m27_results[key] = result

                print(f'  ✅ {key} | AccA={acc_a:.4f} AccB={acc_b:.4f} F1B={f1_b:.4f}'
                      f' | {result["elapsed"]:.0f}s')

            except Exception as e:
                log_fold_error('M27_teacher', seed, fi, 'train_or_eval', e)
                print(f'  🛑 Aborting M27 after error in {key}: {type(e).__name__}: {e}')
                abort_m27 = True
                break

        if abort_m27:
            break

    print(f'\n✅ M27 stage finished in {(time.time()-t0)/60:.1f} min')
else:
    for seed in SEEDS:
        for fi in range(N_FOLDS):
            key = ckpt_key('M27_teacher', seed, fi)
            r = load_ckpt(key)
            if r and 'acc_b' in r:
                m27_results[key] = r

if m27_results:
    accb = [r['acc_b'] for r in m27_results.values()]
    f1b  = [r['f1_b']  for r in m27_results.values()]
    print(f'\nM27 Summary (n={len(accb)}): '
          f'AccB={np.mean(accb):.4f}±{np.std(accb):.4f} '
          f'| F1B={np.mean(f1b):.4f}±{np.std(f1b):.4f}')
else:
    print('\n⚠ No completed M27 folds yet.')



# ==============================================================================
# Notebook cell 13
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification
# ==============================================================================
# ── Cell 11: M28 Graph-Distilled Student LOSO ──────────────────────────────────
# Distills from M27 teacher weights saved in Cell 10.
#
# v3 FIX: CH6_IDX is passed explicitly to distill_epoch / evaluate_student
# v4 STABILITY: AMP disabled by default, CPU-side target validation, graceful error logs

def distill_epoch(student, teacher, loader, opt, scaler, ch6_idx_t, temp=4.0):
    """Knowledge distillation: aligned-MSE (features) + KL (logits) + CE (labels)."""
    student.train()
    teacher.eval()
    total_loss = n = 0

    for X62, y, _ in loader:
        validate_class_targets_cpu(y, N_CLS, 'distill y')

        # Extract 6-ch subset before moving to GPU
        X6  = X62[:, ch6_idx_t].to(device)
        X62 = X62.to(device)
        y   = y.to(device)

        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            with torch.no_grad():
                t_logits, _, _, t_z, _ = teacher(X62, return_feats=True)
            s_logits, s_z, _ = student(X6, return_feats=True)

            # Feature distillation (aligned: 32 → 64 via feat_align)
            loss_mse = F.mse_loss(student.feat_align(s_z), t_z.detach())

            # Logit distillation (KL with temperature scaling)
            loss_kl = F.kl_div(
                F.log_softmax(s_logits / temp, dim=1),
                F.softmax(t_logits.detach() / temp, dim=1),
                reduction='batchmean') * (temp ** 2)

            # Hard-label CE
            loss_ce = F.cross_entropy(s_logits, y)

            loss = loss_mse + 2.0 * loss_kl + loss_ce

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += float(loss.item()) * len(y)
        n += len(y)
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate_student(model, loader, ch6_idx_t):
    """Evaluate student (proto-A) — extracts 6-ch from 62-ch batch."""
    model.eval()
    preds, trues = [], []
    for X62, y, _ in loader:
        validate_class_targets_cpu(y, N_CLS, 'student eval y')
        X6 = X62[:, ch6_idx_t].to(device)
        with amp_ctx():
            logits = model(X6)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return (accuracy_score(trues, preds),
            f1_score(trues, preds, average='macro', zero_division=0))

@torch.no_grad()
def calibrate_and_eval_student(model, X_cal, y_cal, X_ev, y_ev, ch6_idx_np):
    """Proto-B for student."""
    validate_class_targets_cpu(y_cal, N_CLS, 'student y_cal')
    validate_class_targets_cpu(y_ev, N_CLS, 'student y_ev')
    model.eval()
    X6_cal = X_cal[:, ch6_idx_np]
    X6_ev  = X_ev[:,  ch6_idx_np]

    with amp_ctx():
        logits_cal = model(torch.from_numpy(X6_cal).float().to(device))
    prior = F.softmax(logits_cal.float(), dim=-1).mean(0, keepdim=True)

    with amp_ctx():
        logits_ev = model(torch.from_numpy(X6_ev).float().to(device))
    adj = logits_ev.cpu().float() - prior.cpu().log()
    pred = adj.argmax(1).numpy()
    return (accuracy_score(y_ev, pred),
            f1_score(y_ev, pred, average='macro', zero_division=0))

# Prepare CH6_IDX as a torch tensor for in-batch indexing
CH6_IDX_T = torch.from_numpy(CH6_IDX).long()   # CPU LongTensor for indexing

m28_results = {}
abort_m28 = False

print('\n── M28 GraphStudent — resume check ──')
done_m28, total_m28 = print_resume_banner('M28_student', SEEDS, N_FOLDS)

if done_m28 < total_m28:
    t0 = time.time()
    for seed in SEEDS:
        seed_cpu_only(seed + 999)

        for fi, test_subj in enumerate(subjects[:N_FOLDS]):
            key = ckpt_key('M28_student', seed, fi)

            if ckpt_complete('M28_student', seed, fi):
                r = load_ckpt(key)
                if r and 'acc_b' in r:
                    m28_results[key] = r
                    print(f'  SKIP {key} (AccB={r["acc_b"]:.4f})')
                else:
                    print(f'  ⚠ Skip marker present but result incomplete for {key}')
                continue

            teacher_path = teacher_pth(seed, fi)
            if not os.path.exists(teacher_path):
                print(f'  ⚠ Teacher weights missing for {key}: {teacher_path}')
                continue

            t_fold = time.time()
            try:
                X_tr, y_tr, subj_tr, X_cal, y_cal, X_ev, y_ev = get_loso_split(
                    X_raw, y_s, subj_s, test_subj, seed)

                tr_loader = make_loader(X_tr, y_tr, subj_tr, BATCH_SIZE, shuffle=True, weighted=True, seed=seed * 100 + fi)
                ev_loader = make_loader(X_ev, y_ev, np.zeros(len(y_ev), dtype=np.int64), 512, seed=seed * 100 + fi)

                teacher = EEGGraphFormer(n_ch=N_CH, n_bands=N_BANDS, n_subjects=N_SUBJ).to(device)
                ckpt = torch.load(teacher_path, map_location=device)
                teacher.load_state_dict(ckpt['state_dict'])
                teacher.eval()

                student = GraphStudent(n_ch=len(CH6_SEED), n_bands=N_BANDS, teacher_feat_dim=64).to(device)
                scaler = make_scaler()

                opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DISTILL_EP)

                best_f1, patience_cnt, best_state = 0.0, 0, None
                eval_every = max(1, DISTILL_EP // 10)

                for ep in range(DISTILL_EP):
                    loss = distill_epoch(student, teacher, tr_loader, opt, scaler, CH6_IDX_T)
                    sch.step()

                    if (ep + 1) % eval_every == 0 or ep == DISTILL_EP - 1:
                        _, vf1 = evaluate_student(student, ev_loader, CH6_IDX_T)
                        if vf1 > best_f1:
                            best_f1 = vf1
                            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1

                        if (ep + 1) % max(1, DISTILL_EP // 5) == 0:
                            print(f'  [{key}] Distill {ep+1}/{DISTILL_EP} | loss={loss:.4f} | val_f1={vf1:.4f} | best={best_f1:.4f}')

                        if patience_cnt >= max(1, PATIENCE // eval_every):
                            print(f'  [{key}] Early stop at ep {ep+1}')
                            break

                if best_state:
                    student.load_state_dict(best_state)

                acc_a, f1_a = evaluate_student(student, ev_loader, CH6_IDX_T)
                acc_b, f1_b = calibrate_and_eval_student(student, X_cal, y_cal, X_ev, y_ev, CH6_IDX)

                result = {'acc_a': acc_a, 'f1_a': f1_a,
                          'acc_b': acc_b, 'f1_b': f1_b,
                          'seed': seed, 'fold': fi, 'test_subj': int(test_subj),
                          'elapsed': time.time() - t_fold}
                save_ckpt(key, result)
                m28_results[key] = result

                print(f'  ✅ {key} | AccA={acc_a:.4f} AccB={acc_b:.4f} F1B={f1_b:.4f}'
                      f' | {result["elapsed"]:.0f}s')

            except Exception as e:
                log_fold_error('M28_student', seed, fi, 'distill_or_eval', e)
                print(f'  🛑 Aborting M28 after error in {key}: {type(e).__name__}: {e}')
                abort_m28 = True
                break

        if abort_m28:
            break

    print(f'\n✅ M28 stage finished in {(time.time()-t0)/60:.1f} min')
else:
    for seed in SEEDS:
        for fi in range(N_FOLDS):
            key = ckpt_key('M28_student', seed, fi)
            r = load_ckpt(key)
            if r and 'acc_b' in r:
                m28_results[key] = r

if m28_results:
    accb = [r['acc_b'] for r in m28_results.values()]
    f1b  = [r['f1_b']  for r in m28_results.values()]
    print(f'\nM28 Summary (n={len(accb)}): '
          f'AccB={np.mean(accb):.4f}±{np.std(accb):.4f} '
          f'| F1B={np.mean(f1b):.4f}±{np.std(f1b):.4f}')
else:
    print('\n⚠ No completed M28 folds yet.')



# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ── Cell 12: Session Status Summary ────────────────────────────────────────────
m27_ckpts = len([f for f in os.listdir(CKPT_DIR)    if f.startswith('M27_teacher') and f.endswith('.json')])
m27_pths  = len([f for f in os.listdir(WEIGHTS_DIR) if f.startswith('M27_teacher') and f.endswith('.pth')])
m28_ckpts = len([f for f in os.listdir(CKPT_DIR)    if f.startswith('M28_student') and f.endswith('.json')])
err_logs  = len([f for f in os.listdir(os.path.join(BASE, 'results/graph_former')) if f.endswith('_ERROR.json')]) if os.path.exists(os.path.join(BASE, 'results/graph_former')) else 0

TOTAL_CKPTS = TOTAL_RUNS

print()
print('=' * 62)
mode_tag = '[SMOKE TEST]' if SMOKE_TEST else '[FULL RUN]'
print(f'SESSION STATUS  {time.strftime("%Y-%m-%d %H:%M")}  {mode_tag}')
print('=' * 62)
print(f'M27 EEGGraphFormer  : {m27_ckpts}/{TOTAL_CKPTS} checkpoints')
print(f'M27 .pth weights    : {m27_pths}/{TOTAL_CKPTS}')
print(f'M28 GraphStudent    : {m28_ckpts}/{TOTAL_CKPTS} checkpoints')
print(f'Error logs          : {err_logs}')
print()
print(f'Output dir          : {os.path.join(BASE, "results/graph_former")}')
all_done = (m27_ckpts == TOTAL_CKPTS and m27_pths == TOTAL_CKPTS and m28_ckpts == TOTAL_CKPTS)
if all_done:
    print('✅ ALL COMPLETE — proceed to Cell 13 for final results.')
else:
    remaining_m27 = TOTAL_CKPTS - m27_ckpts
    remaining_m28 = TOTAL_CKPTS - m28_ckpts
    if remaining_m27 > 0:
        print(f'⏳ M27: {remaining_m27} folds remaining — re-run Cell 10 to resume.')
    if remaining_m28 > 0:
        print(f'⏳ M28: {remaining_m28} folds remaining — re-run Cell 11 to resume.')
    if err_logs > 0:
        print('⚠ Some folds logged errors; inspect results/graph_former/*_ERROR.json')
print('=' * 62)



# ==============================================================================
# Notebook cell 15
# Categories: preprocessing, model_definition, results_tables
# ==============================================================================
# ── Cell 13: Final Results Report ──────────────────────────────────────────────

def aggregate_results(results_dict, model_name):
    if not results_dict:
        print(f'  ⚠ No results for {model_name}')
        return None
    acc_a = [r['acc_a'] for r in results_dict.values()]
    f1_a  = [r['f1_a']  for r in results_dict.values()]
    acc_b = [r['acc_b'] for r in results_dict.values()]
    f1_b  = [r['f1_b']  for r in results_dict.values()]
    print(f'\n── {model_name} (n={len(acc_a)}) ──')
    print(f'  acc_a : {np.mean(acc_a):.4f} ± {np.std(acc_a):.4f}')
    print(f'  f1_a  : {np.mean(f1_a):.4f} ± {np.std(f1_a):.4f}')
    print(f'  acc_b : {np.mean(acc_b):.4f} ± {np.std(acc_b):.4f}')
    print(f'  f1_b  : {np.mean(f1_b):.4f} ± {np.std(f1_b):.4f}')
    for seed in SEEDS:
        sr = {k: v for k, v in results_dict.items() if v['seed'] == seed}
        if sr:
            print(f'    seed={seed}: AccB={np.mean([r["acc_b"] for r in sr.values()]):.4f}'
                  f'  F1={np.mean([r["f1_b"] for r in sr.values()]):.4f}')
    return {'model': model_name,
            'acc_a_mean': float(np.mean(acc_a)), 'acc_a_std': float(np.std(acc_a)),
            'f1_a_mean':  float(np.mean(f1_a)),  'f1_a_std':  float(np.std(f1_a)),
            'acc_b_mean': float(np.mean(acc_b)), 'acc_b_std': float(np.std(acc_b)),
            'f1_b_mean':  float(np.mean(f1_b)),  'f1_b_std':  float(np.std(f1_b)),
            'n_runs': len(acc_a)}

m27_summary = aggregate_results(m27_results, 'M27_EEGGraphFormer_SEED-IV')
m28_summary = aggregate_results(m28_results, 'M28_GraphStudent_SEED-IV')

if not SMOKE_TEST and m27_summary:
    print('\n── Comparison vs Baselines ──')
    baselines = [
        ('M20 CLISA',          0.6801),
        ('M19 DANN',           0.6619),
        ('M25 DANCE Teacher',  0.5190),
        ('M09 XGBoost F1',     0.4791),
        ('Chance (4-class)',   0.2500),
    ]
    for name, val in baselines:
        print(f'  {name:<22}: {val:.4f}')
    d = m27_summary['acc_b_mean']
    print(f'  M27 AccB               : {d:.4f}  '
          f'Δ CLISA {d-0.6801:+.4f}  Δ DANCE {d-0.5190:+.4f}')

results_dir = os.path.join(BASE, 'results/graph_former')
os.makedirs(results_dir, exist_ok=True)
suffix = '_smoke' if SMOKE_TEST else ''

if m27_summary and m27_results:
    pd.DataFrame([m27_summary]).to_csv(
        os.path.join(results_dir, f'M27_teacher_summary{suffix}.csv'), index=False)
    pd.DataFrame([{**v, 'key': k} for k, v in m27_results.items()]).to_csv(
        os.path.join(results_dir, f'M27_teacher_loso_results{suffix}.csv'), index=False)

if m28_summary and m28_results:
    pd.DataFrame([m28_summary]).to_csv(
        os.path.join(results_dir, f'M28_student_summary{suffix}.csv'), index=False)
    pd.DataFrame([{**v, 'key': k} for k, v in m28_results.items()]).to_csv(
        os.path.join(results_dir, f'M28_student_loso_results{suffix}.csv'), index=False)

print(f'\n✅ Results saved to {results_dir}/')
if m27_summary and m27_results:
    print(f'   M27_teacher_summary{suffix}.csv')
    print(f'   M27_teacher_loso_results{suffix}.csv')
if m28_summary and m28_results:
    print(f'   M28_student_summary{suffix}.csv')
    print(f'   M28_student_loso_results{suffix}.csv')

err_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_ERROR.json')])
if err_files:
    print('\n⚠ Error logs:')
    for f in err_files:
        print(f'   {f}')

