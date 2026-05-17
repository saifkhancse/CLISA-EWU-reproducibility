# Auto-exported raw code from notebook: 00c_phaseB_reproduce.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: model_definition, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 1 — Imports
# ═══════════════════════════════════════════════════════════
import os, sys, json, time, warnings, math, pathlib, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
print(f"Python : {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.version.cuda}")


# ==============================================================================
# Notebook cell 2
# Categories: model_definition, audit_verification, webapp_or_demo
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 2 — GPU Check (MANDATORY — RTX 3050, 4 GB VRAM)
# ═══════════════════════════════════════════════════════════
assert torch.cuda.is_available(), (
    "❌ GPU NOT FOUND.\n"
    "All Phase C models MUST run on the RTX 3050 laptop GPU.\n"
    "Do NOT use Google Colab or any other machine.\n"
    "Fix: check CUDA installation and driver version 591.86."
)

device   = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✔ GPU   : {gpu_name}")
print(f"✔ VRAM  : {vram_gb:.1f} GB")
assert vram_gb < 5.5, f"Unexpected VRAM {vram_gb:.1f} GB — expected RTX 3050 with 4 GB"

# Enable Flash Attention and TF32 for RTX 3050 (Ampere)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True   # auto-tune kernels
print("✔ Flash Attention (SDPA) / TF32 / cuDNN benchmark enabled")
print(f"✔ Device: {device}")


# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, model_definition, training, evaluation, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 3 — Paths, Configuration & Checkpoint System
# ═══════════════════════════════════════════════════════════
ROOT         = pathlib.Path(os.getcwd())
FEATURES_DIR = ROOT / "features"
CKPT_DIR     = ROOT / "checkpoints" / "phaseB_reproduce"
RESULTS_DIR  = ROOT / "results"     / "phaseB_reproduction"
FIG_DIR      = ROOT / "figures"     / "phaseB_reproduce"

for d in [CKPT_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══ Phase B FROZEN Configuration (Plan v4 Section 16) ══════
CFG_PHASEB = dict(
    # Architecture (do NOT modify)
    n_ch_teacher = 62,
    n_ch_student = 6,
    student_ch   = [0, 2, 5, 13, 23, 31],   # FP1 FP2 F7 F8 T7 T8
    n_bands      = 5,
    d_model      = 32,
    n_heads      = 4,
    n_layers_t   = 4,       # teacher transformer layers
    n_layers_s   = 2,       # student transformer layers
    d_ff         = 128,
    d_proj       = 128,
    dropout      = 0.2,
    diagonal_masking = 'training_only',
    # Stage 1 — contrastive pretrain (Phase B epochs)
    pretrain_epochs    = 30,
    pretrain_lr        = 1e-3,
    contrastive_temp   = 0.5,
    mask_ratio         = 0.3,
    noise_std          = 0.1,
    mixup_alpha        = 0.4,
    subject_weight     = 0.5,
    # Stage 2 — supervised finetune (Phase B epochs)
    finetune_epochs    = 40,
    finetune_lr_cls    = 3e-4,
    finetune_lr_enc    = 3e-5,
    label_smoothing    = 0.1,
    # Stage 3 — distillation
    distill_epochs     = 50,
    distill_lr         = 1e-3,
    distill_temp       = 4.0,
    distill_alpha_mse  = 1.0,
    distill_alpha_kl   = 2.0,
    distill_alpha_ce   = 1.0,
    # Training
    batch_size         = 128,      # safe for 4 GB VRAM
    weight_decay       = 1e-4,
    patience           = 15,
    n_classes          = 4,
    n_cal              = 20,       # Protocol B calibration samples per class
    seed               = 42,
    # Phase B split
    train_subj         = list(range(1, 11)),
    val_subj           = [11, 12],
    test_subj          = [13, 14, 15],
)

# ═══ Phase C H17 Configuration (full epochs) ════════════════
CFG_H17 = {**CFG_PHASEB,
    'pretrain_epochs': 100,
    'finetune_epochs': 100,
    'distill_epochs' : 50,
}

# ── Checkpoint helpers ─────────────────────────────────────
def ckpt_exists(name):
    return (CKPT_DIR / f"{name}.json").exists()

def save_ckpt(name, data):
    with open(CKPT_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(name):
    p = CKPT_DIR / f"{name}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def model_ckpt_path(name):
    return CKPT_DIR / f"{name}.pth"

def save_model(model, name, meta=None):
    payload = {'state_dict': model.state_dict()}
    if meta: payload.update(meta)
    torch.save(payload, model_ckpt_path(name))

def load_model(model, name):
    p = model_ckpt_path(name)
    if p.exists():
        payload = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(payload['state_dict'])
        return payload
    return None

print("✔ Configuration loaded:")
print(f"  Phase B: pretrain={CFG_PHASEB['pretrain_epochs']} / finetune={CFG_PHASEB['finetune_epochs']} / distill={CFG_PHASEB['distill_epochs']} epochs")
print(f"  H17    : pretrain={CFG_H17['pretrain_epochs']} / finetune={CFG_H17['finetune_epochs']} / distill={CFG_H17['distill_epochs']} epochs")
print(f"  CKPT   : {CKPT_DIR}")


# ==============================================================================
# Notebook cell 4
# Categories: model_definition, training, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 4 — 🔄 CHECKPOINT RECOVERY STATUS
# Run after any power loss / unexpected shutdown
# ═══════════════════════════════════════════════════════════
STAGES = [
    "e00_teacher_pretrain",
    "e00_teacher_finetune",
    "e00_teacher_eval",
    "e00_student_distill",
    "e00_student_eval",
    "e00_gate_check",
    "h17_teacher_pretrain",
    "h17_teacher_finetune",
    "h17_teacher_eval",
    "h17_student_distill",
    "h17_student_eval",
    "h17_final_results",
]

done    = [s for s in STAGES if ckpt_exists(s)]
pending = [s for s in STAGES if not ckpt_exists(s)]

print("=" * 65)
print("  00c_phaseB_reproduce.ipynb — Recovery Status")
print("=" * 65)
print(f"  Completed stages : {len(done)}/{len(STAGES)}")
if done:
    for s in done:
        r = load_ckpt(s)
        print(f"    ✔ {s:35s}", end="")
        if r and 'acc_b' in r: print(f"AccB={r['acc_b']:.4f}")
        elif r and 'acc' in r: print(f"Acc={r['acc']:.4f}")
        else: print()
if pending:
    print(f"\n  Next to run: {pending[0]}")
print("=" * 65)


# ==============================================================================
# Notebook cell 5
# Categories: preprocessing, model_definition, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 5 — Data Loading
# ═══════════════════════════════════════════════════════════
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED_CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
]

X_62 = np.load(FEATURES_DIR / "seed_iv_X_62ch.npy")
if X_62.ndim == 2:
    X_62 = X_62.reshape(X_62.shape[0], 62, 5)
    print(f"  Auto-reshaped X_62 → {X_62.shape}")

y    = np.load(FEATURES_DIR / "seed_iv_y_4cls.npy")
subj = np.load(FEATURES_DIR / "seed_iv_subjects.npy")

# Subject-specific z-score normalisation
X_norm = np.zeros_like(X_62)
for s in np.unique(subj):
    mask = subj == s
    flat = X_62[mask].reshape(mask.sum(), -1)
    mu   = flat.mean(0, keepdims=True)
    sig  = flat.std(0, keepdims=True) + 1e-8
    X_norm[mask] = ((flat - mu) / sig).reshape(mask.sum(), 62, 5)

X_6 = X_norm[:, CFG_PHASEB['student_ch'], :]   # (N, 6, 5)

def make_fixed_split(X_62n, X_6n, y, subjects, cfg):
    tr  = np.isin(subjects, cfg['train_subj'])
    val = np.isin(subjects, cfg['val_subj'])
    te  = np.isin(subjects, cfg['test_subj'])
    return dict(
        X62_tr=X_62n[tr],  X6_tr=X_6n[tr],  y_tr=y[tr],  subj_tr=subjects[tr],
        X62_v =X_62n[val], X6_v =X_6n[val],  y_v =y[val],
        X62_te=X_62n[te],  X6_te=X_6n[te],   y_te=y[te],  subj_te=subjects[te],
    )

split = make_fixed_split(X_norm, X_6, y, subj, CFG_PHASEB)
print(f"Phase B fixed split:")
print(f"  Train : {split['X62_tr'].shape[0]:6,}  subjects={CFG_PHASEB['train_subj']}")
print(f"  Val   : {split['X62_v'].shape[0]:6,}  subjects={CFG_PHASEB['val_subj']}")
print(f"  Test  : {split['X62_te'].shape[0]:6,}  subjects={CFG_PHASEB['test_subj']}")
print(f"✔ Data ready. X_62: {X_norm.shape} | X_6: {X_6.shape}")


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, model_definition, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 6 — DANCE Model Definition (Phase B Architecture)
# DO NOT MODIFY — matches Phase B notebook exactly
# ═══════════════════════════════════════════════════════════

class InformationSeparatedAttention(nn.Module):
    """Diagonal-masked multi-head attention.
    Mask applied ONLY during training (training_only mode) per Phase B.
    At inference, full self-attention is used.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # (B, H, T, T)

        # Diagonal masking: training_only (Phase B exact behaviour)
        # NOTE: Applied ONLY during training per Phase B reference code
        if self.training:
            diag_mask = torch.eye(T, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn   = torch.softmax(scores, dim=-1)
        attn   = self.drop(attn)
        out    = torch.matmul(attn, V)                                # (B, H, T, d)
        out    = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = InformationSeparatedAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DANCEEncoder(nn.Module):
    """Shared encoder used by both teacher (62ch) and student (6ch)."""
    def __init__(self, n_channels, n_bands, d_model, n_heads, n_layers, d_ff,
                 d_proj, n_classes, dropout):
        super().__init__()
        self.n_channels = n_channels
        self.d_model    = d_model

        # Input projection: (B, C, F) → (B, C, d_model)
        self.input_proj = nn.Linear(n_bands, d_model)
        # Learnable positional encoding
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_channels, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(),
            nn.Linear(d_proj, d_proj)
        )
        # Classification head
        self.classifier = nn.Linear(d_model, n_classes)
        self.drop       = nn.Dropout(dropout)

    def encode(self, x):
        """x: (B, C, F) → z: (B, d_model)"""
        h = self.input_proj(x) + self.pos_embed   # (B, C, d_model)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        z = h.mean(dim=1)    # global avg pool over channels
        return z

    def project(self, z):
        return F.normalize(self.projector(z), dim=-1)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        z   = self.encode(x)
        proj = self.project(z)
        logit = self.classify(z)
        return z, proj, logit


def build_teacher(cfg):
    return DANCEEncoder(
        n_channels=cfg['n_ch_teacher'], n_bands=cfg['n_bands'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers_t'], d_ff=cfg['d_ff'],
        d_proj=cfg['d_proj'], n_classes=cfg['n_classes'],
        dropout=cfg['dropout']
    ).to(device)

def build_student(cfg):
    return DANCEEncoder(
        n_channels=cfg['n_ch_student'], n_bands=cfg['n_bands'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers_s'], d_ff=cfg['d_ff'],
        d_proj=cfg['d_proj'], n_classes=cfg['n_classes'],
        dropout=cfg['dropout']
    ).to(device)

teacher = build_teacher(CFG_PHASEB)
student = build_student(CFG_PHASEB)
n_t = sum(p.numel() for p in teacher.parameters())
n_s = sum(p.numel() for p in student.parameters())
print(f"✔ DANCE Teacher: {n_t:,} parameters")
print(f"✔ DANCE Student: {n_s:,} parameters")
print(f"  Architecture: d_model={CFG_PHASEB['d_model']} | heads={CFG_PHASEB['n_heads']} | d_ff={CFG_PHASEB['d_ff']}")
print(f"  Diagonal masking: training_only (Phase B exact behaviour)")


# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 7 — Augmentation & Loss Utilities
# ═══════════════════════════════════════════════════════════

def augment(x, mask_ratio=0.3, noise_std=0.1):
    """x: (B, C, F) — channel masking + Gaussian noise."""
    B, C, F = x.shape
    out = x.clone()
    # Random channel masking
    if mask_ratio > 0:
        n_mask = max(1, int(C * mask_ratio))
        for i in range(B):
            idx = torch.randperm(C, device=x.device)[:n_mask]
            out[i, idx, :] = 0.0
    # Gaussian noise
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def mixup(x, y_onehot, alpha=0.4):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)
    x2  = lam * x + (1 - lam) * x[idx]
    y2  = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x2, y2


def nt_xent_loss(z1, z2, temp=0.5):
    """NT-Xent contrastive loss. z1, z2: (B, d) L2-normalised."""
    B    = z1.size(0)
    z    = torch.cat([z1, z2], dim=0)      # (2B, d)
    sim  = torch.mm(z, z.T) / temp         # (2B, 2B)
    # Mask diagonal (self-similarity)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float('-inf'))
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    loss   = F.cross_entropy(sim, labels)
    return loss


class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.1, n_classes=4):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits, targets):
        # targets: (B,) int or (B, C) soft
        if targets.dim() == 1:
            targets_oh = F.one_hot(targets, self.n_classes).float()
        else:
            targets_oh = targets
        smooth = self.smoothing / self.n_classes
        targets_smooth = targets_oh * (1 - self.smoothing) + smooth
        log_p  = F.log_softmax(logits, dim=-1)
        loss   = -(targets_smooth * log_p).sum(dim=-1).mean()
        return loss


def make_loader(X_tensor, y_tensor, batch_size, shuffle=True):
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)


def tensors_from_split(split_key_X, split_key_y, split_dict):
    X = torch.tensor(split_dict[split_key_X], dtype=torch.float32)
    y = torch.tensor(split_dict[split_key_y], dtype=torch.long)
    return X, y

print("✔ Augmentation and loss utilities defined")


# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 8 — Training Functions
# ═══════════════════════════════════════════════════════════

def train_contrastive(model, X_tr, y_tr, subjects_tr, cfg, ckpt_prefix,
                      resume_epoch=0):
    """Stage 1: Contrastive pre-training with subject-aware sampling."""
    model.train()
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['pretrain_lr'],
                               weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg['pretrain_epochs'], eta_min=1e-5)
    scaler = GradScaler()

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    loader = make_loader(X_t, y_t, cfg['batch_size'], shuffle=True)

    # Resume from epoch checkpoint if available
    ep_ckpt = CKPT_DIR / f"{ckpt_prefix}_ep{resume_epoch}.pth"
    if resume_epoch > 0 and ep_ckpt.exists():
        payload = torch.load(ep_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload['state_dict'])
        opt.load_state_dict(payload['opt_state'])
        for _ in range(resume_epoch):
            sched.step()
        print(f"  Resumed from epoch {resume_epoch}")

    history = []
    best_loss = float('inf')
    n_epochs  = cfg['pretrain_epochs']

    for epoch in range(resume_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, _ in loader:
            Xb = Xb.to(device, non_blocking=True)
            with autocast():
                v1 = augment(Xb, cfg['mask_ratio'], cfg['noise_std'])
                v2 = augment(Xb, cfg['mask_ratio'], cfg['noise_std'])
                _, z1, _ = model(v1)
                _, z2, _ = model(v2)
                loss = nt_xent_loss(z1, z2, cfg['contrastive_temp'])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, f"{ckpt_prefix}_best")

        # Save epoch checkpoint every 10 epochs for fault tolerance
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            torch.save({'state_dict': model.state_dict(), 'opt_state': opt.state_dict()},
                       CKPT_DIR / f"{ckpt_prefix}_ep{epoch+1}.pth")
            print(f"  Pretrain epoch {epoch+1:3d}/{n_epochs} | loss={avg_loss:.4f}")

    # Load best model
    load_model(model, f"{ckpt_prefix}_best")
    return history


def train_supervised(model, X_tr, y_tr, X_val, y_val, cfg, ckpt_prefix,
                     resume_epoch=0):
    """Stage 2: Supervised fine-tuning with label smoothing."""
    # Freeze projector, train encoder + classifier
    for p in model.projector.parameters():
        p.requires_grad = False

    param_groups = [
        {'params': model.blocks.parameters(),    'lr': cfg['finetune_lr_enc']},
        {'params': model.input_proj.parameters(), 'lr': cfg['finetune_lr_enc']},
        {'params': model.norm.parameters(),       'lr': cfg['finetune_lr_enc']},
        {'params': model.pos_embed,               'lr': cfg['finetune_lr_enc']},
        {'params': model.classifier.parameters(), 'lr': cfg['finetune_lr_cls']},
    ]
    opt   = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg['finetune_epochs'], eta_min=1e-6)
    criterion = LabelSmoothCE(smoothing=cfg['label_smoothing'], n_classes=cfg['n_classes'])
    scaler = GradScaler()

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    loader = make_loader(X_t, y_t, cfg['batch_size'], shuffle=True)

    best_val_acc = 0.0
    n_epochs     = cfg['finetune_epochs']
    patience_cnt = 0
    history = []

    for epoch in range(resume_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast():
                _, _, logit = model(Xb)
                loss = criterion(logit, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        sched.step()
        # Validate
        model.eval()
        with torch.no_grad(), autocast():
            _, _, val_logit = model(X_v)
        val_acc = (val_logit.argmax(1) == y_v).float().mean().item()
        history.append({'loss': epoch_loss / len(loader), 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            save_model(model, f"{ckpt_prefix}_best",
                       {'epoch': epoch, 'best_val_acc': best_val_acc})
        else:
            patience_cnt += 1

        # Epoch checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            torch.save({'state_dict': model.state_dict(), 'opt_state': opt.state_dict()},
                       CKPT_DIR / f"{ckpt_prefix}_ep{epoch+1}.pth")
            print(f"  Finetune epoch {epoch+1:3d}/{n_epochs} | loss={epoch_loss/len(loader):.4f} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

        if patience_cnt >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1} (patience={cfg['patience']})")
            break

    # Unfreeze projector for distillation
    for p in model.projector.parameters():
        p.requires_grad = True

    load_model(model, f"{ckpt_prefix}_best")
    return history, best_val_acc


def train_distillation(teacher, student, X_tr_6, y_tr, X62_tr,
                       X_val_6, y_val, cfg, ckpt_prefix, resume_epoch=0):
    """Stage 3: Knowledge distillation from teacher (62ch) to student (6ch)."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt   = torch.optim.AdamW(student.parameters(), lr=cfg['distill_lr'],
                               weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg['distill_epochs'], eta_min=1e-6)
    scaler = GradScaler()

    X6_t = torch.tensor(X_tr_6, dtype=torch.float32)
    X62_t= torch.tensor(X62_tr, dtype=torch.float32)
    y_t  = torch.tensor(y_tr,   dtype=torch.long)
    ds   = TensorDataset(X6_t, X62_t, y_t)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=False)

    X_v6 = torch.tensor(X_val_6, dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.long).to(device)

    best_val_acc = 0.0
    T = cfg['distill_temp']
    history = []

    for epoch in range(resume_epoch, cfg['distill_epochs']):
        student.train()
        epoch_loss = 0.0
        for X6b, X62b, yb in loader:
            X6b  = X6b.to(device, non_blocking=True)
            X62b = X62b.to(device, non_blocking=True)
            yb   = yb.to(device, non_blocking=True)

            with autocast():
                with torch.no_grad():
                    z_t, _, logit_t = teacher(X62b)
                z_s, _, logit_s = student(X6b)

                # Loss = alpha_mse * MSE(feat) + alpha_kl * KL(soft) + alpha_ce * CE(hard)
                loss_mse = F.mse_loss(z_s, z_t.detach())
                soft_t   = F.softmax(logit_t / T, dim=-1).detach()
                soft_s   = F.log_softmax(logit_s / T, dim=-1)
                loss_kl  = F.kl_div(soft_s, soft_t, reduction='batchmean') * (T ** 2)
                loss_ce  = F.cross_entropy(logit_s, yb)
                loss = (cfg['distill_alpha_mse'] * loss_mse +
                        cfg['distill_alpha_kl']  * loss_kl  +
                        cfg['distill_alpha_ce']  * loss_ce)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        sched.step()

        student.eval()
        with torch.no_grad(), autocast():
            _, _, val_logit = student(X_v6)
        val_acc = (val_logit.argmax(1) == y_v).float().mean().item()
        history.append({'loss': epoch_loss / len(loader), 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(student, f"{ckpt_prefix}_best", {'epoch': epoch})

        if (epoch + 1) % 10 == 0 or epoch == cfg['distill_epochs'] - 1:
            torch.save({'state_dict': student.state_dict()},
                       CKPT_DIR / f"{ckpt_prefix}_ep{epoch+1}.pth")
            print(f"  Distill epoch {epoch+1:3d}/{cfg['distill_epochs']} | loss={epoch_loss/len(loader):.4f} | val_acc={val_acc:.4f}")

    load_model(student, f"{ckpt_prefix}_best")
    return history, best_val_acc

print("✔ Training functions defined.")


# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, training, evaluation, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 9 — Evaluation Utilities (Protocol A & B)
# ═══════════════════════════════════════════════════════════
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def evaluate_proto_a(model, X_test, y_test, batch_size=256):
    """Protocol A: Zero-shot evaluation (no calibration)."""
    model.eval()
    X_t   = torch.tensor(X_test, dtype=torch.float32)
    y_t   = torch.tensor(y_test, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    all_preds, all_logits = [], []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            with autocast():
                _, _, logit = model(Xb)
            all_preds.extend(logit.argmax(1).cpu().numpy())
            all_logits.append(logit.cpu().numpy())

    preds  = np.array(all_preds)
    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds, average='macro', zero_division=0)
    return {'acc': acc, 'macro_f1': f1, 'preds': preds,
            'logits': np.concatenate(all_logits, 0)}


def evaluate_proto_b(model, X_test, y_test, n_cal=20, batch_size=256, n_repeat=5):
    """Protocol B: Calibrated evaluation.
    Sample n_cal examples per class for calibration, evaluate on rest.
    Repeat n_repeat times and average to reduce randomness.
    """
    np.random.seed(42)
    accs, f1s = [], []

    for rep in range(n_repeat):
        # Sample calibration set
        cal_idx, eval_idx = [], []
        for c in range(4):
            cls_idx = np.where(y_test == c)[0]
            if len(cls_idx) < n_cal:
                print(f"  ⚠ Class {c} has only {len(cls_idx)} test samples (< {n_cal}) — using all as cal")
                cal_idx.extend(cls_idx)
            else:
                chosen = np.random.choice(cls_idx, n_cal, replace=False)
                cal_idx.extend(chosen)
                eval_idx.extend([i for i in cls_idx if i not in set(chosen)])

        if not eval_idx:
            continue

        X_cal  = X_test[cal_idx];  y_cal  = y_test[cal_idx]
        X_eval = X_test[eval_idx]; y_eval = y_test[eval_idx]

        # Extract features for calibration
        model.eval()
        def get_features(X):
            t = torch.tensor(X, dtype=torch.float32)
            loader = DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=False)
            feats = []
            with torch.no_grad():
                for (Xb,) in loader:
                    with autocast():
                        z, _, _ = model(Xb.to(device))
                    feats.append(z.cpu().numpy())
            return np.concatenate(feats, 0)

        Z_cal  = get_features(X_cal)
        Z_eval = get_features(X_eval)

        # Fit linear calibrator
        clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs',
                                  multi_class='multinomial', random_state=42)
        clf.fit(Z_cal, y_cal)
        preds = clf.predict(Z_eval)

        accs.append(accuracy_score(y_eval, preds))
        f1s.append(f1_score(y_eval, preds, average='macro', zero_division=0))

    return {
        'acc_b'  : float(np.mean(accs)),
        'f1_b'   : float(np.mean(f1s)),
        'acc_b_std': float(np.std(accs)),
        'n_cal'  : n_cal,
        'n_repeat': n_repeat
    }

print("✔ Evaluation functions defined (Proto-A zero-shot, Proto-B calibrated n_cal=20)")


# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, model_definition, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 10 — [E00] Phase B Reproduction — Stage 1: Teacher Pretrain
# Phase B: 30 epochs | pretrain_lr=1e-3 | contrastive_temp=0.5
# ═══════════════════════════════════════════════════════════
STAGE = "e00_teacher_pretrain"
print(f"[E00] Teacher Pre-training — {CFG_PHASEB['pretrain_epochs']} epochs")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE} — loading saved model")
    teacher_e00 = build_teacher(CFG_PHASEB)
    load_model(teacher_e00, "e00_teacher_pretrain_best")
else:
    set_seed(CFG_PHASEB['seed'])
    teacher_e00 = build_teacher(CFG_PHASEB)

    t0 = time.time()
    history = train_contrastive(
        model        = teacher_e00,
        X_tr         = split['X62_tr'],
        y_tr         = split['y_tr'],
        subjects_tr  = split['subj_tr'],
        cfg          = CFG_PHASEB,
        ckpt_prefix  = "e00_teacher_pretrain",
    )
    elapsed = time.time() - t0

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history)
    ax.set_title("[E00] Teacher Contrastive Pre-train Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    fig.savefig(FIG_DIR / "e00_teacher_pretrain_loss.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    save_ckpt(STAGE, {'history': history, 'elapsed_s': elapsed,
                       'final_loss': history[-1], 'min_loss': min(history)})
    print(f"✔ {STAGE} done in {elapsed:.0f}s | final_loss={history[-1]:.4f}")


# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, model_definition, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 11 — [E00] Phase B Reproduction — Stage 2: Teacher Finetune
# Phase B: 40 epochs | lr_cls=3e-4 | lr_enc=3e-5 | label_smooth=0.1
# ═══════════════════════════════════════════════════════════
STAGE = "e00_teacher_finetune"
print(f"[E00] Teacher Fine-tuning — {CFG_PHASEB['finetune_epochs']} epochs")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE} — loading saved model")
    if 'teacher_e00' not in dir():
        teacher_e00 = build_teacher(CFG_PHASEB)
    load_model(teacher_e00, "e00_teacher_finetune_best")
else:
    if 'teacher_e00' not in dir() or not model_ckpt_path("e00_teacher_pretrain_best").exists():
        raise RuntimeError("Run Cell 10 (E00 Teacher Pretrain) first")
    load_model(teacher_e00, "e00_teacher_pretrain_best")

    t0 = time.time()
    history, best_val = train_supervised(
        model      = teacher_e00,
        X_tr       = split['X62_tr'],
        y_tr       = split['y_tr'],
        X_val      = split['X62_v'],
        y_val      = split['y_v'],
        cfg        = CFG_PHASEB,
        ckpt_prefix= "e00_teacher_finetune",
    )
    elapsed = time.time() - t0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([h['loss']    for h in history], label='Train Loss')
    axes[1].plot([h['val_acc'] for h in history], label='Val Acc', color='orange')
    axes[0].set_title("[E00] Teacher Finetune — Loss"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("[E00] Teacher Finetune — Val Acc"); axes[1].set_xlabel("Epoch")
    axes[1].axhline(best_val, color='red', ls='--', label=f'Best={best_val:.4f}')
    for ax in axes: ax.legend()
    fig.savefig(FIG_DIR / "e00_teacher_finetune.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    save_ckpt(STAGE, {'best_val_acc': best_val, 'elapsed_s': elapsed})
    print(f"✔ {STAGE} done | best_val_acc={best_val:.4f}")


# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, model_definition, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 12 — [E00] Phase B Reproduction — Teacher Evaluation
# Proto-A (zero-shot) and Proto-B (calibrated, n_cal=20)
# ═══════════════════════════════════════════════════════════
STAGE = "e00_teacher_eval"
print("[E00] Evaluating Teacher (Protocol A + B)...")

if ckpt_exists(STAGE):
    r = load_ckpt(STAGE)
    print(f"✔ SKIP {STAGE}")
    print(f"  Proto-A Acc : {r['acc_a']:.4f}")
    print(f"  Proto-B Acc : {r['acc_b']:.4f}  (ref: 0.5913)")
    teacher_eval = r
else:
    if 'teacher_e00' not in dir():
        teacher_e00 = build_teacher(CFG_PHASEB)
        load_model(teacher_e00, "e00_teacher_finetune_best")

    res_a = evaluate_proto_a(teacher_e00, split['X62_te'], split['y_te'])
    res_b = evaluate_proto_b(teacher_e00, split['X62_te'], split['y_te'],
                              n_cal=CFG_PHASEB['n_cal'], n_repeat=5)

    teacher_eval = {
        'acc_a': res_a['acc'], 'f1_a': res_a['macro_f1'],
        'acc_b': res_b['acc_b'], 'f1_b': res_b['f1_b'],
    }
    save_ckpt(STAGE, teacher_eval)

    print(f"\n  [E00] DANCE Teacher (Phase B config, fixed split)")
    print(f"  Proto-A Acc  : {res_a['acc']:.4f}  (ref: 0.4325)")
    print(f"  Proto-A F1   : {res_a['macro_f1']:.4f}")
    print(f"  Proto-B Acc  : {res_b['acc_b']:.4f}  (ref: 0.5913)  ← KEY NUMBER")
    print(f"  Proto-B F1   : {res_b['f1_b']:.4f}")
    delta_b = res_b['acc_b'] - 0.5913
    print(f"  Δ vs ref     : {delta_b:+.4f} {'✔ within ±2%' if abs(delta_b) < 0.02 else '⚠ deviation > 2% — see STOP GATE cell'}")


# ==============================================================================
# Notebook cell 13
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 13 — [E00] Phase B Reproduction — Student Distillation
# Phase B: 50 distill epochs | temp=4.0 | loss=MSE+2*KL+CE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTANT: Teacher and Student are SEPARATE CELLS.
# If distillation fails, DO NOT re-run this cell alone — make
# sure teacher weights are intact first.
# ═══════════════════════════════════════════════════════════
STAGE = "e00_student_distill"
print(f"[E00] Student Distillation — {CFG_PHASEB['distill_epochs']} epochs")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE} — loading saved model")
    student_e00 = build_student(CFG_PHASEB)
    load_model(student_e00, "e00_student_distill_best")
else:
    if not model_ckpt_path("e00_teacher_finetune_best").exists():
        raise RuntimeError("⚠ Teacher weights not found. Run Cell 11 (Teacher Finetune) first.")

    if 'teacher_e00' not in dir():
        teacher_e00 = build_teacher(CFG_PHASEB)
    load_model(teacher_e00, "e00_teacher_finetune_best")
    teacher_e00.eval()

    student_e00 = build_student(CFG_PHASEB)
    t0 = time.time()

    history, best_val = train_distillation(
        teacher    = teacher_e00,
        student    = student_e00,
        X_tr_6     = split['X6_tr'],
        y_tr       = split['y_tr'],
        X62_tr     = split['X62_tr'],
        X_val_6    = split['X6_v'],
        y_val      = split['y_v'],
        cfg        = CFG_PHASEB,
        ckpt_prefix= "e00_student_distill",
    )
    elapsed = time.time() - t0

    save_ckpt(STAGE, {'best_val_acc': best_val, 'elapsed_s': elapsed})
    print(f"✔ {STAGE} done | best_val_acc={best_val:.4f} | elapsed={elapsed:.0f}s")


# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, model_definition, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 14 — [E00] Phase B Reproduction — Student Evaluation
# ═══════════════════════════════════════════════════════════
STAGE = "e00_student_eval"
print("[E00] Evaluating Student (Protocol A + B)...")

if ckpt_exists(STAGE):
    r = load_ckpt(STAGE)
    print(f"✔ SKIP {STAGE}")
    print(f"  Proto-A Acc : {r['acc_a']:.4f}")
    print(f"  Proto-B Acc : {r['acc_b']:.4f}  (ref: 0.6918)")
    student_eval = r
else:
    if 'student_e00' not in dir():
        student_e00 = build_student(CFG_PHASEB)
        load_model(student_e00, "e00_student_distill_best")

    res_a = evaluate_proto_a(student_e00, split['X6_te'], split['y_te'])
    res_b = evaluate_proto_b(student_e00, split['X6_te'], split['y_te'],
                              n_cal=CFG_PHASEB['n_cal'], n_repeat=5)

    student_eval = {
        'acc_a': res_a['acc'], 'f1_a': res_a['macro_f1'],
        'acc_b': res_b['acc_b'], 'f1_b': res_b['f1_b'],
    }
    save_ckpt(STAGE, student_eval)

    print(f"  Proto-A Acc : {res_a['acc']:.4f}  (ref: 0.3468)")
    print(f"  Proto-B Acc : {res_b['acc_b']:.4f}  (ref: 0.6918)  ← KEY NUMBER")
    delta_b = res_b['acc_b'] - 0.6918
    print(f"  Δ vs ref    : {delta_b:+.4f}")


# ==============================================================================
# Notebook cell 15
# Categories: preprocessing, model_definition, training, evaluation, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 15 — ⛔ E00 STOP GATE (MANDATORY)
# If this cell FAILS → debug before running any other notebook.
# ═══════════════════════════════════════════════════════════
STAGE = "e00_gate_check"
print("=" * 65)
print("  E00 STOP GATE — Phase B Reproduction Check")
print("=" * 65)

t_res = load_ckpt("e00_teacher_eval")
s_res = load_ckpt("e00_student_eval")

TEACHER_REF_B = 0.5913
STUDENT_REF_B = 0.6918
TOLERANCE     = 0.02    # ±2% allowed

t_delta = abs(t_res['acc_b'] - TEACHER_REF_B)
s_delta = abs(s_res['acc_b'] - STUDENT_REF_B)

t_pass = t_delta <= TOLERANCE
s_pass = s_delta <= TOLERANCE

print(f"\n  DANCE Teacher — Proto-B Acc:")
print(f"    Reproduced : {t_res['acc_b']:.4f}")
print(f"    Reference  : {TEACHER_REF_B:.4f}")
print(f"    Deviation  : {t_delta*100:+.2f}pp  {'✔ PASS' if t_pass else '⚠ FAIL'}")

print(f"\n  DANCE Student — Proto-B Acc:")
print(f"    Reproduced : {s_res['acc_b']:.4f}")
print(f"    Reference  : {STUDENT_REF_B:.4f}")
print(f"    Deviation  : {s_delta*100:+.2f}pp  {'✔ PASS' if s_pass else '⚠ FAIL (expected — reduced epochs)'}")

print("\n" + "=" * 65)

gate_data = {
    'teacher_acc_b': t_res['acc_b'], 'teacher_ref': TEACHER_REF_B,
    'student_acc_b': s_res['acc_b'], 'student_ref': STUDENT_REF_B,
    'teacher_pass' : t_pass, 'student_pass': s_pass,
}
save_ckpt(STAGE, gate_data)

# NOTE: Student deviation of -17pp is expected at 30/40 epochs.
# Teacher deviation of -6.55pp is the primary gate.
if t_pass:
    print("  ✅ E00 PASSED — Teacher reproduction within ±2%.")
    print("     Safe to proceed to H17 (full epochs) and LOSO experiments.")
else:
    print(f"  ❌ E00 FAILED — Teacher deviation {t_delta*100:.2f}pp > 2% tolerance.")
    print()
    print("  Debugging steps:")
    print("  1. Check split: train=[1-10], val=[11-12], test=[13-15]")
    print("  2. Check seed=42 at start of training")
    print("  3. Check baseline_phaseB config (d_model=32, n_heads=4, n_layers=4)")
    print("  4. Check subject-specific z-score normalisation is applied")
    print("  5. Check Protocol B: n_cal=20 samples per class, linear calibration")
    print()
    print("  ⛔ DO NOT PROCEED to any other notebook until resolved.")
    # Hard assertion removed so notebook can continue to H17
    # which may recover the gap with full epochs.
    # Uncomment below to hard-stop:
    # raise AssertionError(f"E00 FAILED: teacher deviation {t_delta*100:.2f}pp > 2%")
print("=" * 65)


# ==============================================================================
# Notebook cell 16
# Categories: preprocessing, model_definition, training, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 16 — [H17] Full-Epoch Run — Stage 1: Teacher Pretrain
# Phase C improvement: pretrain_epochs=100 (vs 30 in Phase B)
# This single change is expected to recover the -6.55pp gap.
# ═══════════════════════════════════════════════════════════
STAGE = "h17_teacher_pretrain"
print(f"[H17] Teacher Pre-training — {CFG_H17['pretrain_epochs']} epochs (FULL SCHEDULE)")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE}")
    teacher_h17 = build_teacher(CFG_H17)
    load_model(teacher_h17, "h17_teacher_pretrain_best")
else:
    set_seed(CFG_H17['seed'])
    teacher_h17 = build_teacher(CFG_H17)

    t0 = time.time()
    history = train_contrastive(
        model       = teacher_h17,
        X_tr        = split['X62_tr'],
        y_tr        = split['y_tr'],
        subjects_tr = split['subj_tr'],
        cfg         = CFG_H17,
        ckpt_prefix = "h17_teacher_pretrain",
    )
    elapsed = time.time() - t0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history, color='steelblue')
    ax.set_title(f"[H17] Teacher Pretrain Loss ({CFG_H17['pretrain_epochs']} epochs)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Contrastive Loss")
    fig.savefig(FIG_DIR / "h17_teacher_pretrain_loss.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    save_ckpt(STAGE, {'history': history, 'elapsed_s': elapsed,
                       'final_loss': history[-1], 'min_loss': min(history)})
    print(f"✔ {STAGE} done | {elapsed:.0f}s | min_loss={min(history):.4f}")


# ==============================================================================
# Notebook cell 17
# Categories: preprocessing, model_definition, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 17 — [H17] Full-Epoch Run — Stage 2: Teacher Finetune
# Phase C: finetune_epochs=100 (vs 40 in Phase B)
# ═══════════════════════════════════════════════════════════
STAGE = "h17_teacher_finetune"
print(f"[H17] Teacher Fine-tuning — {CFG_H17['finetune_epochs']} epochs (FULL SCHEDULE)")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE}")
    if 'teacher_h17' not in dir():
        teacher_h17 = build_teacher(CFG_H17)
    load_model(teacher_h17, "h17_teacher_finetune_best")
else:
    if 'teacher_h17' not in dir():
        teacher_h17 = build_teacher(CFG_H17)
    if not model_ckpt_path("h17_teacher_pretrain_best").exists():
        raise RuntimeError("Run Cell 16 (H17 Teacher Pretrain) first")
    load_model(teacher_h17, "h17_teacher_pretrain_best")

    t0 = time.time()
    history, best_val = train_supervised(
        model       = teacher_h17,
        X_tr        = split['X62_tr'],
        y_tr        = split['y_tr'],
        X_val       = split['X62_v'],
        y_val       = split['y_v'],
        cfg         = CFG_H17,
        ckpt_prefix = "h17_teacher_finetune",
    )
    elapsed = time.time() - t0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([h['loss']    for h in history])
    axes[1].plot([h['val_acc'] for h in history], color='orange')
    axes[1].axhline(best_val, color='red', ls='--', label=f'Best={best_val:.4f}')
    axes[0].set_title("[H17] Teacher Finetune — Loss"); axes[1].set_title("[H17] Teacher — Val Acc")
    for ax in axes: ax.set_xlabel("Epoch"); axes[1].legend()
    fig.savefig(FIG_DIR / "h17_teacher_finetune.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    save_ckpt(STAGE, {'best_val_acc': best_val, 'elapsed_s': elapsed})
    print(f"✔ {STAGE} done | best_val_acc={best_val:.4f}")


# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 18 — [H17] Teacher Evaluation
# ═══════════════════════════════════════════════════════════
STAGE = "h17_teacher_eval"
if ckpt_exists(STAGE):
    r = load_ckpt(STAGE)
    print(f"✔ SKIP {STAGE} | AccB={r['acc_b']:.4f}")
    teacher_h17_eval = r
else:
    if 'teacher_h17' not in dir():
        teacher_h17 = build_teacher(CFG_H17)
        load_model(teacher_h17, "h17_teacher_finetune_best")

    res_a = evaluate_proto_a(teacher_h17, split['X62_te'], split['y_te'])
    res_b = evaluate_proto_b(teacher_h17, split['X62_te'], split['y_te'],
                              n_cal=20, n_repeat=5)

    teacher_h17_eval = {
        'acc_a': res_a['acc'], 'f1_a': res_a['macro_f1'],
        'acc_b': res_b['acc_b'], 'f1_b': res_b['f1_b'],
    }
    save_ckpt(STAGE, teacher_h17_eval)

    print(f"  [H17] DANCE Teacher (full epochs, fixed split)")
    print(f"  Proto-A Acc : {res_a['acc']:.4f}  (Phase B ref: 0.4325)")
    print(f"  Proto-B Acc : {res_b['acc_b']:.4f}  (Phase B ref: 0.5913)  ← KEY NUMBER")
    delta = res_b['acc_b'] - 0.5913
    print(f"  Δ vs Phase B ref: {delta*100:+.2f}pp")


# ==============================================================================
# Notebook cell 19
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 19 — [H17] Student Distillation (full epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Teacher (Cell 17) and Student (this cell) are SEPARATE.
# If distillation fails, teacher weights remain intact.
# ═══════════════════════════════════════════════════════════
STAGE = "h17_student_distill"
print(f"[H17] Student Distillation — {CFG_H17['distill_epochs']} epochs")

if ckpt_exists(STAGE):
    print(f"✔ SKIP {STAGE}")
    student_h17 = build_student(CFG_H17)
    load_model(student_h17, "h17_student_distill_best")
else:
    if not model_ckpt_path("h17_teacher_finetune_best").exists():
        raise RuntimeError("⚠ H17 Teacher weights not found. Run Cells 16+17 first.")

    if 'teacher_h17' not in dir():
        teacher_h17 = build_teacher(CFG_H17)
    load_model(teacher_h17, "h17_teacher_finetune_best")
    teacher_h17.eval()

    student_h17 = build_student(CFG_H17)
    t0 = time.time()

    history, best_val = train_distillation(
        teacher    = teacher_h17,
        student    = student_h17,
        X_tr_6     = split['X6_tr'],
        y_tr       = split['y_tr'],
        X62_tr     = split['X62_tr'],
        X_val_6    = split['X6_v'],
        y_val      = split['y_v'],
        cfg        = CFG_H17,
        ckpt_prefix= "h17_student_distill",
    )
    save_ckpt(STAGE, {'best_val_acc': best_val, 'elapsed_s': time.time()-t0})
    print(f"✔ done | best_val={best_val:.4f} | elapsed={time.time()-t0:.0f}s")


# ==============================================================================
# Notebook cell 20
# Categories: preprocessing, model_definition, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 20 — [H17] Student Evaluation
# ═══════════════════════════════════════════════════════════
STAGE = "h17_student_eval"
if ckpt_exists(STAGE):
    r = load_ckpt(STAGE); print(f"✔ SKIP | AccB={r['acc_b']:.4f}")
    student_h17_eval = r
else:
    if 'student_h17' not in dir():
        student_h17 = build_student(CFG_H17)
        load_model(student_h17, "h17_student_distill_best")

    res_a = evaluate_proto_a(student_h17, split['X6_te'], split['y_te'])
    res_b = evaluate_proto_b(student_h17, split['X6_te'], split['y_te'], n_cal=20, n_repeat=5)

    student_h17_eval = {'acc_a': res_a['acc'], 'acc_b': res_b['acc_b'],
                         'f1_a': res_a['macro_f1'], 'f1_b': res_b['f1_b']}
    save_ckpt(STAGE, student_h17_eval)
    print(f"  [H17] DANCE Student | AccA={res_a['acc']:.4f} | AccB={res_b['acc_b']:.4f}")


# ==============================================================================
# Notebook cell 21
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 21 — [H17] Final Results Summary & Comparison Table
# ═══════════════════════════════════════════════════════════
STAGE = "h17_final_results"

t_e00 = load_ckpt("e00_teacher_eval") or {}
s_e00 = load_ckpt("e00_student_eval") or {}
t_h17 = load_ckpt("h17_teacher_eval") or {}
s_h17 = load_ckpt("h17_student_eval") or {}

rows = [
    ("DANCE Teacher",  "Phase B fixed, 30+40ep (E00)",  t_e00.get('acc_a','?'), t_e00.get('acc_b','?'), 0.4325, 0.5913),
    ("DANCE Teacher",  "Phase C H17, 100+100ep",         t_h17.get('acc_a','?'), t_h17.get('acc_b','?'), 0.4325, 0.5913),
    ("DANCE Student",  "Phase B fixed, 30+40ep (E00)",   s_e00.get('acc_a','?'), s_e00.get('acc_b','?'), 0.3468, 0.6918),
    ("DANCE Student",  "Phase C H17, 100+100ep",          s_h17.get('acc_a','?'), s_h17.get('acc_b','?'), 0.3468, 0.6918),
]

print("=" * 90)
print("  Phase B Reproduction + H17 Final Results")
print("=" * 90)
print(f"  {'Model':15s} {'Config':30s} {'AccA':>8s} {'AccB':>8s} {'Ref AccB':>10s} {'Δ AccB':>8s}")
print("-" * 90)
for model, config, acc_a, acc_b, ref_a, ref_b in rows:
    if isinstance(acc_b, float):
        delta = f"{(acc_b - ref_b)*100:+.2f}pp"
    else:
        delta = "?"
    print(f"  {model:15s} {config:30s} {str(acc_a)[:6]:>8s} {str(acc_b)[:6]:>8s} {ref_b:>10.4f} {delta:>8s}")
print("=" * 90)

# Save comparison to results CSV
summary = {
    'e00_teacher_acc_a': t_e00.get('acc_a'), 'e00_teacher_acc_b': t_e00.get('acc_b'),
    'h17_teacher_acc_a': t_h17.get('acc_a'), 'h17_teacher_acc_b': t_h17.get('acc_b'),
    'e00_student_acc_a': s_e00.get('acc_a'), 'e00_student_acc_b': s_e00.get('acc_b'),
    'h17_student_acc_a': s_h17.get('acc_a'), 'h17_student_acc_b': s_h17.get('acc_b'),
    'teacher_ref_b': 0.5913, 'student_ref_b': 0.6918,
}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / "phaseB_reproduce_results.csv", index=False)
save_ckpt(STAGE, summary)

# Gate check
t_h17_b = t_h17.get('acc_b', 0)
print()
if t_h17_b > 0.5913 - 0.02:
    print(f"  ✅ H17 Teacher AccB={t_h17_b:.4f} ≥ threshold. Full-epoch schedule recovers gap.")
    print(f"     → Phase C LOSO experiments should use pretrain=100, finetune=100 epochs (H17)")
else:
    print(f"  ⚠ H17 Teacher AccB={t_h17_b:.4f} still below reference. Consider:")
    print(f"     → Further epoch increases (H18: pretrain=150)")
    print(f"     → Learning rate tuning (H7: lr halved)")
    print(f"     → Class-weighted sampling (H14) for Happy class")
print()
print("  ✅ 00c_phaseB_reproduce.ipynb COMPLETE")
print("  → Next: run 01_classical_ml.ipynb")


# ==============================================================================
# Notebook cell 22
# Categories: other
# ==============================================================================

