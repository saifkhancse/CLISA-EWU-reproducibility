# Auto-exported raw code from notebook: knowledge-distillation.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 0
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification, webapp_or_demo
# ==============================================================================
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════
East West University | CSE Research Pipeline
Phase C: Knowledge Distillation (M20 62ch Teacher -> M27 6ch Student)
Protocol: Option A (Contrastive Pre-training -> KD Fine-tuning)
════════════════════════════════════════════════════════════════════════════
"""

import os, json, time, copy, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 1. DEVICE & PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    print(f'✅ GPU Active: {torch.cuda.get_device_name(0)}')

KAGGLE_INPUT_DIR = Path('/kaggle/input/datasets/saifkhancse/seed-iv-clisa-data')

FEAT_DIR = KAGGLE_INPUT_DIR / 'features'
MDL_DIR  = KAGGLE_INPUT_DIR / 'checkpoints'

# Working directories for this specific notebook
BASE     = Path('/kaggle/working')
CKPT_DIR = BASE / 'checkpoints' / 'loso_results'
OUT_MDL  = BASE / 'checkpoints' / 'model_weights'
RES_DIR  = BASE / 'results' / 'deep_models_seediv'
for d in [CKPT_DIR, OUT_MDL, RES_DIR]: d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 2. EXPERIMENT CONFIG & UTILITIES
# ═══════════════════════════════════════════════════════════════
SEEDS    = [1, 7, 21]
SUBJECTS = list(range(1, 16))
N_CLASSES = 4
N_CAL    = 20
PATIENCE = 15

# Quick Run Toggle (Set to False for full 45-fold run)
QUICK_RUN_BEST_FOLD = False    
BEST_SEED, BEST_FOLD = 7, 15

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weight_path(model_id, seed, fold, ch='62ch', is_teacher=False):
    target_dir = MDL_DIR if is_teacher else OUT_MDL
    return target_dir / f'{model_id}_{ch}_s{seed}_f{fold:02d}_best.pth'

def ck_path(model_id, seed, fold, ch):
    return CKPT_DIR / f'{model_id}_{ch}_seed{seed}_fold{fold:02d}.json'

def ck_save(model_id, seed, fold, ch, result):
    with open(ck_path(model_id, seed, fold, ch), 'w') as f: json.dump(result, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# 3. DATA LOADING & SPLITTING
# ═══════════════════════════════════════════════════════════════
print("Loading features...")
X62 = np.load(FEAT_DIR / 'seed_iv_X_62ch.npy').astype(np.float32)
X6  = np.load(FEAT_DIR / 'seed_iv_X_6ch.npy' ).astype(np.float32)
Y   = np.load(FEAT_DIR / 'seed_iv_y_4cls.npy' ).astype(np.int64)
S   = np.load(FEAT_DIR / 'seed_iv_subjects.npy').astype(np.int64)

in_feat_62 = X62.shape[1]
in_feat_6  = X6.shape[1]

def loso_split(X, Y, S, test_sub, val_frac=0.15, seed=42):
    te_mask = (S == test_sub)
    tr_mask = ~te_mask
    Xtr, Ytr, Str = X[tr_mask], Y[tr_mask], S[tr_mask]
    Xte, Yte      = X[te_mask], Y[te_mask]
    idx = np.arange(len(Xtr))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    return (Xtr[trn_idx], Ytr[trn_idx], Str[trn_idx],
            Xtr[val_idx],  Ytr[val_idx],
            Xte, Yte)

def make_loader(X, Y, batch_size, weighted=True, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
    if weighted and shuffle:
        counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
        weights = 1.0 / (counts[Y] + 1e-6)
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

# ═══════════════════════════════════════════════════════════════
# 4. ARCHITECTURE & EVALUATION LOGIC
# ═══════════════════════════════════════════════════════════════
class CLISA(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=128, d_proj=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d_model), nn.BatchNorm1d(d_model), nn.ReLU())
        self.projector  = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.classifier = nn.Linear(d_model, n_cls)
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
    def project(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=-1)

def nt_xent_loss(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e4)
    pos_sim = torch.cat([torch.diagonal(sim, B), torch.diagonal(sim, -B)])
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []
    for Xb, Yb in loader:
        with autocast():
            out = model(Xb.to(device))
            preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds); all_true.extend(Yb.numpy())
    return accuracy_score(all_true, all_preds), f1_score(all_true, all_preds, average='macro', zero_division=0)

def proto_b_calibrate(model, Xte, Yte, n_cal=N_CAL, cal_epochs=10, lr=1e-3):
    if len(Xte) <= n_cal:
        return evaluate(model, make_loader(Xte, Yte, batch_size=256, weighted=False, shuffle=False))
    idx   = np.random.choice(len(Xte), n_cal, replace=False)
    mask  = np.zeros(len(Xte), dtype=bool); mask[idx] = True
    Xcal, Ycal = Xte[mask],  Yte[mask]
    Xeva, Yeva = Xte[~mask], Yte[~mask]
    
    cal_model = copy.deepcopy(model).to(device)
    for p in cal_model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(cal_model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    cal_loader = make_loader(Xcal, Ycal, batch_size=min(n_cal, 32), weighted=False, shuffle=True)
    
    cal_model.train()
    for _ in range(cal_epochs):
        for Xb, Yb in cal_loader:
            opt.zero_grad()
            loss = crit(cal_model(Xb.to(device)), Yb.to(device))
            loss.backward(); opt.step()
    return evaluate(cal_model, make_loader(Xeva, Yeva, batch_size=256, weighted=False, shuffle=False))

# ═══════════════════════════════════════════════════════════════
# 5. M27 DISTILLATION TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
_MID_STU, _CH_STU = 'M27', '6ch'
_MID_TCH, _CH_TCH = 'M20', '62ch'

PRETRAIN_EPOCHS, DISTILL_EPOCHS = 50, 50
LR_PT, LR_KD = 1e-3, 3e-4
DISTILL_TEMP = 4.0
W_MSE, W_KL, W_CE = 1.0, 2.0, 1.0

print('\n' + '='*72)
print(f'▶ {_MID_STU} [{_CH_STU}] — CLISA Student Distillation (Option A)')
print('='*72)

runs = [(BEST_SEED, BEST_FOLD, BEST_FOLD)] if QUICK_RUN_BEST_FOLD else [(s, f, f) for s in SEEDS for f in SUBJECTS]
all_results = []
n_ch_6 = in_feat_6 // 5

for seed, fi, test_sub in runs:
    set_seed(seed)
    t0 = time.time()
    
    # Paired Train splits
    _, Ytr, Str, _, Yva, _, Yte = loso_split(X62, Y, S, test_sub, seed=seed)
    Xtr62, _, _, Xva62, _, Xte62, _ = loso_split(X62, Y, S, test_sub, seed=seed)
    Xtr6,  _, _, Xva6,  _, Xte6,  _ = loso_split(X6,  Y, S, test_sub, seed=seed)

    # ── Load Teacher ───────────────────────────────────────────
    teacher = CLISA(in_feat_62).to(device)
    t_path  = weight_path(_MID_TCH, seed, fi, _CH_TCH, is_teacher=True)
    
    if t_path.exists():
        teacher.load_state_dict(torch.load(t_path, map_location=device))
        print(f'    [{_MID_STU}|s{seed}|f{fi:02d}] Loaded Teacher weights successfully.')
    else:
        print(f'    ⚠ ERROR: Teacher weights missing at {t_path}')
        print(f'      Ensure KAGGLE_INPUT_DIR is set correctly.')
        break
        
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # ── Initialize Student ─────────────────────────────────────
    student = CLISA(in_feat_6).to(device)
    scaler  = GradScaler()
    
    tr_loader_pt = make_loader(Xtr6, Ytr, 128, weighted=True)
    
    tr_ds_kd = TensorDataset(torch.FloatTensor(Xtr6), torch.FloatTensor(Xtr62), torch.LongTensor(Ytr))
    counts = np.bincount(Ytr, minlength=N_CLASSES).astype(float)
    weights = 1.0 / (counts[Ytr] + 1e-6)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)
    tr_loader_kd = DataLoader(tr_ds_kd, batch_size=128, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    va_loader = make_loader(Xva6, Yva, 256, weighted=False, shuffle=False)

    # ── Stage 1: Contrastive Pre-training ──────────────────────
    print(f'      Stage 1: Contrastive Pre-training ({PRETRAIN_EPOCHS} epochs)...')
    opt_pt   = torch.optim.AdamW(student.parameters(), lr=LR_PT, weight_decay=1e-4)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=PRETRAIN_EPOCHS)
    
    for ep in range(PRETRAIN_EPOCHS):
        student.train()
        for Xb, _ in tr_loader_pt:
            Xb = Xb.to(device)
            v1 = Xb + 0.1 * torch.randn_like(Xb)
            mask = (torch.rand(Xb.size(0), n_ch_6, 1, device=Xb.device) > 0.3).float()
            v2 = Xb * mask.repeat(1, 1, 5).view(Xb.size(0), -1)
            
            opt_pt.zero_grad(set_to_none=True)
            with autocast():
                z1 = student.project(v1); z2 = student.project(v2)
                loss_pt = nt_xent_loss(z1, z2, temp=0.5)
            scaler.scale(loss_pt).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt_pt); scaler.update()
        sched_pt.step()

    # ── Stage 2: Knowledge Distillation ────────────────────────
    print(f'      Stage 2: Knowledge Distillation ({DISTILL_EPOCHS} epochs)...')
    for p in student.encoder.parameters(): p.requires_grad = True
    opt_kd   = torch.optim.AdamW(student.parameters(), lr=LR_KD, weight_decay=1e-4)
    sched_kd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_kd, T_max=DISTILL_EPOCHS)
    ce_crit  = nn.CrossEntropyLoss()
    
    best_f1, best_state, no_imp = 0.0, None, 0
    
    for ep in range(DISTILL_EPOCHS):
        student.train()
        for X6b, X62b, Yb in tr_loader_kd:
            X6b, X62b, Yb = X6b.to(device), X62b.to(device), Yb.to(device)
            opt_kd.zero_grad(set_to_none=True)
            with autocast():
                stu_logits = student(X6b)
                stu_feat   = student.encoder(X6b)
                
                with torch.no_grad():
                    tch_logits = teacher(X62b)
                    tch_feat   = teacher.encoder(X62b)
                
                loss_mse = F.mse_loss(stu_feat, tch_feat.detach())
                
                T = DISTILL_TEMP
                stu_soft = F.log_softmax(stu_logits / T, dim=-1)
                tch_soft = F.softmax(tch_logits / T, dim=-1)
                loss_kl  = F.kl_div(stu_soft, tch_soft, reduction='batchmean') * (T**2)
                
                loss_ce  = ce_crit(stu_logits, Yb)
                loss = W_MSE*loss_mse + W_KL*loss_kl + W_CE*loss_ce
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt_kd); scaler.update()
            
        sched_kd.step()
        _, vf1 = evaluate(student, va_loader)
        
        if vf1 > best_f1:
            best_f1 = vf1; best_state = copy.deepcopy(student.state_dict()); no_imp = 0
            if not QUICK_RUN_BEST_FOLD:
                torch.save(best_state, weight_path(_MID_STU, seed, fi, _CH_STU))
        else:
            no_imp += 1
        if no_imp >= PATIENCE: break

    # ── Evaluation ─────────────────────────────────────────────
    student.load_state_dict(best_state)
    acc_a, f1_a = evaluate(student, make_loader(Xte6, Yte, 256, weighted=False, shuffle=False))
    acc_b, f1_b = proto_b_calibrate(student, Xte6, Yte)

    r = dict(model_id=_MID_STU, ch=_CH_STU, seed=seed, fold=fi, test_sub=test_sub,
             acc_a=round(acc_a,4), f1_a=round(f1_a,4),
             acc_b=round(acc_b,4), f1_b=round(f1_b,4),
             best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
    
    if not QUICK_RUN_BEST_FOLD:
        ck_save(_MID_STU, seed, fi, _CH_STU, r)
        
    all_results.append(r)
    print(f'  ✔ {_MID_STU}|{_CH_STU}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} ({r["elapsed"]:.0f}s)')

if not QUICK_RUN_BEST_FOLD and all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(RES_DIR / f'{_MID_STU}_{_CH_STU}_summary.csv', index=False)
    print(f"\n  AccA: {df['acc_a'].mean():.4f} ± {df['acc_a'].std():.4f}")
    print(f"  AccB: {df['acc_b'].mean():.4f} ± {df['acc_b'].std():.4f}")

print(f'\n✅ {_MID_STU} [{_CH_STU}] pipeline execution complete.')
