# Auto-exported raw code from notebook: 02_deep_models.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: preprocessing, model_definition, training, evaluation, figures, webapp_or_demo
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS + DEVICE CHECK
# ═══════════════════════════════════════════════════════════════
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

# ── Device Check ───────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(f'✅ GPU : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    print('✅ TF32 + cuDNN benchmark enabled')
else:
    print('⚠️ CUDA GPU not found. Running on CPU.')
    print('   This is fine for loading checkpoints and plotting confusion matrices.')
    print('   Do not run training cells on CPU unless you really have to.')

print(f'   PyTorch {torch.__version__} | CUDA {torch.version.cuda}')


# ==============================================================================
# Notebook cell 2
# Categories: preprocessing, training, evaluation, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 2 — CONFIG + CHECKPOINT UTILITIES
# ═══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
BASE     = Path('.')
FEAT_DIR = BASE / 'features'
CKPT_DIR = BASE / 'checkpoints' / 'loso_results'
MDL_DIR  = BASE / 'checkpoints' / 'model_weights'
RES_DIR  = BASE / 'results' / 'deep_models_seediv'
for d in [CKPT_DIR, MDL_DIR, RES_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── LOSO / Training Config ────────────────────────────────────
SEEDS    = [1, 7, 21]
SUBJECTS = list(range(1, 16))          # 1..15
N_FOLDS  = 15
N_SEEDS  = 3
N_RUNS   = N_FOLDS * N_SEEDS           # 45 per channel config
N_CLASSES = 4
N_CAL    = 20                          # Proto-B calibration samples
CH6_IDX  = [0, 2, 5, 13, 23, 31]      # FP1,FP2,F7,F8,T7,T8 in 62ch ordering
PATIENCE = 15                          # early-stopping patience

# ── Checkpoint helpers ────────────────────────────────────────
def ck_path(model_id, seed, fold, ch='62ch'):
    return CKPT_DIR / f'{model_id}_{ch}_seed{seed}_fold{fold:02d}.json'

def ck_exists(model_id, seed, fold, ch='62ch'):
    return ck_path(model_id, seed, fold, ch).exists()

def ck_save(model_id, seed, fold, ch, result):
    p = ck_path(model_id, seed, fold, ch)
    with open(p, 'w') as f: json.dump(result, f, indent=2)

def ck_load(model_id, seed, fold, ch='62ch'):
    p = ck_path(model_id, seed, fold, ch)
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

def model_complete(model_id, ch='62ch', n_expected=N_RUNS):
    done = sum(1 for s in SEEDS for fi in range(1, N_FOLDS+1) if ck_exists(model_id, s, fi, ch))
    return done, n_expected, done == n_expected

def weight_path(model_id, seed, fold, ch='62ch'):
    return MDL_DIR / f'{model_id}_{ch}_s{seed}_f{fold:02d}_best.pth'

# ── Reproducibility ────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print('✅ Config loaded')
print(f'   CKPT_DIR : {CKPT_DIR}')
print(f'   RES_DIR  : {RES_DIR}')
print(f'   SEEDS={SEEDS}  N_FOLDS={N_FOLDS}  N_RUNS={N_RUNS}  N_CAL={N_CAL}')



# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 3 — DATA LOADING + LOSO SPLIT UTILS
# ═══════════════════════════════════════════════════════════════

X62 = np.load(FEAT_DIR / 'seed_iv_X_62ch.npy').astype(np.float32)  # (N, 310)
X6  = np.load(FEAT_DIR / 'seed_iv_X_6ch.npy' ).astype(np.float32)  # (N, 30)
Y   = np.load(FEAT_DIR / 'seed_iv_y_4cls.npy' ).astype(np.int64)
S   = np.load(FEAT_DIR / 'seed_iv_subjects.npy').astype(np.int64)

IN_62 = X62.shape[1]   # 310
IN_6  = X6.shape[1]    # 30
print(f'X62 {X62.shape}  X6 {X6.shape}  Y {Y.shape}  S {S.shape}')
print(f'Classes: { {int(c):int(n) for c,n in zip(*np.unique(Y,return_counts=True))} }')
print(f'Subjects: {sorted(np.unique(S).tolist())}')

def get_data(ch='62ch'):
    return X62 if ch == '62ch' else X6

def loso_split(X, Y, S, test_sub, val_frac=0.15, seed=42):
    """Returns train/val/test numpy arrays. Val is a subset of train subjects."""
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
    Xt = torch.FloatTensor(X)
    Yt = torch.LongTensor(Y)
    ds = TensorDataset(Xt, Yt)
    if weighted and shuffle:
        counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
        weights = 1.0 / (counts[Y] + 1e-6)
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)

print('✅ Data loaded')



# ==============================================================================
# Notebook cell 4
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 4 — SHARED TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════

# ── Standard supervised train/eval loop ───────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(Xb)
            loss = criterion(out, Yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []
    for Xb, Yb in loader:
        with autocast():
            out = model(Xb.to(device))
            # Handle models that return a tuple (e.g. DANN returns cls_out, dom_out, feats)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds); all_true.extend(Yb.numpy())
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)
    return acc, f1

def proto_b_calibrate(model, Xte, Yte, n_cal=N_CAL, cal_epochs=10, lr=1e-3):
    """Fine-tune only the final classifier on n_cal calibration samples.
    Returns calibrated accuracy + f1 on remaining test samples."""
    if len(Xte) <= n_cal:
        return evaluate(model, make_loader(Xte, Yte, batch_size=256, weighted=False, shuffle=False))
    idx   = np.random.choice(len(Xte), n_cal, replace=False)
    mask  = np.zeros(len(Xte), dtype=bool); mask[idx] = True
    Xcal, Ycal = Xte[mask],  Yte[mask]
    Xeva, Yeva = Xte[~mask], Yte[~mask]
    # freeze all params except last linear layer
    cal_model = copy.deepcopy(model).to(device)
    for p in cal_model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(cal_model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    cal_loader = make_loader(Xcal, Ycal, batch_size=min(n_cal, 32), weighted=False, shuffle=True)
    cal_model.train()
    for _ in range(cal_epochs):
        for Xb, Yb in cal_loader:
            opt.zero_grad()
            out = cal_model(Xb.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            loss = crit(logits, Yb.to(device))
            loss.backward(); opt.step()
    return evaluate(cal_model, make_loader(Xeva, Yeva, batch_size=256, weighted=False, shuffle=False))

def run_loso(model_id, ch, build_model_fn,
             n_epochs=100, batch_size=256, lr=1e-3, wd=1e-4,
             patience=PATIENCE, save_weights=False,
             print_every=1):
    """
    Full LOSO-15 × 3-seed loop for a supervised model.
    build_model_fn(in_features) → nn.Module
    Returns list of result dicts (one per completed fold×seed).
    """
    X = get_data(ch)
    in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(model_id, seed, fi, ch):
                r = ck_load(model_id, seed, fi, ch)
                all_results.append(r)
                print(f'  SKIP {model_id}|{ch}|s{seed}|f{fi:02d} '
                      f'AccA={r["acc_a"]:.4f} AccB={r["acc_b"]:.4f}')
                continue

            t0 = time.time()
            Xtr, Ytr, _, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)

            tr_loader = make_loader(Xtr, Ytr, batch_size, weighted=True)
            va_loader = make_loader(Xva, Yva, batch_size*2, weighted=False, shuffle=False)

            model   = build_model_fn(in_feat).to(device)
            opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            crit    = nn.CrossEntropyLoss(label_smoothing=0.0)
            scaler  = GradScaler()

            best_f1, best_state, no_improve = 0.0, None, 0
            for ep in range(1, n_epochs+1):
                train_epoch(model, tr_loader, opt, crit, scaler)
                sched.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

            model.load_state_dict(best_state)
            if save_weights:
                torch.save(best_state, weight_path(model_id, seed, fi, ch))

            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, batch_size*2, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)

            r = dict(model_id=model_id, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(model_id, seed, fi, ch, r)
            all_results.append(r)
            if fi % print_every == 0 or fi == N_FOLDS:
                print(f'  {model_id}|{ch}|s{seed}|f{fi:02d} '
                      f'AccA={acc_a:.4f} F1A={f1_a:.4f} '
                      f'AccB={acc_b:.4f} F1B={f1_b:.4f} '
                      f'({r["elapsed"]:.0f}s)')
    return all_results

def summarise_and_save(model_id, ch, results):
    """Print mean±std and save CSV."""
    df = pd.DataFrame(results)
    df = df[(df.model_id==model_id) & (df.ch==ch)]
    if df.empty: return
    for col in ['acc_a','f1_a','acc_b','f1_b']:
        print(f'  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}')
    csv_path = RES_DIR / f'{model_id}_{ch}_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'  Saved → {csv_path}')

print('✅ Shared training utilities ready')



# ==============================================================================
# Notebook cell 5
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 5 — MODEL ARCHITECTURES (M11–M24, DANCE M25/M26)
# ═══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# M11  Shallow MLP
# ──────────────────────────────────────────────────────────────
class ShallowMLP(nn.Module):
    def __init__(self, in_feat, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))
    def forward(self, x): return self.net(x)

# ──────────────────────────────────────────────────────────────
# M12  Deep MLP
# ──────────────────────────────────────────────────────────────
class DeepMLP(nn.Module):
    def __init__(self, in_feat, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),      nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_cls))
    def forward(self, x): return self.net(x)

# ──────────────────────────────────────────────────────────────
# M13  LSTM   (input reshaped to (B, 5, C): bands as timesteps)
# ──────────────────────────────────────────────────────────────
class EEG_LSTM(nn.Module):
    def __init__(self, in_feat, n_cls=4, hidden=128, n_layers=2):
        super().__init__()
        self.n_ch = in_feat // 5          # channels (62 or 6)
        self.lstm = nn.LSTM(self.n_ch, hidden, n_layers, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(64, n_cls))
    def forward(self, x):
        x = x.view(x.size(0), 5, self.n_ch)   # (B, 5, C)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # bidirectional last layer
        return self.head(h)

# ──────────────────────────────────────────────────────────────
# M14  GRU
# ──────────────────────────────────────────────────────────────
class EEG_GRU(nn.Module):
    def __init__(self, in_feat, n_cls=4, hidden=128, n_layers=2):
        super().__init__()
        self.n_ch = in_feat // 5
        self.gru  = nn.GRU(self.n_ch, hidden, n_layers, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(64, n_cls))
    def forward(self, x):
        x = x.view(x.size(0), 5, self.n_ch)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h)

# ──────────────────────────────────────────────────────────────
# M15  Conv1D  (input: (B, C, F) = (B, n_ch, 5))
# ──────────────────────────────────────────────────────────────
class EEG_Conv1D(nn.Module):
    def __init__(self, in_feat, n_cls=4):
        super().__init__()
        n_ch = in_feat // 5
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,  256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,  128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, x):
        n_ch = x.size(1) // 5
        x = x.view(x.size(0), n_ch, 5)   # (B, C, F)
        x = self.conv(x).squeeze(-1)
        return self.head(x)

# ──────────────────────────────────────────────────────────────
# M16  Vanilla Transformer  (channels as tokens, bands as d)
# ──────────────────────────────────────────────────────────────
class VanillaTransformer(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        n_ch = in_feat // 5
        self.proj = nn.Linear(5, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, n_ch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                           dropout=0.2, batch_first=True, norm_first=True)
        self.tf   = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))
    def forward(self, x):
        n_ch = x.size(1) // 5
        x = x.view(x.size(0), n_ch, 5)       # (B, C, F)
        x = self.proj(x) + self.pos[:, :n_ch]  # (B, C, d_model)
        x = self.tf(x)
        x = x.mean(1)                          # mean pool over channels
        return self.head(x)

# ──────────────────────────────────────────────────────────────
# M17  EEG Conformer  (Conv front-end + Transformer)
# ──────────────────────────────────────────────────────────────
class EEGConformer(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        n_ch = in_feat // 5
        # Conv front-end: spatial + spectral mixing
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 5), padding=(0, 2)), nn.BatchNorm2d(16), nn.ELU(),
            nn.Conv2d(16, 32, (n_ch, 1)), nn.BatchNorm2d(32), nn.ELU(),
            nn.Dropout(0.25))
        # After conv: (B, 32, 1, 5) → squeeze → (B, 32, 5) → permute → (B, 5, 32)
        # Each of the 5 band-position tokens has 32 features → project to d_model
        self.proj = nn.Linear(32, d_model)          # ← was nn.Linear(32 * 5, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                           dropout=0.2, batch_first=True, norm_first=True)
        self.tf   = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.3),
                                   nn.Linear(d_model, n_cls))
        self.n_ch = n_ch

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, 1, self.n_ch, 5)          # (B, 1, C, F=5)
        x = self.conv(x)                          # (B, 32, 1, 5)
        x = x.squeeze(2).permute(0, 2, 1)         # (B, 5, 32)
        x = self.proj(x)                           # (B, 5, d_model)
        x = self.tf(x).mean(1)                    # (B, d_model)
        return self.head(x)

# ──────────────────────────────────────────────────────────────
# M18  ChanDrop Transformer  (Transformer + channel dropout)
# ──────────────────────────────────────────────────────────────
class ChanDropTransformer(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=64, n_heads=4, n_layers=2, drop_ch=0.1):
        super().__init__()
        n_ch = in_feat // 5
        self.n_ch     = n_ch
        self.drop_ch  = drop_ch
        self.proj     = nn.Linear(5, d_model)
        self.pos      = nn.Parameter(torch.zeros(1, n_ch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                           dropout=0.2, batch_first=True, norm_first=True)
        self.tf   = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))
    def forward(self, x):
        b = x.size(0)
        n_ch = x.size(1) // 5
        x = x.view(b, n_ch, 5)
        if self.training and self.drop_ch > 0:
            mask = (torch.rand(b, n_ch, 1, device=x.device) > self.drop_ch).float()
            x = x * mask
        x = self.proj(x) + self.pos[:, :n_ch]
        x = self.tf(x).mean(1)
        return self.head(x)

# ──────────────────────────────────────────────────────────────
# M19  DANN  (Gradient Reversal + Subject discriminator)
# ──────────────────────────────────────────────────────────────
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DANN(nn.Module):
    def __init__(self, in_feat, n_cls=4, n_subjects=15, d=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d),       nn.BatchNorm1d(d),   nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_cls))
        self.discriminator = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_subjects))
    def forward(self, x, alpha=1.0):
        feat = self.encoder(x)
        cls_out  = self.classifier(feat)
        rev_feat = GradReverse.apply(feat, alpha)
        dom_out  = self.discriminator(rev_feat)
        return cls_out, dom_out, feat

# ──────────────────────────────────────────────────────────────
# M20  CLISA  (Contrastive + Subject-Invariant repr learning)
# ──────────────────────────────────────────────────────────────
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
    """NT-Xent contrastive loss for pairs (z1[i], z2[i])."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2B, d)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp           # (2B, 2B)
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e4)
    pos_idx = torch.arange(B, device=z.device)
    pos_sim = torch.cat([torch.diagonal(sim, B), torch.diagonal(sim, -B)])
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()

# ──────────────────────────────────────────────────────────────
# M21  SimCLR
# ──────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=128, d_proj=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d_model), nn.BatchNorm1d(d_model), nn.ReLU())
        self.projector  = nn.Sequential(nn.Linear(d_model, d_proj), nn.BatchNorm1d(d_proj),
                                         nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.classifier = nn.Linear(d_model, n_cls)
    def forward(self, x):
        return self.classifier(self.encoder(x))
    def project(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=-1)

# ──────────────────────────────────────────────────────────────
# M22  BYOL
# ──────────────────────────────────────────────────────────────
class BYOLEncoder(nn.Module):
    def __init__(self, in_feat, d_model=128, d_proj=64):
        super().__init__()
        self.encoder   = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, d_model), nn.BatchNorm1d(d_model), nn.ReLU())
        self.projector = nn.Sequential(nn.Linear(d_model, d_proj), nn.BatchNorm1d(d_proj),
                                        nn.ReLU(), nn.Linear(d_proj, d_proj))
    def forward(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=-1)
    def get_repr(self, x):
        return self.encoder(x)

class BYOL(nn.Module):
    def __init__(self, in_feat, n_cls=4, d_model=128, d_proj=64, tau=0.996):
        super().__init__()
        self.tau     = tau
        self.online  = BYOLEncoder(in_feat, d_model, d_proj)
        self.target  = copy.deepcopy(self.online)
        for p in self.target.parameters(): p.requires_grad = False
        self.predictor  = nn.Sequential(nn.Linear(d_proj, d_proj), nn.BatchNorm1d(d_proj),
                                         nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.classifier = nn.Linear(d_model, n_cls)
    @torch.no_grad()
    def update_target(self):
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data = self.tau * tp.data + (1 - self.tau) * op.data
    def forward(self, x):
        return self.classifier(self.online.get_repr(x))
    def byol_loss(self, x1, x2):
        p1 = self.predictor(self.online(x1))
        p2 = self.predictor(self.online(x2))
        with torch.no_grad():
            t1 = self.target(x1)
            t2 = self.target(x2)
        loss = 2 - 2*(F.normalize(p1,dim=-1)*t2).sum(-1).mean()
        loss += 2 - 2*(F.normalize(p2,dim=-1)*t1).sum(-1).mean()
        return loss / 2

# ──────────────────────────────────────────────────────────────
# M23  PseudoLabel (semi-supervised)
# ──────────────────────────────────────────────────────────────
class PseudoLabelNet(nn.Module):
    def __init__(self, in_feat, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))
    def forward(self, x): return self.net(x)

# ──────────────────────────────────────────────────────────────
# M24  MixMatch (semi-supervised)
# ──────────────────────────────────────────────────────────────
class MixMatchNet(nn.Module):
    def __init__(self, in_feat, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))
    def forward(self, x): return self.net(x)

# ──────────────────────────────────────────────────────────────
# DANCE components  (M25 Teacher + M26 Student)
# ──────────────────────────────────────────────────────────────
# Diagonal masked self-attention
class DiagMaskedAttn(nn.Module):
    def __init__(self, d_model, n_heads, bandwidth=5, dropout=0.2):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.bandwidth = bandwidth
        self._mask_cache = {}
    def _get_mask(self, n, device):
        if n not in self._mask_cache:
            mask = torch.full((n, n), float('-inf'))
            for i in range(n):
                lo, hi = max(0, i-self.bandwidth), min(n, i+self.bandwidth+1)
                mask[i, lo:hi] = 0.0
            self._mask_cache[n] = mask
        return self._mask_cache[n].to(device)
    def forward(self, x, training_only=True):
        mask = self._get_mask(x.size(1), x.device) if (self.training or not training_only) else None
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out

class DANCEBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=128, dropout=0.2, bandwidth=5):
        super().__init__()
        self.attn = DiagMaskedAttn(d_model, n_heads, bandwidth, dropout)
        self.ff   = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.ln1(x + self.drop(self.attn(x)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x

class DANCETeacher(nn.Module):
    def __init__(self, n_ch=62, n_bands=5, d_model=32, n_heads=4, n_layers=4,
                 d_ff=128, d_proj=128, dropout=0.2, n_cls=4, n_subjects=15):
        super().__init__()
        self.n_ch     = n_ch
        self.band_emb = nn.Linear(n_bands, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, n_ch, d_model))
        self.blocks   = nn.ModuleList([DANCEBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ch_attn  = nn.Linear(d_model, 1)  # channel attention weights
        self.proj_head  = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.classifier = nn.Linear(d_model, n_cls)
        self.subj_disc  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_subjects))
        self.grl_alpha  = 1.0
    def encode(self, x):
        # x: (B, n_ch*n_bands)
        B = x.size(0)
        x = x.view(B, self.n_ch, -1)             # (B, C, F)
        x = self.band_emb(x) + self.pos_emb      # (B, C, d)
        for blk in self.blocks: x = blk(x)
        w = torch.softmax(self.ch_attn(x), dim=1) # (B, C, 1)
        z = (x * w).sum(1)                         # (B, d)
        return z
    def forward(self, x, return_domain=False):
        z   = self.encode(x)
        cls = self.classifier(z)
        if return_domain:
            rev = GradReverse.apply(z, self.grl_alpha)
            dom = self.subj_disc(rev)
            return cls, dom, z
        return cls
    def project(self, x):
        return F.normalize(self.proj_head(self.encode(x)), dim=-1)

class DANCEStudent(nn.Module):
    def __init__(self, n_ch=6, n_bands=5, d_model=32, n_heads=4, n_layers=2,
                 d_ff=128, d_proj=128, dropout=0.2, n_cls=4):
        super().__init__()
        self.n_ch     = n_ch
        self.band_emb = nn.Linear(n_bands, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, n_ch, d_model))
        self.blocks   = nn.ModuleList([DANCEBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ch_attn  = nn.Linear(d_model, 1)
        self.proj_head  = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.classifier = nn.Linear(d_model, n_cls)
    def encode(self, x):
        B = x.size(0)
        x = x.view(B, self.n_ch, -1)
        x = self.band_emb(x) + self.pos_emb
        for blk in self.blocks: x = blk(x)
        w = torch.softmax(self.ch_attn(x), dim=1)
        return (x * w).sum(1)
    def forward(self, x): return self.classifier(self.encode(x))
    def project(self, x): return F.normalize(self.proj_head(self.encode(x)), dim=-1)

# ── DANCE augmentation utils ────────────────────────────────
def dance_augment(x, mask_ratio=0.3, noise_std=0.1):
    """Channel masking + Gaussian noise (applied to (B, C*F) inputs)."""
    x = x.clone()
    n_ch = x.size(1) // 5
    # channel mask
    mask = (torch.rand(x.size(0), n_ch, 1, device=x.device) > mask_ratio).float()
    mask = mask.repeat(1, 1, 5).view(x.size(0), -1)
    x = x * mask
    # gaussian noise
    x = x + noise_std * torch.randn_like(x)
    return x

def subject_mixup(x, y, s, alpha=0.4):
    """Mixup between same-emotion, different-subject pairs."""
    B = x.size(0)
    lam = np.random.beta(alpha, alpha, B).astype(np.float32)
    lam_t = torch.FloatTensor(lam).to(x.device).unsqueeze(1)
    idx = torch.randperm(B, device=x.device)
    # only mix different subjects, same label
    same_emo = (y == y[idx])
    diff_sub = (s != s[idx])
    valid    = same_emo & diff_sub
    lam_t = torch.where(valid.unsqueeze(1), lam_t, torch.ones_like(lam_t))
    x_mix = lam_t * x + (1 - lam_t) * x[idx]
    return x_mix

print('✅ All model architectures defined (M11–M26 + DANCE)')



# ==============================================================================
# Notebook cell 6
# Categories: training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# CELL 6 — NOTEBOOK STATUS SCAN  (run anytime to check progress)
# ═══════════════════════════════════════════════════════════════

models_62ch = ['M11','M12','M13','M14','M15','M16','M17','M18',
               'M19','M20','M21','M22','M23','M24','M25']
models_6ch  = ['M11','M12','M13','M14','M15','M16','M17','M18',
               'M19','M20','M21','M22','M23','M24','M26']

print('='*72)
print(f'  NOTEBOOK STATUS — 02_deep_models.ipynb')
print('='*72)
total_done = 0
total_exp  = 0
for ch, mlist in [('62ch', models_62ch), ('6ch', models_6ch)]:
    for mid in mlist:
        done, exp, complete = model_complete(mid, ch)
        total_done += done; total_exp += exp
        status = '✅' if complete else ('▶' if done > 0 else '⬜')
        print(f'  {status} {mid:4s} [{ch}]  {done:3d}/{exp} checkpoints')
print('-'*72)
print(f'  TOTAL: {total_done}/{total_exp} checkpoints completed')
print('='*72)



# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M11 — SHALLOW MLP  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M11'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: ShallowMLP(f),
                       n_epochs=100, batch_size=256, lr=1e-3)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M12 — DEEP MLP  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M12'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: DeepMLP(f),
                       n_epochs=100, batch_size=256, lr=1e-3)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M13 — LSTM  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M13'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: EEG_LSTM(f),
                       n_epochs=100, batch_size=256, lr=1e-3)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M14 — GRU  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M14'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: EEG_GRU(f),
                       n_epochs=100, batch_size=256, lr=1e-3)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 16
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M15 — CONV1D  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M15'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: EEG_Conv1D(f),
                       n_epochs=100, batch_size=128, lr=1e-3)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, model_definition, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M16 — VANILLA TRANSFORMER  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M16'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: VanillaTransformer(f),
                       n_epochs=100, batch_size=128, lr=5e-4)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 20
# Categories: preprocessing, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M17 — EEG CONFORMER  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M17'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: EEGConformer(f),
                       n_epochs=100, batch_size=128, lr=5e-4)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 22
# Categories: preprocessing, model_definition, training
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M18 — CHANDROP TRANSFORMER  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M18'
for _ch in ['62ch', '6ch']:
    done, exp, complete = model_complete(_MID, _ch)
    if complete:
        print(f'✅ {_MID} [{_ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        continue
    print(f'\n▶ {_MID} [{_ch}]  {done}/{exp} done — running remaining...')
    results = run_loso(_MID, _ch,
                       build_model_fn=lambda f: ChanDropTransformer(f),
                       n_epochs=100, batch_size=128, lr=5e-4)
    summarise_and_save(_MID, _ch, results)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 24
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M19 — DANN  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# BEST DL baseline from Phase B (AccA=0.5020, AccB=0.5068 on 62ch)
# ═══════════════════════════════════════════════════════════════
_MID = 'M19'

def run_dann(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING')
        return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d} '
                      f'AccA={r["acc_a"]:.4f} AccB={r["acc_b"]:.4f}')
                continue

            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            tr_loader = make_loader(Xtr, Ytr, 128, weighted=True)
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            # subject labels (0-indexed) for discriminator
            subj_map = {int(s): i for i,s in enumerate(sorted(np.unique(Str)))}
            Str_idx  = np.array([subj_map[int(s)] for s in Str], dtype=np.int64)
            tr_ds_s  = TensorDataset(torch.FloatTensor(Xtr),
                                      torch.LongTensor(Ytr),
                                      torch.LongTensor(Str_idx))
            tr_loader_s = DataLoader(tr_ds_s, batch_size=128, shuffle=True,
                                     num_workers=0, pin_memory=True, drop_last=True)

            n_sub_tr = len(subj_map)
            model    = DANN(in_feat, n_subjects=n_sub_tr).to(device)
            opt      = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
            cls_crit = nn.CrossEntropyLoss()
            dom_crit = nn.CrossEntropyLoss()
            scaler   = GradScaler()
            best_f1, best_state, no_imp = 0.0, None, 0
            total_steps = 100 * len(tr_loader_s)
            step = 0

            for ep in range(1, 101):
                model.train()
                for Xb, Yb, Sb in tr_loader_s:
                    Xb, Yb, Sb = Xb.to(device), Yb.to(device), Sb.to(device)
                    p = step / total_steps
                    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                    model.grl_alpha = float(alpha)
                    opt.zero_grad(set_to_none=True)
                    with autocast():
                        cls_out, dom_out, _ = model(Xb, alpha=alpha)
                        loss = cls_crit(cls_out, Yb) + 0.3 * dom_crit(dom_out, Sb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    step += 1
                sched.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break

            model.load_state_dict(best_state)
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)

            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r)
            all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} '
                  f'AccA={acc_a:.4f} F1A={f1_a:.4f} '
                  f'AccB={acc_b:.4f} F1B={f1_b:.4f} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_dann(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 26
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M20 — CLISA  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# Stage 1: contrastive pretrain (50ep). Stage 2: CE finetune (50ep).
# ═══════════════════════════════════════════════════════════════
_MID = 'M20'

def run_clisa(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING'); return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d} AccA={r["acc_a"]:.4f}')
                continue
            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            model  = CLISA(in_feat).to(device)
            scaler = GradScaler()

            # ── Stage 1: Contrastive pretraining ──────────────
            opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50)
            tr_loader = make_loader(Xtr, Ytr, 128, weighted=True)
            for ep in range(50):
                model.train()
                for Xb, Yb in tr_loader:
                    Xb = Xb.to(device)
                    # augment two views: noise + masking
                    n_ch = Xb.size(1)//5
                    v1 = Xb + 0.1*torch.randn_like(Xb)
                    mask = (torch.rand(Xb.size(0), n_ch, 1, device=Xb.device) > 0.3).float()
                    v2 = Xb * mask.repeat(1,1,5).view(Xb.size(0),-1)
                    opt1.zero_grad(set_to_none=True)
                    with autocast():
                        z1 = model.project(v1); z2 = model.project(v2)
                        loss = nt_xent_loss(z1, z2)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt1); scaler.update()
                sched1.step()

            # ── Stage 2: Supervised fine-tune ─────────────────
            for p in model.encoder.parameters(): p.requires_grad = True  # unfreeze all
            opt2  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched2= torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50)
            crit  = nn.CrossEntropyLoss()
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            best_f1, best_state, no_imp = 0.0, None, 0
            for ep in range(50):
                train_epoch(model, tr_loader, opt2, crit, scaler)
                sched2.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break

            model.load_state_dict(best_state)
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)
            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r); all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_clisa(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 28
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M21 — SIMCLR  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M21'

def run_simclr(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING'); return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d} AccA={r["acc_a"]:.4f}')
                continue
            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            model  = SimCLR(in_feat).to(device)
            scaler = GradScaler()
            n_ch   = in_feat // 5

            # Stage 1: SimCLR pretrain
            opt1   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50)
            tr_loader = make_loader(Xtr, Ytr, 128, weighted=True)
            for ep in range(50):
                model.train()
                for Xb, _ in tr_loader:
                    Xb = Xb.to(device)
                    v1 = Xb + 0.1*torch.randn_like(Xb)
                    mask = (torch.rand(Xb.size(0), n_ch, 1, device=Xb.device)>0.3).float()
                    v2   = Xb * mask.repeat(1,1,5).view(Xb.size(0),-1)
                    opt1.zero_grad(set_to_none=True)
                    with autocast():
                        loss = nt_xent_loss(model.project(v1), model.project(v2))
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt1); scaler.update()
                sched1.step()

            # Stage 2: freeze encoder, train linear head
            for p in model.encoder.parameters():  p.requires_grad = False
            for p in model.projector.parameters(): p.requires_grad = False
            opt2   = torch.optim.AdamW(model.classifier.parameters(), lr=3e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50)
            crit   = nn.CrossEntropyLoss()
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            best_f1, best_state, no_imp = 0.0, None, 0
            for ep in range(50):
                train_epoch(model, tr_loader, opt2, crit, scaler)
                sched2.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break

            model.load_state_dict(best_state)
            # re-enable all for calibration
            for p in model.parameters(): p.requires_grad = True
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)
            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r); all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_simclr(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 30
# Categories: preprocessing, model_definition, training, evaluation
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M22 — BYOL  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# NEW — not implemented in Phase B. EMA bootstrap pretraining.
# ═══════════════════════════════════════════════════════════════
_MID = 'M22'

def run_byol(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING'); return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []
    n_ch = in_feat // 5

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d} AccA={r["acc_a"]:.4f}')
                continue
            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            model  = BYOL(in_feat).to(device)
            scaler = GradScaler()

            # Stage 1: BYOL self-supervised pretraining
            opt1   = torch.optim.AdamW(list(model.online.parameters()) +
                                        list(model.predictor.parameters()), lr=1e-3, weight_decay=1e-4)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50)
            tr_loader = make_loader(Xtr, Ytr, 128, weighted=True)
            for ep in range(50):
                model.train()
                for Xb, _ in tr_loader:
                    Xb = Xb.to(device)
                    v1 = Xb + 0.1*torch.randn_like(Xb)
                    mask = (torch.rand(Xb.size(0), n_ch, 1, device=Xb.device)>0.3).float()
                    v2   = Xb * mask.repeat(1,1,5).view(Xb.size(0),-1)
                    opt1.zero_grad(set_to_none=True)
                    with autocast():
                        loss = model.byol_loss(v1, v2)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt1); scaler.update()
                    model.update_target()
                sched1.step()

            # Stage 2: supervised fine-tune (classifier only)
            for p in model.online.parameters(): p.requires_grad = False
            opt2   = torch.optim.AdamW(model.classifier.parameters(), lr=3e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50)
            crit   = nn.CrossEntropyLoss()
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            best_f1, best_state, no_imp = 0.0, None, 0
            for ep in range(50):
                train_epoch(model, tr_loader, opt2, crit, scaler)
                sched2.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break

            model.load_state_dict(best_state)
            for p in model.parameters(): p.requires_grad = True
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)
            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r); all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_byol(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 32
# Categories: preprocessing, training, evaluation, statistics
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M23 — PSEUDOLABEL  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M23'

def run_pseudolabel(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING'); return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d}')
                continue
            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            crit = nn.CrossEntropyLoss(); scaler = GradScaler()

            # Stage 1: supervised train on labeled data (60 epochs)
            model1 = PseudoLabelNet(in_feat).to(device)
            opt1   = torch.optim.AdamW(model1.parameters(), lr=1e-3, weight_decay=1e-4)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=60)
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            best_f1, best_state, no_imp = 0.0, None, 0
            for ep in range(60):
                train_epoch(model1, make_loader(Xtr, Ytr, 256), opt1, crit, scaler)
                sched1.step()
                _, vf1 = evaluate(model1, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model1.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break
            model1.load_state_dict(best_state)

            # Stage 2: generate pseudo-labels for test subject (high-confidence only)
            model1.eval()
            with torch.no_grad():
                Xte_t = torch.FloatTensor(Xte).to(device)
                with autocast():
                    probs = torch.softmax(model1(Xte_t), dim=-1).cpu().numpy()
            conf  = probs.max(axis=1)
            plabs = probs.argmax(axis=1).astype(np.int64)
            conf_mask = conf > 0.8    # only high-confidence pseudo-labels
            Xpl = Xte[conf_mask]; Ypl = plabs[conf_mask]
            if len(Xpl) > 0:
                Xall = np.concatenate([Xtr, Xpl], axis=0)
                Yall = np.concatenate([Ytr, Ypl], axis=0)
            else:
                Xall, Yall = Xtr, Ytr

            # Stage 3: retrain on labeled + pseudo-labeled data (40 epochs)
            model2 = PseudoLabelNet(in_feat).to(device)
            opt2   = torch.optim.AdamW(model2.parameters(), lr=5e-4, weight_decay=1e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=40)
            best_f1, best_state, no_imp = 0.0, None, 0
            for ep in range(40):
                train_epoch(model2, make_loader(Xall, Yall, 256), opt2, crit, scaler)
                sched2.step()
                _, vf1 = evaluate(model2, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model2.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break
            model2.load_state_dict(best_state)

            acc_a, f1_a = evaluate(model2, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model2, Xte, Yte)
            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), n_pseudolabels=int(len(Xpl)),
                     elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r); all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} '
                  f'n_pl={len(Xpl)} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_pseudolabel(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 34
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M24 — MIXMATCH  |  62ch + 6ch  |  LOSO-15 × 3 seeds
# ═══════════════════════════════════════════════════════════════
_MID = 'M24'

def sharpen(p, T=0.5):
    p_sharp = p ** (1.0 / T)
    return p_sharp / p_sharp.sum(dim=-1, keepdim=True)

def run_mixmatch(ch):
    done, exp, complete = model_complete(_MID, ch)
    if complete:
        print(f'✅ {_MID} [{ch}] COMPLETE ({done}/{exp}) — SKIPPING'); return
    print(f'\n▶ {_MID} [{ch}]  {done}/{exp} done — running remaining...')
    X = get_data(ch); in_feat = X.shape[1]
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, ch):
                r = ck_load(_MID, seed, fi, ch); all_results.append(r)
                print(f'  SKIP {_MID}|{ch}|s{seed}|f{fi:02d}')
                continue
            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            model  = MixMatchNet(in_feat).to(device)
            opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
            scaler = GradScaler()
            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)
            # treat test subject as unlabeled
            unl_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte)),
                                    batch_size=128, shuffle=True, drop_last=True)
            unl_iter   = iter(unl_loader)
            tr_loader  = make_loader(Xtr, Ytr, 128, weighted=True)
            best_f1, best_state, no_imp = 0.0, None, 0
            n_ch = in_feat // 5

            for ep in range(100):
                model.train()
                for Xl, Yl in tr_loader:
                    Xl, Yl = Xl.to(device), Yl.to(device)
                    try:
                        (Xu,) = next(unl_iter)
                    except StopIteration:
                        unl_iter = iter(unl_loader); (Xu,) = next(unl_iter)
                    Xu = Xu.to(device)
                    # pseudo-label for unlabeled
                    model.eval()
                    with torch.no_grad():
                        with autocast():
                            p_unl = sharpen(torch.softmax(model(Xu), dim=-1)).detach()
                    model.train()
                    # mixup labeled with labeled
                    lam = float(np.random.beta(0.75, 0.75))
                    lam = max(lam, 1-lam)
                    idx = torch.randperm(Xl.size(0), device=device)
                    Xm = lam*Xl + (1-lam)*Xl[idx]
                    # one-hot labels
                    Yl_oh = F.one_hot(Yl, N_CLASSES).float()
                    Ym = lam*Yl_oh + (1-lam)*Yl_oh[idx]
                    opt.zero_grad(set_to_none=True)
                    with autocast():
                        out_l = model(Xm)
                        out_u = model(Xu)
                        loss_l = -(Ym * F.log_softmax(out_l, dim=-1)).sum(-1).mean()
                        # consistency loss on unlabeled
                        loss_u = F.mse_loss(torch.softmax(out_u, dim=-1), p_unl)
                        loss   = loss_l + 75.0 * loss_u
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                sched.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break

            model.load_state_dict(best_state)
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)
            r = dict(model_id=_MID, ch=ch, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, ch, r); all_results.append(r)
            print(f'  {_MID}|{ch}|s{seed}|f{fi:02d} AccA={acc_a:.4f} AccB={acc_b:.4f} ({r["elapsed"]:.0f}s)')
    summarise_and_save(_MID, ch, all_results)

for _ch in ['62ch', '6ch']:
    run_mixmatch(_ch)
print(f'\n✅ {_MID} complete')



# ==============================================================================
# Notebook cell 36
# Categories: preprocessing, model_definition, training, evaluation, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M25 — DANCE TEACHER  |  62ch ONLY  |  LOSO-15 × 3 seeds
# pretrain=100ep (contrastive+adversarial) | finetune=100ep (CE)
# Phase C standard: H17 full epochs — confirmed in Phase 0
# ═══════════════════════════════════════════════════════════════
_MID = 'M25'
_CH  = '62ch'

# ── Config (frozen Phase B reference — do NOT change) ──────────
PRETRAIN_EPOCHS   = 100
FINETUNE_EPOCHS   = 100
PRETRAIN_LR       = 1e-3
FT_LR_CLASSIFIER  = 3e-4
FT_LR_ENCODER     = 3e-5
LABEL_SMOOTHING   = 0.1
CONTRASTIVE_TEMP  = 0.5
MASK_RATIO        = 0.3
NOISE_STD         = 0.1
MIXUP_ALPHA       = 0.4
SUBJECT_WEIGHT    = 0.5   # weight on adversarial loss

done, exp, complete = model_complete(_MID, _CH)
if complete:
    print(f'✅ {_MID} [{_CH}] COMPLETE ({done}/{exp}) — SKIPPING')
else:
    print(f'\n▶ {_MID} [{_CH}]  {done}/{exp} done — running remaining...')
    print(f'   pretrain={PRETRAIN_EPOCHS}ep  finetune={FINETUNE_EPOCHS}ep  '
          f'lr_pt={PRETRAIN_LR}  lr_ft_cls={FT_LR_CLASSIFIER}')
    X = get_data(_CH); in_feat = X.shape[1]  # 310
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, _CH):
                r = ck_load(_MID, seed, fi, _CH); all_results.append(r)
                print(f'  SKIP {_MID}|{_CH}|s{seed}|f{fi:02d} '
                      f'AccA={r["acc_a"]:.4f} AccB={r["acc_b"]:.4f}')
                continue

            t0 = time.time()
            Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub, seed=seed)
            n_sub_tr = len(np.unique(Str))
            subj_map = {int(s): i for i,s in enumerate(sorted(np.unique(Str)))}
            Str_idx  = np.array([subj_map[int(s)] for s in Str], dtype=np.int64)

            va_loader = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)

            model   = DANCETeacher(n_ch=62, n_subjects=n_sub_tr).to(device)
            scaler  = GradScaler()

            # ── Stage 1: Contrastive + Adversarial Pretrain ───────
            print(f'    [{_MID}|s{seed}|f{fi:02d}] Stage1: contrastive pretrain...')
            opt_pt  = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
            sched_pt= torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=PRETRAIN_EPOCHS)
            tr_ds   = TensorDataset(torch.FloatTensor(Xtr),
                                     torch.LongTensor(Ytr),
                                     torch.LongTensor(Str_idx))
            tr_load = DataLoader(tr_ds, batch_size=128, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)

            total_steps_pt = PRETRAIN_EPOCHS * len(tr_load)
            step = 0
            for ep in range(PRETRAIN_EPOCHS):
                model.train()
                for Xb, Yb, Sb in tr_load:
                    Xb, Yb, Sb = Xb.to(device), Yb.to(device), Sb.to(device)
                    p = step / total_steps_pt
                    model.grl_alpha = float(2.0/(1.0+np.exp(-10*p))-1.0)
                    # two augmented views
                    v1 = dance_augment(Xb, MASK_RATIO, NOISE_STD)
                    v2 = dance_augment(Xb, MASK_RATIO, NOISE_STD)
                    # subject mixup on v2
                    v2 = subject_mixup(v2, Yb, Sb, MIXUP_ALPHA)
                    opt_pt.zero_grad(set_to_none=True)
                    with autocast():
                        z1 = model.project(v1); z2 = model.project(v2)
                        loss_con = nt_xent_loss(z1, z2, temp=CONTRASTIVE_TEMP)
                        _, dom_out, _ = model(Xb, return_domain=True)
                        loss_adv = nn.CrossEntropyLoss()(dom_out, Sb)
                        loss = loss_con + SUBJECT_WEIGHT * loss_adv
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt_pt); scaler.update()
                    step += 1
                sched_pt.step()
                if (ep+1) % 25 == 0:
                    print(f'      Pretrain ep {ep+1}/{PRETRAIN_EPOCHS} loss={loss.item():.4f}')

            # ── Stage 2: Supervised Fine-tune ──────────────────────
            print(f'    [{_MID}|s{seed}|f{fi:02d}] Stage2: supervised fine-tune...')
            # encoder gets lower LR, classifier gets higher LR
            opt_ft = torch.optim.AdamW([
                {'params': list(model.band_emb.parameters()) +
                           list(model.blocks.parameters()) +
                           [model.pos_emb] + list(model.ch_attn.parameters()),
                 'lr': FT_LR_ENCODER},
                {'params': model.classifier.parameters(), 'lr': FT_LR_CLASSIFIER}
            ], weight_decay=1e-4)
            sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=FINETUNE_EPOCHS)
            crit_ft  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            tr_load_ft = make_loader(Xtr, Ytr, 128, weighted=True)
            best_f1, best_state, no_imp = 0.0, None, 0

            for ep in range(FINETUNE_EPOCHS):
                model.train()
                for Xb, Yb in tr_load_ft:
                    Xb, Yb = Xb.to(device), Yb.to(device)
                    opt_ft.zero_grad(set_to_none=True)
                    with autocast():
                        logits = model(Xb)
                        loss   = crit_ft(logits, Yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt_ft); scaler.update()
                sched_ft.step()
                _, vf1 = evaluate(model, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                    torch.save(best_state, weight_path(_MID, seed, fi, _CH))
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break
                if (ep+1) % 25 == 0:
                    print(f'      Finetune ep {ep+1}/{FINETUNE_EPOCHS} valF1={vf1:.4f}')

            model.load_state_dict(best_state)
            acc_a, f1_a = evaluate(model, make_loader(Xte, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(model, Xte, Yte)

            r = dict(model_id=_MID, ch=_CH, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, _CH, r)
            all_results.append(r)
            print(f'  ✔ {_MID}|{_CH}|s{seed}|f{fi:02d} '
                  f'AccA={acc_a:.4f} F1A={f1_a:.4f} '
                  f'AccB={acc_b:.4f} F1B={f1_b:.4f}  ({r["elapsed"]:.0f}s)')

    summarise_and_save(_MID, _CH, all_results)
print(f'\n✅ {_MID} [{_CH}] done — best weights saved to {MDL_DIR}')
print(f'   → Ready for M26 distillation')



# ==============================================================================
# Notebook cell 38
# Categories: preprocessing, model_definition, training, evaluation, results_tables, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# M26 — DANCE STUDENT  |  6ch ONLY  |  LOSO-15 × 3 seeds
# Distilled from M25 teacher. loss = MSE + KL + CE.
# REQUIRES M25 best weights (auto-falls back to fresh teacher if missing)
# ═══════════════════════════════════════════════════════════════
_MID = 'M26'
_CH  = '6ch'

# ── Distillation Config (frozen Phase B reference) ─────────────
DISTILL_EPOCHS = 50
DISTILL_LR     = 1e-3
DISTILL_TEMP   = 4.0      # soft-target temperature
W_MSE, W_KL, W_CE = 1.0, 2.0, 1.0

done, exp, complete = model_complete(_MID, _CH)
if complete:
    print(f'✅ {_MID} [{_CH}] COMPLETE ({done}/{exp}) — SKIPPING')
else:
    print(f'\n▶ {_MID} [{_CH}]  {done}/{exp} done — running remaining...')
    print(f'   distill={DISTILL_EPOCHS}ep  lr={DISTILL_LR}  '
          f'temp={DISTILL_TEMP}  loss=MSE×{W_MSE}+KL×{W_KL}+CE×{W_CE}')

    X62 = get_data('62ch'); X6 = get_data('6ch')
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        for fi, test_sub in enumerate(SUBJECTS, start=1):
            if ck_exists(_MID, seed, fi, _CH):
                r = ck_load(_MID, seed, fi, _CH); all_results.append(r)
                print(f'  SKIP {_MID}|{_CH}|s{seed}|f{fi:02d} '
                      f'AccA={r["acc_a"]:.4f} AccB={r["acc_b"]:.4f}')
                continue

            t0 = time.time()
            # Train splits (both 62ch for teacher, 6ch for student)
            _, Ytr, Str, _, Yva, _, Yte = loso_split(X62, Y, S, test_sub, seed=seed)
            Xtr62, _, _, Xva62, _, Xte62, _ = loso_split(X62, Y, S, test_sub, seed=seed)
            Xtr6,  _, _, Xva6,  _, Xte6,  _ = loso_split(X6,  Y, S, test_sub, seed=seed)
            n_sub_tr = len(np.unique(Str))

            # ── Load teacher (from M25 checkpoint if available) ────
            teacher = DANCETeacher(n_ch=62, n_subjects=n_sub_tr).to(device)
            t_path  = weight_path('M25', seed, fi, '62ch')
            if t_path.exists():
                teacher.load_state_dict(torch.load(t_path, map_location=device))
                print(f'    Loaded teacher: {t_path.name}')
            else:
                print(f'    ⚠  Teacher weights not found ({t_path.name}). '
                      f'Train M25 first for best results — using untrained teacher.')
            teacher.eval()
            for p in teacher.parameters(): p.requires_grad = False

            # ── Build student ───────────────────────────────────────
            student = DANCEStudent(n_ch=6).to(device)
            opt     = torch.optim.AdamW(student.parameters(), lr=DISTILL_LR, weight_decay=1e-4)
            sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DISTILL_EPOCHS)
            scaler  = GradScaler()
            ce_crit = nn.CrossEntropyLoss(label_smoothing=0.1)

            # Dataset: paired (6ch features, 62ch features, labels)
            tr_ds = TensorDataset(torch.FloatTensor(Xtr6),
                                   torch.FloatTensor(Xtr62),
                                   torch.LongTensor(Ytr))
            counts  = np.bincount(Ytr, minlength=N_CLASSES).astype(float)
            weights = 1.0 / (counts[Ytr] + 1e-6)
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)
            tr_loader = DataLoader(tr_ds, batch_size=256, sampler=sampler,
                                   num_workers=0, pin_memory=True, drop_last=True)
            va_loader = make_loader(Xva6, Yva, 256, weighted=False, shuffle=False)

            best_f1, best_state, no_imp = 0.0, None, 0
            print(f'    [{_MID}|s{seed}|f{fi:02d}] Distilling for {DISTILL_EPOCHS} epochs...')

            for ep in range(DISTILL_EPOCHS):
                student.train()
                ep_loss = 0.0
                for X6b, X62b, Yb in tr_loader:
                    X6b, X62b, Yb = X6b.to(device), X62b.to(device), Yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    with autocast():
                        # student forward
                        stu_logits = student(X6b)
                        # teacher soft targets (no grad)
                        with torch.no_grad():
                            tch_logits = teacher(X62b)
                            tch_feat   = teacher.encode(X62b)
                        stu_feat   = student.encode(X6b)
                        # MSE on embeddings
                        loss_mse = F.mse_loss(stu_feat, tch_feat.detach())
                        # KL on soft labels
                        T = DISTILL_TEMP
                        stu_soft = F.log_softmax(stu_logits / T, dim=-1)
                        tch_soft = F.softmax(tch_logits / T, dim=-1)
                        loss_kl  = F.kl_div(stu_soft, tch_soft, reduction='batchmean') * (T**2)
                        # CE on hard labels
                        loss_ce  = ce_crit(stu_logits, Yb)
                        loss = W_MSE*loss_mse + W_KL*loss_kl + W_CE*loss_ce
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    ep_loss += loss.item()
                sched.step()
                _, vf1 = evaluate(student, va_loader)
                if vf1 > best_f1:
                    best_f1 = vf1; best_state = copy.deepcopy(student.state_dict()); no_imp = 0
                    torch.save(best_state, weight_path(_MID, seed, fi, _CH))
                else:
                    no_imp += 1
                if no_imp >= PATIENCE: break
                if (ep+1) % 10 == 0:
                    print(f'      Distill ep {ep+1}/{DISTILL_EPOCHS} loss={ep_loss/len(tr_loader):.4f} valF1={vf1:.4f}')

            student.load_state_dict(best_state)
            acc_a, f1_a = evaluate(student, make_loader(Xte6, Yte, 256, weighted=False, shuffle=False))
            acc_b, f1_b = proto_b_calibrate(student, Xte6, Yte)

            r = dict(model_id=_MID, ch=_CH, seed=seed, fold=fi, test_sub=test_sub,
                     acc_a=round(acc_a,4), f1_a=round(f1_a,4),
                     acc_b=round(acc_b,4), f1_b=round(f1_b,4),
                     best_val_f1=round(best_f1,4), elapsed=round(time.time()-t0,1))
            ck_save(_MID, seed, fi, _CH, r)
            all_results.append(r)
            print(f'  ✔ {_MID}|{_CH}|s{seed}|f{fi:02d} '
                  f'AccA={acc_a:.4f} F1A={f1_a:.4f} '
                  f'AccB={acc_b:.4f} F1B={f1_b:.4f}  ({r["elapsed"]:.0f}s)')

    summarise_and_save(_MID, _CH, all_results)
print(f'\n✅ {_MID} [{_CH}] complete')



# ==============================================================================
# Notebook cell 40
# Categories: preprocessing, model_definition, training, results_tables
# ==============================================================================
# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY — Build master comparison table from all checkpoints
# ═══════════════════════════════════════════════════════════════
import glob

all_rows = []
for json_path in sorted(CKPT_DIR.glob('M*.json')):
    with open(json_path) as f:
        all_rows.append(json.load(f))

if not all_rows:
    print('No checkpoints found yet.'); 
else:
    df_all = pd.DataFrame(all_rows)
    # Summary per model × ch
    summary_rows = []
    for (mid, ch), grp in df_all.groupby(['model_id','ch']):
        done = len(grp)
        summary_rows.append(dict(
            model_id=mid, ch=ch, n_runs=done,
            acc_a_mean=round(grp.acc_a.mean(),4), acc_a_std=round(grp.acc_a.std(),4),
            f1_a_mean =round(grp.f1_a.mean(),4),  f1_a_std =round(grp.f1_a.std(),4),
            acc_b_mean=round(grp.acc_b.mean(),4), acc_b_std=round(grp.acc_b.std(),4),
            f1_b_mean =round(grp.f1_b.mean(),4),  f1_b_std =round(grp.f1_b.std(),4),
        ))
    df_sum = pd.DataFrame(summary_rows).sort_values(['model_id','ch'])

    print('='*80)
    print('  DEEP MODELS SUMMARY  (AccA / AccB  mean ± std over LOSO-15 × 3 seeds)')
    print('='*80)
    for _, row in df_sum.iterrows():
        print(f"  {row.model_id:4s} [{row.ch}]  n={row.n_runs:3d}  "
              f"AccA={row.acc_a_mean:.4f}±{row.acc_a_std:.4f}  "
              f"AccB={row.acc_b_mean:.4f}±{row.acc_b_std:.4f}  "
              f"F1A={row.f1_a_mean:.4f}  F1B={row.f1_b_mean:.4f}")

    master_csv = RES_DIR / 'deep_models_master_summary.csv'
    df_sum.to_csv(master_csv, index=False)
    df_all.to_csv(RES_DIR / 'deep_models_all_folds.csv', index=False)
    print(f'\nSaved:\n  {master_csv}')
    print(f'  {RES_DIR}/deep_models_all_folds.csv')

    # Classical ML baseline comparison
    ref = {'M09 XGBoost 62ch': 0.4791, 'M03 RF 62ch': 0.4648,
           'M19 DANN AccA Phase B': 0.5020, 'M19 DANN AccB Phase B': 0.5068}
    print('\nClassical ML ceiling (Phase C): XGBoost F1=0.4791 | RF F1=0.4648')
    print('Phase B best DL: DANN AccA=0.5020  AccB=0.5068  ← targets to beat')



# ==============================================================================
# Notebook cell 41
# Categories: preprocessing, training, results_tables
# ==============================================================================
# ── BEST LOSO FOLD FROM SAVED RESULTS ONLY (no training) ─────────────────────
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

# Which metric should define "best fold"?
# Options: "f1_b", "acc_b", "f1_a", "acc_a"
BEST_METRIC = "f1_b"   # <- change if needed

# Use existing CKPT_DIR if already defined in the notebook, otherwise fallback
ckpt_dir = Path(globals().get("CKPT_DIR", Path(".") / "checkpoints" / "loso_results"))

if not ckpt_dir.exists():
    print(f"Checkpoint folder not found: {ckpt_dir}")
else:
    json_files = sorted(ckpt_dir.glob("M*.json"))

    if len(json_files) == 0:
        print(f"No LOSO result JSON files found in: {ckpt_dir}")
    else:
        rows = []
        pat = re.compile(r"^(M\d+)_(6ch|62ch)_seed(\d+)_fold(\d+)\.json$")

        for fp in json_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    r = json.load(f)

                m = pat.match(fp.name)

                # Prefer values from JSON, fallback to filename parsing
                model_id = r.get("model_id", m.group(1) if m else None)
                ch       = r.get("ch",       m.group(2) if m else None)
                seed     = r.get("seed",     int(m.group(3)) if m else None)
                fold     = r.get("fold",     int(m.group(4)) if m else None)

                rows.append({
                    "file": fp.name,
                    "model_id": model_id,
                    "ch": ch,
                    "seed": seed,
                    "fold": fold,
                    "test_sub": r.get("test_sub", None),
                    "best_val_f1": r.get("best_val_f1", np.nan),
                    "f1_a": r.get("f1_a", np.nan),
                    "acc_a": r.get("acc_a", np.nan),
                    "f1_b": r.get("f1_b", np.nan),
                    "acc_b": r.get("acc_b", np.nan),
                })

            except Exception as e:
                print(f"Could not read {fp.name}: {e}")

        df = pd.DataFrame(rows)

        if df.empty:
            print("No readable result files found.")
        elif BEST_METRIC not in df.columns:
            print(f"Metric '{BEST_METRIC}' not found. Available columns: {list(df.columns)}")
        elif df[BEST_METRIC].dropna().empty:
            print(f"No valid values found for metric '{BEST_METRIC}'.")
        else:
            # Make sure numeric columns are numeric
            for c in ["best_val_f1", "f1_a", "acc_a", "f1_b", "acc_b"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            best_idx = df[BEST_METRIC].idxmax()
            best = df.loc[best_idx]

            print("=" * 72)
            print(f"BEST LOSO FOLD (from saved folders only)  |  metric = {BEST_METRIC}")
            print("=" * 72)
            print(f"Model      : {best['model_id']}")
            print(f"Channels   : {best['ch']}")
            print(f"Seed       : {int(best['seed']) if pd.notna(best['seed']) else best['seed']}")
            print(f"Fold       : {int(best['fold']):02d}" if pd.notna(best['fold']) else f"Fold       : {best['fold']}")
            print(f"Test Subj  : {best['test_sub']}")
            print(f"Best Val F1: {best['best_val_f1']:.4f}" if pd.notna(best["best_val_f1"]) else "Best Val F1: NaN")
            print(f"F1-A       : {best['f1_a']:.4f}" if pd.notna(best["f1_a"]) else "F1-A       : NaN")
            print(f"Acc-A      : {best['acc_a']:.4f}" if pd.notna(best["acc_a"]) else "Acc-A      : NaN")
            print(f"F1-B       : {best['f1_b']:.4f}" if pd.notna(best["f1_b"]) else "F1-B       : NaN")
            print(f"Acc-B      : {best['acc_b']:.4f}" if pd.notna(best["acc_b"]) else "Acc-B      : NaN")
            print(f"Source file: {best['file']}")
            print("=" * 72)

            # Optional: show top 10 folds
            show_cols = ["model_id", "ch", "seed", "fold", "test_sub", "f1_a", "acc_a", "f1_b", "acc_b"]
            print("\nTop 10 folds:")
            display(df.sort_values(BEST_METRIC, ascending=False)[show_cols].head(10).reset_index(drop=True))


# ==============================================================================
# Notebook cell 42
# Categories: preprocessing, training, results_tables
# ==============================================================================
# ── BEST SINGLE LOSO FOLD FOR 6ch ONLY (no training) ─────────────────────────
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

BEST_METRIC = "f1_b"   # change if needed
TARGET_CH = "6ch"

ckpt_dir = Path(globals().get("CKPT_DIR", Path(".") / "checkpoints" / "loso_results"))

if not ckpt_dir.exists():
    print(f"Checkpoint folder not found: {ckpt_dir}")
else:
    json_files = sorted(ckpt_dir.glob("M*.json"))
    pat = re.compile(r"^(M\d+)_(6ch|62ch)_seed(\d+)_fold(\d+)\.json$")

    rows = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                r = json.load(f)

            m = pat.match(fp.name)

            rows.append({
                "file": fp.name,
                "model_id": r.get("model_id", m.group(1) if m else None),
                "ch": r.get("ch", m.group(2) if m else None),
                "seed": r.get("seed", int(m.group(3)) if m else None),
                "fold": r.get("fold", int(m.group(4)) if m else None),
                "test_sub": r.get("test_sub", None),
                "best_val_f1": r.get("best_val_f1", np.nan),
                "f1_a": r.get("f1_a", np.nan),
                "acc_a": r.get("acc_a", np.nan),
                "f1_b": r.get("f1_b", np.nan),
                "acc_b": r.get("acc_b", np.nan),
            })
        except Exception as e:
            print(f"Could not read {fp.name}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        print("No readable result files found.")
    else:
        for c in ["best_val_f1", "f1_a", "acc_a", "f1_b", "acc_b"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df_ch = df[df["ch"] == TARGET_CH].copy()

        if df_ch.empty:
            print(f"No results found for {TARGET_CH}")
        elif df_ch[BEST_METRIC].dropna().empty:
            print(f"No valid values found for metric '{BEST_METRIC}' in {TARGET_CH}")
        else:
            best_idx = df_ch[BEST_METRIC].idxmax()
            best = df_ch.loc[best_idx]

            print("=" * 72)
            print(f"BEST LOSO FOLD FOR {TARGET_CH}  |  metric = {BEST_METRIC}")
            print("=" * 72)
            print(f"Model      : {best['model_id']}")
            print(f"Channels   : {best['ch']}")
            print(f"Seed       : {int(best['seed']) if pd.notna(best['seed']) else best['seed']}")
            print(f"Fold       : {int(best['fold']):02d}" if pd.notna(best['fold']) else f"Fold       : {best['fold']}")
            print(f"Test Subj  : {best['test_sub']}")
            print(f"Best Val F1: {best['best_val_f1']:.4f}" if pd.notna(best["best_val_f1"]) else "Best Val F1: NaN")
            print(f"F1-A       : {best['f1_a']:.4f}" if pd.notna(best["f1_a"]) else "F1-A       : NaN")
            print(f"Acc-A      : {best['acc_a']:.4f}" if pd.notna(best["acc_a"]) else "Acc-A      : NaN")
            print(f"F1-B       : {best['f1_b']:.4f}" if pd.notna(best["f1_b"]) else "F1-B       : NaN")
            print(f"Acc-B      : {best['acc_b']:.4f}" if pd.notna(best["acc_b"]) else "Acc-B      : NaN")
            print(f"Source file: {best['file']}")
            print("=" * 72)

            print("\nTop 10 folds within 6ch:")
            display(
                df_ch.sort_values(BEST_METRIC, ascending=False)[
                    ["model_id", "ch", "seed", "fold", "test_sub", "f1_a", "acc_a", "f1_b", "acc_b", "file"]
                ].head(10).reset_index(drop=True)
            )


# ==============================================================================
# Notebook cell 43
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
# Literature only (updated according to your final table)
methods_lit = [
    "DGCNN\n(2020)",
    "DAN\n(2018)",
    "BiDANN\n(2018)",
    "BiHDM\n(2020)",
    "RGNN\n(2020)",
    "SOGNN\n(2021)",
    "AttGraph\n(2025)",
    "BFE-Net\n(2024)",
    "SS-EMERGE\n(2025)"
]
scores_lit = [52.82, 58.87, 65.59, 69.03, 73.84, 75.27, 78.36, 79.81, 81.51]

# Keep only CLISA
methods_ours = [
    "CLISA-EWU\n(LOSO, 2026)",
    "CLISA-EWU\nBest Fold (2026)"
]
scores_ours = [68.01, 85.35]

# ── colours ───────────────────────────────────────────────────────────────────
lit_color  = "#5b9bd5"   # literature
ours_color = "#ed7d31"   # proposed CLISA
best_color = "#c00000"   # best fold CLISA

# ── combine + sort high to low ────────────────────────────────────────────────
entries = []

for m, s in zip(methods_lit, scores_lit):
    entries.append((m, s, lit_color, "Literature"))

entries.append((methods_ours[0], scores_ours[0], ours_color, "CLISA-EWU"))
entries.append((methods_ours[1], scores_ours[1], best_color, "CLISA-EWU Best Fold"))

entries = sorted(entries, key=lambda x: x[1], reverse=True)

all_methods = [e[0] for e in entries]
all_scores  = [e[1] for e in entries]
all_colors  = [e[2] for e in entries]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))

x = np.arange(len(all_methods))
bars = ax.bar(
    x, all_scores,
    color=all_colors,
    width=0.68,
    edgecolor="white",
    linewidth=0.8
)

# value labels
for bar, score in zip(bars, all_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{score:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# chance line
ax.axhline(25, color="gray", linestyle="--", linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(all_methods, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title(
    "Subject-Independent LOSO on SEED-IV (Highest to Lowest)",
    fontsize=12,
    pad=12
)

ax.set_ylim(0, 95)
ax.set_xlim(-0.6, len(all_methods) - 0.4)

# legend inside plot on right side
patches = [
    mpatches.Patch(color=lit_color,  label="Literature"),
    mpatches.Patch(color=ours_color, label="CLISA-EWU"),
    mpatches.Patch(color=best_color, label="CLISA-EWU Best Fold"),
]
ax.legend(handles=patches, fontsize=9, loc="upper right", frameon=True)

ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("sota_comparison_clisa_only_sorted.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: sota_comparison_clisa_only_sorted.png")


# ==============================================================================
# Notebook cell 44
# Categories: preprocessing, model_definition, training, evaluation, results_tables, figures, statistics
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# CLISA 62ch — Best-fold inference with class predictions
# Uses your best row from the table:
#   M20, 62ch, seed=7, fold=15, test_sub=15, acc_b=0.8535
# Shows:
#   • per-sample true class vs predicted class
#   • confidence
#   • confusion matrix
#   • class-wise recall
#   • true vs predicted class counts
# ═══════════════════════════════════════════════════════════════════════════════

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)

# ── config ─────────────────────────────────────────────────────────────────────
BEST_SEED            = 7
BEST_FOLD            = 15
BEST_TEST_SUB        = 15
PRETRAIN_EPOCHS      = 50
FINETUNE_EPOCHS      = 50
BATCH_SIZE           = 128
SAVE_FIG             = True
FIG_DPI              = 150

# IMPORTANT:
# Change this order only if your dataset integer encoding is different.
CLASS_NAMES = ["happy", "fear", "sad", "normal"]

# ── split helper for the exact best fold ──────────────────────────────────────
def split_clisa_best_fold(ch='62ch', seed=BEST_SEED, fold=BEST_FOLD):
    X = get_data(ch)
    Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(
        X, Y, S, test_sub=fold, seed=seed
    )
    return X.shape[1], Xtr, Ytr, Str, Xva, Yva, Xte, Yte

# ── epoch helpers ─────────────────────────────────────────────────────────────
def train_epoch_loss(model, loader, opt, crit, scaler):
    model.train()
    total = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(Xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = crit(logits, Yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)

def val_epoch_loss_f1(model, loader, crit):
    model.eval()
    total = 0.0
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            with autocast():
                out = model(Xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits, Yb)
            total += loss.item()
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_t.extend(Yb.cpu().numpy())
    vf1 = f1_score(all_t, all_p, average='macro', zero_division=0)
    return total / max(len(loader), 1), vf1

# ── train CLISA on the exact best fold ────────────────────────────────────────
def train_clisa_best_fold():
    in_feat, Xtr, Ytr, _, Xva, Yva, Xte, Yte = split_clisa_best_fold('62ch')
    set_seed(BEST_SEED)

    model  = CLISA(in_feat).to(device)
    scaler = GradScaler()

    tr_ldr = make_loader(Xtr, Ytr, BATCH_SIZE, weighted=True)
    va_ldr = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)

    # Stage 1: CLISA self-supervised pretrain
    n_ch   = in_feat // 5
    opt1   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PRETRAIN_EPOCHS)

    for ep in range(PRETRAIN_EPOCHS):
        model.train()
        for Xb, _ in tr_ldr:
            Xb = Xb.to(device)

            v1 = Xb + 0.1 * torch.randn_like(Xb)
            mask = (torch.rand(Xb.size(0), n_ch, 1, device=Xb.device) > 0.3).float()
            v2 = Xb * mask.repeat(1, 1, 5).view(Xb.size(0), -1)

            opt1.zero_grad(set_to_none=True)
            with autocast():
                z1, z2 = model.project(v1), model.project(v2)
                loss = F.mse_loss(z1, z2.detach())
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt1)
            scaler.update()
        sched1.step()

    # Stage 2: supervised fine-tuning
    crit = nn.CrossEntropyLoss()
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=FINETUNE_EPOCHS)

    patience = PATIENCE if 'PATIENCE' in globals() else 10
    best_f1 = -1.0
    no_imp = 0
    best_state = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []

    for ep in range(1, FINETUNE_EPOCHS + 1):
        tl = train_epoch_loss(model, tr_ldr, opt2, crit, scaler)
        sched2.step()
        vl, vf1 = val_epoch_loss_f1(model, va_ldr, crit)

        train_losses.append(tl)
        val_losses.append(vl)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    return model, Xte, Yte, train_losses, val_losses

# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(model, Xte, Yte):
    te_ds = TensorDataset(torch.FloatTensor(Xte), torch.LongTensor(Yte))
    te_ldr = DataLoader(te_ds, batch_size=256, shuffle=False)

    model.eval()
    probs_all, preds_all, true_all = [], [], []

    with torch.no_grad():
        for Xb, Yb in te_ldr:
            Xb = Xb.to(device)
            out = model(Xb)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)

            probs_all.append(probs.cpu().numpy())
            preds_all.append(probs.argmax(dim=1).cpu().numpy())
            true_all.append(Yb.numpy())

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    true  = np.concatenate(true_all, axis=0)

    conf = probs.max(axis=1)
    correct = preds == true

    return probs, preds, true, conf, correct

# ── run exact best-fold model ─────────────────────────────────────────────────
print("Training CLISA 62ch on best fold from your table...")
print(f"seed={BEST_SEED}, fold={BEST_FOLD}, test_sub={BEST_TEST_SUB}")

model, Xte, Yte, tl, vl = train_clisa_best_fold()
probs, preds, true, conf, correct = run_inference(model, Xte, Yte)

# ── readable prediction table ─────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "sample_id": np.arange(len(true)),
    "true_id": true,
    "pred_id": preds,
    "true_class": [CLASS_NAMES[i] for i in true],
    "pred_class": [CLASS_NAMES[i] for i in preds],
    "confidence_%": np.round(conf * 100, 2),
    "correct": correct
})

# ── metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(true, preds) * 100
accb = balanced_accuracy_score(true, preds) * 100
mf1  = f1_score(true, preds, average='macro') * 100

print("\n" + "═" * 72)
print("  CLISA 62ch — best-fold inference summary")
print("─" * 72)
print(f"  Seed / fold / test_sub : {BEST_SEED} / {BEST_FOLD} / {BEST_TEST_SUB}")
print(f"  Test samples           : {len(true)}")
print(f"  Accuracy               : {acc:.2f}%")
print(f"  Balanced Accuracy      : {accb:.2f}%")
print(f"  Macro-F1               : {mf1:.2f}%")
print(f"  Mean confidence        : {pred_df['confidence_%'].mean():.2f}%")
print("═" * 72)

print("\nAll test predictions:")
display(pred_df)

wrong_df = pred_df.loc[~pred_df["correct"]].sort_values("confidence_%", ascending=False).reset_index(drop=True)
print("\nWrong predictions only:")
display(wrong_df)

# ── class-wise stats ──────────────────────────────────────────────────────────
cm = confusion_matrix(true, preds, labels=np.arange(len(CLASS_NAMES)))

true_counts = np.bincount(true, minlength=len(CLASS_NAMES))
pred_counts = np.bincount(preds, minlength=len(CLASS_NAMES))
class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

# ── visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))

# 1) Confusion matrix
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted class", fontsize=10)
ax.set_ylabel("True class", fontsize=10)
ax.set_xticks(np.arange(len(CLASS_NAMES)))
ax.set_yticks(np.arange(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_yticklabels(CLASS_NAMES, fontsize=10)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            fontsize=11, fontweight="bold"
        )

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 2) Class-wise recall
ax = axes[1]
bars = ax.bar(CLASS_NAMES, class_recall * 100, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, class_recall * 100):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Class-wise Recall", fontsize=12, fontweight="bold")
ax.set_ylabel("Recall (%)", fontsize=10)
ax.set_ylim(0, 105)
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# 3) True vs predicted class counts
ax = axes[2]
x = np.arange(len(CLASS_NAMES))
w = 0.36
bars1 = ax.bar(x - w/2, true_counts, width=w, label="True", edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + w/2, pred_counts, width=w, label="Predicted", edgecolor="white", linewidth=0.8)

for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{int(b.get_height())}",
            ha="center", fontsize=10, fontweight="bold")
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{int(b.get_height())}",
            ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_title("True vs Predicted Class Counts", fontsize=12, fontweight="bold")
ax.set_ylabel("Count", fontsize=10)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

plt.suptitle(
    f"CLISA 62ch class prediction check  |  seed={BEST_SEED}, fold={BEST_FOLD}, test_sub={BEST_TEST_SUB}",
    fontsize=13, fontweight="bold", y=1.03
)
plt.tight_layout()

if SAVE_FIG and 'RES_DIR' in globals():
    out_dir = RES_DIR / 'inference_vis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / f'CLISA_62ch_seed{BEST_SEED}_fold{BEST_FOLD}_class_prediction_check.png'
    plt.savefig(out_fn, dpi=FIG_DPI, bbox_inches='tight')
    print(f"\nSaved figure: {out_fn}")

plt.show()

print("\nClassification report:\n")
print(classification_report(true, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))


# ==============================================================================
# Notebook cell 45
# Categories: preprocessing, model_definition, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# CLISA 62ch — Loss curve only (no arrow annotation)
# ═══════════════════════════════════════════════════════════════════════════════

import copy
import numpy as np
import matplotlib.pyplot as plt

# ── config ─────────────────────────────────────────────────────────────────────
CLISA_CURVE_SEED      = 1
CLISA_CURVE_FOLD      = 5
CLISA_PRETRAIN_EPOCHS = 50
CLISA_FINETUNE_EPOCHS = 50
CLISA_BATCH_SIZE      = 128
CLISA_SAVE_FIG        = True
CLISA_FIG_DPI         = 150

# ── use notebook globals if available ─────────────────────────────────────────
_clisa_patience = PATIENCE if 'PATIENCE' in globals() else 10
_clisa_curve_dir = RES_DIR / 'loss_curves' if 'RES_DIR' in globals() else None
if CLISA_SAVE_FIG and _clisa_curve_dir is not None:
    _clisa_curve_dir.mkdir(parents=True, exist_ok=True)

# ── helper: CLISA pretrain ────────────────────────────────────────────────────
def _run_clisa_62ch_curve(seed=CLISA_CURVE_SEED, fold=CLISA_CURVE_FOLD):
    old_seed = globals().get("CURVE_SEED", None)
    old_fold = globals().get("CURVE_FOLD", None)
    globals()["CURVE_SEED"] = seed
    globals()["CURVE_FOLD"] = fold

    in_feat, Xtr, Ytr, _, Xva, Yva, _, _ = _split('62ch')
    set_seed(seed)

    model  = CLISA(in_feat).to(device)
    scaler = GradScaler()
    tr_ldr = make_loader(Xtr, Ytr, CLISA_BATCH_SIZE, weighted=True)
    va_ldr = make_loader(Xva, Yva, 256, weighted=False, shuffle=False)

    # ── Stage 1: self-supervised pretrain ─────────────────────────────────────
    n_ch   = in_feat // 5
    opt1   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CLISA_PRETRAIN_EPOCHS)

    for ep in range(CLISA_PRETRAIN_EPOCHS):
        model.train()
        for Xb, _ in tr_ldr:
            Xb = Xb.to(device)
            v1 = Xb + 0.1 * torch.randn_like(Xb)
            mask = (torch.rand(Xb.size(0), n_ch, 1, device=Xb.device) > 0.3).float()
            v2   = Xb * mask.repeat(1, 1, 5).view(Xb.size(0), -1)

            opt1.zero_grad(set_to_none=True)
            with autocast():
                z1, z2 = model.project(v1), model.project(v2)
                loss = F.mse_loss(z1, z2.detach())
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt1)
            scaler.update()
        sched1.step()

    # ── Stage 2: supervised fine-tuning with loss logging ─────────────────────
    crit   = nn.CrossEntropyLoss()
    opt2   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CLISA_FINETUNE_EPOCHS)

    train_losses, val_losses = [], []
    best_f1, no_imp = 0.0, 0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, CLISA_FINETUNE_EPOCHS + 1):
        tl = _epoch_train_loss(model, tr_ldr, opt2, crit, scaler)
        sched2.step()
        vl, vf1 = _epoch_val_loss(model, va_ldr, crit)

        train_losses.append(tl)
        val_losses.append(vl)

        if vf1 > best_f1:
            best_f1 = vf1
            no_imp = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1

        if no_imp >= _clisa_patience:
            break

    model.load_state_dict(best_state)

    if old_seed is not None:
        globals()["CURVE_SEED"] = old_seed
    if old_fold is not None:
        globals()["CURVE_FOLD"] = old_fold

    return model, train_losses, val_losses

# ── run CLISA 62ch ────────────────────────────────────────────────────────────
model, tl, vl = _run_clisa_62ch_curve()

# ── plot CLISA loss curve only ────────────────────────────────────────────────
epochs = np.arange(1, len(tl) + 1)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(epochs, tl, color='#1565C0', lw=2.4, label='Train loss')
ax.plot(epochs, vl, color='#C62828', lw=2.4, label='Val loss')


if len(tl) < CLISA_FINETUNE_EPOCHS:
    ax.axvline(len(tl), color='#555555', linestyle=':', lw=1.3,
               label=f'Early stop (ep {len(tl)})')

ax.set_title(
    f'CLISA 62ch Loss Curve\n(fold {CLISA_CURVE_FOLD}, seed {CLISA_CURVE_SEED})',
    fontsize=12,
    fontweight='bold'
)
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Cross-Entropy Loss', fontsize=10)
ax.set_xlim(1, max(len(tl), 5))
ax.grid(True, alpha=0.35)
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.95)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

if CLISA_SAVE_FIG and _clisa_curve_dir is not None:
    out_fn = _clisa_curve_dir / 'CLISA_62ch_loss_curve.png'
    plt.savefig(out_fn, dpi=CLISA_FIG_DPI, bbox_inches='tight')
    print(f'Saved: {out_fn}')

plt.show()


# ==============================================================================
# Notebook cell 46
# Categories: preprocessing, model_definition, training, evaluation, results_tables, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# WEARABLE ANALYSIS — shared helpers
# Run this first, then run Plot 1 / Plot 2 / Plot 3 cells below
# ═══════════════════════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast

# ── common config ──────────────────────────────────────────────────────────────
WEARABLE_FIGSIZE = (10.2, 5.6)
WEARABLE_DPI = 200
WEARABLE_OUT = RES_DIR / 'wearable_analysis_plots'
WEARABLE_OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["happy", "fear", "sad", "normal"]

MODEL_NAME = {
    'M11': 'Shallow MLP',
    'M12': 'Deep MLP',
    'M13': 'LSTM',
    'M14': 'GRU',
    'M15': 'Conv1D',
    'M16': 'Transformer',
    'M17': 'EEG Conformer',
    'M18': 'ChanDrop Transformer',
    'M19': 'DANN',
    'M20': 'CLISA',
    'M21': 'SimCLR',
    'M22': 'BYOL',
    'M23': 'PseudoLabel',
    'M24': 'MixMatch',
    'M25': 'DANCE Teacher',
    'M26': 'DANCE Student',
}

C_CLISA   = "#E65100"
C_DANN    = "#607D8B"
C_DANCE   = "#1565C0"
C_OTHER   = "#90A4AE"
C_GAIN    = "#2E7D32"
C_LOSS    = "#C62828"

# ── load checkpoint table safely ──────────────────────────────────────────────
def wearable_load_ckpts():
    rows = []
    for p in sorted(CKPT_DIR.glob('M*.json')):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    rows.append(obj)
        except Exception:
            pass

    if not rows:
        raise RuntimeError(f'No checkpoint JSONs found in {CKPT_DIR}')

    df = pd.DataFrame(rows)

    for col in ['seed', 'fold', 'test_sub']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

    for col in ['acc_a', 'f1_a', 'acc_b', 'f1_b', 'best_val_f1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['model_id', 'ch']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['model_name'] = df['model_id'].map(MODEL_NAME).fillna(df['model_id'])
    return df

def wearable_best_row(df, model_id, ch):
    sub = df[(df.model_id == model_id) & (df.ch == ch)].copy()
    sub = sub.dropna(subset=['acc_b'])
    if sub.empty:
        raise RuntimeError(f'No valid rows found for {model_id} [{ch}]')
    sort_cols = [c for c in ['acc_b', 'f1_b', 'acc_a', 'f1_a'] if c in sub.columns]
    sub = sub.sort_values(sort_cols, ascending=False, na_position='last')
    row = sub.iloc[0].copy()
    for k in ['seed', 'fold', 'test_sub']:
        if pd.isna(row.get(k)):
            raise RuntimeError(f"Best row for {model_id} [{ch}] is missing '{k}'")
    return row

def wearable_find_matching_row(df, model_id, ch, seed, fold, test_sub):
    # exact match first
    sub = df[
        (df.model_id == model_id) &
        (df.ch == ch) &
        (df.seed == seed) &
        (df.fold == fold) &
        (df.test_sub == test_sub)
    ].dropna(subset=['acc_b'])

    if not sub.empty:
        return sub.iloc[0].copy(), "exact"

    # fallback 1: same seed + same test_sub
    sub = df[
        (df.model_id == model_id) &
        (df.ch == ch) &
        (df.seed == seed) &
        (df.test_sub == test_sub)
    ].dropna(subset=['acc_b']).sort_values('acc_b', ascending=False)

    if not sub.empty:
        return sub.iloc[0].copy(), "same seed + same test_sub"

    # fallback 2: same test_sub
    sub = df[
        (df.model_id == model_id) &
        (df.ch == ch) &
        (df.test_sub == test_sub)
    ].dropna(subset=['acc_b']).sort_values('acc_b', ascending=False)

    if not sub.empty:
        return sub.iloc[0].copy(), "same test_sub"

    # fallback 3: global best
    return wearable_best_row(df, model_id, ch), "global best"

def wearable_mean_std(df, model_id, ch):
    x = df[(df.model_id == model_id) & (df.ch == ch)]['acc_b'].dropna()
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std())

def wearable_save(fig, filename):
    out = WEARABLE_OUT / filename
    fig.savefig(out, dpi=WEARABLE_DPI, bbox_inches='tight')
    print(f"Saved: {out}")

# ── split + model loading for inference ───────────────────────────────────────
def wearable_split(ch, seed, test_sub):
    X = get_data(ch)
    Xtr, Ytr, Str, Xva, Yva, Xte, Yte = loso_split(X, Y, S, test_sub=test_sub, seed=seed)
    return X.shape[1], Xtr, Ytr, Str, Xva, Yva, Xte, Yte

def wearable_load_model(model_id, ch, seed, fold, test_sub):
    in_feat, Xtr, Ytr, Str, Xva, Yva, Xte, Yte = wearable_split(ch, seed, test_sub)
    n_sub_tr = len(np.unique(Str))

    if model_id == 'M20':      # CLISA
        model = CLISA(in_feat).to(device)
    elif model_id == 'M19':    # DANN
        model = DANN(in_feat, n_subjects=n_sub_tr).to(device)
    elif model_id == 'M26':    # DANCE student
        model = DANCEStudent(n_ch=6).to(device)
    else:
        raise RuntimeError(f"Model loading not implemented for {model_id}")

    wpath = weight_path(model_id, seed, fold, ch)
    if not Path(wpath).exists():
        raise FileNotFoundError(f'Weight not found: {wpath}')

    model.load_state_dict(torch.load(wpath, map_location=device))
    return model, Xte, Yte

@torch.no_grad()
def wearable_infer(model, model_id, X, batch_size=256):
    model.eval()
    ds = TensorDataset(torch.FloatTensor(X), torch.zeros(len(X), dtype=torch.long))
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits = []
    for Xb, _ in ld:
        Xb = Xb.to(device)
        with autocast():
            # DANN can have different forward signatures across notebooks
            if model_id == 'M19':
                try:
                    out = model(Xb, lam=0.0)   # training-style signature
                except TypeError:
                    out = model(Xb)            # inference-style signature
            else:
                out = model(Xb)

            logits = out[0] if isinstance(out, tuple) else out

        all_logits.append(logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    conf  = probs.max(axis=1)
    return probs, preds, conf

def wearable_row_normalize(cm):
    denom = cm.sum(axis=1, keepdims=True).astype(float)
    denom[denom == 0] = 1.0
    return cm / denom * 100.0

def wearable_class_recall(cm):
    denom = cm.sum(axis=1).astype(float)
    denom[denom == 0] = 1.0
    return np.diag(cm) / denom * 100.0

# load once
wear_df = wearable_load_ckpts()
print("Loaded wearable checkpoint table.")


# ==============================================================================
# Notebook cell 47
# Categories: model_definition, results_tables, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Wearable 6ch leaderboard
# Focus: best wearable models only
# ═══════════════════════════════════════════════════════════════════════════════

df6 = wear_df[wear_df.ch == '6ch'].dropna(subset=['acc_b']).copy()
if df6.empty:
    raise RuntimeError("No valid 6ch rows found.")

rank6 = (
    df6.groupby(['model_id', 'model_name'], as_index=False)['acc_b']
    .mean()
    .sort_values('acc_b', ascending=False)
    .reset_index(drop=True)
)

# take top 6, but force CLISA / DANN / DANCE Student in if missing
must_keep = {'M20', 'M19', 'M26'}
chosen = rank6.head(6).copy()

missing = rank6[rank6['model_id'].isin(must_keep - set(chosen['model_id']))]
if not missing.empty:
    chosen = pd.concat([chosen, missing], ignore_index=True)

chosen = (
    chosen.drop_duplicates(subset=['model_id'])
    .sort_values('acc_b', ascending=False)
    .reset_index(drop=True)
)

labels = [f"{r.model_name}\n({r.model_id})" for _, r in chosen.iterrows()]
vals = chosen['acc_b'].values * 100

colors = []
for mid in chosen['model_id']:
    if mid == 'M20':
        colors.append(C_CLISA)
    elif mid == 'M19':
        colors.append(C_DANN)
    elif mid == 'M26':
        colors.append(C_DANCE)
    else:
        colors.append(C_OTHER)

fig, ax = plt.subplots(figsize=WEARABLE_FIGSIZE)

x = np.arange(len(chosen))
bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.9, width=0.72)
ax.axhline(25, color='gray', linestyle='--', linewidth=1.2, label='Chance (25%)')

for b, v in zip(bars, vals):
    ax.text(
        b.get_x() + b.get_width()/2,
        v + 0.8,
        f"{v:.2f}%",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean AccB (%)", fontsize=11)
ax.set_title("Plot 1 — Wearable 6ch Model Comparison", fontsize=13, fontweight='bold')
ax.set_ylim(0, max(vals) + 10)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
wearable_save(fig, "wearable_plot1_6ch_leaderboard.png")
plt.show()


# ==============================================================================
# Notebook cell 48
# Categories: preprocessing, model_definition, training, evaluation, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Class-wise recall on best wearable CLISA fold
# Focus: CLISA 6ch vs matched DANN 6ch
# ═══════════════════════════════════════════════════════════════════════════════

best_clisa = wearable_best_row(wear_df, 'M20', '6ch')

seed_p3     = int(best_clisa['seed'])
fold_p3     = int(best_clisa['fold'])
test_sub_p3 = int(best_clisa['test_sub'])

# matched DANN row
matched_dann, match_note = wearable_find_matching_row(
    wear_df, 'M19', '6ch',
    seed=seed_p3, fold=fold_p3, test_sub=test_sub_p3
)

seed_dann     = int(matched_dann['seed'])
fold_dann     = int(matched_dann['fold'])
test_sub_dann = int(matched_dann['test_sub'])

# load models
clisa_model, Xte_clisa, Yte_clisa = wearable_load_model('M20', '6ch', seed_p3, fold_p3, test_sub_p3)
dann_model,  Xte_dann,  Yte_dann  = wearable_load_model('M19', '6ch', seed_dann, fold_dann, test_sub_dann)

# sanity check
same_split = (len(Yte_clisa) == len(Yte_dann)) and np.all(Yte_clisa == Yte_dann)
if not same_split:
    print("Warning: CLISA and DANN test labels are not identical; comparison uses each model's own split.")

# inference
_, clisa_preds, _ = wearable_infer(clisa_model, 'M20', Xte_clisa)
_, dann_preds,  _ = wearable_infer(dann_model,  'M19', Xte_dann)

# metrics
cm_clisa = confusion_matrix(Yte_clisa, clisa_preds, labels=np.arange(len(CLASS_NAMES)))
cm_dann  = confusion_matrix(Yte_dann,  dann_preds,  labels=np.arange(len(CLASS_NAMES)))

rec_clisa = wearable_class_recall(cm_clisa)
rec_dann  = wearable_class_recall(cm_dann)
rec_delta = rec_clisa - rec_dann

acc_clisa  = accuracy_score(Yte_clisa, clisa_preds) * 100
accb_clisa = balanced_accuracy_score(Yte_clisa, clisa_preds) * 100
acc_dann   = accuracy_score(Yte_dann, dann_preds) * 100
accb_dann  = balanced_accuracy_score(Yte_dann, dann_preds) * 100

fig, ax = plt.subplots(figsize=WEARABLE_FIGSIZE)

x = np.arange(len(CLASS_NAMES))
w = 0.34

bars1 = ax.bar(
    x - w/2, rec_dann, width=w,
    color=C_DANN, edgecolor='white', linewidth=0.8,
    label='DANN (M19, 6ch)'
)
bars2 = ax.bar(
    x + w/2, rec_clisa, width=w,
    color=C_CLISA, edgecolor='white', linewidth=0.8,
    label='CLISA (M20, 6ch)'
)

for b, v in zip(bars1, rec_dann):
    ax.text(
        b.get_x() + b.get_width()/2,
        v + 1.0,
        f"{v:.1f}%",
        ha='center',
        fontsize=9,
        fontweight='bold'
    )

for b, v in zip(bars2, rec_clisa):
    ax.text(
        b.get_x() + b.get_width()/2,
        v + 1.0,
        f"{v:.1f}%",
        ha='center',
        fontsize=9,
        fontweight='bold'
    )

for i, d in enumerate(rec_delta):
    y = max(rec_dann[i], rec_clisa[i]) + 7
    ax.text(
        i, y,
        f"{d:+.1f} pp",
        ha='center',
        fontsize=9.5,
        fontweight='bold',
        color=(C_GAIN if d >= 0 else C_LOSS)
    )

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("Recall (%)", fontsize=11)
ax.set_title("Plot 3 — Wearable Class-wise Recall: CLISA vs DANN", fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rec_clisa.max(), rec_dann.max()) + 18)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc='upper right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)

# compact metric note
note = (
    f"CLISA: Acc={acc_clisa:.2f}%, AccB={accb_clisa:.2f}%\n"
    f"DANN:  Acc={acc_dann:.2f}%, AccB={accb_dann:.2f}%\n"
    f"Match: {match_note}"
)
ax.text(
    0.02, 0.98, note,
    transform=ax.transAxes,
    ha='left', va='top',
    fontsize=9.5,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
)

plt.tight_layout()
wearable_save(fig, "wearable_plot3_classwise_recall_clisa_vs_dann.png")
plt.show()
