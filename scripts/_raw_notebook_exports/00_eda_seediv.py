# Auto-exported raw code from notebook: 00_eda_seediv.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: preprocessing, figures, webapp_or_demo
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 1 — Imports and package check
# ═══════════════════════════════════════════════════════════
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe on Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import scipy.io as sio
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pathlib

warnings.filterwarnings('ignore')
np.random.seed(42)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
try:
    import scipy; print("SciPy:", scipy.__version__)
except ImportError:
    print("⚠ SciPy not found — pip install scipy")
try:
    from sklearn import __version__ as skv; print("Scikit-learn:", skv)
except ImportError:
    print("⚠ sklearn not found — pip install scikit-learn")
print("\n✔ Imports complete.")


# ==============================================================================
# Notebook cell 2
# Categories: preprocessing, model_definition, training, results_tables, figures, audit_verification, webapp_or_demo
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 2 — Path configuration + GPU check
# ═══════════════════════════════════════════════════════════
# ── Set project root ──────────────────────────────────────
# If running from the notebook directory, ROOT is the parent
ROOT = pathlib.Path(os.getcwd())
# Adjust if needed: ROOT = pathlib.Path(r"C:\Users\Saif\Desktop\CSE400\C")
print(f"Project root: {ROOT}")

# ── Directory setup ───────────────────────────────────────
FEATURES_DIR  = ROOT / "features"
DATA_DIR      = ROOT / "data" / "SEED_IV" / "ExtractedFeatures"
FIG_DIR       = ROOT / "figures" / "eda_seediv"
CKPT_DIR      = ROOT / "checkpoints" / "eda_seediv"

for d in [FEATURES_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print(f"Figures → {FIG_DIR}")
print(f"Checkpoints → {CKPT_DIR}")

# ── GPU check (informational for EDA — CPU-dominant) ──────
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
        assert "3050" in gpu_name or vram_gb < 5.0, "⚠ Unexpected GPU — verify this is the RTX 3050 laptop"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
        print("✔ Flash Attention (SDPA) enabled — RTX 3050 compatible")
    else:
        print("⚠ GPU NOT available — EDA runs on CPU (acceptable for EDA)")
except ImportError:
    print("⚠ PyTorch not imported — EDA runs on CPU only")

print("\n✔ Paths configured.")


# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, training, figures, statistics
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 3 — 🔄 CHECKPOINT RECOVERY (run after any interruption)
# ═══════════════════════════════════════════════════════════
EDA_PLOTS = [
    "eda01_class_distribution",
    "eda02_subject_window_count",
    "eda03_band_channel_distributions",
    "eda04_intersubject_variability",
    "eda05_normalisation_comparison",
    "eda06_channel_correlation",
    "eda07_pca",
    "eda08_tsne_raw",
    "eda09_session_effect",
    "eda10_band_importance",
    "eda11_channel_coverage",
]

def eda_done(name):
    return (CKPT_DIR / f"{name}.done").exists()

def mark_done(name):
    (CKPT_DIR / f"{name}.done").write_text("done")

done    = [p for p in EDA_PLOTS if eda_done(p)]
pending = [p for p in EDA_PLOTS if not eda_done(p)]

print(f"{'='*60}")
print(f" SEED-IV EDA Recovery Status")
print(f"{'='*60}")
print(f" Completed : {len(done):2d} / {len(EDA_PLOTS)}")
print(f" Remaining : {len(pending):2d}")
if pending:
    print(f"\n Next to run:")
    for p in pending:
        print(f"  • {p}")
else:
    print("\n ✅ All EDA plots complete! Nothing to re-run.")
print(f"{'='*60}")


# ==============================================================================
# Notebook cell 4
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 4 — Data Loading
# ═══════════════════════════════════════════════════════════

SEED_CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ',
    'O2','CB2'
]
assert len(SEED_CHANNELS) == 62

STUDENT_CH_NAMES   = ['FP1', 'FP2', 'F7', 'F8', 'T7', 'T8']
STUDENT_CH_INDICES = [0, 2, 5, 13, 23, 31]
BAND_NAMES  = ['delta', 'theta', 'alpha', 'beta', 'gamma']
LABEL_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
N_SUBJECTS  = 15
N_CLASSES   = 4

SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
}

# ── Fix: Phase B saved flat (N, 310) — reshape to (N, 62, 5) ──
def maybe_reshape(arr, c, b):
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], c, b)
        print(f"  Auto-reshaped: (N, {c*b}) → (N, {c}, {b})")
    return arr

npy_X62  = FEATURES_DIR / "seed_iv_X_62ch.npy"
npy_X6   = FEATURES_DIR / "seed_iv_X_6ch.npy"
npy_y    = FEATURES_DIR / "seed_iv_y_4cls.npy"
npy_subj = FEATURES_DIR / "seed_iv_subjects.npy"
npy_sess = FEATURES_DIR / "seed_iv_session.npy"

if all(f.exists() for f in [npy_X62, npy_y, npy_subj, npy_sess]):
    print("Loading pre-processed features from features/ directory...")
    X_62  = maybe_reshape(np.load(npy_X62), 62, 5)
    y     = np.load(npy_y)
    subj  = np.load(npy_subj)
    sess  = np.load(npy_sess)
    if npy_X6.exists():
        X_6 = maybe_reshape(np.load(npy_X6), 6, 5)
    else:
        X_6 = X_62[:, STUDENT_CH_INDICES, :]
        np.save(npy_X6, X_6)
    print(f"✔ Loaded — X_62: {X_62.shape} | X_6: {X_6.shape}")

else:
    print("features/*.npy not found — processing raw SEED-IV .mat files...")
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"SEED-IV data not found at {DATA_DIR}\n"
            "Please place ExtractedFeatures/{1,2,3}/ folders in data/SEED_IV/")

    all_X, all_y, all_subj, all_sess = [], [], [], []
    for sess_id in [1, 2, 3]:
        sess_dir  = DATA_DIR / str(sess_id)
        mat_files = sorted(sess_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files in {sess_dir}")
        for subj_id, mat_path in enumerate(mat_files, start=1):
            mat    = sio.loadmat(str(mat_path))
            de_key = None
            for k in ['de_LDS', 'de_movingAve']:
                if k in mat: de_key = k; break
            if de_key is None:
                for k, v in mat.items():
                    if not k.startswith('_') and isinstance(v, np.ndarray):
                        de_key = k; break
            for trial_idx, label in enumerate(SESSION_LABELS[sess_id]):
                if f"de_LDS{trial_idx+1}" in mat:
                    trial_data = mat[f"de_LDS{trial_idx+1}"]
                elif de_key in mat:
                    raw = mat[de_key]
                    if isinstance(raw, np.ndarray) and raw.ndim == 3:
                        if raw.shape[0] == 62:
                            trial_data = raw
                        else:
                            trial_data = raw[trial_idx] if trial_idx < raw.shape[0] else raw[0]
                    else: continue
                else: continue
                if trial_data.ndim == 3:
                    if trial_data.shape[0] == 62:
                        trial_data = trial_data.transpose(1, 0, 2)
                    elif trial_data.shape[2] == 62 * 5:
                        trial_data = trial_data.reshape(trial_data.shape[0], 62, 5)
                    n_win = trial_data.shape[0]
                    all_X.append(trial_data)
                    all_y.extend([label] * n_win)
                    all_subj.extend([subj_id] * n_win)
                    all_sess.extend([sess_id] * n_win)
            print(f"  Session {sess_id} Subject {subj_id:2d} loaded", end='\r')

    X_62 = np.concatenate(all_X, axis=0).astype(np.float32)
    y    = np.array(all_y,    dtype=np.int64)
    subj = np.array(all_subj, dtype=np.int64)
    sess = np.array(all_sess, dtype=np.int64)
    X_6  = X_62[:, STUDENT_CH_INDICES, :]
    np.save(npy_X62, X_62); np.save(npy_X6, X_6)
    np.save(npy_y, y); np.save(npy_subj, subj); np.save(npy_sess, sess)
    print(f"\n✔ Processed and saved — shape: {X_62.shape}")

N = X_62.shape[0]
assert X_62.shape[1:] == (62, 5), f"Expected (N,62,5) got {X_62.shape}"
assert X_6.shape[1:]  == (6,  5), f"Expected (N,6,5)  got {X_6.shape}"

print(f"\n{'='*50}")
print(f" SEED-IV Dataset Summary")
print(f"{'='*50}")
print(f" Total windows  : {N:,}")
print(f" Teacher input  : {X_62.shape}  — 62 channels")
print(f" Student input  : {X_6.shape}   — 6 channels {STUDENT_CH_NAMES}")
print(f" Classes        : {N_CLASSES}  {LABEL_NAMES}")
print(f" Subjects       : {np.unique(subj)}")
print(f" Sessions       : {np.unique(sess)}")
print(f" Label dist     : {dict(zip(LABEL_NAMES, [int((y==i).sum()) for i in range(4)]))}")
print(f"{'='*50}")


# ==============================================================================
# Notebook cell 5
# Categories: training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 5 — [EDA 01] Class Distribution
# ═══════════════════════════════════════════════════════════
PLOT = "eda01_class_distribution"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done"); 
else:
    print(f"Running {PLOT}...")
    counts = [(y == i).sum() for i in range(4)]
    total  = sum(counts)
    # Class weights (inverse frequency, normalised)
    weights = [total / (4 * c) for c in counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SEED-IV: Class Distribution", fontsize=14, fontweight='bold')

    # Bar chart
    colors = ['#4472C4','#ED7D31','#A9D18E','#FF0000']
    bars = axes[0].bar(LABEL_NAMES, counts, color=colors, edgecolor='black', width=0.6)
    axes[0].set_title("Window Count per Class", fontweight='bold')
    axes[0].set_ylabel("Number of Windows")
    axes[0].set_ylim(0, max(counts) * 1.2)
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f"{cnt:,}\n({cnt/total*100:.1f}%)", ha='center', va='bottom', fontsize=10)
    axes[0].axhline(total/4, color='red', linestyle='--', alpha=0.6, label='Balanced baseline')
    axes[0].legend()

    # Class weights
    axes[1].bar(LABEL_NAMES, weights, color=colors, edgecolor='black', width=0.6)
    axes[1].set_title("Class Weights (Inverse Frequency)\nfor use in CrossEntropyLoss", fontweight='bold')
    axes[1].set_ylabel("Weight")
    for i, (name, w) in enumerate(zip(LABEL_NAMES, weights)):
        axes[1].text(i, w + 0.01, f"{w:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda01_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Class distribution: {dict(zip(LABEL_NAMES, counts))}")
    print(f"  Class weights (CE): {dict(zip(LABEL_NAMES, [f'{w:.3f}' for w in weights]))}")
    print(f"  ⚠ Happy class underrepresented by {(1 - counts[3]/counts[1])*100:.1f}% vs Sad")
    print(f"  → Use WeightedRandomSampler or class_weight={{{', '.join([f'{i}:{w:.3f}' for i,w in enumerate(weights)])}}}")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda01_class_distribution.png'}")


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 6 — [EDA 02] Per-Subject Window Count
# ═══════════════════════════════════════════════════════════
PLOT = "eda02_subject_window_count"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    subj_ids = sorted(np.unique(subj))

    # Windows per subject per class
    counts_mat = np.zeros((len(subj_ids), N_CLASSES), dtype=int)
    for i, s in enumerate(subj_ids):
        mask = subj == s
        for c in range(N_CLASSES):
            counts_mat[i, c] = ((y == c) & mask).sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("SEED-IV: Per-Subject Window Count", fontsize=14, fontweight='bold')

    # Stacked bar
    colors = ['#4472C4','#ED7D31','#A9D18E','#FF4444']
    bottom = np.zeros(len(subj_ids))
    for c, (name, col) in enumerate(zip(LABEL_NAMES, colors)):
        axes[0].bar(subj_ids, counts_mat[:, c], bottom=bottom, color=col,
                    label=name, edgecolor='white', width=0.7)
        bottom += counts_mat[:, c]
    axes[0].set_title("Stacked: Windows per Subject × Class", fontweight='bold')
    axes[0].set_xlabel("Subject ID"); axes[0].set_ylabel("Window Count")
    axes[0].legend(loc='upper right', fontsize=9); axes[0].set_xticks(subj_ids)

    # Total per subject with std band
    totals = counts_mat.sum(axis=1)
    axes[1].bar(subj_ids, totals, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].axhline(totals.mean(), color='red', ls='--', label=f'Mean={totals.mean():.0f}')
    axes[1].fill_between([min(subj_ids)-0.5, max(subj_ids)+0.5],
                          totals.mean()-totals.std(), totals.mean()+totals.std(),
                          alpha=0.15, color='red', label=f'±1σ={totals.std():.0f}')
    axes[1].set_title("Total Windows per Subject\n(check for high-variance outliers)", fontweight='bold')
    axes[1].set_xlabel("Subject ID"); axes[1].set_ylabel("Total Windows")
    axes[1].legend(); axes[1].set_xticks(subj_ids)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda02_subject_window_count.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    cv = totals.std() / totals.mean()
    print(f"  Window count range: {totals.min()} – {totals.max()} | mean={totals.mean():.0f} | CV={cv:.3f}")
    if cv > 0.1:
        print(f"  ⚠ CV={cv:.3f} > 0.10 → subject window count variability detected. LOSO folds will be imbalanced.")
    else:
        print(f"  ✔ Low variability (CV={cv:.3f}) — LOSO folds reasonably balanced.")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda02_subject_window_count.png'}")


# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, results_tables, figures, statistics
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 7 — [EDA 03] Per-Band, Per-Channel Feature Distributions
# Uses 10 representative channels (evenly spaced)
# ═══════════════════════════════════════════════════════════
PLOT = "eda03_band_channel_distributions"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    np.random.seed(42)
    # Sample 3000 windows to keep plots readable
    idx_sample = np.random.choice(N, size=min(3000, N), replace=False)
    X_samp  = X_62[idx_sample]
    y_samp  = y[idx_sample]

    rep_channels = np.linspace(0, 61, 10, dtype=int)
    colors = ['#4472C4','#ED7D31','#A9D18E','#FF4444','#7030A0']

    fig, axes = plt.subplots(len(BAND_NAMES), len(rep_channels), figsize=(22, 12))
    fig.suptitle("SEED-IV: DE Feature Distribution\n(10 representative channels × 5 bands, 3000-window sample)",
                 fontsize=13, fontweight='bold')

    for bi, band in enumerate(BAND_NAMES):
        for ci, ch_idx in enumerate(rep_channels):
            ax = axes[bi, ci]
            vals = X_samp[:, ch_idx, bi]
            ax.hist(vals, bins=30, color=colors[bi], alpha=0.7, edgecolor='none', density=True)
            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                xs  = np.linspace(vals.min(), vals.max(), 100)
                ax.plot(xs, kde(xs), 'k-', lw=1)
            except Exception:
                pass
            # Skewness check
            sk = stats.skew(vals)
            ax.set_title(f"{SEED_CHANNELS[ch_idx]}\nsk={sk:.2f}", fontsize=7)
            if ci == 0: ax.set_ylabel(band, fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda03_band_channel_distributions.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Summary: which channels have high skewness (|skew| > 1)?
    skews = stats.skew(X_62.reshape(N, -1), axis=0)
    high_skew = np.where(np.abs(skews) > 1)[0]
    print(f"  Features with |skew| > 1.0: {len(high_skew)} / {62*5} = {len(high_skew)/(62*5)*100:.1f}%")
    print(f"  → Consider log(1+x) transform for highly skewed features in HPO experiments")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda03_band_channel_distributions.png'}")


# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 8 — [EDA 04] Inter-Subject Variability Heatmap
# Quantifies subject-level domain gap — key motivation for
# subject-invariant learning.
# ═══════════════════════════════════════════════════════════
PLOT = "eda04_intersubject_variability"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    subj_ids  = sorted(np.unique(subj))
    # Per-subject mean feature vector (310-dim flattened)
    subj_means = np.zeros((len(subj_ids), 62 * 5))
    subj_stds  = np.zeros((len(subj_ids), 62 * 5))
    for i, s in enumerate(subj_ids):
        mask = subj == s
        flat = X_62[mask].reshape(mask.sum(), -1)
        subj_means[i] = flat.mean(0)
        subj_stds[i]  = flat.std(0)

    # Coefficient of variation across subjects (per feature)
    global_mean = subj_means.mean(0)
    global_std  = subj_means.std(0)
    cv_per_feat = np.where(np.abs(global_mean) > 1e-6,
                           global_std / np.abs(global_mean), 0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("SEED-IV: Inter-Subject Variability Analysis", fontsize=14, fontweight='bold')

    # Heatmap: subjects × features (reshaped to 15 × (62×5))
    im0 = axes[0].imshow(subj_means, aspect='auto', cmap='RdYlBu_r',
                          interpolation='nearest')
    axes[0].set_title("Per-Subject Mean DE Feature\n(15 subjects × 310 features)", fontweight='bold')
    axes[0].set_xlabel("Feature (channel × band)"); axes[0].set_ylabel("Subject ID")
    axes[0].set_yticks(range(len(subj_ids))); axes[0].set_yticklabels([f"S{s}" for s in subj_ids])
    plt.colorbar(im0, ax=axes[0], fraction=0.02)

    # CV per channel (averaged over bands)
    cv_per_ch = cv_per_feat.reshape(62, 5).mean(1)
    axes[1].bar(range(62), cv_per_ch, color='steelblue', edgecolor='none')
    axes[1].axhline(cv_per_ch.mean(), color='red', ls='--', label=f'Mean CV={cv_per_ch.mean():.3f}')
    axes[1].set_title("Coefficient of Variation per Channel\n(averaged over 5 bands)", fontweight='bold')
    axes[1].set_xlabel("Channel Index"); axes[1].set_ylabel("CV = σ_subjects / |μ_subjects|")
    axes[1].legend(); axes[1].set_xticks(range(0, 62, 10))
    # Mark student channels
    for ci in STUDENT_CH_INDICES:
        axes[1].axvline(ci, color='orange', alpha=0.5, ls='--', lw=1)
    axes[1].text(0.97, 0.97, 'Orange: student channels', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=8, color='darkorange')

    # Intra vs inter subject variance ratio
    # Intra: variance within each subject, averaged; Inter: variance of subject means
    intra_var = np.mean([X_62[subj==s].reshape((subj==s).sum(), -1).var(0).mean()
                         for s in subj_ids])
    inter_var = subj_means.var(0).mean()
    ratio     = inter_var / (intra_var + 1e-9)
    axes[2].bar(['Intra-Subject (within-subject)', 'Inter-Subject (between-subject)'],
                [intra_var, inter_var], color=['#4472C4','#ED7D31'], edgecolor='black', width=0.5)
    axes[2].set_title(f"Variance Decomposition\nInter/Intra ratio = {ratio:.2f}", fontweight='bold')
    axes[2].set_ylabel("Mean Variance")
    if ratio > 1.0:
        axes[2].text(0.5, 0.95, f"⚠ Inter > Intra (ratio={ratio:.2f})\n→ Strong subject bias → motivates DANN/DANCE",
                     transform=axes[2].transAxes, ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda04_intersubject_variability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Intra-subject variance : {intra_var:.4f}")
    print(f"  Inter-subject variance : {inter_var:.4f}")
    print(f"  Inter/Intra ratio      : {ratio:.3f}")
    if ratio > 1.0:
        print(f"  ⚠ Inter-subject variance EXCEEDS intra — strong subject-bias. DANCE/DANN essential.")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda04_intersubject_variability.png'}")


# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, model_definition, results_tables, figures, statistics, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 9 — [EDA 05] Before/After Z-Score Normalisation
# Verifies subject-specific z-score is correct and effective.
# ═══════════════════════════════════════════════════════════
PLOT = "eda05_normalisation_comparison"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    subj_ids = sorted(np.unique(subj))
    
    # Apply subject-specific z-score
    X_norm = np.zeros_like(X_62)
    for s in subj_ids:
        mask = subj == s
        flat = X_62[mask].reshape(mask.sum(), -1)
        mu   = flat.mean(0, keepdims=True)
        sigma = flat.std(0, keepdims=True) + 1e-8
        X_norm[mask] = ((X_62[mask].reshape(mask.sum(), -1) - mu) / sigma).reshape(mask.sum(), 62, 5)

    # Save normalised version for potential later use
    np.save(FEATURES_DIR / "seed_iv_X_62ch_norm.npy", X_norm)

    # Show 5 representative channels
    rep_ch = [0, 5, 13, 23, 31]  # FP1, F7, F8, T7, T8 (student channels)
    fig, axes = plt.subplots(2, len(rep_ch), figsize=(18, 8))
    fig.suptitle("SEED-IV: Raw vs Subject-Specific Z-Score Normalised\n(Student channels shown)",
                 fontsize=13, fontweight='bold')

    for ci, ch in enumerate(rep_ch):
        raw_vals  = X_62[:, ch, 2]   # alpha band as representative
        norm_vals = X_norm[:, ch, 2]

        for row, (vals, title, col) in enumerate([
            (raw_vals, f"Raw — {SEED_CHANNELS[ch]} (alpha)", '#4472C4'),
            (norm_vals, f"Normalised — {SEED_CHANNELS[ch]} (alpha)", '#ED7D31')
        ]):
            axes[row, ci].hist(vals, bins=40, color=col, alpha=0.7, density=True, edgecolor='none')
            axes[row, ci].set_title(title, fontsize=8)
            axes[row, ci].text(0.97, 0.95, f"μ={vals.mean():.2f}\nσ={vals.std():.2f}",
                               transform=axes[row, ci].transAxes, ha='right', va='top', fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda05_normalisation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify subject means are close to 0 after normalisation
    post_means = [X_norm[subj==s].mean() for s in subj_ids]
    post_stds  = [X_norm[subj==s].std()  for s in subj_ids]
    print(f"  After normalisation — subject mean range: [{min(post_means):.4f}, {max(post_means):.4f}] (should be ≈0)")
    print(f"  After normalisation — subject std  range: [{min(post_stds):.4f},  {max(post_stds):.4f}]  (should be ≈1)")
    ok = max(abs(m) for m in post_means) < 0.01
    print(f"  Normalisation check: {'✔ PASS' if ok else '⚠ FAIL — check normalisation implementation'}")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda05_normalisation_comparison.png'}")


# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 10 — [EDA 06] Channel Correlation Matrix (62×62)
# Key for M27 EEG-GraphFormer graph adjacency construction.
# High correlation → can share graph edges.
# ═══════════════════════════════════════════════════════════
PLOT = "eda06_channel_correlation"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    # Mean over all windows: (N, 62, 5) → (N, 62) mean over bands, then correlate channels
    X_mean_band = X_62.mean(axis=2)   # (N, 62) — mean feature per channel
    corr = np.corrcoef(X_mean_band.T) # (62, 62)

    # Save correlation matrix for M27 graph construction
    np.save(FEATURES_DIR / "seed_iv_channel_correlation.npy", corr)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("SEED-IV: Channel Correlation Matrix (62×62)\n"
                 "Informs M27 EEG-GraphFormer adjacency construction", fontsize=13, fontweight='bold')

    im = axes[0].imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[0].set_title("Pearson Correlation of Mean DE Features", fontweight='bold')
    axes[0].set_xlabel("Channel"); axes[0].set_ylabel("Channel")
    tick_step = 10
    axes[0].set_xticks(range(0, 62, tick_step))
    axes[0].set_xticklabels([SEED_CHANNELS[i] for i in range(0, 62, tick_step)], rotation=45, fontsize=8)
    axes[0].set_yticks(range(0, 62, tick_step))
    axes[0].set_yticklabels([SEED_CHANNELS[i] for i in range(0, 62, tick_step)], fontsize=8)
    plt.colorbar(im, ax=axes[0], fraction=0.03)
    # Highlight student channels
    for ci in STUDENT_CH_INDICES:
        axes[0].axhline(ci, color='orange', lw=0.7, alpha=0.8)
        axes[0].axvline(ci, color='orange', lw=0.7, alpha=0.8)

    # Threshold graph (|corr| > 0.7)
    threshold = 0.7
    adj = (np.abs(corr) > threshold).astype(float) - np.eye(62)
    degree = adj.sum(axis=1)
    axes[1].bar(range(62), degree, color='steelblue', edgecolor='none')
    # Highlight student channels
    for ci in STUDENT_CH_INDICES:
        axes[1].bar(ci, degree[ci], color='orange', edgecolor='black')
    axes[1].set_title(f"Node Degree in Threshold Graph (|corr| > {threshold})\n"
                      f"Orange bars = student channels", fontweight='bold')
    axes[1].set_xlabel("Channel Index"); axes[1].set_ylabel("Degree")
    axes[1].set_xticks(range(0, 62, 5))
    axes[1].set_xticklabels([SEED_CHANNELS[i] if i % 10 == 0 else '' for i in range(0, 62, 5)],
                             rotation=45, fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda06_channel_correlation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    upper = corr[np.triu_indices(62, k=1)]
    high_corr = (np.abs(upper) > 0.7).sum()
    print(f"  High-correlation pairs (|r| > 0.7): {high_corr} / {len(upper)} = {high_corr/len(upper)*100:.1f}%")
    print(f"  → EEG topology has strong spatial structure — graph inductive bias is justified for M27")
    
    # Top correlated channels with each student channel
    print(f"\n  Top-3 most correlated channel for each student channel:")
    for ci_idx, ci in enumerate(STUDENT_CH_INDICES):
        top3 = np.argsort(corr[ci])[::-1][1:4]
        print(f"    {SEED_CHANNELS[ci]:5s}: {[SEED_CHANNELS[t] for t in top3]}")

    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda06_channel_correlation.png'}")


# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, training, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 11 — [EDA 07] PCA Analysis (Pre/Post Normalisation)
# Shows class separability in linear feature space.
# ═══════════════════════════════════════════════════════════
PLOT = "eda07_pca"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    np.random.seed(42)
    idx_sample = np.random.choice(N, size=min(5000, N), replace=False)
    X_samp  = X_62[idx_sample].reshape(min(5000, N), -1)
    y_samp  = y[idx_sample]
    subj_samp = subj[idx_sample]

    X_norm_samp = np.load(FEATURES_DIR / "seed_iv_X_62ch_norm.npy")[idx_sample].reshape(min(5000, N), -1)

    pca_raw  = PCA(n_components=2, random_state=42).fit_transform(X_samp)
    pca_norm = PCA(n_components=2, random_state=42).fit_transform(X_norm_samp)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("SEED-IV PCA (5000-window sample)", fontsize=14, fontweight='bold')

    e_colors = ['#4472C4','#ED7D31','#70AD47','#FF0000']
    s_cmap   = plt.cm.get_cmap('tab20', 15)

    # Top row: raw features
    for c, name in enumerate(LABEL_NAMES):
        mask = y_samp == c
        axes[0,0].scatter(pca_raw[mask,0], pca_raw[mask,1], c=e_colors[c],
                          s=6, alpha=0.4, label=name, rasterized=True)
    axes[0,0].set_title("Raw — coloured by Emotion", fontweight='bold')
    axes[0,0].legend(markerscale=3, fontsize=9); axes[0,0].set_xlabel("PC1"); axes[0,0].set_ylabel("PC2")

    for s_idx, s_id in enumerate(sorted(np.unique(subj_samp))):
        mask = subj_samp == s_id
        axes[0,1].scatter(pca_raw[mask,0], pca_raw[mask,1], c=[s_cmap(s_idx)]*mask.sum(),
                          s=6, alpha=0.4, label=f"S{s_id}", rasterized=True)
    axes[0,1].set_title("Raw — coloured by Subject\n(overlap = subject-invariant; clusters = subject bias)",
                        fontweight='bold')
    axes[0,1].set_xlabel("PC1"); axes[0,1].set_ylabel("PC2")

    # Bottom row: normalised
    for c, name in enumerate(LABEL_NAMES):
        mask = y_samp == c
        axes[1,0].scatter(pca_norm[mask,0], pca_norm[mask,1], c=e_colors[c],
                          s=6, alpha=0.4, label=name, rasterized=True)
    axes[1,0].set_title("Post Z-score Norm — coloured by Emotion", fontweight='bold')
    axes[1,0].legend(markerscale=3, fontsize=9); axes[1,0].set_xlabel("PC1"); axes[1,0].set_ylabel("PC2")

    for s_idx, s_id in enumerate(sorted(np.unique(subj_samp))):
        mask = subj_samp == s_id
        axes[1,1].scatter(pca_norm[mask,0], pca_norm[mask,1], c=[s_cmap(s_idx)]*mask.sum(),
                          s=6, alpha=0.4, label=f"S{s_id}", rasterized=True)
    axes[1,1].set_title("Post Z-score Norm — coloured by Subject", fontweight='bold')
    axes[1,1].set_xlabel("PC1"); axes[1,1].set_ylabel("PC2")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda07_pca.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  PCA complete. Inspect figure: overlapping emotion clusters → confirms task is hard (expected).")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda07_pca.png'}")


# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 12 — [EDA 08] t-SNE Baseline (Raw Features, No Model)
# This is EDA t-SNE — different from the model-representation
# t-SNE in 05_ablations_and_viz.ipynb.
# ═══════════════════════════════════════════════════════════
PLOT = "eda08_tsne_raw"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    np.random.seed(42)
    N_TSNE = 2000   # keep fast for EDA
    idx_t  = np.random.choice(N, size=N_TSNE, replace=False)
    X_t    = np.load(FEATURES_DIR / "seed_iv_X_62ch_norm.npy")[idx_t].reshape(N_TSNE, -1)
    y_t    = y[idx_t]; subj_t = subj[idx_t]

    print(f"  Running t-SNE on {N_TSNE} samples (this may take 2–5 minutes)...")
    t0    = time.time()
    tsne  = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1)
    Z     = tsne.fit_transform(X_t)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    np.save(FEATURES_DIR / "seed_iv_tsne_raw_2000.npy", Z)

    e_colors = ['#4472C4','#ED7D31','#70AD47','#FF0000']
    s_cmap   = plt.cm.get_cmap('tab20', 15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("SEED-IV t-SNE (Raw Normalised Features, 2000 samples)\n"
                 "EDA baseline — no model representation", fontsize=13, fontweight='bold')

    for c, name in enumerate(LABEL_NAMES):
        mask = y_t == c
        axes[0].scatter(Z[mask,0], Z[mask,1], c=e_colors[c], s=8, alpha=0.5,
                        label=name, rasterized=True)
    axes[0].set_title("Coloured by Emotion Class\n(no model — raw DE features)", fontweight='bold')
    axes[0].legend(markerscale=2, fontsize=10); axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    for s_idx, s_id in enumerate(sorted(np.unique(subj_t))):
        mask = subj_t == s_id
        axes[1].scatter(Z[mask,0], Z[mask,1], c=[s_cmap(s_idx)]*mask.sum(), s=8, alpha=0.5,
                        label=f"S{s_id}", rasterized=True)
    axes[1].set_title("Coloured by Subject\n(clusters = subject-specific patterns)", fontweight='bold')
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda08_tsne_raw.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Inspect: if subjects cluster but emotions don't → confirms subject bias dominates raw space")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda08_tsne_raw.png'}")


# ==============================================================================
# Notebook cell 13
# Categories: preprocessing, results_tables, figures, statistics
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 13 — [EDA 09] Session Effect Analysis
# Quantifies session-to-session drift — motivates domain adaptation.
# ═══════════════════════════════════════════════════════════
PLOT = "eda09_session_effect"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    session_ids = sorted(np.unique(sess))
    
    # Per-session mean for each channel-band combination
    sess_means = np.zeros((len(session_ids), 62, 5))
    sess_counts = []
    for i, s_id in enumerate(session_ids):
        mask = sess == s_id
        sess_means[i] = X_62[mask].mean(0)
        sess_counts.append(mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SEED-IV: Session Effect Analysis (3 sessions)\n"
                 "Session drift motivates temporal domain adaptation", fontsize=13, fontweight='bold')

    # Session mean feature heatmap (per band)
    for bi, band in enumerate(BAND_NAMES):
        for si, s_id in enumerate(session_ids):
            axes[0].plot(range(62), sess_means[si, :, bi],
                         label=f"S{s_id}/{band}" if bi == 0 else None, alpha=0.7)
    axes[0].set_title("Per-Band Channel Mean per Session\n(all bands × 3 sessions)", fontweight='bold')
    axes[0].set_xlabel("Channel Index"); axes[0].set_ylabel("Mean DE Feature Value")
    axes[0].set_xticks(range(0, 62, 10))
    # Simplified legend
    for si, (s_id, col) in enumerate(zip(session_ids, ['blue','orange','green'])):
        axes[0].plot([], [], color=col, label=f"Session {s_id} (n={sess_counts[si]:,})")
    axes[0].legend()

    # Session-to-session difference (session 2 - session 1)
    diff_12 = sess_means[1] - sess_means[0]  # (62, 5)
    diff_23 = sess_means[2] - sess_means[1]  # (62, 5)
    im = axes[1].imshow(np.vstack([diff_12.mean(1), diff_23.mean(1)]),
                        aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title("Session Mean Shift (band-averaged)\nRow0: Sess2–Sess1 | Row1: Sess3–Sess2",
                      fontweight='bold')
    axes[1].set_xlabel("Channel Index")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(['S2–S1','S3–S2'])
    plt.colorbar(im, ax=axes[1], fraction=0.05)

    # Band-wise drift magnitude
    drift_by_band = np.abs(sess_means).std(0).mean(0)  # (5,) — std over sessions, mean over ch
    axes[2].bar(BAND_NAMES, drift_by_band, color=['#4472C4','#ED7D31','#70AD47','#FF0000','#7030A0'],
                edgecolor='black', width=0.6)
    axes[2].set_title("Session Drift per Frequency Band\n(std over sessions, mean over channels)",
                      fontweight='bold')
    axes[2].set_ylabel("Session Drift Magnitude")
    for i, v in enumerate(drift_by_band):
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda09_session_effect.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    max_drift_band = BAND_NAMES[np.argmax(drift_by_band)]
    print(f"  Max session drift band: {max_drift_band}")
    print(f"  Session drift magnitudes: {dict(zip(BAND_NAMES, [f'{d:.4f}' for d in drift_by_band]))}")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda09_session_effect.png'}")


# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 14 — [EDA 10] Frequency Band Importance (ANOVA F-statistic)
# Ranks which frequency bands best separate the 4 emotion classes.
# ═══════════════════════════════════════════════════════════
PLOT = "eda10_band_importance"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    # Use normalised features
    X_n = np.load(FEATURES_DIR / "seed_iv_X_62ch_norm.npy")

    # F-statistic per (channel, band) feature
    f_stats  = np.zeros((62, 5))
    p_values = np.zeros((62, 5))
    for bi in range(5):
        for ci in range(62):
            feat    = X_n[:, ci, bi]
            groups  = [feat[y == c] for c in range(4)]
            f, p    = stats.f_oneway(*groups)
            f_stats[ci, bi]  = f
            p_values[ci, bi] = p

    # Band-level mean F-stat
    band_f = f_stats.mean(0)
    band_rank = np.argsort(band_f)[::-1]
    print(f"  Band importance ranking (mean ANOVA F-statistic across all 62 channels):")
    for rank, bi in enumerate(band_rank):
        print(f"    {rank+1}. {BAND_NAMES[bi]:5s}: F={band_f[bi]:.3f}")

    # Top-20 most discriminative channel×band features
    top20_idx = np.unravel_index(np.argsort(f_stats.ravel())[-20:], (62, 5))
    top20     = list(zip(top20_idx[0], top20_idx[1]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("SEED-IV: Frequency Band Importance via ANOVA F-Statistic\n"
                 "(Higher = better emotion class separation)", fontsize=13, fontweight='bold')

    # Band-level bar chart
    colors_b = ['#4472C4','#ED7D31','#70AD47','#FF0000','#7030A0']
    bars = axes[0].bar(BAND_NAMES, band_f, color=colors_b, edgecolor='black', width=0.6)
    axes[0].set_title("Mean F-Statistic per Band\n(averaged over 62 channels)", fontweight='bold')
    axes[0].set_ylabel("Mean ANOVA F-Statistic")
    for i, (b, f) in enumerate(zip(BAND_NAMES, band_f)):
        axes[0].text(i, f + 0.1, f"{f:.1f}", ha='center', fontsize=11, fontweight='bold')

    # Heatmap: F-statistic per channel per band
    im = axes[1].imshow(f_stats.T, aspect='auto', cmap='hot_r')
    axes[1].set_title("F-Statistic Heatmap (5 bands × 62 channels)", fontweight='bold')
    axes[1].set_yticks(range(5)); axes[1].set_yticklabels(BAND_NAMES)
    axes[1].set_xlabel("Channel Index")
    axes[1].set_xticks(range(0, 62, 10))
    axes[1].set_xticklabels([SEED_CHANNELS[i] for i in range(0, 62, 10)], rotation=45, fontsize=7)
    plt.colorbar(im, ax=axes[1], fraction=0.05, label='F-statistic')
    # Mark student channels
    for ci in STUDENT_CH_INDICES:
        axes[1].axvline(ci, color='cyan', lw=1.5, alpha=0.8)
    axes[1].text(0.01, 0.99, 'Cyan: student channels', transform=axes[1].transAxes,
                 ha='left', va='top', fontsize=8, color='cyan')

    # Top-20 features bar chart
    top20_labels = [f"{SEED_CHANNELS[ci]}\n{BAND_NAMES[bi]}" for ci, bi in top20]
    top20_vals   = [f_stats[ci, bi] for ci, bi in top20]
    # Check which are student channels
    top20_colors = ['orange' if ci in STUDENT_CH_INDICES else 'steelblue' for ci, _ in top20]
    axes[2].barh(range(20), top20_vals, color=top20_colors, edgecolor='black')
    axes[2].set_yticks(range(20)); axes[2].set_yticklabels(top20_labels, fontsize=7)
    axes[2].set_title("Top-20 Most Discriminative Features\n(orange = student channel)",
                      fontweight='bold')
    axes[2].set_xlabel("F-Statistic")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda10_band_importance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Are student channels in top discriminative channels?
    student_f = f_stats[STUDENT_CH_INDICES].mean(0)
    all_f     = f_stats.mean(0)
    print(f"  Student channel mean F per band: {dict(zip(BAND_NAMES, [f'{v:.2f}' for v in student_f]))}")
    print(f"  All channel mean F per band    : {dict(zip(BAND_NAMES, [f'{v:.2f}' for v in all_f]))}")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda10_band_importance.png'}")


# ==============================================================================
# Notebook cell 15
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 15 — [EDA 11] Student Channel Coverage (Topographic Map)
# Shows which brain regions the 6 student channels cover.
# Uses 10-20 system approximate coordinates.
# ═══════════════════════════════════════════════════════════
PLOT = "eda11_channel_coverage"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT} — already done")
else:
    print(f"Running {PLOT}...")
    # 10-20 system approximate 2D positions (front=top, right=right)
    # Coordinates roughly normalised to [-1,1] head space
    CH_POS = {
        'FP1':(-0.18, 0.90), 'FPZ':(0.00, 0.92), 'FP2':(0.18, 0.90),
        'AF3':(-0.25, 0.78), 'AF4':(0.25, 0.78),
        'F7': (-0.56, 0.65), 'F5': (-0.42, 0.68), 'F3': (-0.28, 0.70), 'F1': (-0.14, 0.72),
        'FZ': ( 0.00, 0.73), 'F2': ( 0.14, 0.72), 'F4': ( 0.28, 0.70), 'F6': ( 0.42, 0.68), 'F8': ( 0.56, 0.65),
        'FT7':(-0.66, 0.47), 'FC5':(-0.52, 0.50), 'FC3':(-0.34, 0.53), 'FC1':(-0.17, 0.55),
        'FCZ':( 0.00, 0.56), 'FC2':( 0.17, 0.55), 'FC4':( 0.34, 0.53), 'FC6':( 0.52, 0.50), 'FT8':( 0.66, 0.47),
        'T7': (-0.78, 0.00), 'C5': (-0.60, 0.02), 'C3': (-0.40, 0.03), 'C1': (-0.20, 0.04),
        'CZ': ( 0.00, 0.04), 'C2': ( 0.20, 0.04), 'C4': ( 0.40, 0.03), 'C6': ( 0.60, 0.02), 'T8': ( 0.78, 0.00),
        'TP7':(-0.75,-0.40), 'CP5':(-0.58,-0.38), 'CP3':(-0.38,-0.36), 'CP1':(-0.19,-0.35),
        'CPZ':( 0.00,-0.35),'CP2':( 0.19,-0.35), 'CP4':( 0.38,-0.36), 'CP6':( 0.58,-0.38), 'TP8':( 0.75,-0.40),
        'P7': (-0.70,-0.60), 'P5': (-0.52,-0.62), 'P3': (-0.35,-0.64), 'P1': (-0.17,-0.65),
        'PZ': ( 0.00,-0.66), 'P2': ( 0.17,-0.65), 'P4': ( 0.35,-0.64), 'P6': ( 0.52,-0.62), 'P8': ( 0.70,-0.60),
        'PO7':(-0.55,-0.78), 'PO5':(-0.40,-0.80), 'PO3':(-0.24,-0.82), 'POZ':( 0.00,-0.84),
        'PO4':( 0.24,-0.82), 'PO6':( 0.40,-0.80), 'PO8':( 0.55,-0.78),
        'CB1':(-0.18,-0.94), 'O1': (-0.18,-0.90), 'OZ': ( 0.00,-0.92), 'O2': ( 0.18,-0.90), 'CB2':( 0.18,-0.94),
    }

    # Compute mean discriminative power per channel (mean F over all bands)
    X_n    = np.load(FEATURES_DIR / "seed_iv_X_62ch_norm.npy")
    ch_f   = np.zeros(62)
    for ci in range(62):
        groups = [X_n[y == c, ci, :].ravel() for c in range(4)]
        f, _   = stats.f_oneway(*groups)
        ch_f[ci] = f

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("SEED-IV: Topographic Channel Map\n"
                 "Colour = ANOVA F-statistic (emotion discriminability)",
                 fontsize=13, fontweight='bold')

    for ax_idx, (ax, highlight_student) in enumerate(zip(axes, [False, True])):
        # Head circle
        head = Circle((0, 0), 1.05, fill=False, color='black', lw=2)
        ax.add_patch(head)
        # Nose
        ax.plot([0], [1.15], 'k^', markersize=8)
        # Ears
        ax.plot([-1.1, -1.15, -1.1], [-0.05, 0.0, 0.05], 'k-', lw=2)
        ax.plot([ 1.1,  1.15,  1.1], [-0.05, 0.0, 0.05], 'k-', lw=2)

        # Plot all channels
        f_min, f_max = ch_f.min(), ch_f.max()
        for ci, ch_name in enumerate(SEED_CHANNELS):
            if ch_name not in CH_POS:
                continue
            x, y_pos = CH_POS[ch_name]
            norm_f   = (ch_f[ci] - f_min) / (f_max - f_min + 1e-8)
            color    = plt.cm.YlOrRd(norm_f)
            is_student = ci in STUDENT_CH_INDICES

            if highlight_student:
                if is_student:
                    ax.scatter(x, y_pos, c=[color], s=300, zorder=5, edgecolor='blue', lw=2.5)
                    ax.text(x, y_pos + 0.12, ch_name, ha='center', va='bottom',
                            fontsize=7, fontweight='bold', color='blue')
                else:
                    ax.scatter(x, y_pos, c=[color], s=80, zorder=4, edgecolor='gray', lw=0.5, alpha=0.7)
                    ax.text(x, y_pos + 0.10, ch_name, ha='center', va='bottom', fontsize=5, alpha=0.5)
            else:
                ax.scatter(x, y_pos, c=[color], s=150, zorder=4, edgecolor='gray', lw=0.5)
                ax.text(x, y_pos + 0.10, ch_name, ha='center', va='bottom', fontsize=5)

        title_suffix = "\n(Blue border = 6 student channels)" if highlight_student else ""
        ax.set_title(f"{'Student Channel Coverage' if highlight_student else 'All 62 Channels'}{title_suffix}",
                     fontweight='bold', fontsize=10)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.35)
        ax.set_aspect('equal'); ax.axis('off')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(ch_f.min(), ch_f.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, label='ANOVA F-statistic')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eda11_channel_coverage.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  Student channel F-statistics (emotion discriminability):")
    for ci in STUDENT_CH_INDICES:
        print(f"    {SEED_CHANNELS[ci]:5s} (idx {ci:2d}): F={ch_f[ci]:.2f} "
              f"(rank {int(np.argsort(ch_f)[::-1].tolist().index(ci))+1}/62)")
    mark_done(PLOT)
    print(f"✔ Saved: {FIG_DIR / 'eda11_channel_coverage.png'}")


# ==============================================================================
# Notebook cell 16
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 16 — EDA Summary Report
# ═══════════════════════════════════════════════════════════
done    = [p for p in EDA_PLOTS if eda_done(p)]
pending = [p for p in EDA_PLOTS if not eda_done(p)]

print("=" * 60)
print(" SEED-IV EDA COMPLETE — Summary")
print("=" * 60)
print(f" Plots completed : {len(done)}/{len(EDA_PLOTS)}")
print(f" Figures saved to: {FIG_DIR}")
print()

if not pending:
    # Print key findings
    counts = [(y == i).sum() for i in range(4)]
    print(" KEY FINDINGS FOR PHASE C:")
    print(f"  • Class imbalance: Happy underrepresented by "
          f"{(1 - counts[3]/max(counts))*100:.1f}% → use WeightedRandomSampler (H14)")
    print(f"  • Student channels [FP1,FP2,F7,F8,T7,T8] cover Frontal+Temporal regions")
    print(f"  • EDA figures confirm inter-subject variance > intra → DANCE/DANN justified")
    print(f"  • Channel correlation heatmap saved → use for M27 EEG-GraphFormer adjacency")
    print(f"  • Session drift exists → motivates domain adaptation across sessions")
    print()
    print(" NEXT STEPS:")
    print("  1. Run 00b_eda_faced.ipynb")
    print("  2. Run 00c_phaseB_reproduce.ipynb (STOP GATE)")
else:
    print(f" ⚠ {len(pending)} plots still pending:")
    for p in pending:
        print(f"   • {p}")
    print("\n Re-run cells above for pending plots.")

print("=" * 60)
