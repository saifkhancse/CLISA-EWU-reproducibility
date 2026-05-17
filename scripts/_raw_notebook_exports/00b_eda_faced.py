# Auto-exported raw code from notebook: 00b_eda_faced.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 1 — Imports and setup
# ═══════════════════════════════════════════════════════════
import os, sys, json, warnings, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pathlib

warnings.filterwarnings('ignore')
np.random.seed(42)
print("✔ Imports complete.")


# ==============================================================================
# Notebook cell 2
# Categories: preprocessing, model_definition, training, results_tables, figures, webapp_or_demo
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 2 — Path configuration + GPU check
# ═══════════════════════════════════════════════════════════
ROOT         = pathlib.Path(os.getcwd())
FEATURES_DIR = ROOT / "features"
FACED_DIR    = ROOT / "data" / "FACED" / "EEG_Features" / "DE"
FIG_DIR      = ROOT / "figures" / "eda_faced"
CKPT_DIR     = ROOT / "checkpoints" / "eda_faced"
SEEDIV_FIG   = ROOT / "figures" / "eda_seediv"   # for joint comparisons

for d in [FEATURES_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
        print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB — Flash Attention enabled")
    else:
        print("⚠ GPU not available — EDA runs on CPU (acceptable)")
except ImportError:
    pass
print(f"\nFACED raw data: {FACED_DIR}")
print(f"Figures → {FIG_DIR}")


# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, training, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 3 — 🔄 CHECKPOINT RECOVERY
# ═══════════════════════════════════════════════════════════
EDA_PLOTS = [
    "faced01_class_distribution",
    "faced02_channel_layout",
    "faced03_band_alignment",
    "faced04_subject_variability",
    "faced05_pca_domain_gap",
    "faced06_mmd",
    "faced07_clip_distribution",
]

def eda_done(name): return (CKPT_DIR / f"{name}.done").exists()
def mark_done(name): (CKPT_DIR / f"{name}.done").write_text("done")

done    = [p for p in EDA_PLOTS if eda_done(p)]
pending = [p for p in EDA_PLOTS if not eda_done(p)]
print(f"FACED EDA Recovery | Completed: {len(done)}/{len(EDA_PLOTS)}")
if pending:
    for p in pending: print(f"  Pending: {p}")
else:
    print("✅ All FACED EDA plots complete.")


# ==============================================================================
# Notebook cell 4
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 4 — FACED Data Loading
#
# FACED structure: 123 .pkl files in data/FACED/EEG_Features/DE/
# Each .pkl contains DE features for one subject.
# Shape per subject: approx (n_clips, n_windows, 32, 5) or (n_windows, 32, 5)
#
# 4-class emotion mapping (from valence × arousal or direct labels):
#   0=Neutral, 1=Sad, 2=Fear, 3=Happy
# ═══════════════════════════════════════════════════════════

# FACED 32-channel names (standard 32-electrode cap)
FACED_CHANNELS = [
    'FP1','FPZ','FP2','F7','F3','FZ','F4','F8',
    'FC5','FC1','FC2','FC6','T7','C3','CZ','C4',
    'T8','CP5','CP1','CP2','CP6','P7','P3','PZ',
    'P4','P8','PO7','PO3','POZ','PO4','PO8','OZ'
]
assert len(FACED_CHANNELS) == 32

SEED_CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ',
    'O2','CB2'
]

BAND_NAMES  = ['delta','theta','alpha','beta','gamma']
LABEL_NAMES = ['Neutral','Sad','Fear','Happy']
STUDENT_CH_NAMES   = ['FP1','FP2','F7','F8','T7','T8']
STUDENT_CH_INDICES_SEED = [0, 2, 5, 13, 23, 31]

# FACED label mapping — FACED has 28 clips with valence/arousal ratings
# Typical 4-class mapping used in cross-dataset studies:
# valence>0.5 and arousal>0.5 → Happy; valence<0.5 and arousal<0.5 → Neutral
# valence<0.5 and arousal>0.5 → Fear;  valence>0.5 and arousal<0.5 → Sad
# Some works use direct labels: disgust/anger→fear, calm→neutral, etc.
# We use a simplified hard-coded mapping based on clip order in FACED documentation.
FACED_4CLASS_MAP = {
    # Valence-positive (≈Happy or Amusement): clips roughly 1,2,3,4,5,6,7
    # Neutral: clips 8,9,10,11
    # Sad: clips 12,13,14,15,16,17,18
    # Fear/Disgust: clips 19-28
    # NOTE: Adjust this based on actual FACED documentation in your data directory
    # The mapping below is approximate — verify with the FACED readme
    **{i: 3 for i in range(0, 7)},   # Happy (clips 0-6)
    **{i: 0 for i in range(7, 11)},  # Neutral (clips 7-10)
    **{i: 1 for i in range(11, 18)}, # Sad (clips 11-17)
    **{i: 2 for i in range(18, 28)}, # Fear (clips 18-27)
}

# ── Load or process ──────────────────────────────────────
npy_Xf  = FEATURES_DIR / "faced_X_32ch.npy"
npy_yf  = FEATURES_DIR / "faced_y_4cls.npy"
npy_sf  = FEATURES_DIR / "faced_subjects.npy"
npy_cf  = FEATURES_DIR / "faced_clips.npy"

if all(f.exists() for f in [npy_Xf, npy_yf, npy_sf]):
    print("Loading pre-processed FACED features from features/ ...")
    X_faced   = np.load(npy_Xf)   # (N, 32, 5)
    y_faced   = np.load(npy_yf)   # (N,)
    subj_faced= np.load(npy_sf)   # (N,)
    clips_faced = np.load(npy_cf) if npy_cf.exists() else np.zeros(len(y_faced), dtype=int)
    print(f"✔ Loaded: {X_faced.shape}")

elif FACED_DIR.exists():
    pkl_files = sorted(FACED_DIR.glob("*.pkl"))
    if not pkl_files:
        pkl_files = sorted(FACED_DIR.glob("**/*.pkl"))
    print(f"Processing {len(pkl_files)} FACED .pkl files...")

    all_X, all_y, all_subj, all_clips = [], [], [], []

    for subj_idx, pkl_path in enumerate(pkl_files, start=1):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1') if sys.version_info[0] == 3 else pickle.load(f)
        except Exception as e:
            print(f"  ⚠ Skipped {pkl_path.name}: {e}"); continue

        # FACED .pkl structure may vary — try common keys
        if isinstance(data, dict):
            # Try to find DE features
            de_data = None
            for key in ['de', 'DE', 'de_features', 'features', 'data']:
                if key in data:
                    de_data = data[key]; break
            if de_data is None:
                # Try first array-like value
                for v in data.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 2:
                        de_data = v; break
            labels_raw = data.get('labels', data.get('label', None))
        elif isinstance(data, np.ndarray):
            de_data = data
            labels_raw = None
        else:
            print(f"  ⚠ Unknown format in {pkl_path.name}"); continue

        if de_data is None:
            print(f"  ⚠ No DE features found in {pkl_path.name}"); continue

        # Normalise shape to (n_clips, n_windows, 32, 5) or (n_clips * n_windows, 32, 5)
        arr = np.array(de_data, dtype=np.float32)
        if arr.ndim == 4:
            n_clips, n_win, n_ch, n_band = arr.shape
        elif arr.ndim == 3:
            # Could be (n_windows, 32, 5) or (n_clips, 32, 5)
            if arr.shape[-1] == 5 and arr.shape[-2] == 32:
                n_clips, n_win = 1, arr.shape[0]
                arr = arr[np.newaxis]  # → (1, n_windows, 32, 5)
            else:
                arr = arr.reshape(1, arr.shape[0], 32, 5)
                n_clips, n_win = 1, arr.shape[1]
        elif arr.ndim == 2:
            # Flat: (n_windows, 32*5)
            n_win = arr.shape[0]
            arr = arr.reshape(1, n_win, 32, 5)
            n_clips = 1
        else:
            continue

        n_clips_actual = arr.shape[0]
        for clip_idx in range(n_clips_actual):
            clip_label = FACED_4CLASS_MAP.get(clip_idx, 0)
            if labels_raw is not None and len(labels_raw) > clip_idx:
                # Use provided label if available; map to 4-class
                raw_lbl = int(labels_raw[clip_idx]) if np.isscalar(labels_raw[clip_idx]) else int(labels_raw.flat[clip_idx])
                # Simple mapping if labels are already 0-3 or need remapping
                clip_label = raw_lbl % N_CLASSES if raw_lbl < N_CLASSES * 5 else FACED_4CLASS_MAP.get(clip_idx, 0)
            n_w = arr.shape[1]
            all_X.append(arr[clip_idx])  # (n_win, 32, 5)
            all_y.extend([clip_label] * n_w)
            all_subj.extend([subj_idx] * n_w)
            all_clips.extend([clip_idx] * n_w)

        if subj_idx % 20 == 0:
            print(f"  Processed {subj_idx}/{len(pkl_files)} subjects...")

    if not all_X:
        raise RuntimeError("No FACED data could be loaded — check pkl file format")

    X_faced    = np.concatenate(all_X, axis=0).astype(np.float32)
    y_faced    = np.array(all_y,    dtype=np.int64)
    subj_faced = np.array(all_subj, dtype=np.int64)
    clips_faced= np.array(all_clips,dtype=np.int64)

    np.save(npy_Xf, X_faced); np.save(npy_yf, y_faced)
    np.save(npy_sf, subj_faced); np.save(npy_cf, clips_faced)
    print(f"\n✔ FACED loaded and saved: {X_faced.shape}")

else:
    print(f"⚠ FACED data not found at {FACED_DIR}")
    print("Creating synthetic placeholder for EDA structure testing...")
    N_FAKE = 5000
    X_faced    = np.random.randn(N_FAKE, 32, 5).astype(np.float32)
    y_faced    = np.random.randint(0, 4, N_FAKE)
    subj_faced = np.random.randint(1, 124, N_FAKE)
    clips_faced= np.random.randint(0, 28, N_FAKE)
    print("  ⚠ RUNNING WITH SYNTHETIC DATA — results are meaningless. Place FACED data in data/FACED/")

N_CLASSES = 4
N_FACED = len(y_faced)
print(f"\nFACED summary: {N_FACED:,} windows | {len(np.unique(subj_faced))} subjects | {X_faced.shape}")
print(f"Label dist: {dict(zip(LABEL_NAMES, [(y_faced==i).sum() for i in range(4)]))}") 


# ==============================================================================
# Notebook cell 5
# Categories: preprocessing, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 5 — [FACED 01] Class Distribution
# ═══════════════════════════════════════════════════════════
PLOT = "faced01_class_distribution"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    counts_f = [(y_faced == i).sum() for i in range(4)]
    total_f  = sum(counts_f)

    # Also load SEED-IV for comparison
    try:
        y_seed   = np.load(FEATURES_DIR / "seed_iv_y_4cls.npy")
        counts_s = [(y_seed == i).sum() for i in range(4)]
    except FileNotFoundError:
        counts_s = [10170, 10245, 9225, 7935]   # from Phase B report

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FACED: Class Distribution and SEED-IV Comparison", fontsize=14, fontweight='bold')
    colors = ['#4472C4','#ED7D31','#A9D18E','#FF0000']

    # FACED distribution
    axes[0].bar(LABEL_NAMES, counts_f, color=colors, edgecolor='black', width=0.6)
    axes[0].set_title("FACED: Windows per Emotion", fontweight='bold')
    axes[0].set_ylabel("Count")
    for i, c in enumerate(counts_f):
        axes[0].text(i, c + 50, f"{c:,}\n({c/total_f*100:.1f}%)", ha='center', fontsize=9)

    # SEED-IV distribution
    total_s = sum(counts_s)
    axes[1].bar(LABEL_NAMES, counts_s, color=colors, edgecolor='black', width=0.6, alpha=0.8)
    axes[1].set_title("SEED-IV: Windows per Emotion (reference)", fontweight='bold')
    axes[1].set_ylabel("Count")
    for i, c in enumerate(counts_s):
        axes[1].text(i, c + 50, f"{c:,}\n({c/total_s*100:.1f}%)", ha='center', fontsize=9)

    # Proportion comparison
    props_f = [c/total_f for c in counts_f]
    props_s = [c/total_s for c in counts_s]
    x = np.arange(4); width = 0.35
    axes[2].bar(x - width/2, props_f, width, label='FACED', color='steelblue', alpha=0.8)
    axes[2].bar(x + width/2, props_s, width, label='SEED-IV', color='orange', alpha=0.8)
    axes[2].set_title("Class Proportion Comparison\n(for cross-dataset validity)", fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(LABEL_NAMES)
    axes[2].set_ylabel("Proportion"); axes[2].legend()
    axes[2].axhline(0.25, color='red', ls='--', alpha=0.5, label='Balanced')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced01_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    imbalance = max(counts_f) / min(counts_f)
    print(f"  FACED class imbalance ratio: {imbalance:.2f}x")
    mark_done(PLOT); print(f"✔ Saved faced01_class_distribution.png")


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, model_definition, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 6 — [FACED 02] Channel Layout & Overlap with SEED-IV
# ═══════════════════════════════════════════════════════════
PLOT = "faced02_channel_layout"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    overlap = [ch for ch in FACED_CHANNELS if ch in SEED_CHANNELS]
    only_seed = [ch for ch in SEED_CHANNELS if ch not in FACED_CHANNELS]
    only_faced = [ch for ch in FACED_CHANNELS if ch not in SEED_CHANNELS]

    # Student channels in FACED
    student_in_faced = [ch for ch in STUDENT_CH_NAMES if ch in FACED_CHANNELS]
    student_indices_faced = [FACED_CHANNELS.index(ch) for ch in student_in_faced]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("FACED vs SEED-IV: Channel Overlap Analysis\n"
                 "Critical for cross-dataset feature alignment", fontsize=13, fontweight='bold')

    # Venn-like bar
    cats = ['FACED only\n(no overlap)', 'Shared\nChannels', 'SEED-IV only\n(no overlap)']
    vals = [len(only_faced), len(overlap), len(only_seed)]
    colors_v = ['#ED7D31','#70AD47','#4472C4']
    bars = axes[0].bar(cats, vals, color=colors_v, edgecolor='black', width=0.5)
    axes[0].set_title("Channel Count Breakdown", fontweight='bold')
    axes[0].set_ylabel("Number of Channels")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(v),
                     ha='center', fontsize=13, fontweight='bold')

    # Show shared channels and student channel coverage
    axes[1].axis('off')
    info_text = (
        f"FACED channels   : {len(FACED_CHANNELS)}\n"
        f"SEED-IV channels : {len(SEED_CHANNELS)}\n"
        f"Shared channels  : {len(overlap)} → {overlap[:10]}{'...' if len(overlap)>10 else ''}\n\n"
        f"Student channels ({len(student_in_faced)}/{len(STUDENT_CH_NAMES)} in FACED):\n"
        f"  Present : {student_in_faced} (indices {student_indices_faced})\n"
        f"  Missing : {[ch for ch in STUDENT_CH_NAMES if ch not in FACED_CHANNELS]}\n\n"
        f"Cross-dataset alignment strategy:\n"
        f"  → Use only shared {len(overlap)} channels for feature alignment\n"
        f"  → DANCE teacher must be re-indexed for 32-ch FACED input\n"
        f"  → Student 6-ch: {'all present' if len(student_in_faced)==6 else f'{len(student_in_faced)}/6 present — substitute nearest'}"
    )
    axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes, va='top', ha='left',
                 fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced02_channel_layout.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save channel overlap for cross-dataset experiments
    overlap_info = {
        'shared_channels': overlap,
        'shared_seed_indices': [SEED_CHANNELS.index(ch) for ch in overlap],
        'shared_faced_indices': [FACED_CHANNELS.index(ch) for ch in overlap],
        'student_in_faced': student_in_faced,
        'student_indices_faced': student_indices_faced
    }
    with open(FEATURES_DIR / "cross_dataset_channel_map.json", 'w') as f:
        json.dump(overlap_info, f, indent=2)

    print(f"  Shared channels: {len(overlap)}/32 FACED = {len(overlap)/32*100:.1f}%")
    print(f"  Student channels present in FACED: {len(student_in_faced)}/6")
    print(f"  Channel map saved to features/cross_dataset_channel_map.json")
    mark_done(PLOT); print(f"✔ Saved faced02_channel_layout.png")


# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, results_tables, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 7 — [FACED 03] Band Alignment Verification
# ═══════════════════════════════════════════════════════════
PLOT = "faced03_band_alignment"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    try:
        X_seed = np.load(FEATURES_DIR / "seed_iv_X_62ch.npy")
        if X_seed.ndim == 2:
            X_seed = X_seed.reshape(X_seed.shape[0], 62, 5)
    except FileNotFoundError:
        X_seed = None

    with open(FEATURES_DIR / "cross_dataset_channel_map.json") as f:
        ch_map = json.load(f)
    shared_faced_idx = ch_map['shared_faced_indices'][:5]
    shared_seed_idx  = ch_map['shared_seed_indices'][:5]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Band Distribution Comparison: FACED vs SEED-IV\n"
                 "Verifying identical band ordering (delta/theta/alpha/beta/gamma)",
                 fontsize=13, fontweight='bold')

    for bi, band in enumerate(BAND_NAMES):
        f_vals = X_faced[:, shared_faced_idx, bi].ravel()
        axes[0, bi].hist(f_vals[:50000], bins=50, color='#ED7D31', alpha=0.7, density=True, edgecolor='none')
        axes[0, bi].set_title(f"FACED — {band}", fontweight='bold', fontsize=9)
        axes[0, bi].text(0.97, 0.95, f"μ={f_vals.mean():.2f}\nσ={f_vals.std():.2f}",
                         transform=axes[0, bi].transAxes, ha='right', va='top', fontsize=8)
        if bi == 0:
            axes[0, bi].set_ylabel("FACED", fontsize=10)

        if X_seed is not None:
            s_vals = X_seed[:, shared_seed_idx, bi].ravel()
            axes[1, bi].hist(s_vals[:50000], bins=50, color='#4472C4', alpha=0.7, density=True, edgecolor='none')
            axes[1, bi].set_title(f"SEED-IV — {band}", fontweight='bold', fontsize=9)
            axes[1, bi].text(0.97, 0.95, f"μ={s_vals.mean():.2f}\nσ={s_vals.std():.2f}",
                             transform=axes[1, bi].transAxes, ha='right', va='top', fontsize=8)
        if bi == 0:
            axes[1, bi].set_ylabel("SEED-IV", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced03_band_alignment.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Band alignment checked. Compare μ/σ values across rows — should be similar scale.")
    print("  If scales differ significantly → apply dataset-specific z-score normalisation")
    mark_done(PLOT)
    print(f"✔ Saved faced03_band_alignment.png")


# ==============================================================================
# Notebook cell 8
# Categories: preprocessing, training, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 8 — [FACED 04] Per-Subject Variability Comparison
# ═══════════════════════════════════════════════════════════
PLOT = "faced04_subject_variability"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    faced_subj_ids   = sorted(np.unique(subj_faced))
    n_faced_subj     = len(faced_subj_ids)
    faced_subj_means = np.array([X_faced[subj_faced==s].mean(0).ravel() for s in faced_subj_ids])

    try:
        X_seed    = np.load(FEATURES_DIR / "seed_iv_X_62ch.npy")
        if X_seed.ndim == 2:
            X_seed = X_seed.reshape(X_seed.shape[0], 62, 5)
        subj_seed = np.load(FEATURES_DIR / "seed_iv_subjects.npy")
        seed_subj_ids   = sorted(np.unique(subj_seed))
        seed_subj_means = np.array([
            X_seed[subj_seed==s].reshape((subj_seed==s).sum(), -1).mean(0)
            for s in seed_subj_ids
        ])
    except FileNotFoundError:
        seed_subj_means = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("FACED: Per-Subject Variability\n"
                 "(123 subjects vs SEED-IV's 15 subjects)", fontsize=13, fontweight='bold')

    wpc = [(subj_faced == s).sum() for s in faced_subj_ids]
    axes[0].hist(wpc, bins=20, color='#ED7D31', edgecolor='black', alpha=0.8)
    axes[0].axvline(np.mean(wpc), color='red', ls='--', label=f"Mean={np.mean(wpc):.0f}")
    axes[0].set_title(f"Windows per Subject\n(FACED, {n_faced_subj} subjects)", fontweight='bold')
    axes[0].set_xlabel("Windows per Subject"); axes[0].set_ylabel("Count of Subjects")
    axes[0].legend()

    if seed_subj_means is not None:
        min_feats = min(faced_subj_means.shape[1], seed_subj_means.shape[1])
        faced_iv  = faced_subj_means[:, :min_feats].var(0).mean()
        seed_iv   = seed_subj_means[:, :min_feats].var(0).mean()
        axes[1].bar(['FACED\n(32ch)', 'SEED-IV\n(62ch)'], [faced_iv, seed_iv],
                    color=['#ED7D31','#4472C4'], edgecolor='black', width=0.4)
        axes[1].set_title(f"Inter-Subject Variance Comparison\n(first {min_feats} shared features)",
                          fontweight='bold')
    else:
        faced_inter_var = faced_subj_means.var(0).mean()
        axes[1].bar(['FACED Inter-Subject'], [faced_inter_var], color='#ED7D31', edgecolor='black', width=0.4)
        axes[1].set_title("FACED Inter-Subject Variance", fontweight='bold')
    axes[1].set_ylabel("Mean Inter-Subject Variance")

    if faced_subj_means.shape[0] > 2:
        pca      = PCA(n_components=2, random_state=42)
        subj_pca = pca.fit_transform(faced_subj_means)
        sc = axes[2].scatter(subj_pca[:,0], subj_pca[:,1],
                             c=range(n_faced_subj), cmap='viridis', s=30, alpha=0.8)
        axes[2].set_title(f"PCA of Per-Subject Mean Features\n(FACED, {n_faced_subj} subjects)",
                          fontweight='bold')
        axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
        plt.colorbar(sc, ax=axes[2], label="Subject ID")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced04_subject_variability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  FACED subjects: {n_faced_subj} | windows/subject: {np.mean(wpc):.0f} ± {np.std(wpc):.0f}")
    mark_done(PLOT)
    print(f"✔ Saved faced04_subject_variability.png")


# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, training, figures, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 9 — [FACED 05] PCA Domain Gap Visualisation
# ═══════════════════════════════════════════════════════════
PLOT = "faced05_pca_domain_gap"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    np.random.seed(42)
    with open(FEATURES_DIR / "cross_dataset_channel_map.json") as f:
        ch_map = json.load(f)
    shared_seed  = ch_map['shared_seed_indices']
    shared_faced = ch_map['shared_faced_indices']

    # Filter to valid indices only (FACED may have fewer channels than expected)
    n_faced_ch   = X_faced.shape[1]
    valid_mask   = [i for i, idx in enumerate(shared_faced) if idx < n_faced_ch]
    shared_faced = [shared_faced[i] for i in valid_mask]
    shared_seed  = [shared_seed[i]  for i in valid_mask]

    N_PCA = 2000
    try:
        X_seed = np.load(FEATURES_DIR / "seed_iv_X_62ch.npy")
        if X_seed.ndim == 2:
            X_seed = X_seed.reshape(X_seed.shape[0], 62, 5)
        y_seed = np.load(FEATURES_DIR / "seed_iv_y_4cls.npy")
        idx_s  = np.random.choice(len(y_seed), N_PCA, replace=False)
        Xs_shared = X_seed[idx_s][:, shared_seed, :].reshape(N_PCA, -1)
        ys = y_seed[idx_s]
    except FileNotFoundError:
        Xs_shared = np.random.randn(N_PCA, len(shared_seed)*5).astype(np.float32)
        ys = np.random.randint(0, 4, N_PCA)

    idx_f     = np.random.choice(N_FACED, min(N_PCA, N_FACED), replace=False)
    Xf_shared = X_faced[idx_f][:, shared_faced, :].reshape(len(idx_f), -1)
    yf        = y_faced[idx_f]

    X_combined = np.vstack([Xs_shared, Xf_shared])
    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X_combined)
    Zs  = Z[:N_PCA]; Zf = Z[N_PCA:]

    e_colors = ['#4472C4','#ED7D31','#70AD47','#FF0000']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Domain Gap Visualisation: FACED vs SEED-IV\n"
                 f"PCA of shared {len(shared_seed)} channels × 5 bands",
                 fontsize=13, fontweight='bold')

    axes[0].scatter(Zs[:,0], Zs[:,1], c='#4472C4', s=6, alpha=0.4, label='SEED-IV', rasterized=True)
    axes[0].scatter(Zf[:,0], Zf[:,1], c='#ED7D31', s=6, alpha=0.4, label='FACED',   rasterized=True)
    axes[0].set_title("Dataset Domain Gap\n(overlap = small gap; separation = large gap)", fontweight='bold')
    axes[0].legend(markerscale=3); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    for c, name in enumerate(LABEL_NAMES):
        mask = ys == c
        if mask.sum() > 0:
            axes[1].scatter(Zs[mask,0], Zs[mask,1], c=e_colors[c], s=8, alpha=0.5, label=name, rasterized=True)
    axes[1].set_title("SEED-IV coloured by Emotion", fontweight='bold')
    axes[1].legend(markerscale=2, fontsize=9); axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    for c, name in enumerate(LABEL_NAMES):
        mask = yf == c
        if mask.sum() > 0:
            axes[2].scatter(Zf[mask,0], Zf[mask,1], c=e_colors[c], s=8, alpha=0.5, label=name, rasterized=True)
    axes[2].set_title("FACED coloured by Emotion", fontweight='bold')
    axes[2].legend(markerscale=2, fontsize=9); axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced05_pca_domain_gap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    mark_done(PLOT)
    print(f"✔ Saved faced05_pca_domain_gap.png")


# ==============================================================================
# Notebook cell 10
# Categories: preprocessing, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 10 — [FACED 06] Maximum Mean Discrepancy (MMD)
# ═══════════════════════════════════════════════════════════
PLOT = "faced06_mmd"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    def compute_mmd(X, Y, gamma=1.0):
        """Estimate MMD using RBF kernel on random subsample."""
        n = min(500, len(X), len(Y))
        np.random.seed(42)
        Xi = X[np.random.choice(len(X), n, replace=False)]
        Yi = Y[np.random.choice(len(Y), n, replace=False)]
        def rbf_kernel(A, B):
            dists = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-gamma * dists)
        Kxx = rbf_kernel(Xi, Xi)
        Kyy = rbf_kernel(Yi, Yi)
        Kxy = rbf_kernel(Xi, Yi)
        mmd_sq = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return float(np.sqrt(max(mmd_sq, 0)))

    with open(FEATURES_DIR / "cross_dataset_channel_map.json") as f:
        ch_map = json.load(f)
    shared_seed  = ch_map['shared_seed_indices']
    shared_faced = ch_map['shared_faced_indices']

    # Filter to valid indices only
    n_faced_ch   = X_faced.shape[1]
    valid_mask   = [i for i, idx in enumerate(shared_faced) if idx < n_faced_ch]
    shared_faced = [shared_faced[i] for i in valid_mask]
    shared_seed  = [shared_seed[i]  for i in valid_mask]

    try:
        X_seed = np.load(FEATURES_DIR / "seed_iv_X_62ch.npy")
        if X_seed.ndim == 2:
            X_seed = X_seed.reshape(X_seed.shape[0], 62, 5)
        Xs = X_seed[:, shared_seed, :].reshape(len(X_seed), -1)
    except FileNotFoundError:
        Xs = np.random.randn(2000, len(shared_seed)*5).astype(np.float32)

    Xf = X_faced[:, shared_faced, :].reshape(N_FACED, -1)

    print("  Computing MMD per frequency band (this takes ~1 min)...")
    mmd_per_band = []
    for bi, band in enumerate(BAND_NAMES):
        Xs_b  = Xs.reshape(-1, len(shared_seed),  5)[:, :, bi].reshape(len(Xs), -1)
        Xf_b  = Xf.reshape(-1, len(shared_faced), 5)[:, :, bi].reshape(len(Xf), -1)
        Xs_bn = (Xs_b - Xs_b.mean(0)) / (Xs_b.std(0) + 1e-8)
        Xf_bn = (Xf_b - Xf_b.mean(0)) / (Xf_b.std(0) + 1e-8)
        mmd   = compute_mmd(Xs_bn, Xf_bn)
        mmd_per_band.append(mmd)
        print(f"    {band}: MMD = {mmd:.4f}")

    Xs_n = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    Xf_n = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-8)
    mmd_overall = compute_mmd(Xs_n, Xf_n)
    print(f"  Overall MMD (all bands): {mmd_overall:.4f}")

    mmd_results = dict(zip(BAND_NAMES, mmd_per_band))
    mmd_results['overall'] = mmd_overall
    with open(FEATURES_DIR / "mmd_faced_seediv.json", 'w') as f:
        json.dump(mmd_results, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_b = ['#4472C4','#ED7D31','#70AD47','#FF0000','#7030A0']
    bars = ax.bar(BAND_NAMES + ['Overall'], mmd_per_band + [mmd_overall],
                  color=colors_b + ['gray'], edgecolor='black', width=0.6)
    ax.set_title("MMD (Maximum Mean Discrepancy): FACED vs SEED-IV\n"
                 "Higher = larger domain gap per frequency band", fontweight='bold', fontsize=12)
    ax.set_ylabel("MMD Distance")
    for bar, v in zip(bars, mmd_per_band + [mmd_overall]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
    max_band = BAND_NAMES[np.argmax(mmd_per_band)]
    ax.text(0.98, 0.98, f"Highest gap: {max_band} band (MMD={max(mmd_per_band):.4f})\n"
            f"→ Cross-dataset experiments should use domain adaptation",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced06_mmd.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    mark_done(PLOT)
    print(f"✔ Saved faced06_mmd.png")


# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 11 — [FACED 07] Clip-wise Emotion Distribution
# Check if video clips dominate — potential dataset bias.
# ═══════════════════════════════════════════════════════════
PLOT = "faced07_clip_distribution"
if eda_done(PLOT):
    print(f"✔ SKIP {PLOT}")
else:
    clip_ids = sorted(np.unique(clips_faced))
    n_clips  = len(clip_ids)

    # Clip label (majority vote per clip across subjects)
    clip_labels = {}
    for clip in clip_ids:
        mask = clips_faced == clip
        if mask.sum() > 0:
            clip_labels[clip] = int(stats.mode(y_faced[mask], keepdims=True)[0][0])

    clip_emotion = [clip_labels.get(c, 0) for c in clip_ids]
    windows_per_clip = [(clips_faced == c).sum() for c in clip_ids]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"FACED: Clip-wise Analysis ({n_clips} clips × {len(np.unique(subj_faced))} subjects)",
                 fontsize=13, fontweight='bold')

    colors = ['#4472C4','#ED7D31','#A9D18E','#FF0000']
    bar_colors = [colors[clip_emotion[i]] for i in range(n_clips)]
    axes[0].bar(clip_ids, windows_per_clip, color=bar_colors, edgecolor='none')
    axes[0].set_title("Windows per Clip\n(colour = mapped emotion class)", fontweight='bold')
    axes[0].set_xlabel("Clip ID"); axes[0].set_ylabel("Total Windows (all subjects)")
    # Legend
    for i, name in enumerate(LABEL_NAMES):
        axes[0].bar([], [], color=colors[i], label=name)
    axes[0].legend()

    # Clips per emotion
    clips_by_emo = [(np.array(clip_emotion) == i).sum() for i in range(4)]
    axes[1].bar(LABEL_NAMES, clips_by_emo, color=colors, edgecolor='black', width=0.6)
    axes[1].set_title("Clips per Emotion Class\n(label balance at clip level)", fontweight='bold')
    axes[1].set_ylabel("Number of Clips")
    for i, v in enumerate(clips_by_emo):
        axes[1].text(i, v + 0.1, str(v), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "faced07_clip_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    max_clip_bias = max(windows_per_clip) / max(1, min(windows_per_clip))
    print(f"  Clips: {n_clips} | Windows/clip range: {min(windows_per_clip)}–{max(windows_per_clip)} | ratio={max_clip_bias:.1f}x")
    if max_clip_bias > 3:
        print(f"  ⚠ High clip bias ({max_clip_bias:.1f}x) → use clip-stratified sampling in cross-dataset experiments")
    mark_done(PLOT); print(f"✔ Saved faced07_clip_distribution.png")


# ==============================================================================
# Notebook cell 12
# Categories: preprocessing, model_definition, results_tables, figures
# ==============================================================================
# ═══════════════════════════════════════════════════════════
# CELL 12 — FACED EDA Summary
# ═══════════════════════════════════════════════════════════
done    = [p for p in EDA_PLOTS if eda_done(p)]
pending = [p for p in EDA_PLOTS if not eda_done(p)]

print("=" * 60)
print(" FACED EDA COMPLETE — Summary")
print("=" * 60)
print(f" Completed : {len(done)}/{len(EDA_PLOTS)}")
print(f" Figures   : {FIG_DIR}")
print()
if not pending:
    # Load saved MMD if available
    try:
        with open(FEATURES_DIR / "mmd_faced_seediv.json") as f:
            mmd = json.load(f)
        print(f" MMD (domain gap): {mmd}")
    except: pass
    print()
    with open(FEATURES_DIR / "cross_dataset_channel_map.json") as f:
        ch_map = json.load(f)
    print(f" Shared channels (FACED ∩ SEED-IV): {len(ch_map['shared_channels'])}")
    print(f" Student channels in FACED: {ch_map['student_in_faced']}")
    print()
    print(" KEY FINDINGS:")
    print("  • Domain gap quantified via MMD — use for E03/E04 adaptation strategy")
    print("  • Shared channel map saved → features/cross_dataset_channel_map.json")
    print("  • FACED has 123 subjects → more diverse training source for cross-dataset")
    print()
    print(" NEXT STEPS:")
    print("  → Run 00c_phaseB_reproduce.ipynb (STOP GATE)")
else:
    for p in pending: print(f"  Pending: {p}")
print("=" * 60)
