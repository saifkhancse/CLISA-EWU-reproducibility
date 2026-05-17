# Auto-exported raw code from notebook: 09_final_results_rebuild_verified.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 2
# Categories: figures
# ==============================================================================
# CELL 1.1 — Core imports
import os
import json
import math
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optional: scipy for stats if available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, statistical tests will be skipped")

# Suppress harmless warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("✓ Imports loaded successfully")


# ==============================================================================
# Notebook cell 3
# Categories: figures
# ==============================================================================
# CELL 1.2 — Matplotlib style configuration
# Publication-style theme: black text, readable fonts, high DPI

plt.style.use('default')  # Start from clean slate

# Set global font sizes
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10

# Black text everywhere
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'

# High DPI for publication quality
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# White background, visible grid
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

# Define color palette (similar to v2 notebook but with black text)
COLORS = {
    'neutral': '#4C72B0',    # Blue
    'sad': '#DD8452',        # Orange
    'fear': '#55A868',       # Green
    'happy': '#C44E52',      # Red
    'classical': '#8C8C8C',  # Gray
    'deep': '#64B5CD',       # Cyan
    'dance': '#9467BD',      # Purple
    'baseline': '#E377C2'    # Pink
}

print("✓ Matplotlib style configured: black text, large fonts, DPI=300")


# ==============================================================================
# Notebook cell 5
# Categories: audit_verification
# ==============================================================================
# CELL 2.1 — Resolve project root

# Expected root from prompt
EXPECTED_ROOT = Path(r"C:\Users\Saif\Desktop\CSE400\C")

# Try to resolve from current working directory
ROOT = Path.cwd()

# If CWD is not the project root, attempt to find it
if not (ROOT / 'results').exists():
    # Check if we're in a subdirectory
    for parent in ROOT.parents:
        if (parent / 'results').exists():
            ROOT = parent
            break
    else:
        # Fall back to expected root
        if EXPECTED_ROOT.exists():
            ROOT = EXPECTED_ROOT
        else:
            print(f"⚠ WARNING: Could not find project root. Using CWD: {ROOT}")

print(f"✓ Project ROOT: {ROOT}")
print(f"  Exists: {ROOT.exists()}")
print(f"  results/ exists: {(ROOT / 'results').exists()}")


# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, results_tables, figures, audit_verification
# ==============================================================================
# CELL 2.2 — Define source-of-truth paths

# Input paths (source of truth)
SOURCE_PATHS = {
    'classical_summary': ROOT / 'results' / 'classical_ml' / 'classical_ml_summary.csv',
    'deep_master_summary': ROOT / 'results' / 'deep_models_seediv' / 'deep_models_master_summary.csv',
    'dance_m25_loso': ROOT / 'results' / 'deep_models_seediv' / 'M25_62ch_summary.csv',
    'dance_m26_loso': ROOT / 'results' / 'deep_models_seediv' / 'M26_6ch_summary.csv',
    'dance_loso_crosscheck': ROOT / 'results' / 'loso' / 'dance_loso.csv',
    'dance_reproduction': ROOT / 'results' / 'phaseB_reproduction' / 'phaseB_reproduce_results.csv',
    'ablations': ROOT / 'results' / 'ablations' / 'ablations_partial_summary.csv'
}

# Per-model deep learning summaries (M11-M24)
DEEP_MODEL_DIR = ROOT / 'results' / 'deep_models_seediv'
DEEP_MODEL_IDS = [f'M{i:02d}' for i in range(11, 25)]  # M11-M24

# Output paths
OUTPUT_DIRS = {
    'tables': ROOT / 'results' / 'final_tables',
    'figures_png': ROOT / 'results' / 'final_figures' / 'png',
    'figures_pdf': ROOT / 'results' / 'final_figures' / 'pdf',
    'audit': ROOT / 'results' / 'final_audit'
}

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {dir_path.relative_to(ROOT)}")

print("\n✓ All output directories ready")


# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, evaluation, results_tables, figures, statistics, audit_verification
# ==============================================================================
# CELL 2.3 — Save output contract JSON

contract = {
    'root': str(ROOT),
    'timestamp': datetime.now().isoformat(),
    'source_of_truth': {k: str(v) for k, v in SOURCE_PATHS.items()},
    'output_dirs': {k: str(v) for k, v in OUTPUT_DIRS.items()},
    'expected_tables': {
        'table_5_1_classical_main.csv': 'Chapter 5.1 classical summary (62ch + 6ch)',
        'table_5_2_deep_main.csv': 'Chapter 5.2 deep baseline summary',
        'table_5_3_dance_loso_verified.csv': 'Chapter 5.3 verified LOSO DANCE',
        'table_5_3b_dance_reproduction.csv': 'Chapter 5.3 reproduction (E00/H17)',
        'table_5_4_ablations.csv': 'Chapter 5.5 ablation table',
        'table_5_5_proto_gain.csv': 'Proto-A vs Proto-B gain summary',
        'table_5_6_channel_efficiency.csv': '62ch vs 6ch retention',
        'table_5_7_per_class_top_models.csv': 'Per-class F1 for top models',
        'table_audit_sources.csv': 'Audit table of source files'
    },
    'expected_figures': {
        'fig_5_1_classical_leaderboard': 'Classical results plot',
        'fig_5_2_deep_leaderboard_62ch': 'Deep baseline 62ch',
        'fig_5_3_deep_leaderboard_6ch': 'Deep baseline 6ch',
        'fig_5_4_protoA_vs_protoB': 'Proto-A vs Proto-B comparison',
        'fig_5_5_dance_headline': 'DANCE headline figure',
        'fig_5_6_tsne_geometry': 't-SNE representation geometry',
        'fig_5_7_ablation_abs': 'Ablation absolute AccB',
        'fig_5_8_ablation_delta': 'Ablation delta vs baseline',
        'fig_5_9_stat_sig': 'Statistical significance',
        'fig_5_10_channel_efficiency_scatter': '62ch vs 6ch scatter',
        'fig_5_11_channel_efficiency_retention': 'Retention ratio/bar',
        'fig_5_12_per_class_all_models': 'Per-class summary',
        'fig_5_13_per_class_top_62ch': 'Top 62ch per-class F1',
        'fig_5_14_per_class_top_6ch': 'Top 6ch per-class F1',
        'fig_5_15_confusion_top_models': 'Confusion matrices top models'
    }
}

contract_path = OUTPUT_DIRS['audit'] / 'output_contract.json'
with open(contract_path, 'w') as f:
    json.dump(contract, f, indent=2)

print(f"✓ Output contract saved: {contract_path.relative_to(ROOT)}")


# ==============================================================================
# Notebook cell 9
# Categories: audit_verification
# ==============================================================================
# CELL 3.1 — Initialize audit log

audit_log = []
audit_warnings = []
audit_errors = []

def log_audit(message, level='INFO'):
    """Add entry to audit log"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    entry = f"[{timestamp}] {level}: {message}"
    audit_log.append(entry)
    
    if level == 'WARNING':
        audit_warnings.append(message)
        print(f"⚠ {message}")
    elif level == 'ERROR':
        audit_errors.append(message)
        print(f"❌ {message}")
    else:
        print(f"  {message}")

log_audit("Audit log initialized")
log_audit(f"Project root: {ROOT}")


# ==============================================================================
# Notebook cell 10
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 3.2 — Check source file existence

log_audit("Checking source file existence...")

source_status = {}
for name, path in SOURCE_PATHS.items():
    exists = path.exists()
    source_status[name] = exists
    
    if exists:
        log_audit(f"✓ Found: {name} ({path.relative_to(ROOT)})")
    else:
        log_audit(f"Missing: {name} ({path.relative_to(ROOT)})", 'WARNING')

# Check per-model deep learning summaries
deep_model_status = {}
for model_id in DEEP_MODEL_IDS:
    for ch_config in ['62ch', '6ch']:
        filename = f"{model_id}_{ch_config}_summary.csv"
        filepath = DEEP_MODEL_DIR / filename
        exists = filepath.exists()
        deep_model_status[f"{model_id}_{ch_config}"] = exists
        
        if not exists:
            log_audit(f"Missing per-model file: {filename}", 'WARNING')

missing_count = len([k for k, v in source_status.items() if not v])
missing_deep = len([k for k, v in deep_model_status.items() if not v])

if missing_count == 0 and missing_deep == 0:
    log_audit("✓ All source files present")
else:
    log_audit(f"Missing {missing_count} core files and {missing_deep} per-model files", 'WARNING')


# ==============================================================================
# Notebook cell 11
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 3.3 — Load classical ML summary

if source_status['classical_summary']:
    classical_df = pd.read_csv(SOURCE_PATHS['classical_summary'])
    log_audit(f"Loaded classical_summary: {classical_df.shape[0]} rows, {classical_df.shape[1]} cols")
    log_audit(f"  Columns: {list(classical_df.columns)}")
    
    # Validate expected schema
    expected_cols = ['Model', 'Name', 'Channels', 'F1_mean', 'F1_std', 'Acc_mean', 'Acc_std']
    missing_cols = [c for c in expected_cols if c not in classical_df.columns]
    if missing_cols:
        log_audit(f"Classical schema missing columns: {missing_cols}", 'WARNING')
    else:
        log_audit("✓ Classical schema validated")
else:
    classical_df = None
    log_audit("Classical summary not available, skipping classical results", 'ERROR')


# ==============================================================================
# Notebook cell 12
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 3.4 — Load deep models master summary

if source_status['deep_master_summary']:
    deep_master_df = pd.read_csv(SOURCE_PATHS['deep_master_summary'])
    log_audit(f"Loaded deep_master_summary: {deep_master_df.shape[0]} rows")
    log_audit(f"  Columns: {list(deep_master_df.columns)}")
    
    # Validate schema
    expected_cols = ['model_id', 'ch', 'n_runs', 'acc_a_mean', 'acc_a_std', 
                     'f1_a_mean', 'f1_a_std', 'acc_b_mean', 'acc_b_std', 'f1_b_mean', 'f1_b_std']
    missing_cols = [c for c in expected_cols if c not in deep_master_df.columns]
    if missing_cols:
        log_audit(f"Deep master schema missing: {missing_cols}", 'WARNING')
    else:
        log_audit("✓ Deep master schema validated")
else:
    deep_master_df = None
    log_audit("Deep master summary not available", 'ERROR')


# ==============================================================================
# Notebook cell 13
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 3.5 — Load per-model deep learning summaries (M11-M24)

deep_per_model = {}

for model_id in DEEP_MODEL_IDS:
    for ch_config in ['62ch', '6ch']:
        key = f"{model_id}_{ch_config}"
        filename = f"{key}_summary.csv"
        filepath = DEEP_MODEL_DIR / filename
        
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                deep_per_model[key] = df
                log_audit(f"  Loaded {key}: {df.shape[0]} rows")
            except Exception as e:
                log_audit(f"Failed to load {filename}: {e}", 'ERROR')
                deep_per_model[key] = None
        else:
            deep_per_model[key] = None

loaded_count = sum(1 for v in deep_per_model.values() if v is not None)
log_audit(f"Loaded {loaded_count}/{len(deep_per_model)} per-model summaries")


# ==============================================================================
# Notebook cell 14
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 3.6 — Load DANCE LOSO summaries (M25, M26)

if source_status['dance_m25_loso']:
    dance_m25_df = pd.read_csv(SOURCE_PATHS['dance_m25_loso'])
    log_audit(f"Loaded M25 LOSO: {dance_m25_df.shape[0]} rows")
else:
    dance_m25_df = None
    log_audit("M25 LOSO summary not available", 'ERROR')

if source_status['dance_m26_loso']:
    dance_m26_df = pd.read_csv(SOURCE_PATHS['dance_m26_loso'])
    log_audit(f"Loaded M26 LOSO: {dance_m26_df.shape[0]} rows")
else:
    dance_m26_df = None
    log_audit("M26 LOSO summary not available", 'ERROR')

if source_status['dance_loso_crosscheck']:
    dance_crosscheck_df = pd.read_csv(SOURCE_PATHS['dance_loso_crosscheck'])
    log_audit(f"Loaded DANCE crosscheck: {dance_crosscheck_df.shape[0]} rows")
else:
    dance_crosscheck_df = None
    log_audit("DANCE crosscheck not available", 'WARNING')


# ==============================================================================
# Notebook cell 15
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 3.7 — Load DANCE reproduction results

if source_status['dance_reproduction']:
    dance_repro_df = pd.read_csv(SOURCE_PATHS['dance_reproduction'])
    log_audit(f"Loaded DANCE reproduction: {dance_repro_df.shape[0]} rows")
    log_audit(f"  Columns: {list(dance_repro_df.columns)}")
else:
    dance_repro_df = None
    log_audit("DANCE reproduction not available", 'WARNING')


# ==============================================================================
# Notebook cell 16
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 3.8 — Load ablation results

if source_status['ablations']:
    ablations_df = pd.read_csv(SOURCE_PATHS['ablations'])
    log_audit(f"Loaded ablations: {ablations_df.shape[0]} rows")
    
    # Note row count for audit
    if ablations_df.shape[0] < 14:
        log_audit(f"Ablation row count is {ablations_df.shape[0]}, expected 14", 'WARNING')
else:
    ablations_df = None
    log_audit("Ablations summary not available", 'WARNING')


# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, results_tables
# ==============================================================================
# CELL 4.1 — Helper functions for data processing

def safe_numeric(value):
    """Convert to float, return NaN if invalid"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def compute_mean_std(df, value_col):
    """Compute mean and std from DataFrame column"""
    values = df[value_col].apply(safe_numeric).dropna()
    if len(values) == 0:
        return np.nan, np.nan
    return values.mean(), values.std()

def standardize_model_id(model_str):
    """Standardize model ID to M01, M02, ... M26 format"""
    if isinstance(model_str, str):
        # Extract number
        import re
        match = re.search(r'(\d+)', model_str)
        if match:
            num = int(match.group(1))
            return f"M{num:02d}"
    return str(model_str)

def standardize_channel_label(ch_str):
    """Standardize channel label to '62ch' or '6ch'"""
    ch_str = str(ch_str).lower()
    if '62' in ch_str:
        return '62ch'
    elif '6' in ch_str:
        return '6ch'
    else:
        return ch_str

print("✓ Helper functions defined")


# ==============================================================================
# Notebook cell 19
# Categories: model_definition
# ==============================================================================
# CELL 4.2 — Model pretty-name mapping

MODEL_NAMES = {
    'M01': 'LDA',
    'M02': 'SVM (RBF)',
    'M03': 'Random Forest',
    'M04': 'k-NN',
    'M05': 'Logistic Regression',
    'M06': 'Naive Bayes',
    'M07': 'Extra Trees',
    'M08': 'Gradient Boosting',
    'M09': 'XGBoost',
    'M10': 'MLP (sklearn)',
    'M11': 'Shallow MLP',
    'M12': 'Deep MLP',
    'M13': 'LSTM',
    'M14': 'GRU',
    'M15': 'Conv1D',
    'M16': 'Vanilla Transformer',
    'M17': 'EEG Conformer',
    'M18': 'ChanDrop Transformer',
    'M19': 'DANN',
    'M20': 'CLISA',
    'M21': 'SimCLR',
    'M22': 'BYOL',
    'M23': 'PseudoLabel',
    'M24': 'MixMatch',
    'M25': 'DANCE Teacher',
    'M26': 'DANCE Student'
}

def get_model_name(model_id):
    """Get pretty name for model ID"""
    return MODEL_NAMES.get(str(model_id), str(model_id))

print(f"✓ Model name mapping defined for {len(MODEL_NAMES)} models")


# ==============================================================================
# Notebook cell 20
# Categories: results_tables, figures, audit_verification
# ==============================================================================
# CELL 4.3 — Figure and table save helpers

def save_figure(fig, name, dpi=300):
    """Save figure as both PNG and PDF"""
    png_path = OUTPUT_DIRS['figures_png'] / f"{name}.png"
    pdf_path = OUTPUT_DIRS['figures_pdf'] / f"{name}.pdf"
    
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    log_audit(f"Saved figure: {name}")
    return png_path, pdf_path

def save_table(df, name):
    """Save DataFrame as CSV table"""
    csv_path = OUTPUT_DIRS['tables'] / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    log_audit(f"Saved table: {name} ({df.shape[0]} rows)")
    return csv_path

print("✓ Save helpers defined")


# ==============================================================================
# Notebook cell 22
# Categories: preprocessing, audit_verification
# ==============================================================================
# CELL 5.1 — Process classical ML results

log_audit("Processing classical ML results...")

if classical_df is not None:
    # Standardize columns
    classical_official = classical_df.copy()
    
    # Standardize model IDs and channels
    if 'Model' in classical_official.columns:
        classical_official['model_id'] = classical_official['Model'].apply(standardize_model_id)
    if 'Channels' in classical_official.columns:
        classical_official['ch'] = classical_official['Channels'].apply(standardize_channel_label)
    
    # Add pretty names
    classical_official['name'] = classical_official['model_id'].apply(get_model_name)
    
    log_audit(f"✓ Classical official dataset: {classical_official.shape[0]} entries")
    log_audit(f"  Models: {sorted(classical_official['model_id'].unique())}")
    log_audit(f"  Channels: {sorted(classical_official['ch'].unique())}")
else:
    classical_official = None
    log_audit("Classical ML processing skipped (no source data)", 'ERROR')


# ==============================================================================
# Notebook cell 23
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 5.2 — Recompute deep baseline aggregated results from per-model CSVs

log_audit("Recomputing deep baseline aggregated results...")

deep_recomputed_rows = []

for model_id in DEEP_MODEL_IDS:
    for ch_config in ['62ch', '6ch']:
        key = f"{model_id}_{ch_config}"
        df = deep_per_model.get(key)
        
        if df is not None and len(df) > 0:
            # Aggregate across all runs
            row = {
                'model_id': model_id,
                'ch': ch_config,
                'n_runs': len(df),
                'acc_a_mean': df['acc_a'].mean() if 'acc_a' in df.columns else np.nan,
                'acc_a_std': df['acc_a'].std() if 'acc_a' in df.columns else np.nan,
                'f1_a_mean': df['f1_a'].mean() if 'f1_a' in df.columns else np.nan,
                'f1_a_std': df['f1_a'].std() if 'f1_a' in df.columns else np.nan,
                'acc_b_mean': df['acc_b'].mean() if 'acc_b' in df.columns else np.nan,
                'acc_b_std': df['acc_b'].std() if 'acc_b' in df.columns else np.nan,
                'f1_b_mean': df['f1_b'].mean() if 'f1_b' in df.columns else np.nan,
                'f1_b_std': df['f1_b'].std() if 'f1_b' in df.columns else np.nan
            }
            deep_recomputed_rows.append(row)

if deep_recomputed_rows:
    deep_recomputed = pd.DataFrame(deep_recomputed_rows)
    deep_recomputed['name'] = deep_recomputed['model_id'].apply(get_model_name)
    log_audit(f"✓ Deep recomputed: {len(deep_recomputed)} model-channel pairs")
else:
    deep_recomputed = None
    log_audit("Deep recomputation failed (no per-model data)", 'ERROR')


# ==============================================================================
# Notebook cell 24
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 5.3 — Compare recomputed vs master summary (audit check)

if deep_recomputed is not None and deep_master_df is not None:
    log_audit("Comparing recomputed deep results vs master summary...")
    
    comparison_rows = []
    TOLERANCE = 1e-4
    
    for _, recomp_row in deep_recomputed.iterrows():
        model_id = recomp_row['model_id']
        ch = recomp_row['ch']
        
        # Find matching row in master
        master_match = deep_master_df[
            (deep_master_df['model_id'] == model_id) & 
            (deep_master_df['ch'].apply(standardize_channel_label) == ch)
        ]
        
        if len(master_match) == 1:
            master_row = master_match.iloc[0]
            
            # Compare acc_b_mean
            recomp_val = recomp_row['acc_b_mean']
            master_val = master_row['acc_b_mean']
            diff = abs(recomp_val - master_val)
            
            comparison_rows.append({
                'model_id': model_id,
                'ch': ch,
                'recomputed_acc_b': recomp_val,
                'master_acc_b': master_val,
                'diff': diff,
                'match': diff < TOLERANCE
            })
            
            if diff >= TOLERANCE:
                log_audit(f"Mismatch {model_id}/{ch}: diff={diff:.6f}", 'WARNING')
    
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        mismatches = (~comparison_df['match']).sum()
        
        if mismatches > 0:
            log_audit(f"Found {mismatches} mismatches, preferring recomputed values", 'WARNING')
            # Save audit comparison
            save_table(comparison_df, 'audit_deep_comparison')
        else:
            log_audit("✓ All recomputed values match master summary")
            
    deep_official = deep_recomputed  # Use recomputed as official
else:
    deep_official = deep_master_df  # Fallback to master if no recomputation
    log_audit("Using master summary as official (no recomputation available)", 'WARNING')


# ==============================================================================
# Notebook cell 25
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# CELL 5.4 — Aggregate DANCE unified LOSO results (M25, M26)

log_audit("Processing DANCE unified LOSO results...")

dance_loso_rows = []

# M25 Teacher (62ch)
if dance_m25_df is not None:
    row = {
        'model_id': 'M25',
        'name': 'DANCE Teacher',
        'ch': '62ch',
        'n_runs': len(dance_m25_df),
        'acc_a_mean': dance_m25_df['acc_a'].mean() if 'acc_a' in dance_m25_df.columns else np.nan,
        'acc_a_std': dance_m25_df['acc_a'].std() if 'acc_a' in dance_m25_df.columns else np.nan,
        'f1_a_mean': dance_m25_df['f1_a'].mean() if 'f1_a' in dance_m25_df.columns else np.nan,
        'f1_a_std': dance_m25_df['f1_a'].std() if 'f1_a' in dance_m25_df.columns else np.nan,
        'acc_b_mean': dance_m25_df['acc_b'].mean() if 'acc_b' in dance_m25_df.columns else np.nan,
        'acc_b_std': dance_m25_df['acc_b'].std() if 'acc_b' in dance_m25_df.columns else np.nan,
        'f1_b_mean': dance_m25_df['f1_b'].mean() if 'f1_b' in dance_m25_df.columns else np.nan,
        'f1_b_std': dance_m25_df['f1_b'].std() if 'f1_b' in dance_m25_df.columns else np.nan,
        'mean_best_val_f1': dance_m25_df['best_val_f1'].mean() if 'best_val_f1' in dance_m25_df.columns else np.nan,
        'mean_elapsed': dance_m25_df['elapsed'].mean() if 'elapsed' in dance_m25_df.columns else np.nan
    }
    dance_loso_rows.append(row)
    log_audit(f"✓ M25 aggregated from {len(dance_m25_df)} runs")

# M26 Student (6ch)
if dance_m26_df is not None:
    row = {
        'model_id': 'M26',
        'name': 'DANCE Student',
        'ch': '6ch',
        'n_runs': len(dance_m26_df),
        'acc_a_mean': dance_m26_df['acc_a'].mean() if 'acc_a' in dance_m26_df.columns else np.nan,
        'acc_a_std': dance_m26_df['acc_a'].std() if 'acc_a' in dance_m26_df.columns else np.nan,
        'f1_a_mean': dance_m26_df['f1_a'].mean() if 'f1_a' in dance_m26_df.columns else np.nan,
        'f1_a_std': dance_m26_df['f1_a'].std() if 'f1_a' in dance_m26_df.columns else np.nan,
        'acc_b_mean': dance_m26_df['acc_b'].mean() if 'acc_b' in dance_m26_df.columns else np.nan,
        'acc_b_std': dance_m26_df['acc_b'].std() if 'acc_b' in dance_m26_df.columns else np.nan,
        'f1_b_mean': dance_m26_df['f1_b'].mean() if 'f1_b' in dance_m26_df.columns else np.nan,
        'f1_b_std': dance_m26_df['f1_b'].std() if 'f1_b' in dance_m26_df.columns else np.nan,
        'mean_best_val_f1': dance_m26_df['best_val_f1'].mean() if 'best_val_f1' in dance_m26_df.columns else np.nan,
        'mean_elapsed': dance_m26_df['elapsed'].mean() if 'elapsed' in dance_m26_df.columns else np.nan
    }
    dance_loso_rows.append(row)
    log_audit(f"✓ M26 aggregated from {len(dance_m26_df)} runs")

if dance_loso_rows:
    dance_loso_official = pd.DataFrame(dance_loso_rows)
    log_audit(f"✓ DANCE LOSO official: {len(dance_loso_official)} models")
else:
    dance_loso_official = None
    log_audit("DANCE LOSO processing failed", 'ERROR')


# ==============================================================================
# Notebook cell 26
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# CELL 5.5 — Cross-check DANCE LOSO vs dance_loso.csv

if dance_loso_official is not None and dance_crosscheck_df is not None:
    log_audit("Cross-checking DANCE LOSO against dance_loso.csv...")
    
    # Aggregate dance_loso.csv by model
    if 'model' in dance_crosscheck_df.columns:
        crosscheck_agg = dance_crosscheck_df.groupby('model').agg({
            'acc_A': ['mean', 'std'],
            'f1_A': ['mean', 'std'],
            'acc_B': ['mean', 'std'],
            'f1_B': ['mean', 'std']
        }).reset_index()
        
        log_audit(f"  Crosscheck aggregated: {len(crosscheck_agg)} models")
        
        # Compare M25 Teacher if present
        m25_crosscheck = crosscheck_agg[crosscheck_agg['model'].str.contains('teacher', case=False)]
        if len(m25_crosscheck) > 0 and 'M25' in dance_loso_official['model_id'].values:
            m25_official = dance_loso_official[dance_loso_official['model_id'] == 'M25'].iloc[0]
            m25_cross = m25_crosscheck.iloc[0]
            
            diff_acc_b = abs(m25_official['acc_b_mean'] - m25_cross[('acc_B', 'mean')])
            if diff_acc_b > 0.001:
                log_audit(f"M25 AccB mismatch: {diff_acc_b:.4f}", 'WARNING')
            else:
                log_audit("✓ M25 crosscheck matches")
else:
    log_audit("DANCE crosscheck skipped (missing data)", 'WARNING')


# ==============================================================================
# Notebook cell 27
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# CELL 5.6 — Process DANCE reproduction results (separate from LOSO)

log_audit("Processing DANCE reproduction results...")

if dance_repro_df is not None and len(dance_repro_df) > 0:
    # Build tidy reproduction table
    repro_rows = []
    
    # Assuming single row with all metrics
    r = dance_repro_df.iloc[0]
    
    # Teacher E00
    if 'e00_teacher_acc_b' in r:
        repro_rows.append({
            'variant': 'E00',
            'model': 'Teacher',
            'acc_a': r.get('e00_teacher_acc_a', np.nan),
            'acc_b': r.get('e00_teacher_acc_b', np.nan),
            'ref_acc_b': r.get('teacher_ref_b', np.nan),
            'delta_vs_ref_b': r.get('e00_teacher_acc_b', np.nan) - r.get('teacher_ref_b', np.nan)
        })
    
    # Teacher H17
    if 'h17_teacher_acc_b' in r:
        repro_rows.append({
            'variant': 'H17',
            'model': 'Teacher',
            'acc_a': r.get('h17_teacher_acc_a', np.nan),
            'acc_b': r.get('h17_teacher_acc_b', np.nan),
            'ref_acc_b': r.get('teacher_ref_b', np.nan),
            'delta_vs_ref_b': r.get('h17_teacher_acc_b', np.nan) - r.get('teacher_ref_b', np.nan)
        })
    
    # Student E00
    if 'e00_student_acc_b' in r:
        repro_rows.append({
            'variant': 'E00',
            'model': 'Student',
            'acc_a': r.get('e00_student_acc_a', np.nan),
            'acc_b': r.get('e00_student_acc_b', np.nan),
            'ref_acc_b': r.get('student_ref_b', np.nan),
            'delta_vs_ref_b': r.get('e00_student_acc_b', np.nan) - r.get('student_ref_b', np.nan)
        })
    
    # Student H17
    if 'h17_student_acc_b' in r:
        repro_rows.append({
            'variant': 'H17',
            'model': 'Student',
            'acc_a': r.get('h17_student_acc_a', np.nan),
            'acc_b': r.get('h17_student_acc_b', np.nan),
            'ref_acc_b': r.get('student_ref_b', np.nan),
            'delta_vs_ref_b': r.get('h17_student_acc_b', np.nan) - r.get('student_ref_b', np.nan)
        })
    
    if repro_rows:
        dance_repro_official = pd.DataFrame(repro_rows)
        log_audit(f"✓ DANCE reproduction table: {len(dance_repro_official)} entries")
    else:
        dance_repro_official = None
        log_audit("DANCE reproduction parsing failed", 'WARNING')
else:
    dance_repro_official = None
    log_audit("DANCE reproduction not available", 'WARNING')


# ==============================================================================
# Notebook cell 29
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 6.1 — Table 5.1: Classical ML Main

log_audit("Generating Table 5.1: Classical ML main results...")

if classical_official is not None:
    table_5_1 = classical_official[[
        'model_id', 'name', 'ch', 'F1_mean', 'F1_std', 'Acc_mean', 'Acc_std'
    ]].copy()
    
    # Add per-class F1 if available
    for col in ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']:
        if col in classical_official.columns:
            table_5_1[col.lower()] = classical_official[col]
    
    # Add N_runs if available
    if 'N_runs' in classical_official.columns:
        table_5_1['n_runs'] = classical_official['N_runs']
    
    save_table(table_5_1, 'table_5_1_classical_main')
else:
    log_audit("Table 5.1 skipped (no classical data)", 'WARNING')


# ==============================================================================
# Notebook cell 30
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 6.2 — Table 5.2: Deep Learning Main

log_audit("Generating Table 5.2: Deep learning main results...")

if deep_official is not None:
    table_5_2 = deep_official[[
        'model_id', 'name', 'ch', 'n_runs',
        'acc_a_mean', 'acc_a_std', 'f1_a_mean', 'f1_a_std',
        'acc_b_mean', 'acc_b_std', 'f1_b_mean', 'f1_b_std'
    ]].copy()
    
    save_table(table_5_2, 'table_5_2_deep_main')
else:
    log_audit("Table 5.2 skipped (no deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 31
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 6.3 — Table 5.3: DANCE LOSO Verified

log_audit("Generating Table 5.3: DANCE LOSO verified...")

if dance_loso_official is not None:
    save_table(dance_loso_official, 'table_5_3_dance_loso_verified')
else:
    log_audit("Table 5.3 skipped (no DANCE LOSO data)", 'WARNING')


# ==============================================================================
# Notebook cell 32
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 6.4 — Table 5.3b: DANCE Reproduction

log_audit("Generating Table 5.3b: DANCE reproduction...")

if dance_repro_official is not None:
    save_table(dance_repro_official, 'table_5_3b_dance_reproduction')
else:
    log_audit("Table 5.3b skipped (no reproduction data)", 'WARNING')


# ==============================================================================
# Notebook cell 33
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 6.5 — Table 5.4: Ablations

log_audit("Generating Table 5.4: Ablation results...")

if ablations_df is not None:
    table_5_4 = ablations_df.copy()
    save_table(table_5_4, 'table_5_4_ablations')
else:
    log_audit("Table 5.4 skipped (no ablation data)", 'WARNING')


# ==============================================================================
# Notebook cell 34
# Categories: preprocessing, evaluation, results_tables, audit_verification
# ==============================================================================
# CELL 6.6 — Table 5.5: Proto-A vs Proto-B Gain

log_audit("Generating Table 5.5: Proto gain summary...")

if deep_official is not None:
    proto_gain_rows = []
    
    for _, row in deep_official.iterrows():
        if pd.notna(row['acc_a_mean']) and pd.notna(row['acc_b_mean']):
            proto_gain_rows.append({
                'model_id': row['model_id'],
                'name': row['name'],
                'ch': row['ch'],
                'delta_acc': row['acc_b_mean'] - row['acc_a_mean'],
                'delta_f1': row['f1_b_mean'] - row['f1_a_mean'] if pd.notna(row['f1_a_mean']) else np.nan
            })
    
    # Add DANCE if available
    if dance_loso_official is not None:
        for _, row in dance_loso_official.iterrows():
            if pd.notna(row['acc_a_mean']) and pd.notna(row['acc_b_mean']):
                proto_gain_rows.append({
                    'model_id': row['model_id'],
                    'name': row['name'],
                    'ch': row['ch'],
                    'delta_acc': row['acc_b_mean'] - row['acc_a_mean'],
                    'delta_f1': row['f1_b_mean'] - row['f1_a_mean'] if pd.notna(row['f1_a_mean']) else np.nan
                })
    
    if proto_gain_rows:
        table_5_5 = pd.DataFrame(proto_gain_rows)
        save_table(table_5_5, 'table_5_5_proto_gain')
else:
    log_audit("Table 5.5 skipped (no deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 35
# Categories: preprocessing, model_definition, results_tables, audit_verification
# ==============================================================================
# CELL 6.7 — Table 5.6: Channel Efficiency

log_audit("Generating Table 5.6: Channel efficiency...")

channel_eff_rows = []

# For classical ML (F1_mean as primary metric)
if classical_official is not None:
    for model_id in classical_official['model_id'].unique():
        model_data = classical_official[classical_official['model_id'] == model_id]
        
        ch_62 = model_data[model_data['ch'] == '62ch']
        ch_6 = model_data[model_data['ch'] == '6ch']
        
        if len(ch_62) > 0 and len(ch_6) > 0:
            f1_62 = ch_62.iloc[0]['F1_mean']
            f1_6 = ch_6.iloc[0]['F1_mean']
            
            channel_eff_rows.append({
                'model_id': model_id,
                'name': get_model_name(model_id),
                'primary_62': f1_62,
                'primary_6': f1_6,
                'retention_primary': f1_6 / f1_62 if f1_62 > 0 else np.nan,
                'family': 'classical'
            })

# For deep learning (acc_b_mean as primary metric)
if deep_official is not None:
    for model_id in deep_official['model_id'].unique():
        model_data = deep_official[deep_official['model_id'] == model_id]
        
        ch_62 = model_data[model_data['ch'] == '62ch']
        ch_6 = model_data[model_data['ch'] == '6ch']
        
        if len(ch_62) > 0 and len(ch_6) > 0:
            acc_62 = ch_62.iloc[0]['acc_b_mean']
            acc_6 = ch_6.iloc[0]['acc_b_mean']
            
            channel_eff_rows.append({
                'model_id': model_id,
                'name': get_model_name(model_id),
                'primary_62': acc_62,
                'primary_6': acc_6,
                'retention_primary': acc_6 / acc_62 if acc_62 > 0 else np.nan,
                'family': 'deep'
            })

# DANCE Teacher->Student paired
if dance_loso_official is not None:
    m25 = dance_loso_official[dance_loso_official['model_id'] == 'M25']
    m26 = dance_loso_official[dance_loso_official['model_id'] == 'M26']
    
    if len(m25) > 0 and len(m26) > 0:
        teacher_acc = m25.iloc[0]['acc_b_mean']
        student_acc = m26.iloc[0]['acc_b_mean']
        
        channel_eff_rows.append({
            'model_id': 'M25->M26',
            'name': 'DANCE Teacher->Student',
            'primary_62': teacher_acc,
            'primary_6': student_acc,
            'retention_primary': student_acc / teacher_acc if teacher_acc > 0 else np.nan,
            'family': 'dance',
            'paired_distillation': True
        })

if channel_eff_rows:
    table_5_6 = pd.DataFrame(channel_eff_rows)
    save_table(table_5_6, 'table_5_6_channel_efficiency')
else:
    log_audit("Table 5.6 skipped (insufficient paired data)", 'WARNING')


# ==============================================================================
# Notebook cell 36
# Categories: results_tables, audit_verification
# ==============================================================================
# CELL 6.8 — Table 5.7: Per-Class Top Models

log_audit("Generating Table 5.7: Per-class top models...")

per_class_rows = []

# Classical per-class (if available in classical_summary)
if classical_official is not None:
    per_class_cols = ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']
    available_per_class = [c for c in per_class_cols if c in classical_official.columns]
    
    if available_per_class:
        for _, row in classical_official.iterrows():
            entry = {
                'model_id': row['model_id'],
                'name': row['name'],
                'ch': row['ch'],
                'source': 'classical_summary'
            }
            for col in available_per_class:
                entry[col.lower()] = row[col]
            per_class_rows.append(entry)

# Note: Deep/DANCE per-class requires separate source files
# If not found, log limitation
if not per_class_rows or len(available_per_class) < 4:
    log_audit("Per-class data incomplete, using available classical subset", 'WARNING')

if per_class_rows:
    table_5_7 = pd.DataFrame(per_class_rows)
    save_table(table_5_7, 'table_5_7_per_class_top_models')
else:
    log_audit("Table 5.7 skipped (no per-class data)", 'WARNING')


# ==============================================================================
# Notebook cell 37
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 6.9 — Table Audit Sources

log_audit("Generating audit sources table...")

audit_sources = [
    {'output': 'table_5_1', 'source': 'classical_ml_summary.csv', 'role': 'primary', 'status': source_status.get('classical_summary', False)},
    {'output': 'table_5_2', 'source': 'M11-M24 per-model CSVs + master', 'role': 'primary+audit', 'status': deep_recomputed is not None},
    {'output': 'table_5_3', 'source': 'M25_62ch + M26_6ch summaries', 'role': 'primary', 'status': dance_loso_official is not None},
    {'output': 'table_5_3b', 'source': 'phaseB_reproduce_results.csv', 'role': 'primary', 'status': source_status.get('dance_reproduction', False)},
    {'output': 'table_5_4', 'source': 'ablations_partial_summary.csv', 'role': 'primary', 'status': source_status.get('ablations', False)},
    {'output': 'table_5_5', 'source': 'derived from table_5_2 + table_5_3', 'role': 'derived', 'status': True},
    {'output': 'table_5_6', 'source': 'derived from paired 62ch/6ch', 'role': 'derived', 'status': True},
    {'output': 'table_5_7', 'source': 'classical_summary + optional deep', 'role': 'partial', 'status': True}
]

table_audit = pd.DataFrame(audit_sources)
save_table(table_audit, 'table_audit_sources')


# ==============================================================================
# Notebook cell 39
# Categories: preprocessing, evaluation, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.1 — Figure 5.1: Classical Leaderboard

log_audit("Generating Figure 5.1: Classical leaderboard...")

if classical_official is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 62ch panel
    data_62 = classical_official[classical_official['ch'] == '62ch'].sort_values('F1_mean', ascending=True)
    if len(data_62) > 0:
        ax1.barh(range(len(data_62)), data_62['F1_mean'], 
                xerr=data_62['F1_std'], color=COLORS['classical'], alpha=0.8)
        ax1.set_yticks(range(len(data_62)))
        ax1.set_yticklabels(data_62['name'])
        ax1.set_xlabel('Macro-F1')
        ax1.set_title('Classical ML - 62 channels')
        ax1.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
        ax1.legend()
    
    # 6ch panel
    data_6 = classical_official[classical_official['ch'] == '6ch'].sort_values('F1_mean', ascending=True)
    if len(data_6) > 0:
        ax2.barh(range(len(data_6)), data_6['F1_mean'], 
                xerr=data_6['F1_std'], color=COLORS['classical'], alpha=0.8)
        ax2.set_yticks(range(len(data_6)))
        ax2.set_yticklabels(data_6['name'])
        ax2.set_xlabel('Macro-F1')
        ax2.set_title('Classical ML - 6 channels')
        ax2.axvline(0.25, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_1_classical_leaderboard')
    plt.close()
else:
    log_audit("Figure 5.1 skipped (no classical data)", 'WARNING')


# ==============================================================================
# Notebook cell 40
# Categories: preprocessing, evaluation, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.2 — Figure 5.2: Deep Baseline Leaderboard (62ch)

log_audit("Generating Figure 5.2: Deep baseline 62ch...")

if deep_official is not None:
    data_62 = deep_official[deep_official['ch'] == '62ch'].sort_values('acc_b_mean', ascending=True)
    
    if len(data_62) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(data_62)), data_62['acc_b_mean'],
               xerr=data_62['acc_b_std'], color=COLORS['deep'], alpha=0.8)
        ax.set_yticks(range(len(data_62)))
        ax.set_yticklabels(data_62['name'])
        ax.set_xlabel('AccB (Proto-B Balanced Accuracy)')
        ax.set_title('Deep Learning Baselines - 62 channels')
        ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        
        plt.tight_layout()
        save_figure(fig, 'fig_5_2_deep_leaderboard_62ch')
        plt.close()
    else:
        log_audit("No 62ch deep data for Figure 5.2", 'WARNING')
else:
    log_audit("Figure 5.2 skipped (no deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 41
# Categories: preprocessing, evaluation, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.3 — Figure 5.3: Deep Baseline Leaderboard (6ch)

log_audit("Generating Figure 5.3: Deep baseline 6ch...")

if deep_official is not None:
    data_6 = deep_official[deep_official['ch'] == '6ch'].sort_values('acc_b_mean', ascending=True)
    
    if len(data_6) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(data_6)), data_6['acc_b_mean'],
               xerr=data_6['acc_b_std'], color=COLORS['deep'], alpha=0.8)
        ax.set_yticks(range(len(data_6)))
        ax.set_yticklabels(data_6['name'])
        ax.set_xlabel('AccB (Proto-B Balanced Accuracy)')
        ax.set_title('Deep Learning Baselines - 6 channels')
        ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        
        plt.tight_layout()
        save_figure(fig, 'fig_5_3_deep_leaderboard_6ch')
        plt.close()
    else:
        log_audit("No 6ch deep data for Figure 5.3", 'WARNING')
else:
    log_audit("Figure 5.3 skipped (no deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 42
# Categories: preprocessing, evaluation, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.4 — Figure 5.4: Proto-A vs Proto-B

log_audit("Generating Figure 5.4: Proto-A vs Proto-B...")

if deep_official is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all deep models + DANCE
    for _, row in deep_official.iterrows():
        if pd.notna(row['acc_a_mean']) and pd.notna(row['acc_b_mean']):
            color = COLORS['deep'] if row['ch'] == '62ch' else COLORS['baseline']
            ax.scatter(row['acc_a_mean'], row['acc_b_mean'], 
                      s=100, alpha=0.7, color=color, label=f"{row['name']} ({row['ch']})")
    
    if dance_loso_official is not None:
        for _, row in dance_loso_official.iterrows():
            if pd.notna(row['acc_a_mean']) and pd.notna(row['acc_b_mean']):
                ax.scatter(row['acc_a_mean'], row['acc_b_mean'],
                          s=200, alpha=0.9, color=COLORS['dance'], 
                          marker='*', label=row['name'])
    
    # Diagonal line (Proto-A = Proto-B)
    lims = [0.2, 0.7]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Proto-A = Proto-B')
    
    ax.set_xlabel('AccA (Proto-A)')
    ax.set_ylabel('AccB (Proto-B)')
    ax.set_title('Proto-A vs Proto-B Calibration Gain')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_4_protoA_vs_protoB')
    plt.close()
else:
    log_audit("Figure 5.4 skipped (no deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 43
# Categories: preprocessing, model_definition, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.5 — Figure 5.5: DANCE Headline

log_audit("Generating Figure 5.5: DANCE headline...")

if dance_loso_official is not None and deep_official is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: M25 vs top 62ch baselines
    top_62 = deep_official[deep_official['ch'] == '62ch'].nlargest(5, 'acc_b_mean')
    m25 = dance_loso_official[dance_loso_official['model_id'] == 'M25']
    
    if len(m25) > 0:
        combined_62 = pd.concat([top_62, m25])
        combined_62 = combined_62.sort_values('acc_b_mean', ascending=True)
        
        colors_62 = [COLORS['dance'] if x == 'M25' else COLORS['deep'] 
                     for x in combined_62['model_id']]
        
        ax1.barh(range(len(combined_62)), combined_62['acc_b_mean'],
                xerr=combined_62['acc_b_std'], color=colors_62, alpha=0.8)
        ax1.set_yticks(range(len(combined_62)))
        ax1.set_yticklabels(combined_62['name'])
        ax1.set_xlabel('AccB')
        ax1.set_title('DANCE Teacher vs Top 62ch Baselines')
    
    # Panel B: M26 vs top 6ch baselines
    top_6 = deep_official[deep_official['ch'] == '6ch'].nlargest(5, 'acc_b_mean')
    m26 = dance_loso_official[dance_loso_official['model_id'] == 'M26']
    
    if len(m26) > 0:
        combined_6 = pd.concat([top_6, m26])
        combined_6 = combined_6.sort_values('acc_b_mean', ascending=True)
        
        colors_6 = [COLORS['dance'] if x == 'M26' else COLORS['baseline'] 
                    for x in combined_6['model_id']]
        
        ax2.barh(range(len(combined_6)), combined_6['acc_b_mean'],
                xerr=combined_6['acc_b_std'], color=colors_6, alpha=0.8)
        ax2.set_yticks(range(len(combined_6)))
        ax2.set_yticklabels(combined_6['name'])
        ax2.set_xlabel('AccB')
        ax2.set_title('DANCE Student vs Top 6ch Baselines')
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_5_dance_headline')
    plt.close()
else:
    log_audit("Figure 5.5 skipped (missing DANCE or deep data)", 'WARNING')


# ==============================================================================
# Notebook cell 44
# Categories: evaluation, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.6 — Figure 5.7: Ablation Absolute

log_audit("Generating Figure 5.7: Ablation absolute...")

if ablations_df is not None:
    abl = ablations_df.sort_values('acc_B_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color A01 baseline differently
    colors = [COLORS['dance'] if 'A01' in str(x) or 'Full' in str(x) 
              else COLORS['baseline'] for x in abl['ablation']]
    
    ax.barh(range(len(abl)), abl['acc_B_mean'],
           xerr=abl['acc_B_std'] if 'acc_B_std' in abl.columns else None,
           color=colors, alpha=0.8)
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels(abl['ablation'])
    ax.set_xlabel('AccB (Proto-B)')
    ax.set_title('DANCE Ablation Study - Absolute Performance')
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_7_ablation_abs')
    plt.close()
else:
    log_audit("Figure 5.7 skipped (no ablation data)", 'WARNING')


# ==============================================================================
# Notebook cell 45
# Categories: results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.7 — Figure 5.8: Ablation Delta

log_audit("Generating Figure 5.8: Ablation delta...")

if ablations_df is not None:
    # Find baseline A01
    baseline_row = ablations_df[ablations_df['ablation'].str.contains('A01|Full', case=False, na=False)]
    
    if len(baseline_row) > 0:
        baseline_acc = baseline_row.iloc[0]['acc_B_mean']
        
        abl = ablations_df.copy()
        abl['delta'] = abl['acc_B_mean'] - baseline_acc
        abl = abl.sort_values('delta', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['red' if x < 0 else 'green' for x in abl['delta']]
        
        ax.barh(range(len(abl)), abl['delta'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(abl)))
        ax.set_yticklabels(abl['ablation'])
        ax.set_xlabel('Δ AccB vs Full DANCE')
        ax.set_title('DANCE Ablation Study - Delta from Baseline')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        save_figure(fig, 'fig_5_8_ablation_delta')
        plt.close()
    else:
        log_audit("Could not find baseline A01 for delta calculation", 'WARNING')
else:
    log_audit("Figure 5.8 skipped (no ablation data)", 'WARNING')


# ==============================================================================
# Notebook cell 46
# Categories: preprocessing, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.8 — Figure 5.10: Channel Efficiency Scatter

log_audit("Generating Figure 5.10: Channel efficiency scatter...")

if channel_eff_rows:
    eff_df = pd.DataFrame(channel_eff_rows)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by family
    for family, color in [('classical', COLORS['classical']), 
                          ('deep', COLORS['deep']),
                          ('dance', COLORS['dance'])]:
        subset = eff_df[eff_df['family'] == family]
        if len(subset) > 0:
            marker = '*' if family == 'dance' else 'o'
            size = 200 if family == 'dance' else 80
            ax.scatter(subset['primary_62'], subset['primary_6'],
                      s=size, alpha=0.7, color=color, marker=marker,
                      label=family.capitalize())
    
    # Diagonal
    lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='62ch = 6ch')
    
    ax.set_xlabel('Primary Metric (62ch)')
    ax.set_ylabel('Primary Metric (6ch)')
    ax.set_title('Channel Efficiency: 62ch vs 6ch Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_10_channel_efficiency_scatter')
    plt.close()
else:
    log_audit("Figure 5.10 skipped (no channel efficiency data)", 'WARNING')


# ==============================================================================
# Notebook cell 47
# Categories: preprocessing, results_tables, figures, audit_verification
# ==============================================================================
# CELL 7.9 — Figure 5.11: Channel Efficiency Retention

log_audit("Generating Figure 5.11: Channel retention...")

if channel_eff_rows:
    eff_df = pd.DataFrame(channel_eff_rows)
    eff_df = eff_df.sort_values('retention_primary', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLORS.get(f, COLORS['baseline']) for f in eff_df['family']]
    
    ax.barh(range(len(eff_df)), eff_df['retention_primary'], 
           color=colors, alpha=0.8)
    ax.set_yticks(range(len(eff_df)))
    ax.set_yticklabels(eff_df['name'])
    ax.set_xlabel('Retention Ratio (6ch / 62ch)')
    ax.set_title('Channel Efficiency: Performance Retention')
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Parity')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'fig_5_11_channel_efficiency_retention')
    plt.close()
else:
    log_audit("Figure 5.11 skipped (no channel efficiency data)", 'WARNING')


# ==============================================================================
# Notebook cell 50
# Categories: preprocessing, training, results_tables, figures, audit_verification
# ==============================================================================
# CELL 8.1 — Search for optional visualization assets

log_audit("Searching for optional visualization assets...")

# Search paths
search_dirs = [
    ROOT / 'results',
    ROOT / 'checkpoints',
    ROOT / 'features'
]

discovered_assets = []

# Asset patterns to search for
patterns = [
    ('tsne', ['*tsne*.npy', '*tsne*.csv', '*embedding*.npy']),
    ('confusion', ['*confusion*.npy', '*confusion*.csv']),
    ('per_class', ['*per_class*.csv', '*class_metrics*.csv'])
]

for search_dir in search_dirs:
    if search_dir.exists():
        for asset_type, file_patterns in patterns:
            for pattern in file_patterns:
                matches = list(search_dir.rglob(pattern))
                for match in matches:
                    discovered_assets.append({
                        'type': asset_type,
                        'path': str(match.relative_to(ROOT)),
                        'size_kb': match.stat().st_size / 1024
                    })
                    log_audit(f"  Found {asset_type}: {match.name}")

if discovered_assets:
    assets_df = pd.DataFrame(discovered_assets)
    save_table(assets_df, 'audit_discovered_optional_assets')
    log_audit(f"✓ Discovered {len(discovered_assets)} optional assets")
else:
    log_audit("No optional visualization assets found", 'WARNING')


# ==============================================================================
# Notebook cell 52
# Categories: audit_verification
# ==============================================================================
# CELL 9.1 — Generate audit report text

log_audit("Generating final audit report...")

audit_text = []
audit_text.append("=" * 80)
audit_text.append("CSE400C FINAL RESULTS REBUILD - AUDIT REPORT")
audit_text.append("=" * 80)
audit_text.append(f"Generated: {datetime.now().isoformat()}")
audit_text.append(f"Project Root: {ROOT}")
audit_text.append("")

audit_text.append("SOURCE FILE STATUS")
audit_text.append("-" * 40)
for name, status in source_status.items():
    status_str = "✓ FOUND" if status else "✗ MISSING"
    audit_text.append(f"{status_str:12} {name}")
audit_text.append("")

audit_text.append("PER-MODEL DEEP LEARNING FILES")
audit_text.append("-" * 40)
loaded = sum(1 for v in deep_model_status.values() if v)
total = len(deep_model_status)
audit_text.append(f"Loaded: {loaded}/{total}")

missing_models = [k for k, v in deep_model_status.items() if not v]
if missing_models:
    if len(missing_models) > 10:
        audit_text.append(f"Missing: {', '.join(missing_models[:10])}...")
    else:
        audit_text.append(f"Missing: {', '.join(missing_models)}")
audit_text.append("")

if audit_warnings:
    audit_text.append("WARNINGS")
    audit_text.append("-" * 40)
    for w in audit_warnings[:20]:
        audit_text.append(f"⚠ {w}")
    audit_text.append("")

if audit_errors:
    audit_text.append("ERRORS")
    audit_text.append("-" * 40)
    for e in audit_errors[:20]:
        audit_text.append(f"❌ {e}")
    audit_text.append("")

audit_text.append("FULL LOG")
audit_text.append("-" * 40)
audit_text.extend(audit_log)

# Save audit text as UTF-8
audit_txt_path = OUTPUT_DIRS['audit'] / 'results_audit.txt'
with open(audit_txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(audit_text))

log_audit(f"✓ Audit text saved: {audit_txt_path.relative_to(ROOT)}")


# ==============================================================================
# Notebook cell 53
# Categories: preprocessing, results_tables, audit_verification
# ==============================================================================
# CELL 9.2 — Generate audit CSV

audit_csv_rows = []

for entry in audit_log:
    # Parse entry format: [HH:MM:SS] LEVEL: message
    parts = entry.split('] ', 1)
    if len(parts) == 2:
        timestamp = parts[0].replace('[', '')
        rest = parts[1].split(': ', 1)
        if len(rest) == 2:
            level, message = rest
        else:
            level = 'INFO'
            message = rest[0]
    else:
        timestamp = ''
        level = 'INFO'
        message = entry
    
    audit_csv_rows.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })

audit_csv_df = pd.DataFrame(audit_csv_rows)
audit_csv_path = OUTPUT_DIRS['audit'] / 'results_audit.csv'
audit_csv_df.to_csv(audit_csv_path, index=False)

log_audit(f"✓ Audit CSV saved: {audit_csv_path.relative_to(ROOT)}")


# ==============================================================================
# Notebook cell 55
# Categories: results_tables, figures, audit_verification
# ==============================================================================
# CELL 10.1 — Display final summary

print("\n" + "=" * 80)
print("NOTEBOOK EXECUTION COMPLETE")
print("=" * 80)

# Count generated outputs
tables_dir = OUTPUT_DIRS['tables']
png_dir = OUTPUT_DIRS['figures_png']
pdf_dir = OUTPUT_DIRS['figures_pdf']

n_tables = len(list(tables_dir.glob('*.csv'))) if tables_dir.exists() else 0
n_png = len(list(png_dir.glob('*.png'))) if png_dir.exists() else 0
n_pdf = len(list(pdf_dir.glob('*.pdf'))) if pdf_dir.exists() else 0

print(f"\n📊 Generated {n_tables} tables")
if n_tables > 0:
    for table_file in sorted(tables_dir.glob('*.csv')):
        print(f"   - {table_file.name}")

print(f"\n📈 Generated {n_png} figures (PNG+PDF)")
if n_png > 0:
    for fig_file in sorted(png_dir.glob('*.png')):
        print(f"   - {fig_file.stem}")

print(f"\n⚠️  Total warnings: {len(audit_warnings)}")
if audit_warnings:
    for w in audit_warnings[:5]:
        print(f"   - {w}")
    if len(audit_warnings) > 5:
        print(f"   ... and {len(audit_warnings) - 5} more (see audit report)")

print(f"\n❌ Total errors: {len(audit_errors)}")
if audit_errors:
    for e in audit_errors:
        print(f"   - {e}")

print(f"\n📁 All outputs saved to: {ROOT / 'results'}")
print(f"📋 Audit report: {OUTPUT_DIRS['audit'] / 'results_audit.txt'}")

if len(audit_errors) == 0 and len(audit_warnings) <= 5:
    print("\n✅ DONE - Notebook completed successfully")
elif len(audit_errors) == 0:
    print("\n⚠️  DONE - Notebook completed with warnings (check audit report)")
else:
    print("\n❌ DONE - Notebook completed with errors (check audit report)")

print("=" * 80)


# ==============================================================================
# Notebook cell 56
# Categories: preprocessing, model_definition, training, evaluation, results_tables, figures, statistics, audit_verification
# ==============================================================================
# # CELL X — MASTER REGEN: all tables + all figures + supplementary figures (separate files)

# from pathlib import Path
# from IPython.display import display, Markdown
# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ------------------------------------------------------------------------------
# # 0) Safety / fallback helpers
# # ------------------------------------------------------------------------------

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 200)
# pd.set_option('display.max_colwidth', 120)

# if 'ROOT' not in globals():
#     ROOT = Path.cwd()

# if 'OUTPUT_DIRS' not in globals():
#     OUTPUT_DIRS = {
#         'tables': ROOT / 'results' / 'tables',
#         'figures_png': ROOT / 'results' / 'figures_png',
#         'figures_pdf': ROOT / 'results' / 'figures_pdf',
#         'audit': ROOT / 'results' / 'final_audit',
#     }
#     for p in OUTPUT_DIRS.values():
#         p.mkdir(parents=True, exist_ok=True)

# if 'log_audit' not in globals():
#     audit_log = []
#     audit_warnings = []
#     audit_errors = []

#     def log_audit(message, level='INFO'):
#         timestamp = datetime.now().strftime('%H:%M:%S')
#         entry = f"[{timestamp}] {level}: {message}"
#         audit_log.append(entry)
#         if level == 'WARNING':
#             audit_warnings.append(message)
#             print(f"⚠ {message}")
#         elif level == 'ERROR':
#             audit_errors.append(message)
#             print(f"❌ {message}")
#         else:
#             print(f"  {message}")

# if 'save_figure' not in globals():
#     def save_figure(fig, name, dpi=300):
#         png_path = OUTPUT_DIRS['figures_png'] / f"{name}.png"
#         pdf_path = OUTPUT_DIRS['figures_pdf'] / f"{name}.pdf"
#         fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
#         fig.savefig(pdf_path, bbox_inches='tight')
#         log_audit(f"Saved figure: {name}")
#         return png_path, pdf_path

# if 'save_table' not in globals():
#     def save_table(df, name):
#         csv_path = OUTPUT_DIRS['tables'] / f"{name}.csv"
#         df.to_csv(csv_path, index=False)
#         log_audit(f"Saved table: {name} ({df.shape[0]} rows)")
#         return csv_path

# if 'get_model_name' not in globals():
#     MODEL_NAMES = {
#         'M01': 'LDA', 'M02': 'SVM (RBF)', 'M03': 'Random Forest', 'M04': 'k-NN',
#         'M05': 'Logistic Regression', 'M06': 'Naive Bayes', 'M07': 'Extra Trees',
#         'M08': 'Gradient Boosting', 'M09': 'XGBoost', 'M10': 'MLP (sklearn)',
#         'M11': 'Shallow MLP', 'M12': 'Deep MLP', 'M13': 'LSTM', 'M14': 'GRU',
#         'M15': 'Conv1D', 'M16': 'Vanilla Transformer', 'M17': 'EEG Conformer',
#         'M18': 'ChanDrop Transformer', 'M19': 'DANN', 'M20': 'CLISA',
#         'M21': 'SimCLR', 'M22': 'BYOL', 'M23': 'PseudoLabel', 'M24': 'MixMatch',
#         'M25': 'DANCE Teacher', 'M26': 'DANCE Student'
#     }

#     def get_model_name(model_id):
#         return MODEL_NAMES.get(str(model_id), str(model_id))

# if 'COLORS' not in globals():
#     COLORS = {
#         'classical': '#4C78A8',
#         'deep': '#F58518',
#         'dance': '#54A24B',
#         'baseline': '#9D755D',
#         'neutral': '#4C78A8',
#         'sad': '#72B7B2',
#         'fear': '#E45756',
#         'happy': '#54A24B'
#     }

# def _slug(text):
#     return re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')

# def _record_table(df, name, generated_tables):
#     save_table(df, name)
#     generated_tables[name] = df
#     print("\n" + "=" * 100)
#     print(f"TABLE: {name} | shape={df.shape}")
#     print("Columns:", list(df.columns))
#     display(df)

# def _record_fig(fig, name, generated_figs):
#     save_figure(fig, name)
#     generated_figs.append(name)
#     plt.close(fig)

# log_audit("MASTER CELL: generating all tables + all figures + supplementary figures (separate plots)...")

# generated_tables = {}
# generated_figs = []

# classical_official = globals().get('classical_official', None)
# deep_official = globals().get('deep_official', None)
# dance_loso_official = globals().get('dance_loso_official', None)
# dance_repro_official = globals().get('dance_repro_official', None)
# ablations_df = globals().get('ablations_df', None)
# deep_per_model = globals().get('deep_per_model', {})
# dance_m25_df = globals().get('dance_m25_df', None)
# source_status = globals().get('source_status', {})
# deep_recomputed = globals().get('deep_recomputed', None)
# SCIPY_AVAILABLE = globals().get('SCIPY_AVAILABLE', False)

# # ------------------------------------------------------------------------------
# # 1) TABLES
# # ------------------------------------------------------------------------------

# log_audit("Regenerating all tables and printing table info...")

# # Table 5.1
# if classical_official is not None:
#     table_5_1 = classical_official[
#         ['model_id', 'name', 'ch', 'F1_mean', 'F1_std', 'Acc_mean', 'Acc_std']
#     ].copy()
#     for col in ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']:
#         if col in classical_official.columns:
#             table_5_1[col.lower()] = classical_official[col]
#     if 'N_runs' in classical_official.columns:
#         table_5_1['n_runs'] = classical_official['N_runs']
#     _record_table(table_5_1, 'table_5_1_classical_main', generated_tables)
# else:
#     log_audit("Table 5.1 skipped (no classical data)", 'WARNING')

# # Table 5.2
# if deep_official is not None:
#     needed = [
#         'model_id', 'name', 'ch', 'n_runs',
#         'acc_a_mean', 'acc_a_std', 'f1_a_mean', 'f1_a_std',
#         'acc_b_mean', 'acc_b_std', 'f1_b_mean', 'f1_b_std'
#     ]
#     cols = [c for c in needed if c in deep_official.columns]
#     table_5_2 = deep_official[cols].copy()
#     _record_table(table_5_2, 'table_5_2_deep_main', generated_tables)
# else:
#     log_audit("Table 5.2 skipped (no deep data)", 'WARNING')

# # Table 5.3
# if dance_loso_official is not None:
#     _record_table(dance_loso_official.copy(), 'table_5_3_dance_loso_verified', generated_tables)
# else:
#     log_audit("Table 5.3 skipped (no DANCE LOSO data)", 'WARNING')

# # Table 5.3b
# if dance_repro_official is not None:
#     _record_table(dance_repro_official.copy(), 'table_5_3b_dance_reproduction', generated_tables)
# else:
#     log_audit("Table 5.3b skipped (no DANCE reproduction data)", 'WARNING')

# # Table 5.4
# if ablations_df is not None:
#     _record_table(ablations_df.copy(), 'table_5_4_ablations', generated_tables)
# else:
#     log_audit("Table 5.4 skipped (no ablation data)", 'WARNING')

# # Table 5.5
# proto_gain_rows = []
# if deep_official is not None:
#     for _, row in deep_official.iterrows():
#         if pd.notna(row.get('acc_a_mean', np.nan)) and pd.notna(row.get('acc_b_mean', np.nan)):
#             proto_gain_rows.append({
#                 'model_id': row['model_id'],
#                 'name': row['name'],
#                 'ch': row['ch'],
#                 'delta_acc': row['acc_b_mean'] - row['acc_a_mean'],
#                 'delta_f1': (
#                     row['f1_b_mean'] - row['f1_a_mean']
#                     if pd.notna(row.get('f1_a_mean', np.nan)) and pd.notna(row.get('f1_b_mean', np.nan))
#                     else np.nan
#                 )
#             })

# if dance_loso_official is not None:
#     for _, row in dance_loso_official.iterrows():
#         if pd.notna(row.get('acc_a_mean', np.nan)) and pd.notna(row.get('acc_b_mean', np.nan)):
#             proto_gain_rows.append({
#                 'model_id': row['model_id'],
#                 'name': row['name'],
#                 'ch': row['ch'],
#                 'delta_acc': row['acc_b_mean'] - row['acc_a_mean'],
#                 'delta_f1': (
#                     row['f1_b_mean'] - row['f1_a_mean']
#                     if pd.notna(row.get('f1_a_mean', np.nan)) and pd.notna(row.get('f1_b_mean', np.nan))
#                     else np.nan
#                 )
#             })

# if proto_gain_rows:
#     table_5_5 = pd.DataFrame(proto_gain_rows)
#     _record_table(table_5_5, 'table_5_5_proto_gain', generated_tables)
# else:
#     log_audit("Table 5.5 skipped (no proto-gain rows)", 'WARNING')

# # Table 5.6
# channel_eff_rows = []

# if classical_official is not None:
#     for model_id in classical_official['model_id'].dropna().unique():
#         model_data = classical_official[classical_official['model_id'] == model_id]
#         ch_62 = model_data[model_data['ch'] == '62ch']
#         ch_6 = model_data[model_data['ch'] == '6ch']
#         if len(ch_62) > 0 and len(ch_6) > 0:
#             f1_62 = ch_62.iloc[0]['F1_mean']
#             f1_6 = ch_6.iloc[0]['F1_mean']
#             channel_eff_rows.append({
#                 'model_id': model_id,
#                 'name': get_model_name(model_id),
#                 'primary_62': f1_62,
#                 'primary_6': f1_6,
#                 'retention_primary': f1_6 / f1_62 if pd.notna(f1_62) and f1_62 > 0 else np.nan,
#                 'family': 'classical'
#             })

# if deep_official is not None:
#     for model_id in deep_official['model_id'].dropna().unique():
#         model_data = deep_official[deep_official['model_id'] == model_id]
#         ch_62 = model_data[model_data['ch'] == '62ch']
#         ch_6 = model_data[model_data['ch'] == '6ch']
#         if len(ch_62) > 0 and len(ch_6) > 0:
#             acc_62 = ch_62.iloc[0]['acc_b_mean']
#             acc_6 = ch_6.iloc[0]['acc_b_mean']
#             channel_eff_rows.append({
#                 'model_id': model_id,
#                 'name': get_model_name(model_id),
#                 'primary_62': acc_62,
#                 'primary_6': acc_6,
#                 'retention_primary': acc_6 / acc_62 if pd.notna(acc_62) and acc_62 > 0 else np.nan,
#                 'family': 'deep'
#             })

# if dance_loso_official is not None:
#     m25 = dance_loso_official[dance_loso_official['model_id'] == 'M25']
#     m26 = dance_loso_official[dance_loso_official['model_id'] == 'M26']
#     if len(m25) > 0 and len(m26) > 0:
#         teacher_acc = m25.iloc[0]['acc_b_mean']
#         student_acc = m26.iloc[0]['acc_b_mean']
#         channel_eff_rows.append({
#             'model_id': 'M25->M26',
#             'name': 'DANCE Teacher->Student',
#             'primary_62': teacher_acc,
#             'primary_6': student_acc,
#             'retention_primary': student_acc / teacher_acc if pd.notna(teacher_acc) and teacher_acc > 0 else np.nan,
#             'family': 'dance',
#             'paired_distillation': True
#         })

# if channel_eff_rows:
#     table_5_6 = pd.DataFrame(channel_eff_rows)
#     globals()['channel_eff_rows'] = channel_eff_rows
#     _record_table(table_5_6, 'table_5_6_channel_efficiency', generated_tables)
# else:
#     log_audit("Table 5.6 skipped (no channel efficiency rows)", 'WARNING')

# # Table 5.7
# per_class_rows = []
# available_per_class = []

# if classical_official is not None:
#     per_class_cols = ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']
#     available_per_class = [c for c in per_class_cols if c in classical_official.columns]
#     if available_per_class:
#         for _, row in classical_official.iterrows():
#             entry = {
#                 'model_id': row['model_id'],
#                 'name': row['name'],
#                 'ch': row['ch'],
#                 'source': 'classical_summary'
#             }
#             for col in available_per_class:
#                 entry[col.lower()] = row[col]
#             per_class_rows.append(entry)

# if not per_class_rows or len(available_per_class) < 4:
#     log_audit("Per-class data incomplete, using available classical subset", 'WARNING')

# if per_class_rows:
#     table_5_7 = pd.DataFrame(per_class_rows)
#     _record_table(table_5_7, 'table_5_7_per_class_top_models', generated_tables)
# else:
#     log_audit("Table 5.7 skipped (no per-class rows)", 'WARNING')

# # Audit source table
# table_audit = pd.DataFrame([
#     {'output': 'table_5_1', 'source': 'classical_ml_summary.csv', 'role': 'primary', 'status': source_status.get('classical_summary', False)},
#     {'output': 'table_5_2', 'source': 'M11-M24 per-model CSVs + master', 'role': 'primary+audit', 'status': deep_recomputed is not None},
#     {'output': 'table_5_3', 'source': 'M25_62ch + M26_6ch summaries', 'role': 'primary', 'status': dance_loso_official is not None},
#     {'output': 'table_5_3b', 'source': 'phaseB_reproduce_results.csv', 'role': 'primary', 'status': source_status.get('dance_reproduction', False)},
#     {'output': 'table_5_4', 'source': 'ablations_partial_summary.csv', 'role': 'primary', 'status': source_status.get('ablations', False)},
#     {'output': 'table_5_5', 'source': 'derived from table_5_2 + table_5_3', 'role': 'derived', 'status': True},
#     {'output': 'table_5_6', 'source': 'derived from paired 62ch/6ch', 'role': 'derived', 'status': True},
#     {'output': 'table_5_7', 'source': 'classical_summary + optional deep', 'role': 'partial', 'status': True},
# ])
# _record_table(table_audit, 'table_audit_sources', generated_tables)

# # ------------------------------------------------------------------------------
# # 2) FIGURES — ALL SAVED AS SEPARATE PLOTS
# # ------------------------------------------------------------------------------

# log_audit("Regenerating all figures as separate plot files...")

# # Figure 5.1a — Classical 62ch
# if classical_official is not None:
#     data_62 = classical_official[classical_official['ch'] == '62ch'].sort_values('F1_mean', ascending=True)
#     if len(data_62) > 0:
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(data_62)), data_62['F1_mean'], xerr=data_62['F1_std'],
#                 color=COLORS['classical'], alpha=0.85)
#         ax.set_yticks(range(len(data_62)))
#         ax.set_yticklabels(data_62['name'])
#         ax.set_xlabel('Macro-F1')
#         ax.set_title('Figure 5.1A — Classical ML Leaderboard (62 channels)')
#         ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
#         ax.legend()
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_1a_classical_leaderboard_62ch', generated_figs)

# # Figure 5.1b — Classical 6ch
# if classical_official is not None:
#     data_6 = classical_official[classical_official['ch'] == '6ch'].sort_values('F1_mean', ascending=True)
#     if len(data_6) > 0:
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(data_6)), data_6['F1_mean'], xerr=data_6['F1_std'],
#                 color=COLORS['classical'], alpha=0.85)
#         ax.set_yticks(range(len(data_6)))
#         ax.set_yticklabels(data_6['name'])
#         ax.set_xlabel('Macro-F1')
#         ax.set_title('Figure 5.1B — Classical ML Leaderboard (6 channels)')
#         ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
#         ax.legend()
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_1b_classical_leaderboard_6ch', generated_figs)

# # Figure 5.2 — Deep 62ch
# if deep_official is not None:
#     data_62 = deep_official[deep_official['ch'] == '62ch'].sort_values('acc_b_mean', ascending=True)
#     if len(data_62) > 0:
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(data_62)), data_62['acc_b_mean'], xerr=data_62['acc_b_std'],
#                 color=COLORS['deep'], alpha=0.85)
#         ax.set_yticks(range(len(data_62)))
#         ax.set_yticklabels(data_62['name'])
#         ax.set_xlabel('AccB (Proto-B Balanced Accuracy)')
#         ax.set_title('Figure 5.2 — Deep Learning Baselines (62 channels)')
#         ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance')
#         ax.legend()
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_2_deep_leaderboard_62ch', generated_figs)

# # Figure 5.3 — Deep 6ch
# if deep_official is not None:
#     data_6 = deep_official[deep_official['ch'] == '6ch'].sort_values('acc_b_mean', ascending=True)
#     if len(data_6) > 0:
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(data_6)), data_6['acc_b_mean'], xerr=data_6['acc_b_std'],
#                 color=COLORS['deep'], alpha=0.85)
#         ax.set_yticks(range(len(data_6)))
#         ax.set_yticklabels(data_6['name'])
#         ax.set_xlabel('AccB (Proto-B Balanced Accuracy)')
#         ax.set_title('Figure 5.3 — Deep Learning Baselines (6 channels)')
#         ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Chance')
#         ax.legend()
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_3_deep_leaderboard_6ch', generated_figs)

# # Figure 5.4 — Proto-A vs Proto-B
# if deep_official is not None:
#     fig, ax = plt.subplots(figsize=(10, 8))

#     for _, row in deep_official.iterrows():
#         if pd.notna(row.get('acc_a_mean', np.nan)) and pd.notna(row.get('acc_b_mean', np.nan)):
#             color = COLORS['deep'] if row['ch'] == '62ch' else COLORS['baseline']
#             ax.scatter(row['acc_a_mean'], row['acc_b_mean'], s=100, alpha=0.75,
#                        color=color, label=f"{row['name']} ({row['ch']})")

#     if dance_loso_official is not None:
#         for _, row in dance_loso_official.iterrows():
#             if pd.notna(row.get('acc_a_mean', np.nan)) and pd.notna(row.get('acc_b_mean', np.nan)):
#                 ax.scatter(row['acc_a_mean'], row['acc_b_mean'], s=220, alpha=0.9,
#                            color=COLORS['dance'], marker='*', label=row['name'])

#     lims = [0.2, 0.7]
#     ax.plot(lims, lims, 'k--', alpha=0.3, label='Proto-A = Proto-B')
#     ax.set_xlabel('AccA (Proto-A)')
#     ax.set_ylabel('AccB (Proto-B)')
#     ax.set_title('Figure 5.4 — Proto-A vs Proto-B Calibration Gain')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_4_protoA_vs_protoB', generated_figs)

# # Figure 5.5a — DANCE teacher vs top 62ch
# if dance_loso_official is not None and deep_official is not None:
#     top_62 = deep_official[deep_official['ch'] == '62ch'].nlargest(5, 'acc_b_mean')
#     m25 = dance_loso_official[dance_loso_official['model_id'] == 'M25']
#     if len(m25) > 0:
#         combined_62 = pd.concat([top_62, m25], ignore_index=True).sort_values('acc_b_mean', ascending=True)
#         colors_62 = [COLORS['dance'] if x == 'M25' else COLORS['deep'] for x in combined_62['model_id']]
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(combined_62)), combined_62['acc_b_mean'],
#                 xerr=combined_62['acc_b_std'], color=colors_62, alpha=0.85)
#         ax.set_yticks(range(len(combined_62)))
#         ax.set_yticklabels(combined_62['name'])
#         ax.set_xlabel('AccB')
#         ax.set_title('Figure 5.5A — DANCE Teacher vs Top 62ch Baselines')
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_5a_dance_teacher_vs_top_62ch', generated_figs)

# # Figure 5.5b — DANCE student vs top 6ch
# if dance_loso_official is not None and deep_official is not None:
#     top_6 = deep_official[deep_official['ch'] == '6ch'].nlargest(5, 'acc_b_mean')
#     m26 = dance_loso_official[dance_loso_official['model_id'] == 'M26']
#     if len(m26) > 0:
#         combined_6 = pd.concat([top_6, m26], ignore_index=True).sort_values('acc_b_mean', ascending=True)
#         colors_6 = [COLORS['dance'] if x == 'M26' else COLORS['baseline'] for x in combined_6['model_id']]
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.barh(range(len(combined_6)), combined_6['acc_b_mean'],
#                 xerr=combined_6['acc_b_std'], color=colors_6, alpha=0.85)
#         ax.set_yticks(range(len(combined_6)))
#         ax.set_yticklabels(combined_6['name'])
#         ax.set_xlabel('AccB')
#         ax.set_title('Figure 5.5B — DANCE Student vs Top 6ch Baselines')
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_5b_dance_student_vs_top_6ch', generated_figs)

# # Figure 5.6 — t-SNE geometry
# log_audit("Generating supplementary Figure 5.6: t-SNE geometry...")
# tsne_candidates = []
# for search_path in [ROOT / 'results', ROOT / 'features', ROOT / 'checkpoints']:
#     if search_path.exists():
#         tsne_candidates.extend(list(search_path.rglob('*tsne*.npy')))
#         tsne_candidates.extend(list(search_path.rglob('*embedding*.npy')))
#         tsne_candidates.extend(list(search_path.rglob('*projection*.npy')))

# tsne_candidates = sorted(set(tsne_candidates), key=lambda p: p.stat().st_mtime, reverse=True) if tsne_candidates else []

# if tsne_candidates:
#     tsne_file = tsne_candidates[0]
#     try:
#         embeddings = np.load(tsne_file)
#         if embeddings.ndim >= 2 and embeddings.shape[1] >= 2:
#             fig, ax = plt.subplots(figsize=(10, 8))

#             label_candidates = [
#                 tsne_file.with_name(tsne_file.name.replace('tsne', 'labels')),
#                 tsne_file.with_name(tsne_file.name.replace('embedding', 'labels')),
#                 tsne_file.with_name(tsne_file.name.replace('projection', 'labels')),
#             ]
#             label_file = next((p for p in label_candidates if p.exists()), None)

#             if label_file is not None:
#                 labels = np.load(label_file)
#                 class_map = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
#                 for class_id, class_name in class_map.items():
#                     mask = labels == class_id
#                     if np.any(mask):
#                         ax.scatter(
#                             embeddings[mask, 0], embeddings[mask, 1],
#                             alpha=0.65, s=30, label=class_name,
#                             color=COLORS.get(class_name.lower(), 'gray')
#                         )
#                 ax.legend()
#             else:
#                 ax.scatter(embeddings[:, 0], embeddings[:, 1],
#                            alpha=0.5, s=20, color=COLORS['deep'])

#             ax.set_xlabel('t-SNE Dimension 1')
#             ax.set_ylabel('t-SNE Dimension 2')
#             ax.set_title('Figure 5.6 — Feature Space Representation (t-SNE)')
#             plt.tight_layout()
#             _record_fig(fig, 'fig_5_6_tsne_geometry', generated_figs)
#         else:
#             log_audit(f"Figure 5.6 skipped ({tsne_file.name} does not contain 2D embeddings)", 'WARNING')
#     except Exception as e:
#         log_audit(f"Figure 5.6 failed: {e}", 'WARNING')
# else:
#     log_audit("Figure 5.6 skipped (no t-SNE / embedding files found)", 'WARNING')

# # Figure 5.7 — Ablation absolute
# if ablations_df is not None:
#     abl = ablations_df.sort_values('acc_B_mean', ascending=True)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     bar_colors = [
#         COLORS['dance'] if ('A01' in str(x) or 'Full' in str(x)) else COLORS['baseline']
#         for x in abl['ablation']
#     ]
#     ax.barh(range(len(abl)), abl['acc_B_mean'],
#             xerr=abl['acc_B_std'] if 'acc_B_std' in abl.columns else None,
#             color=bar_colors, alpha=0.85)
#     ax.set_yticks(range(len(abl)))
#     ax.set_yticklabels(abl['ablation'])
#     ax.set_xlabel('AccB (Proto-B)')
#     ax.set_title('Figure 5.7 — DANCE Ablation Study (Absolute Performance)')
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_7_ablation_abs', generated_figs)

# # Figure 5.8 — Ablation delta
# if ablations_df is not None:
#     baseline_row = ablations_df[ablations_df['ablation'].astype(str).str.contains('A01|Full', case=False, na=False)]
#     if len(baseline_row) > 0:
#         baseline_acc = baseline_row.iloc[0]['acc_B_mean']
#         abl = ablations_df.copy()
#         abl['delta'] = abl['acc_B_mean'] - baseline_acc
#         abl = abl.sort_values('delta', ascending=True)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         delta_colors = ['red' if x < 0 else 'green' for x in abl['delta']]
#         ax.barh(range(len(abl)), abl['delta'], color=delta_colors, alpha=0.85)
#         ax.set_yticks(range(len(abl)))
#         ax.set_yticklabels(abl['ablation'])
#         ax.set_xlabel('Δ AccB vs Full DANCE')
#         ax.set_title('Figure 5.8 — DANCE Ablation Study (Delta from Baseline)')
#         ax.axvline(0, color='black', linestyle='-', linewidth=1)
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_8_ablation_delta', generated_figs)
#     else:
#         log_audit("Figure 5.8 skipped (baseline A01/Full not found)", 'WARNING')

# # Figure 5.9a / 5.9b — statistical significance, split into separate files
# log_audit("Generating supplementary Figure 5.9: statistical significance (split plots)...")
# comparison_results = []

# if SCIPY_AVAILABLE and deep_per_model and dance_m25_df is not None and 'acc_b' in dance_m25_df.columns:
#     m25_values = dance_m25_df['acc_b'].dropna().values

#     if deep_official is not None:
#         top_62 = deep_official[deep_official['ch'] == '62ch'].nlargest(3, 'acc_b_mean')

#         for _, model_row in top_62.iterrows():
#             model_id = model_row['model_id']
#             key = f"{model_id}_62ch"
#             model_df = deep_per_model.get(key)

#             if model_df is not None and 'acc_b' in model_df.columns:
#                 model_values = model_df['acc_b'].dropna().values
#                 n = min(len(m25_values), len(model_values))

#                 if n >= 2:
#                     t_stat, p_value = stats.ttest_rel(m25_values[:n], model_values[:n])
#                     comparison_results.append({
#                         'model': model_row['name'],
#                         'model_mean': model_row['acc_b_mean'],
#                         'm25_mean': (
#                             dance_loso_official[dance_loso_official['model_id'] == 'M25'].iloc[0]['acc_b_mean']
#                             if dance_loso_official is not None and len(dance_loso_official[dance_loso_official['model_id'] == 'M25']) > 0
#                             else np.nan
#                         ),
#                         't_stat': t_stat,
#                         'p_value': p_value,
#                         'significant': p_value < 0.05,
#                         'n_paired': n
#                     })

# if comparison_results:
#     comp_df = pd.DataFrame(comparison_results)
#     _record_table(comp_df, 'audit_statistical_tests', generated_tables)

#     # Figure 5.9a
#     fig, ax = plt.subplots(figsize=(10, 6))
#     x = np.arange(len(comp_df))
#     ax.bar(x, comp_df['model_mean'], alpha=0.8, color=COLORS['deep'], label='Baseline')
#     if pd.notna(comp_df['m25_mean'].iloc[0]):
#         ax.axhline(comp_df['m25_mean'].iloc[0], color=COLORS['dance'], linestyle='--',
#                    linewidth=2, label='DANCE Teacher')

#     for i, row in comp_df.iterrows():
#         if row['significant']:
#             marker = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else '*')
#             y = max(row['model_mean'], row['m25_mean']) + 0.01
#             ax.text(i, y, marker, ha='center', fontsize=14)

#     ax.set_xticks(x)
#     ax.set_xticklabels(comp_df['model'], rotation=45, ha='right')
#     ax.set_ylabel('AccB (Proto-B)')
#     ax.set_title('Figure 5.9A — DANCE vs Top 62ch Baselines')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_9a_stat_sig_mean_comparison', generated_figs)

#     # Figure 5.9b
#     fig, ax = plt.subplots(figsize=(10, 6))
#     p_colors = ['green' if p < 0.05 else 'red' for p in comp_df['p_value']]
#     ax.barh(np.arange(len(comp_df)), -np.log10(comp_df['p_value']), color=p_colors, alpha=0.8)
#     ax.axvline(-np.log10(0.05), color='black', linestyle='--', label='p=0.05 threshold')
#     ax.set_yticks(np.arange(len(comp_df)))
#     ax.set_yticklabels(comp_df['model'])
#     ax.set_xlabel('-log10(p-value)')
#     ax.set_title('Figure 5.9B — Statistical Significance (Paired t-test)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_9b_stat_sig_pvalues', generated_figs)
# else:
#     log_audit("Figure 5.9 skipped (insufficient run-level paired data)", 'WARNING')

# # Figure 5.10 — Channel efficiency scatter
# if channel_eff_rows:
#     eff_df = pd.DataFrame(channel_eff_rows)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     for family, color in [('classical', COLORS['classical']), ('deep', COLORS['deep']), ('dance', COLORS['dance'])]:
#         subset = eff_df[eff_df['family'] == family]
#         if len(subset) > 0:
#             marker = '*' if family == 'dance' else 'o'
#             size = 220 if family == 'dance' else 90
#             ax.scatter(subset['primary_62'], subset['primary_6'],
#                        s=size, alpha=0.75, color=color, marker=marker, label=family.capitalize())
#     lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
#     ax.plot(lims, lims, 'k--', alpha=0.3, label='62ch = 6ch')
#     ax.set_xlabel('Primary Metric (62ch)')
#     ax.set_ylabel('Primary Metric (6ch)')
#     ax.set_title('Figure 5.10 — Channel Efficiency: 62ch vs 6ch Performance')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_10_channel_efficiency_scatter', generated_figs)

# # Figure 5.11 — Channel retention
# if channel_eff_rows:
#     eff_df = pd.DataFrame(channel_eff_rows).sort_values('retention_primary', ascending=True)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     bar_colors = [COLORS.get(f, COLORS['baseline']) for f in eff_df['family']]
#     ax.barh(range(len(eff_df)), eff_df['retention_primary'], color=bar_colors, alpha=0.85)
#     ax.set_yticks(range(len(eff_df)))
#     ax.set_yticklabels(eff_df['name'])
#     ax.set_xlabel('Retention Ratio (6ch / 62ch)')
#     ax.set_title('Figure 5.11 — Channel Efficiency: Performance Retention')
#     ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Parity')
#     ax.legend()
#     plt.tight_layout()
#     _record_fig(fig, 'fig_5_11_channel_efficiency_retention', generated_figs)

# # Figure 5.12 — Per-class all models
# log_audit("Generating supplementary Figure 5.12: per-class all models...")
# if classical_official is not None:
#     per_class_cols = ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']
#     available = [c for c in per_class_cols if c in classical_official.columns]
#     if len(available) == 4:
#         top_62 = classical_official[classical_official['ch'] == '62ch'].nlargest(5, 'F1_mean')
#         top_6 = classical_official[classical_official['ch'] == '6ch'].nlargest(5, 'F1_mean')
#         combined = pd.concat([top_62, top_6], ignore_index=True)

#         fig, ax = plt.subplots(figsize=(12, 8))
#         x = np.arange(len(combined))
#         width = 0.2
#         emotions = ['Neutral', 'Sad', 'Fear', 'Happy']

#         for i, emotion in enumerate(emotions):
#             col = f'F1_{emotion}'
#             ax.bar(
#                 x + i * width,
#                 combined[col].values,
#                 width,
#                 label=emotion,
#                 color=COLORS.get(emotion.lower(), 'gray'),
#                 alpha=0.85
#             )

#         ax.set_xticks(x + width * 1.5)
#         ax.set_xticklabels(
#             [f"{row['name']} ({row['ch']})" for _, row in combined.iterrows()],
#             rotation=45,
#             ha='right'
#         )
#         ax.set_ylabel('F1 Score')
#         ax.set_title('Figure 5.12 — Per-Class F1 Scores (Top Classical Models)')
#         ax.legend()
#         ax.grid(True, alpha=0.3, axis='y')
#         plt.tight_layout()
#         _record_fig(fig, 'fig_5_12_per_class_all_models', generated_figs)
#     else:
#         log_audit(f"Figure 5.12 skipped (only {len(available)}/4 per-class columns present)", 'WARNING')

# # Figure 5.13 — Per-class top 62ch
# log_audit("Generating supplementary Figure 5.13: per-class top 62ch...")
# if classical_official is not None:
#     per_class_cols = ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']
#     available = [c for c in per_class_cols if c in classical_official.columns]
#     if len(available) == 4:
#         top_62 = classical_official[classical_official['ch'] == '62ch'].nlargest(8, 'F1_mean')
#         if len(top_62) > 0:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             x = np.arange(len(top_62))
#             width = 0.2
#             for i, emotion in enumerate(['Neutral', 'Sad', 'Fear', 'Happy']):
#                 col = f'F1_{emotion}'
#                 ax.bar(x + i * width, top_62[col].values, width,
#                        label=emotion, color=COLORS.get(emotion.lower(), 'gray'), alpha=0.85)
#             ax.set_xticks(x + width * 1.5)
#             ax.set_xticklabels(top_62['name'], rotation=45, ha='right')
#             ax.set_ylabel('F1 Score')
#             ax.set_title('Figure 5.13 — Per-Class F1 (Top 62-Channel Models)')
#             ax.legend()
#             ax.grid(True, alpha=0.3, axis='y')
#             plt.tight_layout()
#             _record_fig(fig, 'fig_5_13_per_class_top_62ch', generated_figs)

# # Figure 5.14 — Per-class top 6ch
# log_audit("Generating supplementary Figure 5.14: per-class top 6ch...")
# if classical_official is not None:
#     per_class_cols = ['F1_Neutral', 'F1_Sad', 'F1_Fear', 'F1_Happy']
#     available = [c for c in per_class_cols if c in classical_official.columns]
#     if len(available) == 4:
#         top_6 = classical_official[classical_official['ch'] == '6ch'].nlargest(8, 'F1_mean')
#         if len(top_6) > 0:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             x = np.arange(len(top_6))
#             width = 0.2
#             for i, emotion in enumerate(['Neutral', 'Sad', 'Fear', 'Happy']):
#                 col = f'F1_{emotion}'
#                 ax.bar(x + i * width, top_6[col].values, width,
#                        label=emotion, color=COLORS.get(emotion.lower(), 'gray'), alpha=0.85)
#             ax.set_xticks(x + width * 1.5)
#             ax.set_xticklabels(top_6['name'], rotation=45, ha='right')
#             ax.set_ylabel('F1 Score')
#             ax.set_title('Figure 5.14 — Per-Class F1 (Top 6-Channel Models)')
#             ax.legend()
#             ax.grid(True, alpha=0.3, axis='y')
#             plt.tight_layout()
#             _record_fig(fig, 'fig_5_14_per_class_top_6ch', generated_figs)

# # Figure 5.15 — Confusion matrices, each saved separately
# log_audit("Generating supplementary Figure 5.15: confusion matrices (separate files)...")
# confusion_candidates = []
# for search_path in [ROOT / 'results', ROOT / 'checkpoints']:
#     if search_path.exists():
#         confusion_candidates.extend(list(search_path.rglob('*confusion*.npy')))
#         confusion_candidates.extend(list(search_path.rglob('*confusion*.csv')))

# confusion_candidates = sorted(set(confusion_candidates), key=lambda p: p.stat().st_mtime, reverse=True) if confusion_candidates else []

# if confusion_candidates:
#     plotted = 0
#     for conf_file in confusion_candidates:
#         try:
#             if conf_file.suffix.lower() == '.npy':
#                 cm = np.load(conf_file)
#             else:
#                 cm = pd.read_csv(conf_file, index_col=0).values

#             if cm.ndim != 2:
#                 continue

#             row_sums = cm.sum(axis=1, keepdims=True)
#             row_sums[row_sums == 0] = 1
#             cm_norm = cm.astype(float) / row_sums

#             fig, ax = plt.subplots(figsize=(6, 5))
#             im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')

#             for r in range(cm_norm.shape[0]):
#                 for c in range(cm_norm.shape[1]):
#                     ax.text(
#                         c, r, f"{cm_norm[r, c]:.2f}",
#                         ha='center', va='center',
#                         color='white' if cm_norm[r, c] > 0.5 else 'black'
#                     )

#             if cm.shape == (4, 4):
#                 tick_labels = ['Neutral', 'Sad', 'Fear', 'Happy']
#                 ax.set_xticks(range(4))
#                 ax.set_yticks(range(4))
#                 ax.set_xticklabels(tick_labels)
#                 ax.set_yticklabels(tick_labels)
#             else:
#                 ax.set_xticks(range(cm.shape[1]))
#                 ax.set_yticks(range(cm.shape[0]))

#             ax.set_xlabel('Predicted')
#             ax.set_ylabel('True')

#             model_name = conf_file.stem.replace('confusion_', '').replace('_matrix', '')
#             ax.set_title(f'Figure 5.15 — Confusion Matrix: {model_name}')
#             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#             plt.tight_layout()

#             out_name = f"fig_5_15_confusion_{_slug(model_name)}"
#             _record_fig(fig, out_name, generated_figs)
#             plotted += 1

#         except Exception as e:
#             log_audit(f"Failed to load confusion matrix {conf_file.name}: {e}", 'WARNING')

#     if plotted == 0:
#         log_audit("Figure 5.15 skipped (confusion files found but none could be plotted)", 'WARNING')
# else:
#     log_audit("Figure 5.15 skipped (no confusion matrix files found)", 'WARNING')

# # ------------------------------------------------------------------------------
# # 3) FINAL INVENTORY PRINT
# # ------------------------------------------------------------------------------

# print("\n" + "=" * 100)
# print("MASTER CELL COMPLETE")
# print("=" * 100)

# print(f"\nGenerated tables in this cell: {len(generated_tables)}")
# for name, df in generated_tables.items():
#     print(f"  - {name}.csv  |  {df.shape[0]} rows x {df.shape[1]} cols")

# print(f"\nGenerated figures in this cell: {len(generated_figs)}")
# for name in generated_figs:
#     print(f"  - {name}.png / {name}.pdf")

# print("\nSaved table directory:")
# print(f"  {OUTPUT_DIRS['tables']}")

# print("\nSaved figure directories:")
# print(f"  PNG: {OUTPUT_DIRS['figures_png']}")
# print(f"  PDF: {OUTPUT_DIRS['figures_pdf']}")

# # Optional: refresh audit text safely as UTF-8
# if 'audit_log' in globals():
#     audit_text = []
#     audit_text.append("=" * 80)
#     audit_text.append("CSE400C FINAL RESULTS REBUILD - AUDIT REPORT")
#     audit_text.append("=" * 80)
#     audit_text.append(f"Generated: {datetime.now().isoformat()}")
#     audit_text.append(f"Project Root: {ROOT}")
#     audit_text.append("")
#     audit_text.append("WARNINGS")
#     audit_text.append("-" * 40)
#     for w in globals().get('audit_warnings', [])[:50]:
#         audit_text.append(f"⚠ {w}")
#     audit_text.append("")
#     audit_text.append("ERRORS")
#     audit_text.append("-" * 40)
#     for e in globals().get('audit_errors', [])[:50]:
#         audit_text.append(f"❌ {e}")
#     audit_text.append("")
#     audit_text.append("FULL LOG")
#     audit_text.append("-" * 40)
#     audit_text.extend(globals().get('audit_log', []))

#     audit_txt_path = OUTPUT_DIRS['audit'] / 'results_audit.txt'
#     with open(audit_txt_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(audit_text))

#     print(f"\nAudit report refreshed: {audit_txt_path}")


# ==============================================================================
# Notebook cell 57
# Categories: other
# ==============================================================================

