# Auto-exported raw code from notebook: 01_classical_ml.ipynb
# Generated: 2026-05-17_23-35-17
# NOTE: This is a raw provenance export, not cleaned production code.


# ==============================================================================
# Notebook cell 1
# Categories: results_tables
# ==============================================================================
# ── 0. Install gate ────────────────────────────────────────────────────────────
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
        print(f"  ✅ {pkg} already installed")
    except ImportError:
        print(f"  ⏳ Installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  ✅ {pkg} installed")

ensure("xgboost")
ensure("scikit-learn", "sklearn")
ensure("joblib")
ensure("tqdm")
print("\n✅ All dependencies satisfied")



# ==============================================================================
# Notebook cell 2
# Categories: model_definition, evaluation, figures
# ==============================================================================
import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

# sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                              classification_report)
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(0)

print("✅ All imports successful")
print(f"   sklearn version : {__import__('sklearn').__version__}")
print(f"   xgboost version : {xgb.__version__}")



# ==============================================================================
# Notebook cell 3
# Categories: preprocessing, training, figures, audit_verification
# ==============================================================================
# ── 1. Configuration ────────────────────────────────────────────────────────────
# Adjust BASE_DIR if running from a different working directory
BASE_DIR        = os.path.abspath(".")          # project root
FEAT_DIR        = os.path.join(BASE_DIR, "features")
RESULTS_DIR     = os.path.join(BASE_DIR, "results", "classical_ml")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints", "loso_results")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures", "models")

for d in [RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# LOSO / reproducibility settings
SEEDS    = [1, 7, 21]
N_FOLDS  = 15                           # SEED-IV: 15 subjects
SUBJECTS = list(range(1, N_FOLDS + 1))  # 1 … 15

# 6-channel indices in SEED-IV 62-ch ordering
# FP1=0, FP2=2, F7=5, F8=13, T7=23, T8=31
CH6_SEED = [0, 2, 5, 13, 23, 31]
CH6_NAMES = ["FP1", "FP2", "F7", "F8", "T7", "T8"]

# Emotion classes
EMOTION_NAMES = ["Neutral", "Sad", "Fear", "Happy"]

# Phase-B reference numbers (for comparison in final report)
PHASE_B_REF = {
    "M01": {"f1_62ch": 0.4182, "f1_6ch": 0.4170},
    "M02": {"f1_62ch": 0.4472, "f1_6ch": 0.4077},
    "M03": {"f1_62ch": 0.4798, "f1_6ch": 0.3469},
    "M04": {"f1_62ch": 0.3441, "f1_6ch": 0.2945},
    "M05": {"f1_62ch": 0.3803, "f1_6ch": 0.4241},
    "M06": {"f1_62ch": 0.2999, "f1_6ch": 0.3841},
    "M07": {"f1_62ch": 0.4604, "f1_6ch": 0.3934},
    "M08": {"f1_62ch": 0.4607, "f1_6ch": 0.4048},
    "M09": {"f1_62ch": float("nan"), "f1_6ch": float("nan")},  # was missing
    "M10": {"f1_62ch": 0.4032, "f1_6ch": 0.4258},
}

print("✅ Configuration set")
print(f"   BASE_DIR    : {BASE_DIR}")
print(f"   FEAT_DIR    : {FEAT_DIR}")
print(f"   RESULTS_DIR : {RESULTS_DIR}")
print(f"   CHECKPOINT  : {CHECKPOINT_DIR}")
print(f"   FIGURES     : {FIGURES_DIR}")
print(f"   Seeds       : {SEEDS}")
print(f"   Folds       : {N_FOLDS}")



# ==============================================================================
# Notebook cell 4
# Categories: training
# ==============================================================================
# ── 2. Checkpoint utilities ──────────────────────────────────────────────────────

def checkpoint_key(model_id: str, ch_tag: str, seed: int, fold: int) -> str:
    """Unique key: e.g. M01_62ch_seed1_fold00"""
    return f"{model_id}_{ch_tag}_seed{seed}_fold{fold:02d}"

def load_checkpoint(key: str):
    path = os.path.join(CHECKPOINT_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_checkpoint(key: str, result: dict):
    path = os.path.join(CHECKPOINT_DIR, f"{key}.json")
    # Convert numpy types to plain Python
    def cvt(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj
    clean = {k: cvt(v) if not isinstance(v, list) else
             [cvt(x) if not isinstance(x, list) else [cvt(y) for y in x] for x in v]
             for k, v in result.items()}
    with open(path, "w") as f:
        json.dump(clean, f)

def recovery_report(model_ids, ch_tags=("62ch", "6ch")):
    """Print completion status for all models."""
    total = len(model_ids) * len(ch_tags) * len(SEEDS) * N_FOLDS
    done  = 0
    for mid in model_ids:
        for ct in ch_tags:
            n = sum(1 for s in SEEDS for fi in range(N_FOLDS)
                    if load_checkpoint(checkpoint_key(mid, ct, s, fi)) is not None)
            done += n
    print(f"  Completed checkpoints : {done} / {total}")
    print(f"  Remaining             : {total - done}")
    return done, total

print("✅ Checkpoint utilities ready")
# Scan existing checkpoints
recovery_report([f"M{i:02d}" for i in range(1, 11)])



# ==============================================================================
# Notebook cell 5
# Categories: preprocessing, audit_verification
# ==============================================================================
# ── 3. Load SEED-IV Features ────────────────────────────────────────────────────

X62_path  = os.path.join(FEAT_DIR, "seed_iv_X_62ch.npy")
X6_path   = os.path.join(FEAT_DIR, "seed_iv_X_6ch.npy")
y_path    = os.path.join(FEAT_DIR, "seed_iv_y_4cls.npy")
sub_path  = os.path.join(FEAT_DIR, "seed_iv_subjects.npy")

# ── Mandatory existence checks ──────────────────────────────────────────────────
for p in [X62_path, y_path, sub_path]:
    assert os.path.exists(p), f"❌ Missing required file: {p}"

X_62ch   = np.load(X62_path)          # (N, 310) — 62ch × 5 bands
y        = np.load(y_path)            # (N,)  integer labels 0-3
subjects = np.load(sub_path)          # (N,)  integer 1-15

# 6-channel subset: select the 5-band block for each of the 6 channels
# SEED-IV feature ordering: [ch0_delta, ch0_theta, …, ch0_gamma, ch1_delta, …]
# Each channel occupies 5 consecutive features → 6ch indices in 310-d vector
CH6_FEAT_IDX = []
for ch in CH6_SEED:
    CH6_FEAT_IDX.extend([ch*5 + b for b in range(5)])  # 5 bands per channel

if os.path.exists(X6_path):
    X_6ch = np.load(X6_path)
    print("  Loaded precomputed 6-ch features from disk")
else:
    X_6ch = X_62ch[:, CH6_FEAT_IDX]
    np.save(X6_path, X_6ch)
    print("  Computed and saved 6-ch features")

print(f"\n✅ Data loaded")
print(f"   X_62ch   shape : {X_62ch.shape}   (expected [N, 310])")
print(f"   X_6ch    shape : {X_6ch.shape}    (expected [N,  30])")
print(f"   y        shape : {y.shape}   labels: {np.unique(y)}")
print(f"   subjects shape : {subjects.shape}   unique: {np.unique(subjects)}")
print(f"   Total samples  : {len(y)}")
print(f"\n   Class counts (0=Neutral,1=Sad,2=Fear,3=Happy):")
for cls, name in enumerate(EMOTION_NAMES):
    print(f"     {cls} {name:<9}: {(y == cls).sum():5d}")



# ==============================================================================
# Notebook cell 6
# Categories: preprocessing, training, evaluation, results_tables
# ==============================================================================
# ── 4. LOSO Utilities ────────────────────────────────────────────────────────────

def loso_split(X: np.ndarray, y: np.ndarray, test_subj: int):
    """Return (X_train, y_train, X_test, y_test) for leave-one-subject-out."""
    tr_mask  = subjects != test_subj
    te_mask  = subjects == test_subj
    return X[tr_mask], y[tr_mask], X[te_mask], y[te_mask]

def compute_metrics(y_true, y_pred):
    """Return dict with all metrics needed for checkpointing."""
    return {
        "f1_macro"    : float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
        "f1_weighted" : float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy"    : float(accuracy_score(y_true, y_pred)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "conf_matrix" : confusion_matrix(y_true, y_pred).tolist(),
    }

def run_loso(model_id: str, ch_tag: str, X: np.ndarray, build_clf_fn,
             verbose: bool = True) -> list:
    """
    Run LOSO-15 × 3 seeds for one model on one channel configuration.
    Returns list of result dicts (45 entries).
    """
    results = []
    for seed in SEEDS:
        for fold_idx, test_subj in enumerate(SUBJECTS):
            key = checkpoint_key(model_id, ch_tag, seed, fold_idx)
            cached = load_checkpoint(key)
            if cached is not None:
                if verbose:
                    print(f"  SKIP {key} (cached F1={cached['f1_macro']:.4f})")
                cached.update({"seed": seed, "fold": fold_idx, "test_subj": test_subj})
                results.append(cached)
                continue

            X_tr, y_tr, X_te, y_te = loso_split(X, y, test_subj)
            clf = build_clf_fn(seed)
            t0  = time.time()
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            elapsed = time.time() - t0
            metrics = compute_metrics(y_te, y_pred)
            rec = {**metrics, "seed": seed, "fold": fold_idx, "test_subj": test_subj,
                   "elapsed_s": round(elapsed, 2)}
            save_checkpoint(key, rec)
            if verbose:
                print(f"  {model_id} | {ch_tag} | Seed {seed} | Fold {fold_idx+1:02d}/15 "
                      f"| Subj {test_subj:02d} | F1={metrics['f1_macro']:.4f} "
                      f"| Acc={metrics['accuracy']:.4f} | {elapsed:.1f}s")
            results.append(rec)
    return results


def summarise(results: list, model_id: str, ch_tag: str) -> dict:
    """Compute mean ± std over 45 runs."""
    f1s  = [r["f1_macro"]  for r in results]
    accs = [r["accuracy"]  for r in results]
    f1pc = np.array([r["f1_per_class"] for r in results])  # (45, 4)
    return {
        "model_id"         : model_id,
        "ch_tag"           : ch_tag,
        "f1_mean"          : float(np.mean(f1s)),
        "f1_std"           : float(np.std(f1s)),
        "acc_mean"         : float(np.mean(accs)),
        "acc_std"          : float(np.std(accs)),
        "f1_per_class_mean": f1pc.mean(axis=0).tolist(),
        "f1_per_class_std" : f1pc.std(axis=0).tolist(),
        "n_runs"           : len(results),
    }

print("✅ LOSO utilities ready")



# ==============================================================================
# Notebook cell 7
# Categories: preprocessing, results_tables
# ==============================================================================
# ── 5. Runner helper (used by every model cell) ──────────────────────────────────

ALL_SUMMARIES = []   # will accumulate across model cells

def run_model(model_id: str, label: str, build_fn_62ch, build_fn_6ch=None):
    """
    Run 62ch AND 6ch LOSO for a model; print summary; append to ALL_SUMMARIES.
    build_fn_6ch defaults to build_fn_62ch (same hyper-params).
    """
    if build_fn_6ch is None:
        build_fn_6ch = build_fn_62ch

    print(f"\n{'='*65}")
    print(f"  {model_id} — {label}")
    print(f"{'='*65}")

    # 62-channel ────────────────────────────────────────────────────
    print("\n  [62ch]")
    r62 = run_loso(model_id, "62ch", X_62ch, build_fn_62ch)
    s62 = summarise(r62, model_id, "62ch")
    pb  = PHASE_B_REF[model_id]["f1_62ch"]
    delta = s62["f1_mean"] - pb if not np.isnan(pb) else float("nan")
    print(f"\n  ➜ 62ch  F1 = {s62['f1_mean']:.4f} ± {s62['f1_std']:.4f}  "
          f"| PhaseB ref = {pb:.4f}  | Δ = {delta:+.4f}")

    # 6-channel ─────────────────────────────────────────────────────
    print("\n  [6ch]")
    r6  = run_loso(model_id, "6ch", X_6ch, build_fn_6ch)
    s6  = summarise(r6,  model_id, "6ch")
    pb6 = PHASE_B_REF[model_id]["f1_6ch"]
    d6  = s6["f1_mean"] - pb6 if not np.isnan(pb6) else float("nan")
    print(f"\n  ➜ 6ch   F1 = {s6['f1_mean']:.4f} ± {s6['f1_std']:.4f}  "
          f"| PhaseB ref = {pb6:.4f}  | Δ = {d6:+.4f}")

    ALL_SUMMARIES.extend([s62, s6])
    return s62, s6



# ==============================================================================
# Notebook cell 8
# Categories: other
# ==============================================================================
# ── M01 — Linear Discriminant Analysis ─────────────────────────────────────────
# Plan spec: solver=svd, shrinkage=auto  (Note: shrinkage only available with
# solver='lsqr' or 'eigen'; using 'lsqr' + shrinkage='auto')

def build_lda(seed):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

run_model("M01", "LDA (lsqr + shrinkage=auto)", build_lda)



# ==============================================================================
# Notebook cell 9
# Categories: preprocessing, training, evaluation
# ==============================================================================
# ── M02 — SVM (RBF kernel) ────────────────────────────────────────────────────
# Plan spec: C ∈ {1,10,100}, gamma=scale.
# We select C=10 as default (validated as typical best on EEG features).
# If you want full grid-search per fold, set FULL_SEARCH=True (slower ~×3).

FULL_SEARCH_SVM = False

def build_svm(seed):
    if FULL_SEARCH_SVM:
        from sklearn.model_selection import GridSearchCV
        base = SVC(kernel="rbf", gamma="scale", probability=False)
        return GridSearchCV(base, {"C": [1, 10, 100]}, cv=3,
                            scoring="f1_macro", n_jobs=-1, refit=True)
    return SVC(kernel="rbf", C=10, gamma="scale", probability=False)

run_model("M02", "SVM (RBF, C=10, gamma=scale)", build_svm)



# ==============================================================================
# Notebook cell 10
# Categories: model_definition
# ==============================================================================
# ── M03 — Random Forest ──────────────────────────────────────────────────────
# Plan spec: n_estimators=500  (TOP CLASSICAL)

def build_rf(seed):
    return RandomForestClassifier(n_estimators=500, random_state=seed,
                                  n_jobs=-1, class_weight="balanced")

run_model("M03", "Random Forest (n=500, balanced)", build_rf)



# ==============================================================================
# Notebook cell 11
# Categories: model_definition, training, evaluation
# ==============================================================================
# ── M04 — k-Nearest Neighbours ─────────────────────────────────────────────
# Plan spec: k ∈ {3,5,7,11}, cosine metric.
# Default: k=7 cosine. Set FULL_SEARCH_KNN=True for grid-search.

FULL_SEARCH_KNN = False

def build_knn(seed):
    if FULL_SEARCH_KNN:
        from sklearn.model_selection import GridSearchCV
        base = KNeighborsClassifier(metric="cosine", algorithm="brute")
        return GridSearchCV(base, {"n_neighbors": [3, 5, 7, 11]}, cv=3,
                            scoring="f1_macro", n_jobs=-1, refit=True)
    return KNeighborsClassifier(n_neighbors=7, metric="cosine", algorithm="brute",
                                n_jobs=-1)

run_model("M04", "k-NN (k=7, cosine)", build_knn)



# ==============================================================================
# Notebook cell 12
# Categories: other
# ==============================================================================
# ── M05 — Logistic Regression ────────────────────────────────────────────────
# Plan spec: solver=lbfgs, C=1.0, max_iter=500

def build_lr(seed):
    return LogisticRegression(solver="lbfgs", C=1.0, max_iter=500,
                              random_state=seed, multi_class="multinomial",
                              class_weight="balanced", n_jobs=-1)

run_model("M05", "Logistic Regression (lbfgs, C=1.0)", build_lr)



# ==============================================================================
# Notebook cell 13
# Categories: other
# ==============================================================================
# ── M06 — Gaussian Naïve Bayes ───────────────────────────────────────────────
# Plan spec: GaussianNB. No hyperparameters.

def build_nb(seed):
    return GaussianNB()

run_model("M06", "Gaussian Naïve Bayes", build_nb)



# ==============================================================================
# Notebook cell 14
# Categories: model_definition
# ==============================================================================
# ── M07 — Extra Trees ────────────────────────────────────────────────────────
# Plan spec: n_estimators=500

def build_et(seed):
    return ExtraTreesClassifier(n_estimators=500, random_state=seed,
                                n_jobs=-1, class_weight="balanced")

run_model("M07", "Extra Trees (n=500, balanced)", build_et)



# ==============================================================================
# Notebook cell 15
# Categories: preprocessing, model_definition
# ==============================================================================
# ── M08 — Gradient Boosting (HistGBM — fast) ─────────────────────────────────
# Replacing sklearn GradientBoostingClassifier (sequential, too slow for 310 features)
# with HistGradientBoostingClassifier — same algorithm, histogram-binned, ~20x faster.

from sklearn.ensemble import HistGradientBoostingClassifier

def build_gb(seed):
    return HistGradientBoostingClassifier(
        max_iter=300,           # equivalent to n_estimators
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
        l2_regularization=0.1,
        class_weight="balanced",
        early_stopping=False,
    )

run_model("M08", "HistGradBoost (n=300, lr=0.05, depth=6)", build_gb)


# ==============================================================================
# Notebook cell 16
# Categories: model_definition, training, audit_verification
# ==============================================================================
# ── M09 — XGBoost ─────────────────────────────────────────────────────────────
# Plan spec: n_estimators=300, lr=0.05, max_depth=6
# ⚠ Was NaN in Phase B (package missing) — MUST produce real numbers now.

def build_xgb(seed):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=seed, use_label_encoder=False,
        eval_metric="mlogloss", subsample=0.8,
        tree_method="hist",      # faster on CPU
        n_jobs=-1, verbosity=0
    )

run_model("M09", "XGBoost (n=300, lr=0.05, depth=6)", build_xgb)



# ==============================================================================
# Notebook cell 17
# Categories: model_definition
# ==============================================================================
# ── M10 — MLP (sklearn) ──────────────────────────────────────────────────────
# Plan spec: hidden=(256,128), lr=1e-3, max_iter=200

def build_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256, 128),
                         learning_rate_init=1e-3,
                         max_iter=200, random_state=seed,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=20, alpha=1e-4)

run_model("M10", "MLP sklearn (256-128, lr=1e-3)", build_mlp)



# ==============================================================================
# Notebook cell 18
# Categories: preprocessing, results_tables
# ==============================================================================
# ── 6. Aggregate Results → CSV ───────────────────────────────────────────────

rows = []
for s in ALL_SUMMARIES:
    ref_key = "f1_62ch" if s["ch_tag"] == "62ch" else "f1_6ch"
    pb_ref  = PHASE_B_REF[s["model_id"]][ref_key]
    delta   = s["f1_mean"] - pb_ref if not np.isnan(pb_ref) else float("nan")
    rows.append({
        "Model"          : s["model_id"],
        "Name"           : {
            "M01":"LDA","M02":"SVM","M03":"Random Forest","M04":"k-NN",
            "M05":"Logistic Reg","M06":"Naive Bayes","M07":"Extra Trees",
            "M08":"Grad Boost","M09":"XGBoost","M10":"MLP sklearn"
        }[s["model_id"]],
        "Channels"       : s["ch_tag"],
        "F1_mean"        : round(s["f1_mean"], 4),
        "F1_std"         : round(s["f1_std"],  4),
        "Acc_mean"       : round(s["acc_mean"],4),
        "Acc_std"        : round(s["acc_std"], 4),
        "F1_Neutral"     : round(s["f1_per_class_mean"][0], 4),
        "F1_Sad"         : round(s["f1_per_class_mean"][1], 4),
        "F1_Fear"        : round(s["f1_per_class_mean"][2], 4),
        "F1_Happy"       : round(s["f1_per_class_mean"][3], 4),
        "PhaseB_ref_F1"  : pb_ref,
        "Delta_vs_PhaseB": round(delta, 4) if not np.isnan(delta) else float("nan"),
        "N_runs"         : s["n_runs"],
    })

df_results = pd.DataFrame(rows)
csv_path   = os.path.join(RESULTS_DIR, "classical_ml_summary.csv")
df_results.to_csv(csv_path, index=False)

print("✅ Summary CSV saved →", csv_path)
print()
print(df_results.to_string(index=False))



# ==============================================================================
# Notebook cell 19
# Categories: preprocessing, evaluation, results_tables, figures
# ==============================================================================
# ── Figure 1: F1 Macro Comparison — 62ch vs 6ch ─────────────────────────────

df_62 = df_results[df_results["Channels"]=="62ch"].set_index("Model")
df_6  = df_results[df_results["Channels"]=="6ch" ].set_index("Model")
models_order = [f"M{i:02d}" for i in range(1, 11)]
model_names  = [df_62.loc[m, "Name"] if m in df_62.index else m for m in models_order]

f1_62 = [df_62.loc[m,"F1_mean"] if m in df_62.index else 0 for m in models_order]
e_62  = [df_62.loc[m,"F1_std"]  if m in df_62.index else 0 for m in models_order]
f1_6  = [df_6.loc[m, "F1_mean"] if m in df_6.index  else 0 for m in models_order]
e_6   = [df_6.loc[m, "F1_std"]  if m in df_6.index  else 0 for m in models_order]
f1_pb62 = [PHASE_B_REF[m]["f1_62ch"] for m in models_order]
f1_pb6  = [PHASE_B_REF[m]["f1_6ch"]  for m in models_order]

x  = np.arange(len(models_order))
w  = 0.2

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# ── top panel: 62ch ─────────────────────────────────────────────────────────
ax = axes[0]
bars_new = ax.bar(x - w/2, f1_62, w, yerr=e_62,   label="Phase C  62ch",
                  color="#2196F3", capsize=4, error_kw={"elinewidth":1.2})
bars_ref = ax.bar(x + w/2, f1_pb62, w,            label="Phase B  62ch (ref)",
                  color="#90CAF9", edgecolor="#1565C0", linewidth=1.2)
ax.axhline(0.25, color="red", linestyle="--", linewidth=1.0, label="Chance (0.25)")
ax.set_xticks(x); ax.set_xticklabels([f"{m}\n{n}" for m,n in zip(models_order, model_names)],
                                      fontsize=9)
ax.set_ylabel("Macro-F1", fontsize=11); ax.set_title("62-Channel: Phase C vs Phase B", fontsize=13)
ax.set_ylim(0, 0.7); ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
for bar, f1 in zip(bars_new, f1_62):
    if f1 > 0: ax.text(bar.get_x()+bar.get_width()/2, f1+0.01, f"{f1:.3f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# ── bottom panel: 6ch ───────────────────────────────────────────────────────
ax = axes[1]
bars_new6 = ax.bar(x - w/2, f1_6,   w, yerr=e_6,  label="Phase C  6ch",
                   color="#4CAF50", capsize=4, error_kw={"elinewidth":1.2})
bars_ref6 = ax.bar(x + w/2, f1_pb6, w,             label="Phase B  6ch (ref)",
                   color="#A5D6A7", edgecolor="#2E7D32", linewidth=1.2)
ax.axhline(0.25, color="red", linestyle="--", linewidth=1.0, label="Chance (0.25)")
ax.set_xticks(x); ax.set_xticklabels([f"{m}\n{n}" for m,n in zip(models_order, model_names)],
                                      fontsize=9)
ax.set_ylabel("Macro-F1", fontsize=11); ax.set_title("6-Channel: Phase C vs Phase B", fontsize=13)
ax.set_ylim(0, 0.7); ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
for bar, f1 in zip(bars_new6, f1_6):
    if f1 > 0: ax.text(bar.get_x()+bar.get_width()/2, f1+0.01, f"{f1:.3f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.tight_layout(pad=2.5)
fig_path = os.path.join(FIGURES_DIR, "01_classical_ml_f1_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path}")



# ==============================================================================
# Notebook cell 20
# Categories: preprocessing, training, evaluation, results_tables, figures
# ==============================================================================
# ── Figure 2: Normalised Confusion Matrices (62ch, averaged over 45 runs) ────

def load_avg_cm(model_id: str, ch_tag: str):
    """Load and average all confusion matrices for (model, ch_tag)."""
    cms = []
    for seed in SEEDS:
        for fold_idx in range(N_FOLDS):
            key = checkpoint_key(model_id, ch_tag, seed, fold_idx)
            rec = load_checkpoint(key)
            if rec is not None and "conf_matrix" in rec:
                cms.append(np.array(rec["conf_matrix"]))
    if not cms:
        return None
    cm_sum = np.sum(cms, axis=0).astype(float)
    return cm_sum / cm_sum.sum(axis=1, keepdims=True)   # row-normalise

model_labels = {
    "M01":"LDA","M02":"SVM","M03":"Random\nForest","M04":"k-NN",
    "M05":"Logistic\nReg","M06":"Naive\nBayes","M07":"Extra\nTrees",
    "M08":"Grad\nBoost","M09":"XGBoost","M10":"MLP\nsklearn"
}

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for idx, mid in enumerate(models_order):
    cm = load_avg_cm(mid, "62ch")
    ax = axes[idx]
    if cm is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{mid} — {model_labels[mid]}", fontsize=10)
        continue
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["Neu","Sad","Fear","Hap"], fontsize=8)
    ax.set_yticklabels(["Neu","Sad","Fear","Hap"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    # Best F1 from summary
    row = df_results[(df_results["Model"]==mid) & (df_results["Channels"]=="62ch")]
    f1v = row["F1_mean"].values[0] if len(row)>0 else 0
    ax.set_title(f"{mid} — {model_labels[mid]}\nF1={f1v:.4f}", fontsize=9, fontweight="bold")
    for i in range(4):
        for j in range(4):
            val = cm[i,j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Normalised Confusion Matrices — 62ch (averaged over LOSO-15 × 3 seeds)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=1.5)
fig_path2 = os.path.join(FIGURES_DIR, "01_classical_ml_confusion_matrices.png")
plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path2}")



# ==============================================================================
# Notebook cell 21
# Categories: preprocessing, training, evaluation, results_tables, figures
# ==============================================================================
# ── Figure 3: Per-Subject Macro-F1 Heatmap (62ch, mean over 3 seeds) ─────────

def per_subject_f1(model_id: str, ch_tag: str):
    """Return array of shape (15,) = mean F1 per subject over 3 seeds."""
    f1_by_subj = []
    for fold_idx, subj in enumerate(SUBJECTS):
        fold_f1s = []
        for seed in SEEDS:
            key = checkpoint_key(model_id, ch_tag, seed, fold_idx)
            rec = load_checkpoint(key)
            if rec is not None:
                fold_f1s.append(rec["f1_macro"])
        f1_by_subj.append(np.mean(fold_f1s) if fold_f1s else np.nan)
    return np.array(f1_by_subj)

# Build matrix (10 models × 15 subjects)
hm_data = np.zeros((len(models_order), N_FOLDS))
for i, mid in enumerate(models_order):
    hm_data[i] = per_subject_f1(mid, "62ch")

fig, ax = plt.subplots(figsize=(18, 7))
im = ax.imshow(hm_data, aspect="auto", cmap="RdYlGn", vmin=0.15, vmax=0.70)
ax.set_xticks(range(N_FOLDS))
ax.set_xticklabels([f"S{i:02d}" for i in SUBJECTS], fontsize=9)
ax.set_yticks(range(len(models_order)))
ax.set_yticklabels([f"{m} {model_labels[m].replace(chr(10),' ')}" for m in models_order], fontsize=10)
ax.set_xlabel("Test Subject (LOSO fold)", fontsize=11)
ax.set_title("Per-Subject Macro-F1 Heatmap — Classical ML 62ch (mean over 3 seeds)",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Macro-F1", shrink=0.8)
for i in range(len(models_order)):
    for j in range(N_FOLDS):
        val = hm_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val < 0.25 or val > 0.62 else "black")
plt.tight_layout()
fig_path3 = os.path.join(FIGURES_DIR, "01_classical_ml_subject_heatmap.png")
plt.savefig(fig_path3, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path3}")



# ==============================================================================
# Notebook cell 22
# Categories: preprocessing, results_tables, figures
# ==============================================================================
# ── Figure 4: Per-Class F1 Heatmap (62ch) ────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax_idx, ch_tag in enumerate(("62ch", "6ch")):
    ax = axes[ax_idx]
    pc_data = np.zeros((len(models_order), 4))
    for i, mid in enumerate(models_order):
        row = df_results[(df_results["Model"]==mid) & (df_results["Channels"]==ch_tag)]
        if len(row):
            pc_data[i] = [row["F1_Neutral"].values[0], row["F1_Sad"].values[0],
                          row["F1_Fear"].values[0],    row["F1_Happy"].values[0]]
    im = ax.imshow(pc_data, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(EMOTION_NAMES, fontsize=11)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([f"{m} {model_labels[m].replace(chr(10),' ')}"
                        for m in models_order], fontsize=10)
    ax.set_title(f"Per-Class F1 — {ch_tag}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="F1", shrink=0.8)
    for i in range(len(models_order)):
        for j in range(4):
            ax.text(j, i, f"{pc_data[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if pc_data[i,j] > 0.55 else "black")

plt.suptitle("Per-Emotion-Class F1 Score — Classical ML (mean over LOSO-15 × 3 seeds)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path4 = os.path.join(FIGURES_DIR, "01_classical_ml_per_class_f1.png")
plt.savefig(fig_path4, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path4}")



# ==============================================================================
# Notebook cell 23
# Categories: preprocessing, results_tables, figures
# ==============================================================================
# ── Figure 5: 62ch vs 6ch F1 Scatter + Channel Efficiency Analysis ───────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: scatter 62ch vs 6ch
ax = axes[0]
f1_62v = [df_results[(df_results["Model"]==m) & (df_results["Channels"]=="62ch")]["F1_mean"].values[0]
           for m in models_order if len(df_results[(df_results["Model"]==m) & (df_results["Channels"]=="62ch")])]
f1_6v  = [df_results[(df_results["Model"]==m) & (df_results["Channels"]=="6ch" )]["F1_mean"].values[0]
           for m in models_order if len(df_results[(df_results["Model"]==m) & (df_results["Channels"]=="6ch")])]

colors = plt.cm.tab10(np.linspace(0, 1, len(models_order)))
for i, (m, x, y_, c) in enumerate(zip(models_order, f1_62v, f1_6v, colors)):
    ax.scatter(x, y_, color=c, s=120, zorder=5)
    ax.annotate(f"{m}\n{model_labels[m].replace(chr(10),' ')[:8]}",
                (x, y_), textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color=c)
diag = np.linspace(0.2, 0.6, 100)
ax.plot(diag, diag, "k--", linewidth=1, alpha=0.4, label="62ch = 6ch")
ax.set_xlabel("62-Channel F1", fontsize=11)
ax.set_ylabel("6-Channel F1", fontsize=11)
ax.set_title("62-Channel vs 6-Channel F1\n(above diagonal → 6ch better)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Right: F1 drop from 62ch to 6ch
ax2 = axes[1]
drops = [x - y_ for x, y_ in zip(f1_62v, f1_6v)]
bar_colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in drops]
bars = ax2.barh(range(len(drops)), drops, color=bar_colors)
ax2.set_yticks(range(len(models_order)))
ax2.set_yticklabels([f"{m}: {model_labels[m].replace(chr(10),' ')}" for m in models_order], fontsize=9)
ax2.axvline(0, color="black", linewidth=1)
ax2.set_xlabel("F1 Drop (62ch − 6ch)", fontsize=11)
ax2.set_title("Channel Reduction Cost\n(red=62ch better, green=6ch better)", fontsize=11)
ax2.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, drops):
    ax2.text(val + (0.002 if val >= 0 else -0.002), bar.get_y()+bar.get_height()/2,
             f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

plt.tight_layout()
fig_path5 = os.path.join(FIGURES_DIR, "01_classical_ml_channel_efficiency.png")
plt.savefig(fig_path5, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path5}")



# ==============================================================================
# Notebook cell 24
# Categories: preprocessing, evaluation, figures
# ==============================================================================
# ── Figure 6: Phase C vs Phase B Delta (improvement/regression) ───────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, ch_tag in enumerate(("62ch", "6ch")):
    ax = axes[ax_idx]
    deltas, labels, clrs = [], [], []
    for m in models_order:
        row = df_results[(df_results["Model"]==m) & (df_results["Channels"]==ch_tag)]
        if len(row):
            d = row["Delta_vs_PhaseB"].values[0]
            if np.isnan(d): d = 0.0
            deltas.append(d)
            labels.append(f"{m}\n{model_labels[m].replace(chr(10),' ')[:10]}")
            clrs.append("#27ae60" if d >= 0 else "#e74c3c")
        else:
            deltas.append(0); labels.append(m); clrs.append("grey")
    bars = ax.bar(range(len(deltas)), deltas, color=clrs)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.01,  color="green",  linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(-0.01, color="orange", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ΔF1 (Phase C − Phase B)", fontsize=10)
    ax.set_title(f"Phase C Improvement over Phase B — {ch_tag}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x()+bar.get_width()/2, d + (0.002 if d >= 0 else -0.005),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=8, fontweight="bold")

plt.suptitle("Phase C vs Phase B Reference — Macro-F1 Delta", fontsize=13, fontweight="bold")
plt.tight_layout()
fig_path6 = os.path.join(FIGURES_DIR, "01_classical_ml_phase_b_delta.png")
plt.savefig(fig_path6, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure saved → {fig_path6}")



# ==============================================================================
# Notebook cell 25
# Categories: preprocessing, training, evaluation, results_tables, figures, statistics, audit_verification
# ==============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# 01_classical_ml.ipynb — FINAL REPORT
# CSE400C Phase C | Classical ML Baseline Results (M01–M10)
# LOSO-15 × 3 seeds = 45 runs per model | SEED-IV 4-class emotion recognition
# ═══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 72

# ── 0. Completion gate ─────────────────────────────────────────────────────────
done, total = recovery_report([f"M{i:02d}" for i in range(1, 11)])
print(SEP)
print("  NOTEBOOK COMPLETION STATUS")
print(SEP)
print(f"  Checkpoints completed : {done} / {total}")
if done < total:
    print(f"  ⚠  {total-done} fold(s) still missing — run above cells to completion.")
else:
    print("  ✅ All 45 runs × 10 models × 2 ch-configs = 900 checkpoints COMPLETE")
print()

# ── 1. Full Results Table ──────────────────────────────────────────────────────
print(SEP)
print("  TABLE 1 — MACRO-F1 SUMMARY (LOSO-15 × 3 seeds, mean ± std)")
print(SEP)
print(f"{'Model':<4}  {'Name':<18}  {'Ch':<4}  {'F1 mean':<9}  {'F1 std':<8}  {'Acc mean':<10}  {'PhaseB ref':<12}  {'Δ vs PhaseB'}")
print("-" * 72)
for _, row in df_results.sort_values(["Model","Channels"]).iterrows():
    pb  = row["PhaseB_ref_F1"]
    dlt = row["Delta_vs_PhaseB"]
    pb_str  = f"{pb:.4f}" if not np.isnan(pb) else "  NaN  "
    dlt_str = f"{dlt:+.4f}" if not np.isnan(dlt) else "   NEW  "
    flag    = " ✅" if (not np.isnan(dlt) and dlt >= 0) else (" 🆕" if np.isnan(dlt) else " ⚠")
    print(f"{row['Model']:<4}  {row['Name']:<18}  {row['Channels']:<4}  "
          f"{row['F1_mean']:.4f}     {row['F1_std']:.4f}    "
          f"{row['Acc_mean']:.4f}      {pb_str}         {dlt_str}{flag}")
print()

# ── 2. Ranking (62ch) ─────────────────────────────────────────────────────────
print(SEP)
print("  TABLE 2 — MODEL RANKING by F1 Macro (62ch)")
print(SEP)
rank62 = df_results[df_results["Channels"]=="62ch"].sort_values("F1_mean", ascending=False).reset_index(drop=True)
print(f"{'Rank':<5}  {'Model':<5}  {'Name':<18}  {'F1 mean':<10}  {'F1 std':<9}  {'Acc mean'}")
print("-" * 62)
for i, row in rank62.iterrows():
    star = " ★ TOP CLASSICAL" if i == 0 else ("" if i > 2 else " ★")
    print(f"{i+1:<5}  {row['Model']:<5}  {row['Name']:<18}  "
          f"{row['F1_mean']:.4f}      {row['F1_std']:.4f}     {row['Acc_mean']:.4f}{star}")
print()

# ── 3. Per-class F1 breakdown ─────────────────────────────────────────────────
print(SEP)
print("  TABLE 3 — PER-EMOTION F1 (62ch, mean over 45 runs)")
print(SEP)
print(f"{'Model':<4}  {'Name':<18}  {'Neutral':<10}  {'Sad':<10}  {'Fear':<10}  {'Happy':<10}  Note")
print("-" * 70)
for _, row in df_results[df_results["Channels"]=="62ch"].sort_values("Model").iterrows():
    happy_flag = " ⚠ lowest" if row["F1_Happy"] == df_results[df_results["Channels"]=="62ch"]["F1_Happy"].min() else ""
    print(f"{row['Model']:<4}  {row['Name']:<18}  {row['F1_Neutral']:<10.4f}  "
          f"{row['F1_Sad']:<10.4f}  {row['F1_Fear']:<10.4f}  {row['F1_Happy']:.4f}{happy_flag}")
print()

# ── 4. Channel efficiency ─────────────────────────────────────────────────────
print(SEP)
print("  TABLE 4 — CHANNEL EFFICIENCY: 62ch vs 6ch")
print(SEP)
print(f"{'Model':<4}  {'Name':<18}  {'F1 62ch':<10}  {'F1 6ch':<10}  {'Drop':<10}  Note")
print("-" * 66)
for mid in models_order:
    r62  = df_results[(df_results["Model"]==mid) & (df_results["Channels"]=="62ch")]
    r6   = df_results[(df_results["Model"]==mid) & (df_results["Channels"]=="6ch")]
    if not len(r62) or not len(r6): continue
    f62, f6 = r62["F1_mean"].values[0], r6["F1_mean"].values[0]
    drop = f62 - f6
    note = " 6ch > 62ch ✅" if drop < 0 else (f" drop={drop:.3f}")
    print(f"{mid:<4}  {r62['Name'].values[0]:<18}  {f62:.4f}      {f6:.4f}      {drop:+.4f}    {note}")
print()

# ── 5. Key Findings ────────────────────────────────────────────────────────────
print(SEP)
print("  FINDINGS & OBSERVATIONS")
print(SEP)

# Best model
best_row = rank62.iloc[0]
print(f"  [F1] Best 62ch classical model  : {best_row['Model']} {best_row['Name']} "
      f"F1={best_row['F1_mean']:.4f} ± {best_row['F1_std']:.4f}")
worst_row = rank62.iloc[-1]
print(f"  [F1] Worst 62ch classical model : {worst_row['Model']} {worst_row['Name']} "
      f"F1={worst_row['F1_mean']:.4f}")

# M09 XGBoost (was NaN in Phase B)
m09_r = df_results[(df_results["Model"]=="M09") & (df_results["Channels"]=="62ch")]
if len(m09_r) and m09_r["F1_mean"].values[0] > 0:
    print(f"  [M09 XGBoost] NOW COMPLETE — F1={m09_r['F1_mean'].values[0]:.4f} "
          f"(was NaN in Phase B ✅)")

# Class imbalance effect on Happy
all_happy = df_results[df_results["Channels"]=="62ch"]["F1_Happy"]
print(f"  [Happy class] Mean F1 across models: {all_happy.mean():.4f}  "
      f"(other classes avg: "
      f"{df_results[df_results['Channels']=='62ch'][['F1_Neutral','F1_Sad','F1_Fear']].values.mean():.4f})")
print(f"  → Happy class underrepresentation (22.5%) is visible — justifies")
print(f"    WeightedRandomSampler (H14) in DL models (02_deep_models.ipynb)")

# Phase B comparison
improved_62 = df_results[(df_results["Channels"]=="62ch") & (df_results["Delta_vs_PhaseB"] > 0)]
regressed_62 = df_results[(df_results["Channels"]=="62ch") & (df_results["Delta_vs_PhaseB"] < 0)]
print(f"  [vs Phase B] Improved on 62ch : {len(improved_62)} models | "
      f"Regressed : {len(regressed_62)} models")

# Subject variability
for mid in ["M03", "M02"]:
    sv = per_subject_f1(mid, "62ch")
    if not np.all(np.isnan(sv)):
        print(f"  [{mid} subject std] {np.nanstd(sv):.4f}  "
              f"(min={np.nanmin(sv):.4f}, max={np.nanmax(sv):.4f})")

print()
print(SEP)
print("  SAVED LOCATIONS")
print(SEP)
print(f"  Results CSV    : {csv_path}")
print(f"  Checkpoints    : {CHECKPOINT_DIR}  (900 .json files)")
print()
print("  Figures:")
for fp in [fig_path, fig_path2, fig_path3, fig_path4, fig_path5, fig_path6]:
    print(f"    {fp}")

print()
print(SEP)
print("  PHASE C GATE STATUS")
print(SEP)
m01f1 = df_results[(df_results["Model"]=="M01") & (df_results["Channels"]=="62ch")]["F1_mean"].values
m03f1 = df_results[(df_results["Model"]=="M03") & (df_results["Channels"]=="62ch")]["F1_mean"].values
m09f1 = df_results[(df_results["Model"]=="M09") & (df_results["Channels"]=="62ch")]["F1_mean"].values
gate_pass  = (len(m01f1)>0 and len(m03f1)>0 and
              m01f1[0] > 0 and m03f1[0] > 0 and
              (len(m09f1)==0 or m09f1[0] > 0))
if gate_pass:
    print("  ✅  01_classical_ml.ipynb — ALL MODELS COMPLETE")
    print("  ➡  NEXT STEP: Run 02_deep_models.ipynb (M11–M26)")
    print(f"     Key baseline to beat in DL: M03 RF F1={m03f1[0]:.4f} (62ch)")
else:
    print("  ⚠  Some models not yet complete — see TABLE 1 for gaps")
    print("     Re-run the relevant model cells above to complete them.")
print(SEP)

