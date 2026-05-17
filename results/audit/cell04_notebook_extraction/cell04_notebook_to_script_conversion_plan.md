# Cell 4 Notebook-to-Script Conversion Plan

Generated: `2026-05-17_23-35-17`

## Purpose

This report identifies which notebook code should be moved into clean repository scripts.

Raw notebook exports were created under:

`scripts/_raw_notebook_exports/`

These raw exports are for traceability only. They should not be treated as final clean code.

## Notebook summary

| notebook | num_code_cells | code_cells_with_outputs | raw_export | main_categories | defined_classes |
| --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 16 | 16 | scripts/_raw_notebook_exports/00_eda_seediv.py | preprocessing:15; figures:15; results_tables:11; model_definition:8; training:5; audit_verification:4; statistics:4; webapp_or_demo:2 |  |
| 00b_eda_faced.ipynb | 12 | 12 | scripts/_raw_notebook_exports/00b_eda_faced.py | figures:11; preprocessing:11; results_tables:6; model_definition:4; training:4; audit_verification:4; webapp_or_demo:1 |  |
| 00c_phaseB_reproduce.ipynb | 22 | 21 | scripts/_raw_notebook_exports/00c_phaseB_reproduce.py | model_definition:20; preprocessing:18; training:15; evaluation:10; results_tables:7; figures:6; audit_verification:4; webapp_or_demo:1; other:1 | InformationSeparatedAttention, TransformerBlock, DANCEEncoder, LabelSmoothCE |
| 01_classical_ml.ipynb | 25 | 24 | scripts/_raw_notebook_exports/01_classical_ml.py | preprocessing:14; results_tables:10; evaluation:9; figures:9; training:9; model_definition:7; audit_verification:4; other:3; statistics:1 |  |
| 02_deep_models.ipynb | 31 | 25 | scripts/_raw_notebook_exports/02_deep_models.py | preprocessing:29; training:29; model_definition:19; evaluation:15; results_tables:12; figures:7; audit_verification:6; statistics:2; webapp_or_demo:1 | ShallowMLP, DeepMLP, EEG_LSTM, EEG_GRU, EEG_Conv1D, VanillaTransformer, EEGConformer, ChanDropTransformer, GradReverse, DANN, CLISA, SimCLR, BYOLEn... |
| 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb | 24 | 17 | scripts/_raw_notebook_exports/03_5_and_04b_faced_models_kaggle_v2_Incomplete.py | preprocessing:20; training:14; model_definition:13; results_tables:12; evaluation:9; audit_verification:5; figures:4; other:2; webapp_or_demo:1 | EEGDataset, ShallowMLP, DeepMLP, LSTMModel, GRUModel, Conv1DModel, VanillaTransformer, EEGConformer, ChanDropTransformer, GradientReversalFn, DANN,... |
| 06-graph-former-v3.ipynb | 15 | 0 | scripts/_raw_notebook_exports/06_graph_former_v3.py | training:12; preprocessing:12; model_definition:11; results_tables:10; audit_verification:5; evaluation:4; webapp_or_demo:1 | GradientReversalFn, GRL, AdaptiveAdjacency, BandGCN, DiagMaskedTransformer, EEGGraphFormer, GraphStudent, EEGDataset |
| 08_final_stats_and_figures.ipynb | 25 | 25 | scripts/_raw_notebook_exports/08_final_stats_and_figures.py | preprocessing:23; results_tables:21; figures:20; evaluation:17; model_definition:16; training:9; audit_verification:6; statistics:2 |  |
| 08_final_stats_and_figures_v2.ipynb | 40 | 13 | scripts/_raw_notebook_exports/08_final_stats_and_figures_v2.py | preprocessing:31; figures:28; results_tables:27; model_definition:26; evaluation:16; audit_verification:11; training:7; statistics:2 |  |
| 09_final_results_rebuild_verified.ipynb | 46 | 44 | scripts/_raw_notebook_exports/09_final_results_rebuild_verified.py | audit_verification:41; results_tables:38; preprocessing:25; figures:17; evaluation:8; model_definition:7; statistics:2; training:2; other:1 |  |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 42 | 39 | scripts/_raw_notebook_exports/10_wearkd_seediv_calibrated_clip_loso_test.py | preprocessing:35; results_tables:32; evaluation:17; model_definition:15; training:11; figures:11; audit_verification:7; statistics:4; webapp_or_dem... | EEGWindowDataset, BandEmbedding, ChannelTransformerEncoder, ProjectionHead, EEGTeacherTransformer, WearableStudentTransformer, SupConLoss, LogitKDL... |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 43 | 40 | scripts/_raw_notebook_exports/10_wearkd_seediv_calibrated_clip_loso.py | preprocessing:36; results_tables:34; evaluation:18; model_definition:16; training:12; figures:11; statistics:6; audit_verification:5; webapp_or_dem... | EEGWindowDataset, BandEmbedding, ChannelTransformerEncoder, ProjectionHead, EEGTeacherTransformer, WearableStudentTransformer, SupConLoss, LogitKDL... |
| CSE400C_PhaseBC_Master.ipynb | 38 | 33 | scripts/_raw_notebook_exports/CSE400C_PhaseBC_Master.py | preprocessing:33; model_definition:29; results_tables:26; training:22; evaluation:21; figures:11; audit_verification:8; statistics:4; webapp_or_demo:2 | EEGAugmentor, InformationSeparatedAttention, DiagonalTransformerBlock, DANCEEncoder, DANCEModel, SEEDDataset, PairedDataset, CrossSubjectContrastiv... |
| dataset.ipynb | 2 | 2 | scripts/_raw_notebook_exports/dataset.py | other:1; preprocessing:1; audit_verification:1 |  |
| knowledge-distillation.ipynb | 1 | 0 | scripts/_raw_notebook_exports/knowledge_distillation.py | preprocessing:1; model_definition:1; training:1; evaluation:1; results_tables:1; audit_verification:1; webapp_or_demo:1 | CLISA |
| neurosync_data_setup.ipynb | 11 | 9 | scripts/_raw_notebook_exports/neurosync_data_setup.py | training:5; webapp_or_demo:5; preprocessing:5; results_tables:4; other:3; audit_verification:3; model_definition:1; figures:1 |  |
| save the pts.ipynb | 27 | 27 | scripts/_raw_notebook_exports/save_the_pts.py | training:22; model_definition:18; preprocessing:17; audit_verification:17; webapp_or_demo:13; results_tables:12; evaluation:5; other:1; figures:1; ... | CLISAClassifier, CLISA |
| seediv_raw_clisa_first_sota_fullrun.ipynb | 31 | 28 | scripts/_raw_notebook_exports/seediv_raw_clisa_first_sota_fullrun.py | preprocessing:30; results_tables:25; model_definition:18; evaluation:17; training:13; audit_verification:7; figures:4; statistics:1; webapp_or_demo... | FeatureWindowDataset, GradientReversalFunction, GRL, TinyCLISA, CLISARawMLP, ResidualBlock, CLISARawResidualMLP, CLISARawConformer, CLISASSTFallbac... |


## Category counts

| category | cell_count | target_script |
| --- | --- | --- |
| preprocessing | 356 | scripts/preprocess_seediv.py or scripts/preprocess_faced.py |
| results_tables | 288 | scripts/reproduce_tables.py |
| model_definition | 229 | src/clisa_ewu/models.py |
| training | 192 | scripts/train_clisa.py / scripts/train_baselines.py |
| evaluation | 167 | scripts/evaluate_loso.py |
| figures | 156 | scripts/reproduce_figures.py |
| audit_verification | 139 | scripts/verify_results.py |
| webapp_or_demo | 31 | separate webapp repository or docs/demo only |
| statistics | 29 | scripts/statistical_tests.py |
| other | 15 | manual review |


## preprocessing

**Target:** `scripts/preprocess_seediv.py or scripts/preprocess_faced.py`

**Number of matching code cells:** 356

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, dataset.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 1 | 33 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 3 | 38 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00_eda_seediv.ipynb | 4 | 124 | # ═══════════════════════════════════════════════════════════ | maybe_reshape |  | True |
| 00_eda_seediv.ipynb | 6 | 54 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 7 | 52 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 8 | 76 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 10 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 11 | 63 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 12 | 51 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 13 | 66 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 14 | 82 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 15 | 96 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 16 | 34 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 3 | 23 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00b_eda_faced.ipynb | 4 | 168 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 5 | 54 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 6 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |


## model_definition

**Target:** `src/clisa_ewu/models.py`

**Number of matching code cells:** 229

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 4 | 124 | # ═══════════════════════════════════════════════════════════ | maybe_reshape |  | True |
| 00_eda_seediv.ipynb | 8 | 76 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 10 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 14 | 82 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 15 | 96 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 16 | 34 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 4 | 168 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 6 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 12 | 35 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 1 | 20 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 2 | 24 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 3 | 102 | # ═══════════════════════════════════════════════════════════ | ckpt_exists, save_ckpt, load_ckpt, model_ckpt_path, save_model, load_model |  | True |
| 00c_phaseB_reproduce.ipynb | 4 | 36 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 5 | 52 | # ═══════════════════════════════════════════════════════════ | set_seed, make_fixed_split |  | False |
| 00c_phaseB_reproduce.ipynb | 6 | 142 | # ═══════════════════════════════════════════════════════════ | __init__, forward, __init__, forward, __init__, encode, project, classify, forward, build_teacher, build_student | InformationSeparatedAttention, TransformerBlock, DANCEEncoder | False |
| 00c_phaseB_reproduce.ipynb | 7 | 75 | # ═══════════════════════════════════════════════════════════ | augment, mixup, nt_xent_loss, __init__, forward, make_loader, tensors_from_split | LabelSmoothCE | False |
| 00c_phaseB_reproduce.ipynb | 8 | 230 | # ═══════════════════════════════════════════════════════════ | train_contrastive, train_supervised, train_distillation |  | True |


## training

**Target:** `scripts/train_clisa.py / scripts/train_baselines.py`

**Number of matching code cells:** 192

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 3 | 38 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00_eda_seediv.ipynb | 5 | 45 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 11 | 63 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 12 | 51 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 3 | 23 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00b_eda_faced.ipynb | 8 | 65 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 9 | 71 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00c_phaseB_reproduce.ipynb | 1 | 20 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 3 | 102 | # ═══════════════════════════════════════════════════════════ | ckpt_exists, save_ckpt, load_ckpt, model_ckpt_path, save_model, load_model |  | True |
| 00c_phaseB_reproduce.ipynb | 4 | 36 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 7 | 75 | # ═══════════════════════════════════════════════════════════ | augment, mixup, nt_xent_loss, __init__, forward, make_loader, tensors_from_split | LabelSmoothCE | False |
| 00c_phaseB_reproduce.ipynb | 8 | 230 | # ═══════════════════════════════════════════════════════════ | train_contrastive, train_supervised, train_distillation |  | True |
| 00c_phaseB_reproduce.ipynb | 9 | 90 | # ═══════════════════════════════════════════════════════════ | evaluate_proto_a, evaluate_proto_b, get_features |  | False |
| 00c_phaseB_reproduce.ipynb | 10 | 36 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00c_phaseB_reproduce.ipynb | 11 | 41 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00c_phaseB_reproduce.ipynb | 13 | 42 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 15 | 62 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 16 | 37 | # ═══════════════════════════════════════════════════════════ |  |  | True |


## evaluation

**Target:** `scripts/evaluate_loso.py`

**Number of matching code cells:** 167

**Notebooks involved:** 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, knowledge-distillation.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00c_phaseB_reproduce.ipynb | 3 | 102 | # ═══════════════════════════════════════════════════════════ | ckpt_exists, save_ckpt, load_ckpt, model_ckpt_path, save_model, load_model |  | True |
| 00c_phaseB_reproduce.ipynb | 8 | 230 | # ═══════════════════════════════════════════════════════════ | train_contrastive, train_supervised, train_distillation |  | True |
| 00c_phaseB_reproduce.ipynb | 9 | 90 | # ═══════════════════════════════════════════════════════════ | evaluate_proto_a, evaluate_proto_b, get_features |  | False |
| 00c_phaseB_reproduce.ipynb | 12 | 35 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 13 | 42 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 14 | 31 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 15 | 62 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 18 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 19 | 38 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 20 | 19 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 01_classical_ml.ipynb | 2 | 31 | import os, json, warnings, time |  |  | False |
| 01_classical_ml.ipynb | 6 | 72 | # ── 4. LOSO Utilities ──────────────────────────────────────────────────────────── | loso_split, compute_metrics, run_loso, summarise |  | False |
| 01_classical_ml.ipynb | 9 | 16 | # ── M02 — SVM (RBF kernel) ──────────────────────────────────────────────────── | build_svm |  | False |
| 01_classical_ml.ipynb | 11 | 16 | # ── M04 — k-Nearest Neighbours ───────────────────────────────────────────── | build_knn |  | False |
| 01_classical_ml.ipynb | 19 | 56 | # ── Figure 1: F1 Macro Comparison — 62ch vs 6ch ───────────────────────────── |  |  | True |
| 01_classical_ml.ipynb | 20 | 55 | # ── Figure 2: Normalised Confusion Matrices (62ch, averaged over 45 runs) ──── | load_avg_cm |  | True |
| 01_classical_ml.ipynb | 21 | 41 | # ── Figure 3: Per-Subject Macro-F1 Heatmap (62ch, mean over 3 seeds) ───────── | per_subject_f1 |  | True |
| 01_classical_ml.ipynb | 24 | 36 | # ── Figure 6: Phase C vs Phase B Delta (improvement/regression) ─────────────── |  |  | True |
| 01_classical_ml.ipynb | 25 | 147 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 02_deep_models.ipynb | 1 | 32 | # ═══════════════════════════════════════════════════════════════ |  |  | False |


## results_tables

**Target:** `scripts/reproduce_tables.py`

**Number of matching code cells:** 288

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 4 | 124 | # ═══════════════════════════════════════════════════════════ | maybe_reshape |  | True |
| 00_eda_seediv.ipynb | 6 | 54 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 7 | 52 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 8 | 76 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 10 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 13 | 66 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 14 | 82 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 15 | 96 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 16 | 34 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 4 | 168 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 7 | 49 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 8 | 65 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 10 | 84 | # ═══════════════════════════════════════════════════════════ | compute_mmd, rbf_kernel |  | True |
| 00b_eda_faced.ipynb | 12 | 35 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 3 | 102 | # ═══════════════════════════════════════════════════════════ | ckpt_exists, save_ckpt, load_ckpt, model_ckpt_path, save_model, load_model |  | True |
| 00c_phaseB_reproduce.ipynb | 5 | 52 | # ═══════════════════════════════════════════════════════════ | set_seed, make_fixed_split |  | False |
| 00c_phaseB_reproduce.ipynb | 6 | 142 | # ═══════════════════════════════════════════════════════════ | __init__, forward, __init__, forward, __init__, encode, project, classify, forward, build_teacher, build_student | InformationSeparatedAttention, TransformerBlock, DANCEEncoder | False |


## figures

**Target:** `scripts/reproduce_figures.py`

**Number of matching code cells:** 156

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 1 | 33 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 3 | 38 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00_eda_seediv.ipynb | 5 | 45 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 6 | 54 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 7 | 52 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 8 | 76 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 10 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 11 | 63 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 12 | 51 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 13 | 66 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 14 | 82 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 15 | 96 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 16 | 34 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 1 | 17 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 3 | 23 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00b_eda_faced.ipynb | 5 | 54 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 6 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |


## statistics

**Target:** `scripts/statistical_tests.py`

**Number of matching code cells:** 29

**Notebooks involved:** 00_eda_seediv.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 3 | 38 | # ═══════════════════════════════════════════════════════════ | eda_done, mark_done |  | False |
| 00_eda_seediv.ipynb | 7 | 52 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 13 | 66 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 01_classical_ml.ipynb | 25 | 147 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 02_deep_models.ipynb | 32 | 88 | # ═══════════════════════════════════════════════════════════════ | run_pseudolabel |  | False |
| 02_deep_models.ipynb | 44 | 301 | # ═══════════════════════════════════════════════════════════════════════════════ | split_clisa_best_fold, train_epoch_loss, val_epoch_loss_f1, train_clisa_best_fold, run_inference |  | True |
| 08_final_stats_and_figures.ipynb | 19 | 108 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 08_final_stats_and_figures.ipynb | 24 | 75 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 08_final_stats_and_figures_v2.ipynb | 33 | 85 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 08_final_stats_and_figures_v2.ipynb | 40 | 84 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 09_final_results_rebuild_verified.ipynb | 7 | 42 | # CELL 2.3 — Save output contract JSON |  |  | False |
| 09_final_results_rebuild_verified.ipynb | 56 | 856 | # # CELL X — MASTER REGEN: all tables + all figures + supplementary figures (separate files) |  |  | True |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 2 | 38 | import os, sys, json, time, shutil, random, re, warnings, math |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 40 | 98 | # ─── Inference & metrics ────────────────────────────────────────────────────── | predict_window_level, aggregate_clip_level, evaluate_window_metrics, evaluate_clip_metrics |  | False |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 48 | 80 | def compute_fold_stats(series: pd.Series): | compute_fold_stats, fold_bacc_series |  | True |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 55 | 61 | # ── Confusion matrices ──────────────────────────────────────────────────────── |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 2 | 38 | import os, sys, json, time, shutil, random, re, warnings, math |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 3 | 110 | # ─── MASTER CONFIGURATION ──────────────────────────────────────────────────── |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 40 | 98 | # ─── Inference & metrics ────────────────────────────────────────────────────── | predict_window_level, aggregate_clip_level, evaluate_window_metrics, evaluate_clip_metrics |  | False |


## audit_verification

**Target:** `scripts/verify_results.py`

**Number of matching code cells:** 139

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 08_final_stats_and_figures.ipynb, 08_final_stats_and_figures_v2.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, dataset.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 4 | 124 | # ═══════════════════════════════════════════════════════════ | maybe_reshape |  | True |
| 00_eda_seediv.ipynb | 9 | 55 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00_eda_seediv.ipynb | 11 | 63 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 4 | 168 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 6 | 67 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 7 | 49 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00b_eda_faced.ipynb | 9 | 71 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 00c_phaseB_reproduce.ipynb | 2 | 24 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 4 | 36 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 15 | 62 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 16 | 37 | # ═══════════════════════════════════════════════════════════ |  |  | True |
| 01_classical_ml.ipynb | 3 | 46 | # ── 1. Configuration ──────────────────────────────────────────────────────────── |  |  | False |
| 01_classical_ml.ipynb | 5 | 39 | # ── 3. Load SEED-IV Features ──────────────────────────────────────────────────── |  |  | True |
| 01_classical_ml.ipynb | 16 | 14 | # ── M09 — XGBoost ───────────────────────────────────────────────────────────── | build_xgb |  | False |
| 01_classical_ml.ipynb | 25 | 147 | # ═══════════════════════════════════════════════════════════════════════════════ |  |  | False |
| 02_deep_models.ipynb | 2 | 57 | # ═══════════════════════════════════════════════════════════════ | ck_path, ck_exists, ck_save, ck_load, model_complete, weight_path, set_seed |  | False |
| 02_deep_models.ipynb | 34 | 95 | # ═══════════════════════════════════════════════════════════════ | sharpen, run_mixmatch |  | False |
| 02_deep_models.ipynb | 38 | 129 | # ═══════════════════════════════════════════════════════════════ |  |  | True |
| 02_deep_models.ipynb | 46 | 211 | # ═══════════════════════════════════════════════════════════════════════════════ | wearable_load_ckpts, wearable_best_row, wearable_find_matching_row, wearable_mean_std, wearable_save, wearable_split, wearable_load_model, wearable... |  | True |


## webapp_or_demo

**Target:** `separate webapp repository or docs/demo only`

**Number of matching code cells:** 31

**Notebooks involved:** 00_eda_seediv.ipynb, 00b_eda_faced.ipynb, 00c_phaseB_reproduce.ipynb, 02_deep_models.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 06-graph-former-v3.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, CSE400C_PhaseBC_Master.ipynb, knowledge-distillation.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00_eda_seediv.ipynb | 1 | 33 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00_eda_seediv.ipynb | 2 | 39 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00b_eda_faced.ipynb | 2 | 28 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 00c_phaseB_reproduce.ipynb | 2 | 24 | # ═══════════════════════════════════════════════════════════ |  |  | False |
| 02_deep_models.ipynb | 1 | 32 | # ═══════════════════════════════════════════════════════════════ |  |  | False |
| 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb | 4 | 37 | import subprocess, sys |  |  | False |
| 06-graph-former-v3.ipynb | 4 | 56 | # ── Cell 2: Imports ──────────────────────────────────────────────────────────── | amp_ctx, make_scaler, seed_cpu_only |  | False |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 4 | 13 | # ─── REPRODUCIBILITY ───────────────────────────────────────────────────────── | set_seed |  | False |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 4 | 13 | # ─── REPRODUCIBILITY ───────────────────────────────────────────────────────── | set_seed |  | False |
| CSE400C_PhaseBC_Master.ipynb | 4 | 44 | # ── Global Seed ────────────────────────────────────────────────────────────── | set_global_seed |  | False |
| CSE400C_PhaseBC_Master.ipynb | 29 | 272 | # ── Generic DL training engine ──────────────────────────────────────────────── | train_dl_generic, build_dl_model, get_calibration_params, freeze_all_except_head, eval_dl |  | True |
| knowledge-distillation.ipynb | 0 | 316 | # -*- coding: utf-8 -*- | set_seed, weight_path, ck_path, ck_save, loso_split, make_loader, __init__, forward, project, nt_xent_loss, evaluate, proto_b_calibrate | CLISA | True |
| neurosync_data_setup.ipynb | 0 | 53 | from pathlib import Path | pick_state, detect_family |  | False |
| neurosync_data_setup.ipynb | 4 | 18 | import torch |  |  | False |
| neurosync_data_setup.ipynb | 5 | 38 | # ── CELL 2: CONFIGURE YOUR PATHS — edit these two lines only ───────────────── |  |  | False |
| neurosync_data_setup.ipynb | 8 | 43 | # ── CELL 5: Write manifest.json ──────────────────────────────────────────────── |  |  | False |
| neurosync_data_setup.ipynb | 9 | 56 | # ── CELL 6: Verify everything ────────────────────────────────────────────────── |  |  | False |
| save the pts.ipynb | 14 | 204 | # CLISA only: export/copy best .pth -> .pt into web app checkpoints dir | metric_from_json |  | True |
| save the pts.ipynb | 15 | 322 | from pathlib import Path | find_project_root, backup_file, __init__, forward, input_dim, build_clisa, detect_family, build_clisa, detect_family | CLISAClassifier | False |
| save the pts.ipynb | 16 | 89 | from pathlib import Path | find_compose_root |  | False |


## other

**Target:** `manual review`

**Number of matching code cells:** 15

**Notebooks involved:** 00c_phaseB_reproduce.ipynb, 01_classical_ml.ipynb, 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb, 09_final_results_rebuild_verified.ipynb, 10-wearkd-seediv-calibrated-clip-loso-test.ipynb, 10-wearkd-seediv-calibrated-clip-loso.ipynb, dataset.ipynb, neurosync_data_setup.ipynb, save the pts.ipynb, seediv_raw_clisa_first_sota_fullrun.ipynb

| notebook | cell_index | num_lines | first_line | defined_functions | defined_classes | has_save_call |
| --- | --- | --- | --- | --- | --- | --- |
| 00c_phaseB_reproduce.ipynb | 22 | 0 |  |  |  | False |
| 01_classical_ml.ipynb | 8 | 8 | # ── M01 — Linear Discriminant Analysis ───────────────────────────────────────── | build_lda |  | False |
| 01_classical_ml.ipynb | 12 | 9 | # ── M05 — Logistic Regression ──────────────────────────────────────────────── | build_lr |  | False |
| 01_classical_ml.ipynb | 13 | 7 | # ── M06 — Gaussian Naïve Bayes ─────────────────────────────────────────────── | build_nb |  | False |
| 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb | 22 | 12 | # ADD this at the top of Cell 11 (right after imports, before class EEGDataset) | maybe_autocast |  | False |
| 03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb | 46 | 0 |  |  |  | False |
| 09_final_results_rebuild_verified.ipynb | 57 | 0 |  |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 44 | 0 |  |  |  | False |
| 10-wearkd-seediv-calibrated-clip-loso.ipynb | 64 | 0 |  |  |  | False |
| dataset.ipynb | 0 | 12 | import os, shutil |  |  | False |
| neurosync_data_setup.ipynb | 2 | 3 | # ── CELL 1: Install dependencies ────────────────────────────────────────────── |  |  | False |
| neurosync_data_setup.ipynb | 10 | 0 |  |  |  | False |
| neurosync_data_setup.ipynb | 11 | 0 |  |  |  | False |
| save the pts.ipynb | 2 | 44 | # ============================================================ | looks_relevant |  | False |
| seediv_raw_clisa_first_sota_fullrun.ipynb | 56 | 0 |  |  |  | False |


## Recommended extraction order

1. `src/clisa_ewu/models.py` — extract model classes from `02_deep_models.ipynb` and related notebooks.
2. `scripts/evaluate_loso.py` — extract LOSO, Proto-A, Proto-B, balanced accuracy, macro-F1, confusion matrix, and per-class metrics.
3. `scripts/reproduce_tables.py` — extract final table-building logic from `09_final_results_rebuild_verified.ipynb` and `08_final_stats_and_figures_v2.ipynb`.
4. `scripts/reproduce_figures.py` — extract figure-generation code from final stats notebooks.
5. `scripts/preprocess_seediv.py` and `scripts/preprocess_faced.py` — extract dataset loading, DE feature formatting, channel mapping, and normalization logic.
6. `scripts/verify_results.py` — extract audit checks for BYOL, DANCE/M25, ablation rows, FACED shape, and variance-claim safety.

## Files still missing for final public repository

- Clean source code modules under `src/clisa_ewu/`.
- Real preprocessing scripts instead of placeholders.
- Real LOSO evaluation script instead of placeholder.
- Real table/figure reproduction scripts instead of placeholders.
- Data split JSON files under `data_splits/`.
- Exact dependency versions.
- License.
- Citation file, for example `CITATION.cff`.
