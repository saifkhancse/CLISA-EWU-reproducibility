# Cell 2 Deep Audit Report

Generated: 2026-05-17_23-29-58

## What this cell inspected

- NPY/NPZ files inspected: 141
- CSV files inspected: 61
- JSON files inspected: 10628
- Notebooks inspected: 24
- Python files inspected: 23
- Duplicate small-file entries: 3094

## Repository decision counts

| Decision | Files | Size |
|---|---:|---:|
| REVIEW_KEEP | 10626 | 38.12 MB |
| EXCLUDE_OR_RELEASE | 2850 | 1.39 GB |
| EXCLUDE_DATA | 310 | 2.05 GB |
| REVIEW | 130 | 23.37 MB |
| KEEP_REVIEW | 117 | 29.70 MB |
| EXCLUDE_OR_DOCS_ONLY | 111 | 368.83 MB |
| OPTIONAL_SEPARATE_REPO | 79 | 858.03 KB |
| EXCLUDE | 33 | 198.89 MB |
| EXCLUDE_OR_ARCHIVE | 30 | 88.67 MB |
| KEEP_PROVENANCE | 16 | 6.63 MB |
| KEEP | 14 | 32.17 KB |

## Top NPY/NPZ files by size

| relative_path | size_human | shape | dtype | repo_action_hint |
| --- | --- | --- | --- | --- |
| C\features\faced_X_32ch.npy | 63.06 MB | (110208, 30, 5) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f1_noise.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f2_bandmask.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f3_chanmask.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f4_subject_mixup.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f5_combined.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f6_regionmask.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f7_magwarp.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_aug_f8_combined_m27.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f1_zscore_subject.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f3_minmax_subject.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f4_robust_subject.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f5_zscore_clip3.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f6_bandwise_zscore.npy | 63.06 MB | (110208, 150) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\backend_assets_clisa\data\features\seed_iv_X_62ch.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_X_62ch.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_X_62ch_norm.npy | 44.43 MB | (37575, 62, 5) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s1_noise.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s2_bandmask.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s3_chanmask.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s4_regionmask.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s5_subject_mixup.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s6_magwarp.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_aug_s7_combined.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s1_zscore_subject.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s3_minmax_subject.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s4_robust_subject.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s5_zscore_clip3.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s6_bandwise_zscore.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\seed-iv-clisa-data\features\seed_iv_X_62ch.npy | 44.43 MB | (37575, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\faced_prep_f1_zscore_subject_6ch.npy | 12.61 MB | (110208, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\backend_assets_clisa\data\features\seed_iv_X_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_X_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s1_zscore_subject_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s3_minmax_subject_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s4_robust_subject_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\preprocessed\seed_iv_prep_s5_zscore_clip3_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\seed-iv-clisa-data\features\seed_iv_X_6ch.npy | 4.30 MB | (37575, 30) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\faced_clips.npy | 861.12 KB | (110208,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\faced_subjects.npy | 861.12 KB | (110208,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\faced_y_4cls.npy | 861.12 KB | (110208,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\backend_assets_clisa\data\features\seed_iv_subjects.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\backend_assets_clisa\data\features\seed_iv_y_4cls.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_session.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_subjects.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_y_4cls.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\seed-iv-clisa-data\features\seed_iv_subjects.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\seed-iv-clisa-data\features\seed_iv_y_4cls.npy | 293.68 KB | (37575,) | int64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\features\seed_iv_channel_correlation.npy | 30.16 KB | (62, 62) | float64 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub001_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub001_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub001_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub002_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub002_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub002_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub003_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub003_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub003_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub004_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub004_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub004_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub005_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub005_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub005_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub006_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub006_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub006_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub007_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub007_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub007_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub008_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub008_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub008_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub009_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub009_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub009_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub010_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub010_ses02.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub010_ses03.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |
| C\webapp\neurosync_webapp\neurosync\data\processed\sub011_ses01.npy | 29.19 KB | (24, 310) | float32 | EXCLUDE_FROM_GITHUB_OR_USE_RELEASE/LFS_IF_LICENSE_ALLOWS |

## Important notebook audit

| relative_path | num_cells | num_code_cells | has_outputs | defined_functions | defined_classes | repo_action_hint |
| --- | --- | --- | --- | --- | --- | --- |
| C\08_final_stats_and_figures.ipynb | 26 | 25 | True | ck_path_seediv, ck_path_faced, load_json, aggregate_seediv, aggregate_faced, try_load_csv, get_deep_results, save_fig, add_value_labels, ... |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\CSE400C_PhaseBC_Master.ipynb | 53 | 38 | True | set_global_seed, load_seed_iv_lds, apply_subject_zscore, __init__, mixup, channel_mask, add_noise, forward, _sinusoidal, forward_contrast... | EEGAugmentor, InformationSeparatedAttention, DiagonalTransformerBlock, DANCEEncoder, DANCEModel, SEEDDataset, PairedDataset, CrossSubject... | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\save the pts.ipynb | 27 | 27 | True | find_dir_by_name, count_files, recursive_find, try_load_state_dict, fmt_shape, looks_relevant, sha1_first_mb, is_relevant_pth, quick_hash... | CLISAClassifier, CLISA | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\neurosync_data_setup.ipynb | 12 | 11 | True | pick_state, detect_family |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| Journal\10-wearkd-seediv-calibrated-clip-loso.ipynb | 65 | 43 | True | set_seed, elapsed_hours, should_stop_for_time_limit, save_json, load_json, copy_previous_run_if_available, save_figure, load_seediv_de_fe... | EEGWindowDataset, BandEmbedding, ChannelTransformerEncoder, ProjectionHead, EEGTeacherTransformer, WearableStudentTransformer, SupConLoss... | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| Journal\seediv_raw_clisa_first_sota_fullrun.ipynb | 57 | 31 | True | set_seed, mem_stats, free_memory, config_hash, bandpass_filter, apply_notch, compute_de, compute_psd_bandpower, extract_features_from_tri... | FeatureWindowDataset, GradientReversalFunction, GRL, TinyCLISA, CLISARawMLP, ResidualBlock, CLISARawResidualMLP, CLISARawConformer, CLISA... | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\08_final_stats_and_figures_v2.ipynb | 41 | 40 | True | ck_path_seediv, ck_path_faced, load_json, aggregate_seediv, aggregate_faced, try_load_csv, get_deep_results, save_fig, add_value_labels, ... |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\02_deep_models.ipynb | 49 | 31 | True | ck_path, ck_exists, ck_save, ck_load, model_complete, weight_path, set_seed, get_data, loso_split, make_loader, train_epoch, evaluate, pr... | ShallowMLP, DeepMLP, EEG_LSTM, EEG_GRU, EEG_Conv1D, VanillaTransformer, EEGConformer, ChanDropTransformer, GradReverse, DANN, CLISA, SimC... | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\03-5-and-04b-faced-models-kaggle-v2_Incomplete.ipynb | 47 | 24 | True | ckpt_key, save_ckpt, load_ckpt, ckpt_exists, model_done_count, prep_done, mark_prep_done, save_prep, session_status, _load, get_faced_fol... | EEGDataset, ShallowMLP, DeepMLP, LSTMModel, GRUModel, Conv1DModel, VanillaTransformer, EEGConformer, ChanDropTransformer, GradientReversa... | REVIEW_NOTEBOOK |
| C\01_classical_ml.ipynb | 26 | 25 | True | ensure, checkpoint_key, load_checkpoint, save_checkpoint, cvt, recovery_report, loso_split, compute_metrics, run_loso, summarise, run_mod... |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\03-5-and-04b-faced-models-kaggle-v2.ipynb | 47 | 24 | True | ckpt_key, save_ckpt, load_ckpt, ckpt_exists, model_done_count, prep_done, mark_prep_done, save_prep, session_status, _load, get_faced_fol... | EEGDataset, ShallowMLP, DeepMLP, LSTMModel, GRUModel, Conv1DModel, VanillaTransformer, EEGConformer, ChanDropTransformer, GradientReversa... | REVIEW_NOTEBOOK |
| Journal\10-wearkd-seediv-calibrated-clip-loso-test.ipynb | 64 | 42 | True | set_seed, elapsed_hours, should_stop_for_time_limit, save_json, load_json, copy_previous_run_if_available, save_figure, load_seediv_de_fe... | EEGWindowDataset, BandEmbedding, ChannelTransformerEncoder, ProjectionHead, EEGTeacherTransformer, WearableStudentTransformer, SupConLoss... | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\09_final_results_rebuild_verified.ipynb | 58 | 46 | True | log_audit, safe_numeric, compute_mean_std, standardize_model_id, standardize_channel_label, get_model_name, save_figure, save_table |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| Journal\seed-iv-dataset-explore.ipynb | 6 | 6 | True | quick_file_signature, list_mat_files, parse_subject_date_session, inspect_mat_file, parse_subject_file, mat_var_shapes, get_trial_shape, ... |  | REVIEW_NOTEBOOK |
| C\Untitled.ipynb | 12 | 12 | True | load_notebook, join_cells, extract_model_block, extract_model_lines, preview_text_file |  | REVIEW_NOTEBOOK |
| C\00c_phaseB_reproduce.ipynb | 23 | 22 | True | ckpt_exists, save_ckpt, load_ckpt, model_ckpt_path, save_model, load_model, set_seed, make_fixed_split, __init__, forward, encode, projec... | InformationSeparatedAttention, TransformerBlock, DANCEEncoder, LabelSmoothCE | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\00_eda_seediv.ipynb | 17 | 16 | True | eda_done, mark_done, maybe_reshape |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\06-graph-former-v3.ipynb | 17 | 15 | False | run, sh, get_gpu_name, pip_show_version, amp_ctx, make_scaler, seed_cpu_only, ckpt_key, ckpt_path, teacher_pth, save_ckpt, load_ckpt, ckp... | GradientReversalFn, GRL, AdaptiveAdjacency, BandGCN, DiagMaskedTransformer, EEGGraphFormer, GraphStudent, EEGDataset | REVIEW_NOTEBOOK |
| C\00b_eda_faced.ipynb | 13 | 12 | True | eda_done, mark_done, compute_mmd, rbf_kernel |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\dataset.ipynb | 2 | 2 | True |  |  | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |
| C\knowledge-distillation.ipynb | 1 | 1 | False | set_seed, weight_path, ck_path, ck_save, loso_split, make_loader, __init__, forward, project, nt_xent_loss, evaluate, proto_b_calibrate | CLISA | KEEP_AS_PROVENANCE_THEN_CLEAN_OR_CONVERT |

## Python file audit

| relative_path | defined_functions | defined_classes | repo_action_hint |
| --- | --- | --- | --- |
| misc\notebockgenB.py | md, code, add, set_global_seed, load_seed_iv_lds, apply_subject_zscore, __init__, mixup, channel_mask, add_noise, forward, _sinusoidal, f... | EEGAugmentor, InformationSeparatedAttention, DiagonalTransformerBlock, DANCEEncoder, DANCEModel, SEEDDataset, PairedDataset, CrossSubject... | LIKELY_EXCLUDE_OR_REVIEW |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\model_loader.py | _load_ch_map, _resolve_model_config, to_dict, __init__, _load_and_verify, _build_model, _load_norm_stats, _get_real_test_sample, _adapt_i... | VerificationStatus, NeuroSyncModel | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\inference.py |  |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\device.py | _make_default_state, get_device_state, set_device_state, _apply_stability_settings, _connect_simulator, _connect_real | ConnectDeviceRequest, StabilitySettingsRequest, DeviceStateResponse | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\eeg_simulator.py | __init__, update_config, apply_preset, generate_de_features, generate_raw_waveform, infer_emotion, _prototype_fallback, get_band_powers, ... | SimulatorConfig, AugmentationPreset, EEGSimulator | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\dataset_loader.py | load_manifest, list_processed_files, get_dataset_status, load_features, load_norm_stats, run_validation | DatasetManifest, ValidationResult | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\tracking.py | _calculate_wellness_score, _calculate_streak |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\dance_model.py | __init__, forward, input_dim, detect_family, build_dance_student, build_eeg_band_transformer, build_clisa | ChannelAttention, DANCEEncoder, DANCEStudent, _AttentionWrapper, _BandTransformerBlock, EEGBandTransformer, CLISAClassifier | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\calibration.py |  | StartCalibrationResponse, AddWindowRequest, CalibrationStatusResponse, CalibrationRecord_, Config | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\inference_stabilizer.py | __init__, update, reset, _accept, _build_output, compute_margin | InferenceStabilizer | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\sessions.py |  | SessionResponse, Config, StartSessionRequest | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\auth.py | username_valid, password_valid, new_password_valid | RegisterRequest, TokenResponse, UserResponse, Config, UpdateProfileRequest, ChangePasswordRequest | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\model_management.py |  | LoadModelRequest | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\main.py |  |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\dataset.py |  | ValidateRequest | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\models\session.py |  | EEGSession, EmotionRecord, CalibrationRecord, DailyEmotionSummary | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\auth_service.py | verify_password, get_password_hash, create_access_token, decode_access_token |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\models\user.py |  | User | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\database.py |  | Base | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\config.py | get_settings | Settings, Config | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\routers\__init__.py |  |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\models\__init__.py |  |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |
| C\webapp\neurosync_webapp\neurosync\backend\app\services\__init__.py |  |  | OPTIONAL_WEBAPP_REPO_NOT_MAIN_METHOD_REPO |

## Final table CSV schemas

| relative_path | num_rows | num_columns | columns | repo_action_hint |
| --- | --- | --- | --- | --- |
| C\results\final_tables\table_5_2_deep_main.csv | 28 | 12 | model_id, name, ch, n_runs, acc_a_mean, acc_a_std, f1_a_mean, f1_a_std, acc_b_mean, acc_b_std, f1_b_mean, f1_b_std | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_6_channel_efficiency.csv | 25 | 7 | model_id, name, primary_62, primary_6, retention_primary, family, paired_distillation | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_5_proto_gain.csv | 30 | 5 | model_id, name, ch, delta_acc, delta_f1 | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_4_ablations.csv | 11 | 9 | ablation, acc_A_mean, acc_A_std, f1_A_mean, f1_A_std, acc_B_mean, acc_B_std, f1_B_mean, f1_B_std | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_1_classical_main.csv | 20 | 12 | model_id, name, ch, F1_mean, F1_std, Acc_mean, Acc_std, f1_neutral, f1_sad, f1_fear, f1_happy, n_runs | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_7_per_class_top_models.csv | 20 | 8 | model_id, name, ch, source, f1_neutral, f1_sad, f1_fear, f1_happy | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_3_dance_loso_verified.csv | 2 | 14 | model_id, name, ch, n_runs, acc_a_mean, acc_a_std, f1_a_mean, f1_a_std, acc_b_mean, acc_b_std, f1_b_mean, f1_b_std, mean_best_val_f1, mea... | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_audit_sources.csv | 8 | 4 | output, source, role, status | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\table_5_3b_dance_reproduction.csv | 4 | 6 | variant, model, acc_a, acc_b, ref_acc_b, delta_vs_ref_b | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\audit_statistical_tests.csv | 3 | 7 | model, model_mean, m25_mean, t_stat, p_value, significant, n_paired | KEEP_RESULT_PROVENANCE |
| C\results\final_tables\audit_discovered_optional_assets.csv | 2 | 3 | type, path, size_kb | KEEP_RESULT_PROVENANCE |
