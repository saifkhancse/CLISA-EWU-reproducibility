# Reproduced manuscript tables

Generated: `2026-05-18_00-25-02`

## Audit table. Optional assets

Source: `results/final_tables/audit_discovered_optional_assets.csv`

| type | path | size kb |
| --- | --- | --- |
| per_class | results\final_tables\table_5_7_per_class_top_models.csv | 1.363 |
| tsne | features\seed_iv_tsne_raw_2000.npy | 15.750 |

## Audit table. Statistical tests

Source: `results/final_tables/audit_statistical_tests.csv`

| model | model mean | m25 mean | t stat | p value | significant | n paired |
| --- | --- | --- | --- | --- | --- | --- |
| CLISA | 0.6801 | 0.5190 | -16.306 | 0 | 1.000 | 45.000 |
| DANN | 0.6619 | 0.5190 | -12.757 | 0 | 1.000 | 45.000 |
| PseudoLabel | 0.6574 | 0.5190 | -12.673 | 0 | 1.000 | 45.000 |

## Table 5.1. Classical machine-learning benchmark

Source: `results/final_tables/table_5_1_classical_main.csv`

| model id | name | ch | F1 mean | F1 std | Acc mean | Acc std | f1 neutral | f1 sad | f1 fear | f1 happy | n runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | LDA | 62ch | 0.4047 | 0.1063 | 0.4081 | 0.1074 | 0.4633 | 0.3976 | 0.3581 | 0.3998 | 45.000 |
| M01 | LDA | 6ch | 0.4345 | 0.1095 | 0.4408 | 0.1078 | 0.5123 | 0.4218 | 0.3288 | 0.4752 | 45.000 |
| M02 | SVM (RBF) | 62ch | 0.4763 | 0.0959 | 0.4811 | 0.0961 | 0.5617 | 0.4841 | 0.3907 | 0.4687 | 45.000 |
| M02 | SVM (RBF) | 6ch | 0.3520 | 0.0718 | 0.3653 | 0.0722 | 0.3890 | 0.3598 | 0.2689 | 0.3904 | 45.000 |
| M03 | Random Forest | 62ch | 0.4648 | 0.1310 | 0.4702 | 0.1275 | 0.4715 | 0.4850 | 0.4186 | 0.4840 | 45.000 |
| M03 | Random Forest | 6ch | 0.4039 | 0.1016 | 0.4140 | 0.0978 | 0.4445 | 0.4272 | 0.3388 | 0.4051 | 45.000 |
| M04 | k-NN | 62ch | 0.3408 | 0.0601 | 0.3487 | 0.0582 | 0.3009 | 0.3728 | 0.3383 | 0.3510 | 45.000 |
| M04 | k-NN | 6ch | 0.3040 | 0.0475 | 0.3061 | 0.0483 | 0.2962 | 0.3028 | 0.2924 | 0.3247 | 45.000 |
| M05 | Logistic Regression | 62ch | 0.4148 | 0.1150 | 0.4233 | 0.1086 | 0.4627 | 0.4405 | 0.3698 | 0.3861 | 45.000 |
| M05 | Logistic Regression | 6ch | 0.4257 | 0.1030 | 0.4304 | 0.1015 | 0.4842 | 0.4160 | 0.3322 | 0.4706 | 45.000 |
| M06 | Naive Bayes | 62ch | 0.3791 | 0.1091 | 0.3986 | 0.1044 | 0.4173 | 0.2294 | 0.4348 | 0.4349 | 45.000 |
| M06 | Naive Bayes | 6ch | 0.3759 | 0.0868 | 0.3834 | 0.0812 | 0.4114 | 0.3232 | 0.3037 | 0.4653 | 45.000 |
| M07 | Extra Trees | 62ch | 0.4717 | 0.1250 | 0.4765 | 0.1230 | 0.4929 | 0.4926 | 0.4161 | 0.4851 | 45.000 |
| M07 | Extra Trees | 6ch | 0.4076 | 0.0977 | 0.4195 | 0.0935 | 0.4338 | 0.4440 | 0.3370 | 0.4155 | 45.000 |
| M08 | Gradient Boosting | 62ch | 0.4765 | 0.1298 | 0.4821 | 0.1299 | 0.5073 | 0.5080 | 0.4142 | 0.4767 | 45.000 |
| M08 | Gradient Boosting | 6ch | 0.4152 | 0.0871 | 0.4221 | 0.0859 | 0.4537 | 0.4134 | 0.3784 | 0.4151 | 45.000 |
| XGBoost | XGBoost | 62ch | 0.4791 | 0.1321 | 0.4844 | 0.1323 | 0.4933 | 0.5216 | 0.4153 | 0.4862 | 45.000 |
| XGBoost | XGBoost | 6ch | 0.4089 | 0.0919 | 0.4162 | 0.0893 | 0.4527 | 0.3969 | 0.3711 | 0.4149 | 45.000 |
| M10 | MLP (sklearn) | 62ch | 0.4790 | 0.1169 | 0.4825 | 0.1161 | 0.5406 | 0.4823 | 0.3949 | 0.4983 | 45.000 |
| M10 | MLP (sklearn) | 6ch | 0.3978 | 0.0830 | 0.4116 | 0.0816 | 0.4173 | 0.4142 | 0.3313 | 0.4282 | 45.000 |

## Table 5.2. Deep-learning and adaptation benchmark

Source: `results/final_tables/table_5_2_deep_main.csv`

| model id | name | ch | n runs | acc a mean | acc a std | f1 a mean | f1 a std | acc b mean | acc b std | f1 b mean | f1 b std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M11 | Shallow MLP | 62ch | 45.000 | 0.4765 | 0.1042 | 0.4739 | 0.1045 | 0.4961 | 0.0983 | 0.4930 | 0.0985 |
| M11 | Shallow MLP | 6ch | 45.000 | 0.4004 | 0.0875 | 0.3929 | 0.0859 | 0.4118 | 0.0870 | 0.4049 | 0.0859 |
| M12 | Deep MLP | 62ch | 45.000 | 0.4787 | 0.1098 | 0.4763 | 0.1097 | 0.4921 | 0.1023 | 0.4888 | 0.1015 |
| M12 | Deep MLP | 6ch | 45.000 | 0.3978 | 0.0843 | 0.3888 | 0.0844 | 0.4005 | 0.0786 | 0.3955 | 0.0789 |
| M13 | LSTM | 62ch | 45.000 | 0.4298 | 0.1174 | 0.4197 | 0.1280 | 0.4312 | 0.1179 | 0.4213 | 0.1284 |
| M13 | LSTM | 6ch | 45.000 | 0.3840 | 0.0713 | 0.3761 | 0.0721 | 0.3847 | 0.0702 | 0.3773 | 0.0705 |
| M14 | GRU | 62ch | 45.000 | 0.4390 | 0.1060 | 0.4310 | 0.1110 | 0.4411 | 0.1055 | 0.4339 | 0.1101 |
| M14 | GRU | 6ch | 45.000 | 0.3757 | 0.0779 | 0.3688 | 0.0796 | 0.3777 | 0.0768 | 0.3713 | 0.0785 |
| M15 | Conv1D | 62ch | 45.000 | 0.4303 | 0.1235 | 0.4192 | 0.1276 | 0.4455 | 0.1062 | 0.4380 | 0.1101 |
| M15 | Conv1D | 6ch | 45.000 | 0.3753 | 0.0750 | 0.3664 | 0.0784 | 0.3789 | 0.0788 | 0.3728 | 0.0803 |
| M16 | Vanilla Transformer | 62ch | 45.000 | 0.4047 | 0.0788 | 0.3929 | 0.0822 | 0.4068 | 0.0792 | 0.3952 | 0.0825 |
| M16 | Vanilla Transformer | 6ch | 45.000 | 0.3684 | 0.0665 | 0.3576 | 0.0678 | 0.3710 | 0.0664 | 0.3601 | 0.0679 |
| M17 | EEG Conformer | 62ch | 45.000 | 0.4528 | 0.1206 | 0.4467 | 0.1246 | 0.4594 | 0.1134 | 0.4544 | 0.1163 |
| M17 | EEG Conformer | 6ch | 45.000 | 0.3527 | 0.0743 | 0.3443 | 0.0790 | 0.3608 | 0.0740 | 0.3535 | 0.0784 |
| M18 | ChanDrop Transformer | 62ch | 45.000 | 0.4303 | 0.0744 | 0.4180 | 0.0802 | 0.4315 | 0.0745 | 0.4195 | 0.0800 |
| M18 | ChanDrop Transformer | 6ch | 45.000 | 0.3908 | 0.0626 | 0.3828 | 0.0635 | 0.3977 | 0.0628 | 0.3905 | 0.0635 |
| M19 | DANN | 62ch | 45.000 | 0.4584 | 0.1133 | 0.4547 | 0.1155 | 0.6619 | 0.0915 | 0.6543 | 0.0954 |
| M19 | DANN | 6ch | 45.000 | 0.3964 | 0.0920 | 0.3863 | 0.0917 | 0.5904 | 0.0731 | 0.5795 | 0.0791 |
| CLISA-EWU | CLISA | 62ch | 45.000 | 0.4845 | 0.1175 | 0.4797 | 0.1198 | 0.6801 | 0.0901 | 0.6742 | 0.0931 |
| CLISA-EWU | CLISA | 6ch | 45.000 | 0.4055 | 0.0749 | 0.3979 | 0.0761 | 0.6199 | 0.0744 | 0.6114 | 0.0770 |
| M21 | SimCLR | 62ch | 45.000 | 0.4114 | 0.0935 | 0.4059 | 0.0955 | 0.5745 | 0.0848 | 0.5568 | 0.0896 |
| M21 | SimCLR | 6ch | 45.000 | 0.4465 | 0.1067 | 0.4405 | 0.1092 | 0.6020 | 0.0972 | 0.5822 | 0.1084 |
| BYOL | BYOL | 62ch | 45.000 | 0.4308 | 0.0977 | 0.4225 | 0.1059 | 0.6089 | 0.0906 | 0.5944 | 0.0978 |
| BYOL | BYOL | 6ch | 45.000 | 0.4403 | 0.1038 | 0.4328 | 0.1043 | 0.6124 | 0.0928 | 0.5977 | 0.0974 |
| M23 | PseudoLabel | 62ch | 45.000 | 0.4682 | 0.1153 | 0.4656 | 0.1168 | 0.6574 | 0.0856 | 0.6538 | 0.0854 |
| M23 | PseudoLabel | 6ch | 45.000 | 0.4047 | 0.0947 | 0.3943 | 0.0952 | 0.6102 | 0.0843 | 0.5999 | 0.0888 |
| M24 | MixMatch | 62ch | 45.000 | 0.2853 | 0.0787 | 0.2429 | 0.0812 | 0.5350 | 0.1018 | 0.5217 | 0.1094 |
| M24 | MixMatch | 6ch | 45.000 | 0.2704 | 0.0704 | 0.2263 | 0.0654 | 0.3642 | 0.0746 | 0.3343 | 0.0698 |

## Table 5.3. DANCE LOSO verified results

Source: `results/final_tables/table_5_3_dance_loso_verified.csv`

| model id | name | ch | n runs | acc a mean | acc a std | f1 a mean | f1 a std | acc b mean | acc b std | f1 b mean | f1 b std | mean best val f1 | mean elapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DANCE Teacher | DANCE Teacher | 62ch | 45.000 | 0.4193 | 0.0934 | 0.4077 | 0.0943 | 0.5190 | 0.0914 | 0.5026 | 0.0973 | 0.5866 | 2009.9 |
| DANCE Student | DANCE Student | 6ch | 45.000 | 0.3901 | 0.0740 | 0.3775 | 0.0745 | 0.5706 | 0.0873 | 0.5412 | 0.1009 | 0.7102 | 264.0 |

## Table 5.3b. DANCE reproduction/caution table

Source: `results/final_tables/table_5_3b_dance_reproduction.csv`

| variant | model | acc a | balanced accuracy (AccB) | ref acc b | delta vs ref b |
| --- | --- | --- | --- | --- | --- |
| E00 | Teacher | 0.4436 | 0.6170 | 0.5913 | 0.0257 |
| H17 | Teacher | 0.4576 | 0.6080 | 0.5913 | 0.0167 |
| E00 | Student | 0.3866 | 0.5406 | 0.6918 | -0.1512 |
| H17 | Student | 0.3964 | 0.5444 | 0.6918 | -0.1474 |

## Table 5.4. Ablation study

Source: `results/final_tables/table_5_4_ablations.csv`

| ablation | acc A mean | acc A std | f1 A mean | f1 A std | acc B mean | acc B std | f1 B mean | f1 B std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A06 | 0.3540 | 0.0126 | 0.3469 | 0.0073 | 0.5340 | 0.0238 | 0.5307 | 0.0244 |
| A10 | 0.3932 | 0.0146 | 0.3737 | 0.0129 | 0.5307 | 0.0333 | 0.5249 | 0.0343 |
| A04 | 0.3616 | 0.0063 | 0.3542 | 0.0040 | 0.5296 | 0.0384 | 0.5290 | 0.0369 |
| A13 | 0.3616 | 0.0063 | 0.3542 | 0.0040 | 0.5296 | 0.0384 | 0.5290 | 0.0369 |
| A08 | 0.3616 | 0.0063 | 0.3542 | 0.0040 | 0.5296 | 0.0384 | 0.5290 | 0.0369 |
| A07 | 0.3616 | 0.0063 | 0.3542 | 0.0040 | 0.5296 | 0.0384 | 0.5290 | 0.0369 |
| A01 | 0.3923 | 0.0360 | 0.3801 | 0.0353 | 0.5274 | 0.0232 | 0.5232 | 0.0298 |
| A12 | 0.3753 | 0.0165 | 0.3679 | 0.0233 | 0.5254 | 0.0258 | 0.5253 | 0.0249 |
| A16 | 0.3961 | 0.0387 | 0.3769 | 0.0267 | 0.5242 | 0.0150 | 0.5164 | 0.0240 |
| A11 | 0.3484 | 0.0028 | 0.3395 | 0.0035 | 0.5227 | 0.0422 | 0.5204 | 0.0423 |
| A05 | 0.3724 | 0.0037 | 0.3631 | 0.0045 | 0.5175 | 0.0115 | 0.5103 | 0.0147 |

## Table 5.5. Proto-A versus Proto-B gain

Source: `results/final_tables/table_5_5_proto_gain.csv`

| model id | name | ch | delta acc | delta f1 |
| --- | --- | --- | --- | --- |
| M11 | Shallow MLP | 62ch | 0.0196 | 0.0191 |
| M11 | Shallow MLP | 6ch | 0.0114 | 0.0120 |
| M12 | Deep MLP | 62ch | 0.0134 | 0.0125 |
| M12 | Deep MLP | 6ch | 0.0028 | 0.0067 |
| M13 | LSTM | 62ch | 0.0014 | 0.0016 |
| M13 | LSTM | 6ch | 6.56e-04 | 0.0012 |
| M14 | GRU | 62ch | 0.0022 | 0.0028 |
| M14 | GRU | 6ch | 0.0020 | 0.0025 |
| M15 | Conv1D | 62ch | 0.0151 | 0.0187 |
| M15 | Conv1D | 6ch | 0.0036 | 0.0064 |
| M16 | Vanilla Transformer | 62ch | 0.0021 | 0.0023 |
| M16 | Vanilla Transformer | 6ch | 0.0025 | 0.0025 |
| M17 | EEG Conformer | 62ch | 0.0066 | 0.0077 |
| M17 | EEG Conformer | 6ch | 0.0081 | 0.0092 |
| M18 | ChanDrop Transformer | 62ch | 0.0012 | 0.0014 |
| M18 | ChanDrop Transformer | 6ch | 0.0069 | 0.0076 |
| M19 | DANN | 62ch | 0.2035 | 0.1995 |
| M19 | DANN | 6ch | 0.1940 | 0.1931 |
| CLISA-EWU | CLISA | 62ch | 0.1956 | 0.1945 |
| CLISA-EWU | CLISA | 6ch | 0.2145 | 0.2135 |
| M21 | SimCLR | 62ch | 0.1631 | 0.1509 |
| M21 | SimCLR | 6ch | 0.1555 | 0.1417 |
| BYOL | BYOL | 62ch | 0.1781 | 0.1719 |
| BYOL | BYOL | 6ch | 0.1721 | 0.1649 |
| M23 | PseudoLabel | 62ch | 0.1892 | 0.1882 |
| M23 | PseudoLabel | 6ch | 0.2055 | 0.2055 |
| M24 | MixMatch | 62ch | 0.2497 | 0.2788 |
| M24 | MixMatch | 6ch | 0.0939 | 0.1080 |
| DANCE Teacher | DANCE Teacher | 62ch | 0.0997 | 0.0950 |
| DANCE Student | DANCE Student | 6ch | 0.1805 | 0.1637 |

## Table 5.6. Channel-efficiency retention

Source: `results/final_tables/table_5_6_channel_efficiency.csv`

| model id | name | primary 62 | primary 6 | retention primary | family | paired distillation |
| --- | --- | --- | --- | --- | --- | --- |
| M01 | LDA | 0.4047 | 0.4345 | 1.074 | classical |  |
| M02 | SVM (RBF) | 0.4763 | 0.3520 | 0.7390 | classical |  |
| M03 | Random Forest | 0.4648 | 0.4039 | 0.8690 | classical |  |
| M04 | k-NN | 0.3408 | 0.3040 | 0.8920 | classical |  |
| M05 | Logistic Regression | 0.4148 | 0.4257 | 1.026 | classical |  |
| M06 | Naive Bayes | 0.3791 | 0.3759 | 0.9916 | classical |  |
| M07 | Extra Trees | 0.4717 | 0.4076 | 0.8641 | classical |  |
| M08 | Gradient Boosting | 0.4765 | 0.4152 | 0.8714 | classical |  |
| XGBoost | XGBoost | 0.4791 | 0.4089 | 0.8535 | classical |  |
| M10 | MLP (sklearn) | 0.4790 | 0.3978 | 0.8305 | classical |  |
| M11 | Shallow MLP | 0.4961 | 0.4118 | 0.8301 | deep |  |
| M12 | Deep MLP | 0.4921 | 0.4005 | 0.8139 | deep |  |
| M13 | LSTM | 0.4312 | 0.3847 | 0.8920 | deep |  |
| M14 | GRU | 0.4411 | 0.3777 | 0.8561 | deep |  |
| M15 | Conv1D | 0.4455 | 0.3789 | 0.8506 | deep |  |
| M16 | Vanilla Transformer | 0.4068 | 0.3710 | 0.9120 | deep |  |
| M17 | EEG Conformer | 0.4594 | 0.3608 | 0.7854 | deep |  |
| M18 | ChanDrop Transformer | 0.4315 | 0.3977 | 0.9217 | deep |  |
| M19 | DANN | 0.6619 | 0.5904 | 0.8920 | deep |  |
| CLISA-EWU | CLISA | 0.6801 | 0.6199 | 0.9116 | deep |  |
| M21 | SimCLR | 0.5745 | 0.6020 | 1.048 | deep |  |
| BYOL | BYOL | 0.6089 | 0.6124 | 1.006 | deep |  |
| M23 | PseudoLabel | 0.6574 | 0.6102 | 0.9282 | deep |  |
| M24 | MixMatch | 0.5350 | 0.3642 | 0.6808 | deep |  |
| DANCE Teacher->DANCE Student | DANCE Teacher->Student | 0.5190 | 0.5706 | 1.100 | dance | True |

## Table 5.7. Per-class top-model comparison

Source: `results/final_tables/table_5_7_per_class_top_models.csv`

| model id | name | ch | source | f1 neutral | f1 sad | f1 fear | f1 happy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | LDA | 62ch | classical_summary | 0.4633 | 0.3976 | 0.3581 | 0.3998 |
| M01 | LDA | 6ch | classical_summary | 0.5123 | 0.4218 | 0.3288 | 0.4752 |
| M02 | SVM (RBF) | 62ch | classical_summary | 0.5617 | 0.4841 | 0.3907 | 0.4687 |
| M02 | SVM (RBF) | 6ch | classical_summary | 0.3890 | 0.3598 | 0.2689 | 0.3904 |
| M03 | Random Forest | 62ch | classical_summary | 0.4715 | 0.4850 | 0.4186 | 0.4840 |
| M03 | Random Forest | 6ch | classical_summary | 0.4445 | 0.4272 | 0.3388 | 0.4051 |
| M04 | k-NN | 62ch | classical_summary | 0.3009 | 0.3728 | 0.3383 | 0.3510 |
| M04 | k-NN | 6ch | classical_summary | 0.2962 | 0.3028 | 0.2924 | 0.3247 |
| M05 | Logistic Regression | 62ch | classical_summary | 0.4627 | 0.4405 | 0.3698 | 0.3861 |
| M05 | Logistic Regression | 6ch | classical_summary | 0.4842 | 0.4160 | 0.3322 | 0.4706 |
| M06 | Naive Bayes | 62ch | classical_summary | 0.4173 | 0.2294 | 0.4348 | 0.4349 |
| M06 | Naive Bayes | 6ch | classical_summary | 0.4114 | 0.3232 | 0.3037 | 0.4653 |
| M07 | Extra Trees | 62ch | classical_summary | 0.4929 | 0.4926 | 0.4161 | 0.4851 |
| M07 | Extra Trees | 6ch | classical_summary | 0.4338 | 0.4440 | 0.3370 | 0.4155 |
| M08 | Gradient Boosting | 62ch | classical_summary | 0.5073 | 0.5080 | 0.4142 | 0.4767 |
| M08 | Gradient Boosting | 6ch | classical_summary | 0.4537 | 0.4134 | 0.3784 | 0.4151 |
| XGBoost | XGBoost | 62ch | classical_summary | 0.4933 | 0.5216 | 0.4153 | 0.4862 |
| XGBoost | XGBoost | 6ch | classical_summary | 0.4527 | 0.3969 | 0.3711 | 0.4149 |
| M10 | MLP (sklearn) | 62ch | classical_summary | 0.5406 | 0.4823 | 0.3949 | 0.4983 |
| M10 | MLP (sklearn) | 6ch | classical_summary | 0.4173 | 0.4142 | 0.3313 | 0.4282 |

## Audit table. Source mapping

Source: `results/final_tables/table_audit_sources.csv`

| output | source | role | status |
| --- | --- | --- | --- |
| table_5_1 | classical_ml_summary.csv | primary | 1.000 |
| table_5_2 | M11-M24 per-model CSVs + master | primary+audit | 1.000 |
| table_5_3 | M25_62ch + M26_6ch summaries | primary | 1.000 |
| table_5_3b | phaseB_reproduce_results.csv | primary | 1.000 |
| table_5_4 | ablations_partial_summary.csv | primary | 1.000 |
| table_5_5 | derived from table_5_2 + table_5_3 | derived | 1.000 |
| table_5_6 | derived from paired 62ch/6ch | derived | 1.000 |
| table_5_7 | classical_summary + optional deep | partial | 1.000 |
