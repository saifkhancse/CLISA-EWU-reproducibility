# Table 5.6. Channel-efficiency retention

Source file: `results/final_tables/table_5_6_channel_efficiency.csv`

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
