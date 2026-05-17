# Table 5.5. Proto-A versus Proto-B gain

Source file: `results/final_tables/table_5_5_proto_gain.csv`

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
