# PM-KMNet: Physics-guided Mamba-KAN Hybrid Network

[![Paper](https://img.shields.io/badge/Advanced%20Engineering%20Informatics-10.1016%2Fj.aei.2026.104545-blue)](https://doi.org/10.1016/j.aei.2026.104545)

Official research code and de-identified data accompanying the paper **“Physics-guided Mamba-KAN Hybrid Network (PM-KMNet): Robust prognostics for ship propulsion systems under distribution shifts”**, published in *Advanced Engineering Informatics* (2026).

PM-KMNet predicts ship main-engine thrust-bearing pad temperature under changing voyage and vessel conditions. It combines an interpretable thermal recurrence with a data-driven residual model so that physical structure governs the prediction while learned dynamics compensate for unmodeled effects.

## Method at a glance

- **Physics meta-learner:** a compact KAN estimates operating-condition-dependent heat-generation, cooling, and thermal-inertia parameters.
- **Residual compensator:** Mamba models vibration, hydraulic, and other residual dynamics that are not captured by the simplified thermal balance.
- **Explicit integration:** an Euler recurrence converts the learned energy terms into a temperature trajectory.
- **Progressive optimization:** three phases prioritize physics learning, calibrate the residual branch, and then jointly fine-tune the full model.

In the paper's main cross-voyage test, PM-KMNet achieved an RMSE of **0.667 °C**, an MAE of **0.527 °C**, and an R² of **0.9656**. Please consult the paper for the complete protocol, baselines, ablations, uncertainty analysis, and cross-vessel results.

## Repository contents

| Path | Description |
| --- | --- |
| `PMK.py` | Complete training, ablation, evaluation, and visualization script for PM-KMNet. |
| `Dataset_A.csv` | Primary de-identified dataset included with the release. |
| `Dataset_B.csv` | Primary de-identified dataset included with the release. |
| `Voyage_3.csv`–`Voyage_8.csv` | Additional de-identified voyage datasets used for cross-voyage and cross-vessel evaluation. |

## Environment

Python 3.10 and a CUDA-capable Linux environment are recommended for the full Mamba configuration.

```bash
pip install -r requirements.txt
pip install mamba-ssm
```

`mamba-ssm` is required to reproduce the proposed Mamba branch. If it is unavailable, `PMK.py` automatically falls back to a smaller GRU implementation; that fallback is an ablation-compatible substitute, not the full PM-KMNet architecture reported in the paper.

## Running the released script

The script retains the directory layout used in the research environment. Before running it:

1. Open `PMK.py` and set `Config.base_dir` to a writable local directory.
2. Set `Config.path_train` and `Config.path_test`, or provide the expected file names `Voyage1_Train.csv` and `Voyage2_Test.csv` under `base_dir`.
3. Confirm that the selected CSV files contain the feature names listed below.
4. Run:

```bash
python PMK.py
```

The default entry point trains the proposed model and five ablation variants, evaluates them on the test voyage, and writes checkpoints, metrics, predictions, figures, and a LaTeX table to `Config.save_dir`. This is substantially more expensive than a single-model training run.

The published main experiment uses a chronological 70/30 train/validation split from the first voyage and evaluates on the second voyage. The default sequence length is 50, training uses a stride of 5, and evaluation uses a stride of 1.

## Dataset / 数据集

The repository contains normalized or desensitized sensor data from two large vessels, **Yuan Fu Yang (远福洋)** and **Yuan Bei Hai (远北海)**. Voyage data are released without absolute timestamps, GPS coordinates, or MMSI identifiers.

本仓库包含远福洋与远北海两艘船舶的归一化或脱敏主机传感器数据。数据不包含绝对时间、GPS 坐标或 MMSI 标识。

Each CSV contains the following fields:

| Variable | Meaning | Model role |
| --- | --- | --- |
| `step` | Relative sequence index | Index |
| `Me1Thrust_bearing_pad_temp` | Main-engine thrust-bearing pad temperature | Prediction target |
| `Me1Nms` | Main-engine speed | Dynamic input |
| `Me1Load` | Main-engine load | Dynamic input |
| `Me1Axial_vib` | Axial vibration | Dynamic input |
| `Me1Main_brg_lo_in_press` | Main-bearing lubricating-oil inlet pressure | Dynamic input |
| `Me1Main_brg_lo_in_temp` | Main-bearing lubricating-oil inlet temperature | Dynamic input |
| `Me1Sw_com_temp` | Seawater/common inlet temperature | Context input |
| `Df` | Forward draft | Context input |
| `Da` | Aft draft | Context input |
| `Rudder` | Rudder angle | Context input |
| `TrueVwr` | True wind speed | Context input |
| `EeIndex8` | Slip-ratio/efficiency-related operating index | Context input |

For consistent processing across vessels, the seawater-temperature field is unified as `Me1Sw_com_temp`. Columns unavailable because of sensor failure are zero-filled in the released files. When substituting other data, preserve the column names and review the fixed normalization constants in `Config` and `DataProcessor`.

## Citation

If this code or dataset supports your work, please cite:

```bibtex
@article{zhang2026physics,
  title   = {Physics-guided Mamba-KAN Hybrid Network (PM-KMNet): Robust prognostics for ship propulsion systems under distribution shifts},
  author  = {Zhang, Meng and Liu, Jilong and Han, Bing and Dong, Shengli and Cui, Tong and Ren, Yan},
  journal = {Advanced Engineering Informatics},
  volume  = {73},
  pages   = {104545},
  year    = {2026},
  doi     = {10.1016/j.aei.2026.104545}
}
```

GitHub also exposes the same metadata through [`CITATION.cff`](CITATION.cff).

## Contribution and maintenance

The published CRediT statement identifies **Jilong Liu** with conceptualization, methodology, software, formal analysis, validation, visualization, resources, project administration, original-draft preparation, and review/editing contributions.

Jilong Liu maintains this repository and is currently pursuing a Ph.D. at Northeastern University, China.

Contact: [liujilong@mails.neu.edu.cn](mailto:liujilong@mails.neu.edu.cn)

## Responsible use

The released datasets are intended for research and educational use. They are de-identified and should not be used to infer vessel identity, reconstruct routes, or make safety-critical operational decisions without independent validation.


