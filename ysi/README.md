# YSI (Yield Sooting Index) Model Training and Inference

This sub-project handles the neural network modeling, training, validation, and ONNX export for Yield Sooting Index (YSI) across diverse molecular components and fuel surrogates.

---

## 1. Mathematical Formulation & Architecture

Unlike ignition delay times or cetane numbers where blend interactions are nonlinear, mixture soot tendency follows an exact linear mole-fraction mixing law:

$$\text{YSI}_{\text{mix}} = \sum_{i=1}^{10} x_i \cdot \text{YSI}_i$$

where:
* $x_i$ is the mole fraction of component $i$ ($\sum_{i=1}^{10} x_i = 1.0$)
* $\text{YSI}_i$ is the pure-component Yield Sooting Index predicted directly from its SELFIES representation:

$$\text{YSI}_i = \text{Model}(\text{SELFIES}_i)$$

### Model Architecture
* **Input**: Padded integer token sequences representing SELFIES strings (`max_len = 65`).
* **Sequence Encoder**: 2-layer Bidirectional Recurrent Network (`BiGRU`, hidden dimension 128) with attentive pooling to summarize variable-length molecular tokens into a dense invariant chemical fingerprint.
* **Property Head**: 3-layer MLP with LayerNorm, GELU, and Dropout predicting $\log(1 + \text{YSI})$.
* **End-to-End ONNX Graph**:
  * Inputs: `component_tokens` `(batch, 10, 65)` and `mole_fracs` `(batch, 10)`.
  * Computes per-component values $\text{YSI}_i = \max(\exp(\hat{y}_i) - 1, 0)$.
  * Blends components linearly via inner product: $\text{YSI}_{\text{mix}} = \sum_{i=1}^{10} x_i \cdot \text{YSI}_i$.
  * Output: `ysi_mix` `(batch,)` in linear YSI units.

---

## 2. Directory Structure

```
ysi/
├── README.md                  # Documentation (this file)
├── inference.py               # Standalone CLI inference runner for pure components & mixtures
├── scripts/
│   └── train_ysi_dha_export.py  # Production training, CV validation, and ONNX export
└── models/
    └── ysi_dha/
        ├── vocab.json                 # 35-token extended SELFIES vocabulary
        ├── ysi_dha.onnx               # End-to-end ONNX model (opset 17, dynamic batch)
        ├── ysi_dha_meta.json          # Metrics and hyperparameter metadata
        ├── ysi_dha_component_parity.png # Out-of-fold pure component parity plot
        ├── ysi_dha_parity.png         # Mixture linear blend parity plot
        ├── ysi_predictor_seed0.pt     # Ensemble seed 0 weights
        ├── ysi_predictor_seed1.pt     # Ensemble seed 1 weights
        ├── ysi_predictor_seed2.pt     # Ensemble seed 2 weights
        ├── ysi_predictor_seed3.pt     # Ensemble seed 3 weights
        └── ysi_predictor_seed4.pt     # Ensemble seed 4 weights
```

---

## 3. Dataset & Vocabulary

* **Dataset Path**: `model_training/ysi_dha/ysi_mix_selfies_train.dat` (2,104,723 rows)
* **Pure Molecules**: 567 unique pure chemical compounds with measured ground-truth YSI spanning from 0.5 to 1,338.9.
* **Extended Vocabulary (35 tokens)**:
  Includes 12 new functional tokens present in the YSI space:
  `[#Branch1]`, `[#Branch2]`, `[#C]`, `[-/Ring1]`, `[=N]`, `[Br]`, `[C@@H1]`, `[C@H1]`, `[Cl]`, `[F]`, `[P]`, `[S]`.
* **Resource Guarding**: Data is streamed with minimal memory footprint (<100 MB RAM) and threads capped to 6 CPU cores to prevent throttling background workloads.

---

## 4. Benchmark Performance

| Metric | Component Level (567 Pure Compounds, OOF) | Mixture Level (30,000 Sampled Mixtures) |
| :--- | :---: | :---: |
| **MAE** | **25.56 YSI** | **18.86 YSI** |
| **Median Absolute Error** | **5.18 YSI** | **11.59 YSI** |
| **R² Score** | **0.9395** | **0.7145** |

*Note: For pure components, the best individual fold achieved an MAE of 14.22 YSI and $R^2 = 0.9805$. Because linear blending averages independent component errors, mixture predictions demonstrate substantial variance reduction.*

---

## 5. Quickstart & Inference

### Single Component Prediction
```bash
python ysi/inference.py --selfies "[C][#C][C][C][C][C][C][C]"
```
```
==================================================
Yield Sooting Index (YSI) Prediction
==================================================
Component SELFIES : [C][#C][C][C][C][C][C][C]
Predicted YSI     : 72.36
==================================================
```

### Multi-Component Mixture Prediction
```bash
python ysi/inference.py \
  --selfies "[C][#C][C][C][C][C][C][C]" "[C][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=C][Ring1][#Branch2]" \
  --mole_fracs 0.6 0.4
```
```
==================================================
Yield Sooting Index (YSI) Prediction
==================================================
Components (2):
  [ 1] x=0.6000 | [C][#C][C][C][C][C][C][C]
  [ 2] x=0.4000 | [C][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=C][Ring1][#Branch2]

Linear Mixture Predicted YSI_mix : 287.34
==================================================
```
