# SIDT Model Training and Inference

This sub-project handles training, export, and inference for Soot/Similar Ignition Delay Time (SIDT) models as well as Negative Temperature Coefficient (NTC) bounds models.

Each compound dataset is trained to produce:
1. **Forward IDT Model**: Predicts ignition delay time (`idt_s` in seconds).
2. **Inverse Condition Models**: Predicts 1 physical condition (`pressure`, `temperature`, `phi`, or `egr_fraction`) from remaining inputs + IDT.
3. **NTC Bounds Models**: Predicts NTC region presence (`has_ntc`: 0/1) and bounds (`ntc_t_min_K`, `ntc_t_max_K`).

---

## 1. Directory Structure

```
sidt/
├── README.md                  # Documentation (this file)
├── inference.py               # Unified forward/inverse/ntc CLI inference runner
├── scripts/
│   ├── train_export.py                    # Parameterized training & ONNX export (Forward/Inverse IDT)
│   ├── train_ntc_export.py                # Training & ONNX export for NTC bounds
│   ├── train_natgas_mix_export.py         # 6-component natural gas mixture model
│   ├── train_gasoline_mix_export.py       # 10-component gasoline mixture model (RandomForest baseline)
│   └── train_sidt_gasoline_selfies_export.py  # 10-component gasoline SELFIES VAE + MLP (CN-MXTR)
└── models/
    ├── methane/               # Exported ONNX models for methane
    ├── ethane/                # Exported ONNX models for ethane
    ├── propane/               # Exported ONNX models for propane
    ├── butane/                # Exported ONNX models for butane
    ├── pentane/               # Exported ONNX models for pentane
    ├── hexane/                # Exported ONNX models for hexane
    ├── heptane/               # Exported ONNX models for heptane
    ├── natgas_mix/            # Exported ONNX models for 6-component natural gas mix
    │   ├── forward_model.onnx
    │   └── idt_400k_model.onnx
    ├── gasoline_mix/          # RandomForest baseline for 10-component gasoline surrogate
    │   ├── forward_model.onnx
    │   ├── idt_400k_model.onnx
    │   └── ntc/               # Exported ONNX models for NTC bounds
    │       ├── has_ntc_classifier.onnx
    │       ├── ntc_t_min_model.onnx
    │       └── ntc_t_max_model.onnx
    └── gasoline_mix_selfies/  # SELFIES VAE + MLP ensemble for 10-component gasoline surrogate
        ├── idt_gasoline_selfies.onnx      # Full end-to-end ONNX export (opset 17)
        ├── predictor_seed0.pt             # MLP checkpoint — ensemble seed 0
        ├── predictor_seed1.pt             # MLP checkpoint — ensemble seed 1
        ├── predictor_seed2.pt             # MLP checkpoint — ensemble seed 2
        ├── predictor_seed3.pt             # MLP checkpoint — ensemble seed 3
        └── predictor_seed4.pt             # MLP checkpoint — ensemble seed 4
```

---

## 2. Model Architecture

* **Forward IDT Model (Pure Compounds)**: Predicts `idt_s` from physical conditions `(pressure_bar, temperature_K, phi, egr_fraction)`.
* **Forward Mixture IDT Model (`natgas_mix`)**: Predicts `idt_400K_s` from 10 inputs: 6 fuel component mole fractions `(cpnt_mole_frac_1..6)` and operating conditions `(pressure_pa, temperature_K, phi, egr_fraction)`.
* **Forward Mixture IDT Model (`gasoline_mix`)**: RandomForest baseline predicting `idt_400K_s` from 14 inputs: 10 fuel component mole fractions `(cpnt_mole_frac_1..10)` (ethanol, 1-hexene, toluene, 2-methylhexane, cyclopentane, isopentane, isooctane, n-hexane, n-heptane, 1,2,4-trimethylbenzene) and operating conditions `(pressure_pa, temperature_K, phi, egr_fraction)`.
* **Forward Mixture IDT Model (`gasoline_mix_selfies`) — CN-MXTR architecture**:
  * Uses 5 pretrained SELFIES VAE encoders (`SELFIES/checkpoints_opt/seed{i}_s1_vae.pt`) to embed each of the 10 component SELFIES strings into a 128-dim latent vector μᵢ.
  * Mixture latent: **z_mix = Σᵢ xᵢ · μᵢ** (mole-fraction weighted sum).
  * z_mix + standardised reactor conditions → 3-layer MLP (512→256→128→1, LayerNorm+SiLU).
  * 5-seed ensemble averaged at inference.
  * Trained on 89,567 LHS simulation runs (Sarathy 2016 compositional mechanism).
  * Full end-to-end export: `idt_gasoline_selfies.onnx` — inputs: `component_tokens [10,65]`, `mole_fracs [N,10]`, `reactor_conds [N,4]`; output: `idt_log1p [N]`.
* **Inverse Condition Models**: Predicts one parameter from the remaining conditions and `idt_s`.
* **NTC Bounds Models**:
  * **Classifier (`has_ntc_classifier.onnx`)**: Random Forest Classifier predicting whether an NTC pocket exists (`has_ntc` = 1 or 0) for given operating conditions `(pressure_bar, phi, egr_fraction)`.
  * **Regressors (`ntc_t_min_model.onnx` & `ntc_t_max_model.onnx`)**: Random Forest Regressors predicting lower (`ntc_t_min_K`) and upper (`ntc_t_max_K`) NTC temperature thresholds in Kelvin.

---

## 3. Training & Exporting Models

To train and export models, run the corresponding scripts:

```bash
# 1. Train Forward & Inverse IDT models (Pure Compounds)
python sidt/scripts/train_export.py \
    --input model_training/sidt/sidt_selfies_heptane.dat \
    --out_dir sidt/models/heptane

# 2. Train Natural Gas Mixture Model (6-Component Fuel Blend)
python sidt/scripts/train_natgas_mix_export.py \
    --input model_training/sidt/sidt_selfies_natgas_mix.dat \
    --out_dir sidt/models/natgas_mix

# 3. Train Gasoline Mixture Model — RandomForest baseline (10-Component Surrogate Blend)
python sidt/scripts/train_gasoline_mix_export.py \
    --input model_training/sidt/sidt_selfies_gasoline_mix.dat \
    --out_dir sidt/models/gasoline_mix

# 4. Train Gasoline Mixture Model — SELFIES VAE + MLP Ensemble (CN-MXTR, 10-Component)
#    Requires: SELFIES/checkpoints_opt/seed{0..4}_s1_vae.pt (pretrained VAE encoders)
#    Dataset:  model_training/sidt/sidt_lhs_gasoline_k10_10k.dat (89,567 LHS runs)
python sidt/scripts/train_sidt_gasoline_selfies_export.py \
    --input model_training/sidt/sidt_lhs_gasoline_k10_10k.dat \
    --out_dir sidt/models/gasoline_mix_selfies \
    --seeds 5 \
    --epochs 60

# 5. Train NTC Bounds models
python sidt/scripts/train_ntc_export.py \
    --input model_training/sidt/sidt_ntc_bounds_propane.dat \
    --out_dir sidt/models/propane/ntc
```

---

## 4. CLI Inference Engine

`sidt/inference.py` provides a unified CLI runner supporting `--mode forward`, `--mode inverse`, and `--mode ntc`.

### A. Forward Mode (Predict IDT & Generate Arrhenius Plot)
```bash
# Pure Compound (e.g. Propane)
python sidt/inference.py \
    --mode forward \
    --compound propane \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0

# 6-Component Natural Gas Mixture
python sidt/inference.py \
    --mode forward \
    --compound natgas_mix \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0

# 10-Component Gasoline Surrogate Mixture — RandomForest baseline
# (equimolar 0.1 each, or custom --cpnt_mol_fracs)
python sidt/inference.py \
    --mode forward \
    --compound gasoline_mix \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0

# 10-Component Gasoline Surrogate Mixture — SELFIES VAE + MLP (CN-MXTR)
# Uses: sidt/models/gasoline_mix_selfies/idt_gasoline_selfies.onnx
python sidt/inference.py \
    --mode forward \
    --compound gasoline_mix_selfies \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0
```

### B. Inverse Mode (Predict Operating Condition)
```bash
# Predict Temperature for Propane
python sidt/inference.py \
    --mode inverse \
    --compound propane \
    --target temperature \
    --pressure 10.0 \
    --phi 1.0 \
    --egr_fraction 0.0 \
    --idt 0.01
```

### C. NTC Mode (Predict NTC Presence & Bounds)
```bash
python sidt/inference.py \
    --mode ntc \
    --compound propane \
    --pressure 10.0 \
    --phi 0.5 \
    --egr_fraction 0.0
```
* **Output**: Prints `has_ntc`, `T_min`, and `T_max`, and automatically generates the Arrhenius NTC curve plot saved at `sidt/propane_ntc_curve.png`.

