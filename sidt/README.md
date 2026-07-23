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
│   ├── train_export.py        # Parameterized training & ONNX export script (Forward/Inverse IDT)
│   └── train_ntc_export.py    # Training & ONNX export script for NTC bounds
└── models/
    ├── methane/
    │   ├── forward_model.onnx
    │   ├── inverse_*.onnx
    │   └── ntc/               # Exported ONNX models for NTC bounds
    │       ├── has_ntc_classifier.onnx
    │       ├── ntc_t_min_model.onnx
    │       └── ntc_t_max_model.onnx
    └── ethane/
```

---

## 2. Model Architecture

* **Forward IDT Model**: Predicts `idt_s` from physical conditions `(pressure_bar, temperature_K, phi, egr_fraction)`.
* **Inverse Condition Models**: Predicts one parameter from the remaining conditions and `idt_s`.
* **NTC Bounds Models**:
  * **Classifier (`has_ntc_classifier.onnx`)**: Random Forest Classifier predicting whether an NTC pocket exists (`has_ntc` = 1 or 0) for given operating conditions `(pressure_bar, phi, egr_fraction)`.
  * **Regressors (`ntc_t_min_model.onnx` & `ntc_t_max_model.onnx`)**: Random Forest Regressors predicting lower (`ntc_t_min_K`) and upper (`ntc_t_max_K`) NTC temperature thresholds in Kelvin.

---

## 3. Training & Exporting Models

To train and export models, run the corresponding scripts:

```bash
# 1. Train Forward & Inverse IDT models for Methane & Ethane
python sidt/scripts/train_export.py \
    --input model_training/sidt/sidt_selfies_methane.dat \
    --out_dir sidt/models/methane

python sidt/scripts/train_export.py \
    --input model_training/sidt/sidt_selfies_ethane.dat \
    --out_dir sidt/models/ethane

# 2. Train NTC Bounds models for Methane
python sidt/scripts/train_ntc_export.py \
    --input model_training/sidt/sidt_ntc_bounds_methane.dat \
    --out_dir sidt/models/methane/ntc
```

---

## 4. CLI Inference Engine

`sidt/inference.py` provides a unified CLI runner supporting `--mode forward`, `--mode inverse`, and `--mode ntc`.

### A. Forward Mode (Predict IDT & Generate Arrhenius Plot)
```bash
python sidt/inference.py \
    --mode forward \
    --compound methane \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0
```

### B. Inverse Mode (Predict Operating Condition)
```bash
# Predict Temperature for Methane
python sidt/inference.py \
    --mode inverse \
    --compound methane \
    --target temperature \
    --pressure 10.0 \
    --phi 1.0 \
    --egr_fraction 0.0 \
    --idt 0.1
```

### C. NTC Mode (Predict NTC Presence & Bounds)
```bash
python sidt/inference.py \
    --mode ntc \
    --compound methane \
    --pressure 10.0 \
    --phi 0.5 \
    --egr_fraction 0.0
```
* **Output**: Prints `has_ntc`, `T_min`, and `T_max`, and automatically generates the Arrhenius NTC curve plot saved at `sidt/methane_ntc_curve.png`.
