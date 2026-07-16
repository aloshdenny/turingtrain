# SIDT Model Training and Inference

This sub-project handles training, export, and inference for Soot/Similar Ignition Delay Time (SIDT) models. Each compound dataset is trained to produce 1 forward model and 4 inverse models using Random Forest Regressors, which are then compiled into ONNX binaries.

---

## 1. Directory Structure

```
sidt/
├── README.md              # Documentation (this file)
├── inference.py           # Unified forward/inverse CLI inference runner
├── scripts/
│   └── train_export.py    # Parameterized training & ONNX export script
└── models/
    ├── methane/           # Exported ONNX models for methane
    └── ethane/            # Exported ONNX models for ethane
```

---

## 2. Model Architecture

* **Forward Model**: Predicts ignition delay time (`idt_s` in seconds) from physical conditions (`pressure_bar`, `temperature_K`, `phi`, `egr_fraction`).
* **Inverse Models**: Predicts one physical condition from the remaining conditions and the ignition delay time:
  * **Inverse Pressure**: Predicts `pressure_bar` from `(temperature, phi, egr_fraction, idt)`.
  * **Inverse Temperature**: Predicts `temperature_K` from `(pressure, phi, egr_fraction, idt)`.
  * **Inverse Phi**: Predicts `phi` from `(pressure, temperature, egr_fraction, idt)`.
  * **Inverse EGR Fraction**: Predicts `egr_fraction` from `(pressure, temperature, phi, idt)`.

To handle the highly non-linear nature of ignition delay times (Arrhenius kinetics), all models are trained using **Random Forest Regressors** (`n_estimators=100`, `max_depth=12`), which significantly outperform linear models.

---

## 3. Training & Exporting Models

To train and export models for any compound dataset (including new compounds), run the parameterized training script:

```bash
# Train Methane models
python sidt/scripts/train_export.py \
    --input model_training/sidt/sidt_selfies_methane.dat \
    --out_dir sidt/models/methane

# Train Ethane models
python sidt/scripts/train_export.py \
    --input model_training/sidt/sidt_selfies_ethane.dat \
    --out_dir sidt/models/ethane
```

---

## 4. CLI Inference Engine

`sidt/inference.py` provides a unified runner for both forward and inverse prediction modes across all trained compounds.

### Forward Mode (Predict IDT)
```bash
python sidt/inference.py \
    --mode forward \
    --compound methane \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --egr_fraction 0.0
```

### Inverse Mode (Predict Condition)
You must specify the `--target` variable (`pressure`, `temperature`, `phi`, or `egr_fraction`) along with the remaining 4 conditions:
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

# Predict EGR Fraction for Ethane
python sidt/inference.py \
    --mode inverse \
    --compound ethane \
    --target egr_fraction \
    --pressure 10.0 \
    --temperature 1000.0 \
    --phi 1.0 \
    --idt 0.1
```
