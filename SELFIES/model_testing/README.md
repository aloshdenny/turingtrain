# SELFIES VAE Model Testing & Inference

This folder contains the production-ready checkpoints, dataset caches, evaluation utilities, and a self-contained inference script for the optimized attention-based Cetane Number (CN) mixture model.

## Directory Structure

```
SELFIES/
├── data/                         # ← Single source of truth for data
│   ├── preprocess_selfies.py     # Run this to rebuild the dataset cache
│   ├── cn_mixtures_selfies.pkl   # Tokenised dataset cache
│   └── vocab.json                # Tokenizer vocabulary
├── vae/                          # ← Single source of truth for all model code
│   ├── train_vae_optimized.py    # Training script (Stage 1/2/3)
│   ├── selfies_vae.py            # VAE model definition
│   ├── mixture_cn_predictor.py   # Baseline predictor
│   ├── selfies_rf_benchmark.py   # Random Forest baseline
│   ├── train_vae.py              # Baseline training script
│   ├── inverse_design.py         # (legacy) latent-space inverse design
│   └── evaluate_and_export.py    # (legacy) evaluation script
├── selfies_tokenizer.py          # Tokenizer implementation
├── inchi_to_selfies.py           # InChI → SELFIES conversion utility
└── model_testing/                # ← Production artifacts only
    ├── checkpoints_opt/          # Trained model weights
    │   ├── seed0_s1_vae.pt       # Stage 1: Pre-trained VAE
    │   ├── seed0_s2_model.pt     # Stage 2: Mixture predictor
    │   └── seed0_s3_model.pt     # Stage 3: Fine-tuned joint model ← used for inference
    ├── checkpoints/              # Legacy baseline checkpoints
    ├── vae/
    │   └── evaluate_and_export.py  # Evaluate model + export to ONNX
    ├── selfies_vae_optimized.onnx  # Exported ONNX model
    ├── selfies_vae_predictions.csv # Validation predictions
    ├── inverse_design.py           # Inverse design: CN target → novel mixture
    └── inference.py                # Standalone inference engine
```

---

## Quick Start: Run Inference

> [!IMPORTANT]
> Make sure you are inside the `intensors` conda environment before running anything.
> See the root [`README.md`](../../README.md) for setup instructions.

All scripts print progress in real-time. **No environment variables needed.**

### 1. Single Mixture Prediction (PyTorch checkpoint)

Run from the `turingtrain/` root directory:

```bash
python SELFIES/model_testing/inference.py \
    --model SELFIES/model_testing/checkpoints_opt/seed0_s3_model.pt \
    --selfies "[C][C]" "[C][O]" \
    --vols 0.5 0.5 \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"
```

### 2. Single Mixture Prediction (ONNX — faster, no PyTorch required)

```bash
python SELFIES/model_testing/inference.py \
    --model SELFIES/model_testing/selfies_vae_optimized.onnx \
    --selfies "[C][C]" "[C][O]" \
    --vols 0.5 0.5 \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"
```

> [!NOTE]
> The `selfies_vae_optimized.onnx` file has model weights directly embedded — no separate `.data` file needed.

### 3. Batch Inference from CSV

```bash
python SELFIES/model_testing/inference.py \
    --model SELFIES/model_testing/selfies_vae_optimized.onnx \
    --csv SELFIES/model_testing/data/cn_mixtures_selfies.pkl \
    --out predictions_out.csv
```

---

## Retraining

Follow these steps **in order**. Do not skip steps.

### Step 1 — Preprocess the dataset

Run this whenever the raw `.dat` file changes:

```bash
python SELFIES/data/preprocess_selfies.py
```

This rebuilds `SELFIES/data/cn_mixtures_selfies.pkl` and `vocab.json`.

### Step 2 — Train the model

This takes several hours on CPU. Progress is printed in real-time.

```bash
python SELFIES/vae/train_vae_optimized.py --no-ensemble
```

Checkpoints are saved to `SELFIES/checkpoints_opt/` automatically.

### Step 3 — Evaluate and export to ONNX

```bash
python SELFIES/model_testing/vae/evaluate_and_export.py
```

This produces:
- `SELFIES/model_testing/selfies_vae_predictions.csv` — validation predictions
- `SELFIES/model_testing/selfies_vae_optimized.onnx` — exportable model

> [!NOTE]
> No manual file copying is ever needed. All scripts read from `SELFIES/checkpoints_opt/` and `SELFIES/data/` directly.

---

## Inverse Design (`inverse_design.py`)

Given a target cetane number, finds novel fuel mixture compositions the model predicts will achieve it.

### How it works
1. **Warm-start**: Encode ~2000 dataset mixtures into a latent bank; select nearest-CN as seeds.
2. **Gradient optimisation**: Jointly relax per-component latent vectors *z_i* and volume fractions to minimise `(pred_CN − target_CN)²` with diversity + entropy regularisation.
3. **Decode**: Optimised *z_i* → SELFIES → SMILES (RDKit validated).
4. **Report**: Write ranked candidates to a CSV file.

### Usage

Run from the `turingtrain/` root directory:

```bash
# Single target
python SELFIES/model_testing/inverse_design.py \
    --target-cn 90 \
    --n-candidates 10 \
    --opt-steps 500

# Multiple targets at once
python SELFIES/model_testing/inverse_design.py \
    --target-cn 60 80 100 \
    --n-candidates 5 \
    --output inverse_results.csv

# Fast search (fewer components, fewer steps)
python SELFIES/model_testing/inverse_design.py \
    --target-cn 85 \
    --n-comp 3 \
    --n-candidates 8 \
    --opt-steps 300 \
    --lr 5e-3
```

| Argument | Default | Description |
|---|---|---|
| `--target-cn` | **required** | Target cetane number(s), space-separated |
| `--n-candidates` | 10 | Candidate mixtures per target |
| `--n-comp` | 10 | Max components per mixture |
| `--opt-steps` | 500 | Gradient optimisation steps per candidate |
| `--n-restarts` | 5 | Random restarts per candidate (best is kept) |
| `--lr` | 0.01 | Adam learning rate for latent optimisation |
| `--noise-std` | 0.5 | Gaussian noise on warm-start latents |
| `--output` | auto | Output CSV path |
| `--ckpt-dir` | `checkpoints_opt/` | Checkpoint directory |

Results are saved to `SELFIES/model_testing/inverse_design_results/inverse_cn<target>.csv`.

---

## Programmatic API

You can also use the inference engine in your own Python code:

```python
from SELFIES.model_testing.inference import CNInferenceModel

# Load model (PyTorch or ONNX — both work)
model = CNInferenceModel("SELFIES/model_testing/selfies_vae_optimized.onnx")

# Define a mixture
mixture = {
    "components": [
        {"selfies": "[C][C]", "vol": 0.5, "inchi": "InChI=1S/C2H6/c1-2/h1-2H3"},
        {"selfies": "[C][O]", "vol": 0.5, "inchi": "InChI=1S/CH4O/c1-2/h2H,1H3"}
    ]
}

# Predict
predicted_cn = model.predict([mixture])[0]
print(f"Predicted Cetane Number: {predicted_cn:.4f}")
```
