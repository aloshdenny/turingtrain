# SELFIES VAE Model Testing & Inference

This folder contains the production-ready checkpoints, dataset caches, evaluation utilities, and a self-contained inference script for the optimized attention-based Cetane Number (CN) mixture model.

## Directory Structure

```
model_testing/
├── checkpoints_opt/        # Final stage checkpoints of the optimized attention-based model
│   ├── seed0_s1_vae.pt     # Stage 1: Pre-trained VAE
│   ├── seed0_s2_model.pt   # Stage 2: Mixture predictor
│   └── seed0_s3_model.pt   # Stage 3: Fine-tuned joint model (PyTorch checkpoint)
├── checkpoints/            # Baseline model checkpoints (for comparison)
├── data/                   # Preprocessed dataset and tokenizer vocabulary
│   ├── cn_mixtures_selfies.pkl
│   └── vocab.json
├── vae/                    # Component implementation and utility scripts
│   ├── train_vae_optimized.py    # Training logic with self-attention slot-encoder
│   ├── evaluate_and_export.py    # Evaluates performance and exports to ONNX
│   ├── mixture_cn_predictor.py   # Baseline linear mixture predictor definition
│   ├── selfies_vae.py            # Core VAE module definition
│   └── selfies_rf_benchmark.py   # Random Forest fingerprint baseline
├── selfies_tokenizer.py    # Tokenizer implementation
├── selfies_vae_optimized.onnx    # Unified exported ONNX model (weights embedded)
├── selfies_vae_predictions.csv   # Predictions log on validation split
├── inverse_design.py       # Gradient-based inverse design: CN target → novel fuel mixture
└── inference.py            # Standalone, two-file execution inference engine
```

---

## Standalone Inference (`inference.py`)

The [`inference.py`](inference.py) script is designed for production deployment. It embeds the tokenizer vocabulary, structural chemistry descriptor extractors, and model class architectures internally. 

To run inferences, you only need **two files**:
1. The inference engine script: `inference.py`
2. The model file: either the PyTorch checkpoint `checkpoints_opt/seed0_s3_model.pt` or the ONNX model `selfies_vae_optimized.onnx`.

> [!NOTE]
> The exported `selfies_vae_optimized.onnx` file has its weights directly embedded (compiled with `external_data=False`). No separate `.data` files are required.

### Requirements & Setup
Please refer to the root [`README.md`](../../README.md) for unified environment creation (`intensors` conda environment) and library installation.

> [!IMPORTANT]
> For single mixture predictions, the `--selfies`, `--vols`, and `--inchis` parameters are now **mandatory arguments** and must all be provided with matching component counts (up to 10 components).


### Usage Examples

#### 1. Single Mixture Prediction (PyTorch Backend)
```bash
python inference.py \
    --model checkpoints_opt/seed0_s3_model.pt \
    --selfies "[C][C]" "[C][O]" \
    --vols 0.5 0.5 \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"
```

#### 2. Single Mixture Prediction (ONNX Runtime Backend)
```bash
python inference.py \
    --model selfies_vae_optimized.onnx \
    --selfies "[C][C]" "[C][O]" \
    --vols 0.5 0.5 \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"
```

#### 3. Batch Inference on Database CSV
If you have a CSV file formatted with columns matching the dataset (e.g. `cpnt_selfies_1`, `cpnt_vol_1`, `cpnt_inchi_1` up to 10 components):
```bash
python inference.py \
    --model selfies_vae_optimized.onnx \
    --csv data/cn_mixtures_selfies.csv \
    --out predictions_out.csv
```

---

## Inverse Design (`inverse_design.py`)

The [`inverse_design.py`](inverse_design.py) script performs **gradient-based inverse design** in the VAE latent space: given a target cetane number, it discovers novel fuel mixture compositions that the trained model predicts will achieve it.

### Strategy
1. **Warm-start**: Encode known dataset mixtures into a latent bank; select nearest-CN mixtures as initial seeds.
2. **Gradient optimisation**: Jointly relax per-component latent vectors *z_i* and volume fractions (via softmax) to minimise `(pred_CN − target_CN)²` with diversity and entropy regularisation.
3. **Decode**: Optimised *z_i* → SELFIES → SMILES (validated with RDKit if available).
4. **Report**: Write ranked candidate mixtures to a CSV file.

### Usage Examples

```bash
# Single target
python model_testing/inverse_design.py \
    --target-cn 90 \
    --n-candidates 10 \
    --opt-steps 500

# Multiple targets
python model_testing/inverse_design.py \
    --target-cn 60 80 100 \
    --n-candidates 5 \
    --output inverse_results.csv

# Faster search (fewer components and steps)
python model_testing/inverse_design.py \
    --target-cn 85 \
    --n-comp 3 \
    --n-candidates 8 \
    --opt-steps 300 \
    --lr 5e-3
```

| Argument | Default | Description |
|---|---|---|
| `--target-cn` | *required* | Target cetane number(s), space-separated |
| `--n-candidates` | 10 | Candidate mixtures per target |
| `--n-comp` | 10 | Max components per mixture |
| `--opt-steps` | 500 | Gradient optimisation steps per candidate |
| `--n-restarts` | 5 | Random restarts per candidate (best kept) |
| `--lr` | 0.01 | Adam learning rate for latent optimisation |
| `--noise-std` | 0.5 | Gaussian noise std on warm-start latents |
| `--output` | auto | Output CSV path |
| `--ckpt-dir` | `checkpoints_opt/` | Checkpoint directory |

Results are saved to `inverse_design_results/inverse_cn<target>.csv`.

---

## Programmatic API

You can also use the inference engine in your own Python code:

```python
from inference import CNInferenceModel

# Load model (PyTorch or ONNX)
model = CNInferenceModel("selfies_vae_optimized.onnx")

# Define candidate mixture
mixture = {
    "components": [
        {"selfies": "[C][C]", "vol": 0.5, "inchi": "InChI=1S/C2H6/c1-2/h1-2H3"},
        {"selfies": "[C][O]", "vol": 0.5, "inchi": "InChI=1S/CH4O/c1-2/h2H,1H3"}
    ]
}

# Run prediction
predicted_cn = model.predict([mixture])[0]
print(f"Predicted Cetane Number: {predicted_cn:.4f}")
```
