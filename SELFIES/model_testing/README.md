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
│   ├── inverse_design.py         # Latent space Bayesian Optimization for fuel design
│   └── selfies_rf_benchmark.py   # Random Forest fingerprint baseline
├── selfies_tokenizer.py    # Tokenizer implementation
├── selfies_vae_optimized.onnx    # Unified exported ONNX model (weights embedded)
├── selfies_vae_predictions.csv   # Predictions log on validation split
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

### Requirements
Install all dependencies (including optional ones like `onnxruntime`, `rdkit`, `lightgbm`, etc.) for the `intensors` conda environment from the provided [`requirements.txt`](requirements.txt):
```bash
pip install -r requirements.txt
```

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
