# TuringTrain Machine Learning Models

This repository houses multiple machine learning models and pipelines for predicting Cetane Number (CN) properties of hydrocarbon fuel mixtures, including:
1. **KeroML** - Feedforward neural network models for direct and inverse Cetane Number predictions based on compound class distributions.
2. **Inchi** - RandomForest and GNN baseline models predicting Cetane Number from molecular structure (InChIs) and volume fractions.
3. **SELFIES** - Transformer-based SELFIES VAE + Attention Mixture models capturing nonlinear blending phenomena in multi-component oxygenate blends.

---

## Environment Setup

To run predictions across all projects in this repository, it is recommended to set up the unified Conda environment `intensors`.

### 1. Create Conda Environment
Create a new conda environment named `intensors` with Python 3.11:
```bash
conda create -n intensors python=3.11 -y
```

### 2. Activate Conda Environment
Activate the environment:
```bash
conda activate intensors
```

### 3. Install Dependencies
Install all required packages from the root `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Deactivate Conda Environment
When done, you can deactivate the environment:
```bash
conda deactivate
```

---

## Projects and Quick Start

### 1. [KeroML](KeroML/)
Predict Cetane Number from compound class distributions, or design fuel compositions inversely.
- **Inference Script**: [`KeroML/inference.py`](KeroML/inference.py)
- **Documentation**: See [`KeroML/README.md`](KeroML/README.md) for usage commands and details.

### 2. [Inchi](Inchi/)
Predict Cetane Number directly from molecular InChI strings and volume fractions.
- **Inference Script**: [`Inchi/inference.py`](Inchi/inference.py)
- **Documentation**: See [`Inchi/README.md`](Inchi/README.md) for usage commands and details.

### 3. [SELFIES](SELFIES/)
Predict Cetane Number using a Transformer-based VAE + Attention Mixture model for complex oxygenate blends.
- **Inference Script**: [`SELFIES/model_testing/inference.py`](SELFIES/model_testing/inference.py)
- **Inverse Design**: [`SELFIES/model_testing/inverse_design.py`](SELFIES/model_testing/inverse_design.py) — gradient-based latent space search to discover novel mixtures with a target CN.
- **Documentation**: See [`SELFIES/model_testing/README.md`](SELFIES/model_testing/README.md) for usage commands and details.

### 4. [Freezing Point (2DGC)](freezing_point/)
Predict fuel blend freezing point (`T_freeze_K`) from 2DGC hydrocarbon bin distributions.
- **Training Script**: [`freezing_point/scripts/train_fp_2dgc_export.py`](freezing_point/scripts/train_fp_2dgc_export.py)
- **Documentation**: See [`freezing_point/README.md`](freezing_point/README.md) for usage commands, metrics, and ONNX inference.

### 5. [Viscosity (2DGC)](viscosity/)
Predict liquid hydrocarbon viscosity (`eta_internal_cP` and `eta_loglinear_cP`) from 2DGC bins, temperature (`T_K`), and pressure (`P_bar`).
- **Training Script**: [`viscosity/scripts/train_visc_2dgc_export.py`](viscosity/scripts/train_visc_2dgc_export.py)
- **Documentation**: See [`viscosity/README.md`](viscosity/README.md) for usage commands, metrics, and ONNX inference.

---

## Retraining

To retrain the SELFIES VAE model from scratch on updated data, follow these steps **in order**:

```bash
# Step 1 — Rebuild the tokenizer cache from the raw dataset
python SELFIES/data/preprocess_selfies.py

# Step 2 — Train (takes several hours on CPU; progress is printed in real-time)
python SELFIES/vae/train_vae_optimized.py --no-ensemble

# Step 3 — Evaluate and export to ONNX
python SELFIES/model_testing/vae/evaluate_and_export.py
```

No manual file copying is ever needed. All scripts read checkpoints from `SELFIES/checkpoints_opt/` and data from `SELFIES/data/` directly.

---

### KeroML Models

To retrain the KeroML forward and inverse models:

```bash
python KeroML/scripts/train_export.py
```

This trains the Pre-BRIX model, folds BRIX constraints directly into the Post-BRIX model coefficients, and trains the multi-output inverse regressor. The resulting ONNX models are written to `KeroML/models/`.
