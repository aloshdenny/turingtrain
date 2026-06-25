# Cetane Number Prediction Scripts

This folder contains scripts for predicting cetane numbers from chemical mixture compositions using various machine learning approaches.

## Data
- `cn_mixtues_inchi.dat` - Main dataset with 1143 mixture records (tab-separated)
  - Columns: No, cpnt_inchi_1-10, cpnt_vol_1-10, CN
  - 2-10 component mixtures with volume fractions and cetane number targets

## GNN Approach (Molecular Graph Neural Networks)

### Core Scripts
- **gnn_model.py** - GNN architecture with split-head design
  - `AtomEncoder` - Embed atomic features
  - `BondEncoder` - Embed bond features
  - `MessagePassingLayer` - Single GNN layer
  - `MolecularGNN` - Molecular encoding with 5 MP layers
  - `MixtureGNN` - Per-component GNN + linear blending

- **train_gnn.py** - Training pipeline
  - Converts InChI strings to molecular graphs
  - Trains MixtureGNN with Adam optimizer
  - Usage: `python train_gnn.py --epochs 50 --batch-size 16`

- **debug_gnn.py** - Diagnostic utilities
  - Inspect InChI conversions
  - Sample model predictions
  - Usage: `python debug_gnn.py`

- **evaluate_gnn.py** - Model evaluation
  - Compare GNN vs RandomForest baseline
  - Usage: `python evaluate_gnn.py`

### Performance
- Test MAE: ~17.05 (poor)
- Test R²: ~-0.28
- **Verdict**: GNN architecture mismatched to problem (topology-based features for chemistry-driven problem)

## Neural Network Approaches (Chemical Features)

### Simple NN
- **train_chemical_nn.py** - 3-layer NN with direct element counts
  - Features: C, H, O, N, S, P, F, Cl, Br, I, other + metadata (66 total)
  - Performance: MAE ~12.25 (better than GNN, worse than RF)

### Notebook's NN
- **train_nn_proper.py** - NN with full feature engineering (matching notebook)
  - Features: 87 features including mixture statistics (volume entropy, weighted components)
  - Batch-normalized volumes, per-component element counts, chemical ratios
  - Performance: MAE ~10.24 (much better, but still worse than RF)
  - Usage: `python train_nn_proper.py`

## Analysis
- **analyze_training.py** - Extract and summarize training metrics
  - Parse terminal output for loss/MAE/R² over epochs
  - Usage: `python analyze_training.py < training_output.txt`

## Baselines

### RandomForest (Notebook's Approach)
- **inchi_cn_model.pkl** - Trained RandomForest (250 estimators)
- Test MAE: **7.29** ✓ Best performance
- Test R²: 0.620
- Reason: RF naturally captures additive blending rules; small dataset favors ensemble methods

## Key Findings

### Why RandomForest Wins
1. **Chemistry is additive**: CN_mixture ≈ Σ(mole_fraction_i × CN_i) + interactions
2. **Element composition matters**: C/H/O ratios are most predictive
3. **Tree-based learning**: Decision trees naturally encode blending rules
4. **Dataset size**: 1143 samples too small for DL to outperform traditional ML

### Why GNN Failed
- Problem driven by **element composition**, not molecular **topology**
- Molecular graph structure indirect signal for cetane prediction
- Message-passing learns connectivity, not chemistry
- Result: Predictions plateau at MAE ~17 despite architecture improvements

### Why NNs Underperform
- Require more data to learn additive rules
- Gradient descent struggles with simple decision boundaries
- Trees more efficient at capturing if-then chemistry logic

## Standalone Inference (`inference.py`)

A self-contained, standardized inference script is located at [`Inchi/inference.py`](inference.py). It supports predicting Cetane Numbers from InChI strings and volume fractions using either the ONNX model (default) or the Pickle (RandomForest) backend.

### Requirements & Setup
Please refer to the root [`README.md`](../README.md) for unified environment creation (`intensors` conda environment) and library installation.

### Usage Examples

#### 1. Single Mixture Prediction (ONNX Backend - Default)
```bash
python Inchi/inference.py \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3" \
    --vols 0.5 0.5
```

#### 2. Single Mixture Prediction (Pickle Backend)
```bash
python Inchi/inference.py \
    --model Inchi/inchi_cn_model.pkl \
    --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3" \
    --vols 0.5 0.5
```

#### 3. Batch Inference on database CSV/DAT file
```bash
python Inchi/inference.py \
    --csv Inchi/cn_mixtues_inchi.dat \
    --out predictions_out.csv
```

