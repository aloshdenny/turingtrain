# TuringTrain KeroML (Kerosene ML Models)

This folder contains KeroML models and scripts to perform direct (forward) and inverse Cetane Number (CN) predictions.

## Directory Structure
- `models/`: Contains the exported ONNX models:
  - `pre_brix_model.onnx` - Forward prediction model trained on pre-BRIX compositions.
  - `post_brix_model.onnx` - Forward prediction model trained on post-BRIX compositions.
  - `inverse_model.onnx` - Inverse design model mapping Cetane Number targets back to composition matrices.
- `scripts/`: Internal training, exporting, and original diagnostic scripts.
- `brix analysis/`: Branching-sensitive BRIX/BI scenario analysis scripts.
- `inference.py`: Standardized, unified inference runner for both forward and inverse prediction modes.

---

## Standalone Inference (`inference.py`)

The unified [`KeroML/inference.py`](inference.py) script is the entry point for running predictions.

### Requirements & Setup
Please refer to the root [`README.md`](../README.md) for environment creation (`intensors` conda environment) and library installation.

### Usage Examples

#### 1. Forward Mode (Composition -> Cetane Number)
Predict the Cetane Number from a fuel composition CSV file. This file must be a 20x8 matrix with compound classes as columns (e.g. `n-paraffins`, `iso-paraffins`, etc.) and carbon numbers C5-C24 as rows.

```bash
# Using the default pre_brix_model.onnx
python KeroML/inference.py --input KeroML/scripts/inverse_cn_20.0.csv

# Using the alternative post_brix_model.onnx
python KeroML/inference.py \
    --model KeroML/models/post_brix_model.onnx \
    --input KeroML/scripts/inverse_cn_20.0.csv
```
*Note: This generates a probability distribution density plot (e.g. `pre_brix_model_dist.png`). You can customize the plot destination using the `--out <path>` argument.*

#### 2. Inverse Mode (Cetane Number -> Composition)
Predict the ideal carbon number and compound class distribution matrix back from a target Cetane Number constraint.

```bash
python KeroML/inference.py --cn 45.5
```
*Note: This generates both a `.csv` matrix containing the predicted mass fractions and a stacked bar chart `.png` visualizing the designed fuel makeup.*
*You can customize the outputs using the `--out <csv_path>` and `--out_plot <png_path>` arguments.*
