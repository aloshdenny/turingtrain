# TuringTrain Distillation 2D-GC Models

This folder contains distillation models and scripts to perform direct (forward) and inverse distillation curve temperature predictions based on 2D-GC composition matrices.

## Directory Structure
- `models/`: Contains the exported ONNX models:
  - `distillation_model.onnx` - Forward prediction model mapping 2D-GC composition (390 features) to 11 ASTM D86 distillation temperatures (`T05`-`T95`).
  - `inverse_model.onnx` - Inverse model mapping target distillation curve temperatures (11 features) to the 2D-GC composition matrix.
- `scripts/`: Internal training and export scripts.
  - `train_export.py` - Core pipeline script responsible for cross-validating, fitting, and exporting the models to ONNX.
- `inference.py`: Standardized, unified inference runner for both forward and inverse prediction modes.

---

## Standalone Inference (`inference.py`)

The unified [`dist_2dgc/inference.py`](inference.py) script is the entry point for running predictions.

### Requirements & Setup
Please refer to the root [`README.md`](../README.md) for environment creation (`intensors` conda environment) and library installation.

### Usage Examples

#### 1. Forward Mode (Composition -> Distillation Curve)
Predict the ASTM D86 distillation curve temperatures from a fuel composition CSV file.
- **Matrix Layout**: A 30x13 matrix with carbon numbers C1-C30 as rows and 13 classes (`n-paraffins`, `iso-paraffins`, `1R-cycloparaffins`, `2R-cycloparaffins`, `3R-cycloparaffins`, `1R-aromatics`, `2R-aromatics`, `cycloaromatics`, `olefins`, `synthetic-oxygenates`, `antioxidant-oxygenates`, `dienes`, and `indenes`) as columns.

```bash
# Using the default distillation_model.onnx
python dist_2dgc/inference.py --mode forward --input dist_2dgc/test_composition.csv
```
*Note: This generates a line plot of the distillation curve (`distillation_curve.png`). You can customize the plot destination using the `--out <path>` argument.*

#### 2. Inverse Mode (Distillation Curve -> Composition)
Predict the carbon number and compound class distribution matrix back from a target set of 11 distillation temperatures (corresponding to `T05 T10 T20 T30 T40 T50 T60 T70 T80 T90 T95`).

```bash
python dist_2dgc/inference.py --mode inverse --temps 150 160 175 190 205 220 235 250 265 280 290
```
*Note: This generates both a `.csv` matrix containing the predicted mass fractions (30x13 matrix) and a stacked bar chart `.png` visualizing the designed fuel makeup.*
*You can customize the outputs using the `--out <csv_path>` and `--out_plot <png_path>` arguments.*

---

## Retraining

To retrain the forward and inverse models on updated data:

```bash
python dist_2dgc/scripts/train_export.py
```

This script will automatically:
1. Load `dist_2dgc_dataset.dat` from the `model_training/dist_2dgc/training_data/` folder.
2. Train the forward model (using target-specific optimal Ridge $\alpha_i$ regularizations) and the inverse model.
3. Export both models directly to `dist_2dgc/models/` in ONNX format.
