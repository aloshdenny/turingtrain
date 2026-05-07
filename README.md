# TuringTrain KeroML

## ONNX Inference Workflow

The repository includes a set of ONNX models and scripts to perform direct and inverse Cetane Number (CN) predictions.

### Directory Structure
- `KeroML/models/`: Contains the exported ONNX models (`pre_brix_model.onnx`, `post_brix_model.onnx`, `inverse_model.onnx`).
- `KeroML/scripts/`: Contains the Python scripts for training/exporting models and running inference.

### 1. Training & Exporting the ONNX Models
To parse the dataset and generate the ONNX binaries in the `models/` directory, run:
```bash
python KeroML/scripts/train_export.py
```

### 2. Forward Inference (Composition -> Cetane Number)
Predict the Cetane Number from a given composition CSV (formatted with carbon numbers C5-C24 as columns and compound classes as rows). 
This outputs a normal probability distribution plot (`.png`) for the predicted CN.

```bash
python KeroML/scripts/infer_cn.py --input KeroML/input_sample.csv --model pre_brix_model.onnx
```
*You can swap `--model pre_brix_model.onnx` with `--model post_brix_model.onnx` to test the alternative model.*

### 3. Inverse Inference (Cetane Number -> Composition)
Predict the ideal carbon number/class distribution back from a target Cetane Number constraint.
This generates both a `.csv` matrix containing the mass fractions and a stacked bar chart `.png` visualizing the predicted fuel makeup.

```bash
python KeroML/scripts/infer_inverse.py --cn 45.5
```
