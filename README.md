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

---

## SELFIES VAE Mixture Model

In addition to KeroML, this repository contains a **Transformer-based SELFIES VAE + Attention Mixture Model** for Cetane Number (CN) predictions of complex hydrocarbon fuel mixtures.

The final optimized model employs a `MixtureSlotEncoder` (attention-based nonlinear mixing) to accurately capture interactions in multi-component oxygenate blends.

### Quick Start (model_testing)
A self-contained testing directory is located at [`SELFIES/model_testing/`](SELFIES/model_testing).

* **Standalone Inference**: Running inference requires only two files: [`inference.py`](SELFIES/model_testing/inference.py) and the model file (either the PyTorch `.pt` checkpoint or the embedded `.onnx` binary).
* **Usage**: See the dedicated [`SELFIES/model_testing/README.md`](SELFIES/model_testing/README.md) for execution commands, batch CSV processing, and programmatic API integration.
