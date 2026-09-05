import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt

def main():
    parser = argparse.ArgumentParser(description="Train and export SIDT idt_400K_s model for 10-component gasoline surrogate mixture.")
    parser.add_argument("--input", type=str, default="model_training/sidt/sidt_selfies_gasoline_mix.dat", help="Path to input dataset .dat file")
    parser.add_argument("--out_dir", type=str, default="sidt/models/gasoline_mix", help="Directory to save exported ONNX model")
    args = parser.parse_args()

    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent.parent

    input_path = _ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_dir = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if not input_path.exists():
        print(f"Error: Dataset not found at {input_path}")
        sys.exit(1)

    print(f"Loading gasoline mixture dataset from: {input_path}")
    df = pd.read_csv(input_path, sep='\t', comment='#')

    # Feature columns: 10 mole fractions + pressure_pa + temperature_K + phi + egr_fraction
    feature_cols = [f"cpnt_mole_frac_{i}" for i in range(1, 11)] + [
        'pressure_pa', 'temperature_K', 'phi', 'egr_fraction'
    ]
    target_col = 'idt_400K_s'

    df_clean = df[feature_cols + [target_col]].dropna()
    print(f"Loaded {len(df_clean)} clean data samples.")

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # Train / test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print("Training Random Forest Regressor on 90% train split...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Validation Metrics (Test Split 10%):")
    print(f"  R² Score : {r2:.5f}")
    print(f"  MAE      : {mae:.6e} s ({mae * 1000.0:.3f} ms)")
    print(f"  RMSE     : {rmse:.6e} s ({rmse * 1000.0:.3f} ms)")

    # Retrain on full dataset
    print("Fitting model on 100% of samples for final export...")
    pipeline.fit(X, y)

    # Convert to ONNX
    print("Converting model to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    forward_model_path = out_dir / "forward_model.onnx"
    idt_400k_model_path = out_dir / "idt_400k_model.onnx"

    serialized = onx.SerializeToString()
    sz_mb = len(serialized) / (1024 * 1024)
    print(f"ONNX Model Binary Size: {sz_mb:.2f} MB")

    with open(forward_model_path, "wb") as f:
        f.write(serialized)
    with open(idt_400k_model_path, "wb") as f:
        f.write(serialized)

    print(f"✓ Exported ONNX model to: {forward_model_path}")
    print(f"✓ Exported ONNX model to: {idt_400k_model_path}")

    # Verify with onnxruntime
    sess = rt.InferenceSession(str(forward_model_path))
    in_name = sess.get_inputs()[0].name
    sample_input = np.array([[
        0.1] * 10 + [
        1.0e6, 1000.0, 1.0, 0.0
    ]], dtype=np.float32)
    sample_res = sess.run(None, {in_name: sample_input})[0]
    sample_pred = float(sample_res.item() if sample_res.size == 1 else sample_res.flatten()[0])
    print(f"\nSample ONNX Inference (P = 10 bar / 1.0 MPa, T = 1000 K, φ = 1.0, EGR = 0.0, equimolar 10-mix):")
    print(f"  Predicted idt_400K_s: {sample_pred:.6f} s ({sample_pred * 1000.0:.3f} ms)")

if __name__ == "__main__":
    main()
