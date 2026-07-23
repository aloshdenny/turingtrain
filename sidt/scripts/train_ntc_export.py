import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

def train_and_export_classifier(X, y, out_path):
    """Trains a Random Forest Classifier for has_ntc and exports to ONNX."""
    print("Training NTC Classifier (has_ntc)...")
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✓ Exported NTC Classifier ONNX model to: {out_path}")

def train_and_export_regressor(X, y, target_name, out_path):
    """Trains a Random Forest Regressor for NTC bounds and exports to ONNX."""
    print(f"Training NTC Regressor for '{target_name}'...")
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✓ Exported ONNX model for '{target_name}' to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and export SIDT NTC models for methane.")
    parser.add_argument("--input", type=str, default="model_training/sidt/sidt_ntc_bounds_methane.dat", help="Path to SIDT NTC bounds .dat file")
    parser.add_argument("--out_dir", type=str, default="sidt/models/methane/ntc", help="Directory to save exported NTC ONNX models")
    args = parser.parse_args()
    
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent.parent
    
    input_path = _ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_dir = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    
    if not input_path.exists():
        print(f"Error: Dataset not found at {input_path}")
        sys.exit(1)
        
    print(f"Loading SIDT NTC dataset from: {input_path}")
    df = pd.read_csv(input_path, sep='\t', comment='#')
    
    # Feature columns: pressure_bar, phi, egr_fraction
    features = ['pressure_bar', 'phi', 'egr_fraction']
    X = df[features].values
    
    # 1. Train has_ntc Classifier
    y_has_ntc = df['has_ntc'].astype(int).values
    train_and_export_classifier(X, y_has_ntc, out_dir / "has_ntc_classifier.onnx")
    
    # Filter dataset for valid NTC bounds (where has_ntc == 1)
    df_ntc = df[df['has_ntc'] == 1].dropna(subset=['ntc_t_min_K', 'ntc_t_max_K'])
    X_ntc = df_ntc[features].values
    
    # 2. Train ntc_t_min_K Regressor
    y_t_min = df_ntc['ntc_t_min_K'].values
    train_and_export_regressor(X_ntc, y_t_min, 'ntc_t_min_K', out_dir / "ntc_t_min_model.onnx")
    
    # 3. Train ntc_t_max_K Regressor
    y_t_max = df_ntc['ntc_t_max_K'].values
    train_and_export_regressor(X_ntc, y_t_max, 'ntc_t_max_K', out_dir / "ntc_t_max_model.onnx")
    
    print(f"\n✓ All NTC bounds models for methane trained and exported successfully to: {out_dir}")

if __name__ == "__main__":
    main()
