import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    # Configure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    # Resolve paths relative to this script
    _HERE = Path(__file__).resolve().parent  # dist_2dgc/scripts/
    _ROOT = _HERE.parent.parent              # turingtrain/
    
    data_path = _ROOT / "model_training" / "dist_2dgc" / "training_data" / "dist_2dgc_dataset.dat"
    models_dir = _HERE.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading distillation dataset from: {data_path}")
    # Handle mixed types columns to prevent Performance/DtypeWarning
    df = pd.read_csv(data_path, sep='\t', comment='#', low_memory=False)
    
    # Define standard outputs
    targets = ['T05', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90', 'T95']
    df = df.dropna(subset=targets)
    
    # Define standard classes map
    classes_map = {
        'n-paraffins': 'nor_par',
        'iso-paraffins': 'iso_par',
        '1R-cycloparaffins': 'mon_nap',
        '2R-cycloparaffins': 'di_nap',
        '3R-cycloparaffins': 'tri_nap',
        '1R-aromatics': 'mon_aro',
        '2R-aromatics': 'di_aro',
        'cycloaromatics': 'nap_aro',
        'olefins': 'olef',
        'synthetic-oxygenates': 'syn_oxy',
        'antioxidant-oxygenates': 'ant_oxy',
        'dienes': 'dien',
        'indenes': 'inde'
    }
    
    canonical_classes = sorted(list(set(classes_map.values())))
    
    # Construct standardized feature columns template: 13 classes x C1-C30 (390 features)
    feature_cols = []
    for cls in canonical_classes:
        for c in range(1, 31):
            col = f"2dgc_{cls}_c{c:02d}"
            feature_cols.append(col)
            # Pad column with 0 if it's missing in dataset
            if col not in df.columns:
                df[col] = 0.0
                
    # Prepare training features X (N, 390) and target Y (N, 11)
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    Y_raw = df[targets].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    
    print(f"Dataset Loaded. Samples count: {len(X_raw)}")
    print(f"X shape: {X_raw.shape}, Y shape: {Y_raw.shape}")
    
    # 1. Train Forward Model (Ridge regression with target-specific alphas)
    # Alphas chosen from cross-validation: 50.0 for T60, 10.0 for other targets
    alphas = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0, 10.0, 10.0, 10.0, 10.0])
    
    forward_model = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alphas, random_state=42))
    ])
    print("Training Forward Model (Composition -> Distillation Curve)...")
    forward_model.fit(X_raw, Y_raw)
    
    # 2. Train Inverse Model (Distillation Curve -> Composition)
    # Alpha chosen from cross-validation: 1.0
    inverse_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
    print("Training Inverse Model (Distillation Curve -> Composition)...")
    inverse_model.fit(Y_raw, X_raw)
    
    # --- Export to ONNX ---
    print("Converting models to ONNX...")
    
    # Forward Model ONNX Conversion
    initial_type_fwd = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    onx_fwd = convert_sklearn(forward_model, initial_types=initial_type_fwd)
    fwd_path = models_dir / "distillation_model.onnx"
    with open(fwd_path, "wb") as f:
        f.write(onx_fwd.SerializeToString())
    print(f"✓ Exported Forward Model to: {fwd_path}")
        
    # Inverse Model ONNX Conversion
    initial_type_inv = [('float_input', FloatTensorType([None, len(targets)]))]
    onx_inv = convert_sklearn(inverse_model, initial_types=initial_type_inv)
    inv_path = models_dir / "inverse_model.onnx"
    with open(inv_path, "wb") as f:
        f.write(onx_inv.SerializeToString())
    print(f"✓ Exported Inverse Model to: {inv_path}")
    
    print("✓ All distillation models trained and exported successfully.")

if __name__ == "__main__":
    main()
