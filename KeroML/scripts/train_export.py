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
    # Resolve paths relative to this script
    _HERE = Path(__file__).resolve().parent  # KeroML/scripts/
    _ROOT = _HERE.parent.parent              # turingtrain/
    
    data_path = _ROOT / "model_training" / "keroml" / "training_data" / "keroml_dataset.dat"
    brix_path = _ROOT / "model_training" / "keroml" / "training_data" / "keroml_theoretical_brix.dat"
    models_dir = _HERE.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep='\t', comment='#')
    df = df.dropna(subset=['cetane_number_val'])
    
    print(f"Loading BRIX limits from: {brix_path}")
    brix = pd.read_csv(brix_path, sep='\t', comment='#')

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
    
    # Construct standardized feature columns template: 13 classes x C1-C30
    feature_cols = []
    for cls in canonical_classes:
        for c in range(1, 31):
            col = f"2dgc_{cls}_c{c:02d}"
            feature_cols.append(col)
            # Pad column with 0 if it's missing in dataset
            if col not in df.columns:
                df[col] = 0.0
                
    # Prepare training features X (N, 390) and target y (N,)
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y = df['cetane_number_val'].values
    
    # 1. Train Pre-BRIX Model (Ridge with alpha=50.0 on raw compositions)
    pre_brix_model = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=50.0, random_state=42))
    ])
    pre_brix_model.fit(X_raw, y)

    # 2. Train Post-BRIX Model (Ridge with alpha=50.0 using BRIX linear subspace constraint)
    # Parse BRIX limits for lookup
    brix_lookup = {}
    for col in brix.columns:
        if col.endswith('_carbon_number'):
            cls = col.replace('_carbon_number', '')
            min_col = f'{cls}_brix_min'
            max_col = f'{cls}_brix_max'
            if min_col not in brix.columns or max_col not in brix.columns:
                continue
            sub = brix[[col, min_col, max_col]].dropna().copy()
            sub[col] = sub[col].astype(int)
            brix_lookup[cls] = {
                int(r[col]): (float(r[min_col]), float(r[max_col]))
                for _, r in sub.iterrows()
            }
            
    # Build projection matrix W of shape (390, 39)
    W = np.zeros((len(feature_cols), 3 * len(canonical_classes)))
    pat = re.compile(r'^2dgc_([a-z_]+)_c(\d{2})$')
    for col_idx, col in enumerate(feature_cols):
        m = pat.match(col)
        if not m:
            continue
        raw_cls, carbon = m.group(1), int(m.group(2))
        brix_cls = 'oxy' if raw_cls in ('syn_oxy', 'ant_oxy') else raw_cls
        if brix_cls not in brix_lookup or carbon not in brix_lookup[brix_cls]:
            continue
        bmin, bmax = brix_lookup[brix_cls][carbon]
        bmid = 0.5 * (bmin + bmax)
        
        cls_idx = canonical_classes.index(raw_cls)
        W[col_idx, 3 * cls_idx] = 1.0
        W[col_idx, 3 * cls_idx + 1] = bmid
        W[col_idx, 3 * cls_idx + 2] = bmax - bmin
        
    X_linear = X_raw @ W
    X_concat = np.hstack([X_raw, X_linear])
    
    # Fit Ridge on the concatenated features
    pipeline_post = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=50.0, random_state=42))
    ])
    pipeline_post.fit(X_concat, y)
    
    # Retrieve scaling parameters
    scaler = pipeline_post.named_steps['scaler']
    mean = scaler.mean_
    scale = scaler.scale_
    
    # Retrieve coefficients and intercept
    reg = pipeline_post.named_steps['regressor']
    coef = reg.coef_
    intercept = reg.intercept_
    
    # Fold scaling and W matrix into raw composition coefficients:
    # y = X_raw @ beta_equiv + intercept_equiv
    n_raw = X_raw.shape[1]
    beta_raw = coef[:n_raw]
    beta_linear = coef[n_raw:]
    
    mean_raw = mean[:n_raw]
    scale_raw = scale[:n_raw]
    mean_linear = mean[n_raw:]
    scale_linear = scale[n_raw:]
    
    beta_equiv = (beta_raw / scale_raw) + W @ (beta_linear / scale_linear)
    intercept_equiv = intercept - np.dot(mean_raw / scale_raw, beta_raw) - np.dot(mean_linear / scale_linear, beta_linear)
    
    # Create and fit the post-BRIX pipeline on the raw dataset
    post_brix_model = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('regressor', Ridge(alpha=50.0, random_state=42))
    ])
    post_brix_model.fit(X_raw, y)
    
    # Overwrite the fitted regressor's coefficients and intercept with our folded BRIX weights
    post_brix_model.named_steps['regressor'].coef_ = beta_equiv
    post_brix_model.named_steps['regressor'].intercept_ = intercept_equiv
    
    # 3. Train Inverse Model (y -> X)
    inverse_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=0.1, random_state=42))
    ])
    inverse_model.fit(y.reshape(-1, 1), X_raw)
    
    # --- Export to ONNX ---
    print("Converting models to ONNX...")
    
    # Pre-Brix
    initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    onx_pre = convert_sklearn(pre_brix_model, initial_types=initial_type)
    pre_path = models_dir / "pre_brix_model.onnx"
    with open(pre_path, "wb") as f:
        f.write(onx_pre.SerializeToString())
    print(f"Exported: {pre_path}")
        
    # Post-Brix
    onx_post = convert_sklearn(post_brix_model, initial_types=initial_type)
    post_path = models_dir / "post_brix_model.onnx"
    with open(post_path, "wb") as f:
        f.write(onx_post.SerializeToString())
    print(f"Exported: {post_path}")
        
    # Inverse
    initial_type_inv = [('float_input', FloatTensorType([None, 1]))]
    onx_inv = convert_sklearn(inverse_model, initial_types=initial_type_inv)
    inv_path = models_dir / "inverse_model.onnx"
    with open(inv_path, "wb") as f:
        f.write(onx_inv.SerializeToString())
    print(f"Exported: {inv_path}")
    
    print("KeroML forward and inverse models trained and exported to ONNX successfully.")

if __name__ == "__main__":
    main()
