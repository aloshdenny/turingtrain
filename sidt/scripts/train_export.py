import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

def train_and_export_onnx(X, y, feature_names, target_name, out_path):
    """Trains a Random Forest Regressor and exports it to ONNX format."""
    print(f"Training model for target '{target_name}'...")
    
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    
    print(f"Converting target '{target_name}' to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    
    # Save ONNX model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✓ Exported ONNX model to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and export SIDT models for a specific compound.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset .dat file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the exported ONNX models")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
        
    print(f"Loading SIDT dataset from: {input_path}")
    df = pd.read_csv(input_path, sep='\t', comment='#')
    
    # Define standard physical condition columns
    # We clean/impute target columns if needed
    cols = ['pressure_bar', 'temperature_K', 'phi', 'egr_fraction', 'idt_s']
    df_clean = df[cols].dropna(subset=cols)
    
    print(f"Loaded {len(df_clean)} clean data samples.")
    
    # 1. Forward Model: pressure, temperature, phi, egr_fraction -> idt
    X_fwd = df_clean[['pressure_bar', 'temperature_K', 'phi', 'egr_fraction']].values
    y_fwd = df_clean['idt_s'].values
    train_and_export_onnx(
        X_fwd, y_fwd, 
        ['pressure_bar', 'temperature_K', 'phi', 'egr_fraction'], 
        'idt_s', 
        out_dir / "forward_model.onnx"
    )
    
    # 2. Inverse Pressure: temperature, phi, egr_fraction, idt -> pressure
    X_inv_p = df_clean[['temperature_K', 'phi', 'egr_fraction', 'idt_s']].values
    y_inv_p = df_clean['pressure_bar'].values
    train_and_export_onnx(
        X_inv_p, y_inv_p, 
        ['temperature_K', 'phi', 'egr_fraction', 'idt_s'], 
        'pressure_bar', 
        out_dir / "inverse_pressure_model.onnx"
    )
    
    # 3. Inverse Temperature: pressure, phi, egr_fraction, idt -> temperature
    X_inv_t = df_clean[['pressure_bar', 'phi', 'egr_fraction', 'idt_s']].values
    y_inv_t = df_clean['temperature_K'].values
    train_and_export_onnx(
        X_inv_t, y_inv_t, 
        ['pressure_bar', 'phi', 'egr_fraction', 'idt_s'], 
        'temperature_K', 
        out_dir / "inverse_temperature_model.onnx"
    )
    
    # 4. Inverse Phi: pressure, temperature, egr_fraction, idt -> phi
    X_inv_phi = df_clean[['pressure_bar', 'temperature_K', 'egr_fraction', 'idt_s']].values
    y_inv_phi = df_clean['phi'].values
    train_and_export_onnx(
        X_inv_phi, y_inv_phi, 
        ['pressure_bar', 'temperature_K', 'egr_fraction', 'idt_s'], 
        'phi', 
        out_dir / "inverse_phi_model.onnx"
    )
    
    # 5. Inverse EGR Fraction: pressure, temperature, phi, idt -> egr_fraction
    X_inv_egr = df_clean[['pressure_bar', 'temperature_K', 'phi', 'idt_s']].values
    y_inv_egr = df_clean['egr_fraction'].values
    train_and_export_onnx(
        X_inv_egr, y_inv_egr, 
        ['pressure_bar', 'temperature_K', 'phi', 'idt_s'], 
        'egr_fraction', 
        out_dir / "inverse_egr_fraction_model.onnx"
    )
    
    print(f"✓ Successfully trained and exported all 5 SIDT models to: {out_dir}")

if __name__ == "__main__":
    main()
