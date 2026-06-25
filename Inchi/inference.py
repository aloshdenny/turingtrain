#!/usr/bin/env python3
"""
inference.py
============
Self-contained inference script for the InChI-based CN prediction model.
Supports both ONNX Runtime and Pickle (RandomForest) backends.

Requirements:
    Install dependencies using the root requirements.txt:
    pip install -r requirements.txt

Usage (CLI):
    # Predict CN for a single custom mixture using ONNX model (default):
    python Inchi/inference.py \
        --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3" \
        --vols 0.5 0.5

    # Predict CN for a single custom mixture using Pickle model:
    python Inchi/inference.py \
        --model Inchi/inchi_cn_model.pkl \
        --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3" \
        --vols 0.5 0.5

    # Predict CN for a batch of mixtures from a CSV/DAT file:
    python Inchi/inference.py \
        --csv Inchi/cn_mixtues_inchi.dat \
        --out predictions.csv
"""
import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

def parse_inchi_formula(inchi):
    """Extract element counts from InChI."""
    if not isinstance(inchi, str) or not inchi.startswith('InChI='):
        return {e: 0 for e in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other', 'formula_len']}
    
    try:
        formula_part = inchi.split('/')[1]
    except IndexError:
        formula_part = ''
    
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    counts = {e: 0 for e in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']}
    other = 0
    
    for match in pattern.finditer(formula_part):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        if element in counts:
            counts[element] += count
        else:
            other += count
    
    counts['other'] = other
    counts['formula_len'] = len(formula_part)
    return counts

def extract_inchi_features(inchi):
    """Extract full feature set from InChI."""
    if not isinstance(inchi, str):
        inchi = ''
    
    counts = parse_inchi_formula(inchi)
    valid = inchi.startswith('InChI=')
    
    features = {
        'has_inchi': int(valid),
        'string_len': len(inchi),
        'has_stereo': int('/t' in inchi or '/m' in inchi or '/s' in inchi),
        'has_charge': int('/q' in inchi),
        'has_isotope': int('/i' in inchi),
        'has_reconnected': int('/r' in inchi),
    }
    
    features.update(counts)
    features['num_heavy_atoms'] = sum(counts[e] for e in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    features['num_heteroatoms'] = sum(counts[e] for e in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    features['CH_ratio'] = counts['C'] / counts['H'] if counts['H'] > 0 else 0.0
    features['OC_ratio'] = counts['O'] / counts['C'] if counts['C'] > 0 else 0.0
    
    return features

def build_feature_matrix(df):
    """Build feature matrix from dataframe."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]
    
    rows = []
    for _, row in df.iterrows():
        volumes = [float(row.get(col, 0.0) or 0.0) for col in vol_cols]
        total_volume = sum(volumes) or 1.0
        normalized = [v / total_volume for v in volumes]
        
        base = {
            'mix_total_volume': total_volume,
            'mix_nonzero_components': sum(1 for v in normalized if v > 0),
            'mix_max_volume': max(normalized),
            'mix_min_nonzero_volume': min((v for v in normalized if v > 0), default=0.0),
            'mix_volume_entropy': -sum(v * np.log(v) for v in normalized if v > 0),
        }
        
        for idx, inchi_col in enumerate(inchi_cols, start=1):
            inchi_value = row.get(inchi_col, '')
            comp = extract_inchi_features(inchi_value)
            vol = normalized[idx - 1]
            
            for key, value in comp.items():
                mix_key = f'mix_{key}'
                base[mix_key] = base.get(mix_key, 0.0) + vol * value
            
            base[f'comp_{idx}_C'] = comp['C']
            base[f'comp_{idx}_H'] = comp['H']
            base[f'comp_{idx}_O'] = comp['O']
            base[f'comp_{idx}_N'] = comp['N']
            base[f'comp_{idx}_S'] = comp['S']
            base[f'comp_{idx}_num_heteroatoms'] = comp['num_heteroatoms']
        
        rows.append(base)
    
    return pd.DataFrame(rows).fillna(0.0)

class InchiCNPredictor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.use_onnx = model_path.endswith('.onnx')
        
        if not os.path.exists(model_path):
            # Try to resolve relative path if not absolute
            possible_path = os.path.join(os.path.dirname(__file__), os.path.basename(model_path))
            if os.path.exists(possible_path):
                self.model_path = possible_path
            else:
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
        if self.use_onnx:
            import onnxruntime as rt
            print(f"Loading ONNX session from {self.model_path}...")
            self.session = rt.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            print(f"Loading Pickle model from {self.model_path}...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features.values.astype(np.float32)
        if self.use_onnx:
            preds = self.session.run(None, {self.input_name: X})[0]
            if len(preds.shape) > 1:
                preds = preds.squeeze(-1)
            return preds
        else:
            return self.model.predict(X)

def main():
    parser = argparse.ArgumentParser(description="Self-contained inference runner for InChI CN predictions.")
    parser.add_argument("--model", type=str, default="Inchi/inchi_cn_model.onnx", help="Path to ONNX (.onnx) or Pickle (.pkl) model.")
    
    # CLI inputs for single mixture
    parser.add_argument("--inchis", type=str, nargs="+", help="List of InChI strings for the components.")
    parser.add_argument("--vols", type=float, nargs="+", help="Corresponding volume fractions.")
    
    # CLI inputs for batch CSV processing
    parser.add_argument("--csv", type=str, help="Path to database-style CSV/TSV file to run inference on.")
    parser.add_argument("--out", type=str, help="Output path for CSV containing predictions.")

    args = parser.parse_args()

    # Load model
    try:
        predictor = InchiCNPredictor(args.model)
    except FileNotFoundError as e:
        # If default fails, try directly resolving in current directory
        if args.model == "Inchi/inchi_cn_model.onnx":
            try:
                predictor = InchiCNPredictor("inchi_cn_model.onnx")
            except FileNotFoundError:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print(f"Error: {e}")
            sys.exit(1)

    # Mode 1: Single mixture prediction
    if not args.csv:
        if not args.inchis or not args.vols:
            parser.error("the following arguments are required when --csv is not specified: --inchis, --vols")
        if len(args.inchis) != len(args.vols):
            parser.error("--inchis and --vols parameters must have matching lengths.")
        if len(args.inchis) > 10:
            parser.error("A maximum of 10 components is supported.")

        # Construct dataframe row representing this single mixture
        row_data = {}
        for i in range(1, 11):
            row_data[f'cpnt_inchi_{i}'] = args.inchis[i-1] if i <= len(args.inchis) else ''
            row_data[f'cpnt_vol_{i}'] = args.vols[i-1] if i <= len(args.vols) else 0.0
        
        df = pd.DataFrame([row_data])
        X = build_feature_matrix(df)
        pred = predictor.predict(X)[0]
        print(f"\nPredicted Mixture Cetane Number (CN): {pred:.4f}")

    # Mode 2: Batch prediction
    else:
        print(f"Processing CSV dataset: {args.csv}")
        # Detect delimiter
        sep = '\t' if args.csv.endswith('.dat') or args.csv.endswith('.tsv') or args.csv.endswith('.txt') else ','
        df_in = pd.read_csv(args.csv, sep=sep)
        
        X = build_feature_matrix(df_in)
        preds = predictor.predict(X)
        
        df_out = df_in.copy()
        df_out["predicted_CN"] = preds
        
        out_path = args.out or "predictions_out.csv"
        # Keep same delimiter for output if tsv/dat, else csv
        df_out.to_csv(out_path, index=False, sep=sep)
        print(f"Saved prediction results to: {out_path}")

if __name__ == "__main__":
    main()
