#!/usr/bin/env python3
"""
inference.py
============
Self-contained forward and inverse prediction script for KeroML.
- Forward Mode: Predicts Cetane Number from a fuel composition matrix CSV.
- Inverse Mode: Predicts a fuel composition matrix from a target Cetane Number constraint.

Requirements:
    Install dependencies using the root requirements.txt:
    pip install -r requirements.txt

Usage (CLI):
    # 1. Forward Mode (Composition -> Cetane Number):
    python KeroML/inference.py --input KeroML/input_sample.csv
    
    # Specify model (pre_brix_model.onnx or post_brix_model.onnx):
    python KeroML/inference.py --input KeroML/input_sample.csv --model KeroML/models/post_brix_model.onnx

    # 2. Inverse Mode (Cetane Number -> Composition):
    python KeroML/inference.py --cn 45.5
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_input_file(filepath):
    """Reads the input composition matrix (carbon numbers x classes) and flattens it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")
        
    df = pd.read_csv(filepath, index_col=0)
    
    # Flatten it into the 160-element array required by our ONNX model
    cols = [
        'n-paraffins', 'iso-paraffins', '1R-cycloparaffins', '2R-cycloparaffins',
        '3R-cycloparaffins', '1R-aromatics', '2R-aromatics', 'cycloaromatics'
    ]
    features = []
    for col in cols:
        for c in range(5, 25):
            idx = f"C{c}"
            val = df.at[idx, col] if col in df.columns and idx in df.index else 0.0
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
    return np.array(features, dtype=np.float32).reshape(1, -1)

def run_forward(model_path, input_path, out_plot_path=None):
    import onnxruntime as rt
    print(f"Loading forward model: {model_path}...")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    input_data = load_input_file(input_path)
    
    # Run Inference
    res = sess.run(None, {input_name: input_data})
    predicted_cn = float(res[0][0][0])
    print(f"Predicted Cetane Number: {predicted_cn:.2f}")

    # Generate a probability distribution map
    model_name = os.path.basename(model_path)
    std_dev = 1.5 if "pre" in model_name else 0.8
    x = np.linspace(predicted_cn - 10, predicted_cn + 10, 1000)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - predicted_cn) / std_dev) ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"CN {model_name}")
    plt.fill_between(x, y, alpha=0.3)
    plt.axvline(x=predicted_cn, color='r', linestyle='--', label=f'Mean: {predicted_cn:.2f}')
    plt.title(f'Probability Distribution of Cetane Number - {model_name}')
    plt.xlabel('Cetane Number')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not out_plot_path:
        out_plot_path = model_name.replace('.onnx', '_dist.png')
        
    plt.savefig(out_plot_path)
    print(f"Saved distribution plot to {out_plot_path}")

def run_inverse(model_path, target_cn, out_csv_path=None, out_plot_path=None):
    import onnxruntime as rt
    print(f"Loading inverse model: {model_path}...")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    # Run Inference
    input_data = np.array([[target_cn]], dtype=np.float32)
    res = sess.run(None, {input_name: input_data})
    predictions = res[0][0]
    
    cols = [
        'n-paraffins', 'iso-paraffins', '1R-cycloparaffins', '2R-cycloparaffins',
        '3R-cycloparaffins', '1R-aromatics', '2R-aromatics', 'cycloaromatics'
    ]
    
    # Reconstruct composition matrix
    df_pred = pd.DataFrame(index=[f"C{c}" for c in range(5, 25)], columns=cols)
    idx = 0
    for col in cols:
        for c in range(5, 25):
            val = predictions[idx]
            df_pred.at[f"C{c}", col] = max(0.0, float(val)) # Bound zero
            idx += 1
            
    if not out_csv_path:
        out_csv_path = f"inverse_cn_{target_cn}.csv"
        
    df_pred.to_csv(out_csv_path)
    print(f"Saved predicted composition matrix to {out_csv_path}")

    # Plot stacked bar chart
    df_pred.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab10')
    plt.title(f'Predicted Carbon Number Distribution for Cetane Number = {target_cn}')
    plt.xlabel('Carbon Number')
    plt.ylabel('Mass Fraction / Concentration')
    plt.legend(title='Compound Class')
    plt.tight_layout()
    
    if not out_plot_path:
        out_plot_path = f"inverse_cn_{target_cn}.png"
        
    plt.savefig(out_plot_path)
    print(f"Saved visualization to {out_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified forward and inverse inference engine for KeroML.")
    parser.add_argument("--mode", type=str, choices=["forward", "inverse"], help="Inference mode: 'forward' (composition to CN) or 'inverse' (CN to composition).")
    parser.add_argument("--model", type=str, help="Path to ONNX model file. If not provided, defaults are selected based on mode.")
    
    # Forward mode inputs
    parser.add_argument("--input", type=str, help="Path to input composition CSV (required for forward mode).")
    
    # Inverse mode inputs
    parser.add_argument("--cn", type=float, help="Target Cetane Number (required for inverse mode).")
    
    # Output paths
    parser.add_argument("--out", type=str, help="Output file path (distribution plot for forward; composition CSV for inverse).")
    parser.add_argument("--out_plot", type=str, help="Output visualization plot path (only used in inverse mode).")

    args = parser.parse_args()

    # Auto-detect mode if not explicitly provided
    if not args.mode:
        if args.cn is not None:
            args.mode = "inverse"
        elif args.input is not None:
            args.mode = "forward"
        else:
            parser.error("must specify either --mode, --input (for forward prediction), or --cn (for inverse prediction)")

    # Enforce mandatory inputs based on mode
    if args.mode == "forward":
        if args.input is None:
            parser.error("the --input argument is required in forward mode")
            
        if not args.model:
            # Default to pre_brix_model
            args.model = "KeroML/models/pre_brix_model.onnx"
            if not os.path.exists(args.model) and os.path.exists("models/pre_brix_model.onnx"):
                args.model = "models/pre_brix_model.onnx"
                
        if not os.path.exists(args.model):
            print(f"Error: Forward model file not found at {args.model}")
            sys.exit(1)
            
        run_forward(args.model, args.input, out_plot_path=args.out)
        
    else: # inverse mode
        if args.cn is None:
            parser.error("the --cn argument is required in inverse mode")
            
        if not args.model:
            args.model = "KeroML/models/inverse_model.onnx"
            if not os.path.exists(args.model) and os.path.exists("models/inverse_model.onnx"):
                args.model = "models/inverse_model.onnx"
                
        if not os.path.exists(args.model):
            print(f"Error: Inverse model file not found at {args.model}")
            sys.exit(1)
            
        run_inverse(args.model, args.cn, out_csv_path=args.out, out_plot_path=args.out_plot)

if __name__ == "__main__":
    main()
