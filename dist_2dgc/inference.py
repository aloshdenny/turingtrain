#!/usr/bin/env python3
"""
inference.py
============
Self-contained forward and inverse prediction script for distillation (dist_2dgc).
- Forward Mode: Predicts distillation curve temperatures (T05-T95) from a composition CSV.
- Inverse Mode: Predicts a composition matrix CSV from target distillation curve temperatures.

Requirements:
    Install dependencies using the root requirements.txt:
    pip install -r requirements.txt

Usage (CLI):
    # 1. Forward Mode (Composition -> Distillation Curve):
    python dist_2dgc/inference.py --input KeroML/scripts/inverse_cn_20.0.csv
    
    # 2. Inverse Mode (Distillation Curve -> Composition):
    python dist_2dgc/inference.py --temps 150 160 175 190 205 220 235 250 265 280 290
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FORWARD_MODEL = os.path.join(
    _SCRIPT_DIR, "models", "distillation_model.onnx"
)
_DEFAULT_INVERSE_MODEL = os.path.join(
    _SCRIPT_DIR, "models", "inverse_model.onnx"
)
_DEFAULT_FORWARD_PLOT = os.path.join(_SCRIPT_DIR, "distillation_curve.png")
_DEFAULT_INVERSE_CSV = os.path.join(_SCRIPT_DIR, "inverse_dist_composition.csv")
_DEFAULT_INVERSE_PLOT = os.path.join(_SCRIPT_DIR, "inverse_dist_composition.png")


def _ensure_parent_dir(filepath):
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)

def load_input_file(filepath):
    """Reads the input composition matrix (carbon numbers x classes) and flattens it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")
        
    df = pd.read_csv(filepath, index_col=0)
    
    # Map input index to standard uppercase format (e.g. C1-C30)
    df.index = [str(idx).strip().upper() for idx in df.index]
    
    # Map input columns using classes_map (supporting both short and long names to internal names)
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
        'synergistic_oxygenates': 'syn_oxy',
        'syn_oxy': 'syn_oxy',
        'antioxidant-oxygenates': 'ant_oxy',
        'antagonistic_oxygenates': 'ant_oxy',
        'ant_oxy': 'ant_oxy',
        'dienes': 'dien',
        'indenes': 'inde'
    }
    
    canonical_classes = sorted(list(set(classes_map.values())))
    
    # Standard carbon numbers 1 to 30
    features = []
    for cls in canonical_classes:
        # Find which column in df maps to this class
        df_col = None
        for col in df.columns:
            col_clean = str(col).strip().lower()
            if col_clean == cls or classes_map.get(col_clean) == cls:
                df_col = col
                break
                
        for c in range(1, 31):
            idx = f"C{c}"
            val = 0.0
            if df_col is not None and idx in df.index:
                val = df.at[idx, df_col]
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
            
    return np.array(features, dtype=np.float32).reshape(1, -1)

def run_forward(model_path, input_path, out_plot_path=None):
    import onnxruntime as rt
    print(f"Loading forward distillation model: {model_path}...")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    input_data = load_input_file(input_path)
    
    # Run Inference
    res = sess.run(None, {input_name: input_data})
    predicted_temps = res[0][0]
    
    targets = ['T05', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90', 'T95']
    vol_fractions = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    
    # 5-fold cross-validated RMSE values per target for uncertainty estimation
    rmse_values = np.array([3.28, 3.23, 3.51, 3.60, 3.83, 4.39, 3.59, 6.34, 7.00, 6.40, 7.39])
    
    print("\nPredicted Distillation Temperatures:")
    for t, temp in zip(targets, predicted_temps):
        print(f"  {t}: {temp:.2f} °C")

    # Generate a beautiful dual-panel visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Subplot 1: Distillation Curve with Shaded Uncertainty Bands ---
    ax1.plot(vol_fractions, predicted_temps, marker='o', linewidth=2.5, color='#1e3a8a', label='Predicted Curve')
    ax1.fill_between(vol_fractions, predicted_temps - rmse_values, predicted_temps + rmse_values, 
                     color='#1e3a8a', alpha=0.15, label='Prediction Uncertainty ($\pm$1 SD)')
    ax1.set_title('Predicted Distillation Curve (ASTM D86)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Volume Distilled (%)', fontsize=10)
    ax1.set_ylabel('Temperature (°C)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(vol_fractions)
    
    # Annotate predicted temperatures on the curve
    for x, y in zip(vol_fractions, predicted_temps):
        ax1.annotate(f"{y:.1f}°C", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#374151')
    ax1.legend(loc='upper left')
    
    # --- Subplot 2: Probability Distribution Curves for Key Points (T10, T50, T90) ---
    key_indices = [1, 5, 9]  # Indices for T10, T50, T90
    colors = ['#2563eb', '#10b981', '#f97316']  # Blue, Emerald, Orange
    
    for idx, color in zip(key_indices, colors):
        target_name = targets[idx]
        mean_val = predicted_temps[idx]
        std_dev = rmse_values[idx]
        
        # Define x range for normal distribution
        x_axis = np.linspace(mean_val - 4 * std_dev, mean_val + 4 * std_dev, 1000)
        y_axis = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mean_val) / std_dev) ** 2)
        
        ax2.plot(x_axis, y_axis, color=color, linewidth=2, label=f'{target_name} ({mean_val:.1f} $\pm$ {std_dev:.2f}°C)')
        ax2.fill_between(x_axis, y_axis, color=color, alpha=0.2)
        ax2.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
        
    ax2.set_title('Probability Distribution Curves (T10, T50, T90)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Temperature (°C)', fontsize=10)
    ax2.set_ylabel('Probability Density', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    if not out_plot_path:
        out_plot_path = _DEFAULT_FORWARD_PLOT
 
    _ensure_parent_dir(out_plot_path)
    plt.savefig(out_plot_path, dpi=300)
    print(f"\nSaved distillation curve and probability distribution plots to: {out_plot_path}")

def run_inverse(model_path, target_temps, out_csv_path=None, out_plot_path=None):
    import onnxruntime as rt
    print(f"Loading inverse distillation model: {model_path}...")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    # Run Inference
    input_data = np.array([target_temps], dtype=np.float32)
    res = sess.run(None, {input_name: input_data})
    predictions = res[0][0]
    
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
        'synergistic_oxygenates': 'syn_oxy',
        'syn_oxy': 'syn_oxy',
        'antioxidant-oxygenates': 'ant_oxy',
        'antagonistic_oxygenates': 'ant_oxy',
        'ant_oxy': 'ant_oxy',
        'dienes': 'dien',
        'indenes': 'inde'
    }
    
    canonical_classes = sorted(list(set(classes_map.values())))
    
    # Reconstruct composition matrix (30 carbons x 13 classes)
    df_pred = pd.DataFrame(index=[f"C{c}" for c in range(1, 31)], columns=canonical_classes)
    idx = 0
    for cls in canonical_classes:
        for c in range(1, 31):
            val = predictions[idx]
            df_pred.at[f"C{c}", cls] = max(0.0, float(val)) # Bound zero
            idx += 1
            
    # Normalize composition so columns/total sums to 1.0 if not zero
    total_sum = df_pred.sum().sum()
    if total_sum > 0:
        df_pred = df_pred / total_sum
        
    # Map column headers back to user-friendly names for display/save
    display_names = {
        'nor_par': 'n-paraffins',
        'iso_par': 'iso-paraffins',
        'mon_nap': '1R-cycloparaffins',
        'di_nap': '2R-cycloparaffins',
        'tri_nap': '3R-cycloparaffins',
        'mon_aro': '1R-aromatics',
        'di_aro': '2R-aromatics',
        'nap_aro': 'cycloaromatics',
        'olef': 'olefins',
        'syn_oxy': 'synergistic_oxygenates',
        'ant_oxy': 'antagonistic_oxygenates',
        'dien': 'dienes',
        'inde': 'indenes'
    }
    df_pred.columns = [display_names.get(col, col) for col in df_pred.columns]
    
    if not out_csv_path:
        out_csv_path = _DEFAULT_INVERSE_CSV
 
    _ensure_parent_dir(out_csv_path)
    df_pred.to_csv(out_csv_path)
    print(f"✓ Saved predicted composition matrix to: {out_csv_path}")

    # Plot stacked bar chart
    df_pred.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
    plt.title(f'Predicted Carbon Number Distribution for Target Distillation Profile', fontsize=13, fontweight='bold')
    plt.xlabel('Carbon Number')
    plt.ylabel('Mass Fraction / Concentration')
    plt.legend(title='Compound Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if not out_plot_path:
        out_plot_path = _DEFAULT_INVERSE_PLOT
 
    _ensure_parent_dir(out_plot_path)
    plt.savefig(out_plot_path, dpi=300)
    print(f"✓ Saved visualization to: {out_plot_path}")


def main():
    # Configure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(description="Unified forward and inverse inference engine for distillation 2D-GC.")
    parser.add_argument("--mode", type=str, choices=["forward", "inverse"], help="Inference mode: 'forward' (composition to curve) or 'inverse' (curve to composition).")
    parser.add_argument("--model", type=str, help="Path to ONNX model file. If not provided, defaults are selected based on mode.")
    
    # Forward mode inputs
    parser.add_argument("--input", type=str, help="Path to input composition CSV (required for forward mode).")
    
    # Inverse mode inputs
    parser.add_argument("--temps", type=float, nargs='+', help="Target temperatures for T05 T10 T20 T30 T40 T50 T60 T70 T80 T90 T95 (11 values required for inverse mode).")
    
    # Output paths
    parser.add_argument("--out", type=str, help="Output file path (distillation curve plot for forward; composition CSV for inverse).")
    parser.add_argument("--out_plot", type=str, help="Output visualization plot path (only used in inverse mode).")

    args = parser.parse_args()

    # Auto-detect mode if not explicitly provided
    if not args.mode:
        if args.temps is not None:
            args.mode = "inverse"
        elif args.input is not None:
            args.mode = "forward"
        else:
            parser.error("must specify either --mode, --input (for forward prediction), or --temps (for inverse prediction)")

    # Enforce inputs based on mode
    if args.mode == "forward":
        if args.input is None:
            parser.error("the --input argument is required in forward mode")
            
        if not args.model:
            args.model = _DEFAULT_FORWARD_MODEL
                
        if not os.path.exists(args.model):
            print(f"Error: Forward model file not found at {args.model}")
            sys.exit(1)
            
        run_forward(args.model, args.input, out_plot_path=args.out)
        
    else: # inverse mode
        if args.temps is None or len(args.temps) != 11:
            parser.error("the --temps argument requires exactly 11 temperature values corresponding to T05 T10 T20 T30 T40 T50 T60 T70 T80 T90 T95")
            
        if not args.model:
            args.model = _DEFAULT_INVERSE_MODEL
                
        if not os.path.exists(args.model):
            print(f"Error: Inverse model file not found at {args.model}")
            sys.exit(1)
            
        run_inverse(args.model, args.temps, out_csv_path=args.out, out_plot_path=args.out_plot)

if __name__ == "__main__":
    main()
