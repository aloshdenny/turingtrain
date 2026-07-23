import os
import sys
import argparse
import numpy as np
import pandas as pd
import onnxruntime as rt

def load_and_predict(model_path, inputs):
    """Loads an ONNX model and runs a single prediction."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
        
    sess = rt.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    input_data = np.array([inputs], dtype=np.float32)
    
    res = sess.run(None, {input_name: input_data})
    val = res[0].item() if isinstance(res[0], np.ndarray) and res[0].size == 1 else res[0][0]
    return float(val)

def main():
    parser = argparse.ArgumentParser(description="Inference runner for SIDT forward, inverse, and NTC models.")
    parser.add_argument("--mode", type=str, choices=["forward", "inverse", "ntc"], required=True,
                        help="Prediction mode: 'forward' (predict idt), 'inverse' (predict condition), or 'ntc' (predict NTC presence & bounds)")
    parser.add_argument("--compound", type=str, default="methane",
                        help="Compound model directory to use (e.g., 'methane', 'ethane')")
    parser.add_argument("--target", type=str, choices=["pressure", "temperature", "phi", "egr_fraction"],
                        help="Target variable to predict in inverse mode (required for inverse mode)")
    parser.add_argument("--out_plot", type=str, help="Output plot file path (for forward or NTC temperature sweep)")
    
    # Optional inputs for conditions
    parser.add_argument("--pressure", type=float, help="Pressure (bar)")
    parser.add_argument("--temperature", type=float, help="Temperature (Kelvin)")
    parser.add_argument("--phi", type=float, help="Equivalence ratio phi")
    parser.add_argument("--egr_fraction", type=float, help="Exhaust Gas Recirculation (EGR) fraction (0-1)")
    parser.add_argument("--idt", type=float, help="Ignition Delay Time (s)")
    
    args = parser.parse_args()
    
    # Resolve compound model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compound_dir = os.path.normpath(os.path.join(script_dir, "models", args.compound))
    
    if args.mode == "forward":
        # Forward inputs: pressure, temperature, phi, egr_fraction
        missing = [param for param, val in [("pressure", args.pressure),
                                            ("temperature", args.temperature),
                                            ("phi", args.phi),
                                            ("egr_fraction", args.egr_fraction)] if val is None]
        if missing:
            print(f"Error: Missing required input parameters for forward mode: {', '.join(missing)}")
            sys.exit(1)
            
        model_path = os.path.join(compound_dir, "forward_model.onnx")
        inputs = [args.pressure, args.temperature, args.phi, args.egr_fraction]
        
        try:
            pred_idt = load_and_predict(model_path, inputs)
            print(f"\nForward Model Prediction for {args.compound.capitalize()}:")
            print(f"  Inputs:")
            print(f"    Pressure: {args.pressure:.2f} bar")
            print(f"    Temperature: {args.temperature:.2f} K")
            print(f"    Phi: {args.phi:.4f}")
            print(f"    EGR Fraction: {args.egr_fraction:.4f}")
            print(f"  Output:")
            print(f"    Predicted IDT: {pred_idt:.6f} seconds ({pred_idt * 1000.0:.3f} ms)")
            
            # Generate Temperature Sweep and Arrhenius plot
            try:
                import matplotlib.pyplot as plt
                
                temp_sweep = np.linspace(600.0, 1600.0, 101)
                sweep_inputs = np.array([[args.pressure, t, args.phi, args.egr_fraction] for t in temp_sweep], dtype=np.float32)
                
                sess_sweep = rt.InferenceSession(str(model_path))
                input_name_sweep = sess_sweep.get_inputs()[0].name
                sweep_preds = sess_sweep.run(None, {input_name_sweep: sweep_inputs})[0]
                sweep_idt_ms = sweep_preds.flatten() * 1000.0
                
                df_points = None
                dataset_path = os.path.join(script_dir, "..", "model_training", "sidt", f"sidt_selfies_{args.compound}.dat")
                if os.path.exists(dataset_path):
                    try:
                        df_dat = pd.read_csv(dataset_path, sep='\t', comment='#')
                        mask = (
                            (df_dat['pressure_bar'].between(args.pressure - 0.5, args.pressure + 0.5)) &
                            (df_dat['phi'].between(args.phi - 0.05, args.phi + 0.05)) &
                            (df_dat['egr_fraction'].between(args.egr_fraction - 0.02, args.egr_fraction + 0.02))
                        )
                        df_points = df_dat[mask]
                    except Exception:
                        pass
                
                fig, ax1 = plt.subplots(figsize=(8, 6))
                recip_temp_sweep = 1000.0 / temp_sweep
                ax1.plot(recip_temp_sweep, sweep_idt_ms, label=f"Model Prediction ({args.pressure:.1f} bar, phi={args.phi:.2f}, egr={args.egr_fraction:.2f})", color='#00d2ff', linewidth=2.5)
                
                if df_points is not None and len(df_points) > 0:
                    recip_temp_data = 1000.0 / df_points['temperature_K'].values
                    idt_data_ms = df_points['idt_s'].values * 1000.0
                    ax1.scatter(recip_temp_data, idt_data_ms, color='#ff007f', alpha=0.8, edgecolors='black', zorder=5, label=f"Dataset Points ({len(df_points)} samples)")
                
                ax1.scatter([1000.0 / args.temperature], [pred_idt * 1000.0], color='#ffea00', s=120, edgecolors='black', marker='*', zorder=6, label=f"Prediction: {pred_idt * 1000.0:.2f} ms")
                
                ax1.set_yscale('log')
                ax1.set_xlabel("1000 / Temperature (1/K)", fontsize=11, fontweight='bold')
                ax1.set_ylabel("Ignition Delay Time (ms)", fontsize=11, fontweight='bold')
                ax1.set_title(f"Arrhenius Plot of {args.compound.capitalize()} Ignition Delay Time", fontsize=13, fontweight='bold', pad=15)
                ax1.grid(True, which="both", ls="--", alpha=0.5)
                ax1.legend(frameon=True, facecolor='white', edgecolor='lightgray', loc='upper right')
                
                ax2 = ax1.twiny()
                x1_ticks = ax1.get_xticks()
                x2_labels = [f"{int(round(1000.0 / x)) if x > 0 else 0} K" for x in x1_ticks]
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_xticks(x1_ticks)
                ax2.set_xticklabels(x2_labels)
                ax2.set_xlabel("Temperature (K)", fontsize=10, alpha=0.7)
                
                plot_file = args.out_plot if args.out_plot else os.path.join(script_dir, f"{args.compound}_forward_plot.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved validation Arrhenius plot to: {plot_file}")
                
            except Exception as plot_err:
                print(f"Warning: Could not generate validation plot: {plot_err}")
                
        except Exception as e:
            print(f"Error running forward prediction: {e}")
            sys.exit(1)
            
    elif args.mode == "inverse":
        if not args.target:
            print("Error: Target parameter (--target) is required in inverse mode.")
            sys.exit(1)
            
        if args.target == "pressure":
            missing = [param for param, val in [("temperature", args.temperature),
                                                ("phi", args.phi),
                                                ("egr_fraction", args.egr_fraction),
                                                ("idt", args.idt)] if val is None]
            if missing:
                print(f"Error: Missing required inputs to predict pressure: {', '.join(missing)}")
                sys.exit(1)
                
            model_path = os.path.join(compound_dir, "inverse_pressure_model.onnx")
            inputs = [args.temperature, args.phi, args.egr_fraction, args.idt]
            output_name = "Pressure"
            unit = "bar"
            
        elif args.target == "temperature":
            missing = [param for param, val in [("pressure", args.pressure),
                                                ("phi", args.phi),
                                                ("egr_fraction", args.egr_fraction),
                                                ("idt", args.idt)] if val is None]
            if missing:
                print(f"Error: Missing required inputs to predict temperature: {', '.join(missing)}")
                sys.exit(1)
                
            model_path = os.path.join(compound_dir, "inverse_temperature_model.onnx")
            inputs = [args.pressure, args.phi, args.egr_fraction, args.idt]
            output_name = "Temperature"
            unit = "K"
            
        elif args.target == "phi":
            missing = [param for param, val in [("pressure", args.pressure),
                                                ("temperature", args.temperature),
                                                ("egr_fraction", args.egr_fraction),
                                                ("idt", args.idt)] if val is None]
            if missing:
                print(f"Error: Missing required inputs to predict phi: {', '.join(missing)}")
                sys.exit(1)
                
            model_path = os.path.join(compound_dir, "inverse_phi_model.onnx")
            inputs = [args.pressure, args.temperature, args.egr_fraction, args.idt]
            output_name = "Phi"
            unit = ""
            
        elif args.target == "egr_fraction":
            missing = [param for param, val in [("pressure", args.pressure),
                                                ("temperature", args.temperature),
                                                ("phi", args.phi),
                                                ("idt", args.idt)] if val is None]
            if missing:
                print(f"Error: Missing required inputs to predict egr_fraction: {', '.join(missing)}")
                sys.exit(1)
                
            model_path = os.path.join(compound_dir, "inverse_egr_fraction_model.onnx")
            inputs = [args.pressure, args.temperature, args.phi, args.idt]
            output_name = "EGR Fraction"
            unit = ""
            
        try:
            pred_val = load_and_predict(model_path, inputs)
            print(f"\nInverse Model Prediction for {args.compound.capitalize()}:")
            print(f"  Inputs:")
            if args.target != "pressure": print(f"    Pressure: {args.pressure:.2f} bar")
            if args.target != "temperature": print(f"    Temperature: {args.temperature:.2f} K")
            if args.target != "phi": print(f"    Phi: {args.phi:.4f}")
            if args.target != "egr_fraction": print(f"    EGR Fraction: {args.egr_fraction:.4f}")
            print(f"    IDT: {args.idt:.6f} seconds ({args.idt * 1000.0:.3f} ms)")
            print(f"  Output:")
            print(f"    Predicted {output_name}: {pred_val:.4f} {unit}".strip())
        except Exception as e:
            print(f"Error running inverse prediction: {e}")
            sys.exit(1)

    elif args.mode == "ntc":
        # NTC inputs: pressure, phi, egr_fraction
        missing = [param for param, val in [("pressure", args.pressure),
                                            ("phi", args.phi),
                                            ("egr_fraction", args.egr_fraction)] if val is None]
        if missing:
            print(f"Error: Missing required input parameters for NTC mode: {', '.join(missing)}")
            sys.exit(1)
            
        ntc_dir = os.path.join(compound_dir, "ntc")
        classifier_path = os.path.join(ntc_dir, "has_ntc_classifier.onnx")
        t_min_path = os.path.join(ntc_dir, "ntc_t_min_model.onnx")
        t_max_path = os.path.join(ntc_dir, "ntc_t_max_model.onnx")
        
        inputs = [args.pressure, args.phi, args.egr_fraction]
        
        try:
            has_ntc_val = int(load_and_predict(classifier_path, inputs))
            t_min_val = load_and_predict(t_min_path, inputs)
            t_max_val = load_and_predict(t_max_path, inputs)
            
            print(f"\nNTC Bounds Model Prediction for {args.compound.capitalize()}:")
            print(f"  Inputs:")
            print(f"    Pressure: {args.pressure:.2f} bar")
            print(f"    Phi: {args.phi:.4f}")
            print(f"    EGR Fraction: {args.egr_fraction:.4f}")
            print(f"  Outputs:")
            print(f"    Has NTC Region: {'Yes (1)' if has_ntc_val == 1 else 'No (0)'}")
            print(f"    NTC Lower Temp (T_min): {t_min_val:.1f} K ({t_min_val - 273.15:.1f} °C)")
            print(f"    NTC Upper Temp (T_max): {t_max_val:.1f} K ({t_max_val - 273.15:.1f} °C)")
            
            # Generate Arrhenius curve with highlighted NTC region
            try:
                import matplotlib.pyplot as plt
                fwd_model_path = os.path.join(compound_dir, "forward_model.onnx")
                if os.path.exists(fwd_model_path):
                    temp_sweep = np.linspace(600.0, 1600.0, 200)
                    sweep_inputs = np.column_stack([
                        np.full_like(temp_sweep, args.pressure),
                        temp_sweep,
                        np.full_like(temp_sweep, args.phi),
                        np.full_like(temp_sweep, args.egr_fraction)
                    ]).astype(np.float32)
                    
                    sess_fwd = rt.InferenceSession(fwd_model_path)
                    fwd_in_name = sess_fwd.get_inputs()[0].name
                    idt_preds = sess_fwd.run(None, {fwd_in_name: sweep_inputs})[0].flatten() * 1000.0
                    
                    fig, ax = plt.subplots(figsize=(9, 6))
                    inv_T = 1000.0 / temp_sweep
                    ax.plot(inv_T, idt_preds, color='#1d4ed8', linewidth=2.5, label='Predicted Arrhenius Curve')
                    
                    if has_ntc_val == 1:
                        inv_t_min = 1000.0 / t_min_val
                        inv_t_max = 1000.0 / t_max_val
                        ax.axvspan(inv_t_max, inv_t_min, color='#f59e0b', alpha=0.25, label=f'Predicted NTC Region ({t_min_val:.0f}K – {t_max_val:.0f}K)')
                        ax.axvline(inv_t_min, color='#d97706', linestyle='--', linewidth=1.5)
                        ax.axvline(inv_t_max, color='#d97706', linestyle='--', linewidth=1.5)
                    
                    ax.set_yscale('log')
                    ax.set_title(f"{args.compound.capitalize()} SIDT Ignition Delay & NTC Region (P={args.pressure:.1f} bar, phi={args.phi:.2f})", fontsize=12, fontweight='bold')
                    ax.set_xlabel("1000 / Temperature (1/K)", fontsize=11)
                    ax.set_ylabel("Ignition Delay Time (ms)", fontsize=11)
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax.legend(loc='upper right')
                    
                    plot_file = args.out_plot if args.out_plot else os.path.join(script_dir, f"{args.compound}_ntc_curve.png")
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved NTC curve plot to: {plot_file}")
            except Exception as plot_err:
                print(f"Warning: Could not generate NTC plot: {plot_err}")
                
        except Exception as e:
            print(f"Error running NTC prediction: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
