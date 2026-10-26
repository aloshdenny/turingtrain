"""
inference.py
============
Unified Forward and Inverse Inference Runner for Freezing Point 2DGC Models.

Modes:
  1. Inverse Mode (default when --t_freeze or --celsius is provided):
     Predicts 189 2DGC hydrocarbon mass fraction bins from target freezing point constraint.
       python freezing_point/inference.py --mode inverse --t_freeze 230.0
       python freezing_point/inference.py --celsius -40.0 --out composition_minus40C.csv

  2. Forward Mode:
     Predicts T_freeze_K from a 189-bin composition vector or CSV.
       python freezing_point/inference.py --mode forward --input composition.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "models" / "fp_2dgc"
_FORWARD_ONNX = _MODEL_DIR / "fp_2dgc_seed0.onnx"
_INVERSE_MODEL_DIR = _HERE / "models" / "inverse_fp_2dgc"
_INVERSE_ONNX = _INVERSE_MODEL_DIR / "inverse_fp_2dgc.onnx"
_META_PATH = _INVERSE_MODEL_DIR / "inverse_fp_2dgc_meta.json"

FEATURE_GROUPS = {
    "n_paraffin":         list(range(1,  31)),  # 30 bins
    "iso_paraffin":       list(range(4,  31)),  # 27 bins
    "mono_naphthene":     list(range(5,  31)),  # 26 bins
    "di_naphthene":       list(range(10, 31)),  # 21 bins
    "tri_naphthene":      list(range(14, 31)),  # 17 bins
    "mono_aromatic":      list(range(6,  31)),  # 25 bins
    "naphtheno_aromatic": list(range(9,  31)),  # 22 bins
    "di_aromatic":        list(range(10, 31)),  # 21 bins
}
PREFIXES = {
    "n_paraffin":         "w_n_paraffin_C",
    "iso_paraffin":       "w_iso_paraffin_C",
    "mono_naphthene":     "w_mono_naphthene_C",
    "di_naphthene":       "w_di_naphthene_C",
    "tri_naphthene":      "w_tri_naphthene_C",
    "mono_aromatic":      "w_mono_aromatic_C",
    "naphtheno_aromatic": "w_naphtheno_aromatic_C",
    "di_aromatic":        "w_di_aromatic_C",
}

def build_feature_cols() -> list[str]:
    cols = []
    for group, carbons in FEATURE_GROUPS.items():
        prefix = PREFIXES[group]
        for c in carbons:
            cols.append(f"{prefix}{c}")
    return cols


# ── Forward Inference ──────────────────────────────────────────────────────────

def predict_forward_freezing_point(
    X_189: np.ndarray,
    onnx_path: Path = _FORWARD_ONNX,
) -> float:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Forward ONNX model not found at {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path))
    X_in = X_189.reshape(1, -1).astype(np.float32)
    session_res = sess.run(None, {"float_input": X_in})[0]
    return float(session_res.flatten()[0])


# ── Inverse Inference ──────────────────────────────────────────────────────────

def predict_inverse_composition(
    target_T_K: float,
    onnx_path: Path = _INVERSE_ONNX,
    forward_onnx_path: Path = _FORWARD_ONNX,
) -> dict:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Inverse ONNX model not found at {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path))
    inp = np.array([[float(target_T_K)]], dtype=np.float32)
    w_pred = sess.run(None, {"T_freeze_K": inp})[0][0]  # shape (189,)

    # Verification via forward model if available
    realized_K = None
    parity_error_K = None
    if forward_onnx_path.exists():
        realized_K = predict_forward_freezing_point(w_pred, onnx_path=forward_onnx_path)
        parity_error_K = abs(realized_K - float(target_T_K))

    # Compute group mass fractions
    cols = build_feature_cols()
    class_totals = {}
    idx = 0
    for grp, carbons in FEATURE_GROUPS.items():
        n = len(carbons)
        class_totals[grp] = float(np.sum(w_pred[idx:idx+n]))
        idx += n

    return {
        "target_T_K": float(target_T_K),
        "target_T_C": float(target_T_K - 273.15),
        "realized_T_K": realized_K,
        "realized_T_C": (realized_K - 273.15) if realized_K is not None else None,
        "parity_error_K": parity_error_K,
        "composition_189": w_pred,
        "feature_cols": cols,
        "class_totals": class_totals,
        "sum_mass_fraction": float(np.sum(w_pred)),
    }


# ── CLI Interface ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Freezing Point 2DGC Forward & Inverse Inference Engine.")
    parser.add_argument("--mode", choices=["forward", "inverse"], default=None,
                        help="Inference mode: 'forward' (composition -> T_freeze) or 'inverse' (T_freeze -> composition)")

    # Inverse inputs
    parser.add_argument("--t_freeze", "--t_k", type=float, default=None, help="Target freezing point in Kelvin")
    parser.add_argument("--celsius", "--t_c", type=float, default=None, help="Target freezing point in Celsius")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path for predicted 189-bin composition")

    # Forward inputs
    parser.add_argument("--input", type=Path, default=None, help="Input CSV containing 189-bin composition vector")

    args = parser.parse_args()

    # Determine mode automatically
    if args.mode is None:
        if args.t_freeze is not None or args.celsius is not None:
            args.mode = "inverse"
        elif args.input is not None:
            args.mode = "forward"
        else:
            parser.error("Specify either --t_freeze <Kelvin> (inverse mode) or --input <csv> (forward mode).")

    # ── Execute Inverse Mode ──
    if args.mode == "inverse":
        if args.t_freeze is not None:
            target_K = args.t_freeze
        elif args.celsius is not None:
            target_K = args.celsius + 273.15
        else:
            parser.error("Inverse mode requires --t_freeze <Kelvin> or --celsius <Celsius>.")

        res = predict_inverse_composition(target_K)

        print("\n" + "=" * 65)
        print("Inverse Freezing Point Model: Composition Discovery")
        print("=" * 65)
        print(f"Target Freezing Point   : {res['target_T_K']:.2f} K  ({res['target_T_C']:.2f} °C)")
        if res["realized_T_K"] is not None:
            print(f"Realized Freezing Point : {res['realized_T_K']:.2f} K  ({res['realized_T_C']:.2f} °C)  [Forward Verification]")
            print(f"Parity Error            : {res['parity_error_K']:.2f} K")
        print(f"Total Mass Fraction Sum : {res['sum_mass_fraction']:.5f} (Exact Simplex Conservation)")

        print("\nPredicted Hydrocarbon Class Breakdown:")
        print("-" * 50)
        for grp, total_w in res["class_totals"].items():
            bar = "█" * int(round(total_w * 40))
            print(f"  {grp:<20} : {total_w*100:5.2f}%  {bar}")

        if args.out is not None:
            df_out = pd.DataFrame([res["composition_189"]], columns=res["feature_cols"])
            df_out.to_csv(args.out, index=False)
            print(f"\n✓ Saved 189-bin composition vector to: {args.out}")

        print("=" * 65 + "\n")

    # ── Execute Forward Mode ──
    else:
        if not args.input or not args.input.exists():
            parser.error(f"Input file not found: {args.input}")
        df = pd.read_csv(args.input)
        cols = build_feature_cols()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            parser.error(f"Missing required 2DGC columns: {missing[:5]}")
        X = df[cols].fillna(0.0).values
        pred_K = predict_forward_freezing_point(X[0])
        print("\n" + "=" * 60)
        print("Freezing Point Forward Prediction")
        print("=" * 60)
        print(f"Predicted T_freeze_K : {pred_K:.2f} K")
        print(f"Predicted T_freeze_C : {pred_K - 273.15:.2f} °C")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
