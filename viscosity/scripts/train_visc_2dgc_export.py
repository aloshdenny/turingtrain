"""
train_visc_2dgc_export.py
=========================
Train a dual-output viscosity regressor (eta_internal_cP and eta_loglinear_cP)
from 2DGC hydrocarbon-class bin mass fractions plus temperature (T_K) and
pressure (P_bar) using MultiOutputRegressor(HistGradientBoostingRegressor).

Input features (191 total):
  n-paraffin      C1–C30   (30 bins)
  iso-paraffin    C4–C30   (27 bins)
  mono-naphthene  C5–C30   (26 bins)
  di-naphthene    C10–C30  (21 bins)
  tri-naphthene   C14–C30  (17 bins)
  mono-aromatic   C6–C30   (25 bins)
  naphtheno-arom  C9–C30   (22 bins)
  di-aromatic     C10–C30  (21 bins)
  T_K             Temperature in Kelvin
  P_bar           Pressure in bar

Target outputs:
  eta_internal_cP   f-theory viscosity (cP)
  eta_loglinear_cP  log-linear mixing viscosity (cP)

Trained in log10 space for numerical stability and wide dynamic range.
Exported ONNX model includes an ONNX Pow node so predictions are directly in cP.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import joblib

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: skl2onnx / onnxruntime not installed — ONNX export will be skipped.")

# ── Feature spec ──────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "n_paraffin":         list(range(1,  31)),
    "iso_paraffin":       list(range(4,  31)),
    "mono_naphthene":     list(range(5,  31)),
    "di_naphthene":       list(range(10, 31)),
    "tri_naphthene":      list(range(14, 31)),
    "mono_aromatic":      list(range(6,  31)),
    "naphtheno_aromatic": list(range(9,  31)),
    "di_aromatic":        list(range(10, 31)),
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
    cols.extend(["T_K", "P_bar"])
    return cols

TARGETS = ["eta_internal_cP", "eta_loglinear_cP"]

HGBT_PARAMS = dict(
    max_iter=300,
    max_leaf_nodes=63,
    max_depth=None,
    min_samples_leaf=20,
    learning_rate=0.05,
    l2_regularization=0.1,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)

N_SEEDS  = 5
CV_FOLDS = 5


def load_data(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", comment="#")
    print(f"  Total raw rows: {len(df):,}")
    
    # Filter for valid window and liquid phase if columns exist
    if "window_ok" in df.columns:
        df = df[df["window_ok"] == True].copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "liquid"].copy()
    print(f"  Filtered rows (liquid & window_ok): {len(df):,}")
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}")
    missing_targets = [c for c in TARGETS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
        
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df[TARGETS].values.astype(np.float32)
    return X, y, df


def export_to_onnx(model: MultiOutputRegressor, feature_cols: list[str], out_path: Path):
    """
    Exports MultiOutputRegressor to ONNX and appends a Pow(10.0, y) node
    so the graph directly outputs linear viscosity in cP.
    """
    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    
    raw_output_name = onx.graph.output[0].name
    
    # Add base 10.0 constant
    base_tensor = numpy_helper.from_array(np.array([10.0], dtype=np.float32), name="base_10")
    onx.graph.initializer.append(base_tensor)
    
    # Add Pow node: viscosity_cP = 10.0 ** log10_preds
    pow_node = helper.make_node(
        "Pow",
        inputs=["base_10", raw_output_name],
        outputs=["viscosity_cP"],
        name="pow_10"
    )
    onx.graph.node.append(pow_node)
    
    # Replace output description
    onx.graph.output.pop(0)
    onx.graph.output.append(helper.make_tensor_value_info("viscosity_cP", TensorProto.FLOAT, [None, len(TARGETS)]))
    
    onnx.checker.check_model(onx)
    out_path.write_bytes(onx.SerializeToString())
    print(f"  Exported -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def make_parity_plots(y_true_linear: np.ndarray, y_pred_linear: np.ndarray, out_path: Path):
    """Generate high quality side-by-side parity plot for both targets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)
    
    # Target 1: eta_internal_cP (log-log scale because of large dynamic range)
    ax1 = axes[0]
    yt1 = y_true_linear[:, 0]
    yp1 = y_pred_linear[:, 0]
    
    log_yt1 = np.log10(yt1)
    log_yp1 = np.log10(yp1)
    log_mae1 = mean_absolute_error(log_yt1, log_yp1)
    log_r2_1 = r2_score(log_yt1, log_yp1)
    bulk_mask1 = yt1 <= 10.0
    bulk_mae1 = mean_absolute_error(yt1[bulk_mask1], yp1[bulk_mask1])
    
    # Subsample points for fast, crisp scatter plotting if large
    if len(yt1) > 15000:
        idx = np.random.choice(len(yt1), size=15000, replace=False)
    else:
        idx = np.arange(len(yt1))
        
    ax1.scatter(yt1[idx], yp1[idx], alpha=0.25, s=8, color="#1f77b4", edgecolors="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    
    min_val1 = max(1e-2, min(yt1.min(), yp1.min()) * 0.7)
    max_val1 = min(1e9, max(yt1.max(), yp1.max()) * 1.4)
    ax1.plot([min_val1, max_val1], [min_val1, max_val1], color="#d62728", linestyle="--", linewidth=1.5, label="1:1 Identity")
    ax1.set_xlim(min_val1, max_val1)
    ax1.set_ylim(min_val1, max_val1)
    ax1.set_xlabel("True $\eta_{internal}$ (cP)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted $\eta_{internal}$ (cP)", fontsize=11, fontweight="bold")
    ax1.set_title("Viscosity (Internal / f-theory)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(loc="upper left")
    
    textstr1 = f"Log10 MAE: {log_mae1:.3f} decades\nLog10 R²: {log_r2_1:.4f}\nBulk MAE (≤10 cP): {bulk_mae1:.3f} cP"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc")
    ax1.text(0.05, 0.60, textstr1, transform=ax1.transAxes, fontsize=9.5, verticalalignment="top", bbox=props)
    
    # Target 2: eta_loglinear_cP
    ax2 = axes[1]
    yt2 = y_true_linear[:, 1]
    yp2 = y_pred_linear[:, 1]
    
    mae2 = mean_absolute_error(yt2, yp2)
    r2_2 = r2_score(yt2, yp2)
    log_mae2 = mean_absolute_error(np.log10(yt2), np.log10(yp2))
    log_r2_2 = r2_score(np.log10(yt2), np.log10(yp2))
    
    ax2.scatter(yt2[idx], yp2[idx], alpha=0.25, s=8, color="#2ca02c", edgecolors="none")
    min_val2 = 0.0
    max_val2 = max(yt2.max(), yp2.max()) * 1.05
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], color="#d62728", linestyle="--", linewidth=1.5, label="1:1 Identity")
    ax2.set_xlim(min_val2, max_val2)
    ax2.set_ylim(min_val2, max_val2)
    ax2.set_xlabel("True $\eta_{loglinear}$ (cP)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted $\eta_{loglinear}$ (cP)", fontsize=11, fontweight="bold")
    ax2.set_title("Viscosity (Log-linear Mixing)", fontsize=12, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="upper left")
    
    textstr2 = f"Linear MAE: {mae2:.3f} cP\nLinear R²: {r2_2:.4f}\nLog10 MAE: {log_mae2:.3f} decades\nLog10 R²: {log_r2_2:.4f}"
    ax2.text(0.05, 0.70, textstr2, transform=ax2.transAxes, fontsize=9.5, verticalalignment="top", bbox=props)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Parity plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 2DGC viscosity model")
    parser.add_argument("--input", type=Path,
                        default=Path("model_training/viscosity_2dgc/visc_2dgc_selfies_train.dat"),
                        help="Path to training dataset .dat file")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("viscosity/models/visc_2dgc"),
                        help="Output directory for model checkpoints and metadata")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                        help="Number of ensemble seeds")
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="Max parallel jobs (default: 2 to leave cores free)")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = build_feature_cols()
    print(f"Features ({len(feature_cols)}): {feature_cols[:4]} ... {feature_cols[-4:]}")
    print(f"Targets: {TARGETS}")

    X, y, df = load_data(args.input, feature_cols)

    # Transform targets to log10 space
    y_log10 = np.log10(y).astype(np.float32)

    # 1. 5-Fold CV diagnostic on seed 0
    print(f"\n{'='*60}")
    print("5-Fold CV diagnostic (seed 0) ...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_log_maes_int = []
    cv_log_maes_ll = []
    
    fold_idx = 0
    for tr_idx, val_idx in kf.split(X):
        fold_idx += 1
        X_tr_f, X_val_f = X[tr_idx], X[val_idx]
        y_tr_f, y_val_f = y_log10[tr_idx], y_log10[val_idx]
        
        fold_est = MultiOutputRegressor(
            HistGradientBoostingRegressor(**{**HGBT_PARAMS, "random_state": 0}),
            n_jobs=args.n_jobs
        )
        fold_est.fit(X_tr_f, y_tr_f)
        preds_f = fold_est.predict(X_val_f)
        mae_int_f = mean_absolute_error(y_val_f[:, 0], preds_f[:, 0])
        mae_ll_f = mean_absolute_error(y_val_f[:, 1], preds_f[:, 1])
        cv_log_maes_int.append(mae_int_f)
        cv_log_maes_ll.append(mae_ll_f)
        print(f"  Fold {fold_idx}: eta_int Log10 MAE={mae_int_f:.4f} | eta_loglin Log10 MAE={mae_ll_f:.4f}")

    print(f"  CV Log10 MAE eta_internal: {np.mean(cv_log_maes_int):.4f} +/- {np.std(cv_log_maes_int):.4f} decades")
    print(f"  CV Log10 MAE eta_loglinear: {np.mean(cv_log_maes_ll):.4f} +/- {np.std(cv_log_maes_ll):.4f} decades")

    # 2. Train N-seed ensemble (each on 90/10 split)
    print(f"\n{'='*60}")
    print(f"Training {args.n_seeds}-seed ensemble ...")
    models = []
    seed_results = []

    for seed in range(args.n_seeds):
        ts = time.time()
        X_tr, X_val, y_tr, y_val = train_test_split(X, y_log10, test_size=0.1, random_state=seed)
        est = MultiOutputRegressor(
            HistGradientBoostingRegressor(**{**HGBT_PARAMS, "random_state": seed}),
            n_jobs=args.n_jobs
        )
        est.fit(X_tr, y_tr)
        preds_log = est.predict(X_val)
        preds_linear = 10.0 ** preds_log
        y_val_linear = 10.0 ** y_val

        # Metrics for eta_internal
        int_log_mae = mean_absolute_error(y_val[:, 0], preds_log[:, 0])
        int_log_r2 = r2_score(y_val[:, 0], preds_log[:, 0])
        int_bulk_mae = mean_absolute_error(
            y_val_linear[y_val_linear[:, 0] <= 10.0, 0],
            preds_linear[y_val_linear[:, 0] <= 10.0, 0]
        )

        # Metrics for eta_loglinear
        ll_lin_mae = mean_absolute_error(y_val_linear[:, 1], preds_linear[:, 1])
        ll_lin_r2 = r2_score(y_val_linear[:, 1], preds_linear[:, 1])
        ll_log_mae = mean_absolute_error(y_val[:, 1], preds_log[:, 1])

        print(f"  Seed {seed}: eta_int Log10 MAE={int_log_mae:.4f} (Bulk MAE={int_bulk_mae:.3f} cP) | "
              f"eta_loglin MAE={ll_lin_mae:.3f} cP (R2={ll_lin_r2:.4f})  [{time.time()-ts:.1f}s]")

        joblib.dump(est, out_dir / f"visc_2dgc_seed{seed}.joblib")
        models.append(est)
        seed_results.append({
            "seed": seed,
            "eta_internal_log10_mae": float(int_log_mae),
            "eta_internal_log10_r2": float(int_log_r2),
            "eta_internal_bulk_mae_cP": float(int_bulk_mae),
            "eta_loglinear_linear_mae_cP": float(ll_lin_mae),
            "eta_loglinear_linear_r2": float(ll_lin_r2),
            "eta_loglinear_log10_mae": float(ll_log_mae),
        })

    # 3. Ensemble evaluation on independent test split (seed 999)
    print(f"\n{'='*60}")
    print("Ensemble evaluation (independent holdout) ...")
    X_tr_e, X_val_e, y_tr_e, y_val_e = train_test_split(X, y_log10, test_size=0.1, random_state=999)
    
    # Average log predictions across ensemble models
    preds_ens_log = np.mean([m.predict(X_val_e) for m in models], axis=0)
    preds_ens_linear = 10.0 ** preds_ens_log
    y_val_e_linear = 10.0 ** y_val_e

    # Internal metrics
    ens_int_log_mae = mean_absolute_error(y_val_e[:, 0], preds_ens_log[:, 0])
    ens_int_log_r2 = r2_score(y_val_e[:, 0], preds_ens_log[:, 0])
    ens_int_bulk_mae = mean_absolute_error(
        y_val_e_linear[y_val_e_linear[:, 0] <= 10.0, 0],
        preds_ens_linear[y_val_e_linear[:, 0] <= 10.0, 0]
    )
    ens_int_median_ae = median_absolute_error(y_val_e_linear[:, 0], preds_ens_linear[:, 0])

    # Loglinear metrics
    ens_ll_lin_mae = mean_absolute_error(y_val_e_linear[:, 1], preds_ens_linear[:, 1])
    ens_ll_lin_r2 = r2_score(y_val_e_linear[:, 1], preds_ens_linear[:, 1])
    ens_ll_log_mae = mean_absolute_error(y_val_e[:, 1], preds_ens_log[:, 1])
    ens_ll_log_r2 = r2_score(y_val_e[:, 1], preds_ens_log[:, 1])
    ens_ll_median_ae = median_absolute_error(y_val_e_linear[:, 1], preds_ens_linear[:, 1])

    print("  --- Ensemble Results ---")
    print(f"  eta_internal_cP:")
    print(f"    Log10 MAE: {ens_int_log_mae:.4f} decades | Log10 R2: {ens_int_log_r2:.4f}")
    print(f"    Bulk MAE (<= 10 cP): {ens_int_bulk_mae:.3f} cP | Median AE: {ens_int_median_ae:.3f} cP")
    print(f"  eta_loglinear_cP:")
    print(f"    Linear MAE: {ens_ll_lin_mae:.3f} cP | Linear R2: {ens_ll_lin_r2:.4f}")
    print(f"    Log10 MAE: {ens_ll_log_mae:.4f} decades | Log10 R2: {ens_ll_log_r2:.4f}")

    # 4. Save Parity Plot
    parity_path = out_dir / "visc_2dgc_parity.png"
    make_parity_plots(y_val_e_linear, preds_ens_linear, parity_path)

    # Also copy to artifacts directory for user preview if available
    artifacts_dir = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")
    if artifacts_dir.exists():
        import shutil
        shutil.copy2(parity_path, artifacts_dir / "visc_2dgc_parity.png")

    # 5. ONNX Export (seed 0)
    if HAS_ONNX:
        print("\nExporting seed-0 to ONNX with direct cP outputs ...")
        onnx_path = out_dir / "visc_2dgc_seed0.onnx"
        export_to_onnx(models[0], feature_cols, onnx_path)
        
        # Verify ONNX model
        sess = ort.InferenceSession(str(onnx_path))
        test_in = X_val_e[:5].astype(np.float32)
        onnx_preds = sess.run(None, {"float_input": test_in})[0]
        sk_preds = 10.0 ** models[0].predict(test_in)
        np.testing.assert_allclose(onnx_preds, sk_preds, rtol=1e-4)
        print("  ONNX verification PASSED: outputs match sklearn 10**log10 predictions within 1e-4.")

    # 6. Save metadata
    meta = {
        "model": "MultiOutputRegressor(HistGradientBoostingRegressor)",
        "training_space": "log10(viscosity)",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "targets": TARGETS,
        "n_seeds": args.n_seeds,
        "hgbt_params": HGBT_PARAMS,
        "cv_folds": CV_FOLDS,
        "cv_log10_mae_eta_internal": float(np.mean(cv_log_maes_int)),
        "cv_log10_mae_eta_loglinear": float(np.mean(cv_log_maes_ll)),
        "ensemble_metrics": {
            "eta_internal": {
                "log10_mae_decades": float(ens_int_log_mae),
                "log10_r2": float(ens_int_log_r2),
                "bulk_linear_mae_cP": float(ens_int_bulk_mae),
                "median_ae_cP": float(ens_int_median_ae),
            },
            "eta_loglinear": {
                "linear_mae_cP": float(ens_ll_lin_mae),
                "linear_r2": float(ens_ll_lin_r2),
                "log10_mae_decades": float(ens_ll_log_mae),
                "log10_r2": float(ens_ll_log_r2),
                "median_ae_cP": float(ens_ll_median_ae),
            }
        },
        "seed_results": seed_results,
        "train_time_s": time.time() - t0,
    }
    (out_dir / "visc_2dgc_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Metadata saved to {out_dir / 'visc_2dgc_meta.json'}")

    elapsed = time.time() - t0
    print(f"\nAll Done in {elapsed/60:.1f} min. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
