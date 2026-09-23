"""
train_fp_2dgc_export.py
========================
Train a freezing point (T_freeze_K) regressor from 2DGC hydrocarbon-class
bin mass fractions using HistGradientBoostingRegressor (sklearn).

Input features (189 bins total):
  n-paraffin      C1–C30   (30 bins)
  iso-paraffin    C4–C30   (27 bins)
  mono-naphthene  C5–C30   (26 bins)
  di-naphthene    C10–C30  (21 bins)
  tri-naphthene   C14–C30  (17 bins)
  mono-aromatic   C6–C30   (25 bins)
  naphtheno-arom  C9–C30   (22 bins)
  di-aromatic     C10–C30  (21 bins)

Target: T_freeze_K (K)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: skl2onnx not installed — ONNX export will be skipped.")

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
    return cols

TARGET = "T_freeze_K"

HGBT_PARAMS = dict(
    max_iter=500,
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


def load_data(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df["status"] == "ok"].copy()
    print(f"  Clean rows: {len(df):,}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:5]}")
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="model_training/freezing_point_2dgc/fp_2dgc_selfies_train.dat")
    parser.add_argument("--out_dir", default="freezing_point/models/fp_2dgc")
    args = parser.parse_args()

    t0 = time.time()
    feature_cols = build_feature_cols()
    print(f"Features: {len(feature_cols)} bins")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(Path(args.input), feature_cols)
    print(f"X: {X.shape}  |  T_freeze_K: [{y.min():.1f}, {y.max():.1f}] K  mean={y.mean():.1f} K")

    # 5-fold CV diagnostic
    print(f"\n{'='*60}")
    print("5-Fold CV diagnostic (seed 0) ...")
    est0 = HistGradientBoostingRegressor(**{**HGBT_PARAMS, "random_state": 0})
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(est0, X, y, cv=kf,
                              scoring="neg_mean_absolute_error",
                              n_jobs=2, verbose=0)
    print(f"  CV MAE: {cv_mae.mean():.4f} +/- {cv_mae.std():.4f} K")

    # 5-seed ensemble
    print(f"\n{'='*60}")
    print(f"Training {N_SEEDS}-seed ensemble ...")
    models = []
    seed_results = []

    for seed in range(N_SEEDS):
        ts = time.time()
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=seed)
        est = HistGradientBoostingRegressor(**{**HGBT_PARAMS, "random_state": seed})
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2  = r2_score(y_val, y_pred)
        print(f"  Seed {seed}: MAE={mae:.3f} K  R2={r2:.4f}  ({time.time()-ts:.1f}s)  iters={est.n_iter_}")
        joblib.dump(est, out_dir / f"fp_2dgc_seed{seed}.joblib")
        models.append(est)
        seed_results.append({"seed": seed, "val_mae_K": float(mae), "val_r2": float(r2)})

    # Ensemble evaluation
    print(f"\n{'='*60}")
    print("Ensemble evaluation ...")
    X_tr_e, X_val_e, y_tr_e, y_val_e = train_test_split(X, y, test_size=0.1, random_state=999)
    preds_ens = np.mean([m.predict(X_val_e) for m in models], axis=0)
    ens_mae = mean_absolute_error(y_val_e, preds_ens)
    ens_r2  = r2_score(y_val_e, preds_ens)
    print(f"  Ensemble MAE: {ens_mae:.3f} K  R2: {ens_r2:.4f}")

    # Save metadata
    meta = {
        "model": "HistGradientBoostingRegressor",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "target": TARGET,
        "n_seeds": N_SEEDS,
        "hgbt_params": HGBT_PARAMS,
        "cv_mae_mean_K": float(cv_mae.mean()),
        "cv_mae_std_K":  float(cv_mae.std()),
        "ensemble_mae_K": float(ens_mae),
        "ensemble_r2":    float(ens_r2),
        "seed_results": seed_results,
        "train_time_s": time.time() - t0,
    }
    (out_dir / "fp_2dgc_meta.json").write_text(json.dumps(meta, indent=2))

    # ONNX export
    if HAS_ONNX:
        print("\nExporting seed-0 to ONNX ...")
        onnx_model = convert_sklearn(
            models[0],
            initial_types=[("float_input", FloatTensorType([None, len(feature_cols)]))],
            target_opset=17,
        )
        onnx_path = out_dir / "fp_2dgc_seed0.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        print(f"  Exported -> {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Models at: {out_dir}")


if __name__ == "__main__":
    main()
