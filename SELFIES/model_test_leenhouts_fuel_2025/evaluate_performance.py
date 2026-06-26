#!/usr/bin/env python3
"""
evaluate_performance.py
=======================
Evaluate ONNX model prediction performance on the Leenhouts et al. (2025)
experimental DCN test dataset.

Uses the local inference module and selfies_vae_optimized.onnx model.

Usage:
    python evaluate_performance.py
    python evaluate_performance.py --threshold 5.0 --no-plot
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from inference import CNInferenceModel  # noqa: E402

DEFAULT_DATA = _HERE / "leenhouts_2025_test_dataset_for_model.dat"
DEFAULT_MODEL = _HERE / "selfies_vae_optimized.onnx"
DEFAULT_CN_MIX = _HERE.parent / "perturbation_testing" / "cn_mixture_selfies.dat"
DEFAULT_OUT_DIR = _HERE / "evaluation_results"
N_COMPONENTS = 10
HIGH_CN_THR = 80.0
PRF_NAMES = {
    "PRF 100", "PRF 95", "PRF 90", "PRF 80", "PRF 75", "PRF 70",
    "PRF 60", "PRF 50", "PRF 40", "PRF 30", "PRF 20", "PRF 10", "PRF 0",
}


def load_mixture_dat(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def _is_active(value) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return bool(s) and s.lower() != "nan"


def selfies_to_inchi(selfies: str) -> str:
    try:
        import selfies as sf
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        smiles = sf.decoder(selfies)
        if not smiles:
            return ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchi(mol) or ""
    except Exception:
        return ""


def extract_components(row: pd.Series, inchi_cache: dict[str, str]) -> list[dict]:
    components: list[dict] = []
    for i in range(1, N_COMPONENTS + 1):
        selfies = row.get(f"cpnt_selfies_{i}")
        vol = row.get(f"cpnt_vol_{i}", 0.0)
        if not _is_active(selfies):
            continue
        try:
            v = float(vol)
        except (TypeError, ValueError):
            v = 0.0
        if v <= 0 or np.isnan(v):
            continue

        sf = selfies.strip()
        if sf not in inchi_cache:
            inchi_cache[sf] = selfies_to_inchi(sf)

        components.append(
            {
                "selfies": sf,
                "vol": v,
                "inchi": inchi_cache[sf],
            }
        )
    return components


def row_to_mixture(row: pd.Series, inchi_cache: dict[str, str]) -> dict | None:
    components = extract_components(row, inchi_cache)
    if not components:
        return None
    return {"components": components}


def predict_all(model: CNInferenceModel, mixtures: list[dict]) -> np.ndarray:
    """Run ONNX inference one mixture at a time (required by the exported model)."""
    preds: list[float] = []
    for mixture in mixtures:
        preds.append(float(model.predict([mixture])[0]))
    return np.array(preds, dtype=np.float64)


def plot_results(work: pd.DataFrame, metrics: dict[str, float], plot_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    y_true = work["cn_actual"].values
    y_pred = work["cn_predicted"].values
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo or 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.5, c="steelblue", edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Experimental DCN")
    ax.set_ylabel("Predicted CN")
    ax.set_title(
        f"Leenhouts 2025 test set (n={len(work)})\n"
        f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.3f}"
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    parity_path = plot_dir / "parity_plot.png"
    fig.savefig(parity_path, dpi=300)
    plt.close(fig)
    saved.append(parity_path)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(work["cn_error"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Prediction error (predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    hist_path = plot_dir / "error_histogram.png"
    fig.savefig(hist_path, dpi=300)
    plt.close(fig)
    saved.append(hist_path)

    return saved


def label_outliers(abs_error: pd.Series, threshold: float | None) -> pd.Series:
    if threshold is not None:
        return abs_error > threshold
    q1 = abs_error.quantile(0.25)
    q3 = abs_error.quantile(0.75)
    return abs_error > (q3 + 1.5 * (q3 - q1))


def outlier_cutoff(abs_error: pd.Series, threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    q1 = abs_error.quantile(0.25)
    q3 = abs_error.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def _row_selfies(row: pd.Series) -> list[str]:
    out: list[str] = []
    for i in range(1, N_COMPONENTS + 1):
        val = row.get(f"cpnt_selfies_{i}")
        if _is_active(val):
            out.append(val.strip())
    return out


def build_selfies_name_map(cn_df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def register(sf: str, name: str, priority: int) -> None:
        current = mapping.get(sf)
        if current is None:
            mapping[sf] = name
            return
        if priority == 2 and current in PRF_NAMES:
            mapping[sf] = name

    for _, row in cn_df.iterrows():
        name = str(row.get("mixture_name", "")).strip()
        if not name or name.lower() == "nan":
            continue
        selfies = _row_selfies(row)
        if row.get("mixture_type", "").strip().lower() == "pure component":
            for sf in selfies:
                register(sf, name, priority=2)
            continue
        if len(selfies) == 1:
            register(selfies[0], name, priority=1)
    return mapping


def selfies_to_common_name(
    selfies: str,
    name_map: dict[str, str],
    cache: dict[str, str],
) -> str:
    if selfies in cache:
        return cache[selfies]
    if selfies in name_map:
        cache[selfies] = name_map[selfies]
        return cache[selfies]
    try:
        import selfies as sf
        from rdkit import Chem

        smiles = sf.decoder(selfies)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            cache[selfies] = Chem.MolToSmiles(mol)
            return cache[selfies]
    except Exception:
        pass
    cache[selfies] = selfies if len(selfies) <= 48 else selfies[:45] + "..."
    return cache[selfies]


def outliers_with_common_names(
    outliers: pd.DataFrame,
    cn_mix_path: Path,
) -> pd.DataFrame:
    out = outliers.copy()
    cn_df = load_mixture_dat(cn_mix_path)
    name_map = build_selfies_name_map(cn_df)
    cache: dict[str, str] = {}

    for i in range(1, N_COMPONENTS + 1):
        sf_col = f"cpnt_selfies_{i}"
        name_col = f"cpnt_name_{i}"
        if sf_col not in out.columns:
            continue
        out[name_col] = out[sf_col].apply(
            lambda val: selfies_to_common_name(val.strip(), name_map, cache)
            if _is_active(val)
            else ""
        )
        out.drop(columns=[sf_col], inplace=True)

    # cpnt_name_* before cpnt_vol_* for readability
    prefix = [c for c in out.columns if not c.startswith("cpnt_")]
    comp_cols: list[str] = []
    for i in range(1, N_COMPONENTS + 1):
        name_col = f"cpnt_name_{i}"
        vol_col = f"cpnt_vol_{i}"
        if name_col in out.columns:
            comp_cols.append(name_col)
        if vol_col in out.columns:
            comp_cols.append(vol_col)
    return out[prefix + comp_cols]


def save_outliers_dat(
    outliers: pd.DataFrame,
    path: Path,
    *,
    cutoff: float,
    threshold: float | None,
    n_total: int,
) -> None:
    if threshold is None:
        rule = f"|error| > Q3 + 1.5×IQR = {cutoff:.3f}"
    else:
        rule = f"|error| > {cutoff:.3f}"

    header = [
        "# Leenhouts 2025 evaluation outliers",
        f"# Outlier rule: {rule}",
        f"# Outliers: {len(outliers)} of {n_total} samples",
        "# Component names from Cheetah pure-component entries in cn_mixture_selfies.dat.",
        "#",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        outliers.to_csv(fh, sep="\t", index=False, float_format="%.6E")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    hi = y_true > HIGH_CN_THR
    mae_hi = float(mean_absolute_error(y_true[hi], y_pred[hi])) if hi.any() else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_hi_cn": mae_hi, "n_hi_cn": int(hi.sum())}


def evaluate(
    data_path: Path,
    model_path: Path,
    out_dir: Path,
    plot: bool,
    threshold: float | None,
    top_n: int,
    cn_mix_path: Path,
) -> pd.DataFrame:
    print(f"Loading dataset: {data_path}")
    df = load_mixture_dat(data_path)
    if "DCN" not in df.columns:
        raise ValueError(f"Expected 'DCN' column in {data_path}")

    actual_col = "DCN"
    inchi_cache: dict[str, str] = {}
    mixtures: list[dict] = []
    row_indices: list[int] = []

    for idx, row in df.iterrows():
        mixture = row_to_mixture(row, inchi_cache)
        if mixture is None:
            continue
        mixtures.append(mixture)
        row_indices.append(idx)

    n_skipped = len(df) - len(mixtures)
    if n_skipped:
        print(f"  Skipped {n_skipped} rows with no active components")

    print(f"Loading model: {model_path}")
    model = CNInferenceModel(model_path)

    print(f"Running inference on {len(mixtures)} mixtures...")
    t0 = time.perf_counter()
    preds = predict_all(model, mixtures)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({len(mixtures) / elapsed:.1f} mixtures/s)")

    work = df.loc[row_indices].copy().reset_index(drop=True)
    work["predicted_CN"] = preds
    work["cn_actual"] = pd.to_numeric(work[actual_col], errors="coerce")
    work["cn_predicted"] = pd.to_numeric(work["predicted_CN"], errors="coerce")
    work = work.dropna(subset=["cn_actual", "cn_predicted"]).reset_index(drop=True)
    work["cn_error"] = work["cn_predicted"] - work["cn_actual"]
    work["cn_abs_error"] = work["cn_error"].abs()
    work["is_outlier"] = label_outliers(work["cn_abs_error"], threshold)
    cutoff = outlier_cutoff(work["cn_abs_error"], threshold)

    y_true = work["cn_actual"].values
    y_pred = work["cn_predicted"].values
    metrics = compute_metrics(y_true, y_pred)

    print()
    print(f"Dataset:         {data_path.name}")
    print(f"Model:           {model_path.name}")
    print(f"Samples:         {len(work)}")
    print(f"MAE:             {metrics['mae']:.3f}")
    print(f"RMSE:            {metrics['rmse']:.3f}")
    print(f"R²:              {metrics['r2']:.3f}")
    if metrics["n_hi_cn"] > 0:
        print(f"MAE (DCN>{HIGH_CN_THR:.0f}): {metrics['mae_hi_cn']:.3f}  (n={metrics['n_hi_cn']})")
    n_outliers = int(work["is_outlier"].sum())
    print(f"Outliers:        {n_outliers} ({100 * n_outliers / len(work):.1f}%)")
    if threshold is None:
        print(f"Outlier rule:    |error| > Q3 + 1.5×IQR = {cutoff:.3f}")
    else:
        print(f"Outlier rule:    |error| > {cutoff:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "leenhouts_2025_predictions.csv"
    work.to_csv(results_path, index=False)
    print(f"\nSaved predictions to: {results_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"dataset: {data_path}",
                f"model: {model_path}",
                f"samples: {len(work)}",
                f"mae: {metrics['mae']:.4f}",
                f"rmse: {metrics['rmse']:.4f}",
                f"r2: {metrics['r2']:.4f}",
                f"mae_dcn_gt_{int(HIGH_CN_THR)}: {metrics['mae_hi_cn']:.4f}",
                f"n_dcn_gt_{int(HIGH_CN_THR)}: {metrics['n_hi_cn']}",
            ]
        )
        + "\n"
    )
    print(f"Saved summary to: {summary_path}")

    outliers = work.loc[work["is_outlier"]].sort_values("cn_abs_error", ascending=False)
    outliers_path = out_dir / "leenhouts_2025_outliers.dat"
    save_outliers_dat(
        outliers_with_common_names(outliers, cn_mix_path),
        outliers_path,
        cutoff=cutoff,
        threshold=threshold,
        n_total=len(work),
    )
    print(f"Saved outliers to: {outliers_path}")

    if plot:
        plot_dir = out_dir / "plots"
        plot_paths = plot_results(work, metrics, plot_dir)
        print("\nSaved plots:")
        for path in plot_paths:
            print(f"  {path}")

    id_cols = [c for c in ("dcn_trn_selfies_id", "mixture_id", "mixture_name", "mixture_type") if c in work.columns]
    show_cols = id_cols + ["cn_actual", "cn_predicted", "cn_error", "cn_abs_error"]
    print(f"\nTop {min(top_n, len(outliers))} outliers:")
    if outliers.empty:
        print("  (none)")
    else:
        print(outliers[show_cols].head(top_n).to_string(index=False))

    return work


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX CN model on the Leenhouts 2025 test dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to leenhouts_2025_test_dataset_for_model.dat",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to selfies_vae_optimized.onnx",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for predictions CSV, summary, and plots",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute-error threshold for outliers; default uses 1.5×IQR rule",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of largest outliers to print",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--cn-mix",
        type=Path,
        default=DEFAULT_CN_MIX,
        help="Cheetah mixture dat for component common-name lookup",
    )
    args = parser.parse_args()

    evaluate(
        data_path=args.data,
        model_path=args.model,
        out_dir=args.out_dir,
        plot=not args.no_plot,
        threshold=args.threshold,
        top_n=args.top_n,
        cn_mix_path=args.cn_mix,
    )


if __name__ == "__main__":
    main()
