"""
prediction_comparison.py
========================
Compare predicted vs actual Cetane Number (CN) and flag outlier samples.

Supports the two prediction exports produced by this project:
  - selfies_vae_predictions.csv              (single CN prediction, val split)
  - selfies_vae_three_output_predictions.csv (branching model outputs)

Usage
-----
    python SELFIES/prediction_comparison.py
    python SELFIES/prediction_comparison.py --input selfies_vae_three_output_predictions.csv --pred-col cn_expected
    python SELFIES/prediction_comparison.py --threshold 5.0
    python SELFIES/prediction_comparison.py --plot-dir SELFIES/plots_prediction_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = _HERE / "selfies_vae_predictions.csv"
DEFAULT_PLOT_DIR = _HERE / "plots_prediction_comparison"


def _resolve_columns(df: pd.DataFrame, pred_col: str | None) -> tuple[str, str]:
    """Return (actual_col, predicted_col) from known CSV layouts."""
    if "CN" in df.columns and "predicted_CN" in df.columns:
        return "CN", pred_col or "predicted_CN"

    if "cn_true" in df.columns:
        candidates = [pred_col] if pred_col else ["cn_mean", "cn_expected", "cn_less_branch", "cn_more_branch"]
        for col in candidates:
            if col and col in df.columns:
                return "cn_true", col
        raise ValueError(
            f"No prediction column found. Available: {list(df.columns)}. "
            "Pass --pred-col explicitly."
        )

    raise ValueError(
        "Unrecognized CSV format. Expected columns like (CN, predicted_CN) "
        "or (cn_true, cn_mean/cn_expected/...)."
    )


def label_outliers(abs_error: pd.Series, threshold: float | None) -> pd.Series:
    """Flag outliers using a fixed MAE threshold or the 1.5×IQR rule."""
    if threshold is not None:
        return abs_error > threshold

    q1 = abs_error.quantile(0.25)
    q3 = abs_error.quantile(0.75)
    iqr = q3 - q1
    cutoff = q3 + 1.5 * iqr
    return abs_error > cutoff


def _outlier_cutoff(abs_error: pd.Series, threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    q1 = abs_error.quantile(0.25)
    q3 = abs_error.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def plot_comparisons(
    work: pd.DataFrame,
    predicted_col: str,
    mae: float,
    rmse: float,
    r2: float,
    cutoff: float,
    plot_dir: Path,
    stem: str,
) -> list[Path]:
    """Save parity, residual, and error-distribution plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    inliers = work[~work["is_outlier"]]
    outliers = work[work["is_outlier"]]
    saved: list[Path] = []

    # 1) Parity plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        inliers["cn_actual"],
        inliers["cn_predicted"],
        s=18,
        alpha=0.45,
        c="steelblue",
        edgecolors="none",
        label="inlier",
    )
    if not outliers.empty:
        ax.scatter(
            outliers["cn_actual"],
            outliers["cn_predicted"],
            s=36,
            alpha=0.85,
            c="crimson",
            edgecolors="white",
            linewidths=0.4,
            label="outlier",
        )

    lo = min(work["cn_actual"].min(), work["cn_predicted"].min())
    hi = max(work["cn_actual"].max(), work["cn_predicted"].max())
    pad = 0.03 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "k--", lw=1.2, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual CN")
    ax.set_ylabel("Predicted CN")
    ax.set_title(f"Predicted vs actual CN\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    parity_path = plot_dir / f"{stem}_parity.png"
    fig.savefig(parity_path, dpi=300)
    plt.close(fig)
    saved.append(parity_path)

    # 2) Residuals vs actual CN
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(
        inliers["cn_actual"],
        inliers["cn_error"],
        s=18,
        alpha=0.45,
        c="steelblue",
        edgecolors="none",
        label="inlier",
    )
    if not outliers.empty:
        ax.scatter(
            outliers["cn_actual"],
            outliers["cn_error"],
            s=36,
            alpha=0.85,
            c="crimson",
            edgecolors="white",
            linewidths=0.4,
            label="outlier",
        )
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(cutoff, color="darkorange", ls="--", lw=1.0, label=f"+{cutoff:.1f} CN")
    ax.axhline(-cutoff, color="darkorange", ls="--", lw=1.0, label=f"-{cutoff:.1f} CN")
    ax.set_xlabel("Actual CN")
    ax.set_ylabel("Prediction error (predicted − actual)")
    ax.set_title("Residuals vs actual CN")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    residual_path = plot_dir / f"{stem}_residuals.png"
    fig.savefig(residual_path, dpi=300)
    plt.close(fig)
    saved.append(residual_path)

    # 3) Absolute error distribution
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        work["cn_abs_error"],
        bins=40,
        color="steelblue",
        alpha=0.75,
        edgecolor="white",
    )
    ax.axvline(cutoff, color="crimson", ls="--", lw=1.5, label=f"outlier cutoff = {cutoff:.2f}")
    ax.set_xlabel("Absolute error |predicted − actual|")
    ax.set_ylabel("Count")
    ax.set_title("Absolute error distribution")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    hist_path = plot_dir / f"{stem}_error_hist.png"
    fig.savefig(hist_path, dpi=300)
    plt.close(fig)
    saved.append(hist_path)

    return saved


def compare_predictions(
    input_path: Path,
    pred_col: str | None = None,
    threshold: float | None = None,
    top_n: int = 15,
    save_path: Path | None = None,
    plot_dir: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    actual_col, predicted_col = _resolve_columns(df, pred_col)

    work = df.copy()
    work["cn_actual"] = pd.to_numeric(work[actual_col], errors="coerce")
    work["cn_predicted"] = pd.to_numeric(work[predicted_col], errors="coerce")
    work = work.dropna(subset=["cn_actual", "cn_predicted"]).reset_index(drop=True)

    work["cn_error"] = work["cn_predicted"] - work["cn_actual"]
    work["cn_abs_error"] = work["cn_error"].abs()
    work["is_outlier"] = label_outliers(work["cn_abs_error"], threshold)

    y_true = work["cn_actual"].values
    y_pred = work["cn_predicted"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    n_outliers = int(work["is_outlier"].sum())

    print(f"Input:      {input_path}")
    print(f"Samples:    {len(work)}")
    print(f"Actual:     {actual_col}")
    print(f"Predicted:  {predicted_col}")
    print(f"MAE:        {mae:.3f}")
    print(f"RMSE:       {rmse:.3f}")
    print(f"R²:         {r2:.3f}")
    print(f"Outliers:   {n_outliers} ({100 * n_outliers / len(work):.1f}%)")

    cutoff = _outlier_cutoff(work["cn_abs_error"], threshold)
    if threshold is None:
        print(f"Outlier rule: |error| > Q3 + 1.5×IQR = {cutoff:.3f}")
    else:
        print(f"Outlier rule: |error| > {cutoff:.3f}")

    id_cols = [c for c in ("No", "mixture_id", "mixture_name", "mixture_type", "split") if c in work.columns]
    outlier_cols = id_cols + [
        "cn_actual",
        "cn_predicted",
        "cn_error",
        "cn_abs_error",
        "is_outlier",
    ]
    outliers = work.loc[work["is_outlier"], outlier_cols].sort_values("cn_abs_error", ascending=False)

    print(f"\nTop {min(top_n, len(outliers))} outliers:")
    if outliers.empty:
        print("  (none)")
    else:
        print(outliers.head(top_n).to_string(index=False))

    if save_path is not None:
        work.to_csv(save_path, index=False)
        print(f"\nSaved comparison CSV to: {save_path}")

    if plot_dir is not None:
        stem = input_path.stem
        plot_paths = plot_comparisons(
            work=work,
            predicted_col=predicted_col,
            mae=mae,
            rmse=rmse,
            r2=r2,
            cutoff=cutoff,
            plot_dir=plot_dir,
            stem=stem,
        )
        print("\nSaved plots:")
        for path in plot_paths:
            print(f"  {path}")

    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CN predictions and label outliers.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Predictions CSV (default: selfies_vae_predictions.csv)",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default=None,
        help="Prediction column for three-output CSVs (default: predicted_CN or cn_mean)",
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
        "--save",
        type=Path,
        default=None,
        help="Optional output CSV path with comparison columns",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory for output plots (default: SELFIES/plots_prediction_comparison)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    compare_predictions(
        input_path=args.input,
        pred_col=args.pred_col,
        threshold=args.threshold,
        top_n=args.top_n,
        save_path=args.save,
        plot_dir=None if args.no_plot else args.plot_dir,
    )


if __name__ == "__main__":
    main()
