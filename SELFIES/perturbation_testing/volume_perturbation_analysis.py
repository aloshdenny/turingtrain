#!/usr/bin/env python3
"""
volume_perturbation_analysis.py
================================
Perturb component volume fractions for mixtures in cn_mixture_selfies.dat
and evaluate how the CN model responds.

Each active component is perturbed by ±pct of its own volume fraction
(e.g. 0.35 → 0.35 ± 0.0035 at 1%), with remaining components adjusted
proportionally so volumes always sum to 1.

By default the script processes **all** mixtures with ≥2 active components.
Use --row-id or --row-index for a single-mixture deep-dive with per-mixture plots.

Usage:
    python volume_perturbation_analysis.py
    python volume_perturbation_analysis.py --limit 50
    python volume_perturbation_analysis.py --row-id CNMX_TRDS_A0_00018
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from inference import CNInferenceModel  # noqa: E402

DEFAULT_DATA = _HERE / "cn_mixture_selfies.dat"
DEFAULT_MODEL = _HERE / "selfies_vae_optimized.onnx"
DEFAULT_OUT_DIR = _HERE / "perturbation_results"
N_COMPONENTS = 10
DEFAULT_PCT = 0.01
DEFAULT_MAX_COMBINED_COMPONENTS = 6


def _log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _count_perturbations(n_comp: int, max_combined_components: int) -> int:
    """Approximate perturbation count for one mixture (after deduplication)."""
    n = 1 + 2 * n_comp  # baseline + single ±pct per component
    if n_comp <= max_combined_components:
        n += max(0, 2**n_comp - 2)  # combined grid minus all-+ / all-− duplicates
    return n


def _collect_eligible_rows(
    df: pd.DataFrame, limit: int | None = None
) -> list[tuple[pd.Series, int]]:
    """Return (row, n_active_components) for mixtures with ≥2 components."""
    eligible: list[tuple[pd.Series, int]] = []
    for _, row in df.iterrows():
        n_comp = len(extract_components(row))
        if n_comp >= 2:
            eligible.append((row, n_comp))
    if limit is not None:
        eligible = eligible[:limit]
    return eligible


def load_mixture_dat(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def _is_active(value) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return bool(s) and s.lower() != "nan"


def extract_components(row: pd.Series) -> list[dict]:
    components: list[dict] = []
    for i in range(1, N_COMPONENTS + 1):
        selfies = row.get(f"cpnt_selfies_{i}")
        vol = row.get(f"cpnt_vol_{i}", 0.0)
        inchi = row.get(f"cpnt_inchi_{i}", "")
        if not _is_active(selfies):
            continue
        try:
            v = float(vol)
        except (TypeError, ValueError):
            v = 0.0
        if v <= 0 or np.isnan(v):
            continue
        components.append(
            {
                "slot": i,
                "selfies": selfies.strip(),
                "vol": v,
                "inchi": inchi if isinstance(inchi, str) else "",
            }
        )
    total = sum(c["vol"] for c in components) or 1.0
    for c in components:
        c["vol"] = c["vol"] / total
    return components


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


def enrich_inchi(components: list[dict], cache: dict[str, str] | None = None) -> list[dict]:
    cache = cache if cache is not None else {}
    out = []
    for c in components:
        inchi = c.get("inchi", "")
        if not inchi:
            sf = c["selfies"]
            if sf not in cache:
                cache[sf] = selfies_to_inchi(sf)
            inchi = cache[sf]
        out.append({**c, "inchi": inchi})
    return out


def components_to_mixture(components: list[dict]) -> dict:
    return {
        "components": [
            {"selfies": c["selfies"], "vol": c["vol"], "inchi": c.get("inchi", "")}
            for c in components
        ]
    }


def perturb_single_component(
    base_vols: np.ndarray,
    target_idx: int,
    sign: int,
    pct: float,
) -> np.ndarray:
    delta = base_vols[target_idx] * pct * sign
    new_vols = base_vols.copy()
    new_vols[target_idx] = base_vols[target_idx] + delta
    others_sum = 1.0 - base_vols[target_idx]
    for j in range(len(base_vols)):
        if j != target_idx:
            new_vols[j] = base_vols[j] - delta * (base_vols[j] / others_sum)
    return new_vols


def perturb_all_components(
    base_vols: np.ndarray,
    signs: tuple[int, ...],
    pct: float,
) -> np.ndarray:
    factors = np.array([1.0 + s * pct for s in signs], dtype=np.float64)
    new_vols = base_vols * factors
    return new_vols / new_vols.sum()


def _sign_pattern_label(signs: tuple[int, ...]) -> str:
    return ",".join(f"{s:+d}%" for s in signs)


def build_perturbations(
    components: list[dict],
    pct: float,
    max_combined_components: int = DEFAULT_MAX_COMBINED_COMPONENTS,
) -> pd.DataFrame:
    """Generate ±pct relative volume perturbations that sum to 1."""
    base_vols = np.array([c["vol"] for c in components], dtype=np.float64)
    n_comp = len(components)
    records: list[dict] = []
    seen_vols: set[tuple] = set()

    def add_record(
        perturbation_type: str,
        perturbed_component: int | None,
        vols: np.ndarray,
        meta: dict | None = None,
    ) -> None:
        key = tuple(round(v, 10) for v in vols)
        if key in seen_vols:
            return
        seen_vols.add(key)
        rec = {
            "perturbation_type": perturbation_type,
            "perturbed_component": perturbed_component,
            "n_active_components": n_comp,
            "perturbation_pct": pct,
        }
        for i, v in enumerate(vols):
            rec[f"vol_comp_{i + 1}"] = v
            rec[f"vol_delta_comp_{i + 1}"] = v - base_vols[i]
        if meta:
            rec.update(meta)
        records.append(rec)

    add_record("baseline", None, base_vols, {"sign_pattern": "baseline", "sign": 0})

    for idx in range(n_comp):
        for sign, label in ((+1, f"+{pct:.2%}"), (-1, f"-{pct:.2%}")):
            vols = perturb_single_component(base_vols, idx, sign, pct)
            add_record(
                "single_component",
                idx + 1,
                vols,
                {"sign": sign, "sign_pattern": label, "target_component": idx + 1},
            )

    if n_comp <= max_combined_components:
        for signs in product((-1, +1), repeat=n_comp):
            vols = perturb_all_components(base_vols, signs, pct)
            add_record(
                "combined",
                None,
                vols,
                {"sign": 0, "sign_pattern": _sign_pattern_label(signs), "signs": str(signs)},
            )

    return pd.DataFrame(records)


def run_predictions(
    perturb_df: pd.DataFrame,
    components: list[dict],
    engine: CNInferenceModel,
    verbose: bool = False,
    mixture_label: str = "",
) -> np.ndarray:
    preds: list[float] = []
    rows = list(perturb_df.iterrows())
    n = len(rows)
    row_iter = rows
    if verbose and n > 1:
        row_iter = tqdm(
            rows,
            desc=f"  inference {mixture_label[:40]}",
            unit="pert",
            leave=False,
            file=sys.stderr,
        )

    for idx, (_, row) in enumerate(row_iter):
        perturbed = [
            {
                "selfies": components[i]["selfies"],
                "vol": float(row[f"vol_comp_{i + 1}"]),
                "inchi": components[i].get("inchi", ""),
            }
            for i in range(len(components))
        ]
        pred = engine.predict([components_to_mixture(perturbed)])
        val = float(np.atleast_1d(pred)[0])
        preds.append(val)
        if verbose and n <= 25:
            ptype = row.get("perturbation_type", "")
            pattern = row.get("sign_pattern", "")
            _log(f"    [{idx + 1}/{n}] {ptype} {pattern}: pred={val:.4f}", verbose=verbose)
    return np.array(preds, dtype=np.float64)


def _mixture_meta(row: pd.Series) -> dict:
    return {
        "cn_mxtr_selfies_id": row["cn_mxtr_selfies_id"],
        "mixture_id": row.get("mixture_id", ""),
        "mixture_name": row.get("mixture_name", ""),
        "mixture_type": row.get("mixture_type", ""),
        "actual_CN": float(row["CN"]),
    }


def process_one_mixture(
    row: pd.Series,
    engine: CNInferenceModel,
    pct: float,
    inchi_cache: dict[str, str],
    max_combined_components: int,
    verbose: bool = False,
) -> pd.DataFrame | None:
    components = extract_components(row)
    if len(components) < 2:
        return None

    components = enrich_inchi(components, inchi_cache)
    perturb_df = build_perturbations(components, pct, max_combined_components)
    label = str(row.get("mixture_name", row["cn_mxtr_selfies_id"]))
    if verbose:
        _log(
            f"  Building {len(perturb_df)} perturbations for {label} "
            f"({len(components)} components, CN={float(row['CN']):.2f})",
            verbose=verbose,
        )
    preds = run_predictions(perturb_df, components, engine, verbose=verbose, mixture_label=label)

    meta = _mixture_meta(row)
    out = perturb_df.copy()
    out["predicted_CN"] = preds
    for key, val in meta.items():
        out[key] = val

    baseline_pred = float(out.loc[out["perturbation_type"] == "baseline", "predicted_CN"].iloc[0])
    out["baseline_predicted_CN"] = baseline_pred
    out["error"] = out["predicted_CN"] - meta["actual_CN"]
    out["abs_error"] = out["error"].abs()
    out["delta_pred_from_baseline"] = out["predicted_CN"] - baseline_pred
    out["abs_delta_pred_from_baseline"] = out["delta_pred_from_baseline"].abs()
    return out


def build_mixture_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, grp in all_results.groupby("cn_mxtr_selfies_id", sort=False):
        baseline = grp[grp["perturbation_type"] == "baseline"].iloc[0]
        perturbed = grp[grp["perturbation_type"] != "baseline"]
        rows.append(
            {
                "cn_mxtr_selfies_id": mid,
                "mixture_id": baseline["mixture_id"],
                "mixture_name": baseline["mixture_name"],
                "mixture_type": baseline.get("mixture_type", ""),
                "actual_CN": baseline["actual_CN"],
                "n_active_components": int(baseline["n_active_components"]),
                "baseline_predicted_CN": baseline["predicted_CN"],
                "baseline_error": baseline["error"],
                "baseline_abs_error": baseline["abs_error"],
                "n_perturbations": len(grp),
                "n_nonbaseline": len(perturbed),
                "pred_min": grp["predicted_CN"].min(),
                "pred_max": grp["predicted_CN"].max(),
                "pred_range": grp["predicted_CN"].max() - grp["predicted_CN"].min(),
                "max_abs_delta_from_baseline": perturbed["abs_delta_pred_from_baseline"].max()
                if len(perturbed)
                else 0.0,
                "mean_abs_delta_from_baseline": perturbed["abs_delta_pred_from_baseline"].mean()
                if len(perturbed)
                else 0.0,
                "max_abs_error": grp["abs_error"].max(),
                "mean_abs_error": grp["abs_error"].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_all_mixtures(
    df: pd.DataFrame,
    engine: CNInferenceModel,
    pct: float,
    max_combined_components: int,
    limit: int | None = None,
    verbose: bool = True,
    log_every: int = 25,
) -> pd.DataFrame:
    inchi_cache: dict[str, str] = {}
    frames: list[pd.DataFrame] = []

    _log("Scanning dataset for eligible mixtures (≥2 active components)...", verbose=verbose)
    eligible = _collect_eligible_rows(df, limit)
    est_inferences = sum(
        _count_perturbations(n_comp, max_combined_components) for _, n_comp in eligible
    )
    _log(f"  Eligible mixtures : {len(eligible):,}", verbose=verbose)
    _log(f"  Est. inferences   : {est_inferences:,}  (±{pct:.2%} perturbations)", verbose=verbose)
    _log(f"  Progress interval : every {log_every} mixtures\n", verbose=verbose)

    t0 = time.perf_counter()
    total_perturbations = 0

    mixture_iter = enumerate(
        tqdm(eligible, desc="Mixtures", unit="mixture", file=sys.stderr, mininterval=0.5),
        start=1,
    )
    for i, (row, n_comp) in mixture_iter:
        row_id = row["cn_mxtr_selfies_id"]
        name = str(row.get("mixture_name", row_id))

        result = process_one_mixture(
            row, engine, pct, inchi_cache, max_combined_components, verbose=False
        )
        if result is not None:
            frames.append(result)
            total_perturbations += len(result)

        elapsed = time.perf_counter() - t0
        should_log = verbose and (i == 1 or i % log_every == 0 or i == len(eligible))
        if should_log and result is not None:
            baseline = result[result["perturbation_type"] == "baseline"].iloc[0]
            rate = (i / elapsed) * 60 if elapsed > 0 else 0.0
            eta = ((len(eligible) - i) / (i / elapsed)) if i > 0 and elapsed > 0 else 0.0
            _log(
                f"[{i:,}/{len(eligible):,}] {name[:45]:45s}  "
                f"n_comp={n_comp}  perts={len(result)}  "
                f"CN={baseline['actual_CN']:.2f}  pred={baseline['predicted_CN']:.2f}  "
                f"err={baseline['error']:+.3f}  "
                f"elapsed={_format_duration(elapsed)}  "
                f"rate={rate:.1f} mix/min  "
                f"ETA={_format_duration(eta)}  "
                f"inferences={total_perturbations:,}",
                verbose=verbose,
            )

    elapsed = time.perf_counter() - t0
    if not frames:
        raise RuntimeError("No eligible mixtures were processed.")

    _log(
        f"\nFinished {len(frames):,} mixtures in {_format_duration(elapsed)} "
        f"({total_perturbations:,} inferences, "
        f"{total_perturbations / elapsed:.1f} inf/s)",
        verbose=verbose,
    )
    return pd.concat(frames, ignore_index=True)


def make_aggregate_plots(all_results: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, pct: float) -> list[Path]:
    plot_dir = out_dir / "aggregate_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    pct_label = f"{pct:.2%}"

    baselines = all_results[all_results["perturbation_type"] == "baseline"].copy()
    perturbed = all_results[all_results["perturbation_type"] != "baseline"].copy()

    # 1) Baseline parity across all mixtures
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        baselines["actual_CN"],
        baselines["predicted_CN"],
        s=12,
        alpha=0.35,
        c="steelblue",
        edgecolors="none",
    )
    lo = min(baselines["actual_CN"].min(), baselines["predicted_CN"].min())
    hi = max(baselines["actual_CN"].max(), baselines["predicted_CN"].max())
    pad = (hi - lo) * 0.03
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Actual CN")
    ax.set_ylabel("Predicted CN (baseline, unperturbed)")
    ax.set_title(f"Baseline predictions — {len(baselines):,} mixtures")
    fig.tight_layout()
    p = plot_dir / "01_baseline_parity_all_mixtures.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 2) Baseline error distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(baselines["error"], bins=60, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Baseline prediction error (pred − actual)")
    ax.set_ylabel("Number of mixtures")
    ax.set_title("Baseline error distribution")
    fig.tight_layout()
    p = plot_dir / "02_baseline_error_histogram.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 3) |Δpred| from baseline for all perturbations
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        perturbed["abs_delta_pred_from_baseline"],
        bins=60,
        color="darkorange",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xlabel("|Δ predicted CN| from baseline")
    ax.set_ylabel("Count (perturbed compositions)")
    ax.set_title(f"Model response to ±{pct_label} volume perturbations ({len(perturbed):,} rows)")
    fig.tight_layout()
    p = plot_dir / "03_delta_pred_histogram.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 4) Boxplot of |Δpred| by perturbation type
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["single_component", "combined"]
    data = perturbed[perturbed["perturbation_type"].isin(order)]
    if not data.empty:
        sns.boxplot(
            data=data,
            x="perturbation_type",
            y="abs_delta_pred_from_baseline",
            hue="perturbation_type",
            order=order,
            ax=ax,
            palette={"single_component": "steelblue", "combined": "teal"},
            legend=False,
        )
        ax.set_xlabel("Perturbation type")
        ax.set_ylabel("|Δ predicted CN| from baseline")
        ax.set_title(f"Perturbation sensitivity by type (±{pct_label})")
    fig.tight_layout()
    p = plot_dir / "04_delta_pred_by_perturbation_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 5) Max sensitivity per mixture vs baseline error
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        summary["baseline_abs_error"],
        summary["max_abs_delta_from_baseline"],
        s=14,
        alpha=0.4,
        c=summary["n_active_components"],
        cmap="viridis",
        edgecolors="none",
    )
    cbar = plt.colorbar(ax.collections[0], ax=ax, label="# active components")
    ax.set_xlabel("Baseline |error| (pred − actual)")
    ax.set_ylabel("Max |Δpred| from baseline under ±1% perturbations")
    ax.set_title("Accuracy vs volume-perturbation sensitivity")
    fig.tight_layout()
    p = plot_dir / "05_baseline_error_vs_sensitivity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 6) Top 25 most sensitive mixtures
    top = summary.nlargest(25, "max_abs_delta_from_baseline")
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = top["mixture_name"].astype(str).str.slice(0, 40)
    ax.barh(range(len(top)), top["max_abs_delta_from_baseline"], color="indianred", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Max |Δpred| from baseline")
    ax.set_title("Top 25 most volume-sensitive mixtures")
    fig.tight_layout()
    p = plot_dir / "06_top_sensitive_mixtures.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 7) Sensitivity vs number of components
    fig, ax = plt.subplots(figsize=(8, 5))
    by_n = summary.groupby("n_active_components")["max_abs_delta_from_baseline"]
    stats = by_n.agg(["median", "mean", "max"]).reset_index()
    ax.plot(stats["n_active_components"], stats["median"], "o-", label="Median", lw=2)
    ax.plot(stats["n_active_components"], stats["mean"], "s--", label="Mean", lw=2)
    ax.set_xlabel("Number of active components")
    ax.set_ylabel("Max |Δpred| from baseline")
    ax.set_title(f"Perturbation sensitivity vs mixture complexity (±{pct_label})")
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "07_sensitivity_vs_n_components.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 8) Perturbed error vs actual CN (all perturbations)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        perturbed["actual_CN"],
        perturbed["error"],
        c=perturbed["abs_delta_pred_from_baseline"],
        s=8,
        alpha=0.25,
        cmap="YlOrRd",
        edgecolors="none",
    )
    ax.axhline(0, color="black", lw=1)
    plt.colorbar(sc, ax=ax, label="|Δpred| from baseline")
    ax.set_xlabel("Actual CN")
    ax.set_ylabel("Prediction error (pred − actual)")
    ax.set_title("Perturbed prediction errors coloured by model response")
    fig.tight_layout()
    p = plot_dir / "08_perturbed_errors_vs_actual_cn.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def make_single_mixture_plots(
    results: pd.DataFrame,
    actual_cn: float,
    mixture_name: str,
    row_id: str,
    out_dir: Path,
    pct: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

    results = results.copy()
    if "error" not in results.columns:
        results["error"] = results["predicted_CN"] - actual_cn
        results["abs_error"] = results["error"].abs()

    baseline = results[results["perturbation_type"] == "baseline"].iloc[0]
    baseline_pred = float(baseline["predicted_CN"])
    baseline_err = float(baseline["error"])
    n_comp = int(baseline["n_active_components"])
    pct_label = f"{pct:.2%}"
    single = results[results["perturbation_type"] == "single_component"].copy()
    combined = results[results["perturbation_type"] == "combined"].copy()

    if not single.empty:
        labels, preds, colors = [], [], []
        for i in range(n_comp):
            for sign, color in ((+1, "steelblue"), (-1, "salmon")):
                row = single[(single["perturbed_component"] == i + 1) & (single["sign"] == sign)]
                if row.empty:
                    continue
                labels.append(f"Comp {i + 1}\n{'+' if sign > 0 else '-'}{pct_label}")
                preds.append(float(row.iloc[0]["predicted_CN"]))
                colors.append(color)

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
        x = np.arange(len(labels))
        ax.bar(x, preds, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(actual_cn, color="crimson", ls="--", lw=2, label=f"Actual CN ({actual_cn:.2f})")
        ax.axhline(baseline_pred, color="gray", ls=":", lw=2, label=f"Baseline pred ({baseline_pred:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Predicted CN")
        ax.set_title(f"Single-component ±{pct_label} — {mixture_name}")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "01_single_component_predictions.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        errors = [p - actual_cn for p in preds]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
        ax.bar(x, errors, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(baseline_err, color="gray", ls=":", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Prediction error (pred − actual)")
        ax.set_title(f"Single-component ±{pct_label} errors — {mixture_name}")
        fig.tight_layout()
        p = out_dir / "02_single_component_errors.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    if not combined.empty:
        comb = combined.sort_values("sign_pattern")
        fig, ax = plt.subplots(figsize=(max(10, len(comb) * 0.45), 5))
        x = np.arange(len(comb))
        ax.bar(x, comb["predicted_CN"], color="teal", alpha=0.75)
        ax.axhline(actual_cn, color="crimson", ls="--", lw=2)
        ax.axhline(baseline_pred, color="gray", ls=":", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(comb["sign_pattern"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Predicted CN")
        ax.set_title(f"Combined ±{pct_label} — {mixture_name}")
        fig.tight_layout()
        p = out_dir / "03_combined_predictions.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    perturbed = results[results["perturbation_type"] != "baseline"].copy()
    perturbed["delta_pred"] = perturbed["predicted_CN"] - baseline_pred
    fig, ax = plt.subplots(figsize=(6.5, 6))
    palette = {"baseline": "black", "single_component": "steelblue", "combined": "teal"}
    for ptype, grp in results.groupby("perturbation_type"):
        ax.scatter(
            [actual_cn] * len(grp),
            grp["predicted_CN"],
            s=80 if ptype == "baseline" else 40,
            alpha=0.95 if ptype == "baseline" else 0.65,
            c=palette.get(ptype, "gray"),
            label=ptype,
            edgecolors="black",
            linewidths=0.3,
        )
    spread = results["predicted_CN"].max() - results["predicted_CN"].min()
    pad = max(0.5, spread * 0.3)
    lims = [actual_cn - pad, actual_cn + pad]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim([results["predicted_CN"].min() - pad * 0.2, results["predicted_CN"].max() + pad * 0.2])
    ax.set_xlabel("Actual CN")
    ax.set_ylabel("Predicted CN")
    ax.set_title(f"All ±{pct_label} perturbations — {mixture_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "04_all_perturbations_parity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def print_aggregate_report(all_results: pd.DataFrame, summary: pd.DataFrame, pct: float) -> None:
    baselines = all_results[all_results["perturbation_type"] == "baseline"]
    perturbed = all_results[all_results["perturbation_type"] != "baseline"]
    print("\n" + "=" * 72)
    print("VOLUME PERTURBATION ANALYSIS — ALL MIXTURES")
    print("=" * 72)
    print(f"Perturbation size     : ±{pct:.2%} of each component volume")
    print(f"Mixtures processed    : {len(summary):,}")
    print(f"Total perturbation rows: {len(all_results):,}  (incl. {len(baselines):,} baselines)")
    print(f"Non-baseline rows     : {len(perturbed):,}")
    print(f"Baseline MAE          : {baselines['abs_error'].mean():.4f}")
    print(f"Baseline RMSE         : {np.sqrt((baselines['error'] ** 2).mean()):.4f}")
    print(f"Mean |Δpred| (pert.)  : {perturbed['abs_delta_pred_from_baseline'].mean():.6f}")
    print(f"Max |Δpred| (pert.)   : {perturbed['abs_delta_pred_from_baseline'].max():.6f}")
    print(f"Mean max |Δpred|/mix  : {summary['max_abs_delta_from_baseline'].mean():.6f}")
    print("\nBy perturbation type:")
    for ptype, grp in perturbed.groupby("perturbation_type"):
        print(
            f"  {ptype:18s}  n={len(grp):7,d}  "
            f"mean|Δpred|={grp['abs_delta_pred_from_baseline'].mean():.6f}  "
            f"max|Δpred|={grp['abs_delta_pred_from_baseline'].max():.6f}"
        )
    print("\nMost sensitive mixtures:")
    for _, r in summary.nlargest(5, "max_abs_delta_from_baseline").iterrows():
        print(
            f"  {r['mixture_name'][:50]:50s}  "
            f"max|Δpred|={r['max_abs_delta_from_baseline']:.4f}  "
            f"baseline_err={r['baseline_error']:+.3f}"
        )
    print("=" * 72 + "\n")


def select_row(df: pd.DataFrame, row_index: int | None, row_id: str | None) -> pd.Series:
    if row_id is not None:
        mask = df["cn_mxtr_selfies_id"] == row_id
        if not mask.any():
            raise ValueError(f"Row id not found: {row_id}")
        return df.loc[mask].iloc[0]
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index {row_index} out of range [0, {len(df) - 1}]")
        return df.iloc[row_index]
    raise ValueError("No row selector provided")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturb mixture volume fractions and analyse CN model sensitivity."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--row-index", type=int, default=None, help="Analyse one mixture by index")
    parser.add_argument("--row-id", type=str, default=None, help="Analyse one mixture by cn_mxtr_selfies_id")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pct", type=float, default=DEFAULT_PCT, help="Relative perturbation (default 0.01 = 1%%)")
    parser.add_argument(
        "--max-combined-components",
        type=int,
        default=DEFAULT_MAX_COMBINED_COMPONENTS,
        help="Max active components for combined ±pct sign-grid (default 6)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N eligible mixtures")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Print progress (default: on)")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log a status line every N mixtures in all-mixtures mode (default: 25)",
    )
    args = parser.parse_args()
    verbose = args.verbose

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    if args.pct <= 0 or args.pct >= 1:
        parser.error("--pct must be between 0 and 1 (e.g. 0.01 for 1%)")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")

    _log(f"Loading dataset: {args.data}", verbose=verbose)
    df = load_mixture_dat(args.data)
    _log(f"  Loaded {len(df):,} rows", verbose=verbose)
    _log(f"Loading model: {args.model}", verbose=verbose)
    engine = CNInferenceModel(args.model, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    single_mode = args.row_id is not None or args.row_index is not None

    if single_mode:
        row = select_row(df, args.row_index, args.row_id)
        row_id = row["cn_mxtr_selfies_id"]
        mixture_name = row.get("mixture_name", row_id)
        actual_cn = float(row["CN"])
        inchi_cache: dict[str, str] = {}

        _log(f"Single mixture: {mixture_name} ({row_id})", verbose=verbose)
        results = process_one_mixture(
            row, engine, args.pct, inchi_cache, args.max_combined_components, verbose=verbose
        )
        if results is None:
            raise ValueError(f"Mixture {row_id} has fewer than 2 active components.")

        mix_dir = args.out_dir / row_id.replace("/", "_")
        mix_dir.mkdir(parents=True, exist_ok=True)
        csv_path = mix_dir / f"{row_id.replace('/', '_')}_perturbation_results.csv"
        results.to_csv(csv_path, index=False)
        _log(f"Saved: {csv_path}", verbose=verbose)

        _log("Generating plots...", verbose=verbose)
        plot_paths = make_single_mixture_plots(results, actual_cn, mixture_name, row_id, mix_dir, args.pct)
        for p in plot_paths:
            _log(f"  {p}", verbose=verbose)
        _log(f"Done. Outputs in {mix_dir}", verbose=verbose)
        return

    # All mixtures mode (default)
    all_results = run_all_mixtures(
        df,
        engine,
        args.pct,
        args.max_combined_components,
        limit=args.limit,
        verbose=verbose,
        log_every=args.log_every,
    )

    _log("Building per-mixture summary...", verbose=verbose)
    summary = build_mixture_summary(all_results)

    full_csv = args.out_dir / "all_mixtures_perturbation_results.csv"
    summary_csv = args.out_dir / "all_mixtures_summary.csv"
    all_results.to_csv(full_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _log(f"Saved full results : {full_csv}  ({len(all_results):,} rows)", verbose=verbose)
    _log(f"Saved summary      : {summary_csv}  ({len(summary):,} mixtures)", verbose=verbose)

    _log("Generating aggregate plots...", verbose=verbose)
    plot_paths = make_aggregate_plots(all_results, summary, args.out_dir, args.pct)
    for p in plot_paths:
        _log(f"  {p}", verbose=verbose)

    if verbose:
        print_aggregate_report(all_results, summary, args.pct)
    _log(f"Done. Outputs in {args.out_dir}", verbose=verbose)


if __name__ == "__main__":
    main()
