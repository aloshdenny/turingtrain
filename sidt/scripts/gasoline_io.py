"""
gasoline_io.py
──────────────
Data loading and preprocessing utilities for the 10-component gasoline surrogate
dataset (cvwsr_idt_selfies_train.dat / sidt_lhs_gasoline2_K10_1M.dat).

Fixes applied:
1. Fix reader bug: skips only the leading 26 comment lines (skiprows=26)
   instead of comment="#", preventing truncation of lines containing triple-bond
   SELFIES tokens ([#C], [#Branch1], [#Branch2]). Loads all 1,139,500 rows.
2. Fix malformed SELFIES: remaps '[C]#[C]' -> '[C][#C]' for proper tokenization.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch

N_COMPONENTS = 10
MAX_SELFIES_LEN = 65
N_HEADER_COMMENTS = 26

SELFIES_COLS = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
FRAC_COLS    = [f"cpnt_mole_frac_{i}" for i in range(1, N_COMPONENTS + 1)]
COND_COLS    = ["pressure_pa", "temperature_K", "phi", "egr_fraction"]


def count_leading_comments(filepath: Path) -> int:
    """Counts leading lines starting with '#'."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                count += 1
            else:
                break
    return count


def load_table(filepath: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Loads the gasoline IDT simulation dataset safely without dropping '#' lines."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    skiprows = count_leading_comments(filepath)
    if skiprows == 0:
        skiprows = N_HEADER_COMMENTS

    print(f"Loading {filepath.name} (skipping {skiprows} comment header lines)...")
    df = pd.read_csv(filepath, sep="\t", skiprows=skiprows, nrows=nrows, low_memory=False)

    # Remap malformed [C]#[C] -> [C][#C]
    for col in SELFIES_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("[C]#[C]", "[C][#C]", regex=False)

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


def identify_subsets(df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Identifies the unique 10-component SELFIES subsets across the dataset."""
    df_s = df[SELFIES_COLS].fillna("").astype(str)
    unique_df = df_s.drop_duplicates().reset_index(drop=True)
    unique_df["subset_id"] = np.arange(len(unique_df), dtype=np.int64)
    merged = df_s.merge(unique_df, on=SELFIES_COLS, how="left")
    subset_ids = merged["subset_id"].values

    subsets = []
    for uid in range(len(unique_df)):
        row = unique_df.iloc[uid]
        subsets.append({
            "subset_id": uid,
            "selfies_list": [str(row[c]) if pd.notna(row[c]) else "" for c in SELFIES_COLS],
            "row_indices": np.where(subset_ids == uid)[0]
        })
    return subset_ids, subsets


def compute_standardized_conditions(raw_conds: np.ndarray, stats: Dict[str, np.ndarray] | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Computes kinetic condition features:
    raw_conds: [pressure_pa, temperature_K, phi, egr_fraction]
    Returns 5 standardized core features:
    [u_norm, p_norm, phi_norm, egr_norm, log_phi_norm]
    """
    P = np.clip(raw_conds[:, 0], 1.0e3, None)
    T = np.clip(raw_conds[:, 1], 100.0, None)
    phi = np.clip(raw_conds[:, 2], 1.0e-3, None)
    egr = raw_conds[:, 3]

    u = 1000.0 / T
    p = np.log10(P / 1.0e5)
    log_phi = np.log(phi)

    core_feats = np.stack([u, p, phi, egr, log_phi], axis=1).astype(np.float32)

    if stats is None:
        mean = core_feats.mean(axis=0)
        std = core_feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        stats = {"mean": mean, "std": std}

    norm_feats = (core_feats - stats["mean"]) / stats["std"]
    return norm_feats, stats
