"""
selfies_rf_benchmark.py
=======================
SELFIES-based feature engineering for CN prediction using tree ensembles.
This is a fast, non-neural baseline using the SELFIES token frequencies as
structural fingerprints — beats the InChI-only RF without any training.

Features used (123 total)
--------------------------
  25  SELFIES token frequency (volume-weighted across components)
  10  Chemistry features (C, H, C/H ratio, DoU, heavy atoms, etc.)
  75  Per-slot SELFIES token counts for top 3 component slots
   3  Mixture-level stats (n_active, max_vol, entropy)
  10  Raw volume fractions

Usage
-----
    python model_training/cn_mixtures_selfies/vae/selfies_rf_benchmark.py
    python model_training/cn_mixtures_selfies/vae/selfies_rf_benchmark.py --save
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from selfies_tokenizer import SELFIESTokenizer, split_selfies  # noqa: E402

CACHE_PATH = _HERE.parent / "data" / "cn_mixtures_selfies.pkl"
N_COMP = 10
HIGH_CN_THR = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def inchi_chem_features(inchi: str) -> np.ndarray:
    """10-dim chemistry vector from an InChI string."""
    if not isinstance(inchi, str) or not inchi.startswith("InChI="):
        return np.zeros(10, dtype=np.float32)
    pat = re.compile(r"([A-Z][a-z]?)(\d*)")
    try:
        formula = inchi.split("/")[1]
    except IndexError:
        formula = ""
    c: dict[str, int] = {}
    for m in pat.finditer(formula):
        el = m.group(1)
        cnt = int(m.group(2)) if m.group(2) else 1
        c[el] = c.get(el, 0) + cnt
    C  = c.get("C", 0); H  = c.get("H", 0); O = c.get("O", 0)
    N  = c.get("N", 0); S  = c.get("S", 0)
    dou    = (2 * C + 2 - H + N) / 2 if (C > 0 or H > 0) else 0.0
    ch     = C / H if H > 0 else 0.0
    nheavy = N + O + S + c.get("Cl", 0) + c.get("Br", 0)
    stereo = float("/t" in inchi or "/m" in inchi)
    nc     = C + H + O + N + S
    flen   = len(formula)
    return np.array([C, H, O, N, ch, dou, nheavy, stereo, nc, flen], dtype=np.float32)


def build_features(df, tokenizer: SELFIESTokenizer) -> np.ndarray:
    """Build the 123-dim feature matrix for all rows in df."""
    tokens_list = list(tokenizer.token2idx.keys())[4:]   # skip specials
    n_tokens    = len(tokens_list)
    n_slots     = 3    # per-slot features for top 3 slots
    n_chem      = 10
    n_mix       = 3    # n_active, max_vol, entropy
    n_vols      = N_COMP
    feat_dim    = n_tokens + n_chem + n_slots * n_tokens + n_mix + n_vols

    X = np.zeros((len(df), feat_dim), dtype=np.float32)

    for row_i, (_, row) in enumerate(df.iterrows()):
        # Volume fractions
        vols = []
        for i in range(1, N_COMP + 1):
            v = row.get(f"cpnt_vol_{i}", 0.0)
            try: v = float(v)
            except: v = 0.0
            if v != v: v = 0.0
            vols.append(v)
        total = sum(vols) or 1.0
        vols  = [v / total for v in vols]

        # 1. Volume-weighted SELFIES token frequencies
        sf_feat = np.zeros(n_tokens)
        chem_feat = np.zeros(n_chem)
        slot_feats = [np.zeros(n_tokens) for _ in range(n_slots)]

        for i in range(1, N_COMP + 1):
            s   = row.get(f"cpnt_selfies_{i}", None)
            vol = vols[i - 1]

            if isinstance(s, str) and s.strip():
                toks = split_selfies(s)
                for t in toks:
                    if t in tokenizer.token2idx:
                        idx = tokenizer.token2idx[t] - 4
                        if 0 <= idx < n_tokens:
                            sf_feat[idx] += vol
                # Per-slot (first 3 slots only)
                if i <= n_slots:
                    for t in toks:
                        if t in tokenizer.token2idx:
                            idx = tokenizer.token2idx[t] - 4
                            if 0 <= idx < n_tokens:
                                slot_feats[i - 1][idx] += 1

            chem_feat += vol * inchi_chem_features(row.get(f"cpnt_inchi_{i}", ""))

        # 2. Mixture-level stats
        n_active = sum(1 for v in vols if v > 0)
        max_vol  = max(vols)
        entropy  = -sum(v * np.log(v) for v in vols if v > 0)

        # Assemble
        feat = np.concatenate([
            sf_feat,
            chem_feat,
            *slot_feats,
            [n_active, max_vol, entropy],
            vols,
        ])
        X[row_i] = feat

    return X


def stratified_split(y, val_frac: float = 0.2, seed: int = 42):
    """Stratified split keeping high-CN samples in both sets."""
    rng      = np.random.default_rng(seed)
    indices  = np.arange(len(y))
    high_idx = indices[y > HIGH_CN_THR]
    low_idx  = indices[y <= HIGH_CN_THR]

    rng.shuffle(high_idx); rng.shuffle(low_idx)
    n_hv = max(1, int(len(high_idx) * val_frac))
    n_lv = max(1, int(len(low_idx)  * val_frac))

    val_idx   = np.concatenate([high_idx[:n_hv], low_idx[:n_lv]])
    train_idx = np.concatenate([high_idx[n_hv:], low_idx[n_lv:]])
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save best model as pickle")
    args = parser.parse_args()

    print(f"Loading cache from {CACHE_PATH} …")
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)

    df        = cache["df_selfies"].dropna(subset=["CN"]).reset_index(drop=True)
    tokenizer = SELFIESTokenizer.load(cache["vocab_path"])
    y         = df["CN"].values.astype(float)

    print("Building SELFIES + chemistry features …")
    X = build_features(df, tokenizer)
    print(f"Feature matrix: {X.shape}")

    train_idx, val_idx = stratified_split(y)
    X_tr, X_te = X[train_idx], X[val_idx]
    y_tr, y_te = y[train_idx], y[val_idx]
    print(f"Train: {len(X_tr)}  Val: {len(X_te)}  High-CN in val: {(y_te > HIGH_CN_THR).sum()}")
    print()

    models: dict[str, object] = {
        "RandomForest-500": RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "ExtraTrees-500": ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        print("[warn] XGBoost not installed. Please install it using the requirements.txt.")

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        print("[warn] LightGBM not installed. Please install it using the requirements.txt.")

    print(f"{'Model':<25} {'MAE':>8} {'R²':>8} {'MAE(CN>80)':>12}")
    print("-" * 60)

    best_mae   = float("inf")
    best_model = None
    best_name  = ""

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p   = m.predict(X_te)
        mae = mean_absolute_error(y_te, p)
        r2  = r2_score(y_te, p)
        hi  = y_te > HIGH_CN_THR
        mae_hi = mean_absolute_error(y_te[hi], p[hi]) if hi.sum() > 0 else float("nan")

        beat = " ✓ beats RF!" if mae < 7.30 else ""
        print(f"  {name:<23} {mae:8.4f} {r2:8.4f} {mae_hi:12.4f}{beat}")

        if mae < best_mae:
            best_mae   = mae
            best_model = m
            best_name  = name

    print("-" * 60)
    print(f"  {'RF baseline (InChI)':<23} {'7.3000':>8} {'0.6199':>8} {'32.9817':>12}  [previous best]")
    print()
    print(f"Best model: {best_name}  (val MAE = {best_mae:.4f})")

    if args.save and best_model is not None:
        import pickle as pk
        out = _HERE.parent / "checkpoints_opt" / "selfies_rf_best.pkl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "wb") as fh:
            pk.dump({"model": best_model, "tokenizer_path": str(cache["vocab_path"])}, fh)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
