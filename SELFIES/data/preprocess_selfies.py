"""
preprocess_selfies.py
=====================
Convert the raw CN-mixture InChI dataset into a SELFIES-tokenised cache.

Run once before training:

    cd turingtrain
    python model_training/cn_mixtures_selfies/data/preprocess_selfies.py

Outputs (in the same directory as this script)
-----------------------------------------------
cn_mixtures_selfies.pkl  — pickled dict with keys:
    'df_selfies'   : pd.DataFrame with all original columns plus
                     cpnt_selfies_1 … cpnt_selfies_10
    'selfies_all'  : list[str]  — unique non-empty SELFIES (for vocab build)
    'vocab_path'   : str        — path to vocab.json
    'max_seq_len'  : int        — max token count + 2 (BOS + EOS)

vocab.json — SELFIESTokenizer vocabulary (saved alongside)
"""
from __future__ import annotations

import pickle
import sys
sys.stdout.reconfigure(line_buffering=True)  # real-time output — no PYTHONUNBUFFERED=1 needed
from pathlib import Path

import pandas as pd

# ── Resolve project root so we can import from SELFIES/ ──────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]          # turingtrain/
sys.path.insert(0, str(_ROOT / "SELFIES"))

from inchi_to_selfies import convert_series, MoleculeConversionError  # noqa: E402
from selfies_tokenizer import SELFIESTokenizer                          # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE   = _ROOT / "model_training" / "cn_mixture_selfies" / "cn_mixture_selfies.dat"
OUT_PKL     = _HERE / "cn_mixtures_selfies.pkl"
OUT_VOCAB   = _HERE / "vocab.json"

N_COMPONENTS = 10
INCHI_COLS   = [f"cpnt_inchi_{i}" for i in range(1, N_COMPONENTS + 1)]
VOL_COLS     = [f"cpnt_vol_{i}"   for i in range(1, N_COMPONENTS + 1)]


def preprocess(data_file: Path = DATA_FILE) -> dict:
    """Load raw data, convert InChI → SELFIES, build vocabulary.

    Returns a dict that is also written to ``OUT_PKL``.
    """
    print(f"Loading data from:\n  {data_file}")
    df = pd.read_csv(data_file, sep="\t", comment="#")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    if "No" not in df.columns:
        df["No"] = df.index + 1

    # ── Convert each SELFIES column to InChI for compatibility ────────────────
    print("\nConverting SELFIES → InChI for compatibility...")
    import selfies as sf
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    for col in INCHI_COLS:
        selfies_col = col.replace("inchi", "selfies")
        print(f"  {selfies_col} → {col}", end="  ", flush=True)
        
        def _to_inchi(s):
            if not isinstance(s, str) or not s.strip():
                return None
            try:
                smiles = sf.decoder(s)
                if not smiles:
                    return None
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                return Chem.MolToInchi(mol)
            except Exception:
                return None
                
        df[col] = df[selfies_col].apply(_to_inchi)
        n_ok = df[col].notna().sum()
        print(f"({n_ok} ok)")

    # ── Collect all unique SELFIES (for vocab) ────────────────────────────────
    selfies_cols = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    all_selfies: list[str] = []
    for col in selfies_cols:
        vals = df[col].dropna().tolist()
        all_selfies.extend(v for v in vals if isinstance(v, str) and v.strip())

    unique_selfies = list(dict.fromkeys(all_selfies))   # deduplicate, preserve order
    print(f"\nTotal SELFIES strings collected : {len(all_selfies):,}")
    print(f"Unique SELFIES strings           : {len(unique_selfies):,}")

    # ── Build vocabulary ──────────────────────────────────────────────────────
    tokenizer = SELFIESTokenizer.from_corpus(unique_selfies)
    tokenizer.save(OUT_VOCAB)
    print(f"Vocabulary size  : {tokenizer.vocab_size} tokens")
    print(f"Vocab saved to   : {OUT_VOCAB}")

    max_seq_len = tokenizer.max_len_for_corpus(unique_selfies, margin=2)
    print(f"Max sequence len : {max_seq_len}  (tokens + BOS + EOS)")

    # ── Compute per-row statistics for QA ─────────────────────────────────────
    _add_mixture_stats(df)

    # ── Persist ───────────────────────────────────────────────────────────────
    payload = {
        "df_selfies":  df,
        "selfies_all": unique_selfies,
        "vocab_path":  str(OUT_VOCAB),
        "max_seq_len": max_seq_len,
    }
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nPickle cache saved to: {OUT_PKL}")

    # ── Quick QA printout ─────────────────────────────────────────────────────
    _qa_report(df)

    return payload


def _add_mixture_stats(df: pd.DataFrame) -> None:
    """Add helper columns: n_components (non-null components per row)."""
    selfies_cols = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    df["n_components"] = df[selfies_cols].notna().sum(axis=1)


def _qa_report(df: pd.DataFrame) -> None:
    print("\n── QA Report ────────────────────────────────────────────────────")
    print(f"Rows total          : {len(df):,}")
    print(f"Rows with CN        : {df['CN'].notna().sum():,}")
    print(f"CN range            : {df['CN'].min():.1f} – {df['CN'].max():.1f}")
    print(f"CN > 80             : {(df['CN'] > 80).sum():,}")

    sc = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    missing = df[sc].isna().sum()
    print("\nMissing SELFIES per component slot:")
    for col, n in missing.items():
        print(f"  {col}: {n:,}")

    print(f"\nn_components distribution:")
    print(df["n_components"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    preprocess()
