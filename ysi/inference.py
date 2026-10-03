"""
inference.py
============
Inference runner for Yield Sooting Index (YSI) model.
Supports both pure-component SELFIES and multi-component mixtures (up to 10 components).

Usage Examples:
  # 1. Single component YSI prediction
  python ysi/inference.py --selfies "[C][#C][C][C][C][C][C][C]"

  # 2. Mixture YSI prediction
  python ysi/inference.py \
    --selfies "[C][#C][C][C][C][C][C][C]" "[C][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=C][Ring1][#Branch2]" \
    --mole_fracs 0.6 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "models" / "ysi_dha"
_VOCAB_PATH = _MODEL_DIR / "vocab.json"
_ONNX_PATH = _MODEL_DIR / "ysi_dha.onnx"

N_COMPONENTS = 10
MAX_SELFIES_LEN = 65
TOKEN_RE = re.compile(r"\[.*?\]")


def load_vocab(vocab_path: Path = _VOCAB_PATH) -> dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("token2idx", data)


def encode_selfies(s: str, vocab: dict[str, int], max_len: int = MAX_SELFIES_LEN) -> list[int]:
    toks = TOKEN_RE.findall(str(s))
    pad_idx = vocab.get("<pad>", 0)
    unk_idx = vocab.get("<unk>", 3)
    ids = [vocab.get(t, unk_idx) for t in toks]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [pad_idx] * (max_len - len(ids))
    return ids


def predict_ysi(
    selfies_list: list[str],
    mole_fracs: list[float] | None = None,
    onnx_path: Path = _ONNX_PATH,
    vocab_path: Path = _VOCAB_PATH,
) -> float:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    vocab = load_vocab(vocab_path)
    n = len(selfies_list)
    if n == 0:
        raise ValueError("Must provide at least 1 SELFIES string.")
    if n > N_COMPONENTS:
        raise ValueError(f"Maximum supported components is {N_COMPONENTS}, got {n}.")

    if mole_fracs is None:
        if n == 1:
            mole_fracs = [1.0]
        else:
            mole_fracs = [1.0 / n] * n

    if len(mole_fracs) != n:
        raise ValueError(f"Length of mole_fracs ({len(mole_fracs)}) does not match selfies ({n}).")

    # Normalize mole fractions
    total_frac = sum(mole_fracs)
    if total_frac <= 0:
        raise ValueError("Mole fractions must sum to a positive number.")
    norm_fracs = [f / total_frac for f in mole_fracs]

    # Prepare inputs: pad to 10 slots
    tokens_10 = np.zeros((1, N_COMPONENTS, MAX_SELFIES_LEN), dtype=np.int64)
    fracs_10 = np.zeros((1, N_COMPONENTS), dtype=np.float32)

    for i, s in enumerate(selfies_list):
        tokens_10[0, i] = encode_selfies(s, vocab)
        fracs_10[0, i] = norm_fracs[i]

    session = ort.InferenceSession(str(onnx_path))
    res = session.run(["ysi_mix"], {"component_tokens": tokens_10, "mole_fracs": fracs_10})
    return float(res[0][0])


def main():
    parser = argparse.ArgumentParser(description="YSI predictor for SELFIES molecules and mixtures.")
    parser.add_argument("--selfies", nargs="+", required=True, help="One or more SELFIES strings (up to 10)")
    parser.add_argument("--mole_fracs", nargs="+", type=float, default=None, help="Mole fractions for each component")
    parser.add_argument("--model", type=Path, default=_ONNX_PATH, help="Path to exported ONNX model")
    args = parser.parse_args()

    pred = predict_ysi(args.selfies, args.mole_fracs, onnx_path=args.model)

    print("\n" + "=" * 50)
    print("Yield Sooting Index (YSI) Prediction")
    print("=" * 50)
    if len(args.selfies) == 1:
        print(f"Component SELFIES : {args.selfies[0]}")
        print(f"Predicted YSI     : {pred:.2f}")
    else:
        fracs = args.mole_fracs if args.mole_fracs is not None else [1.0 / len(args.selfies)] * len(args.selfies)
        s_frac = sum(fracs)
        fracs = [f / s_frac for f in fracs]
        print(f"Components ({len(args.selfies)}):")
        for i, (s, f) in enumerate(zip(args.selfies, fracs), 1):
            print(f"  [{i:2d}] x={f:.4f} | {s}")
        print(f"\nLinear Mixture Predicted YSI_mix : {pred:.2f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
