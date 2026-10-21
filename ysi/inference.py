"""
inference.py
============
Unified Forward and Inverse Inference Runner for Yield Sooting Index (YSI).

Modes:
  1. Forward Mode:
     Predicts YSI from single-component SELFIES or multi-component mixtures (up to 10 components).
       python ysi/inference.py --selfies "[C][#C][C][C][C][C][C][C]"
       python ysi/inference.py --selfies "[C][#C][C][C][C][C][C][C]" "[C][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=C][Ring1][#Branch2]" --mole_fracs 0.6 0.4

  2. Inverse Mode (Physics-Guided Closed-Loop Search):
     Predicts / generates molecular SELFIES targeting a given YSI constraint.
     Uses fast batched parallel generation + forward ONNX scoring for pinpoint accuracy.
       python ysi/inference.py --mode inverse --ysi 85.0
       python ysi/inference.py --mode inverse --ysi 120.0 --candidates 3
       python ysi/inference.py --mode inverse --ysi 400.0 --greedy
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import selfies as sf

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "models" / "ysi_dha"
_VOCAB_PATH = _MODEL_DIR / "vocab.json"
_ONNX_PATH = _MODEL_DIR / "ysi_dha.onnx"
_INVERSE_ONNX_PATH = _HERE / "models" / "inverse_ysi" / "inverse_ysi.onnx"

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


# ── Forward Inference ──────────────────────────────────────────────────────────

def predict_ysi(
    selfies_list: list[str],
    mole_fracs: list[float] | None = None,
    onnx_path: Path = _ONNX_PATH,
    vocab_path: Path = _VOCAB_PATH,
) -> float:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Forward ONNX model not found at {onnx_path}")

    vocab = load_vocab(vocab_path)
    n = len(selfies_list)
    if n == 0:
        raise ValueError("Must provide at least 1 SELFIES string.")
    if n > N_COMPONENTS:
        raise ValueError(f"Maximum supported components is {N_COMPONENTS}, got {n}.")

    if mole_fracs is None:
        mole_fracs = [1.0] if n == 1 else [1.0 / n] * n

    if len(mole_fracs) != n:
        raise ValueError(f"Length of mole_fracs ({len(mole_fracs)}) does not match selfies ({n}).")

    total_frac = sum(mole_fracs)
    if total_frac <= 0:
        raise ValueError("Mole fractions must sum to a positive number.")
    norm_fracs = [f / total_frac for f in mole_fracs]

    tokens_10 = np.zeros((1, N_COMPONENTS, MAX_SELFIES_LEN), dtype=np.int64)
    fracs_10 = np.zeros((1, N_COMPONENTS), dtype=np.float32)

    for i, s in enumerate(selfies_list):
        tokens_10[0, i] = encode_selfies(s, vocab)
        fracs_10[0, i] = norm_fracs[i]

    session = ort.InferenceSession(str(onnx_path))
    res = session.run(["ysi_mix"], {"component_tokens": tokens_10, "mole_fracs": fracs_10})
    return float(res[0][0])


def predict_ysi_batch(
    selfies_list: list[str],
    onnx_path: Path = _ONNX_PATH,
    vocab_path: Path = _VOCAB_PATH,
) -> np.ndarray:
    """Evaluates a batch of pure-component SELFIES in one forward ONNX inference call."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"Forward ONNX model not found at {onnx_path}")

    vocab = load_vocab(vocab_path)
    B = len(selfies_list)
    tokens_10 = np.zeros((B, N_COMPONENTS, MAX_SELFIES_LEN), dtype=np.int64)
    fracs_10 = np.zeros((B, N_COMPONENTS), dtype=np.float32)

    for i, s in enumerate(selfies_list):
        tokens_10[i, 0] = encode_selfies(s, vocab)
        fracs_10[i, 0] = 1.0

    session = ort.InferenceSession(str(onnx_path))
    res = session.run(["ysi_mix"], {"component_tokens": tokens_10, "mole_fracs": fracs_10})
    return res[0]


# ── Inverse Inference (Physics-Guided Best-of-N Search) ────────────────────────

def predict_inverse_selfies(
    target_ysi: float,
    num_candidates: int = 1,
    greedy: bool = False,
    pool_size: int = 32,
    temperature: float = 0.65,
    onnx_path: Path = _INVERSE_ONNX_PATH,
    forward_onnx_path: Path = _ONNX_PATH,
    vocab_path: Path = _VOCAB_PATH,
    max_len: int = 35,
) -> list[dict[str, str | float]]:
    """
    Predicts / generates molecular candidate SELFIES from target YSI.
    Uses inverse ONNX model + forward model guidance for high-precision matching.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"Inverse ONNX model not found at {onnx_path}")

    vocab = load_vocab(vocab_path)
    idx2tok = {v: k for k, v in vocab.items()}
    session = ort.InferenceSession(str(onnx_path))

    target_val = float(target_ysi)
    n_samples = 1 if greedy else max(pool_size, num_candidates * 8)

    y_in = np.full((n_samples, 1), target_val, dtype=np.float32)
    cur_tokens = np.full((n_samples, 1), 1, dtype=np.int64)  # <bos>
    finished = np.zeros(n_samples, dtype=bool)

    # Autoregressive generation loop
    for _ in range(max_len):
        logits = session.run(["next_token_logits"], {"target_ysi": y_in, "current_tokens": cur_tokens})[0]
        step_logits = logits[:, -1, :]

        if greedy:
            next_toks = np.argmax(step_logits, axis=-1, keepdims=True)
        else:
            scaled = step_logits / max(temperature, 0.1)
            exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            next_toks = np.array([[np.random.choice(len(p), p=p)] for p in probs])

        next_toks[finished] = 0
        cur_tokens = np.append(cur_tokens, next_toks, axis=1)
        finished |= (next_toks.squeeze(-1) == 2)
        if finished.all():
            break

    # Extract unique SELFIES sequences
    unique_selfies = []
    seen = set()
    for row in cur_tokens:
        tok_strs = [idx2tok.get(tid, "") for tid in row[1:] if tid not in (0, 1, 2)]
        s = "".join([t for t in tok_strs if t and not t.startswith("<")])
        if s and s not in seen:
            seen.add(s)
            unique_selfies.append(s)

    if not unique_selfies:
        return []

    # Score realized YSI using forward model
    realized_ys = predict_ysi_batch(unique_selfies, onnx_path=forward_onnx_path, vocab_path=vocab_path)

    results = []
    for s, r in zip(unique_selfies, realized_ys):
        try:
            smi = sf.decoder(s)
        except Exception:
            smi = "Invalid SELFIES"
        err = abs(float(r) - target_val)
        results.append({
            "target_ysi": target_val,
            "selfies": s,
            "smiles": smi,
            "realized_ysi": float(r),
            "error_ysi": err,
        })

    # Sort candidates by lowest error
    results.sort(key=lambda x: x["error_ysi"])
    return results[:num_candidates]


# ── CLI Interface ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified YSI forward and inverse ML inference engine.")
    parser.add_argument("--mode", choices=["forward", "inverse"], default=None,
                        help="Inference mode: 'forward' (SELFIES -> YSI) or 'inverse' (YSI -> SELFIES)")

    # Forward arguments
    parser.add_argument("--selfies", nargs="+", default=None, help="One or more SELFIES strings (up to 10)")
    parser.add_argument("--mole_fracs", nargs="+", type=float, default=None, help="Mole fractions for components")

    # Inverse arguments
    parser.add_argument("--ysi", type=float, default=None, help="Target YSI value (required for inverse mode)")
    parser.add_argument("--candidates", type=int, default=1, help="Number of candidate molecules to return (default: 1)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of guided search")
    parser.add_argument("--temperature", type=float, default=0.65, help="Sampling temperature for search pool (default: 0.65)")

    # Optional model paths
    parser.add_argument("--forward_model", type=Path, default=_ONNX_PATH, help="Path to forward YSI ONNX model")
    parser.add_argument("--inverse_model", type=Path, default=_INVERSE_ONNX_PATH, help="Path to inverse YSI ONNX model")
    args = parser.parse_args()

    # Determine mode automatically if not specified
    if args.mode is None:
        if args.ysi is not None:
            args.mode = "inverse"
        elif args.selfies is not None:
            args.mode = "forward"
        else:
            parser.error("Specify either --mode forward/inverse, --selfies, or --ysi.")

    # ── Execute Forward Mode ──
    if args.mode == "forward":
        if not args.selfies:
            parser.error("Forward mode requires --selfies.")
        pred = predict_ysi(args.selfies, args.mole_fracs, onnx_path=args.forward_model)
        print("\n" + "=" * 60)
        print("Yield Sooting Index (YSI) Forward Prediction")
        print("=" * 60)
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
        print("=" * 60 + "\n")

    # ── Execute Inverse Mode ──
    else:
        if args.ysi is None:
            parser.error("Inverse mode requires --ysi <float>.")
        t0 = time.time()
        candidates = predict_inverse_selfies(
            target_ysi=args.ysi,
            num_candidates=args.candidates,
            greedy=args.greedy,
            temperature=args.temperature,
            onnx_path=args.inverse_model,
            forward_onnx_path=args.forward_model,
        )
        print("\n" + "=" * 65)
        mode_desc = "Greedy" if args.greedy else "Physics-Guided Search"
        print(f"Inverse ML Model ({mode_desc}): Target YSI = {args.ysi:.1f}  ({time.time()-t0:.2f}s)")
        print("=" * 65)
        for i, c in enumerate(candidates, 1):
            header = f"Candidate #{i}" if len(candidates) > 1 else "Result"
            print(f"\n--- {header} ---")
            print(f"Generated SELFIES : {c['selfies']}")
            print(f"Decoded SMILES    : {c['smiles']}")
            print(f"Realized YSI      : {c['realized_ysi']:.2f} (Forward Verification)")
            print(f"Parity Error      : {c['error_ysi']:.2f} YSI")
        print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
