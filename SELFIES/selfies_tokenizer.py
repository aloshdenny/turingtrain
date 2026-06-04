"""
selfies_tokenizer.py
====================
Vocabulary construction and integer-encoding utilities for SELFIES strings.

This module is the single source of truth for how SELFIES tokens are mapped to
integers throughout the CN-mixture ML pipeline.  Every model (VAE, predictor,
inverse design) imports from here so that vocabularies stay consistent.

Usage
-----
>>> from selfies_tokenizer import SELFIESTokenizer
>>> tok = SELFIESTokenizer.from_corpus(selfies_list)
>>> tok.save("vocab.json")
>>> tok2 = SELFIESTokenizer.load("vocab.json")
>>> ids = tok.encode("[C][=C][N]", max_len=32)  # padded tensor
>>> selfies_str = tok.decode(ids)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"   # padding (index 0 by convention)
BOS_TOKEN = "<bos>"  # beginning-of-sequence (for decoder input)
EOS_TOKEN = "<eos>"  # end-of-sequence
UNK_TOKEN = "<unk>"  # unknown symbol (should never appear if vocab is complete)

SPECIALS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ─────────────────────────────────────────────────────────────────────────────
# SELFIES splitting regex — matches bracketed tokens like [C], [=N], [Ring1]
# ─────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"\[.*?\]")


def split_selfies(selfies_str: str) -> list[str]:
    """Return a list of individual SELFIES tokens from a SELFIES string.

    Works by collecting every ``[…]`` bracketed symbol in order.  This is the
    canonical way to tokenise SELFIES (the `selfies` library also exposes
    ``selfies.split_selfies`` which does the same thing).
    """
    return _TOKEN_RE.findall(selfies_str)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer class
# ─────────────────────────────────────────────────────────────────────────────

class SELFIESTokenizer:
    """Integer-encodes SELFIES strings with a fixed vocabulary.

    Parameters
    ----------
    token2idx : dict[str, int]
        Mapping from token string to integer index.  Must include all
        SPECIALS and every token that will be encountered at runtime.
    """

    def __init__(self, token2idx: dict[str, int]) -> None:
        self.token2idx: dict[str, int] = token2idx
        self.idx2token: dict[int, str] = {v: k for k, v in token2idx.items()}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK_TOKEN]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        selfies_iterable: Iterable[str],
        *,
        extra_tokens: Sequence[str] = (),
    ) -> "SELFIESTokenizer":
        """Build a tokeniser by scanning *selfies_iterable* for all unique tokens.

        Parameters
        ----------
        selfies_iterable : Iterable[str]
            Any iterable of SELFIES strings (e.g. a list or generator).
        extra_tokens : sequence of str, optional
            Additional tokens to add to the vocabulary beyond what is found in
            the corpus (e.g. domain-specific symbols you know will appear at
            inference time).
        """
        seen: set[str] = set()
        for s in selfies_iterable:
            if not s:
                continue
            seen.update(split_selfies(s))

        # Specials first (deterministic ordering), then sorted corpus tokens
        all_tokens = SPECIALS + sorted(seen) + [t for t in extra_tokens if t not in seen]
        token2idx = {tok: i for i, tok in enumerate(all_tokens)}
        return cls(token2idx)

    @classmethod
    def load(cls, path: str | Path) -> "SELFIESTokenizer":
        """Load a previously saved vocabulary from *path* (JSON)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["token2idx"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the vocabulary to a JSON file at *path*."""
        Path(path).write_text(
            json.dumps({"token2idx": self.token2idx}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        selfies_str: str,
        max_len: int,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> torch.Tensor:
        """Encode a SELFIES string to a fixed-length integer tensor.

        The sequence layout is::

            [<bos>, tok1, tok2, …, tokN, <eos>, <pad>, <pad>, …]

        The tensor has length ``max_len``; excess tokens beyond ``max_len``
        are silently truncated (BOS/EOS are always included if requested).

        Parameters
        ----------
        selfies_str : str
        max_len : int
            Total length of the output tensor (including BOS/EOS).
        add_bos : bool
            Prepend a ``<bos>`` token.
        add_eos : bool
            Append a ``<eos>`` token.

        Returns
        -------
        torch.Tensor  shape (max_len,)  dtype long
        """
        tokens = split_selfies(selfies_str)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        for t in tokens:
            ids.append(self.token2idx.get(t, self.unk_idx))
        if add_eos:
            ids.append(self.eos_idx)

        # Truncate if necessary (keep BOS and EOS)
        if len(ids) > max_len:
            if add_eos:
                ids = ids[: max_len - 1] + [self.eos_idx]
            else:
                ids = ids[:max_len]

        # Pad
        ids += [self.pad_idx] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
        *,
        strip_specials: bool = True,
    ) -> str:
        """Decode an integer sequence back to a SELFIES string.

        Parameters
        ----------
        token_ids : Tensor or list of int
        strip_specials : bool
            If True (default), remove BOS, EOS and PAD tokens from the output.

        Returns
        -------
        str  — concatenated SELFIES string (e.g. ``"[C][=C][N]"``)
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        specials_set = {self.pad_idx, self.bos_idx, self.eos_idx} if strip_specials else set()
        parts = []
        for idx in token_ids:
            if idx in specials_set:
                continue
            tok = self.idx2token.get(idx, UNK_TOKEN)
            parts.append(tok)
        return "".join(parts)

    def batch_encode(
        self,
        selfies_list: list[str],
        max_len: int,
    ) -> torch.Tensor:
        """Encode a list of SELFIES strings into a single 2-D tensor.

        Returns
        -------
        torch.Tensor  shape (N, max_len)  dtype long
        """
        return torch.stack([self.encode(s, max_len) for s in selfies_list])

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        *,
        strip_specials: bool = True,
    ) -> list[str]:
        """Decode a 2-D integer tensor into a list of SELFIES strings."""
        return [self.decode(row, strip_specials=strip_specials) for row in token_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def max_len_for_corpus(self, selfies_list: list[str], *, margin: int = 2) -> int:
        """Return ``max(token_count) + margin`` across the corpus.

        *margin* accounts for BOS and EOS tokens (default 2).
        """
        lengths = [len(split_selfies(s)) for s in selfies_list if s]
        return max(lengths, default=0) + margin

    def __repr__(self) -> str:
        return (
            f"SELFIESTokenizer(vocab_size={self.vocab_size}, "
            f"pad={self.pad_idx}, bos={self.bos_idx}, eos={self.eos_idx})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test / round-trip helper (used by verify step)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """Quick round-trip test: encode → decode should reproduce the original string."""
    import selfies as sf

    test_smiles = [
        "CCCCCCCCCC",        # n-decane
        "c1ccccc1C",          # toluene
        "CC(C)CC(C)(C)C",     # 2,2,4-trimethylpentane (iso-octane)
        "CCCC(CC)CCC",        # 4-ethylheptane
    ]

    selfies_list = [sf.encoder(s) for s in test_smiles]
    tok = SELFIESTokenizer.from_corpus(selfies_list)

    print(f"Vocabulary: {tok.vocab_size} tokens")
    print(f"Tokens: {list(tok.token2idx.keys())}\n")

    max_len = tok.max_len_for_corpus(selfies_list)
    print(f"Max sequence length (with BOS+EOS): {max_len}")

    all_ok = True
    for original in selfies_list:
        ids = tok.encode(original, max_len)
        recovered = tok.decode(ids)
        ok = recovered == original
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  {status}  {original!r}")
        if not ok:
            print(f"      → got {recovered!r}")

    if all_ok:
        print("\nAll round-trip tests passed.")
    else:
        raise RuntimeError("Round-trip test FAILED — check the tokenizer logic.")


if __name__ == "__main__":
    smoke_test()
