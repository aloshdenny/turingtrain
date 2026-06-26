#!/usr/bin/env python3
"""
inference.py
============
Self-contained inference script for the optimized attention-based CN mixture model.
This script embeds the vocabulary tokenizer and model architectures, allowing you 
to make predictions with just this script and the model checkpoint or ONNX file.

Requirements:
    Install dependencies using the provided requirements.txt in the root directory:
    pip install -r requirements.txt

Usage (CLI):
    # Predict CN for a single custom mixture using PyTorch checkpoint:
    python model_testing/inference.py \\
        --model model_testing/checkpoints_opt/seed0_s3_model.pt \\
        --selfies "[C][C]" "[C][O]" \\
        --vols 0.6 0.4 \\
        --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"

    # Predict CN for a single custom mixture using ONNX model:
    python model_testing/inference.py \\
        --model model_testing/selfies_vae_optimized.onnx \\
        --selfies "[C][C]" "[C][O]" \\
        --vols 0.6 0.4 \\
        --inchis "InChI=1S/C2H6/c1-2/h1-2H3" "InChI=1S/CH4O/c1-2/h2H,1H3"

    # Predict CN for a batch of mixtures from a CSV file (matching the database columns):
    python model_testing/inference.py \\
        --model model_testing/checkpoints_opt/seed0_s3_model.pt \\
        --csv model_testing/data/cn_mixtures_selfies.csv \\
        --out predictions.csv
"""
import os
import re
import sys
sys.stdout.reconfigure(line_buffering=True)  # real-time output — no PYTHONUNBUFFERED=1 needed
import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing RDKit for structural chemistry features (falls back to zero if missing)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedded Tokenizer & Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded vocabulary representing the exact index mapping used during training
VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "[/C]": 4,
    "[/H]": 5,
    "[=Branch1]": 6,
    "[=Branch2]": 7,
    "[=C]": 8,
    "[=O]": 9,
    "[=Ring1]": 10,
    "[Branch1]": 11,
    "[Branch2]": 12,
    "[C@@]": 13,
    "[C@]": 14,
    "[C]": 15,
    "[H]": 16,
    "[N]": 17,
    "[O]": 18,
    "[Ring1]": 19,
    "[Ring2]": 20,
    "[\\C]": 21,
    "[\\H]": 22
}

_TOKEN_RE = re.compile(r"\[.*?\]")

def split_selfies(selfies_str: str) -> list[str]:
    """Split a SELFIES string into a list of bracketed tokens."""
    if not isinstance(selfies_str, str):
        return []
    return _TOKEN_RE.findall(selfies_str)

class SELFIESTokenizer:
    def __init__(self, token2idx: dict[str, int]) -> None:
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}

    @property
    def pad_idx(self) -> int: return self.token2idx["<pad>"]
    @property
    def bos_idx(self) -> int: return self.token2idx["<bos>"]
    @property
    def eos_idx(self) -> int: return self.token2idx["<eos>"]
    @property
    def unk_idx(self) -> int: return self.token2idx["<unk>"]
    @property
    def vocab_size(self) -> int: return len(self.token2idx)

    def encode(self, selfies_str: str, max_len: int, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
        tokens = split_selfies(selfies_str)
        ids = []
        if add_bos:
            ids.append(self.bos_idx)
        for t in tokens:
            ids.append(self.token2idx.get(t, self.unk_idx))
        if add_eos:
            ids.append(self.eos_idx)

        # Truncate if necessary (retaining EOS)
        if len(ids) > max_len:
            if add_eos:
                ids = ids[:max_len - 1] + [self.eos_idx]
            else:
                ids = ids[:max_len]

        # Pad
        ids += [self.pad_idx] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

tokenizer = SELFIESTokenizer(VOCAB)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Chemistry Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _parse_formula(inchi: str) -> dict:
    if not isinstance(inchi, str) or not inchi.startswith("InChI="):
        return {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0, "Cl": 0, "Br": 0, "I": 0}
    try:
        formula = inchi.split("/")[1]
    except IndexError:
        formula = ""
    pat = re.compile(r"([A-Z][a-z]?)(\d*)")
    counts = {}
    for m in pat.finditer(formula):
        el  = m.group(1)
        cnt = int(m.group(2)) if m.group(2) else 1
        counts[el] = counts.get(el, 0) + cnt
    return counts

def inchi_chem_features(inchi: str) -> np.ndarray:
    """Extracts a 12-dimensional chemistry feature vector from an InChI string."""
    if not isinstance(inchi, str) or not inchi.strip():
        return np.zeros(12, dtype=np.float32)

    c = _parse_formula(inchi)
    C  = c.get("C", 0)
    H  = c.get("H", 0)
    O  = c.get("O", 0)
    N  = c.get("N", 0)
    S  = c.get("S", 0)
    halogens = c.get("Cl", 0) + c.get("Br", 0) + c.get("I", 0) + c.get("F", 0)
    n_heavy  = N + O + S + halogens
    dou      = (2 * C + 2 - H + N) / 2 if (C > 0 or H > 0) else 0.0
    ch_ratio = C / H if H > 0 else 0.0
    stereo   = float("/t" in inchi or "/m" in inchi)

    # RDKit descriptors
    hbd = 0.0
    tpsa = 0.0
    rot_bonds = 0.0
    num_rings = 0.0
    branching = 0.0

    if RDKIT_OK and inchi.startswith("InChI="):
        try:
            mol = Chem.MolFromInchi(inchi)
            if mol is not None:
                hbd = float(rdMolDescriptors.CalcNumLipinskiHBD(mol))
                tpsa = float(Descriptors.TPSA(mol))
                rot_bonds = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
                num_rings = float(rdMolDescriptors.CalcNumRings(mol))
                branch_cnt = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'C' and atom.GetDegree() >= 3:
                        branch_cnt += 1
                branching = float(branch_cnt)
        except Exception:
            pass

    return np.array([
        C, H, ch_ratio, dou, n_heavy, O,
        hbd, tpsa, rot_bonds, num_rings, branching, stereo
    ], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture Definitions (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])

class SELFIESEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, latent_dim: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, pad_idx: int, max_len: int) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = (src == self.pad_idx).clone()
        pad_mask[:, 0] = False
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        enc_out = self.transformer(emb, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

class MixtureSlotEncoder(nn.Module):
    def __init__(self, latent_dim: int, chem_dim: int, d_slot: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        slot_in_dim = latent_dim + 1 + chem_dim
        self.slot_proj = nn.Linear(slot_in_dim, d_slot)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_slot, nhead=n_heads, dim_feedforward=d_slot * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_slot, latent_dim)

    def forward(self, mu: torch.Tensor, vf: torch.Tensor, chem_pc: torch.Tensor) -> torch.Tensor:
        vf_exp = vf.unsqueeze(-1)
        slots = torch.cat([mu, vf_exp, chem_pc], dim=-1)
        slots = self.slot_proj(slots)
        pad_mask = (vf == 0.0)
        all_pad = pad_mask.all(dim=1, keepdim=True).expand_as(pad_mask)
        pad_mask = pad_mask & ~all_pad
        h = self.transformer(slots, src_key_padding_mask=pad_mask)
        vf_safe = vf.clone()
        vf_safe[pad_mask] = 0.0
        vf_norm = vf_safe / (vf_safe.sum(dim=1, keepdim=True) + 1e-8)
        z_mix = (h * vf_norm.unsqueeze(-1)).sum(dim=1)
        return self.out_proj(z_mix)

class HybridCNPredictor(nn.Module):
    def __init__(self, latent_dim: int, chem_dim: int, hidden_dims: tuple, dropout: float) -> None:
        super().__init__()
        in_dim = latent_dim + chem_dim
        self.chem_norm = nn.LayerNorm(chem_dim)
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mix: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_mix, self.chem_norm(chem)], dim=-1)
        return self.net(x).squeeze(-1)

class AttentionMixtureCNModel(nn.Module):
    def __init__(self, vae_encoder, slot_encoder, predictor, n_comp=10):
        super().__init__()
        self.encoder = vae_encoder
        self.slot_encoder = slot_encoder
        self.predictor = predictor
        self.n_comp = n_comp

    def forward(self, comp_tok, vf, chem_mix, chem_pc):
        B, nc, seq = comp_tok.shape
        flat = comp_tok.reshape(B * nc, seq)
        mu, _ = self.encoder(flat)
        mu = mu.view(B, nc, -1)
        z_mix = self.slot_encoder(mu, vf, chem_pc)
        return self.predictor(z_mix, chem_mix), z_mix

# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference Orchestrator (Supports PyTorch or ONNX Runtime)
# ─────────────────────────────────────────────────────────────────────────────

class CNInferenceModel:
    """Wraps model initialization, parameter loading, and prediction logic."""
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_len = 65
        self.n_comp = 10

        if self.model_path.suffix.lower() == ".onnx":
            self.use_onnx = True
            try:
                import onnxruntime as ort
            except ImportError:
                print("Error: onnxruntime is required for running ONNX models. Install it using the requirements.txt.")
                sys.exit(1)
            print(f"Loading ONNX session from {self.model_path}...")
            self.session = ort.InferenceSession(str(self.model_path))
        else:
            self.use_onnx = False
            # Build architecture
            vae_encoder = SELFIESEncoder(
                vocab_size=23, d_model=128, latent_dim=128, n_heads=4, n_layers=4,
                d_ff=512, dropout=0.0, pad_idx=0, max_len=self.max_len
            )
            slot_encoder = MixtureSlotEncoder(
                latent_dim=128, chem_dim=12, d_slot=128, n_heads=4, n_layers=2, dropout=0.0
            )
            predictor = HybridCNPredictor(
                latent_dim=128, chem_dim=12, hidden_dims=(512, 256, 128), dropout=0.0
            )
            self.model = AttentionMixtureCNModel(vae_encoder, slot_encoder, predictor, n_comp=self.n_comp)
            
            # Load weights
            print(f"Loading PyTorch checkpoint from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

    def predict(self, mixtures: list[dict]) -> np.ndarray:
        """Runs predictions for a list of mixtures.
        
        Input Format:
            mixtures = [
                {
                    "components": [
                        {"selfies": "[C][C]", "vol": 0.5, "inchi": "InChI=..."},
                        {"selfies": "[C][O]", "vol": 0.5, "inchi": "InChI=..."}
                    ]
                }
            ]
        """
        all_comp_tok = []
        all_vf = []
        all_chem_mix = []
        all_chem_pc = []

        for mix in mixtures:
            components = mix.get("components", [])
            # Extract attributes
            raw_vols = [float(c.get("vol", 0.0)) for c in components]
            sfs = [c.get("selfies", "") for c in components]
            inchis = [c.get("inchi", "") for c in components]

            # Normalize active volume fractions
            total_vol = sum(raw_vols) or 1.0
            vols = [v / total_vol for v in raw_vols]

            # Pad to n_comp slots (10)
            tokens = []
            per_comp_chem = []
            pad_vols = []

            for i in range(self.n_comp):
                if i < len(components):
                    s = sfs[i]
                    if s.strip():
                        toks = tokenizer.encode(s, max_len=self.max_len)
                    else:
                        toks = torch.full((self.max_len,), 0, dtype=torch.long)
                    cf = inchi_chem_features(inchis[i])
                    vol = vols[i]
                else:
                    toks = torch.full((self.max_len,), 0, dtype=torch.long)
                    cf = np.zeros(12, dtype=np.float32)
                    vol = 0.0

                tokens.append(toks)
                per_comp_chem.append(cf)
                pad_vols.append(vol)

            # Volume-weighted average for direct chemistry signal
            chem_mix = np.zeros(12, dtype=np.float32)
            for v, cf in zip(vols, [inchi_chem_features(inc) for inc in inchis]):
                chem_mix += v * cf

            all_comp_tok.append(torch.stack(tokens))
            all_vf.append(torch.tensor(pad_vols, dtype=torch.float32))
            all_chem_mix.append(torch.tensor(chem_mix, dtype=torch.float32))
            all_chem_pc.append(torch.tensor(np.stack(per_comp_chem), dtype=torch.float32))

        # Batch inputs
        comp_tok_batch = torch.stack(all_comp_tok)
        vf_batch       = torch.stack(all_vf)
        chem_mix_batch = torch.stack(all_chem_mix)
        chem_pc_batch  = torch.stack(all_chem_pc)

        if self.use_onnx:
            # Run using ONNX Runtime
            inputs = {
                "comp_tok": comp_tok_batch.numpy(),
                "vf": vf_batch.numpy(),
                "chem_mix": chem_mix_batch.numpy(),
                "chem_pc": chem_pc_batch.numpy()
            }
            outputs = self.session.run(None, inputs)
            preds = outputs[0]  # predicted_CN is the first output
            if len(preds.shape) > 1:
                preds = preds.squeeze(-1)
            return preds
        else:
            # Run using PyTorch
            comp_tok_batch = comp_tok_batch.to(self.device)
            vf_batch       = vf_batch.to(self.device)
            chem_mix_batch = chem_mix_batch.to(self.device)
            chem_pc_batch  = chem_pc_batch.to(self.device)

            with torch.no_grad():
                preds, _ = self.model(comp_tok_batch, vf_batch, chem_mix_batch, chem_pc_batch)
            return preds.cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Entrypoint & Batch CSV Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_csv(csv_path: str, model: CNInferenceModel) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    mixtures = []
    for idx, row in df.iterrows():
        components = []
        for i in range(1, 11):
            s = row.get(f"cpnt_selfies_{i}")
            v = row.get(f"cpnt_vol_{i}", 0.0)
            inc = row.get(f"cpnt_inchi_{i}")
            
            # Skip empty columns
            if isinstance(s, str) and s.strip():
                components.append({
                    "selfies": s,
                    "vol": v,
                    "inchi": inc if isinstance(inc, str) else ""
                })
        
        mixtures.append({"components": components})
    return mixtures

def main():
    parser = argparse.ArgumentParser(description="Self-contained inference runner for SELFIES CN property predictions.")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch (.pt) checkpoint or ONNX (.onnx) model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda, mps) for PyTorch backend.")
    
    # CLI inputs for single mixture
    parser.add_argument("--selfies", type=str, nargs="+", help="List of SELFIES strings for the components.")
    parser.add_argument("--vols", type=float, nargs="+", help="Corresponding volume fractions.")
    parser.add_argument("--inchis", type=str, nargs="+", help="Corresponding InChI strings.")
    
    # CLI inputs for batch CSV processing
    parser.add_argument("--csv", type=str, help="Path to database-style CSV file to run inference on.")
    parser.add_argument("--out", type=str, help="Output path for CSV containing predictions.")

    args = parser.parse_args()

    # Load inference engine (auto-detects backend based on extension)
    engine = CNInferenceModel(args.model, device=args.device)

    # Mode 1: Single mixture prediction
    if not args.csv:
        if not args.selfies or not args.vols or not args.inchis:
            parser.error("the following arguments are required when --csv is not specified: --selfies, --vols, --inchis")
        if len(args.selfies) != len(args.vols) or len(args.selfies) != len(args.inchis):
            parser.error("--selfies, --vols, and --inchis parameters must have matching lengths.")
            
        mixture = {
            "components": [
                {"selfies": s, "vol": v, "inchi": inc}
                for s, v, inc in zip(args.selfies, args.vols, args.inchis)
            ]
        }
        pred = engine.predict([mixture])[0]
        print(f"\nPredicted Mixture Cetane Number (CN): {pred:.4f}")
        
    # Mode 2: Batch CSV prediction
    else:
        import pandas as pd
        print(f"Processing CSV dataset: {args.csv}")
        mixtures = process_csv(args.csv, engine)
        preds = engine.predict(mixtures)
        
        df_out = pd.read_csv(args.csv)
        df_out["predicted_CN"] = preds
        
        out_path = args.out or "predictions_out.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved prediction results to: {out_path}")

if __name__ == "__main__":
    main()
