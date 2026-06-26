"""
inverse_design.py — model_testing/
====================================
Inverse design: given a target cetane number (CN), discover novel fuel mixture
compositions whose CN the trained optimized VAE + predictor is expected to match.

Strategy
--------
1. Load optimized model (SELFIESVAE encoder + MixtureSlotEncoder + HybridCNPredictor)
   from checkpoints_opt/seed0_s3_model.pt (Stage 3 joint fine-tuned checkpoint).
2. Encode the training dataset → collect per-component latent means μᵢ and the
   predicted CN for every known mixture.  High-CN or low-CN seeds are selected
   as warm-start depending on the requested target.
3. Gradient optimisation: relax per-component latent vectors z_i and log-volume
   logits jointly to minimise (pred_CN − target_CN)² + diversity penalty.
   Chemistry features are held at the seed mixture's values during optimisation
   and recomputed after decoding.
4. Decode: z_i → SELFIES → SMILES via greedy decoding + RDKit validation.
5. Report: write top-N candidate mixtures to a CSV file.

Usage
-----
    # Single target:
    python model_testing/inverse_design.py \\
        --target-cn 90 \\
        --n-candidates 10 \\
        --opt-steps 500

    # Multiple targets:
    python model_testing/inverse_design.py \\
        --target-cn 60 80 100 \\
        --n-candidates 5 \\
        --output inverse_results.csv

    # Fewer components per mixture (faster):
    python model_testing/inverse_design.py \\
        --target-cn 85 \\
        --n-comp 3 \\
        --n-candidates 8 \\
        --opt-steps 300 \\
        --lr 5e-3
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths: import from canonical SELFIES/ source (no local copies needed) ────
_HERE     = Path(__file__).resolve().parent          # model_testing/
_SELFIES  = _HERE.parent                             # SELFIES/
sys.path.insert(0, str(_SELFIES))                   # for selfies_tokenizer
sys.path.insert(0, str(_SELFIES / "vae"))           # for selfies_vae, train_vae_optimized, etc.

from selfies_tokenizer import SELFIESTokenizer        # noqa: E402

# Import model classes + helper functions from the training script
from train_vae_optimized import (                     # noqa: E402
    HybridCNPredictor,
    MixtureSlotEncoder,
    AttentionMixtureCNModel,
    inchi_chem_features,
    mixture_chem_features,
    CHEM_FEAT_DIM,
    LATENT_DIM,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    D_FF,
    SLOT_D_SLOT,
    SLOT_N_HEADS,
    SLOT_N_LAYERS,
    N_COMP,
)
from selfies_vae import SELFIESVAE                    # noqa: E402

import selfies as sf
try:
    from rdkit import Chem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("[warn] RDKit unavailable — SMILES validation skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
CKPT_OPT   = _SELFIES / "checkpoints_opt"             # same dir training writes to — no copy needed
DATA_CACHE = _SELFIES / "data" / "cn_mixtures_selfies.pkl"   # single source of truth
OUT_DIR    = _HERE / "inverse_design_results"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: torch.device, ckpt_dir: Path = CKPT_OPT):
    """Load optimized VAE + slot encoder + predictor from checkpoints."""
    # Prefer Stage-3 checkpoint; fall back to Stage-2
    s3 = ckpt_dir / "seed0_s3_model.pt"
    s2 = ckpt_dir / "seed0_s2_model.pt"
    s1 = ckpt_dir / "seed0_s1_vae.pt"

    if not s1.exists():
        raise FileNotFoundError(
            f"Stage-1 VAE checkpoint not found at {s1}.\n"
            "Run SELFIES/vae/train_vae_optimized.py first."
        )

    # Load data cache to get vocab info
    with open(DATA_CACHE, "rb") as fh:
        cache = pickle.load(fh)

    tokenizer   = SELFIESTokenizer.load(cache["vocab_path"])
    max_seq_len = cache["max_seq_len"]

    vae = SELFIESVAE(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, latent_dim=LATENT_DIM,
        n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=0.0,
        pad_idx=tokenizer.pad_idx, bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx, max_len=max_seq_len,
    ).to(device)

    predictor = HybridCNPredictor(
        latent_dim=LATENT_DIM, chem_dim=CHEM_FEAT_DIM,
        hidden_dims=(512, 256, 128), dropout=0.0,
    ).to(device)

    slot_encoder = MixtureSlotEncoder(
        latent_dim=LATENT_DIM, chem_dim=CHEM_FEAT_DIM,
        d_slot=SLOT_D_SLOT, n_heads=SLOT_N_HEADS,
        n_layers=SLOT_N_LAYERS, dropout=0.0,
    ).to(device)

    model = AttentionMixtureCNModel(vae.encoder, slot_encoder, predictor).to(device)

    # Load Stage-1 VAE (encoder + decoder)
    vae.load_state_dict(torch.load(s1, map_location=device))

    # Load the best joint model (Stage 3 preferred)
    ckpt_m = s3 if s3.exists() else s2
    model.load_state_dict(torch.load(ckpt_m, map_location=device))
    stage = "3" if s3.exists() else "2"
    print(f"[model] Loaded Stage-{stage} checkpoint from {ckpt_m.name}")

    vae.eval()
    model.eval()
    return vae, model, tokenizer, max_seq_len, cache


# ─────────────────────────────────────────────────────────────────────────────
# Dataset encoding — build a latent bank of known mixtures
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(
    df: pd.DataFrame,
    vae: SELFIESVAE,
    model: AttentionMixtureCNModel,
    tokenizer: SELFIESTokenizer,
    max_seq_len: int,
    device: torch.device,
    n_comp: int = N_COMP,
    max_bank_size: int = 2000,
) -> dict:
    """Encode dataset mixtures → per-component μ, chem features, and predicted CN.

    Encodes up to *max_bank_size* mixtures (stratified: half from high-CN ≥ 80,
    half from the rest) to keep latent bank construction fast on CPU.

    Returns a dict with:
        mu_bank      : list of (n_comp, latent_dim) numpy arrays
        vf_bank      : list of (n_comp,) numpy arrays  [volume fractions]
        chem_pc_bank : list of (n_comp, CHEM_FEAT_DIM) numpy arrays
        chem_mix_bank: list of (CHEM_FEAT_DIM,) numpy arrays
        cn_bank      : list of float  [true CN values]
        pred_cn      : list of float  [predicted CN values]
        selfies_bank : list of lists of str  [per-component SELFIES]
    """
    records = {
        "mu_bank": [], "vf_bank": [], "chem_pc_bank": [], "chem_mix_bank": [],
        "cn_bank": [], "pred_cn": [], "selfies_bank": [],
    }

    # Stratified subsample: keep high-CN representation
    high_mask  = df["CN"] >= 80.0
    high_df    = df[high_mask]
    low_df     = df[~high_mask]
    n_high = min(len(high_df), max_bank_size // 2)
    n_low  = min(len(low_df),  max_bank_size - n_high)
    sub_df = pd.concat([
        high_df.sample(n=n_high, random_state=42) if n_high < len(high_df) else high_df,
        low_df.sample(n=n_low,   random_state=42) if n_low  < len(low_df)  else low_df,
    ]).reset_index(drop=True)
    print(f"  Latent bank: {len(sub_df)} mixtures "
          f"({n_high} high-CN≥80, {n_low} others)", flush=True)

    n_total = len(sub_df)
    for row_i, (_, row) in enumerate(sub_df.iterrows()):
        if row_i % 250 == 0:
            print(f"  Encoding {row_i}/{n_total}…", flush=True)

        vols = []
        for i in range(1, n_comp + 1):
            try:
                v = float(row.get(f"cpnt_vol_{i}", 0.0) or 0.0)
                if v != v: v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            vols.append(v)
        total = sum(vols) or 1.0
        vols  = [v / total for v in vols]

        comp_tokens = []
        per_comp_chem = []
        per_comp_selfies = []
        for i in range(1, n_comp + 1):
            s = row.get(f"cpnt_selfies_{i}", None)
            if isinstance(s, str) and s.strip():
                tok = tokenizer.encode(s, max_seq_len)
                per_comp_selfies.append(s)
            else:
                tok = torch.full((max_seq_len,), tokenizer.pad_idx, dtype=torch.long)
                per_comp_selfies.append("")
            comp_tokens.append(tok)
            inchi = row.get(f"cpnt_inchi_{i}", "")
            per_comp_chem.append(inchi_chem_features(inchi if isinstance(inchi, str) else ""))

        comp_tok = torch.stack(comp_tokens).unsqueeze(0).to(device)   # (1, n_comp, seq)
        vf_t     = torch.tensor(vols, dtype=torch.float32).unsqueeze(0).to(device)
        chem_pc  = np.stack(per_comp_chem, axis=0)                    # (n_comp, CHEM_FEAT_DIM)
        chem_mix = mixture_chem_features(row, n_comp)
        chem_pc_t  = torch.tensor(chem_pc, dtype=torch.float32).unsqueeze(0).to(device)
        chem_mix_t = torch.tensor(chem_mix, dtype=torch.float32).unsqueeze(0).to(device)

        # Encode all components at once
        flat = comp_tok.view(-1, max_seq_len)               # (n_comp, seq)
        mu, _ = vae.encoder(flat)                           # (n_comp, latent_dim)
        mu = mu.view(1, n_comp, -1)                         # (1, n_comp, latent_dim)

        # Predict CN
        pred_cn, _ = model(comp_tok, vf_t, chem_mix_t, chem_pc_t)

        records["mu_bank"].append(mu.squeeze(0).cpu().numpy())
        records["vf_bank"].append(np.array(vols, dtype=np.float32))
        records["chem_pc_bank"].append(chem_pc)
        records["chem_mix_bank"].append(chem_mix)
        records["cn_bank"].append(float(row["CN"]))
        records["pred_cn"].append(float(pred_cn.item()))
        records["selfies_bank"].append(per_comp_selfies)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-based latent space optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _project_simplex(logits: torch.Tensor) -> torch.Tensor:
    """Convert unconstrained logits to volume fractions via softmax."""
    return F.softmax(logits, dim=-1)


def optimise_latent(
    target_cn: float,
    seed_mu: np.ndarray,           # (n_comp, latent_dim)  warm-start latent vectors
    seed_vf: np.ndarray,           # (n_comp,)             warm-start volume fractions
    seed_chem_pc: np.ndarray,      # (n_comp, CHEM_FEAT_DIM)
    seed_chem_mix: np.ndarray,     # (CHEM_FEAT_DIM,)
    model: AttentionMixtureCNModel,
    device: torch.device,
    *,
    n_comp: int         = N_COMP,
    n_steps: int        = 500,
    lr: float           = 1e-2,
    noise_std: float    = 0.5,     # perturbation on warm start
    lambda_div: float   = 0.02,    # diversity regulariser (push components apart)
    lambda_vf: float    = 0.01,    # entropy reg (encourage non-degenerate mixing)
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimise per-component latent vectors z_i and log-volume logits toward target_cn.

    Returns
    -------
    z_opt   : (n_comp, latent_dim) numpy array
    vf_opt  : (n_comp,)            numpy array
    pred_cn : float  predicted CN at the optimised point
    """
    model.eval()

    # Initiallise from warm-start with small Gaussian noise
    z0    = torch.tensor(seed_mu,   dtype=torch.float32, device=device)
    z0   += torch.randn_like(z0) * noise_std
    z_var = nn.Parameter(z0.clone())

    # Log-volume logits → softmax → volume fractions
    # Initialise from seed fractions (with jitter to allow exploration)
    eps   = 1e-6
    vf0   = torch.tensor(seed_vf,  dtype=torch.float32, device=device).clamp(eps, 1.0)
    lv0   = torch.log(vf0)
    lv0  += torch.randn_like(lv0) * 0.2
    lv_var = nn.Parameter(lv0.clone())

    # Fixed chemistry tensors (updated after decoding, held constant during opt)
    chem_pc_t  = torch.tensor(seed_chem_pc,  dtype=torch.float32, device=device).unsqueeze(0)
    chem_mix_t = torch.tensor(seed_chem_mix, dtype=torch.float32, device=device).unsqueeze(0)

    target_t = torch.tensor([target_cn], dtype=torch.float32, device=device)
    opt = torch.optim.Adam([z_var, lv_var], lr=lr)

    best_z, best_vf, best_loss = z_var.detach().clone(), lv_var.detach().clone(), float("inf")

    for step in range(n_steps):
        opt.zero_grad()

        vf_soft = _project_simplex(lv_var).unsqueeze(0)   # (1, n_comp)
        z_bat   = z_var.unsqueeze(0)                       # (1, n_comp, latent_dim)

        # Forward through slot encoder + predictor (skip VAE encoder – we directly use z)
        z_mix = model.slot_encoder(z_bat, vf_soft, chem_pc_t)   # (1, latent_dim)
        pred  = model.predictor(z_mix, chem_mix_t)               # (1,)

        # Primary: squared error to target
        loss_cn = F.mse_loss(pred, target_t)

        # Diversity: maximise pairwise distances between component latents
        # (stops all components from collapsing to same molecule)
        diffs = z_bat[0].unsqueeze(0) - z_bat[0].unsqueeze(1)   # (nc, nc, latent)
        pairwise_dist = diffs.pow(2).sum(-1).clamp(min=0.0)      # (nc, nc)
        loss_div = -lambda_div * pairwise_dist.mean()

        # Entropy reg: encourage diverse mixing (avoid all-weight-on-one component)
        entropy = -(vf_soft * (vf_soft + 1e-8).log()).sum()
        loss_vf = -lambda_vf * entropy

        loss = loss_cn + loss_div + loss_vf
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([z_var, lv_var], max_norm=5.0)
        opt.step()

        if loss_cn.item() < best_loss:
            best_loss = loss_cn.item()
            best_z  = z_var.detach().clone()
            best_vf = lv_var.detach().clone()

    # Final prediction at best point
    with torch.no_grad():
        vf_f   = _project_simplex(best_vf).unsqueeze(0)
        z_f    = best_z.unsqueeze(0)
        z_mix  = model.slot_encoder(z_f, vf_f, chem_pc_t)
        pred_f = model.predictor(z_mix, chem_mix_t).item()

    return (
        best_z.cpu().numpy(),
        _project_simplex(best_vf).detach().cpu().numpy(),
        pred_f,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decoding: latent → SELFIES → SMILES
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_latents(
    z: np.ndarray,               # (n_comp, latent_dim)
    vae: SELFIESVAE,
    tokenizer: SELFIESTokenizer,
    device: torch.device,
) -> list[str]:
    """Greedy-decode per-component latent vectors to SELFIES strings."""
    z_t   = torch.tensor(z, dtype=torch.float32, device=device)    # (n_comp, latent_dim)
    token_ids = vae.decode_latent(z_t)                              # (n_comp, seq_len)
    selfies_list = []
    for ids in token_ids:
        # Use tokenizer.decode() which already handles BOS/EOS/PAD stripping
        sel = tokenizer.decode(ids.tolist(), strip_specials=True)
        selfies_list.append(sel)
    return selfies_list


def selfies_to_smiles(selfies_str: str) -> str | None:
    """Convert SELFIES → SMILES, returning None on failure."""
    try:
        smi = sf.decoder(selfies_str)
        if RDKIT_OK:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            smi = Chem.MolToSmiles(mol, canonical=True)
        return smi
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Recompute chemistry features from decoded molecules
# ─────────────────────────────────────────────────────────────────────────────

def _selfies_to_inchi(selfies_str: str) -> str:
    if not selfies_str:
        return ""
    if RDKIT_OK:
        try:
            from rdkit.Chem.inchi import MolToInchi
            smi = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return MolToInchi(mol) or ""
        except Exception:
            pass
    return ""


def recompute_chem_features(selfies_list: list[str], vols: np.ndarray):
    """Recompute per-component and mixture-averaged chemistry features from decoded molecules."""
    chem_pc = np.stack(
        [inchi_chem_features(_selfies_to_inchi(s)) for s in selfies_list], axis=0
    )
    vf_norm = vols / (vols.sum() + 1e-8)
    chem_mix = (chem_pc * vf_norm[:, None]).sum(0)
    return chem_pc, chem_mix


# ─────────────────────────────────────────────────────────────────────────────
# Refine prediction with decoded chemistry features (one forward pass)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def refine_prediction(
    z: np.ndarray,
    vf: np.ndarray,
    selfies_list: list[str],
    model: AttentionMixtureCNModel,
    device: torch.device,
) -> float:
    """Recompute CN prediction using actual decoded-molecule chemistry features."""
    chem_pc, chem_mix = recompute_chem_features(selfies_list, vf)

    z_t     = torch.tensor(z,        dtype=torch.float32, device=device).unsqueeze(0)
    vf_t    = torch.tensor(vf,       dtype=torch.float32, device=device).unsqueeze(0)
    cp_t    = torch.tensor(chem_pc,  dtype=torch.float32, device=device).unsqueeze(0)
    cm_t    = torch.tensor(chem_mix, dtype=torch.float32, device=device).unsqueeze(0)

    z_mix = model.slot_encoder(z_t, vf_t, cp_t)
    pred  = model.predictor(z_mix, cm_t)
    return float(pred.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main inverse design loop
# ─────────────────────────────────────────────────────────────────────────────

def inverse_design(
    target_cn_list: list[float],
    vae: SELFIESVAE,
    model: AttentionMixtureCNModel,
    tokenizer: SELFIESTokenizer,
    latent_bank: dict,
    device: torch.device,
    *,
    n_comp: int        = N_COMP,
    n_candidates: int  = 10,
    opt_steps: int     = 500,
    lr: float          = 1e-2,
    n_restarts: int    = 5,
    noise_std: float   = 0.5,
    verbose: bool      = True,
) -> pd.DataFrame:
    """Run gradient-based inverse design for each target CN.

    Parameters
    ----------
    target_cn_list : list of float  target cetane numbers
    n_candidates   : int   number of candidate mixtures to return per target
    opt_steps      : int   gradient optimisation steps
    lr             : float Adam learning rate
    n_restarts     : int   random restarts per candidate (best is kept)

    Returns
    -------
    pd.DataFrame with columns: target_cn, pred_cn, selfies_i, smiles_i, vol_i, pred_cn_refined
    """
    cn_bank   = np.array(latent_bank["cn_bank"])
    rows_out  = []

    for target_cn in target_cn_list:
        print(f"\n{'='*60}")
        print(f"  Target CN = {target_cn}")
        print(f"{'='*60}")

        # Select warm-start seeds: nearest neighbours in CN space
        dist = np.abs(cn_bank - target_cn)
        n_pool = min(n_restarts * 3, len(cn_bank))
        seed_indices = np.argsort(dist)[:n_pool]

        candidates: list[tuple[np.ndarray, np.ndarray, float, list]] = []

        for restart_i in range(n_candidates):
            # Pick a seed (cycle through nearest neighbours)
            seed_idx = seed_indices[restart_i % len(seed_indices)]
            seed_mu  = latent_bank["mu_bank"][seed_idx][:n_comp]          # (n_comp, latent_dim)
            seed_vf  = latent_bank["vf_bank"][seed_idx][:n_comp]
            seed_cpc = latent_bank["chem_pc_bank"][seed_idx][:n_comp]
            seed_cmx = latent_bank["chem_mix_bank"][seed_idx]

            best_z, best_vf, best_pred = None, None, float("inf")
            best_err  = float("inf")

            for _ in range(n_restarts):
                z_opt, vf_opt, pred_cn = optimise_latent(
                    target_cn, seed_mu, seed_vf, seed_cpc, seed_cmx,
                    model, device,
                    n_comp=n_comp, n_steps=opt_steps, lr=lr,
                    noise_std=noise_std,
                )
                err = abs(pred_cn - target_cn)
                if err < best_err:
                    best_err  = err
                    best_z    = z_opt
                    best_vf   = vf_opt
                    best_pred = pred_cn

            if best_z is None:
                continue

            # Decode latents to SELFIES
            selfies_list = decode_latents(best_z, vae, tokenizer, device)

            # Refine prediction with decoded-molecule chemistry features
            pred_refined = refine_prediction(best_z, best_vf, selfies_list, model, device)

            candidates.append((best_z, best_vf, best_pred, pred_refined, selfies_list))

            if verbose:
                print(f"  Candidate {restart_i+1}/{n_candidates}: "
                      f"pred_cn={best_pred:.2f}  refined={pred_refined:.2f}  "
                      f"err={abs(best_pred - target_cn):.2f}")

        # Build output rows
        for z, vf, pred, pred_ref, selfies_list in candidates:
            smiles_list = [selfies_to_smiles(s) or "" for s in selfies_list]
            row: dict = {
                "target_cn"       : target_cn,
                "pred_cn"         : round(pred, 3),
                "pred_cn_refined" : round(pred_ref, 3),
                "abs_error"       : round(abs(pred - target_cn), 3),
            }
            active = sum(1 for v in vf if v > 0.01)
            row["n_active_components"] = active

            for i in range(n_comp):
                idx = i + 1
                row[f"selfies_{idx}"] = selfies_list[i] if i < len(selfies_list) else ""
                row[f"smiles_{idx}"]  = smiles_list[i]  if i < len(smiles_list)  else ""
                row[f"vol_{idx}"]     = round(float(vf[i]), 4) if i < len(vf) else 0.0
            rows_out.append(row)

    df_out = pd.DataFrame(rows_out)
    if not df_out.empty:
        df_out = df_out.sort_values(["target_cn", "abs_error"]).reset_index(drop=True)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inverse design: find fuel mixtures with a target cetane number."
    )
    p.add_argument(
        "--target-cn",   type=float, nargs="+", required=True,
        help="One or more target cetane numbers (e.g. 80 90 100).",
    )
    p.add_argument(
        "--n-candidates", type=int, default=10,
        help="Number of candidate mixtures per target CN (default: 10).",
    )
    p.add_argument(
        "--n-comp",       type=int, default=N_COMP,
        help=f"Max mixture components (default: {N_COMP}).",
    )
    p.add_argument(
        "--opt-steps",    type=int, default=500,
        help="Gradient optimisation steps per candidate (default: 500).",
    )
    p.add_argument(
        "--n-restarts",   type=int, default=5,
        help="Random restarts per candidate; best is kept (default: 5).",
    )
    p.add_argument(
        "--lr",           type=float, default=1e-2,
        help="Adam learning rate for latent optimisation (default: 1e-2).",
    )
    p.add_argument(
        "--noise-std",    type=float, default=0.5,
        help="Std of Gaussian noise added to warm-start latents (default: 0.5).",
    )
    p.add_argument(
        "--output",       type=str,   default=None,
        help="Output CSV path (default: inverse_design_results/inverse_<target>.csv).",
    )
    p.add_argument(
        "--ckpt-dir",     type=str,   default=str(CKPT_OPT),
        help=f"Checkpoint directory (default: {CKPT_OPT}).",
    )
    p.add_argument(
        "--quiet",        action="store_true",
        help="Suppress per-candidate progress output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")  # CPU for stable inference; GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Device        : {device}")
    print(f"Target CN(s)  : {args.target_cn}")
    print(f"Candidates    : {args.n_candidates}  |  Restarts: {args.n_restarts}")
    print(f"Opt steps     : {args.opt_steps}  |  LR: {args.lr}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    vae, model, tokenizer, max_seq_len, cache = load_model(device, ckpt_dir)

    # ── Encode dataset → latent bank ─────────────────────────────────────────
    print("\nEncoding dataset mixtures into latent bank (warm-start seeds)…")
    df = cache["df_selfies"].dropna(subset=["CN"]).reset_index(drop=True)
    latent_bank = encode_dataset(
        df, vae, model, tokenizer, max_seq_len, device, n_comp=args.n_comp
    )
    print(f"  {len(latent_bank['cn_bank'])} mixtures encoded.")
    cn_arr = np.array(latent_bank["cn_bank"])
    print(f"  CN range in dataset: [{cn_arr.min():.1f}, {cn_arr.max():.1f}]")

    # ── Run inverse design ────────────────────────────────────────────────────
    df_out = inverse_design(
        target_cn_list = args.target_cn,
        vae            = vae,
        model          = model,
        tokenizer      = tokenizer,
        latent_bank    = latent_bank,
        device         = device,
        n_comp         = args.n_comp,
        n_candidates   = args.n_candidates,
        opt_steps      = args.opt_steps,
        lr             = args.lr,
        n_restarts     = args.n_restarts,
        noise_std      = args.noise_std,
        verbose        = not args.quiet,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        targets_str = "_".join(str(int(t)) for t in args.target_cn)
        out_path = OUT_DIR / f"inverse_cn{targets_str}.csv"

    df_out.to_csv(out_path, index=False)
    print(f"\n✓ {len(df_out)} candidate mixtures saved to: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    summary_cols = ["target_cn", "pred_cn", "pred_cn_refined", "abs_error", "n_active_components"]
    available = [c for c in summary_cols if c in df_out.columns]
    print(df_out[available].to_string(index=False))


if __name__ == "__main__":
    main()
