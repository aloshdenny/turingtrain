"""
inverse_design.py
=================
Bayesian Optimisation in the VAE latent space to discover new fuel mixtures
with target cetane numbers, with emphasis on the high-CN regime (CN > 80).

Workflow
--------
1. Load trained VAE (Stage 3) + CN predictor + tokenizer
2. Encode known mixtures → collect latent vectors μ
3. Initialise BO candidates from high-CN samples (warm start)
4. Run BoTorch GP-EI to maximise predicted CN
5. Decode top candidates: latent → SELFIES → SMILES
6. Validate SMILES with RDKit
7. Write top-N candidate mixtures to CSV

Usage
-----
    python model_training/cn_mixtures_selfies/vae/inverse_design.py \\
        --target-cn 100 --n-candidates 20 --bo-iters 200

    # To generate a range:
    python model_training/cn_mixtures_selfies/vae/inverse_design.py \\
        --target-cn 90 100 110 --n-candidates 10

Optional BoTorch install:
    pip install botorch
If BoTorch is unavailable, falls back to random search + hill-climbing.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project imports ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "SELFIES"))
sys.path.insert(0, str(_HERE))

from selfies_tokenizer import SELFIESTokenizer        # noqa: E402
from selfies_vae import SELFIESVAE                    # noqa: E402
from mixture_cn_predictor import CNPredictor, MixtureCNModel  # noqa: E402

import selfies as sf
try:
    from rdkit import Chem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("[warn] RDKit not available — SMILES validation skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR   = _HERE.parent / "checkpoints"
CACHE_PATH = _HERE.parent / "data" / "cn_mixtures_selfies.pkl"
OUT_DIR    = _HERE.parent / "inverse_design_results"

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(device: torch.device):
    """Load tokenizer + VAE + predictor from Stage 3 checkpoints."""
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)

    tokenizer   = SELFIESTokenizer.load(cache["vocab_path"])
    max_seq_len = cache["max_seq_len"]

    # Build model architecture (must match train_vae.py defaults)
    vae = SELFIESVAE(
        vocab_size=tokenizer.vocab_size,
        d_model=128, latent_dim=64, n_heads=4, n_layers=4,
        d_ff=512, dropout=0.0,
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        max_len=max_seq_len,
    ).to(device)

    predictor = CNPredictor(latent_dim=64, hidden_dims=(256, 128)).to(device)
    model     = MixtureCNModel(vae.encoder, predictor, n_components=10).to(device)

    # Prefer Stage 3 joint checkpoint; fall back to Stage 2
    s3_model = CKPT_DIR / "stage3_joint_best.pt"
    s3_vae   = CKPT_DIR / "stage3_vae_best.pt"
    s2_ckpt  = CKPT_DIR / "stage2_pred_best.pt"
    s1_ckpt  = CKPT_DIR / "stage1_vae_best.pt"

    if s3_vae.exists() and s3_model.exists():
        vae.load_state_dict(torch.load(s3_vae,   map_location=device))
        model.load_state_dict(torch.load(s3_model, map_location=device))
        print("Loaded Stage 3 joint checkpoint.")
    elif s2_ckpt.exists() and s1_ckpt.exists():
        vae.load_state_dict(torch.load(s1_ckpt,  map_location=device))
        model.load_state_dict(torch.load(s2_ckpt, map_location=device))
        print("Loaded Stage 1 VAE + Stage 2 predictor checkpoint.")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {CKPT_DIR}. Run train_vae.py first."
        )

    vae.eval()
    model.eval()
    return tokenizer, max_seq_len, vae, model, cache["df_selfies"]


# ─────────────────────────────────────────────────────────────────────────────
# Latent space encoding of known mixtures
# ─────────────────────────────────────────────────────────────────────────────

def encode_dataset(
    df,
    tokenizer: SELFIESTokenizer,
    max_seq_len: int,
    vae: SELFIESVAE,
    device: torch.device,
    n_comp: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all dataset mixtures to mixture latent vectors.

    Returns
    -------
    z_mix : (N, latent_dim)  mixture latent vectors
    cn    : (N,)             observed cetane numbers
    """
    z_list, cn_list = [], []
    pad_idx = tokenizer.pad_idx

    df_clean = df.dropna(subset=["CN"]).reset_index(drop=True)

    with torch.no_grad():
        for _, row in df_clean.iterrows():
            vols = []
            toks = []
            for i in range(1, n_comp + 1):
                v = row.get(f"cpnt_vol_{i}", 0.0)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = 0.0
                if v != v:   # NaN check
                    v = 0.0
                vols.append(v)

                s = row.get(f"cpnt_selfies_{i}", None)
                if isinstance(s, str) and s.strip():
                    toks.append(tokenizer.encode(s, max_seq_len))
                else:
                    toks.append(torch.full((max_seq_len,), pad_idx, dtype=torch.long))

            total = sum(vols)
            if total <= 0:
                total = 1.0
            vols = [v / total for v in vols]

            comp_tok = torch.stack(toks).unsqueeze(0).to(device)    # (1, n_comp, seq)
            vf       = torch.tensor(vols, dtype=torch.float32).unsqueeze(0).to(device)

            _, z_mix = vae.encoder(comp_tok.reshape(n_comp, max_seq_len))
            # Re-compute properly via model
            mu_list = []
            for i in range(n_comp):
                mu, _ = vae.encoder(comp_tok[0, i].unsqueeze(0))
                mu_list.append(mu.squeeze(0))
            mus = torch.stack(mu_list, dim=0)         # (n_comp, latent_dim)
            vf_t = torch.tensor(vols, dtype=torch.float32, device=device).unsqueeze(-1)
            z_mix_v = (mus * vf_t).sum(dim=0)         # (latent_dim,)

            z_list.append(z_mix_v.cpu())
            cn_list.append(float(row["CN"]))

    return torch.stack(z_list), torch.tensor(cn_list, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimisation (with BoTorch fallback)
# ─────────────────────────────────────────────────────────────────────────────

def predict_cn_from_z(
    z: torch.Tensor,
    predictor: CNPredictor,
    device: torch.device,
) -> torch.Tensor:
    """Predict CN from a batch of latent vectors. Returns (N,) tensor."""
    predictor.eval()
    with torch.no_grad():
        return predictor(z.to(device)).cpu()


def run_botorch_optimization(
    init_z: torch.Tensor,
    init_cn: torch.Tensor,
    predictor: CNPredictor,
    device: torch.device,
    target_cn: float,
    n_iters: int = 200,
    latent_dim: int = 64,
) -> torch.Tensor:
    """GP-EI Bayesian Optimisation in latent space using BoTorch.

    Returns the best latent vectors found (shape: (n_iters, latent_dim)).
    """
    try:
        import botorch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError:
        print("[warn] BoTorch not installed. Falling back to hill-climbing.")
        return _hill_climbing(init_z, predictor, device, target_cn, n_iters, latent_dim)

    print(f"  Running BoTorch GP-EI ({n_iters} iterations, target CN={target_cn})")

    # Scale train_Y for GP (maximise predicted CN)
    train_X = init_z.double()
    train_Y = init_cn.unsqueeze(-1).double()

    best_candidates = []

    for iteration in range(n_iters):
        gp  = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq = ExpectedImprovement(model=gp, best_f=train_Y.max(), maximize=True)

        bounds = torch.stack([
            train_X.mean(0) - 4 * train_X.std(0).clamp(min=0.1),
            train_X.mean(0) + 4 * train_X.std(0).clamp(min=0.1),
        ]).double()

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )
        candidate = candidate.squeeze(0)  # (latent_dim,)

        # Evaluate with predictor
        pred_cn = predict_cn_from_z(candidate.float().unsqueeze(0), predictor, device)
        new_y   = pred_cn.item()

        train_X = torch.cat([train_X, candidate.unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[new_y]]).double()], dim=0)

        best_candidates.append(candidate.float())

        if (iteration + 1) % 20 == 0:
            print(f"    Iter {iteration+1:4d}/{n_iters} | best pred CN so far: {train_Y.max():.2f}")

    return torch.stack(best_candidates)


def _hill_climbing(
    init_z: torch.Tensor,
    predictor: CNPredictor,
    device: torch.device,
    target_cn: float,
    n_iters: int,
    latent_dim: int,
    step_size: float = 0.05,
) -> torch.Tensor:
    """Gradient-free hill-climbing in latent space (BoTorch fallback)."""
    print(f"  Running hill-climbing ({n_iters} iterations, target CN={target_cn})")

    # Start from the known high-CN seed
    best_idx = init_cn_global.argmax()
    z = init_z[best_idx].clone().float()

    candidates = [z.clone()]
    best_cn    = predict_cn_from_z(z.unsqueeze(0), predictor, device).item()

    for _ in range(n_iters):
        perturb   = torch.randn(latent_dim) * step_size
        candidate = z + perturb
        pred_cn   = predict_cn_from_z(candidate.unsqueeze(0), predictor, device).item()
        if pred_cn > best_cn:
            z       = candidate
            best_cn = pred_cn
        candidates.append(z.clone())

    return torch.stack(candidates)


# Expose for fallback reference
init_cn_global: torch.Tensor = torch.zeros(1)


# ─────────────────────────────────────────────────────────────────────────────
# Decoding: latent → SELFIES → SMILES → validate
# ─────────────────────────────────────────────────────────────────────────────

def decode_latent_to_smiles(
    z: torch.Tensor,
    vae: SELFIESVAE,
    tokenizer: SELFIESTokenizer,
    device: torch.device,
) -> tuple[str | None, str | None]:
    """Decode a single latent vector z → SELFIES → SMILES.

    Returns
    -------
    (selfies_str, smiles_str)  — either may be None if decoding fails.
    """
    z = z.unsqueeze(0).to(device)  # (1, latent_dim)
    token_ids = vae.decode_latent(z)
    selfies_str = tokenizer.decode(token_ids[0])

    if not selfies_str:
        return None, None

    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        return selfies_str, None

    if RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return selfies_str, None
        smiles = Chem.MolToSmiles(mol)   # canonical SMILES

    return selfies_str, smiles


# ─────────────────────────────────────────────────────────────────────────────
# Build candidate mixtures
# ─────────────────────────────────────────────────────────────────────────────

def propose_mixture(
    component_latents: list[torch.Tensor],
    n_comp: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of per-component latent vectors, assemble a mixture.

    For simplicity, uses equal volume fractions for the valid components.
    Returns (stacked_latents, volume_fractions).
    """
    n_valid = len(component_latents)
    vf = torch.zeros(n_comp)
    vf[:n_valid] = 1.0 / n_valid

    padding = [torch.zeros_like(component_latents[0])] * (n_comp - n_valid)
    all_latents = component_latents + padding
    return torch.stack(all_latents), vf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inverse design: find high-CN fuel mixtures")
    p.add_argument("--target-cn", type=float, nargs="+", default=[100.0],
                   help="Target CN values to optimise towards (can specify multiple)")
    p.add_argument("--n-candidates", type=int, default=20,
                   help="Number of top candidates to report per target CN")
    p.add_argument("--bo-iters", type=int, default=200,
                   help="Number of BO / hill-climbing iterations")
    p.add_argument("--n-components", type=int, default=3,
                   help="Number of components per candidate mixture (1–10)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    global init_cn_global

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load models & data ────────────────────────────────────────────────────
    tokenizer, max_seq_len, vae, mixture_model, df = load_models(device)
    predictor = mixture_model.predictor

    # ── Encode known mixtures ─────────────────────────────────────────────────
    print("Encoding dataset mixtures…")
    z_known, cn_known = encode_dataset(df, tokenizer, max_seq_len, vae, device)
    init_cn_global = cn_known
    print(f"  Encoded {len(z_known)} mixtures.")

    # Warm-start BO from high-CN samples
    high_cn_mask = cn_known > 80.0
    if high_cn_mask.sum() > 0:
        seed_z  = z_known[high_cn_mask]
        seed_cn = cn_known[high_cn_mask]
        print(f"  High-CN warm-start seeds: {len(seed_z)}")
    else:
        # Fallback: top-10% by CN
        topk = max(5, int(0.1 * len(z_known)))
        idx  = cn_known.argsort(descending=True)[:topk]
        seed_z  = z_known[idx]
        seed_cn = cn_known[idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for target_cn in args.target_cn:
        print(f"\n{'='*60}")
        print(f"Optimising towards target CN = {target_cn}")
        print(f"{'='*60}")

        # Run BO
        candidate_z = run_botorch_optimization(
            init_z=seed_z,
            init_cn=seed_cn,
            predictor=predictor,
            device=device,
            target_cn=target_cn,
            n_iters=args.bo_iters,
            latent_dim=vae.latent_dim,
        )

        # Score all candidates
        pred_cns = predict_cn_from_z(candidate_z, predictor, device)

        # Keep top-N by proximity to target
        dist_to_target = (pred_cns - target_cn).abs()
        top_idx = dist_to_target.argsort()[:args.n_candidates]

        print(f"\nTop {args.n_candidates} candidates (target CN={target_cn}):")
        print(f"{'#':>4}  {'pred_CN':>9}  {'valid':>5}  SELFIES (component 1)")

        for rank, idx in enumerate(top_idx, start=1):
            z_c   = candidate_z[idx]
            p_cn  = pred_cns[idx].item()

            # Decode into component SELFIES/SMILES (use same z for all n_comp components
            # in this simplified implementation — a real run would optimise each separately)
            selfies_str, smiles_str = decode_latent_to_smiles(z_c, vae, tokenizer, device)
            valid = smiles_str is not None

            print(
                f"{rank:4d}  {p_cn:9.3f}  {'yes' if valid else 'no ':>5}  "
                f"{(selfies_str or 'decode_failed')[:60]}"
            )

            results.append({
                "target_cn":    target_cn,
                "rank":         rank,
                "pred_cn":      round(p_cn, 4),
                "valid_smiles": valid,
                "selfies":      selfies_str or "",
                "smiles":       smiles_str  or "",
            })

    # ── Write results CSV ─────────────────────────────────────────────────────
    import csv
    out_csv = OUT_DIR / "candidates.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "target_cn", "rank", "pred_cn", "valid_smiles", "selfies", "smiles"
        ])
        writer.writeheader()
        writer.writerows(results)

    n_valid = sum(r["valid_smiles"] for r in results)
    print(f"\nResults saved to {out_csv}")
    print(f"Total candidates: {len(results)}  |  Valid SMILES: {n_valid}")


if __name__ == "__main__":
    main()
