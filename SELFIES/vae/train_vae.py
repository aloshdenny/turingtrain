"""
train_vae.py
============
Three-stage training pipeline for the SELFIES-VAE + CN-predictor.

Stage 1  Pre-train VAE   (unsupervised, on individual SELFIES strings)
Stage 2  Train predictor  (frozen VAE encoder, supervised on CN)
Stage 3  Fine-tune joint  (both modules, combined loss)

Usage
-----
# Full training from project root:
    python model_training/cn_mixtures_selfies/vae/train_vae.py

# Quick smoke-test (few epochs, small batch):
    python model_training/cn_mixtures_selfies/vae/train_vae.py --fast

# Resume from checkpoint:
    python model_training/cn_mixtures_selfies/vae/train_vae.py \\
        --resume checkpoints/stage1_vae_best.pt

Outputs (written to checkpoints/ next to this file)
-----------------------------------------------------
    stage1_vae_best.pt      best VAE (by recon loss on val set)
    stage2_pred_best.pt     best CN predictor (by MAE on val set)
    stage3_joint_best.pt    best joint model (by val CN MAE)
    vocab.json              tokenizer vocabulary
    training_log.csv        per-epoch metrics
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ── Project imports ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]           # turingtrain/
sys.path.insert(0, str(_ROOT / "SELFIES"))

from selfies_tokenizer import SELFIESTokenizer          # noqa: E402

sys.path.insert(0, str(_HERE))
from selfies_vae import SELFIESVAE, vae_loss, beta_schedule   # noqa: E402
from mixture_cn_predictor import (                              # noqa: E402
    CNPredictor, MixtureCNModel, weighted_mse_loss, joint_loss
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CACHE_PATH  = _HERE.parent / "data" / "cn_mixtures_selfies.pkl"
CKPT_DIR    = _HERE.parent / "checkpoints"
LOG_PATH    = _HERE.parent / "training_log.csv"
N_COMP      = 10

# Model hyper-parameters
D_MODEL     = 128
LATENT_DIM  = 64
N_HEADS     = 4
N_LAYERS    = 4
D_FF        = 512
DROPOUT     = 0.1

# Training hyper-parameters
BATCH_SIZE          = 64
LR_VAE              = 3e-4
LR_PRED             = 1e-3
LR_JOINT            = 5e-5
BETA_ANNEAL_EPOCHS  = 20
BETA_MAX            = 0.002

S1_EPOCHS = 100   # Stage 1
S2_EPOCHS = 60    # Stage 2
S3_EPOCHS = 40    # Stage 3

HIGH_CN_THRESHOLD = 80.0
HIGH_CN_WEIGHT    = 5.0
LAMBDA_PRED       = 10.0
LAMBDA_VAE        = 1.0

VAL_FRACTION = 0.15
SEED         = 42


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeDataset(Dataset):
    """Dataset of individual molecule SELFIES strings (for Stage 1 VAE pre-training).

    Each item is a single integer-encoded SELFIES tensor.
    """

    def __init__(self, selfies_list: list[str], tokenizer: SELFIESTokenizer, max_len: int) -> None:
        self.samples = [
            tokenizer.encode(s, max_len)
            for s in selfies_list if isinstance(s, str) and s.strip()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class MixtureDataset(Dataset):
    """Dataset of fuel mixtures for Stages 2 & 3.

    Each item is:
        component_tokens  : (n_components, max_len)  padded integer tensors
        volume_fractions  : (n_components,)           normalised volumes
        cn                : scalar float              cetane number
    """

    def __init__(
        self,
        df,
        tokenizer: SELFIESTokenizer,
        max_len: int,
        n_comp: int = N_COMP,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.n_comp    = n_comp
        self.pad_idx   = tokenizer.pad_idx

        rows = []
        for _, row in df.iterrows():
            # Volume fractions
            vols = []
            for i in range(1, n_comp + 1):
                v = row.get(f"cpnt_vol_{i}", 0.0)
                vols.append(float(v) if not _isnan(v) else 0.0)
            total = sum(vols)
            if total <= 0:
                total = 1.0
            vols = [v / total for v in vols]

            # SELFIES token tensors
            tokens = []
            for i in range(1, n_comp + 1):
                s = row.get(f"cpnt_selfies_{i}", None)
                if isinstance(s, str) and s.strip():
                    tokens.append(tokenizer.encode(s, max_len))
                else:
                    tokens.append(torch.full((max_len,), self.pad_idx, dtype=torch.long))

            cn = float(row["CN"])
            rows.append((
                torch.stack(tokens),                             # (n_comp, max_len)
                torch.tensor(vols, dtype=torch.float32),         # (n_comp,)
                torch.tensor(cn,   dtype=torch.float32),         # scalar
            ))

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def _isnan(v) -> bool:
    import math
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Select best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> float:
    """Fraction of non-padding positions predicted correctly."""
    preds = logits.argmax(-1)                          # (batch, tgt_len)
    mask  = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0


class EarlyStopper:
    """Stops training if validation metric does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10, mode: str = "min") -> None:
        self.patience  = patience
        self.mode      = mode
        self.best      = math.inf if mode == "min" else -math.inf
        self.counter   = 0

    def step(self, value: float) -> bool:
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best    = value
            self.counter = 0
            return False   # continue
        self.counter += 1
        return self.counter >= self.patience   # stop


# ─────────────────────────────────────────────────────────────────────────────
# Training stages
# ─────────────────────────────────────────────────────────────────────────────

def stage1_pretrain_vae(
    vae: SELFIESVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    ckpt_path: Path,
    log_writer,
) -> None:
    """Stage 1: Pre-train the VAE on individual SELFIES strings."""
    print("\n" + "=" * 60)
    print("STAGE 1: VAE Pre-training")
    print("=" * 60)

    opt   = torch.optim.AdamW(vae.parameters(), lr=LR_VAE, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    stopper = EarlyStopper(patience=15, mode="min")

    best_val = math.inf

    for epoch in range(1, epochs + 1):
        beta = BETA_MAX * beta_schedule(epoch, BETA_ANNEAL_EPOCHS)

        # ── Train ──────────────────────────────────────────────────────────
        vae.train()
        t0 = time.time()
        train_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        train_acc    = 0.0
        n_batches    = 0

        for batch in train_loader:
            src = batch.to(device)                          # (B, seq_len)
            logits, mu, logvar = vae(src)
            targets = src[:, 1:]                            # shift left (remove BOS)

            total, recon, kl = vae_loss(
                logits, targets, mu, logvar, beta=beta, pad_idx=vae.pad_idx
            )

            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()

            train_losses["total"] += total.item()
            train_losses["recon"] += recon.item()
            train_losses["kl"]    += kl.item()
            train_acc             += token_accuracy(logits, targets, vae.pad_idx)
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= n_batches
        train_acc /= n_batches

        # ── Validate ───────────────────────────────────────────────────────
        vae.eval()
        val_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        val_acc    = 0.0
        n_val      = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch.to(device)
                logits, mu, logvar = vae(src)
                targets = src[:, 1:]
                total, recon, kl = vae_loss(
                    logits, targets, mu, logvar, beta=beta, pad_idx=vae.pad_idx
                )
                val_losses["total"] += total.item()
                val_losses["recon"] += recon.item()
                val_losses["kl"]    += kl.item()
                val_acc             += token_accuracy(logits, targets, vae.pad_idx)
                n_val += 1

        for k in val_losses:
            val_losses[k] /= n_val
        val_acc /= n_val

        sched.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:4d}/{epochs}  β={beta:.3f} | "
            f"train [tot={train_losses['total']:.4f} rc={train_losses['recon']:.4f} "
            f"kl={train_losses['kl']:.4f} acc={train_acc:.3f}] | "
            f"val  [tot={val_losses['total']:.4f} rc={val_losses['recon']:.4f} "
            f"acc={val_acc:.3f}]  [{elapsed:.1f}s]"
        )

        # Log
        log_writer.writerow({
            "stage": 1, "epoch": epoch,
            "train_total": train_losses["total"],
            "train_recon": train_losses["recon"],
            "train_kl":    train_losses["kl"],
            "train_acc":   train_acc,
            "val_total":   val_losses["total"],
            "val_recon":   val_losses["recon"],
            "val_kl":      val_losses["kl"],
            "val_acc":     val_acc,
            "cn_mae": "", "beta": beta,
        })

        # Checkpoint
        if val_losses["recon"] < best_val:
            best_val = val_losses["recon"]
            torch.save(vae.state_dict(), ckpt_path)
            print(f"    ✓ Saved best VAE (val_recon={best_val:.4f})")

        if stopper.step(val_losses["total"]):
            print(f"  Early stop at epoch {epoch}")
            break

    vae.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Best VAE val_recon = {best_val:.4f}")


def stage2_train_predictor(
    model: MixtureCNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    ckpt_path: Path,
    log_writer,
) -> None:
    """Stage 2: Train CN predictor with frozen VAE encoder."""
    print("\n" + "=" * 60)
    print("STAGE 2: CN Predictor Training (encoder frozen)")
    print("=" * 60)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    opt     = torch.optim.AdamW(model.predictor.parameters(), lr=LR_PRED, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    stopper = EarlyStopper(patience=15, mode="min")
    best_mae = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()   # keep BN/dropout in eval mode for frozen encoder

        train_mae, n_train = 0.0, 0
        for comp_tok, vf, cn in train_loader:
            comp_tok = comp_tok.to(device)
            vf       = vf.to(device)
            cn       = cn.to(device)

            cn_pred, _ = model(comp_tok, vf)
            loss = weighted_mse_loss(
                cn_pred, cn,
                high_cn_threshold=HIGH_CN_THRESHOLD,
                high_cn_weight=HIGH_CN_WEIGHT,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            opt.step()

            train_mae += (cn_pred.detach() - cn).abs().sum().item()
            n_train   += len(cn)

        train_mae /= n_train

        # Validate
        model.eval()
        val_mae, n_val = 0.0, 0
        val_mae_hicn, n_hicn = 0.0, 0
        with torch.no_grad():
            for comp_tok, vf, cn in val_loader:
                comp_tok = comp_tok.to(device)
                vf       = vf.to(device)
                cn_pred, _ = model(comp_tok, vf)

                abs_err = (cn_pred.cpu() - cn).abs()
                val_mae += abs_err.sum().item()
                n_val   += len(cn)

                hi_mask = cn > HIGH_CN_THRESHOLD
                if hi_mask.sum() > 0:
                    val_mae_hicn += abs_err[hi_mask].sum().item()
                    n_hicn       += hi_mask.sum().item()

        val_mae  /= n_val
        hicn_str  = f"{val_mae_hicn/n_hicn:.3f}" if n_hicn > 0 else "N/A"

        sched.step(val_mae)

        print(
            f"  Ep {epoch:4d}/{epochs} | "
            f"train_MAE={train_mae:.4f} | val_MAE={val_mae:.4f} | "
            f"val_MAE(CN>80)={hicn_str}"
        )

        log_writer.writerow({
            "stage": 2, "epoch": epoch,
            "train_total": "", "train_recon": "", "train_kl": "",
            "train_acc": "", "val_total": "", "val_recon": "", "val_kl": "",
            "val_acc": "", "cn_mae": val_mae, "beta": "",
        })

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✓ Saved best predictor (val_MAE={best_mae:.4f})")

        if stopper.step(val_mae):
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Best predictor val_MAE = {best_mae:.4f}")

    # Unfreeze for Stage 3
    for p in model.encoder.parameters():
        p.requires_grad_(True)


def stage3_finetune_joint(
    vae: SELFIESVAE,
    model: MixtureCNModel,
    mol_train_loader: DataLoader,
    mix_train_loader: DataLoader,
    mix_val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    ckpt_vae_path: Path,
    ckpt_model_path: Path,
    log_writer,
) -> None:
    """Stage 3: Fine-tune VAE + predictor jointly.

    Alternates between:
    - a VAE reconstruction step (sampled SELFIES mini-batch)
    - a CN prediction step (mixture mini-batch)
    with a combined loss.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Joint Fine-tuning")
    print("=" * 60)

    all_params = list(vae.parameters()) + list(model.predictor.parameters())
    opt     = torch.optim.AdamW(all_params, lr=LR_JOINT, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    stopper = EarlyStopper(patience=12, mode="min")
    best_mae = math.inf

    mol_iter = iter(mol_train_loader)

    for epoch in range(1, epochs + 1):
        beta = BETA_MAX   # Small KL in Stage 3 to prevent posterior collapse

        vae.train()
        model.train()

        ep_cn_loss, ep_vae_loss, n_batches = 0.0, 0.0, 0

        for comp_tok, vf, cn in mix_train_loader:
            # ── Get a molecule batch for VAE loss ──────────────────────────
            try:
                mol_batch = next(mol_iter)
            except StopIteration:
                mol_iter  = iter(mol_train_loader)
                mol_batch = next(mol_iter)

            mol_src = mol_batch.to(device)
            comp_tok = comp_tok.to(device)
            vf       = vf.to(device)
            cn       = cn.to(device)

            # VAE loss
            logits_vae, mu_vae, logvar_vae = vae(mol_src)
            vae_targets = mol_src[:, 1:]
            _, recon, kl = vae_loss(
                logits_vae, vae_targets, mu_vae, logvar_vae,
                beta=beta, pad_idx=vae.pad_idx
            )

            # CN loss
            cn_pred, _ = model(comp_tok, vf)
            cn_loss = weighted_mse_loss(
                cn_pred, cn,
                high_cn_threshold=HIGH_CN_THRESHOLD,
                high_cn_weight=HIGH_CN_WEIGHT,
            )

            loss = joint_loss(
                recon, kl, cn_loss,
                lambda_pred=LAMBDA_PRED, lambda_vae=LAMBDA_VAE, beta=beta
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_cn_loss  += cn_loss.item()
            ep_vae_loss += recon.item()
            n_batches   += 1

        ep_cn_loss  /= n_batches
        ep_vae_loss /= n_batches
        sched.step()

        # Validate CN
        model.eval()
        val_mae, n_val = 0.0, 0
        val_mae_hicn, n_hicn = 0.0, 0
        with torch.no_grad():
            for comp_tok, vf, cn in mix_val_loader:
                comp_tok = comp_tok.to(device)
                vf       = vf.to(device)
                cn_pred, _ = model(comp_tok, vf)
                abs_err = (cn_pred.cpu() - cn).abs()
                val_mae += abs_err.sum().item()
                n_val   += len(cn)
                hi_mask = cn > HIGH_CN_THRESHOLD
                if hi_mask.sum() > 0:
                    val_mae_hicn += abs_err[hi_mask].sum().item()
                    n_hicn       += hi_mask.sum().item()

        val_mae  /= n_val
        hicn_str  = f"{val_mae_hicn/n_hicn:.3f}" if n_hicn > 0 else "N/A"

        print(
            f"  Ep {epoch:4d}/{epochs} | cn_loss={ep_cn_loss:.4f} "
            f"recon={ep_vae_loss:.4f} | val_MAE={val_mae:.4f} | "
            f"val_MAE(CN>80)={hicn_str}"
        )

        log_writer.writerow({
            "stage": 3, "epoch": epoch,
            "train_total": ep_cn_loss + ep_vae_loss, "train_recon": ep_vae_loss,
            "train_kl": "", "train_acc": "", "val_total": "",
            "val_recon": "", "val_kl": "", "val_acc": "",
            "cn_mae": val_mae, "beta": beta,
        })

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(vae.state_dict(),   ckpt_vae_path)
            torch.save(model.state_dict(), ckpt_model_path)
            print(f"    ✓ Saved joint best (val_MAE={best_mae:.4f})")

        if stopper.step(val_mae):
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"Best joint val_MAE = {best_mae:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SELFIES VAE + CN predictor")
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke-test: 3 epochs per stage, batch_size=16")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                   help="Run only a single stage (default: run all 3)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint .pt file (Stage 1 VAE only)")
    p.add_argument("--s1-epochs", type=int, default=S1_EPOCHS)
    p.add_argument("--s2-epochs", type=int, default=S2_EPOCHS)
    p.add_argument("--s3-epochs", type=int, default=S3_EPOCHS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast:
        args.s1_epochs = 3
        args.s2_epochs = 3
        args.s3_epochs = 3
        batch_size = 16
    else:
        batch_size = BATCH_SIZE

    # ── Load preprocessed data ────────────────────────────────────────────────
    if not CACHE_PATH.exists():
        print(f"Cache not found at {CACHE_PATH}")
        print("Run: python model_training/cn_mixtures_selfies/data/preprocess_selfies.py")
        sys.exit(1)

    print(f"Loading cached data from {CACHE_PATH} …")
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)

    df          = cache["df_selfies"]
    selfies_all = cache["selfies_all"]
    max_seq_len = cache["max_seq_len"]

    tokenizer   = SELFIESTokenizer.load(cache["vocab_path"])
    print(f"  Vocab size   : {tokenizer.vocab_size}")
    print(f"  Max seq len  : {max_seq_len}")
    print(f"  Dataset rows : {len(df)}")

    device = get_device()
    print(f"  Device       : {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build datasets ────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Molecule-level dataset (Stage 1)
    mol_ds = MoleculeDataset(selfies_all, tokenizer, max_seq_len)
    n_mol_val = max(1, int(len(mol_ds) * VAL_FRACTION))
    n_mol_tr  = len(mol_ds) - n_mol_val
    mol_tr_ds, mol_val_ds = random_split(mol_ds, [n_mol_tr, n_mol_val])

    # Mixture-level dataset (Stages 2 & 3)
    df_clean = df.dropna(subset=["CN"]).reset_index(drop=True)
    mix_ds   = MixtureDataset(df_clean, tokenizer, max_seq_len)
    n_mix_val = max(1, int(len(mix_ds) * VAL_FRACTION))
    n_mix_tr  = len(mix_ds) - n_mix_val
    mix_tr_ds, mix_val_ds = random_split(mix_ds, [n_mix_tr, n_mix_val])

    def mk_loader(ds, shuffle, bs=None):
        return DataLoader(ds, batch_size=bs or batch_size, shuffle=shuffle, num_workers=0)

    mol_tr_dl  = mk_loader(mol_tr_ds,  shuffle=True)
    mol_val_dl = mk_loader(mol_val_ds, shuffle=False)
    mix_tr_dl  = mk_loader(mix_tr_ds,  shuffle=True)
    mix_val_dl = mk_loader(mix_val_ds, shuffle=False)

    print(f"  Mol train/val : {len(mol_tr_ds)}/{len(mol_val_ds)}")
    print(f"  Mix train/val : {len(mix_tr_ds)}/{len(mix_val_ds)}")

    # ── Build models ──────────────────────────────────────────────────────────
    vae = SELFIESVAE(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        max_len=max_seq_len,
    ).to(device)

    predictor = CNPredictor(latent_dim=LATENT_DIM, hidden_dims=(256, 128)).to(device)
    model     = MixtureCNModel(vae.encoder, predictor, n_components=N_COMP).to(device)

    n_vae_params  = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    n_pred_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"  VAE params   : {n_vae_params:,}")
    print(f"  Pred params  : {n_pred_params:,}")

    if args.resume:
        print(f"  Resuming VAE from {args.resume}")
        vae.load_state_dict(torch.load(args.resume, map_location=device))

    # ── Logging ───────────────────────────────────────────────────────────────
    fieldnames = [
        "stage", "epoch", "train_total", "train_recon", "train_kl",
        "train_acc", "val_total", "val_recon", "val_kl", "val_acc",
        "cn_mae", "beta",
    ]
    log_fh     = open(LOG_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_fh, fieldnames=fieldnames)
    log_writer.writeheader()

    run_stages = [1, 2, 3] if args.stage is None else [args.stage]

    try:
        if 1 in run_stages:
            stage1_pretrain_vae(
                vae, mol_tr_dl, mol_val_dl,
                epochs=args.s1_epochs,
                device=device,
                ckpt_path=CKPT_DIR / "stage1_vae_best.pt",
                log_writer=log_writer,
            )

        if 2 in run_stages:
            # Load best Stage 1 VAE if available and we're running Stage 2 standalone
            s1_ckpt = CKPT_DIR / "stage1_vae_best.pt"
            if 1 not in run_stages and s1_ckpt.exists():
                print(f"Loading Stage 1 VAE from {s1_ckpt}")
                vae.load_state_dict(torch.load(s1_ckpt, map_location=device))

            stage2_train_predictor(
                model, mix_tr_dl, mix_val_dl,
                epochs=args.s2_epochs,
                device=device,
                ckpt_path=CKPT_DIR / "stage2_pred_best.pt",
                log_writer=log_writer,
            )

        if 3 in run_stages:
            s2_ckpt = CKPT_DIR / "stage2_pred_best.pt"
            if 2 not in run_stages and s2_ckpt.exists():
                print(f"Loading Stage 2 model from {s2_ckpt}")
                model.load_state_dict(torch.load(s2_ckpt, map_location=device))

            stage3_finetune_joint(
                vae=vae,
                model=model,
                mol_train_loader=mol_tr_dl,
                mix_train_loader=mix_tr_dl,
                mix_val_loader=mix_val_dl,
                epochs=args.s3_epochs,
                device=device,
                ckpt_vae_path=CKPT_DIR / "stage3_vae_best.pt",
                ckpt_model_path=CKPT_DIR / "stage3_joint_best.pt",
                log_writer=log_writer,
            )
    finally:
        log_fh.close()

    print(f"\nTraining log saved to {LOG_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
