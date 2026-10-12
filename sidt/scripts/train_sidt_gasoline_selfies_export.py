"""
train_sidt_gasoline_selfies_export.py
======================================
Train a deep-learning IDT model for the 10-component gasoline surrogate
dataset using the CN-MXTR approach:

  - Stage 1 (frozen encoder): Use the pretrained VAE encoder
    (SELFIES/checkpoints_opt/seed{i}_s1_vae.pt) to embed each of the 10
    component SELFIES strings.
  - Mixture latent: z_mix = Σᵢ xᵢ · μᵢ (mole-fraction weighted sum)
  - Concatenate z_mix with reactor conditions [pressure_pa, temperature_K,
    phi, egr_fraction] to form the full feature vector.
  - Train an MLP regressor on log1p(idt_400K_s) using MSE loss on MPS/CPU.
  - Final model is exported as ONNX for onnxruntime inference.

This mirrors mixture_cn_predictor.py / MixtureCNModel but adapted for
the SIDT regression task (idt_400K_s) with reactor condition concatenation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SELFIES_DIR = _ROOT / "SELFIES"

sys.path.insert(0, str(_SELFIES_DIR))
sys.path.insert(0, str(_SELFIES_DIR / "vae"))
from selfies_tokenizer import SELFIESTokenizer, split_selfies  # noqa: E402
from selfies_vae import SELFIESEncoder, SELFIESVAE               # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
N_COMPONENTS    = 10
MAX_SELFIES_LEN = 65          # must match the pretrained VAE checkpoint (pos_enc shape [1, 65, 128])
LATENT_DIM      = 128         # from the optimized checkpoints (latent_dim=128)
D_MODEL         = 128         # from the optimized checkpoints
N_HEADS         = 4
N_LAYERS        = 4
D_FF            = 512
REACTOR_DIM     = 4           # pressure_pa, temperature_K, phi, egr_fraction
HIDDEN_DIMS     = (512, 256, 128)
DROPOUT         = 0.2
LR              = 3e-4
BATCH_SIZE      = 2048
EPOCHS          = 40
N_SEEDS         = 5           # ensemble over 5 pretrained VAE seeds

# ── Reactor condition normalisation constants ─────────────────────────────────
# Defaults (recomputed dynamically from dataset in main)
COND_MEAN = np.array([2499338.3, 1001.4, 0.9998, 0.1498], dtype=np.float32)
COND_STD  = np.array([1170061.3, 222.11,  0.3019, 0.0891], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GasolineSIDTDataset:
    """Stores dataset tensors and component tokens for fast vectorized training."""

    def __init__(
        self,
        df: pd.DataFrame,
        comp_tokens_10: torch.Tensor,
        cond_mean: np.ndarray = COND_MEAN,
        cond_std: np.ndarray = COND_STD,
    ) -> None:
        self.comp_tokens = comp_tokens_10  # (10, MAX_SELFIES_LEN)

        # Mole fractions: (N, 10) — normalised so they sum to 1
        frac_cols = [f"cpnt_mole_frac_{i}" for i in range(1, N_COMPONENTS + 1)]
        fracs = df[frac_cols].values.astype(np.float32)
        row_sums = fracs.sum(axis=1, keepdims=True).clip(min=1e-8)
        self.fracs = torch.tensor(fracs / row_sums)

        # Reactor conditions: standardised (N, 4)
        conds = df[["pressure_pa", "temperature_K", "phi", "egr_fraction"]].values.astype(np.float32)
        self.conds = torch.tensor((conds - cond_mean) / cond_std)

        # Target: log1p transform for heavy-tail stabilisation
        idt = df["idt_400K_s"].values.astype(np.float32)
        self.y = torch.tensor(np.log1p(idt))

    def __len__(self) -> int:
        return self.fracs.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# IDT Predictor MLP
# ─────────────────────────────────────────────────────────────────────────────

class IDTPredictor(nn.Module):
    """MLP: (z_mix, reactor_conds) → scalar log1p(idt_400K_s)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        reactor_dim: int = REACTOR_DIM,
        hidden_dims: tuple = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + reactor_dim
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mix: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        """(batch, latent_dim), (batch, 4) → (batch,)"""
        x = torch.cat([z_mix, conds], dim=-1)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(
    seed_idx: int,
    train_ds: GasolineSIDTDataset,
    val_ds: GasolineSIDTDataset,
    tokenizer: SELFIESTokenizer,
    device: torch.device,
    out_dir: Path,
    n_epochs: int = EPOCHS,
) -> tuple[IDTPredictor, SELFIESEncoder, torch.Tensor]:
    torch.manual_seed(seed_idx)
    np.random.seed(seed_idx)

    # Load pretrained VAE encoder for this seed
    ck_path = _SELFIES_DIR / "checkpoints_opt" / f"seed{seed_idx}_s1_vae.pt"
    if not ck_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ck_path}")

    state = torch.load(ck_path, map_location="cpu", weights_only=False)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}

    encoder = SELFIESEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        pad_idx=tokenizer.pad_idx,
        max_len=MAX_SELFIES_LEN,
    ).to(device)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Precompute latent representations for the 10 components
    with torch.no_grad():
        mu_10, _ = encoder(train_ds.comp_tokens.to(device))  # (10, 128)
        # Vectorized mixture latent: z_mix = fracs @ mu_10
        train_z_mix = torch.matmul(train_ds.fracs.to(device), mu_10)  # (N_train, 128)
        val_z_mix   = torch.matmul(val_ds.fracs.to(device),   mu_10)  # (N_val, 128)

    train_tensor_ds = torch.utils.data.TensorDataset(train_z_mix, train_ds.conds.to(device), train_ds.y.to(device))
    val_tensor_ds   = torch.utils.data.TensorDataset(val_z_mix,   val_ds.conds.to(device),   val_ds.y.to(device))

    train_loader = DataLoader(train_tensor_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_tensor_ds,   batch_size=BATCH_SIZE, shuffle=False)

    predictor = IDTPredictor().to(device)
    optimiser = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        predictor.train()
        train_loss = 0.0
        for z, conds, y in train_loader:
            pred = predictor(z, conds)
            loss = F.mse_loss(pred, y)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        predictor.eval()
        all_pred, all_y = [], []
        with torch.no_grad():
            for z, conds, y in val_loader:
                pred = predictor(z, conds)
                all_pred.append(pred.cpu())
                all_y.append(y.cpu())
        all_pred = torch.cat(all_pred)
        all_y    = torch.cat(all_y)
        # Back-transform for real-unit MAE
        pred_idt = torch.expm1(all_pred).clamp(min=0)
        true_idt = torch.expm1(all_y).clamp(min=0)
        val_mae  = (pred_idt - true_idt).abs().mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in predictor.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [seed {seed_idx}] Epoch {epoch:3d}/{n_epochs} | "
                f"Train MSE(log): {train_loss:.4f} | Val MAE: {val_mae*1000:.2f} ms"
            )

    print(f"  [seed {seed_idx}] Best Val MAE = {best_val_mae*1000:.2f} ms")
    predictor.load_state_dict(best_state)
    torch.save(best_state, out_dir / f"predictor_seed{seed_idx}.pt")
    return predictor, encoder, val_z_mix


# ─────────────────────────────────────────────────────────────────────────────
import types

# ─────────────────────────────────────────────────────────────────────────────
# ONNX export helper (single predictor, full end-to-end forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_encoder_layers_for_onnx(encoder: SELFIESEncoder) -> None:
    """Patch TransformerEncoderLayer forward methods to avoid aten::_transformer_encoder_layer_fwd
    which is not supported by torch.onnx at opset 17.
    """
    def layer_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pre-LayerNorm (norm_first=True)
        x = self.norm1(src)
        x2, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(x2)
        x = self.norm2(src)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x2)
        return src

    for layer in encoder.transformer.layers:
        layer.forward = types.MethodType(layer_forward, layer)


def export_full_model_onnx(
    encoder: SELFIESEncoder,
    predictor: IDTPredictor,
    out_path: Path,
    device: torch.device,
    n_components: int = N_COMPONENTS,
    max_len: int = MAX_SELFIES_LEN,
) -> None:
    """Export a single (encoder + predictor) forward pass to ONNX on CPU.

    Inputs:
        component_tokens: (1, n_components, max_len) int64
        mole_fracs:       (1, n_components)          float32
        reactor_conds:    (1, 4)                     float32  (standardised)

    Output:
        idt_log1p: (1,) float32  — call expm1() to get seconds
    """

    class FullModel(nn.Module):
        def __init__(self, enc, pred):
            super().__init__()
            self.encoder = enc
            self.predictor = pred

        def forward(self, tokens: torch.Tensor, fracs: torch.Tensor, conds: torch.Tensor):
            B, n_comp, seq_len = tokens.shape
            flat = tokens.reshape(B * n_comp, seq_len)
            mu, _ = self.encoder(flat)
            mu = mu.view(B, n_comp, -1)
            vf = fracs.unsqueeze(-1)
            z_mix = (mu * vf).sum(dim=1)
            return self.predictor(z_mix, conds)

    # Clone encoder to CPU and patch for ONNX export
    enc_cpu = encoder.to("cpu")
    pred_cpu = predictor.to("cpu")
    _patch_encoder_layers_for_onnx(enc_cpu)

    full = FullModel(enc_cpu, pred_cpu).eval()

    dummy_tokens = torch.zeros(1, n_components, max_len, dtype=torch.long, device="cpu")
    dummy_fracs  = torch.full((1, n_components), 1.0 / n_components, device="cpu")
    dummy_conds  = torch.zeros(1, REACTOR_DIM, device="cpu")

    torch.onnx.export(
        full,
        (dummy_tokens, dummy_fracs, dummy_conds),
        str(out_path),
        input_names=["component_tokens", "mole_fracs", "reactor_conds"],
        output_names=["idt_log1p"],
        dynamic_axes={
            "component_tokens": {0: "batch"},
            "mole_fracs":       {0: "batch"},
            "reactor_conds":    {0: "batch"},
            "idt_log1p":        {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Exported ONNX model to {out_path} ({sz_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Train SIDT IDT model for gasoline mixture using SELFIES VAE encoder.")
    parser.add_argument("--input", type=str,
                        default="model_training/sidt/sidt_lhs_gasoline2_K10_1M.dat",
                        help="Path to dataset .dat file")
    parser.add_argument("--out_dir", type=str,
                        default="sidt/models/gasoline_mix_selfies",
                        help="Output directory for checkpoints and ONNX models")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of VAE seeds to use (1–5)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs per seed")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    args = parser.parse_args()

    input_path = _ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_dir    = _ROOT / args.out_dir  if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {input_path}…")
    t0_load = time.time()
    feature_cols = (
        [f"cpnt_mole_frac_{i}" for i in range(1, N_COMPONENTS + 1)]
        + [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
        + ["pressure_pa", "temperature_K", "phi", "egr_fraction", "idt_400K_s"]
    )
    df = pd.read_csv(input_path, sep="\t", comment="#", usecols=feature_cols, low_memory=False)
    df_clean = df.dropna(subset=["idt_400K_s"]).reset_index(drop=True)
    print(f"Clean rows: {len(df_clean):,} (loaded in {time.time()-t0_load:.1f}s)")

    # Compute reactor condition standardisation constants dynamically
    cond_cols = ["pressure_pa", "temperature_K", "phi", "egr_fraction"]
    cond_mean = df_clean[cond_cols].mean().values.astype(np.float32)
    cond_std  = df_clean[cond_cols].std().values.astype(np.float32)
    print(f"Condition Mean : {cond_mean.tolist()}")
    print(f"Condition Std  : {cond_std.tolist()}")

    # Train / val split (90/10)
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df_clean, test_size=0.1, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = _SELFIES_DIR / "data" / "vocab.json"
    print(f"Loading tokenizer from {tok_path}…")
    tokenizer = SELFIESTokenizer.load(tok_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Tokenize the 10 fixed component SELFIES strings
    selfies_cols = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    comp_tokens = torch.zeros(N_COMPONENTS, MAX_SELFIES_LEN, dtype=torch.long)
    for i, col in enumerate(selfies_cols):
        s_val = df_clean[col].iloc[0]
        comp_tokens[i] = tokenizer.encode(s_val, MAX_SELFIES_LEN)
    print(f"Tokenized {N_COMPONENTS} component SELFIES strings successfully.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = GasolineSIDTDataset(df_train, comp_tokens, cond_mean=cond_mean, cond_std=cond_std)
    val_ds   = GasolineSIDTDataset(df_val,   comp_tokens, cond_mean=cond_mean, cond_std=cond_std)

    # ── Train ensemble ────────────────────────────────────────────────────────
    n_seeds = min(args.seeds, N_SEEDS)
    n_epochs = args.epochs
    predictors = []
    encoders   = []
    val_z_mixes = []
    t_train_start = time.time()
    for seed_idx in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Training predictor — Seed {seed_idx}")
        print(f"{'='*60}")
        pred, enc, val_z = train_one_seed(
            seed_idx, train_ds, val_ds, tokenizer, device, out_dir,
            n_epochs=n_epochs
        )
        predictors.append(pred)
        encoders.append(enc)
        val_z_mixes.append(val_z)
    print(f"\nEnsemble training completed in {(time.time()-t_train_start)/60:.1f} min.")

    # ── Ensemble validation ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Ensemble validation…")
    for pred in predictors:
        pred.eval()

    all_preds_ensemble = []
    with torch.no_grad():
        conds_val = val_ds.conds.to(device)
        seed_preds = []
        for i in range(n_seeds):
            pred_i = predictors[i](val_z_mixes[i], conds_val)
            seed_preds.append(pred_i)
        ensemble_pred = torch.stack(seed_preds).mean(dim=0).cpu()

    true_y = val_ds.y.cpu()
    pred_idt = torch.expm1(ensemble_pred).clamp(min=0).numpy()
    true_idt = torch.expm1(true_y).clamp(min=0).numpy()
    ensemble_mae  = float(np.mean(np.abs(pred_idt - true_idt)))
    ensemble_rmse = float(np.sqrt(np.mean((pred_idt - true_idt)**2)))
    r2_score = 1.0 - float(np.sum((true_idt - pred_idt)**2)) / (float(np.sum((true_idt - np.mean(true_idt))**2)) + 1e-8)

    print(f"Ensemble MAE  = {ensemble_mae*1000:.3f} ms ({ensemble_mae:.6f} s)")
    print(f"Ensemble RMSE = {ensemble_rmse*1000:.3f} ms ({ensemble_rmse:.6f} s)")
    print(f"Ensemble R²   = {r2_score:.4f}")

    # Sample prediction (P = 10 bar / 1.0 MPa, T = 1000 K, phi = 1.0, EGR = 0.0, equimolar mix)
    sample_fracs = torch.full((1, 10), 0.1, dtype=torch.float32, device=device)
    raw_cond = np.array([1.0e6, 1000.0, 1.0, 0.0], dtype=np.float32)
    sample_cond = torch.tensor((raw_cond - cond_mean) / cond_std, device=device).unsqueeze(0)
    with torch.no_grad():
        preds_s = []
        for i in range(n_seeds):
            mu_10, _ = encoders[i](comp_tokens.to(device))
            z_s = torch.matmul(sample_fracs, mu_10)
            preds_s.append(predictors[i](z_s, sample_cond))
        sample_pred_val = torch.stack(preds_s).mean(0)
    sample_pred_idt = float(torch.expm1(sample_pred_val).clamp(min=0).item())
    print(f"\nStandard Sample (10 bar, 1000 K, phi=1, egr=0, 10-mix): {sample_pred_idt*1000:.3f} ms ({sample_pred_idt:.6f} s)")

    # ── Parity Plot ──────────────────────────────────────────────────────────
    parity_path = _ROOT / "sidt/gasoline_mix_selfies_parity.png"
    plt.figure(figsize=(7, 6), dpi=200)
    n_pts = min(15000, len(true_idt))
    idx = np.random.choice(len(true_idt), n_pts, replace=False)
    plt.scatter(true_idt[idx]*1000.0, pred_idt[idx]*1000.0, alpha=0.3, s=10, c="#1f77b4", edgecolors="none")
    min_val = min(true_idt[idx].min()*1000.0, pred_idt[idx].min()*1000.0)
    max_val = max(true_idt[idx].max()*1000.0, pred_idt[idx].max()*1000.0)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="1:1 Identity")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True IDT (ms)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted IDT (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline Surrogate IDT — Parity (CN-MXTR SELFIES)", fontsize=12, fontweight="bold")
    stats_text = f"MAE: {ensemble_mae*1000:.2f} ms\nRMSE: {ensemble_rmse*1000:.2f} ms\nR²: {r2_score:.4f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(parity_path, dpi=200)
    plt.close()
    print(f"✓ Saved parity plot to {parity_path}")

    # ── Forward Sweep (Arrhenius) Plot ───────────────────────────────────────
    fwd_path = _ROOT / "sidt/gasoline_mix_selfies_forward_plot.png"
    plt.figure(figsize=(7, 6), dpi=200)
    temps = np.linspace(600.0, 1500.0, 100)
    pressures_bar = [10.0, 20.0, 40.0]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    sweep_fracs = torch.full((len(temps), 10), 0.1, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_blend, _ = encoders[0](comp_tokens.to(device))
        blend_z = torch.matmul(sweep_fracs, mu_blend)

        for p_bar, c in zip(pressures_bar, colors):
            p_pa = p_bar * 1.0e5
            conds_sweep = np.array([[p_pa, t, 1.0, 0.0] for t in temps], dtype=np.float32)
            conds_tensor = torch.tensor((conds_sweep - cond_mean) / cond_std, device=device)

            preds_sweep = []
            for i in range(n_seeds):
                pred_i = predictors[i](blend_z, conds_tensor)
                preds_sweep.append(pred_i)
            pred_val = torch.stack(preds_sweep).mean(0)
            idt_sweep_ms = torch.expm1(pred_val).clamp(min=0).cpu().numpy() * 1000.0

            inv_t = 1000.0 / temps
            plt.plot(inv_t, idt_sweep_ms, lw=2.0, color=c, label=f"{p_bar:.0f} bar")

    plt.yscale("log")
    plt.xlabel("1000 / T (1/K)", fontsize=11, fontweight="bold")
    plt.ylabel("Ignition Delay Time (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline Surrogate Equimolar Blend — Arrhenius Sweep", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(title="Pressure (φ=1.0, EGR=0.0)")
    plt.tight_layout()
    plt.savefig(fwd_path, dpi=200)
    plt.close()
    print(f"✓ Saved forward sweep plot to {fwd_path}")

    # Copy plots to artifacts directory
    artifacts_dir = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")
    if artifacts_dir.exists():
        import shutil
        shutil.copy2(parity_path, artifacts_dir / "gasoline_mix_selfies_parity.png")
        shutil.copy2(fwd_path, artifacts_dir / "gasoline_mix_selfies_forward_plot.png")

    # ── Export best-seed ONNX ────────────────────────────────────────────────
    print("\nExporting ONNX model (seed 0)…")
    best_pred = predictors[0]
    best_enc  = encoders[0]
    export_full_model_onnx(
        best_enc, best_pred,
        out_path=out_dir / "idt_gasoline_selfies.onnx",
        device=device,
    )

    # Save normalisation constants and metadata for inference
    meta = {
        "dataset": str(input_path.name),
        "n_samples_clean": len(df_clean),
        "n_components": N_COMPONENTS,
        "max_selfies_len": MAX_SELFIES_LEN,
        "latent_dim": LATENT_DIM,
        "reactor_dim": REACTOR_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "cond_mean": cond_mean.tolist(),
        "cond_std": cond_std.tolist(),
        "vocab_path": str(tok_path),
        "n_seeds_trained": n_seeds,
        "val_mae_s": ensemble_mae,
        "val_rmse_s": ensemble_rmse,
        "val_r2": r2_score,
        "sample_pred_idt_s": sample_pred_idt,
    }
    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved model metadata to {meta_path}")
    print(f"\n✓ Training complete. Models saved to {out_dir}")


if __name__ == "__main__":
    main()
