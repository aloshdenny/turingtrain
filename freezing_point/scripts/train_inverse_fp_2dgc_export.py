"""
train_inverse_fp_2dgc_export.py  (v2 — Maximum Accuracy)
=========================================================
Trains and exports an Inverse Machine Learning Model for Freezing Point:
  Input:  T_freeze_K (scalar freezing point constraint in Kelvin)
  Output: 189 2DGC hydrocarbon mass fraction bins (8 chemical classes, C1-C30)

Physical Constraints Enforced:
  1. Non-negativity: w_i >= 0 for all 189 bins
  2. Conservation of mass (simplex constraint): sum(w_i) = 1.0 strictly via Softmax

Architecture Improvements (v2):
  - FiLM-conditioned deep residual MLP (512-dim, 8 blocks) for richer capacity
  - Fourier temperature embedding with wider frequency range (32 freqs)
  - Two-stage curriculum:
      Phase 1 (epochs 1..n_warmup): Pure composition MSE + Huber + JS-divergence
      Phase 2 (n_warmup..end):     Add end-to-end closed-loop forward surrogate loss
  - Differentiable forward surrogate trained on same data for gradient flow
  - OneCycleLR scheduler for aggressive but stable convergence

Closed-Loop Validation:
  Predicted 189-bin composition distributions are passed into the production
  forward freezing point model (fp_2dgc_seed0.onnx) to verify target fidelity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -- Paths
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_DATA_PATH = _ROOT / "model_training/freezing_point_2dgc/fp_2dgc_selfies_train.dat"
_FORWARD_ONNX = _ROOT / "freezing_point/models/fp_2dgc/fp_2dgc_seed0.onnx"
_OUT_DIR = _ROOT / "freezing_point/models/inverse_fp_2dgc"
_ARTIFACTS_DIR = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")

# -- Feature Specification (189 bins across 8 classes)
FEATURE_GROUPS = {
    "n_paraffin":         list(range(1,  31)),  # 30 bins
    "iso_paraffin":       list(range(4,  31)),  # 27 bins
    "mono_naphthene":     list(range(5,  31)),  # 26 bins
    "di_naphthene":       list(range(10, 31)),  # 21 bins
    "tri_naphthene":      list(range(14, 31)),  # 17 bins
    "mono_aromatic":      list(range(6,  31)),  # 25 bins
    "naphtheno_aromatic": list(range(9,  31)),  # 22 bins
    "di_aromatic":        list(range(10, 31)),  # 21 bins
}
PREFIXES = {
    "n_paraffin":         "w_n_paraffin_C",
    "iso_paraffin":       "w_iso_paraffin_C",
    "mono_naphthene":     "w_mono_naphthene_C",
    "di_naphthene":       "w_di_naphthene_C",
    "tri_naphthene":      "w_tri_naphthene_C",
    "mono_aromatic":      "w_mono_aromatic_C",
    "naphtheno_aromatic": "w_naphtheno_aromatic_C",
    "di_aromatic":        "w_di_aromatic_C",
}
N_BINS = 189


def build_feature_cols():
    cols = []
    for group, carbons in FEATURE_GROUPS.items():
        prefix = PREFIXES[group]
        for c in carbons:
            cols.append(f"{prefix}{c}")
    return cols


# Normalisation constants (fit from data at runtime, global for use in model)
T_MEAN = 227.0
T_STD  = 50.0


def normalise_T(T):
    return (T - T_MEAN) / T_STD


# -- Architecture

class FourierEmbedding(nn.Module):
    """Log-spaced sinusoidal Fourier features for continuous temperature input."""
    def __init__(self, d_model=512, n_freqs=32):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(0.05), math.log(100.0), n_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs + 1, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t_norm):
        angles = t_norm * self.freqs.unsqueeze(0) * (2.0 * math.pi)
        feat = torch.cat([t_norm, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feat)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation residual block.
    Temperature embedding modulates each hidden layer via scale & bias.
    """
    def __init__(self, hidden_dim=512, cond_dim=512, dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff1   = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ff2   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.film  = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, cond):
        film_params = self.film(cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + gamma) + beta
        h = F.silu(self.ff1(h))
        h = self.drop(self.ff2(h))
        x = x + h
        h = self.norm2(x)
        return x + h


class InverseFreezingPointModel(nn.Module):
    """
    v2 Inverse neural regressor: T_freeze_K -> 189 2DGC hydrocarbon mass fractions.
    Fourier embedding + FiLM-conditioned residual blocks + Softmax simplex output.
    """
    def __init__(self, out_features=N_BINS, hidden_dim=512, n_blocks=8, dropout=0.05, n_freqs=32):
        super().__init__()
        self.emb = FourierEmbedding(d_model=hidden_dim, n_freqs=n_freqs)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim=hidden_dim, cond_dim=hidden_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, out_features)

    def forward(self, T_freeze_K):
        t_norm = normalise_T(T_freeze_K)
        cond = self.emb(t_norm)
        x = F.silu(self.input_proj(cond))
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_norm(x)
        logits = self.out_head(x)
        return F.softmax(logits, dim=-1)


# -- Differentiable Forward Surrogate

class ForwardSurrogate(nn.Module):
    """Deeper differentiable surrogate: 189 bins -> T_freeze_K."""
    def __init__(self, in_features=N_BINS, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, w):
        return self.net(w)


def train_surrogate(X_train, y_train, X_val, y_val, device, epochs=30, batch_size=512):
    print("\nPre-training deeper forward surrogate (composition -> T_freeze, 30 epochs) ...")
    surrogate = ForwardSurrogate(in_features=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(surrogate.parameters(), lr=3e-3, weight_decay=1e-5)
    n_steps = math.ceil(len(X_train) / batch_size)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs, steps_per_epoch=n_steps)
    X_t = torch.tensor(X_train, device=device)
    y_t = torch.tensor(y_train, device=device)
    ds  = torch.utils.data.TensorDataset(X_t, y_t)
    dl  = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        surrogate.train()
        ep_loss = 0.0
        for xb, yb in dl:
            pred = surrogate(xb)
            loss = F.huber_loss(pred, yb, delta=10.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_t)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            surrogate.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, device=device)
                yv = torch.tensor(y_val, device=device)
                mae_v = F.l1_loss(surrogate(xv), yv).item()
            print(f"  Surrogate Epoch {ep:2d}/{epochs} | Loss: {ep_loss:.4f} | Val MAE: {mae_v:.2f} K")

    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(True)
    return surrogate


# -- Loss Functions

def composition_loss(pred_w, true_w):
    """MSE + Huber + Jensen-Shannon divergence in composition space."""
    mse  = F.mse_loss(pred_w, true_w) * 1000.0
    hub  = F.huber_loss(pred_w, true_w, delta=0.02) * 200.0
    eps  = 1e-8
    m    = 0.5 * (pred_w + true_w) + eps
    p    = pred_w + eps
    q    = true_w + eps
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    jsd  = 0.5 * (kl_pm + kl_qm).mean() * 2.0
    return mse + hub + jsd


def forward_loop_loss(pred_w, target_T, surrogate, T_std):
    """End-to-end closed-loop forward temperature penalty."""
    pred_T = surrogate(pred_w)
    err_K  = (pred_T - target_T) / T_std
    return (err_K ** 2).mean() * 50.0


# -- Main

def main():
    parser = argparse.ArgumentParser(description="Train v2 Inverse Freezing Point Model.")
    parser.add_argument("--epochs",     type=int,   default=80,   help="Total epochs")
    parser.add_argument("--warmup",     type=int,   default=20,   help="Warmup epochs (composition loss only)")
    parser.add_argument("--batch_size", type=int,   default=512,  help="Batch size")
    parser.add_argument("--lr",         type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--hidden",     type=int,   default=512,  help="Hidden dimension")
    parser.add_argument("--blocks",     type=int,   default=8,    help="Number of FiLM residual blocks")
    parser.add_argument("--dropout",    type=float, default=0.04, help="Dropout rate")
    parser.add_argument("--fwd_weight", type=float, default=0.30, help="Max forward surrogate loss weight (Phase 2)")
    args = parser.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    feature_cols = build_feature_cols()
    n_bins = len(feature_cols)
    print(f"Features: {n_bins} hydrocarbon bins")

    # Load data
    print(f"\nLoading {_DATA_PATH} ...")
    t0 = time.time()
    df = pd.read_csv(_DATA_PATH, sep="\t", comment="#")
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"  Clean rows: {len(df):,}  ({time.time()-t0:.2f}s)")

    X_all = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y_all = df[["T_freeze_K"]].values.astype(np.float32)

    # Fit normalisation constants
    global T_MEAN, T_STD
    T_MEAN = float(y_all.mean())
    T_STD  = float(y_all.std()) + 1e-8
    print(f"  T_freeze_K  mean={T_MEAN:.2f} K, std={T_STD:.2f} K")

    # 90/10 split
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.10, random_state=42)
    X_train_np, y_train_np = X_all[idx_train], y_all[idx_train]
    X_val_np,   y_val_np   = X_all[idx_val],   y_all[idx_val]
    print(f"  Train: {len(X_train_np):,}  |  Val: {len(X_val_np):,}")

    # Pre-train surrogate
    surrogate = train_surrogate(
        X_train_np, y_train_np, X_val_np, y_val_np, device,
        epochs=30, batch_size=512,
    )

    # Build DataLoaders
    X_train_t = torch.tensor(X_train_np, device=device)
    y_train_t = torch.tensor(y_train_np, device=device)
    X_val_t   = torch.tensor(X_val_np,   device=device)
    y_val_t   = torch.tensor(y_val_np,   device=device)

    train_ds = torch.utils.data.TensorDataset(y_train_t, X_train_t)
    val_ds   = torch.utils.data.TensorDataset(y_val_t,   X_val_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Build inverse model
    model = InverseFreezingPointModel(
        out_features=n_bins,
        hidden_dim=args.hidden,
        n_blocks=args.blocks,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInverse model parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    best_val_loss = float("inf")
    best_weights  = None

    print(f"\n{'='*65}")
    print(f"Training Inverse FP Model (v2) | {args.epochs} epochs | warmup={args.warmup}")
    print(f"  Phase 1 (1-{args.warmup}): Composition loss only")
    print(f"  Phase 2 ({args.warmup+1}-{args.epochs}): + Fwd loss (max_weight={args.fwd_weight}, ramp=20 ep)")
    print(f"{'='*65}\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        surrogate.train()
        use_fwd = (epoch > args.warmup)
        # Ramp forward weight over 20 epochs, capped at args.fwd_weight (default 0.3)
        fwd_w   = min(args.fwd_weight, args.fwd_weight * (epoch - args.warmup) / 20.0) if use_fwd else 0.0

        train_loss = 0.0
        for y_b, x_b in train_loader:
            pred_w = model(y_b)
            loss = composition_loss(pred_w, x_b)
            if use_fwd and fwd_w > 0:
                loss = loss + fwd_w * forward_loop_loss(pred_w, y_b, surrogate, T_STD)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(X_train_t)

        model.eval()
        surrogate.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_b, x_b in val_loader:
                pred_w = model(y_b)
                loss = composition_loss(pred_w, x_b)
                if use_fwd and fwd_w > 0:
                    loss = loss + fwd_w * forward_loop_loss(pred_w, y_b, surrogate, T_STD)
                val_loss += loss.item() * len(y_b)
        val_loss /= len(X_val_t)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch <= 3 or epoch == args.epochs:
            lr_now = optimiser.param_groups[0]["lr"]
            phase  = "P1-Comp" if epoch <= args.warmup else f"P2-FwdW={fwd_w:.2f}"
            print(f"  Epoch {epoch:3d}/{args.epochs} | {phase:16s} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.2e}")

    print(f"\nTraining done in {(time.time()-t_start)/60:.1f} min  |  Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_weights)
    model.eval()

    # Closed-loop verification with production ONNX forward model
    print("\n" + "=" * 65)
    print("Closed-Loop Verification with Forward Model (fp_2dgc_seed0.onnx)")
    print("=" * 65)

    fwd_session = ort.InferenceSession(str(_FORWARD_ONNX))

    val_sample_size = min(20000, len(y_val_np))
    val_y_np_sub = y_val_np[:val_sample_size]

    with torch.no_grad():
        pred_w_val = model(
            torch.tensor(val_y_np_sub, device=device)
        ).cpu().numpy().astype(np.float32)

    realized_T_val = fwd_session.run(None, {"float_input": pred_w_val})[0]

    target_flat   = val_y_np_sub.flatten()
    realized_flat = realized_T_val.flatten()

    mae_K     = float(mean_absolute_error(target_flat, realized_flat))
    rmse_K    = float(np.sqrt(mean_squared_error(target_flat, realized_flat)))
    r2        = float(r2_score(target_flat, realized_flat))
    within_5K = float(np.mean(np.abs(target_flat - realized_flat) <= 5.0)  * 100.0)
    within_10K = float(np.mean(np.abs(target_flat - realized_flat) <= 10.0) * 100.0)

    print(f"\n--- Closed-Loop Validation Metrics ---")
    print(f"  Target Parity R2  : {r2:.4f}")
    print(f"  Parity MAE        : {mae_K:.2f} K")
    print(f"  Parity RMSE       : {rmse_K:.2f} K")
    print(f"  Within +/-5 K     : {within_5K:.1f}%")
    print(f"  Within +/-10 K    : {within_10K:.1f}%")
    print(f"  Mass Sum Check    : min={pred_w_val.sum(axis=1).min():.5f}, max={pred_w_val.sum(axis=1).max():.5f}")

    # Operational sweep
    sweep_targets = [100.0, 150.0, 180.0, 200.0, 216.5, 230.0, 245.0, 260.0, 280.0, 300.0, 325.0, 350.0]
    sweep_t = torch.tensor([[t] for t in sweep_targets], dtype=torch.float32, device=device)
    with torch.no_grad():
        sweep_w = model(sweep_t).cpu().numpy().astype(np.float32)
    sweep_realized = fwd_session.run(None, {"float_input": sweep_w})[0].flatten()

    print("\nOperational Temperature Sweep Validation:")
    print(f"{'Target T_freeze':>16} | {'Realized T_freeze':>18} | {'Error':>8} | Top Class")
    print("-" * 80)
    for t_tgt, t_rl, w_row in zip(sweep_targets, sweep_realized, sweep_w):
        err = abs(t_rl - t_tgt)
        class_sums = {}
        idx_c = 0
        for grp, carbons in FEATURE_GROUPS.items():
            n_c = len(carbons)
            class_sums[grp] = w_row[idx_c:idx_c + n_c].sum()
            idx_c += n_c
        top_cls = max(class_sums.items(), key=lambda x: x[1])
        print(f"{t_tgt:14.1f} K | {t_rl:16.2f} K | {err:8.2f} K | {top_cls[0]} ({top_cls[1]*100:.1f}%)")

    # Parity plot + residual histogram
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    parity_path = _OUT_DIR / "inverse_fp_2dgc_parity.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=180)

    ax = axes[0]
    idx_plot = np.random.choice(len(target_flat), min(10000, len(target_flat)), replace=False)
    ax.scatter(target_flat[idx_plot], realized_flat[idx_plot],
               alpha=0.30, s=10, c="#1c6ea4", edgecolors="none")
    lims = [min(target_flat.min(), realized_flat.min()) - 5,
            max(target_flat.max(), realized_flat.max()) + 5]
    ax.plot(lims, lims, "r--", lw=1.8, label="1:1 Parity")
    ax.set_xlabel("Target T_freeze_K (Input)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Realized T_freeze_K (Forward)", fontsize=11, fontweight="bold")
    ax.set_title("Inverse FP Model v2 - Closed-Loop Parity", fontsize=11, fontweight="bold")
    stats = (f"R2   = {r2:.4f}\nMAE  = {mae_K:.2f} K\nRMSE = {rmse_K:.2f} K\n"
             f"+-5K = {within_5K:.1f}%\n+-10K= {within_10K:.1f}%")
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88))
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    ax2 = axes[1]
    residuals = realized_flat - target_flat
    ax2.hist(residuals, bins=80, color="#1c6ea4", edgecolor="none", alpha=0.75)
    ax2.axvline(0, color="red", ls="--", lw=1.5)
    ax2.set_xlabel("Residual: Realized - Target (K)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Residual Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)

    plt.suptitle("Inverse Freezing Point 2DGC Model (v2) - Max Accuracy", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(parity_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved parity plot -> {parity_path}")

    if _ARTIFACTS_DIR.exists():
        shutil.copy2(parity_path, _ARTIFACTS_DIR / "inverse_fp_2dgc_parity.png")

    # ONNX Export
    print("\nExporting Inverse Model to ONNX ...")
    onnx_path = _OUT_DIR / "inverse_fp_2dgc.onnx"
    export_model = InverseFreezingPointModel(
        out_features=n_bins,
        hidden_dim=args.hidden,
        n_blocks=args.blocks,
        dropout=0.0,
    ).eval()
    export_model.load_state_dict(best_weights)

    dummy_input = torch.tensor([[230.0]], dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        input_names=["T_freeze_K"],
        output_names=["hydrocarbon_bins_189"],
        dynamic_axes={"T_freeze_K": {0: "batch"}, "hydrocarbon_bins_189": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported ONNX -> {onnx_path} ({onnx_path.stat().st_size/1024/1024:.2f} MB)")

    pt_path = _OUT_DIR / "inverse_fp_2dgc.pt"
    torch.save(best_weights, pt_path)
    print(f"Saved PyTorch weights -> {pt_path}")

    norm_path = _OUT_DIR / "inverse_fp_2dgc_norm.json"
    with open(norm_path, "w") as f:
        json.dump({"T_MEAN": T_MEAN, "T_STD": T_STD}, f, indent=2)
    print(f"Saved normalisation constants -> {norm_path}")

    meta = {
        "model_type": "InverseFreezingPointModel v2 (Fourier + FiLM ResNet + Simplex Softmax)",
        "description": "Predicts 189 2DGC hydrocarbon mass fractions from target T_freeze_K",
        "dataset": "model_training/freezing_point_2dgc/fp_2dgc_selfies_train.dat",
        "n_samples": len(df),
        "n_features_out": n_bins,
        "feature_cols": feature_cols,
        "feature_groups": FEATURE_GROUPS,
        "architecture": {
            "hidden_dim": args.hidden,
            "n_blocks": args.blocks,
            "n_freqs": 32,
            "dropout": args.dropout,
            "conditioning": "FiLM per-block",
            "output_activation": "Softmax (simplex)",
        },
        "training": {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "scheduler": "OneCycleLR",
            "phase1_loss": "MSE + Huber + JS-Divergence",
            "phase2_loss": "MSE + Huber + JS-Divergence + ForwardSurrogateLoss",
        },
        "closed_loop_metrics": {
            "parity_r2": r2,
            "mae_K": mae_K,
            "rmse_K": rmse_K,
            "within_5K_pct": within_5K,
            "within_10K_pct": within_10K,
        },
        "onnx_export": onnx_path.name,
    }
    meta_path = _OUT_DIR / "inverse_fp_2dgc_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {meta_path}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Closed-Loop Parity R2 : {r2:.4f}")
    print(f"  Closed-Loop MAE       : {mae_K:.2f} K")
    print(f"  Closed-Loop RMSE      : {rmse_K:.2f} K")
    print(f"  Within +/-5 K         : {within_5K:.1f}%")
    print(f"  Within +/-10 K        : {within_10K:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
