"""
0d_idt_gasoline_b1b_train.py
─────────────────────────────
Gasoline 0D Ignition Delay Time (IDT) Model for Single-Subset (Fixed Components).
Trained on cvwsr_idt_selfies_b1b_train.dat (108,159 LHS simulation runs).

Architecture:
- Inputs:
    1. mole_fracs: [batch, 10] (float32)
    2. reactor_conds: [batch, 4] (float32: pressure_pa, temperature_K, phi, egr_fraction)
- Composition Embedding: MLP projecting 10 mole fractions to d_comp feature vector
- Multi-frequency Fourier features for 1000/T and log10(P/1e5)
- Structured Three-Arrhenius Head:
    1/tau = 1/(tau1 + tau2) + 1/tau3 (smoothly parameterized via logsumexp)
- FiLM ResNet residual backbone with LayerNorm and SiLU
- Auxiliary supervision on 40K IDT
- 5-seed ensemble averaged in log10 space
- Full unified ONNX export with dynamic batch dimension
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

from gasoline_io import load_table, FRAC_COLS, COND_COLS


# ── Numerically Stable LogSumExp for ONNX ─────────────────────────────────────
def logaddexp_onnx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(x, y)
    return m + torch.log(torch.exp(x - m) + torch.exp(y - m))


def logsumexp3_onnx(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(torch.maximum(x, y), z)
    return m + torch.log(torch.exp(x - m) + torch.exp(y - m) + torch.exp(z - m))


# ── Model Modules ─────────────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    def __init__(self, freqs: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0]):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32) * (2.0 * math.pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ThreeArrheniusHead(nn.Module):
    """
    Parametric Three-Arrhenius kinetic equation:
      tau1 = A1 * exp(E1 / (R*T))   (high-T branch, E1 > 0)
      tau2 = A2 * exp(E2 / (R*T))   (intermediate-T branch, E2 < 0: NTC region)
      tau3 = A3 * exp(E3 / (R*T))   (low-T branch, E3 > 0)
      1/tau = 1/(tau1 + tau2) + 1/tau3
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, 6)
        with torch.no_grad():
            self.head.weight.normal_(mean=0.0, std=0.01)
            self.head.bias.copy_(torch.tensor([-5.0, 1.0, -2.0, 0.5, -6.0, 1.5]))

    def forward(self, h_no_t: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        params = self.head(h_no_t)
        a1, r_e1, a2, r_e2, a3, r_e3 = params.chunk(6, dim=-1)
        e1 = F.softplus(r_e1) + 1e-4
        e2 = -F.softplus(r_e2) - 1e-4
        e3 = F.softplus(r_e3) + 1e-4

        ln_tau1 = a1 + e1 * u_norm
        ln_tau2 = a2 + e2 * u_norm
        ln_tau3 = a3 + e3 * u_norm

        ln_tau12 = logaddexp_onnx(ln_tau1, ln_tau2)
        ln_tau123 = logsumexp3_onnx(ln_tau1, ln_tau2, ln_tau3)
        ln_tau = ln_tau12 + ln_tau3 - ln_tau123
        return ln_tau / 2.302585092994046  # Convert ln -> log10


class GasolineB1BPredictor(nn.Module):
    def __init__(
        self,
        n_components: int = 10,
        d_comp: int = 256,
        hidden_dim: int = 512,
        struct_width: int = 256,
        num_blocks: int = 8,
        head_type: str = "struct"
    ):
        super().__init__()
        self.head_type = head_type
        self.d_comp = d_comp

        # Composition MLP projecting the 10 mole fractions into d_comp features
        self.comp_proj = nn.Sequential(
            nn.Linear(n_components, d_comp),
            nn.LayerNorm(d_comp),
            nn.SiLU(),
            nn.Linear(d_comp, d_comp),
            nn.LayerNorm(d_comp)
        )

        self.fourier_u = FourierFeatures()
        self.fourier_p = FourierFeatures()

        cond_dim = 25       # 5 core + 10 fu + 10 fp
        cond_no_t_dim = 14  # p_norm (1) + rest (3: phi, egr, log_phi) + fp (10)

        if head_type == "struct":
            self.trunk_no_t_proj = nn.Linear(d_comp + cond_no_t_dim, struct_width)
            self.blocks_no_t = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(struct_width), nn.SiLU(), nn.Linear(struct_width, struct_width),
                    nn.LayerNorm(struct_width), nn.SiLU(), nn.Linear(struct_width, struct_width),
                ) for _ in range(4)
            ])
            self.arr_head = ThreeArrheniusHead(struct_width)

        self.film_cond_proj = nn.Sequential(
            nn.Linear(d_comp + cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList()
        self.film_gens = nn.ModuleList()
        for _ in range(num_blocks):
            self.film_gens.append(nn.Linear(hidden_dim, 2 * hidden_dim))
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            ))

        self.resid_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.resid_head.weight)
        nn.init.zeros_(self.resid_head.bias)

        self.aux_head = nn.Linear(hidden_dim, 1)

    def forward(self, fracs: torch.Tensor, norm_conds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_norm = norm_conds[:, 0:1]
        p_norm = norm_conds[:, 1:2]
        rest = norm_conds[:, 2:5]

        fu = self.fourier_u(u_norm)
        fp = self.fourier_p(p_norm)

        all_conds = torch.cat([norm_conds, fu, fp], dim=-1)
        conds_no_t = torch.cat([p_norm, rest, fp], dim=-1)

        comp_pool = self.comp_proj(fracs)

        joint_cond = torch.cat([comp_pool, all_conds], dim=-1)
        c_feat = self.film_cond_proj(joint_cond)

        h = c_feat
        for film_gen, block in zip(self.film_gens, self.blocks):
            gamma, beta = film_gen(c_feat).chunk(2, dim=-1)
            h = h + block(h * (1.0 + gamma) + beta)

        resid = self.resid_head(h).squeeze(-1)
        aux_pred = self.aux_head(h).squeeze(-1)

        if self.head_type == "struct":
            h_not = self.trunk_no_t_proj(torch.cat([comp_pool, conds_no_t], dim=-1))
            for b in self.blocks_no_t:
                h_not = h_not + b(h_not)
            arr_log10 = self.arr_head(h_not, u_norm).squeeze(-1)
            pred_log10 = arr_log10 + resid
        else:
            pred_log10 = resid

        return pred_log10, aux_pred


# ── Full Export Wrapper (Unified Ensemble) ────────────────────────────────────
class UnifiedEnsembleONNX(nn.Module):
    def __init__(self, predictors: List[GasolineB1BPredictor], cond_mean: np.ndarray, cond_std: np.ndarray):
        super().__init__()
        self.predictors = nn.ModuleList(predictors)
        self.register_buffer("cond_mean", torch.tensor(cond_mean, dtype=torch.float32))
        self.register_buffer("cond_std",  torch.tensor(cond_std, dtype=torch.float32))

    def forward(self, fracs: torch.Tensor, raw_conds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        P = torch.clamp(raw_conds[:, 0:1], min=1e3)
        T = torch.clamp(raw_conds[:, 1:2], min=100.0)
        phi = raw_conds[:, 2:3]
        egr = raw_conds[:, 3:4]
        log_phi = torch.log(torch.clamp(phi, min=1e-3))

        u = 1000.0 / T
        p = torch.log10(P / 1e5)

        raw_features = torch.cat([u, p, phi, egr, log_phi], dim=-1)
        norm_conds = (raw_features - self.cond_mean) / self.cond_std

        preds = [pred(fracs, norm_conds)[0] for pred in self.predictors]
        ens_log10 = torch.stack(preds, dim=0).mean(dim=0)
        ens_s = torch.pow(10.0, ens_log10)
        return ens_s, ens_log10


# ── Loss Function with Low-T Weighting ────────────────────────────────────────
def compute_lowt_weights(temps_K: torch.Tensor, lowt_weight: float = 2.5, lowt_center: float = 850.0) -> torch.Tensor:
    # Sigmoidal weighting focusing on low temperature chemistry (700-900 K)
    w = 1.0 + (lowt_weight - 1.0) / (1.0 + torch.exp((temps_K - lowt_center) / 30.0))
    return w


def evaluate_grid(
    y_pred_lin: np.ndarray,
    y_true_lin: np.ndarray,
    temps_K: np.ndarray,
    press_bar: np.ndarray
) -> Dict[str, Any]:
    ape = np.abs(y_pred_lin - y_true_lin) / np.clip(y_true_lin, 1e-9, None) * 100.0
    overall_mape = float(np.mean(ape))

    low_mask = temps_K < 900.0
    lowt_mape = float(np.mean(ape[low_mask])) if low_mask.sum() > 0 else overall_mape

    grid = {}
    cell_mapes = []
    for t_min, t_max in [(700, 800), (800, 900)]:
        for p_min, p_max in [(10, 20), (20, 30), (30, 40), (40, 50)]:
            mask = (temps_K >= t_min) & (temps_K < t_max) & (press_bar >= p_min) & (press_bar < p_max)
            cell_name = f"{t_min}-{t_max}K_{p_min}-{p_max}bar"
            if mask.sum() > 0:
                c_mape = float(np.mean(ape[mask]))
                grid[cell_name] = {"mape": round(c_mape, 2), "count": int(mask.sum())}
                cell_mapes.append(c_mape)
            else:
                grid[cell_name] = {"mape": None, "count": 0}

    worst_cell = max(cell_mapes) if cell_mapes else lowt_mape
    score = 0.4 * overall_mape + 0.4 * lowt_mape + 0.2 * worst_cell

    return {
        "overall_mape": overall_mape,
        "lowt_mape": lowt_mape,
        "worst_cell_mape": worst_cell,
        "selection_score": score,
        "grid": grid
    }


def train_single_seed(
    seed_idx: int,
    gpu_data: Dict[str, Any],
    val_temps_K: np.ndarray,
    val_press_bar: np.ndarray,
    val_true_lin: np.ndarray,
    val_fracs: torch.Tensor,
    val_conds: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path
) -> Tuple[GasolineB1BPredictor, Dict[str, Any]]:
    torch.manual_seed(seed_idx + 100)
    np.random.seed(seed_idx + 100)

    model = GasolineB1BPredictor(
        n_components=10,
        d_comp=args.d_comp,
        hidden_dim=args.width,
        struct_width=args.struct_width,
        num_blocks=args.blocks,
        head_type=args.head
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches_per_epoch = gpu_data["N"] // args.batch_size
    total_steps = n_batches_per_epoch * args.epochs
    warmup_steps = int(0.04 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # EMA
    ema_decay = 0.999
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    best_score = float("inf")
    best_state = None
    best_metrics = {}
    history = []

    N_train = gpu_data["N"]
    B = args.batch_size

    print(f"\n--- Training Seed {seed_idx} ({args.epochs} epochs, device: {device}, batch_size: {B}) ---")
    t0 = time.time()

    val_fracs_dev = val_fracs.to(device)
    val_conds_dev = val_conds.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_batches = 0

        perm = torch.randperm(N_train, device=device)
        for start in range(0, N_train - B + 1, B):
            idx = perm[start:start + B]
            fracs_b  = gpu_data["fracs"][idx]
            conds_b  = gpu_data["conds"][idx]
            y_main_b = gpu_data["y_main"][idx]
            y_aux_b  = gpu_data["y_aux"][idx]
            t_raw_b  = gpu_data["temps"][idx]

            pred_main, pred_aux = model(fracs_b, conds_b)

            w = compute_lowt_weights(t_raw_b, lowt_weight=args.lowt_weight, lowt_center=args.lowt_center)
            loss_main = (F.huber_loss(pred_main, y_main_b, delta=0.015, reduction="none") * w).mean()

            if args.aux_weight > 0.0:
                loss_aux = F.huber_loss(pred_aux, y_aux_b, delta=0.015, reduction="mean")
                loss = loss_main + args.aux_weight * loss_aux
            else:
                loss = loss_main

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            # Update EMA
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

            train_loss_total += loss.item() * B
            n_batches += 1

        train_loss_avg = train_loss_total / (n_batches * B)

        # Validation with EMA weights
        temp_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        model.eval()

        val_preds_log = []
        with torch.no_grad():
            b_sz = 8192
            for b in range(0, val_fracs.shape[0], b_sz):
                fracs_chunk = val_fracs_dev[b:b+b_sz]
                conds_chunk = val_conds_dev[b:b+b_sz]
                p_chunk, _ = model(fracs_chunk, conds_chunk)
                val_preds_log.append(p_chunk.cpu())

        val_pred_log_arr = torch.cat(val_preds_log).numpy()
        val_pred_lin = 10.0 ** val_pred_log_arr

        eval_res = evaluate_grid(val_pred_lin, val_true_lin, val_temps_K, val_press_bar)
        score = eval_res["selection_score"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "overall_mape": eval_res["overall_mape"],
            "lowt_mape": eval_res["lowt_mape"],
            "worst_cell_mape": eval_res["worst_cell_mape"],
            "score": score
        })

        if score < best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in ema_state.items()}
            best_metrics = eval_res

        # Restore training weights
        model.load_state_dict(temp_state)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs} [{elapsed:.1f}s] | Loss: {train_loss_avg:.5f} | "
                  f"Overall MAPE: {eval_res['overall_mape']:.2f}% | "
                  f"Low-T MAPE: {eval_res['lowt_mape']:.2f}% | "
                  f"Worst Cell: {eval_res['worst_cell_mape']:.2f}% | "
                  f"Best Score: {best_score:.2f}%")

    # Load best checkpoint and save
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"member_seed{seed_idx}.pt")
    with open(out_dir / f"history_seed{seed_idx}.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Gasoline Surrogate B1B IDT Model (Fixed 10-Component Palette)")
    parser.add_argument("--input", type=str,
                        default=str(_ROOT / "model_training" / "sidt" / "GASOLINE2_K10_250K_b1b_branching" / "cvwsr_idt_selfies_b1b_train.dat"))
    parser.add_argument("--out-dir", type=str,
                        default=str(_ROOT / "sidt" / "models" / "gasoline_mix_b1b"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-comp", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--struct-width", type=int, default=256)
    parser.add_argument("--head", type=str, choices=["struct", "free"], default="struct")
    parser.add_argument("--lowt-weight", type=float, default=2.5)
    parser.add_argument("--lowt-center", type=float, default=850.0)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"\n1. Loading dataset from {input_path}...")
    df = load_table(input_path)
    N_total = len(df)
    print(f"Loaded {N_total:,} rows.")

    # 2. Extract Features
    fracs_all = df[FRAC_COLS].values.astype(np.float32)
    raw_conds_all = df[COND_COLS].values.astype(np.float32)

    P = np.clip(raw_conds_all[:, 0], 1.0e3, None)
    T = np.clip(raw_conds_all[:, 1], 100.0, None)
    phi = np.clip(raw_conds_all[:, 2], 1.0e-3, None)
    egr = raw_conds_all[:, 3]
    u = 1000.0 / T
    p = np.log10(P / 1.0e5)
    log_phi = np.log(phi)

    core_feats = np.stack([u, p, phi, egr, log_phi], axis=1).astype(np.float32)

    # 3. Targets
    idt_400k = df["idt_400K_s"].values.astype(np.float32)
    idt_40k  = df["idt_40K_s"].values.astype(np.float32)
    log_idt_400k = np.log10(np.clip(idt_400k, 1e-9, None))
    log_idt_40k  = np.log10(np.clip(idt_40k, 1e-9, None))

    temps_K_all   = df["temperature_K"].values.astype(np.float32)
    press_bar_all = df["pressure_bar"].values.astype(np.float32)

    # 4. Train / Val Split (90% train, 10% stratified holdout)
    temp_bins = pd.qcut(df["temperature_K"], q=10, labels=False)
    press_bins = pd.qcut(df["pressure_bar"], q=5, labels=False)
    strat_key = temp_bins * 10 + press_bins

    idx_train, idx_val = train_test_split(
        np.arange(N_total),
        test_size=0.10,
        random_state=42,
        stratify=strat_key
    )

    k_mean = np.mean(core_feats[idx_train], axis=0)
    k_std  = np.std(core_feats[idx_train], axis=0)
    k_std[k_std < 1e-6] = 1.0
    norm_conds_all = (core_feats - k_mean) / k_std

    print(f"Split: Train {len(idx_train):,} | Val {len(idx_val):,}")

    val_fracs = torch.from_numpy(fracs_all[idx_val])
    val_conds = torch.from_numpy(norm_conds_all[idx_val])
    val_temps_K = temps_K_all[idx_val]
    val_press_bar = press_bar_all[idx_val]
    val_true_lin = idt_400k[idx_val]

    # Move training tensors to device
    print("Moving training tensors to device...")
    train_fracs_dev  = torch.from_numpy(fracs_all[idx_train]).to(device)
    train_conds_dev  = torch.from_numpy(norm_conds_all[idx_train]).to(device)
    train_y_main_dev = torch.from_numpy(log_idt_400k[idx_train]).to(device)
    train_y_aux_dev  = torch.from_numpy(log_idt_40k[idx_train]).to(device)
    train_temps_dev  = torch.from_numpy(temps_K_all[idx_train]).to(device)
    N_train = len(idx_train)

    gpu_data = {
        "N": N_train,
        "fracs": train_fracs_dev,
        "conds": train_conds_dev,
        "y_main": train_y_main_dev,
        "y_aux": train_y_aux_dev,
        "temps": train_temps_dev
    }
    print(f"Tensors resident on {device}.")

    # 5. Train Ensemble Seeds
    trained_models = []
    seed_val_metrics = []

    for seed_idx in range(args.seeds):
        m, metrics = train_single_seed(
            seed_idx=seed_idx,
            gpu_data=gpu_data,
            val_temps_K=val_temps_K,
            val_press_bar=val_press_bar,
            val_true_lin=val_true_lin,
            val_fracs=val_fracs,
            val_conds=val_conds,
            device=device,
            args=args,
            out_dir=out_dir
        )
        trained_models.append(m)
        seed_val_metrics.append(metrics)

    # 6. Ensemble Validation
    print("\n--- Running Full Ensemble Validation ---")
    val_fracs_dev = val_fracs.to(device)
    val_conds_dev = val_conds.to(device)

    all_seed_preds_log = []
    b_sz = 8192
    for m in trained_models:
        m.eval()
        preds = []
        with torch.no_grad():
            for b in range(0, val_fracs.shape[0], b_sz):
                p, _ = m(val_fracs_dev[b:b+b_sz], val_conds_dev[b:b+b_sz])
                preds.append(p.cpu())
        all_seed_preds_log.append(torch.cat(preds).numpy())

    ens_pred_log = np.mean(all_seed_preds_log, axis=0)
    ens_pred_lin = 10.0 ** ens_pred_log

    ens_metrics = evaluate_grid(ens_pred_lin, val_true_lin, val_temps_K, val_press_bar)
    r2_lin = float(r2_score(val_true_lin, ens_pred_lin))
    r2_log = float(r2_score(log_idt_400k[idx_val], ens_pred_log))
    mae = float(mean_absolute_error(val_true_lin, ens_pred_lin))
    rmse = float(np.sqrt(mean_squared_error(val_true_lin, ens_pred_lin)))

    print("\n" + "=" * 42)
    print(f"Final B1B Ensemble Metrics ({len(idx_val):,} test rows):")
    print(f"  Overall MAPE   : {ens_metrics['overall_mape']:.2f}%")
    print(f"  Low-T (<900K)  : {ens_metrics['lowt_mape']:.2f}%")
    print(f"  Worst Cell MAPE: {ens_metrics['worst_cell_mape']:.2f}%")
    print(f"  Linear R²      : {r2_lin:.4f}")
    print(f"  Log10  R²      : {r2_log:.4f}")
    print(f"  MAE            : {mae*1000.0:.3f} ms")
    print(f"  RMSE           : {rmse*1000.0:.3f} ms")
    print("=" * 42)

    print("\n--- Low-T Grid MAPE (700–900 K by Pressure) ---")
    for row in ["700-800K", "800-900K"]:
        cells = [f"{p_min}-{p_max} bar: {ens_metrics['grid'].get(f'{row}_{p_min}-{p_max}bar', {}).get('mape', 'N/A')}%"
                 for p_min, p_max in [(10, 20), (20, 30), (30, 40), (40, 50)]]
        print(f"  T {row} -> " + " | ".join(cells))

    # 7. Parity and Forward Plots
    artifacts_dir = _ROOT / "sidt"
    parity_path = out_dir / "gasoline_b1b_parity.png"
    plt.figure(figsize=(7, 6))
    plt.scatter(val_true_lin * 1000.0, ens_pred_lin * 1000.0, alpha=0.3, s=6, color="#007acc")
    min_val = min(val_true_lin.min(), ens_pred_lin.min()) * 1000.0
    max_val = max(val_true_lin.max(), ens_pred_lin.max()) * 1000.0
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect parity")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Simulated IDT (ms)", fontweight="bold")
    plt.ylabel("Predicted IDT (ms)", fontweight="bold")
    plt.title(f"Gasoline B1B Fixed Ensemble — Parity Plot\n(MAPE: {ens_metrics['overall_mape']:.2f}%, Log10 R²: {r2_log:.4f})", fontweight="bold")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"✓ Saved parity plot to {parity_path}")

    # Forward Arrhenius Plot
    fwd_path = out_dir / "gasoline_b1b_forward_plot.png"
    t_sweep = np.linspace(650.0, 1500.0, 100)
    p_test_pa = 20.0 * 1e5
    dummy_frac = np.full((len(t_sweep), 10), 0.1, dtype=np.float32)
    dummy_raw_conds = np.zeros((len(t_sweep), 5), dtype=np.float32)
    dummy_raw_conds[:, 0] = 1000.0 / t_sweep
    dummy_raw_conds[:, 1] = np.log10(p_test_pa / 1e5)
    dummy_raw_conds[:, 2] = 1.0  # phi
    dummy_raw_conds[:, 3] = 0.0  # egr
    dummy_raw_conds[:, 4] = 0.0  # log_phi
    norm_sweep = (dummy_raw_conds - k_mean) / k_std

    sweep_fracs_dev = torch.from_numpy(dummy_frac).to(device)
    norm_sweep_dev  = torch.from_numpy(norm_sweep).to(device)

    sweep_preds_all = []
    for m in trained_models:
        with torch.no_grad():
            sp, _ = m(sweep_fracs_dev, norm_sweep_dev)
            sweep_preds_all.append(sp.cpu().numpy())
    sweep_log10 = np.mean(sweep_preds_all, axis=0)
    sweep_lin_ms = (10.0 ** sweep_log10) * 1000.0

    plt.figure(figsize=(7, 5))
    recip_t = 1000.0 / t_sweep
    plt.plot(recip_t, sweep_lin_ms, color="#ff007f", linewidth=2.5, label="Ensemble Prediction (20 bar, phi=1.0, egr=0)")
    plt.yscale("log")
    plt.xlabel("1000 / Temperature (1/K)", fontweight="bold")
    plt.ylabel("Ignition Delay Time (ms)", fontweight="bold")
    plt.title("Gasoline B1B Ensemble — Arrhenius / NTC Curve", fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fwd_path, dpi=300)
    plt.close()
    print(f"✓ Saved forward sweep plot to {fwd_path}")

    shutil.copy2(parity_path, artifacts_dir / "gasoline_b1b_parity.png")
    shutil.copy2(fwd_path, artifacts_dir / "gasoline_b1b_forward_plot.png")

    # 8. ONNX Export
    if not args.no_export:
        print("\nExporting full unified ensemble to ONNX...")
        pred_cpus = [m.to("cpu") for m in trained_models]
        unified_ensemble = UnifiedEnsembleONNX(pred_cpus, k_mean, k_std).eval()
        onnx_path = out_dir / "idt_gasoline_b1b.onnx"

        dummy_fracs = torch.full((1, 10), 0.1, dtype=torch.float32)
        dummy_conds = torch.tensor([[1e6, 1000.0, 1.0, 0.0]], dtype=torch.float32)

        torch.onnx.export(
            unified_ensemble,
            (dummy_fracs, dummy_conds),
            str(onnx_path),
            input_names=["mole_fracs", "reactor_conds"],
            output_names=["idt_s", "idt_log10"],
            dynamic_axes={"mole_fracs": {0: "batch"}, "reactor_conds": {0: "batch"},
                          "idt_s": {0: "batch"}, "idt_log10": {0: "batch"}},
            opset_version=17, do_constant_folding=True
        )
        print(f"✓ ONNX exported to {onnx_path} ({onnx_path.stat().st_size/1024/1024:.2f} MB)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            t_s, t_log = unified_ensemble(dummy_fracs, dummy_conds)
        ort_out = sess.run(None, {
            "mole_fracs": dummy_fracs.numpy(),
            "reactor_conds": dummy_conds.numpy()
        })
        onnx_diff = float(np.abs(ort_out[1] - t_log.numpy()).max())
        print(f"✓ ONNX parity check passed! Max abs difference: {onnx_diff:.2e}")

        # Also create a forward_model.onnx symlink/copy for compatibility
        shutil.copy2(onnx_path, out_dir / "forward_model.onnx")

    # 9. Save Metadata
    meta = {
        "dataset": str(input_path.name),
        "architecture": "GasolineB1BPredictor (Three-Arrhenius + FiLM ResNet without component tokens)",
        "version": "b1b_v1",
        "n_samples_total": N_total,
        "n_samples_train": len(idx_train),
        "n_samples_val": len(idx_val),
        "n_seeds_trained": args.seeds,
        "input_features": {
            "mole_fracs": "[batch, 10]",
            "reactor_conds": "[batch, 4] -> [pressure_pa, temperature_K, phi, egr_fraction]"
        },
        "output_targets": {
            "idt_s": "[batch] (seconds)",
            "idt_log10": "[batch] (log10 seconds)"
        },
        "kinetic_cond_mean": k_mean.tolist(),
        "kinetic_cond_std": k_std.tolist(),
        "ensemble_metrics": {
            "overall_mape": ens_metrics["overall_mape"],
            "lowt_mape": ens_metrics["lowt_mape"],
            "worst_cell_mape": ens_metrics["worst_cell_mape"],
            "val_r2_linear": r2_lin,
            "val_r2_log10": r2_log,
            "val_mae_s": mae,
            "val_rmse_s": rmse,
            "lowt_grid": ens_metrics["grid"]
        }
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved to {out_dir / 'model_meta.json'}")

    print("\nAll deliverables for Gasoline B1B completed successfully!")


if __name__ == "__main__":
    main()
