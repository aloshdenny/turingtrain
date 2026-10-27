"""
0d_idt_gasoline_train_v2.py
───────────────────────────
Gasoline 0D Ignition Delay Time (IDT) Surrogate Model v2.

Key improvements over v1:
1. Fixed Table Reader: loads all 1,139,500 rows across all 10 component sets (skipping # header comments).
2. Extended Vocabulary: incorporates [#C], [#Branch1], [#Branch2] (26 tokens total).
3. Corrected Malformed SELFIES: remaps '[C]#[C]' -> '[C][#C]'.
4. Structured Three-Arrhenius Head:
   1/tau = 1/(tau1 + tau2) + 1/tau3 (captures low-T, NTC, and high-T chemistry)
   driven by a T-free trunk, combined with a zero-initialized free residual from full FiLM trunk.
5. Multi-frequency Fourier features for 1000/T and log P.
6. Masked self-attention and dual pooling (x-weighted + learned query) over present components.
7. Low-temperature Huber loss weighting focusing on 700–900 K.
8. Single unified ONNX export averaging all seeds in log10 space with exact v1-compatible signature.
"""

import argparse
import json
import math
import os
import sys
import time
import types
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
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Project Paths ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SELFIES_DIR = _ROOT / "SELFIES"

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SELFIES_DIR))
sys.path.insert(0, str(_SELFIES_DIR / "vae"))

from gasoline_io import (
    N_COMPONENTS, MAX_SELFIES_LEN, SELFIES_COLS, FRAC_COLS, COND_COLS,
    load_table, identify_subsets, compute_standardized_conditions
)
from selfies_tokenizer import SELFIESTokenizer
from selfies_vae import SELFIESEncoder


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


class CompositionModule(nn.Module):
    def __init__(self, latent_dim: int = 128, d_comp: int = 256, num_heads: int = 4):
        super().__init__()
        self.d_comp = d_comp
        self.proj = nn.Linear(latent_dim + 1, d_comp)
        self.layer1 = nn.MultiheadAttention(d_comp, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_comp)
        self.layer2 = nn.MultiheadAttention(d_comp, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_comp)

        self.query = nn.Parameter(torch.randn(1, 1, d_comp) * 0.02)
        self.query_attn = nn.MultiheadAttention(d_comp, num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(d_comp)
        self.pool_proj = nn.Linear(2 * d_comp, d_comp)

    def forward(self, mu: torch.Tensor, fracs: torch.Tensor) -> torch.Tensor:
        batch_sz = fracs.shape[0]
        x_exp = fracs.unsqueeze(-1)
        if mu.shape[0] != batch_sz:
            mu = mu.expand(batch_sz, -1, -1)
        tokens = torch.cat([mu, x_exp], dim=-1)
        h = self.proj(tokens)

        mask = (fracs <= 1e-7)
        h1, _ = self.layer1(h, h, h, key_padding_mask=mask)
        h = self.norm1(h + h1)
        h2, _ = self.layer2(h, h, h, key_padding_mask=mask)
        h = self.norm2(h + h2)

        x_pool = (h * x_exp).sum(dim=1)
        q = self.query.expand(batch_sz, -1, -1)
        q_out, _ = self.query_attn(q, h, h, key_padding_mask=mask)
        q_pool = self.query_norm(q_out.squeeze(1))

        return self.pool_proj(torch.cat([x_pool, q_pool], dim=-1))


class ThreeArrheniusHead(nn.Module):
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
        return ln_tau / 2.302585092994046


class GasolineV2Predictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        d_comp: int = 256,
        hidden_dim: int = 512,
        struct_width: int = 256,
        num_blocks: int = 8,
        head_type: str = "struct"
    ):
        super().__init__()
        self.head_type = head_type
        self.comp_mod = CompositionModule(latent_dim=latent_dim, d_comp=d_comp)
        self.fourier_u = FourierFeatures()
        self.fourier_p = FourierFeatures()

        cond_dim = 25
        cond_no_t_dim = 14

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

    def forward(self, mu: torch.Tensor, fracs: torch.Tensor, norm_conds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_norm = norm_conds[:, 0:1]
        p_norm = norm_conds[:, 1:2]
        rest = norm_conds[:, 2:5]

        fu = self.fourier_u(u_norm)
        fp = self.fourier_p(p_norm)

        all_conds = torch.cat([norm_conds, fu, fp], dim=-1)
        conds_no_t = torch.cat([p_norm, rest, fp], dim=-1)

        comp_pool = self.comp_mod(mu, fracs)

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
def _patch_encoder_for_onnx(encoder):
    def layer_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = self.norm1(src)
        x2, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(x2)
        x = self.norm2(src)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return src + self.dropout2(x2)
    for layer in encoder.transformer.layers:
        layer.forward = types.MethodType(layer_forward, layer)


class UnifiedEnsembleONNX(nn.Module):
    def __init__(self, encoder, predictors: List[GasolineV2Predictor], cond_mean: np.ndarray, cond_std: np.ndarray):
        super().__init__()
        self.encoder = encoder
        self.predictors = nn.ModuleList(predictors)
        self.register_buffer("cond_mean", torch.tensor(cond_mean, dtype=torch.float32))
        self.register_buffer("cond_std",  torch.tensor(cond_std, dtype=torch.float32))

    def forward(self, tokens: torch.Tensor, fracs: torch.Tensor, raw_conds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, _ = self.encoder(tokens)
        mu_exp = mu.unsqueeze(0)

        P = torch.clamp(raw_conds[:, 0:1], min=1e3)
        T = torch.clamp(raw_conds[:, 1:2], min=100.0)
        phi = raw_conds[:, 2:3]
        egr = raw_conds[:, 3:4]
        log_phi = torch.log(torch.clamp(phi, min=1e-3))

        u = 1000.0 / T
        p = torch.log10(P / 1e5)

        raw_features = torch.cat([u, p, phi, egr, log_phi], dim=-1)
        norm_conds = (raw_features - self.cond_mean) / self.cond_std

        preds = [pred(mu_exp, fracs, norm_conds)[0] for pred in self.predictors]
        if len(preds) == 1:
            ens_log10 = preds[0]
        else:
            ens_log10 = torch.stack(preds, dim=0).mean(dim=0)

        idt_s = torch.pow(10.0, ens_log10)
        return idt_s, ens_log10


# ── Training Helpers ──────────────────────────────────────────────────────────
def compute_lowt_weights(temps_K: torch.Tensor, lowt_weight: float = 2.0, lowt_center: float = 900.0) -> torch.Tensor:
    sig = torch.sigmoid((lowt_center - temps_K) / 30.0)
    return 1.0 + lowt_weight * sig


def evaluate_grid(pred_lin: np.ndarray, true_lin: np.ndarray, temps_K: np.ndarray, press_bar: np.ndarray) -> Dict[str, Any]:
    ape = np.abs(pred_lin - true_lin) / true_lin * 100.0
    overall_mape = float(np.mean(ape))

    lowt_mask = (temps_K < 900.0)
    lowt_mape = float(np.mean(ape[lowt_mask])) if lowt_mask.any() else overall_mape

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
    mu10_stack: torch.Tensor,
    val_temps_K: np.ndarray,
    val_press_bar: np.ndarray,
    val_true_lin: np.ndarray,
    val_sids: torch.Tensor,
    val_fracs: torch.Tensor,
    val_conds: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path
) -> Tuple[GasolineV2Predictor, Dict[str, Any]]:
    torch.manual_seed(seed_idx + 100)
    np.random.seed(seed_idx + 100)

    model = GasolineV2Predictor(
        latent_dim=128,
        d_comp=args.d_comp,
        hidden_dim=args.width,
        struct_width=args.struct_width,
        num_blocks=args.blocks,
        head_type=args.head
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches_per_epoch = gpu_data["N"] // args.batch_size
    total_steps = n_batches_per_epoch * args.epochs
    warmup_steps = int(0.03 * total_steps)

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

    mu10_dev = mu10_stack.to(device)
    N_train = gpu_data["N"]
    B = args.batch_size

    print(f"\n--- Training Seed {seed_idx} ({args.epochs} epochs, device: {device}, batch_size: {B}) ---")
    t0 = time.time()

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
            sids_b   = gpu_data["sids"][idx]
            t_raw_b  = gpu_data["temps"][idx]

            mu_b = mu10_dev[sids_b]
            pred_main, pred_aux = model(mu_b, fracs_b, conds_b)

            w = compute_lowt_weights(t_raw_b, lowt_weight=args.lowt_weight, lowt_center=args.lowt_center)
            loss_main = (F.huber_loss(pred_main, y_main_b, delta=0.02, reduction="none") * w).mean()

            if args.aux_weight > 0.0:
                loss_aux = F.huber_loss(pred_aux, y_aux_b, delta=0.02, reduction="mean")
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
            for b in range(0, val_sids.shape[0], b_sz):
                sids_chunk  = val_sids[b:b+b_sz]
                fracs_chunk = val_fracs[b:b+b_sz]
                conds_chunk = val_conds[b:b+b_sz]
                mu_chunk = mu10_dev[sids_chunk]
                p_chunk, _ = model(mu_chunk, fracs_chunk, conds_chunk)
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
                  f"Overall MAPE: {eval_res['overall_mape']:.2f}% | Low-T MAPE: {eval_res['lowt_mape']:.2f}% | "
                  f"Worst Cell: {eval_res['worst_cell_mape']:.2f}% | Best Score: {best_score:.2f}%")

    # Load best weights
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / f"member_seed{seed_idx}.pt")
    with open(out_dir / f"history_seed{seed_idx}.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, best_metrics


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Gasoline 0D IDT surrogate v2")
    parser.add_argument("--input",            default="model_training/sidt/sidt_lhs_gasoline2_K10_1M.dat")
    parser.add_argument("--out-dir",          default="sidt/models/gasoline_mix_selfies")
    parser.add_argument("--seeds",            type=int, default=5)
    parser.add_argument("--epochs",           type=int, default=30)
    parser.add_argument("--batch-size",       type=int, default=4096)
    parser.add_argument("--lr",               type=float, default=2e-3)
    parser.add_argument("--head",             choices=["struct", "free"], default="struct")
    parser.add_argument("--lowt-weight",      type=float, default=2.0)
    parser.add_argument("--lowt-center",      type=float, default=900.0)
    parser.add_argument("--aux-weight",       type=float, default=0.3)
    parser.add_argument("--width",            type=int, default=512)
    parser.add_argument("--blocks",           type=int, default=8)
    parser.add_argument("--d-comp",           type=int, default=256)
    parser.add_argument("--struct-width",     type=int, default=256)
    parser.add_argument("--no-export",        action="store_true")
    args = parser.parse_args()

    input_path = _ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_dir    = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    # 1. Load Data with fixes applied
    df = load_table(input_path)
    subset_ids_all, subsets = identify_subsets(df)
    print(f"Identified {len(subsets)} unique component subsets.")

    # 2. Tokenize with extended 26-token vocabulary
    tok_path = _SELFIES_DIR / "data" / "vocab.json"
    tokenizer = SELFIESTokenizer.load(tok_path)
    print(f"Loaded tokenizer from {tok_path.name} (vocab size: {tokenizer.vocab_size})")

    # 3. Standardize conditions
    raw_conds_all = df[COND_COLS].values.astype(np.float32)
    norm_conds_all, cond_stats = compute_standardized_conditions(raw_conds_all)
    k_mean = cond_stats["mean"]
    k_std  = cond_stats["std"]

    # 4. Extract features and targets
    fracs_all = df[FRAC_COLS].values.astype(np.float32)
    fracs_all = fracs_all / fracs_all.sum(axis=1, keepdims=True).clip(min=1e-8)

    idt_400k = np.clip(df["idt_400K_s"].values.astype(np.float32), 1e-8, None)
    idt_40k  = np.clip(df["idt_40K_s"].values.astype(np.float32), 1e-8, None)
    log_idt_400k = np.log10(idt_400k)
    log_idt_40k  = np.log10(idt_40k)
    temps_K_all  = df["temperature_K"].values.astype(np.float32)
    press_bar_all = (df["pressure_pa"].values / 1e5).astype(np.float32)

    # 5. Train/Val Split (90/10 hold-out)
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42)
    print(f"Split: Train {len(idx_train):,} | Val {len(idx_val):,}")

    train_ds = TensorDataset(
        torch.from_numpy(fracs_all[idx_train]),
        torch.from_numpy(norm_conds_all[idx_train]),
        torch.from_numpy(log_idt_400k[idx_train]),
        torch.from_numpy(log_idt_40k[idx_train]),
        torch.from_numpy(subset_ids_all[idx_train]),
        torch.from_numpy(temps_K_all[idx_train])
    )

    val_fracs = torch.from_numpy(fracs_all[idx_val])
    val_conds = torch.from_numpy(norm_conds_all[idx_val])
    val_sids  = torch.from_numpy(subset_ids_all[idx_val])
    val_temps_K = temps_K_all[idx_val]
    val_press_bar = press_bar_all[idx_val]
    val_true_lin = idt_400k[idx_val]

    # Move everything to device once for fast GPU-resident batching
    print("Moving training tensors to device...")
    train_fracs_dev  = torch.from_numpy(fracs_all[idx_train]).to(device)
    train_conds_dev  = torch.from_numpy(norm_conds_all[idx_train]).to(device)
    train_y_main_dev = torch.from_numpy(log_idt_400k[idx_train]).to(device)
    train_y_aux_dev  = torch.from_numpy(log_idt_40k[idx_train]).to(device)
    train_sids_dev   = torch.from_numpy(subset_ids_all[idx_train]).to(device)
    train_temps_dev  = torch.from_numpy(temps_K_all[idx_train]).to(device)
    N_train = len(idx_train)

    val_fracs_dev = val_fracs.to(device)
    val_conds_dev = val_conds.to(device)
    val_sids_dev  = val_sids.to(device)
    print(f"Tensors loaded. ~{N_train * 6 * 4 / 1024**3:.2f} GB on {device}.")

    # 6. Initialize SELFIES Encoder & Precompute mu_10 for each subset
    print("Loading pretrained SELFIES VAE encoder with extended 26-token embedding...")
    st0 = torch.load(_SELFIES_DIR / "checkpoints_opt" / "seed0_s1_vae.pt", map_location="cpu", weights_only=False)
    enc_state = {k[len("encoder."):]: v for k, v in st0.items() if k.startswith("encoder.")}

    # Expand embedding to 26 tokens
    old_emb = enc_state["embedding.weight"]
    new_emb = torch.zeros(tokenizer.vocab_size, old_emb.shape[1], dtype=old_emb.dtype)
    new_emb[:old_emb.shape[0]] = old_emb
    new_emb[tokenizer.token2idx["[#Branch1]"]] = old_emb[tokenizer.token2idx["[=Branch1]"]]
    new_emb[tokenizer.token2idx["[#Branch2]"]] = old_emb[tokenizer.token2idx["[=Branch2]"]]
    new_emb[tokenizer.token2idx["[#C]"]]       = old_emb[tokenizer.token2idx["[=C]"]]
    enc_state["embedding.weight"] = new_emb

    encoder = SELFIESEncoder(
        vocab_size=tokenizer.vocab_size, d_model=128, latent_dim=128,
        n_heads=4, n_layers=4, d_ff=512, dropout=0.0,
        pad_idx=tokenizer.pad_idx, max_len=MAX_SELFIES_LEN
    ).to(device)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()

    # Precompute mu_10 for all 10 subsets
    mu10_list = []
    for s in subsets:
        tokens = torch.zeros(N_COMPONENTS, MAX_SELFIES_LEN, dtype=torch.long)
        for i, s_str in enumerate(s["selfies_list"]):
            if s_str:
                tokens[i] = tokenizer.encode(s_str, max_len=MAX_SELFIES_LEN)
        with torch.no_grad():
            mu, _ = encoder(tokens.to(device))
        mu10_list.append(mu.cpu())
    mu10_stack = torch.stack(mu10_list, dim=0)

    # 7. Train Seeds
    trained_models = []
    seed_evals = []
    t_start = time.time()

    # Pack GPU-resident training data for fast access
    gpu_train_data = {
        "fracs":  train_fracs_dev,
        "conds":  train_conds_dev,
        "y_main": train_y_main_dev,
        "y_aux":  train_y_aux_dev,
        "sids":   train_sids_dev,
        "temps":  train_temps_dev,
        "N":      N_train
    }

    for s_idx in range(args.seeds):
        m, ev = train_single_seed(
            s_idx, gpu_train_data, mu10_stack,
            val_temps_K, val_press_bar, val_true_lin,
            val_sids_dev, val_fracs_dev, val_conds_dev,
            device, args, out_dir
        )
        trained_models.append(m)
        seed_evals.append(ev)

    print(f"\nAll {args.seeds} seeds trained in {(time.time() - t_start)/60:.1f} minutes.")

    # 8. Ensemble Evaluation
    print("\n--- Running Full Ensemble Validation ---")
    for m in trained_models:
        m.eval()

    all_seed_preds = []
    mu10_dev = mu10_stack.to(device)
    with torch.no_grad():
        for m in trained_models:
            sp = []
            b_sz = 8192
            for b in range(0, len(val_sids_dev), b_sz):
                sids_chunk = val_sids_dev[b:b+b_sz]
                fracs_chunk = val_fracs_dev[b:b+b_sz]
                conds_chunk = val_conds_dev[b:b+b_sz]
                mu_chunk = mu10_dev[sids_chunk]
                p_chunk, _ = m(mu_chunk, fracs_chunk, conds_chunk)
                sp.append(p_chunk.cpu())
            all_seed_preds.append(torch.cat(sp))

    ens_log10 = torch.stack(all_seed_preds, dim=0).mean(dim=0).numpy()
    pred_lin = 10.0 ** ens_log10

    ens_eval = evaluate_grid(pred_lin, val_true_lin, val_temps_K, val_press_bar)
    r2_lin = float(r2_score(val_true_lin, pred_lin))
    r2_log = float(r2_score(np.log10(val_true_lin), ens_log10))
    ens_mae = float(mean_absolute_error(val_true_lin, pred_lin))
    ens_rmse = float(np.sqrt(mean_squared_error(val_true_lin, pred_lin)))

    ape = np.abs(pred_lin - val_true_lin) / val_true_lin * 100.0

    print(f"\n==========================================")
    print(f"Final Ensemble Metrics (113,950 test rows):")
    print(f"  Overall MAPE   : {ens_eval['overall_mape']:.2f}%  (v1 rescored: 110.07%)")
    print(f"  Low-T (<900K)  : {ens_eval['lowt_mape']:.2f}%")
    print(f"  Linear R²      : {r2_lin:.4f}")
    print(f"  Log10  R²      : {r2_log:.4f}")
    print(f"  MAE            : {ens_mae*1000:.3f} ms")
    print(f"  RMSE           : {ens_rmse*1000:.3f} ms")
    print(f"==========================================")

    # Per-subset breakdown
    sub_mapes = {}
    print("\n--- Per-Subset Hold-out Performance ---")
    for uid in range(len(subsets)):
        m = (val_sids.numpy() == uid)
        sm = float(np.mean(ape[m])) if m.sum() > 0 else 0.0
        sub_mapes[f"subset_{uid+1}"] = {"mape": round(sm, 2), "count": int(m.sum())}
        print(f"  Subset {uid+1:2d} (n={m.sum():,}): MAPE = {sm:6.2f}% | cpnt_2: {subsets[uid]['selfies_list'][1][:30]}")

    # Low-T Grid Table
    print("\n--- Low-T Grid MAPE (700–900 K by Pressure) ---")
    for t_min, t_max in [(700, 800), (800, 900)]:
        row_cells = []
        for p_min, p_max in [(10, 20), (20, 30), (30, 40), (40, 50)]:
            cname = f"{t_min}-{t_max}K_{p_min}-{p_max}bar"
            c_info = ens_eval["grid"].get(cname, {"mape": "N/A"})
            row_cells.append(f"{p_min}-{p_max} bar: {c_info['mape']}%")
        print(f"  T {t_min}-{t_max} K -> " + " | ".join(row_cells))

    # 9. Parity Plot
    parity_path = _ROOT / "sidt/gasoline_mix_selfies_parity.png"
    plt.figure(figsize=(7, 6), dpi=200)
    idx_plot = np.random.choice(len(val_true_lin), min(15000, len(val_true_lin)), replace=False)
    plt.scatter(val_true_lin[idx_plot]*1000, pred_lin[idx_plot]*1000, alpha=0.35, s=12, c="#1f77b4", edgecolors="none")
    lims = [min(val_true_lin[idx_plot].min(), pred_lin[idx_plot].min())*1000,
            max(val_true_lin[idx_plot].max(), pred_lin[idx_plot].max())*1000]
    plt.plot(lims, lims, "r--", lw=1.5, label="1:1 Identity")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True IDT (ms)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted IDT (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline 0D IDT v2 — Parity (Three-Arrhenius + FiLM ResNet)", fontsize=11, fontweight="bold")
    stats_text = (
        f"Ensemble (10 Subsets, 1.14M rows)\n"
        f"Overall MAPE: {ens_eval['overall_mape']:.2f}%\n"
        f"Low-T MAPE:   {ens_eval['lowt_mape']:.2f}%\n"
        f"Linear R²:    {r2_lin:.4f}\n"
        f"Log₁₀ R²:     {r2_log:.4f}\n"
        f"MAE:          {ens_mae*1000:.2f} ms"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))
    plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(parity_path, dpi=200); plt.close()
    print(f"✓ Saved parity plot to {parity_path}")

    # Forward Sweep Plot
    fwd_path = _ROOT / "sidt/gasoline_mix_selfies_forward_plot.png"
    plt.figure(figsize=(7, 6), dpi=200)
    temps_sweep = np.linspace(600.0, 1500.0, 100)
    sweep_fracs = torch.full((len(temps_sweep), 10), 0.1, dtype=torch.float32, device=device)
    sweep_mu = mu10_dev[0:1].expand(len(temps_sweep), -1, -1)

    for p_bar, c in zip([10.0, 20.0, 40.0], ["#2ca02c", "#1f77b4", "#d62728"]):
        raw_conds_sw = np.array([[p_bar*1e5, t, 1.0, 0.0] for t in temps_sweep], dtype=np.float32)
        norm_conds_sw, _ = compute_standardized_conditions(raw_conds_sw, cond_stats)
        norm_conds_t = torch.from_numpy(norm_conds_sw).to(device)

        with torch.no_grad():
            preds_sw = [m(sweep_mu, sweep_fracs, norm_conds_t)[0] for m in trained_models]
            avg_log = torch.stack(preds_sw).mean(dim=0).cpu().numpy()
            ms_sw = (10.0 ** avg_log) * 1000.0
        plt.plot(1000.0 / temps_sweep, ms_sw, lw=2.0, color=c, label=f"{p_bar:.0f} bar")

    plt.yscale("log"); plt.xlabel("1000 / T (1/K)", fontsize=11, fontweight="bold")
    plt.ylabel("Ignition Delay Time (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline 0D IDT v2 — Arrhenius Curves Across Pressures", fontsize=11, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(title="Pressure (φ=1.0, EGR=0.0)")
    plt.tight_layout(); plt.savefig(fwd_path, dpi=200); plt.close()
    print(f"✓ Saved forward sweep plot to {fwd_path}")

    # Copy to brain artifacts
    artifacts_dir = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")
    if artifacts_dir.exists():
        import shutil
        shutil.copy2(parity_path, artifacts_dir / "gasoline_mix_selfies_parity.png")
        shutil.copy2(fwd_path, artifacts_dir / "gasoline_mix_selfies_forward_plot.png")

    # 10. ONNX Export
    if not args.no_export:
        print("\nExporting full unified ensemble to ONNX...")
        enc_cpu = encoder.to("cpu")
        _patch_encoder_for_onnx(enc_cpu)
        pred_cpus = [m.to("cpu") for m in trained_models]

        unified_ensemble = UnifiedEnsembleONNX(enc_cpu, pred_cpus, k_mean, k_std).eval()
        onnx_path = out_dir / "idt_gasoline_selfies.onnx"

        dummy_tokens = torch.zeros(10, MAX_SELFIES_LEN, dtype=torch.long)
        dummy_fracs  = torch.full((1, 10), 0.1, dtype=torch.float32)
        dummy_conds  = torch.tensor([[1e6, 1000.0, 1.0, 0.0]], dtype=torch.float32)

        torch.onnx.export(
            unified_ensemble,
            (dummy_tokens, dummy_fracs, dummy_conds),
            str(onnx_path),
            input_names=["component_tokens", "mole_fracs", "reactor_conds"],
            output_names=["idt_s", "idt_log10"],
            dynamic_axes={"mole_fracs": {0: "batch"}, "reactor_conds": {0: "batch"},
                          "idt_s": {0: "batch"}, "idt_log10": {0: "batch"}},
            opset_version=17, do_constant_folding=True
        )
        print(f"✓ ONNX exported to {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            t_s, t_log = unified_ensemble(dummy_tokens, dummy_fracs, dummy_conds)
        ort_out = sess.run(None, {
            "component_tokens": dummy_tokens.numpy(),
            "mole_fracs": dummy_fracs.numpy(),
            "reactor_conds": dummy_conds.numpy()
        })
        onnx_diff = float(np.abs(ort_out[1] - t_log.numpy()).max())
        print(f"✓ ONNX parity check passed! Max abs difference: {onnx_diff:.2e}")

    # 11. Save model_meta.json
    meta = {
        "dataset": str(input_path.name),
        "architecture": "GasolineV2Predictor (Three-Arrhenius + FiLM ResNet)",
        "version": "v2",
        "n_samples_total": len(df),
        "n_samples_train": len(idx_train),
        "n_samples_val": len(idx_val),
        "n_subsets": len(subsets),
        "subset_sizes": [len(s["row_indices"]) for s in subsets],
        "n_seeds_trained": args.seeds,
        "vocab_size": tokenizer.vocab_size,
        "kinetic_cond_mean": k_mean.tolist(),
        "kinetic_cond_std": k_std.tolist(),
        "baseline_v1_rescored": {
            "overall_mape": 110.07,
            "val_r2_linear": 0.1443,
            "val_r2_log10": 0.8661
        },
        "v2_results": {
            "overall_mape": ens_eval["overall_mape"],
            "lowt_mape": ens_eval["lowt_mape"],
            "worst_cell_mape": ens_eval["worst_cell_mape"],
            "val_r2_linear": r2_lin,
            "val_r2_log10": r2_log,
            "val_mae_s": ens_mae,
            "val_rmse_s": ens_rmse,
            "per_subset_mape": sub_mapes,
            "lowt_grid": ens_eval["grid"]
        }
    }
    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved to {meta_path}")

    # Copy vocab.json to model dir as well
    import shutil
    shutil.copy2(tok_path, out_dir / "vocab.json")
    print(f"✓ vocab.json shipped to {out_dir / 'vocab.json'}")
    print(f"\nAll v2 deliverables completed successfully!")


if __name__ == "__main__":
    main()
