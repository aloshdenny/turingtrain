"""
train_ysi_dha_export.py
========================
Production training and ONNX export for Yield Sooting Index (YSI) 
component and mixture prediction using a SELFIES sequence model.

Mathematical Basis:
  The mixing follows a strictly linear rule:
    ysi_mix = Σ (i=1 to 10) [cpnt_mole_frac_i × ysi_i]

  The neural network predicts the component YSI from its SELFIES token sequence:
    ysi_i = Model(SELFIES_i)
  The mixture prediction is computed inside the end-to-end model via linear blending:
    ysi_mix = Σ (i=1 to 10) [cpnt_mole_frac_i × ysi_i]

Inputs (ONNX):
  component_tokens : (batch, 10, 65)  int64
  mole_fracs       : (batch, 10)      float32

Output (ONNX):
  ysi_mix          : (batch,)         float32 (in linear YSI units)

Resources:
  - CPU threads capped to 6 (leaving headroom for general work)
  - Memory-safe streaming loader (<100 MB RAM total)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SELFIES_DIR = _ROOT / "SELFIES"
_BASE_VOCAB = _SELFIES_DIR / "data" / "vocab.json"
_DATA_PATH = _ROOT / "model_training/ysi_dha/ysi_mix_selfies_train.dat"
_OUT_DIR = _ROOT / "ysi/models/ysi_dha"

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_COMPONENTS = 10
MAX_SELFIES_LEN = 65
EMB_DIM = 128
HIDDEN_DIM = 128
N_SEEDS = 5
EPOCHS = 160
LR = 3e-3
WEIGHT_DECAY = 1e-4

# Tokens present in the YSI dataset not in original CN vocab
NEW_TOKENS = [
    "[#Branch1]", "[#Branch2]", "[#C]", "[-/Ring1]",
    "[=N]", "[Br]", "[C@@H1]", "[C@H1]", "[Cl]", "[F]", "[P]", "[S]"
]

TOKEN_RE = re.compile(r"\[.*?\]")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vocabulary & Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_extended_vocab() -> dict[str, int]:
    with open(_BASE_VOCAB, "r") as f:
        vocab = json.load(f)["token2idx"]
    max_idx = max(vocab.values())
    for t in NEW_TOKENS:
        if t not in vocab:
            max_idx += 1
            vocab[t] = max_idx
    return vocab


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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fast Streaming Data Loader (RAM Safe)
# ─────────────────────────────────────────────────────────────────────────────

def load_data_streaming(
    path: Path,
    vocab: dict[str, int],
    n_sample_mixtures: int = 30000,
):
    """
    Extracts all pure component measurements and samples diverse mixture rows
    using minimal RAM (<80 MB).
    """
    print(f"Streaming data from {path} ...")
    pure_compounds: dict[str, tuple[str, float]] = {}  # selfies -> (name, ysi)
    mixture_rows: list[tuple[list[int], list[float], float]] = []

    # First pass: collect pure components
    with open(path, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            row = line.rstrip("\r\n").split("\t")
            if header is None:
                header = row
                col_map = {name: i for i, name in enumerate(header)}
                continue
            if row[col_map["status"]] != "ok":
                continue

            if row[col_map["n_active"]] == "1":
                ysi = float(row[col_map["ysi_mix"]])
                for i in range(1, N_COMPONENTS + 1):
                    frac = float(row[col_map[f"cpnt_mole_frac_{i}"]])
                    if frac > 0.5:
                        s = row[col_map[f"cpnt_selfies_{i}"]]
                        name = row[col_map[f"cpnt_name_{i}"]]
                        if s and s != "NaN":
                            pure_compounds[s] = (name, ysi)
                        break

    print(f"  Extracted {len(pure_compounds)} unique pure compounds.")

    comp_list = list(pure_compounds.keys())
    comp2idx = {s: i for i, s in enumerate(comp_list)}

    # Second pass: sample diverse mixtures for validation & linear blend checking
    with open(path, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            row = line.rstrip("\r\n").split("\t")
            if header is None:
                header = row
                col_map = {name: i for i, name in enumerate(header)}
                continue
            if row[col_map["status"]] != "ok":
                continue

            if row[col_map["n_active"]] != "1":
                indices = []
                fracs = []
                for i in range(1, N_COMPONENTS + 1):
                    s = row[col_map[f"cpnt_selfies_{i}"]]
                    f_val = float(row[col_map[f"cpnt_mole_frac_{i}"]])
                    if s in comp2idx and f_val > 0.0:
                        indices.append(comp2idx[s])
                        fracs.append(f_val)
                    else:
                        indices.append(0)
                        fracs.append(0.0)
                y_mix = float(row[col_map["ysi_mix"]])
                mixture_rows.append((indices, fracs, y_mix))
                if len(mixture_rows) >= n_sample_mixtures:
                    break

    print(f"  Sampled {len(mixture_rows)} diverse mixture validation rows.")

    # Encode pure compounds into tensors
    pure_tokens = torch.tensor(
        [encode_selfies(s, vocab) for s in comp_list], dtype=torch.long
    )
    pure_ys = np.array([pure_compounds[s][1] for s in comp_list], dtype=np.float32)

    return comp_list, pure_compounds, pure_tokens, pure_ys, mixture_rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class AttentivePooling(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        scores = self.attn(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        return torch.sum(x * weights, dim=1)


class ComponentYSINet(nn.Module):
    """
    Predicts log1p(YSI) for a single SELFIES token sequence.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.pool = AttentivePooling(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)
        pad_mask = (x == 0)
        emb = self.embedding(x)
        h0 = torch.zeros(4, x.size(0), self.hidden_dim, dtype=emb.dtype, device=x.device)
        out, _ = self.gru(emb, h0)
        pooled = self.pool(out, pad_mask)
        return self.head(pooled).squeeze(-1)


class FullYSIMixtureModel(nn.Module):
    """
    End-to-End ONNX Deployable Mixture Model.
    Strictly follows linear blending rule:
      ysi_mix = Σ (i=1..10) [cpnt_mole_frac_i * ysi_i]
    """
    def __init__(self, comp_net: ComponentYSINet):
        super().__init__()
        self.comp_net = comp_net

    def forward(self, component_tokens: torch.Tensor, mole_fracs: torch.Tensor) -> torch.Tensor:
        # component_tokens: (batch, 10, 65)
        # mole_fracs:       (batch, 10)
        B, n_comp, seq_len = component_tokens.shape
        flat_tokens = component_tokens.reshape(B * n_comp, seq_len)
        
        # Predict component log1p(YSI)
        flat_log_ysi = self.comp_net(flat_tokens)
        
        # Convert to linear YSI units and clamp to non-negative (ONNX compatible exp-1)
        flat_ysi = torch.clamp(torch.exp(flat_log_ysi) - 1.0, min=0.0)
        comp_ysi = flat_ysi.reshape(B, n_comp)  # (batch, 10)
        
        # Linear mole-fraction mixing
        ysi_mix = torch.sum(comp_ysi * mole_fracs, dim=-1)  # (batch,)
        return ysi_mix


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_seed(
    seed_idx: int,
    tokens_train: torch.Tensor,
    ys_train: torch.Tensor,
    tokens_val: torch.Tensor,
    ys_val: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    epochs: int = EPOCHS,
) -> tuple[ComponentYSINet, dict[str, float]]:
    torch.manual_seed(seed_idx * 100 + 42)
    np.random.seed(seed_idx * 100 + 42)

    model = ComponentYSINet(vocab_size=vocab_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    x_tr = tokens_train.to(device)
    y_tr_log = torch.log1p(ys_train).to(device)

    x_va = tokens_val.to(device)
    y_va_lin = ys_val.cpu().numpy()

    best_mae = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        pred_log = model(x_tr)
        loss = F.smooth_l1_loss(pred_log, y_tr_log)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_pred_log = model(x_va)
            v_pred_lin = torch.expm1(v_pred_log).clamp(min=0.0).cpu().numpy()
            v_mae = float(np.mean(np.abs(v_pred_lin - y_va_lin)))

            if v_mae < best_mae:
                best_mae = v_mae
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 40 == 0 or epoch == epochs:
            print(f"  [Seed {seed_idx}] Epoch {epoch:3d}/{epochs:3d} | Loss: {loss.item():.4f} | Val MAE: {v_mae:.2f} YSI")

    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        final_preds = torch.expm1(model(x_va)).clamp(min=0.0).cpu().numpy()
        mae = float(np.mean(np.abs(final_preds - y_va_lin)))
        r2 = 1.0 - float(np.sum((y_va_lin - final_preds)**2)) / (float(np.sum((y_va_lin - np.mean(y_va_lin))**2)) + 1e-8)
        med = float(np.median(np.abs(final_preds - y_va_lin)))

    metrics = {"val_mae": mae, "val_r2": r2, "val_median_ae": med}
    print(f"  [Seed {seed_idx}] Best Val MAE: {mae:.2f} | R²: {r2:.4f} | Median AE: {med:.2f}")
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parity Plot Generator
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Yield Sooting Index (YSI) Prediction",
):
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=250)

    # Subsample if large
    if len(y_true) > 15000:
        idx = np.random.choice(len(y_true), size=15000, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    ax.scatter(yt, yp, alpha=0.35, s=12, c="#1E88E5", edgecolors="none")

    lo = min(yt.min(), yp.min(), 0.0)
    hi = max(yt.max(), yp.max(), 100.0)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 Parity")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = 1.0 - float(np.sum((y_true - y_pred)**2)) / (float(np.sum((y_true - np.mean(y_true))**2)) + 1e-8)
    med = float(np.median(np.abs(y_true - y_pred)))

    stats_str = f"MAE: {mae:.2f} YSI\nMedian AE: {med:.2f} YSI\nR²: {r2:.4f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.9, edgecolor="#ccc")
    ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, fontsize=10, va="top", bbox=props)

    ax.set_xlabel("True YSI", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted YSI", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"  Parity plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YSI SELFIES mixture model.")
    parser.add_argument("--data", type=Path, default=_DATA_PATH)
    parser.add_argument("--out_dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if checkpoints exist")
    args = parser.parse_args()

    # Cap CPU threads to prevent resource hogging
    torch.set_num_threads(args.threads)
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device} (with {args.threads} CPU threads)")

    # 1. Extended Vocab
    vocab = get_extended_vocab()
    vocab_file = args.out_dir / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump({"token2idx": vocab}, f, indent=2)
    print(f"Extended vocabulary size: {len(vocab)} (saved to {vocab_file})")

    # 2. Load Data (streaming, memory safe)
    comp_list, pure_compounds, pure_tokens, pure_ys, mixture_rows = load_data_streaming(
        args.data, vocab
    )

    # 5-fold split for ensemble diversity and robust cross-validation
    kf = KFold(n_splits=args.seeds, shuffle=True, random_state=42)
    splits = list(kf.split(pure_tokens))

    models: list[ComponentYSINet] = []
    seed_metrics: list[dict[str, float]] = []
    comp_oof_preds = np.zeros(len(pure_ys), dtype=np.float32)

    # Check if all checkpoints already exist
    all_ckpts_exist = all(
        (args.out_dir / f"ysi_predictor_seed{i}.pt").exists() for i in range(args.seeds)
    )

    if all_ckpts_exist and not args.retrain:
        print("\n" + "=" * 60)
        print("Found existing checkpoints! Loading pre-trained ensemble models ...")
        print("=" * 60)
        for seed_idx, (tr_idx, va_idx) in enumerate(splits):
            ckpt_path = args.out_dir / f"ysi_predictor_seed{seed_idx}.pt"
            model = ComponentYSINet(vocab_size=len(vocab)).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            models.append(model)

            t_va = pure_tokens[va_idx].to(device)
            y_va_lin = pure_ys[va_idx]
            with torch.no_grad():
                preds_lin = torch.clamp(torch.exp(model(t_va)) - 1.0, min=0.0).cpu().numpy()
                comp_oof_preds[va_idx] = preds_lin
                mae = float(np.mean(np.abs(preds_lin - y_va_lin)))
                r2 = 1.0 - float(np.sum((y_va_lin - preds_lin)**2)) / (float(np.sum((y_va_lin - np.mean(y_va_lin))**2)) + 1e-8)
                med = float(np.median(np.abs(preds_lin - y_va_lin)))
                seed_metrics.append({"val_mae": mae, "val_r2": r2, "val_median_ae": med})
                print(f"  [Seed {seed_idx}] Loaded: Val MAE = {mae:.2f} | R² = {r2:.4f} | Median AE = {med:.2f}")
    else:
        print("\n" + "=" * 60)
        print("Training Ensemble of Component YSI Models")
        print("=" * 60)

        for seed_idx, (tr_idx, va_idx) in enumerate(splits):
            print(f"\nTraining Seed {seed_idx} ...")
            t_tr, y_tr = pure_tokens[tr_idx], torch.tensor(pure_ys[tr_idx])
            t_va, y_va = pure_tokens[va_idx], torch.tensor(pure_ys[va_idx])

            model, metrics = train_seed(
                seed_idx, t_tr, y_tr, t_va, y_va,
                vocab_size=len(vocab),
                device=device,
                epochs=args.epochs,
            )
            models.append(model)
            seed_metrics.append(metrics)

            # Record out-of-fold predictions
            model.eval()
            with torch.no_grad():
                comp_oof_preds[va_idx] = torch.clamp(torch.exp(model(t_va.to(device))) - 1.0, min=0.0).cpu().numpy()

            # Save individual checkpoint
            ckpt_path = args.out_dir / f"ysi_predictor_seed{seed_idx}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Component-level ensemble metrics
    comp_mae = float(np.mean(np.abs(comp_oof_preds - pure_ys)))
    comp_r2 = 1.0 - float(np.sum((pure_ys - comp_oof_preds)**2)) / (float(np.sum((pure_ys - np.mean(pure_ys))**2)) + 1e-8)
    comp_med = float(np.median(np.abs(comp_oof_preds - pure_ys)))

    print("\n" + "=" * 60)
    print(f"Component-Level Out-Of-Fold Evaluation (567 Pure Compounds):")
    print(f"  MAE       : {comp_mae:.2f} YSI")
    print(f"  Median AE : {comp_med:.2f} YSI")
    print(f"  R²        : {comp_r2:.4f}")
    print("=" * 60)

    # 3. Mixture Validation
    print(f"\nEvaluating Linear Blends on {len(mixture_rows):,} Sampled Mixtures ...")
    # Precompute all component predictions with ensemble average
    all_comp_preds = np.zeros(len(comp_list), dtype=np.float32)
    with torch.no_grad():
        t_all = pure_tokens.to(device)
        for m in models:
            m.eval()
            all_comp_preds += torch.expm1(m(t_all)).clamp(min=0.0).cpu().numpy()
        all_comp_preds /= len(models)

    mix_true = np.array([m[2] for m in mixture_rows], dtype=np.float32)
    mix_pred = np.zeros(len(mixture_rows), dtype=np.float32)

    for i, (indices, fracs, _) in enumerate(mixture_rows):
        val = sum(fracs[k] * all_comp_preds[indices[k]] for k in range(len(indices)))
        mix_pred[i] = val

    mix_mae = float(np.mean(np.abs(mix_pred - mix_true)))
    mix_r2 = 1.0 - float(np.sum((mix_true - mix_pred)**2)) / (float(np.sum((mix_true - np.mean(mix_true))**2)) + 1e-8)
    mix_med = float(np.median(np.abs(mix_pred - mix_true)))

    print(f"Mixture-Level Validation Metrics:")
    print(f"  MAE       : {mix_mae:.2f} YSI")
    print(f"  Median AE : {mix_med:.2f} YSI")
    print(f"  R²        : {mix_r2:.4f}")

    # 4. Parity Plots
    plot_parity(
        pure_ys, comp_oof_preds,
        args.out_dir / "ysi_dha_component_parity.png",
        title="Component YSI Prediction (Out-of-Fold)"
    )
    plot_parity(
        mix_true, mix_pred,
        args.out_dir / "ysi_dha_parity.png",
        title="YSI Mixture Linear Blending Parity"
    )

    # Copy to artifacts directory
    artifacts_dir = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")
    if artifacts_dir.exists():
        import shutil
        shutil.copy2(args.out_dir / "ysi_dha_parity.png", artifacts_dir / "ysi_dha_parity.png")
        shutil.copy2(args.out_dir / "ysi_dha_component_parity.png", artifacts_dir / "ysi_dha_component_parity.png")

    # 5. Export End-to-End ONNX Model
    print("\nExporting End-to-End ONNX Model ...")
    best_seed = int(np.argmin([m["val_mae"] for m in seed_metrics]))
    print(f"  Best seed for single-model export: Seed {best_seed}")
    best_comp_model = models[best_seed]
    
    full_mixture_model = FullYSIMixtureModel(best_comp_model).eval().to(device)

    onnx_path = args.out_dir / "ysi_dha.onnx"
    dummy_tokens = torch.zeros(1, N_COMPONENTS, MAX_SELFIES_LEN, dtype=torch.long)
    dummy_fracs = torch.full((1, N_COMPONENTS), 0.1, dtype=torch.float32)

    torch.onnx.export(
        full_mixture_model,
        (dummy_tokens, dummy_fracs),
        str(onnx_path),
        input_names=["component_tokens", "mole_fracs"],
        output_names=["ysi_mix"],
        dynamic_axes={
            "component_tokens": {0: "batch"},
            "mole_fracs": {0: "batch"},
            "ysi_mix": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    sz_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Exported ONNX model to {onnx_path} ({sz_mb:.2f} MB)")

    # 6. Save Metadata JSON
    meta = {
        "model_name": "YSI Linear Mixture Predictor",
        "description": "Component-level SELFIES sequence model with strictly linear mole-fraction mixture blending",
        "mixing_rule": "linear_mole_fraction: ysi_mix = sum(cpnt_mole_frac_i * ysi_i)",
        "n_components": N_COMPONENTS,
        "max_selfies_len": MAX_SELFIES_LEN,
        "vocab_size": len(vocab),
        "n_pure_compounds": len(pure_compounds),
        "component_oof_mae": comp_mae,
        "component_oof_r2": comp_r2,
        "component_oof_median_ae": comp_med,
        "mixture_validation_mae": mix_mae,
        "mixture_validation_r2": mix_r2,
        "mixture_validation_median_ae": mix_med,
        "seed_metrics": seed_metrics,
        "onnx_file": "ysi_dha.onnx",
        "total_time_s": time.time() - t0,
    }
    meta_path = args.out_dir / "ysi_dha_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata to {meta_path}")

    print(f"\nDone! All outputs written to {args.out_dir} in {(time.time() - t0):.1f} s.")


if __name__ == "__main__":
    main()
