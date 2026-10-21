"""
train_sidt_gasoline_selfies_export.py
──────────────────────────────────────
Trains an Attention-FiLM ResNet on the 10-component gasoline surrogate mixture dataset
(sidt_lhs_gasoline2_K10_1M.dat). The dataset is composed of 10 subsets, each with a
DIFFERENT fixed set of 10 component SELFIES strings.

Bug fixed: previously all rows used the SELFIES tokens from row 0 only (subset 1),
meaning 83.9% of training data used the wrong chemical representations.

Now each subset's 10 SELFIES are independently tokenized and encoded to get the
correct mu_10 latent matrix for that subset. A per-row subset_id maps each sample
to its correct mu_10 at train time.
"""

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SELFIES_DIR = _ROOT / "SELFIES"

sys.path.insert(0, str(_SELFIES_DIR))
sys.path.insert(0, str(_SELFIES_DIR / "vae"))
from selfies_tokenizer import SELFIESTokenizer  # noqa
from selfies_vae import SELFIESEncoder, SELFIESVAE  # noqa

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_COMPONENTS    = 10
MAX_SELFIES_LEN = 65
LATENT_DIM      = 128
REACTOR_DIM     = 4
HIDDEN_DIM      = 256
NUM_BLOCKS      = 4
NUM_HEADS       = 4
LR              = 1e-3
BATCH_SIZE      = 2048
EPOCHS          = 30
N_SEEDS         = 5


def compute_kinetic_conds(raw_conds: np.ndarray) -> np.ndarray:
    P   = np.clip(raw_conds[:, 0], 1.0e3, None)
    T   = np.clip(raw_conds[:, 1], 100.0, None)
    inv_T = 1000.0 / T
    log_P = np.log10(P / 1.0e5)
    return np.stack([inv_T, log_P, raw_conds[:, 2], raw_conds[:, 3]], axis=1).astype(np.float32)


def identify_subsets(df: pd.DataFrame):
    selfies_cols = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    df_s = df[selfies_cols].fillna("").astype(str)
    unique_df = df_s.drop_duplicates().reset_index(drop=True)
    unique_df["subset_id"] = np.arange(len(unique_df), dtype=np.int64)
    merged = df_s.merge(unique_df, on=selfies_cols, how="left")
    subset_ids = merged["subset_id"].values
    subsets = []
    for uid in range(len(unique_df)):
        row = unique_df.iloc[uid]
        subsets.append({
            "selfies_list": [str(row[c]) if pd.notna(row[c]) else "" for c in selfies_cols],
            "row_indices": np.where(subset_ids == uid)[0]
        })
    return subset_ids, subsets


def tokenize_selfies_list(selfies_list, tokenizer):
    tokens = torch.zeros(N_COMPONENTS, MAX_SELFIES_LEN, dtype=torch.long)
    for i, s in enumerate(selfies_list):
        if s:
            tokens[i] = tokenizer.encode(s, MAX_SELFIES_LEN)
    return tokens


class GasolineSIDTDataset:
    def __init__(self, df, subset_ids, k_mean, k_std):
        frac_cols = [f"cpnt_mole_frac_{i}" for i in range(1, N_COMPONENTS + 1)]
        fracs = df[frac_cols].values.astype(np.float32)
        self.fracs = torch.tensor(fracs / fracs.sum(axis=1, keepdims=True).clip(min=1e-8), dtype=torch.float32)
        raw_conds = df[["pressure_pa", "temperature_K", "phi", "egr_fraction"]].values.astype(np.float32)
        k_conds = compute_kinetic_conds(raw_conds)
        self.conds = torch.tensor((k_conds - k_mean) / k_std, dtype=torch.float32)
        idt = np.clip(df["idt_400K_s"].values.astype(np.float32), 1e-8, None)
        self.y = torch.tensor(np.log10(idt), dtype=torch.float32)
        self.idt_linear = torch.tensor(idt, dtype=torch.float32)
        self.subset_ids = torch.tensor(subset_ids, dtype=torch.long)

    def __len__(self):
        return self.fracs.shape[0]


class AttentionFiLMIDTModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=REACTOR_DIM, hidden_dim=HIDDEN_DIM,
                 num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS):
        super().__init__()
        self.comp_proj = nn.Linear(latent_dim + 1, hidden_dim)
        self.comp_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList()
        self.film_gens = nn.ModuleList()
        for _ in range(num_blocks):
            self.film_gens.append(nn.Linear(hidden_dim, 2 * hidden_dim))
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            ))
        self.out_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, mu_per_sample, fracs, cond):
        x_exp = fracs.unsqueeze(-1)
        tokens = torch.cat([mu_per_sample, x_exp], dim=-1)
        h = self.comp_proj(tokens)
        attn_out, _ = self.comp_attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h_pool = (h * x_exp).sum(dim=1)
        c_feat = self.cond_mlp(cond)
        for film_gen, block in zip(self.film_gens, self.blocks):
            gamma, beta = film_gen(c_feat).chunk(2, dim=-1)
            h_pool = h_pool + block(h_pool * (1.0 + gamma) + beta)
        return self.out_head(h_pool).squeeze(-1)


def precompute_mu10_per_subset(subsets, tokenizer, encoder, device):
    mu10_list = []
    for i, subset in enumerate(subsets):
        tokens = tokenize_selfies_list(subset["selfies_list"], tokenizer).to(device)
        with torch.no_grad():
            mu, _ = encoder(tokens)
        mu10_list.append(mu)
        print(f"  Subset {i+1}/{len(subsets)}: mu_10 encoded — {subset['selfies_list'][0][:35]}")
    return mu10_list


def gather_mu_for_batch(subset_ids_batch, mu10_stack):
    return mu10_stack[subset_ids_batch]


def train_one_seed(seed_idx, train_ds, val_ds, subsets, tokenizer, device, out_dir, n_epochs=EPOCHS):
    torch.manual_seed(seed_idx)
    np.random.seed(seed_idx)

    ck_path = _SELFIES_DIR / "checkpoints_opt" / f"seed{seed_idx}_s1_vae.pt"
    if not ck_path.exists():
        ck_path = _SELFIES_DIR / "checkpoints_opt" / "seed0_s1_vae.pt"
    print(f"  Loading VAE encoder from {ck_path.name}")
    state = torch.load(ck_path, map_location="cpu", weights_only=False)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    encoder = SELFIESEncoder(
        vocab_size=tokenizer.vocab_size, d_model=128, latent_dim=LATENT_DIM,
        n_heads=4, n_layers=4, d_ff=512, dropout=0.0,
        pad_idx=tokenizer.pad_idx, max_len=MAX_SELFIES_LEN,
    ).to(device)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"  Precomputing mu_10 for {len(subsets)} subsets…")
    mu10_list = precompute_mu10_per_subset(subsets, tokenizer, encoder, device)
    mu10_stack = torch.stack(mu10_list, dim=0)

    train_tds = torch.utils.data.TensorDataset(
        train_ds.fracs.to(device), train_ds.conds.to(device),
        train_ds.y.to(device), train_ds.subset_ids.to(device)
    )
    val_tds = torch.utils.data.TensorDataset(
        val_ds.fracs.to(device), val_ds.conds.to(device),
        val_ds.y.to(device), val_ds.subset_ids.to(device)
    )

    train_loader = DataLoader(train_tds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_tds,   batch_size=BATCH_SIZE, shuffle=False)

    predictor = AttentionFiLMIDTModel().to(device)
    optimiser = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    best_val_mae, best_state = float("inf"), None

    for epoch in range(1, n_epochs + 1):
        predictor.train()
        train_loss = 0.0
        for fracs_b, conds_b, y_b, sids_b in train_loader:
            mu_b = gather_mu_for_batch(sids_b, mu10_stack)
            pred = predictor(mu_b, fracs_b, conds_b)
            loss = F.smooth_l1_loss(pred, y_b, beta=0.05)
            optimiser.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_ds)
        scheduler.step()

        predictor.eval()
        all_pred, all_y = [], []
        with torch.no_grad():
            for fracs_b, conds_b, y_b, sids_b in val_loader:
                mu_b = gather_mu_for_batch(sids_b, mu10_stack)
                pred = predictor(mu_b, fracs_b, conds_b)
                all_pred.append(pred.cpu()); all_y.append(y_b.cpu())
        all_pred = torch.cat(all_pred); all_y = torch.cat(all_y)
        val_mae = ((10.0 ** all_pred.clamp(max=3)) - (10.0 ** all_y)).abs().mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in predictor.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [seed {seed_idx}] Epoch {epoch:2d}/{n_epochs} | Huber: {train_loss:.5f} | Val MAE: {val_mae*1000:.3f} ms")

    print(f"  [seed {seed_idx}] Best Val MAE = {best_val_mae*1000:.3f} ms")
    predictor.load_state_dict(best_state)
    torch.save(best_state, out_dir / f"predictor_seed{seed_idx}.pt")
    return predictor, mu10_stack


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


def export_full_model_onnx(encoder, predictor, k_mean, k_std, out_path):
    class FullModel(nn.Module):
        def __init__(self, enc, pred, km, ks):
            super().__init__()
            self.encoder = enc
            self.predictor = pred
            self.register_buffer("k_mean", torch.tensor(km, dtype=torch.float32))
            self.register_buffer("k_std",  torch.tensor(ks, dtype=torch.float32))

        def forward(self, tokens, fracs, raw_conds):
            mu_10, _ = self.encoder(tokens)
            mu_10_exp = mu_10.unsqueeze(0).expand(fracs.shape[0], -1, -1)
            P = torch.clamp(raw_conds[:, 0:1], min=1e3)
            T = torch.clamp(raw_conds[:, 1:2], min=100.0)
            k_raw = torch.cat([1000.0/T, torch.log10(P/1e5), raw_conds[:, 2:3], raw_conds[:, 3:4]], dim=-1)
            norm_cond = (k_raw - self.k_mean) / self.k_std
            pred_log10 = self.predictor(mu_10_exp, fracs, norm_cond)
            return torch.pow(10.0, pred_log10), pred_log10

    enc_cpu = encoder.to("cpu"); pred_cpu = predictor.to("cpu")
    _patch_encoder_for_onnx(enc_cpu)
    full = FullModel(enc_cpu, pred_cpu, k_mean, k_std).eval()

    torch.onnx.export(
        full,
        (torch.zeros(N_COMPONENTS, MAX_SELFIES_LEN, dtype=torch.long),
         torch.full((1, N_COMPONENTS), 0.1),
         torch.tensor([[1e6, 1000.0, 1.0, 0.0]])),
        str(out_path),
        input_names=["component_tokens", "mole_fracs", "reactor_conds"],
        output_names=["idt_s", "idt_log10"],
        dynamic_axes={"mole_fracs": {0: "batch"}, "reactor_conds": {0: "batch"},
                      "idt_s": {0: "batch"}, "idt_log10": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  ✓ ONNX exported to {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")


def main():
    import argparse
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="model_training/sidt/sidt_lhs_gasoline2_K10_1M.dat")
    parser.add_argument("--out_dir",    default="sidt/models/gasoline_mix_selfies")
    parser.add_argument("--seeds",      type=int, default=N_SEEDS)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    input_path = _ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_dir    = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    print(f"Loading {input_path}…")
    t0 = time.time()
    selfies_cols = [f"cpnt_selfies_{i}" for i in range(1, N_COMPONENTS + 1)]
    frac_cols    = [f"cpnt_mole_frac_{i}" for i in range(1, N_COMPONENTS + 1)]
    cond_cols    = ["pressure_pa", "temperature_K", "phi", "egr_fraction"]
    feature_cols = frac_cols + selfies_cols + cond_cols + ["idt_400K_s"]

    df = pd.read_csv(input_path, sep="\t", comment="#", usecols=feature_cols, low_memory=False)
    df_clean = df.dropna(subset=["idt_400K_s"]).reset_index(drop=True)
    print(f"Clean rows: {len(df_clean):,}  ({time.time()-t0:.1f}s)")

    print("Identifying component subsets…")
    subset_ids_all, subsets = identify_subsets(df_clean)
    print(f"Found {len(subsets)} unique component subsets:")
    for i, s in enumerate(subsets):
        print(f"  Subset {i+1}: {len(s['row_indices']):,} rows — cpnt_1: {s['selfies_list'][0][:40]}")

    raw_conds_all = df_clean[cond_cols].values.astype(np.float32)
    k_conds_all   = compute_kinetic_conds(raw_conds_all)
    k_mean = k_conds_all.mean(axis=0); k_std = k_conds_all.std(axis=0)
    print(f"Kinetic cond mean: {k_mean.tolist()}")

    idx_train, idx_val = train_test_split(np.arange(len(df_clean)), test_size=0.1, random_state=42)
    df_train = df_clean.iloc[idx_train].reset_index(drop=True)
    df_val   = df_clean.iloc[idx_val].reset_index(drop=True)
    sids_train = subset_ids_all[idx_train]
    sids_val   = subset_ids_all[idx_val]
    print(f"Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    tok_path = _SELFIES_DIR / "data" / "vocab.json"
    tokenizer = SELFIESTokenizer.load(tok_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ds = GasolineSIDTDataset(df_train, sids_train, k_mean, k_std)
    val_ds   = GasolineSIDTDataset(df_val,   sids_val,   k_mean, k_std)

    n_seeds = min(args.seeds, N_SEEDS)
    predictors, all_mu10_lists = [], []
    t_start = time.time()

    for seed_idx in range(n_seeds):
        print(f"\n{'='*60}\nTraining — Seed {seed_idx}\n{'='*60}")
        pred, mu10_list = train_one_seed(
            seed_idx, train_ds, val_ds, subsets, tokenizer, device, out_dir, n_epochs=args.epochs)
        predictors.append(pred); all_mu10_lists.append(mu10_list)

    print(f"\nEnsemble training completed in {(time.time()-t_start)/60:.1f} min.")

    print(f"\n{'='*60}\nEnsemble validation…")
    for pred in predictors: pred.eval()

    fracs_val  = val_ds.fracs.to(device)
    conds_val  = val_ds.conds.to(device)
    sids_val_t = val_ds.subset_ids.to(device)

    all_seed_preds = []
    with torch.no_grad():
        for i in range(n_seeds):
            sp = []
            for b in range(0, len(val_ds), BATCH_SIZE):
                mu_b = gather_mu_for_batch(sids_val_t[b:b+BATCH_SIZE], all_mu10_lists[i])
                p = predictors[i](mu_b, fracs_val[b:b+BATCH_SIZE], conds_val[b:b+BATCH_SIZE])
                sp.append(p.cpu())
            all_seed_preds.append(torch.cat(sp))

    ensemble_log = torch.stack(all_seed_preds).mean(0).numpy()
    true_lin = val_ds.idt_linear.numpy()
    pred_lin = 10.0 ** ensemble_log

    r2_lin  = float(r2_score(true_lin, pred_lin))
    r2_log  = float(r2_score(np.log10(true_lin), ensemble_log))
    ens_mae  = float(mean_absolute_error(true_lin, pred_lin))
    ens_rmse = float(np.sqrt(mean_squared_error(true_lin, pred_lin)))

    print(f"\n--- Ensemble Validation Metrics ---")
    print(f"  Linear R² : {r2_lin:.5f}")
    print(f"  Log10  R² : {r2_log:.5f}")
    print(f"  MAE       : {ens_mae*1000:.3f} ms")
    print(f"  RMSE      : {ens_rmse*1000:.3f} ms")

    parity_path = _ROOT / "sidt/gasoline_mix_selfies_parity.png"
    plt.figure(figsize=(7, 6), dpi=200)
    idx = np.random.choice(len(true_lin), min(15000, len(true_lin)), replace=False)
    plt.scatter(true_lin[idx]*1000, pred_lin[idx]*1000, alpha=0.35, s=12, c="#1f77b4", edgecolors="none")
    lims = [min(true_lin[idx].min(), pred_lin[idx].min())*1000, max(true_lin[idx].max(), pred_lin[idx].max())*1000]
    plt.plot(lims, lims, "r--", lw=1.5, label="1:1 Identity")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True IDT (ms)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted IDT (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline Surrogate IDT — Parity (Attention-FiLM, Per-Subset SELFIES)", fontsize=11, fontweight="bold")
    stats_text = f"Linear R²: {r2_lin:.4f}\nLog₁₀ R²:  {r2_log:.4f}\nMAE:       {ens_mae*1000:.2f} ms\nRMSE:      {ens_rmse*1000:.2f} ms"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))
    plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(parity_path, dpi=200); plt.close()
    print(f"✓ Parity plot: {parity_path}")

    fwd_path = _ROOT / "sidt/gasoline_mix_selfies_forward_plot.png"
    plt.figure(figsize=(7, 6), dpi=200)
    temps = np.linspace(600.0, 1500.0, 100)
    sweep_fracs = torch.full((len(temps), 10), 0.1, dtype=torch.float32, device=device)
    sweep_sids  = torch.zeros(len(temps), dtype=torch.long, device=device)
    for p_bar, c in zip([10.0, 20.0, 40.0], ["#2ca02c", "#1f77b4", "#d62728"]):
        conds_raw = np.array([[p_bar*1e5, t, 1.0, 0.0] for t in temps], dtype=np.float32)
        conds_norm = torch.tensor((compute_kinetic_conds(conds_raw) - k_mean) / k_std, device=device)
        with torch.no_grad():
            preds = []
            for i in range(n_seeds):
                mu_sw = gather_mu_for_batch(sweep_sids, all_mu10_lists[i])
                preds.append(predictors[i](mu_sw, sweep_fracs, conds_norm))
        sweep_ms = (10.0 ** torch.stack(preds).mean(0)).cpu().numpy() * 1000.0
        plt.plot(1000.0 / temps, sweep_ms, lw=2.0, color=c, label=f"{p_bar:.0f} bar")
    plt.yscale("log"); plt.xlabel("1000 / T (1/K)", fontsize=11, fontweight="bold")
    plt.ylabel("Ignition Delay Time (ms)", fontsize=11, fontweight="bold")
    plt.title("Gasoline Surrogate (Subset 1, Alkanes) — Arrhenius Sweep", fontsize=11, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(title="Pressure (φ=1.0, EGR=0.0)")
    plt.tight_layout(); plt.savefig(fwd_path, dpi=200); plt.close()
    print(f"✓ Forward sweep: {fwd_path}")

    artifacts_dir = Path("/Users/aoxo/.gemini/antigravity-ide/brain/05846df6-21e5-4ec0-853f-3ea8b38a7254")
    if artifacts_dir.exists():
        import shutil
        shutil.copy2(parity_path, artifacts_dir / "gasoline_mix_selfies_parity.png")
        shutil.copy2(fwd_path, artifacts_dir / "gasoline_mix_selfies_forward_plot.png")

    print("\nExporting ONNX (seed 0)…")
    ck0 = _SELFIES_DIR / "checkpoints_opt" / "seed0_s1_vae.pt"
    st0 = torch.load(ck0, map_location="cpu", weights_only=False)
    enc0_state = {k[len("encoder."):]: v for k, v in st0.items() if k.startswith("encoder.")}
    enc0 = SELFIESEncoder(vocab_size=tokenizer.vocab_size, d_model=128, latent_dim=LATENT_DIM,
                          n_heads=4, n_layers=4, d_ff=512, dropout=0.0,
                          pad_idx=tokenizer.pad_idx, max_len=MAX_SELFIES_LEN)
    enc0.load_state_dict(enc0_state, strict=True); enc0.eval()
    export_full_model_onnx(enc0, predictors[0], k_mean, k_std,
                           out_path=out_dir / "idt_gasoline_selfies.onnx")

    meta = {
        "dataset": str(input_path.name),
        "architecture": "AttentionFiLMIDTModel",
        "bug_fix": "Per-subset component SELFIES tokenization (10 unique component sets)",
        "n_samples_clean": len(df_clean),
        "n_subsets": len(subsets),
        "subset_sizes": [len(s["row_indices"]) for s in subsets],
        "n_components": N_COMPONENTS, "max_selfies_len": MAX_SELFIES_LEN,
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, "num_blocks": NUM_BLOCKS,
        "kinetic_cond_mean": k_mean.tolist(), "kinetic_cond_std": k_std.tolist(),
        "vocab_path": str(tok_path), "n_seeds_trained": n_seeds,
        "val_mae_s": ens_mae, "val_rmse_s": ens_rmse,
        "val_r2_linear": r2_lin, "val_r2_log10": r2_log,
    }
    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved to {meta_path}")
    print(f"\n✓ Done. Models saved to {out_dir}")


if __name__ == "__main__":
    main()
