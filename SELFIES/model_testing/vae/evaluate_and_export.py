"""
evaluate_and_export.py
======================
Generates selfies_vae_predictions.csv and exports the trained PyTorch
mixture models to ONNX format.
"""
import os
import sys
# Always show progress in real-time — no need for PYTHONUNBUFFERED=1
sys.stdout.reconfigure(line_buffering=True)
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# ── Paths: import from canonical SELFIES/ source (no local copies needed) ────────
_HERE     = Path(__file__).resolve().parent          # model_testing/vae/
_SELFIES  = _HERE.parents[1]                         # SELFIES/
sys.path.insert(0, str(_SELFIES))                   # for selfies_tokenizer
sys.path.insert(0, str(_SELFIES / "vae"))           # for selfies_vae, train_vae_optimized, etc.

from selfies_tokenizer import SELFIESTokenizer
from selfies_vae import SELFIESVAE
from mixture_cn_predictor import CNPredictor, MixtureCNModel
from train_vae_optimized import (
    HybridCNPredictor,
    MixtureSlotEncoder,
    AttentionMixtureCNModel,
    mixture_chem_features,
    stratified_split,
    MixtureDataset,
    CHEM_FEAT_DIM,
    SLOT_D_SLOT,
    SLOT_N_HEADS,
    SLOT_N_LAYERS,
)

def load_data():
    # Single source of truth: SELFIES/data/  (no model_testing/data/ copy needed)
    cache_path = _SELFIES / "data" / "cn_mixtures_selfies.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}\n"
            "Run: python SELFIES/data/preprocess_selfies.py"
        )
    print(f"Loading data from {cache_path}...")
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)
    return cache

def main():
    cache = load_data()
    tokenizer = SELFIESTokenizer.load(cache["vocab_path"])
    max_seq_len = cache["max_seq_len"]
    
    df_all = cache["df_selfies"].dropna(subset=["CN"]).reset_index(drop=True)
    
    # Stratified split using seed 0 (matching train_vae_optimized.py default)
    df_tr, df_val = stratified_split(df_all, seed=0)
    
    # We will compute predictions on the full dataset and label the split
    df_all["split"] = "train"
    
    # Identify which rows are in the validation set.
    val_nos = set(df_val["No"])
    
    split_col = []
    for _, row in df_all.iterrows():
        if row["No"] in val_nos:
            split_col.append("val")
        else:
            split_col.append("train")
    df_all["split"] = split_col
    
    device = torch.device("cpu")  # use cpu for stable ONNX export

    # ── Auto-detect newest trained checkpoint ─────────────────────────────────
    # Checks both the training output dir (SELFIES/checkpoints_opt/) and the
    # local copy (model_testing/checkpoints_opt/). Always picks the newer one.
    # This means you do NOT need to manually 'cp' after retraining.
    def _find_best_checkpoint_dir(*candidates: Path) -> Path:
        """Return the directory containing the most recently modified Stage-3 checkpoint."""
        best, best_mtime = candidates[0], 0
        for d in candidates:
            for ckpt_name in ("seed0_s3_model.pt", "seed0_s2_model.pt", "seed0_s1_vae.pt"):
                p = d / ckpt_name
                if p.exists():
                    t = p.stat().st_mtime
                    if t > best_mtime:
                        best_mtime = t
                        best = d
                    break  # found a ckpt in this dir, no need to check lower stages
        return best

    ckpt_opt_dir = _find_best_checkpoint_dir(
        _HERE.parent / "checkpoints_opt",           # model_testing/checkpoints_opt/ (local copy)
        _HERE.parent.parent / "checkpoints_opt",    # SELFIES/checkpoints_opt/       (training output)
    )
    print(f"Using checkpoint directory: {ckpt_opt_dir}")

    # Paths to baseline checkpoints (legacy, skipped if vocab mismatch)
    ckpt_dir = _HERE.parent / "checkpoints"

    # Load optimized model
    opt_vae_ckpt = ckpt_opt_dir / "seed0_s1_vae.pt"
    opt_model_ckpt = ckpt_opt_dir / "seed0_s3_model.pt"
    
    has_opt = opt_vae_ckpt.exists() and opt_model_ckpt.exists()
    
    if has_opt:
        print("Loading optimized model...")
        try:
            vae_opt = SELFIESVAE(
                vocab_size=tokenizer.vocab_size,
                d_model=128, latent_dim=128, n_heads=4, n_layers=4,
                d_ff=512, dropout=0.0,
                pad_idx=tokenizer.pad_idx,
                bos_idx=tokenizer.bos_idx,
                eos_idx=tokenizer.eos_idx,
                max_len=max_seq_len,
            ).to(device)
            pred_opt = HybridCNPredictor(
                latent_dim=128, chem_dim=CHEM_FEAT_DIM,
                hidden_dims=(512, 256, 128), dropout=0.0
            ).to(device)
            slot_enc_opt = MixtureSlotEncoder(
                latent_dim=128, chem_dim=CHEM_FEAT_DIM,
                d_slot=SLOT_D_SLOT, n_heads=SLOT_N_HEADS,
                n_layers=SLOT_N_LAYERS, dropout=0.0
            ).to(device)
            model_opt = AttentionMixtureCNModel(vae_opt.encoder, slot_enc_opt, pred_opt).to(device)
            model_opt.load_state_dict(torch.load(opt_model_ckpt, map_location=device))
            model_opt.eval()
            print("✓ Optimized model loaded successfully.")
        except Exception as e:
            print(f"⚠ Could not load optimized model checkpoint: {e}")
            has_opt = False
    else:
        print("⚠ Optimized model checkpoint not found.")
        
    # Load baseline model
    base_vae_ckpt = ckpt_dir / "stage3_vae_best.pt"
    base_model_ckpt = ckpt_dir / "stage3_joint_best.pt"
    
    has_base = base_vae_ckpt.exists() and base_model_ckpt.exists()
    
    if has_base:
        print("Loading baseline model...")
        try:
            vae_base = SELFIESVAE(
                vocab_size=tokenizer.vocab_size,
                d_model=128, latent_dim=64, n_heads=4, n_layers=4,
                d_ff=512, dropout=0.0,
                pad_idx=tokenizer.pad_idx,
                bos_idx=tokenizer.bos_idx,
                eos_idx=tokenizer.eos_idx,
                max_len=max_seq_len,
            ).to(device)
            pred_base = CNPredictor(latent_dim=64, hidden_dims=(256, 128)).to(device)
            model_base = MixtureCNModel(vae_base.encoder, pred_base, n_components=10).to(device)
            model_base.load_state_dict(torch.load(base_model_ckpt, map_location=device))
            model_base.eval()
            print("✓ Baseline model loaded successfully.")
        except Exception as e:
            print(f"⚠ Could not load baseline model checkpoint (likely due to vocabulary mismatch after dataset expansion): {e}")
            has_base = False
    else:
        print("⚠ Baseline model checkpoint not found.")
        

    # Generate predictions (batched for speed)
    print("Generating predictions...")
    dataset = MixtureDataset(df_all, tokenizer, max_seq_len)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    preds_opt  = []
    preds_base = []

    with torch.no_grad():
        for batch_i, (comp_tok, vf, chem_mix, chem_pc, _) in enumerate(loader):
            comp_tok = comp_tok.to(device)
            vf       = vf.to(device)
            chem_mix = chem_mix.to(device)
            chem_pc  = chem_pc.to(device)

            if has_opt:
                p_opt, _ = model_opt(comp_tok, vf, chem_mix, chem_pc)
                preds_opt.extend(p_opt.cpu().tolist())
            else:
                preds_opt.extend([np.nan] * len(comp_tok))

            if has_base:
                p_base, _ = model_base(comp_tok, vf)
                preds_base.extend(p_base.cpu().tolist())
            else:
                preds_base.extend([np.nan] * len(comp_tok))

            if (batch_i + 1) % 10 == 0:
                print(f"  {min((batch_i + 1) * 64, len(dataset))}/{len(dataset)} predictions done...",
                      flush=True)

    print(f"  {len(dataset)}/{len(dataset)} predictions done.")
                
    # Save to CSV
    df_pred = pd.DataFrame({
        "No": df_all["No"],
        "mixture_name": df_all["mixture_name"] if "mixture_name" in df_all.columns else "",
        "CN": df_all["CN"],
        "predicted_CN": preds_opt if has_opt else preds_base,
        "split": df_all["split"]
    })
    
    # Only keep the validation split
    df_pred = df_pred[df_pred["split"] == "val"].reset_index(drop=True)
    
    out_csv = _HERE.parent / "selfies_vae_predictions.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"✓ Saved validation-only predictions to {out_csv}")
    
    # ONNX Export
    print("\nAttempting ONNX export...")
    
    # Sample input for ONNX export
    comp_tok, vf, chem_mix, chem_pc, _ = dataset[0]
    comp_tok = comp_tok.unsqueeze(0).to(device)
    vf       = vf.unsqueeze(0).to(device)
    chem_mix = chem_mix.unsqueeze(0).to(device)
    chem_pc  = chem_pc.unsqueeze(0).to(device)

    if has_opt:
        print("Exporting optimized model to ONNX...")
        opt_onnx_path = _HERE.parent / "selfies_vae_optimized.onnx"
        try:
            torch.onnx.export(
                model_opt,
                (comp_tok, vf, chem_mix, chem_pc),
                str(opt_onnx_path),
                input_names=["comp_tok", "vf", "chem_mix", "chem_pc"],
                output_names=["predicted_CN", "z_mix"],
                dynamic_axes={
                    "comp_tok": {0: "batch_size"},
                    "vf":       {0: "batch_size"},
                    "chem_mix": {0: "batch_size"},
                    "chem_pc":  {0: "batch_size"},
                    "predicted_CN": {0: "batch_size"},
                    "z_mix":    {0: "batch_size"},
                },
                opset_version=17,
                verbose=False,
            )
            print(f"✓ Exported optimized model to {opt_onnx_path}")
        except Exception as e:
            print(f"✗ Failed to export optimized model to ONNX: {e}")
            
    if has_base:
        print("Exporting baseline model to ONNX...")
        base_onnx_path = _HERE.parent / "selfies_vae_baseline.onnx"
        try:
            torch.onnx.export(
                model_base,
                (comp_tok, vf),
                str(base_onnx_path),
                input_names=["comp_tok", "vf"],
                output_names=["predicted_CN", "z_mix"],
                dynamic_axes={
                    "comp_tok": {0: "batch_size"},
                    "vf": {0: "batch_size"},
                    "predicted_CN": {0: "batch_size"},
                    "z_mix": {0: "batch_size"},
                },
                opset_version=17,
                verbose=False,
            )
            print(f"✓ Exported baseline model to {base_onnx_path}")
        except Exception as e:
            print(f"✗ Failed to export baseline model to ONNX: {e}")

if __name__ == "__main__":
    main()
