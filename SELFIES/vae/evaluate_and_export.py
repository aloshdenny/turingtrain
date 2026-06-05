"""
evaluate_and_export.py
======================
Generates selfies_vae_predictions.csv and exports the trained PyTorch
mixture models to ONNX format.
"""
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "SELFIES"))
sys.path.insert(0, str(_HERE))

from selfies_tokenizer import SELFIESTokenizer
from selfies_vae import SELFIESVAE
from mixture_cn_predictor import CNPredictor, MixtureCNModel
from train_vae_optimized import (
    HybridCNPredictor,
    HybridMixtureCNModel,
    mixture_chem_features,
    stratified_split,
    MixtureDataset
)

def load_data():
    cache_path = _HERE.parent / "data" / "cn_mixtures_selfies.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")
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
    
    # Paths to checkpoints
    ckpt_dir = _HERE.parent / "checkpoints"
    ckpt_opt_dir = _HERE.parent / "checkpoints_opt"
    
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
                latent_dim=128, chem_dim=8,
                hidden_dims=(512, 256, 128), dropout=0.0
            ).to(device)
            model_opt = HybridMixtureCNModel(vae_opt.encoder, pred_opt).to(device)
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
        
    # Generate predictions
    print("Generating predictions...")
    preds_opt = []
    preds_base = []
    
    dataset = MixtureDataset(df_all, tokenizer, max_seq_len)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            comp_tok, vf, chem, _ = dataset[i]
            comp_tok = comp_tok.unsqueeze(0).to(device) # (1, 10, seq_len)
            vf = vf.unsqueeze(0).to(device) # (1, 10)
            chem = chem.unsqueeze(0).to(device) # (1, 8)
            
            if has_opt:
                p_opt, _ = model_opt(comp_tok, vf, chem)
                preds_opt.append(p_opt.squeeze(0).item())
            else:
                preds_opt.append(np.nan)
                
            if has_base:
                p_base, _ = model_base(comp_tok, vf)
                preds_base.append(p_base.squeeze(0).item())
            else:
                preds_base.append(np.nan)
                
    # Save to CSV
    df_pred = pd.DataFrame({
        "No": df_all["No"],
        "mixture_name": df_all["mixture_name"] if "mixture_name" in df_all.columns else "",
        "CN": df_all["CN"],
        "predicted_CN": preds_opt if has_opt else preds_base,
        "split": df_all["split"]
    })
    
    out_csv = _HERE.parent / "selfies_vae_predictions.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"✓ Saved predictions to {out_csv}")
    
    # ONNX Export
    print("\nAttempting ONNX export...")
    
    # Sample input for ONNX export
    comp_tok, vf, chem, _ = dataset[0]
    comp_tok = comp_tok.unsqueeze(0).to(device)
    vf = vf.unsqueeze(0).to(device)
    chem = chem.unsqueeze(0).to(device)
    
    if has_opt:
        print("Exporting optimized model to ONNX...")
        opt_onnx_path = _HERE.parent / "selfies_vae_optimized.onnx"
        try:
            torch.onnx.export(
                model_opt,
                (comp_tok, vf, chem),
                str(opt_onnx_path),
                input_names=["comp_tok", "vf", "chem"],
                output_names=["predicted_CN", "z_mix"],
                dynamic_axes={
                    "comp_tok": {0: "batch_size"},
                    "vf": {0: "batch_size"},
                    "chem": {0: "batch_size"},
                    "predicted_CN": {0: "batch_size"},
                    "z_mix": {0: "batch_size"},
                },
                opset_version=18,
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
                opset_version=18,
            )
            print(f"✓ Exported baseline model to {base_onnx_path}")
        except Exception as e:
            print(f"✗ Failed to export baseline model to ONNX: {e}")

if __name__ == "__main__":
    main()
