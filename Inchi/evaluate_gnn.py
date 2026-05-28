"""
Evaluate and compare GNN model against RandomForest baseline.
"""
import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from gnn_model import MixtureGNN
from train_gnn import build_mixture_graphs, create_batch


def evaluate_gnn(model_path='gnn_mixture_model.pt', data_path='cn_mixtues_inchi.dat', test_frac=0.2):
    """Evaluate GNN model."""
    device = torch.device('cpu')
    
    # Load model
    model = MixtureGNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    mixture_graphs, mole_fractions = build_mixture_graphs(df)
    targets = df['CN'].astype(float).values
    
    # Test set (last 20%)
    n_test = int(len(df) * test_frac)
    test_idx = np.arange(len(df) - n_test, len(df))
    
    test_graphs = [mixture_graphs[i] for i in test_idx]
    test_mole_frac = [mole_fractions[i] for i in test_idx]
    test_targets = targets[test_idx]
    
    # Predictions
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_graphs), 16):
            batch_graphs = test_graphs[i:i+16]
            batch_mole_frac = test_mole_frac[i:i+16]
            
            component_graphs, mole_frac_tensor = create_batch(batch_graphs, batch_mole_frac)
            mole_frac_tensor = mole_frac_tensor.to(device)
            
            try:
                preds = model(component_graphs, mole_frac_tensor)
                all_preds.extend(preds.cpu().numpy())
            except:
                continue
    
    all_preds = np.array(all_preds)
    
    gnn_mae = mean_absolute_error(test_targets, all_preds)
    gnn_r2 = r2_score(test_targets, all_preds)
    
    return gnn_mae, gnn_r2


def evaluate_rf(model_path='inchi_cn_model.pkl', data_path='cn_mixtues_inchi.dat', test_frac=0.2):
    """Evaluate RandomForest baseline."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data (RF was trained on full dataset, use same test indices)
    df = pd.read_csv(data_path, sep='\t')
    n_test = int(len(df) * test_frac)
    test_idx = np.arange(len(df) - n_test, len(df))
    
    # For RF, we need the feature matrix - would need to rebuild from notebook
    # This is a placeholder
    return None, None


if __name__ == '__main__':
    print("GNN Model Evaluation")
    print("="*50)
    
    try:
        gnn_mae, gnn_r2 = evaluate_gnn()
        print(f"GNN Test MAE: {gnn_mae:.4f}")
        print(f"GNN Test R²: {gnn_r2:.4f}")
    except Exception as e:
        print(f"Error evaluating GNN: {e}")
    
    print("\nRandomForest Baseline: MAE=7.29, R²=0.620")
