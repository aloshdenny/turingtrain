"""
Debug and inspect InChI conversion and model behavior.
"""
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from train_gnn import inchi_to_graph


def inspect_inchi_samples(data_path='cn_mixtues_inchi.dat', num_samples=5):
    """Inspect random InChI samples and their graph conversions."""
    df = pd.read_csv(data_path, sep='\t')
    
    print("="*60)
    print("InChI Sample Inspection")
    print("="*60)
    
    for idx in np.random.choice(len(df), min(num_samples, len(df)), replace=False):
        row = df.iloc[idx]
        print(f"\nRow {idx} (CN={row['CN']}):")
        
        for i in range(1, 11):
            inchi = row.get(f'cpnt_inchi_{i}', '')
            vol = row.get(f'cpnt_vol_{i}', 0.0)
            
            if isinstance(inchi, str) and inchi.strip():
                print(f"  Component {i}: vol={vol}")
                print(f"    InChI: {inchi[:80]}...")
                
                mol = Chem.MolFromInchi(inchi)
                if mol:
                    print(f"    Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")
                    graph = inchi_to_graph(inchi)
                    if graph:
                        print(f"    Graph x shape: {graph.x.shape}, edge_index shape: {graph.edge_index.shape}")
                    else:
                        print(f"    Graph conversion: FAILED")
                else:
                    print(f"    RDKit conversion: FAILED")


def inspect_model_predictions(model_path='gnn_mixture_model.pt', data_path='cn_mixtues_inchi.dat', num_samples=3):
    """Inspect model predictions on samples."""
    from gnn_model import MixtureGNN
    from train_gnn import build_mixture_graphs, create_batch
    
    device = torch.device('cpu')
    
    print("\n" + "="*60)
    print("Model Prediction Inspection")
    print("="*60)
    
    # Load model
    model = MixtureGNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    mixture_graphs, mole_fractions = build_mixture_graphs(df)
    targets = df['CN'].astype(float).values
    
    # Sample predictions
    with torch.no_grad():
        for idx in np.random.choice(len(df), min(num_samples, len(df)), replace=False):
            component_graphs, mole_frac_tensor = create_batch(
                [mixture_graphs[idx]], [mole_fractions[idx]]
            )
            
            try:
                pred = model(component_graphs, mole_frac_tensor)
                print(f"\nSample {idx}:")
                print(f"  Target CN: {targets[idx]:.2f}")
                print(f"  Predicted CN: {pred.item():.2f}")
                print(f"  Error: {abs(targets[idx] - pred.item()):.2f}")
            except Exception as e:
                print(f"\nSample {idx}: Error - {e}")


if __name__ == '__main__':
    inspect_inchi_samples()
    # inspect_model_predictions()  # Uncomment after training
