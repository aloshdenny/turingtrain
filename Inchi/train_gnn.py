"""
Training pipeline for GNN mixture model.
Converts InChI data to molecular graphs and trains the mixture GNN.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem
from torch_geometric.data import Data
from gnn_model import MixtureGNN


def inchi_to_graph(inchi_str):
    """Convert InChI string to PyG Data object with atom and bond features."""
    try:
        mol = Chem.MolFromInchi(inchi_str)
        if mol is None:
            return None
    except:
        return None

    # Atom features: [atomic_num, aromatic, degree, formal_charge+2, hybridization, num_h]
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            min(atom.GetDegree(), 4),
            atom.GetFormalCharge() + 2,  # Map -2 to +2 -> 0 to 4
            int(atom.GetHybridization()),
            min(atom.GetTotalNumHs(), 4)
        ]
        atom_features.append(features)

    if not atom_features:
        return None

    x = torch.tensor(atom_features, dtype=torch.long)

    # Bond features: [bond_type, is_aromatic, is_conjugated]
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        edge_index.append([begin_atom_idx, end_atom_idx])
        edge_index.append([end_atom_idx, begin_atom_idx])

        bond_type = int(bond.GetBondType())
        is_aromatic = int(bond.GetIsAromatic())
        is_conjugated = int(bond.GetIsConjugated())
        features = [bond_type, is_aromatic, is_conjugated]

        edge_features.append(features)
        edge_features.append(features)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        edge_attr = torch.tensor([], dtype=torch.long).reshape(0, 3)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_mixture_graphs(df):
    """Build graph representations for all mixtures."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]

    mixture_graphs = []
    mole_fractions_list = []

    for _, row in df.iterrows():
        volumes = []
        for col in vol_cols:
            val = row.get(col, 0.0)
            volumes.append(float(val) if pd.notna(val) else 0.0)

        total_volume = sum(volumes)
        if total_volume == 0:
            total_volume = 1.0

        mole_frac = [v / total_volume for v in volumes]

        graphs = []
        for inchi_col in inchi_cols:
            inchi = row.get(inchi_col, '')
            if isinstance(inchi, str) and inchi.strip():
                graph = inchi_to_graph(inchi)
                graphs.append(graph)
            else:
                graphs.append(None)

        mixture_graphs.append(graphs)
        mole_fractions_list.append(mole_frac)

    return mixture_graphs, mole_fractions_list


def create_batch(mixture_samples, mole_fraction_samples, max_components=12):
    """Create batches for training."""
    batch_size = len(mixture_samples)
    component_graphs = mixture_samples
    mole_frac = np.array(mole_fraction_samples)

    # Pad to max_components if needed
    if mole_frac.shape[1] < max_components:
        padding = np.zeros((batch_size, max_components - mole_frac.shape[1]))
        mole_frac = np.concatenate([mole_frac, padding], axis=1)

    mole_frac_tensor = torch.from_numpy(mole_frac).float()
    return component_graphs, mole_frac_tensor


def train_epoch(model, mixture_graphs, mole_fractions, targets, optimizer, loss_fn, device, max_components=12):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for i in range(0, len(mixture_graphs), 16):
        batch_graphs = mixture_graphs[i:i+16]
        batch_mole_frac = mole_fractions[i:i+16]
        batch_targets = targets[i:i+16]

        component_graphs, mole_frac_tensor = create_batch(batch_graphs, batch_mole_frac, max_components)
        mole_frac_tensor = mole_frac_tensor.to(device)

        try:
            preds = model(component_graphs, mole_frac_tensor)
            preds = preds.to(device)
            batch_targets_tensor = torch.from_numpy(batch_targets).float().to(device)

            loss = loss_fn(preds, batch_targets_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

    return total_loss / max(1, len(mixture_graphs) // 16)


def evaluate(model, mixture_graphs, mole_fractions, targets, device, max_components=12):
    """Evaluate on test set."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(mixture_graphs), 16):
            batch_graphs = mixture_graphs[i:i+16]
            batch_mole_frac = mole_fractions[i:i+16]

            component_graphs, mole_frac_tensor = create_batch(batch_graphs, batch_mole_frac, max_components)
            mole_frac_tensor = mole_frac_tensor.to(device)

            try:
                preds = model(component_graphs, mole_frac_tensor)
                all_preds.extend(preds.cpu().numpy())
            except Exception as e:
                print(f"Error in eval batch {i}: {e}")
                continue

    all_preds = np.array(all_preds)
    mae = mean_absolute_error(targets, all_preds)
    r2 = r2_score(targets, all_preds)

    return mae, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='cn_mixtues_inchi.dat')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--output-model', default='gnn_mixture_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(args.data, sep='\t')
    print(f"Loaded {len(df)} records")

    # Build graphs
    print("Building mixture graphs...")
    mixture_graphs, mole_fractions = build_mixture_graphs(df)
    targets = df['CN'].astype(float).values

    # Train/test split
    X_train, X_test, y_train, y_test, mf_train, mf_test = train_test_split(
        mixture_graphs, targets, mole_fractions,
        test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create model
    model = MixtureGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training loop
    print("\nTraining...")
    print("Epoch | Train Loss | Test MAE | Test R²")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, X_train, mf_train, y_train, optimizer, loss_fn, device)
        
        if epoch % 5 == 0 or epoch == 1:
            test_mae, test_r2 = evaluate(model, X_test, mf_test, y_test, device)
            print(f"{epoch:4d} | {train_loss:10.4f} | {test_mae:8.4f} | {test_r2:7.4f}")
            torch.save(model.state_dict(), args.output_model)

    print("\nTraining complete!")
    print(f"Model saved to {args.output_model}")


if __name__ == '__main__':
    main()
