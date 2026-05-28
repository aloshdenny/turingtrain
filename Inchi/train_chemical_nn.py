"""
Train a simple neural network using direct chemical features.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def extract_chemical_features(inchi_str):
    """Extract chemical features from InChI string."""
    features = {
        'C': 0, 'H': 0, 'O': 0, 'N': 0,
        'S': 0, 'P': 0, 'F': 0,
        'Cl': 0, 'Br': 0, 'I': 0,
        'other': 0,
        'stereo': 0,
        'charge': 0,
        'isotope': 0,
    }
    
    if not isinstance(inchi_str, str) or not inchi_str.startswith('InChI='):
        return list(features.values())
    
    try:
        mol = Chem.MolFromInchi(inchi_str)
        if mol is None:
            return list(features.values())
        
        # Extract formula
        formula_part = inchi_str.split('/')[1] if '/' in inchi_str else ''
        
        import re
        pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
        
        for match in pattern.finditer(formula_part):
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            if element in features and element != 'isotope':
                features[element] += count
            else:
                features['other'] += count
        
        features['stereo'] = 1 if '/t' in inchi_str or '/m' in inchi_str else 0
        features['charge'] = 1 if '/q' in inchi_str else 0
        features['isotope'] = 1 if '/i' in inchi_str else 0
        
    except:
        pass
    
    return list(features.values())


def build_mixture_features(df):
    """Build feature matrix for mixtures."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]
    
    rows = []
    
    for _, row in df.iterrows():
        # Normalize volumes
        volumes = []
        for col in vol_cols:
            val = row.get(col, 0.0)
            volumes.append(float(val) if pd.notna(val) else 0.0)
        
        total_volume = sum(volumes)
        if total_volume == 0:
            total_volume = 1.0
        
        normalized = [v / total_volume for v in volumes]
        
        # Features for this mixture
        mixture_features = []
        
        # Per-component weighted features
        for idx, inchi_col in enumerate(inchi_cols):
            inchi = row.get(inchi_col, '')
            comp_features = extract_chemical_features(inchi)
            vol_weight = normalized[idx]
            
            # Weighted features
            weighted = [f * vol_weight for f in comp_features]
            mixture_features.extend(weighted)
        
        rows.append(mixture_features)
    
    return np.array(rows)


class ChemicalNN(nn.Module):
    """Simple neural network for chemical property prediction."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def main():
    print("Using device: cpu")
    print("Loading dataset...")
    df = pd.read_csv('cn_mixtues_inchi.dat', sep='\t')
    print(f"Loaded {len(df)} records")
    
    print("Building chemical feature matrix...")
    X = build_mixture_features(df)
    y = df['CN'].astype(float).values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target CN range: {y.min():.2f} - {y.max():.2f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Model
    device = torch.device('cpu')
    model = ChemicalNN(X_train.shape[1], hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)
    
    best_test_mae = float('inf')
    
    print("\nTraining...")
    print("Epoch | Train Loss | Test MAE | Test R²")
    print("-" * 45)
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_t).squeeze()
        train_loss = loss_fn(train_pred, y_train_t)
        train_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t).squeeze().cpu().numpy()
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    torch.save(model.state_dict(), 'nn_chemical_model.pt')
                
                print(f"{epoch:4d} | {train_loss.item():10.4f} | {test_mae:8.4f} | {test_r2:7.4f}")
    
    print("\n" + "="*45)
    print(f"Best Test MAE: {best_test_mae:.4f}")
    print(f"RandomForest Baseline MAE: 7.29")


if __name__ == '__main__':
    main()
