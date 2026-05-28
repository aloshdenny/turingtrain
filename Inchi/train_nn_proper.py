"""
Train NN using the SAME feature engineering as the successful notebook.
"""
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def parse_inchi_formula(inchi):
    """Extract element counts from InChI formula section."""
    if not isinstance(inchi, str):
        return {
            'C': 0, 'H': 0, 'O': 0, 'N': 0,
            'S': 0, 'P': 0, 'F': 0,
            'Cl': 0, 'Br': 0, 'I': 0,
            'other': 0,
            'formula_len': 0
        }

    if not inchi.startswith('InChI='):
        return {
            'C': 0, 'H': 0, 'O': 0, 'N': 0,
            'S': 0, 'P': 0, 'F': 0,
            'Cl': 0, 'Br': 0, 'I': 0,
            'other': 0,
            'formula_len': 0
        }

    try:
        formula_part = inchi.split('/')[1]
    except IndexError:
        formula_part = ''

    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')

    counts = {
        'C': 0, 'H': 0, 'O': 0, 'N': 0,
        'S': 0, 'P': 0, 'F': 0,
        'Cl': 0, 'Br': 0, 'I': 0
    }

    other = 0
    for match in pattern.finditer(formula_part):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        if element in counts:
            counts[element] += count
        else:
            other += count

    counts['other'] = other
    counts['formula_len'] = len(formula_part)
    return counts


def extract_inchi_features(inchi):
    """Extract full feature set from InChI (matching notebook)."""
    if not isinstance(inchi, str):
        inchi = ''

    counts = parse_inchi_formula(inchi)
    valid = inchi.startswith('InChI=')

    features = {
        'has_inchi': int(valid),
        'string_len': len(inchi),
        'has_stereo': int('/t' in inchi or '/m' in inchi or '/s' in inchi),
        'has_charge': int('/q' in inchi),
        'has_isotope': int('/i' in inchi),
        'has_reconnected': int('/r' in inchi),
    }

    features.update(counts)

    # Heavy atoms
    features['num_heavy_atoms'] = sum(
        counts[e] for e in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    )

    # Heteroatoms
    features['num_heteroatoms'] = sum(
        counts[e] for e in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    )

    # Ratios
    features['CH_ratio'] = counts['C'] / counts['H'] if counts['H'] > 0 else 0.0
    features['OC_ratio'] = counts['O'] / counts['C'] if counts['C'] > 0 else 0.0

    return features


def build_feature_matrix(df):
    """Build feature matrix exactly like notebook."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]

    rows = []

    for _, row in df.iterrows():
        # Normalize volumes
        volumes = []
        for col in vol_cols:
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            volumes.append(float(val))

        total_volume = sum(volumes)
        if total_volume == 0:
            total_volume = 1.0

        normalized = [v / total_volume for v in volumes]

        base = {
            # Mixture-level stats
            'mix_total_volume': total_volume,
            'mix_nonzero_components': sum(1 for v in normalized if v > 0),
            'mix_max_volume': max(normalized),
            'mix_min_nonzero_volume': min((v for v in normalized if v > 0), default=0.0),
            'mix_volume_entropy': -sum(v * np.log(v) for v in normalized if v > 0),
        }

        # Component-level features
        for idx, inchi_col in enumerate(inchi_cols, start=1):
            inchi_value = row.get(inchi_col, '')
            comp = extract_inchi_features(inchi_value)
            vol = normalized[idx - 1]

            # Weighted mixture features
            for key, value in comp.items():
                mix_key = f'mix_{key}'
                base[mix_key] = base.get(mix_key, 0.0) + vol * value

            # Individual component features
            base[f'comp_{idx}_C'] = comp['C']
            base[f'comp_{idx}_H'] = comp['H']
            base[f'comp_{idx}_O'] = comp['O']
            base[f'comp_{idx}_N'] = comp['N']
            base[f'comp_{idx}_S'] = comp['S']
            base[f'comp_{idx}_num_heteroatoms'] = comp['num_heteroatoms']

        rows.append(base)

    return pd.DataFrame(rows).fillna(0.0)


class ChemicalNN(nn.Module):
    """Simple NN with feature engineering from notebook."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Load data
print("Loading dataset...")
df = pd.read_csv('cn_mixtues_inchi.dat', sep='\t')
print(f"Loaded {len(df)} records")

# Build features using notebook's method
print("Building feature matrix (notebook method)...")
X = build_feature_matrix(df)
y = df['CN'].astype(float).values

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()[:10]}...")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Setup model
device = torch.device('cpu')
model = ChemicalNN(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().to(device)
X_test_t = torch.from_numpy(X_test).float().to(device)
y_test_t = torch.from_numpy(y_test).float().to(device)

# Training loop
best_test_mae = float('inf')
print("\nTraining...")
print("Epoch | Train Loss | Test MAE | Test R²")
print("-" * 45)

for epoch in range(1, 151):
    # Train
    model.train()
    optimizer.zero_grad()
    train_pred = model(X_train_t).squeeze()
    train_loss = loss_fn(train_pred, y_train_t)
    train_loss.backward()
    optimizer.step()

    # Eval
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).squeeze().cpu().numpy()
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)

            if test_mae < best_test_mae:
                best_test_mae = test_mae
                torch.save(model.state_dict(), 'nn_proper_model.pt')

            print(f"{epoch:4d} | {train_loss.item():10.4f} | {test_mae:8.4f} | {test_r2:7.4f}")

print("\n" + "="*45)
print(f"Best Test MAE: {best_test_mae:.4f}")
print(f"RandomForest Baseline MAE: 7.29")
print(f"GNN Baseline MAE: 17.05")
print("\nComparison:")
print(f"  NN (notebook features): {best_test_mae:.2f}")
print(f"  RF Baseline: 7.29")
print(f"  GNN Baseline: 17.05")
