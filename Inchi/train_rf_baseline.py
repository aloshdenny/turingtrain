"""
Quickly train and save the RandomForest baseline model
(matches notebook's approach for reference).
"""
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.ensemble import RandomForestRegressor
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
    """Extract full feature set from InChI."""
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

    features['num_heavy_atoms'] = sum(
        counts[e] for e in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    )

    features['num_heteroatoms'] = sum(
        counts[e] for e in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    )

    features['CH_ratio'] = counts['C'] / counts['H'] if counts['H'] > 0 else 0.0
    features['OC_ratio'] = counts['O'] / counts['C'] if counts['C'] > 0 else 0.0

    return features


def build_feature_matrix(df):
    """Build feature matrix exactly like notebook."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]

    rows = []

    for _, row in df.iterrows():
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
            'mix_total_volume': total_volume,
            'mix_nonzero_components': sum(1 for v in normalized if v > 0),
            'mix_max_volume': max(normalized),
            'mix_min_nonzero_volume': min((v for v in normalized if v > 0), default=0.0),
            'mix_volume_entropy': -sum(v * np.log(v) for v in normalized if v > 0),
        }

        for idx, inchi_col in enumerate(inchi_cols, start=1):
            inchi_value = row.get(inchi_col, '')
            comp = extract_inchi_features(inchi_value)
            vol = normalized[idx - 1]

            for key, value in comp.items():
                mix_key = f'mix_{key}'
                base[mix_key] = base.get(mix_key, 0.0) + vol * value

            base[f'comp_{idx}_C'] = comp['C']
            base[f'comp_{idx}_H'] = comp['H']
            base[f'comp_{idx}_O'] = comp['O']
            base[f'comp_{idx}_N'] = comp['N']
            base[f'comp_{idx}_S'] = comp['S']
            base[f'comp_{idx}_num_heteroatoms'] = comp['num_heteroatoms']

        rows.append(base)

    return pd.DataFrame(rows).fillna(0.0)


print("Loading data...")
df = pd.read_csv('cn_mixtues_inchi.dat', sep='\t')
print(f"Loaded {len(df)} records")

print("Building feature matrix...")
X = build_feature_matrix(df)
y = df['CN'].astype(float).values

print(f"Feature matrix: {X.shape}")

print("Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("\nTraining RandomForest (250 estimators)...")
model = RandomForestRegressor(
    n_estimators=250,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

print("\nEvaluating...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n" + "="*50)
print(f"Train MAE: {train_mae:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")
print("="*50)

print("\nSaving model...")
with open('inchi_cn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Model saved to inchi_cn_model.pkl")

# Also save feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})

importance_df = importance_df.sort_values(by='importance', ascending=False)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string())

importance_df.to_csv('rf_feature_importance.csv', index=False)
print("\n✓ Feature importance saved to rf_feature_importance.csv")
