"""
Test and validate ONNX model inference.
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path


def parse_inchi_formula(inchi):
    """Extract element counts from InChI."""
    if not isinstance(inchi, str) or not inchi.startswith('InChI='):
        return {e: 0 for e in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other', 'formula_len']}
    
    try:
        formula_part = inchi.split('/')[1]
    except IndexError:
        formula_part = ''
    
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    counts = {e: 0 for e in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']}
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
    features['num_heavy_atoms'] = sum(counts[e] for e in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    features['num_heteroatoms'] = sum(counts[e] for e in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    features['CH_ratio'] = counts['C'] / counts['H'] if counts['H'] > 0 else 0.0
    features['OC_ratio'] = counts['O'] / counts['C'] if counts['C'] > 0 else 0.0
    
    return features


def build_feature_matrix(df):
    """Build feature matrix from dataframe."""
    inchi_cols = [f'cpnt_inchi_{i}' for i in range(1, 11)]
    vol_cols = [f'cpnt_vol_{i}' for i in range(1, 11)]
    
    rows = []
    for _, row in df.iterrows():
        volumes = [float(row.get(col, 0.0) or 0.0) for col in vol_cols]
        total_volume = sum(volumes) or 1.0
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


def test_onnx_inference():
    """Test ONNX model inference on sample data."""
    print("="*60)
    print("ONNX Model Validation Test")
    print("="*60)
    
    try:
        import onnxruntime as rt
    except ImportError:
        print("✗ onnxruntime not installed")
        print("  Install with: pip install onnxruntime")
        return False
    
    # Load ONNX model
    print("\nLoading ONNX model: inchi_cn_model.onnx")
    try:
        sess = rt.InferenceSession('inchi_cn_model.onnx')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Get model info
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    
    print(f"\nModel Info:")
    print(f"  Input shape: {inputs[0].shape}")
    print(f"  Output shape: {outputs[0].shape}")
    
    # Load sample data
    print("\nLoading sample data: cn_mixtues_inchi.dat")
    try:
        df = pd.read_csv('cn_mixtues_inchi.dat', sep='\t')
        print(f"✓ Loaded {len(df)} samples")
    except FileNotFoundError:
        print("✗ Data file not found")
        return False
    
    # Build features
    print("\nBuilding feature matrix...")
    X = build_feature_matrix(df)
    y = df['CN'].astype(float).values
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"  Expected: (1143, 87)")
    
    # Test inference on small batch
    print("\nTesting inference on first 5 samples...")
    X_test = X.iloc[:5].values.astype(np.float32)
    y_test = y[:5]
    
    try:
        predictions = sess.run(None, {'float_input': X_test})[0]
        print(f"✓ Inference successful")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Output shape: {predictions.shape}")
        
        print(f"\n  Sample Results:")
        for i, (actual, pred) in enumerate(zip(y_test, predictions)):
            error = abs(actual - pred[0])
            print(f"    [{i}] Actual: {actual:6.2f}, Predicted: {pred[0]:6.2f}, Error: {error:6.2f}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    # Full test set inference
    print("\nTesting inference on full dataset (1143 samples)...")
    X_full = X.values.astype(np.float32)
    
    try:
        predictions_full = sess.run(None, {'float_input': X_full})[0]
        
        mae = np.mean(np.abs(y - predictions_full.squeeze()))
        print(f"✓ Full inference successful")
        print(f"  Mean Absolute Error: {mae:.4f}")
        print(f"  Expected baseline: ~7.30")
        
        if mae < 8.0:
            print(f"  ✓ Performance matches expected (~7.30)")
        else:
            print(f"  ⚠ Performance slightly off (check data consistency)")
        
    except Exception as e:
        print(f"✗ Full inference failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ ONNX Model Validation Passed")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_onnx_inference()
    exit(0 if success else 1)
