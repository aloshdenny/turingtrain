import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnx import helper, numpy_helper, TensorProto

def main():
    # Configure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    # Resolve paths relative to this script
    _HERE = Path(__file__).resolve().parent  # dist_2dgc/scripts/
    _ROOT = _HERE.parent.parent              # turingtrain/
    
    data_path = _ROOT / "model_training" / "dist_2dgc" / "training_data" / "dist_2dgc_dataset.dat"
    
    # Define outputs path
    models_dir = _HERE.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading distillation dataset from: {data_path}")
    # Handle mixed types columns to prevent Performance/DtypeWarning
    df = pd.read_csv(data_path, sep='\t', comment='#', low_memory=False)
    
    # Define standard outputs
    targets = ['T05', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90', 'T95']
    df = df.dropna(subset=targets)
    
    # Filter out non-monotonic anomalies in the training set (8 samples)
    df_valid = df[targets]
    diffs = df_valid.diff(axis=1)
    monotonic_mask = ~(diffs.iloc[:, 1:] < 0).any(axis=1)
    df = df[monotonic_mask]
    print(f"Filtered out non-monotonic training rows. Remaining samples: {len(df)}")
    
    # Define standard classes map
    classes_map = {
        'n-paraffins': 'nor_par',
        'iso-paraffins': 'iso_par',
        '1R-cycloparaffins': 'mon_nap',
        '2R-cycloparaffins': 'di_nap',
        '3R-cycloparaffins': 'tri_nap',
        '1R-aromatics': 'mon_aro',
        '2R-aromatics': 'di_aro',
        'cycloaromatics': 'nap_aro',
        'olefins': 'olef',
        'synthetic-oxygenates': 'syn_oxy',
        'antioxidant-oxygenates': 'ant_oxy',
        'dienes': 'dien',
        'indenes': 'inde'
    }
    
    canonical_classes = sorted(list(set(classes_map.values())))
    
    # Construct standardized feature columns template: 13 classes x C1-C30 (390 features)
    feature_cols = []
    for cls in canonical_classes:
        for c in range(1, 31):
            col = f"2dgc_{cls}_c{c:02d}"
            feature_cols.append(col)
            # Pad column with 0 if it's missing in dataset
            if col not in df.columns:
                df[col] = 0.0
                
    # Prepare training features X (N, 390) and target Y (N, 11)
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    Y_raw = df[targets].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    
    # Compute target deltas for training:
    # d0 = T05
    # d1 = T10 - T05
    # d2 = T20 - T10, etc.
    Y_deltas = np.zeros_like(Y_raw)
    Y_deltas[:, 0] = Y_raw[:, 0]
    for idx in range(1, 11):
        Y_deltas[:, idx] = Y_raw[:, idx] - Y_raw[:, idx - 1]
    
    print(f"Dataset Loaded. Samples count: {len(X_raw)}")
    print(f"X shape: {X_raw.shape}, Y_deltas shape: {Y_deltas.shape}")
    
    # 1. Train Forward Model (Ridge regression predicting deltas on raw inputs)
    # Alphas optimized for raw composition inputs:
    # - Index 0 (T05 starting temperature) uses alpha=50.0 for stability against OOD scaling explosions
    # - Indices 1-10 (deltas) use alpha=10.0 to prevent cumulative shape explosions on out-of-distribution compositions
    alphas = np.array([50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    forward_model = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('regressor', Ridge(alpha=alphas, random_state=42))
    ])
    print("Training Forward Model on temperature deltas (raw inputs)...")
    forward_model.fit(X_raw, Y_deltas)
    
    # 2. Train Inverse Model (Distillation Curve -> Composition)
    # Alpha chosen from cross-validation: 1.0
    inverse_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
    print("Training Inverse Model (Distillation Curve -> Composition)...")
    inverse_model.fit(Y_raw, X_raw)
    
    # --- Export to ONNX ---
    print("Converting models to ONNX...")
    
    # Convert standard Ridge pipeline predicting deltas to ONNX
    initial_type_fwd = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    onx_fwd = convert_sklearn(forward_model, initial_types=initial_type_fwd)
    
    # Inject ONNX nodes to enforce monotonicity post-regressor
    raw_output_name = onx_fwd.graph.output[0].name
    raw_graph = onx_fwd.graph
    
    # Define constant U (11x11 upper triangular matrix of ones) to compute the forward cumulative sum
    U_matrix = np.triu(np.ones((11, 11), dtype=np.float32))
    U_tensor = numpy_helper.from_array(U_matrix, name="U_matrix")
    raw_graph.initializer.append(U_tensor)
    
    # Slicing parameters to separate d0 from d_rest
    starts_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts_0")
    ends_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends_1")
    starts_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts_1")
    ends_11 = numpy_helper.from_array(np.array([11], dtype=np.int64), name="ends_11")
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_1")
    raw_graph.initializer.extend([starts_0, ends_1, starts_1, ends_11, axes_1])
    
    # ONNX Nodes:
    # 1. Slice d0
    node_slice_d0 = helper.make_node(
        'Slice',
        inputs=[raw_output_name, 'starts_0', 'ends_1', 'axes_1'],
        outputs=['d0'],
        name='slice_d0'
    )
    # 2. Slice d_rest (indices 1 to 10)
    node_slice_drest = helper.make_node(
        'Slice',
        inputs=[raw_output_name, 'starts_1', 'ends_11', 'axes_1'],
        outputs=['d_rest'],
        name='slice_d_rest'
    )
    # 3. Clip rest of deltas using Relu to be non-negative
    node_relu = helper.make_node(
        'Relu',
        inputs=['d_rest'],
        outputs=['d_rest_clipped'],
        name='relu_drest'
    )
    # 4. Concatenate d0 and d_rest_clipped
    node_concat = helper.make_node(
        'Concat',
        inputs=['d0', 'd_rest_clipped'],
        outputs=['clipped_deltas'],
        axis=1,
        name='concat_deltas'
    )
    # 5. Multiply by U to do the cumulative sum: T_i = sum_{j=0}^i d_j
    node_matmul = helper.make_node(
        'MatMul',
        inputs=['clipped_deltas', 'U_matrix'],
        outputs=['monotonic_temps'],
        name='matmul_U'
    )
    
    raw_graph.node.extend([node_slice_d0, node_slice_drest, node_relu, node_concat, node_matmul])
    
    # Update graph output to be the monotonic_temps
    new_output = helper.make_tensor_value_info(
        'monotonic_temps',
        TensorProto.FLOAT,
        [None, 11]
    )
    raw_graph.output.pop()
    raw_graph.output.append(new_output)
    
    # Save the modified ONNX model to both directories
    onnx.checker.check_model(onx_fwd)
    
    fwd_path = models_dir / "distillation_model.onnx"
    with open(fwd_path, "wb") as f:
        f.write(onx_fwd.SerializeToString())
    print(f"✓ Exported Monotonic Forward Model to: {fwd_path}")
        
    # Inverse Model ONNX Conversion
    initial_type_inv = [('float_input', FloatTensorType([None, len(targets)]))]
    onx_inv = convert_sklearn(inverse_model, initial_types=initial_type_inv)
    
    inv_path = models_dir / "inverse_model.onnx"
    with open(inv_path, "wb") as f:
        f.write(onx_inv.SerializeToString())
    print(f"✓ Exported Inverse Model to: {inv_path}")
    
    print("✓ All distillation models trained, compiled, and exported successfully.")

if __name__ == "__main__":
    main()

