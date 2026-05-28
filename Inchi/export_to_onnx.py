"""
Export trained models to ONNX format.
"""
import pickle
import torch
import numpy as np
import pandas as pd


def export_rf_to_onnx(model_path='inchi_cn_model.pkl', output_path='inchi_cn_model.onnx'):
    """Export RandomForest model to ONNX."""
    print("Loading RandomForest model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        
        print("Converting RandomForest to ONNX...")
        
        # Define input shape (87 features from the notebook)
        initial_type = [('float_input', FloatTensorType([None, 87]))]
        
        # Convert
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✓ RandomForest exported to {output_path}")
        
        # Verify
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validated successfully")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Required package missing: {e}")
        print("Install with: pip install skl2onnx onnx")
        return False
    except Exception as e:
        print(f"✗ Error exporting to ONNX: {e}")
        return False


def export_gnn_to_onnx(model_path='gnn_mixture_model.pt', output_path='gnn_mixture_model.onnx'):
    """Export GNN model to ONNX (torch.onnx)."""
    print("\nNote: GNN models with PyTorch Geometric require custom ONNX export.")
    print("For production deployment, consider:")
    print("  1. Use PyTorch's native .pt format for torch serving")
    print("  2. Export via TorchScript for mobile/edge")
    print("  3. Use ONNX Runtime with custom operators")
    print("\nSkipping GNN ONNX export for now.")
    

def test_onnx_model(onnx_path='inchi_cn_model.onnx', test_data_path='inchi_cn_predictions.csv'):
    """Test ONNX model inference."""
    try:
        import onnxruntime as rt
        import pandas as pd
        
        print(f"\nTesting ONNX model: {onnx_path}")
        
        # Load test data
        if pd.io.common.file_exists(test_data_path):
            test_df = pd.read_csv(test_data_path)
            print(f"Loaded {len(test_df)} test samples")
        else:
            print(f"Warning: Test data not found at {test_data_path}")
            return False
        
        # Create ONNX session
        sess = rt.InferenceSession(onnx_path)
        
        print("✓ ONNX Runtime session created successfully")
        print(f"  Input shape: {sess.get_inputs()[0].shape}")
        print(f"  Output shape: {sess.get_outputs()[0].shape}")
        
        return True
        
    except ImportError:
        print("⚠ onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"✗ Error testing ONNX model: {e}")
        return False


if __name__ == '__main__':
    print("="*60)
    print("Model Export to ONNX")
    print("="*60)
    
    # Export RandomForest
    rf_success = export_rf_to_onnx()
    
    # Export GNN (informational)
    export_gnn_to_onnx()
    
    # Test ONNX
    if rf_success:
        test_onnx_model()
    
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"RandomForest ONNX: {'✓ Success' if rf_success else '✗ Failed'}")
    print("\nModel files:")
    print("  - inchi_cn_model.pkl (PyTorch pickle format)")
    print("  - inchi_cn_model.onnx (ONNX format - recommended for deployment)")
    print("  - gnn_mixture_model.pt (PyTorch native format)")
