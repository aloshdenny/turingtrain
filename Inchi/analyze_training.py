"""
Analyze and summarize training metrics from terminal output.
"""
import re
import sys


def extract_metrics(output_text):
    """Extract training metrics from terminal output."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'test_mae': [],
        'test_r2': []
    }
    
    # Pattern for lines like: "  20 | 430.9215 | 13.9662 |  0.0018"
    pattern = r'(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+)'
    
    for match in re.finditer(pattern, output_text):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        mae = float(match.group(3))
        r2 = float(match.group(4))
        
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(loss)
        metrics['test_mae'].append(mae)
        metrics['test_r2'].append(r2)
    
    return metrics


def summarize_training(metrics):
    """Print summary of training."""
    if not metrics['epochs']:
        print("No metrics found")
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nTotal Epochs: {len(metrics['epochs'])}")
    
    print(f"\nInitial Metrics (Epoch {metrics['epochs'][0]}):")
    print(f"  Train Loss: {metrics['train_loss'][0]:.4f}")
    print(f"  Test MAE: {metrics['test_mae'][0]:.4f}")
    print(f"  Test R²: {metrics['test_r2'][0]:.4f}")
    
    print(f"\nFinal Metrics (Epoch {metrics['epochs'][-1]}):")
    print(f"  Train Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"  Test MAE: {metrics['test_mae'][-1]:.4f}")
    print(f"  Test R²: {metrics['test_r2'][-1]:.4f}")
    
    print(f"\nBest Test MAE: {min(metrics['test_mae']):.4f}")
    print(f"Best Test R²: {max(metrics['test_r2']):.4f}")
    
    # Check for convergence
    recent_mae = metrics['test_mae'][-5:]
    mae_change = recent_mae[-1] - recent_mae[0]
    print(f"\nLast 5 epochs MAE change: {mae_change:.4f}")
    
    if abs(mae_change) < 0.1:
        print("⚠️  Model appears to have converged (minimal improvement)")
    else:
        print("✓ Model still improving")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            output = f.read()
    else:
        output = sys.stdin.read()
    
    metrics = extract_metrics(output)
    summarize_training(metrics)
