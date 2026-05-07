import onnxruntime as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser(description="Infer composition from Cetane Number")
    parser.add_argument("--cn", type=float, required=True, help="Target Cetane Number")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "inverse_model.onnx")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    # Run Inference
    input_data = np.array([[args.cn]], dtype=np.float32)
    res = sess.run(None, {input_name: input_data})
    predictions = res[0][0]
    
    cols = [
        'n-paraffins', 'iso-paraffins', '1R-cycloparaffins', '2R-cycloparaffins',
        '3R-cycloparaffins', '1R-aromatics', '2R-aromatics', 'cycloaromatics'
    ]
    
    # Reconstruct matrix
    df_pred = pd.DataFrame(index=[f"C{c}" for c in range(5, 25)], columns=cols)
    idx = 0
    for col in cols:
        for c in range(5, 25):
            val = predictions[idx]
            df_pred.at[f"C{c}", col] = max(0.0, float(val)) # Bound zero
            idx += 1
            
    output_csv = f"inverse_cn_{args.cn}.csv"
    df_pred.to_csv(output_csv)
    print(f"Saved predicted composition to {output_csv}")

    # Plot
    df_pred.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab10')
    plt.title(f'Predicted Carbon Number Distribution for Cetane Number = {args.cn}')
    plt.xlabel('Carbon Number')
    plt.ylabel('Mass Fraction / Concentration')
    plt.legend(title='Compound Class')
    plt.tight_layout()
    output_img = f"inverse_cn_{args.cn}.png"
    plt.savefig(output_img)
    print(f"Saved visualization to {output_img}")

if __name__ == "__main__":
    main()
