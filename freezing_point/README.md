# Freezing Point 2DGC Model

Predicts the freezing point (`T_freeze_K`) of petroleum-derived fuel blends from their
2D-GC hydrocarbon-class bin mass-fraction distributions.

---

## Directory Structure

```
freezing_point/
├── README.md
├── scripts/
│   └── train_fp_2dgc_export.py   # Training & ONNX export script
└── models/
    └── fp_2dgc/
        ├── fp_2dgc_seed{0..4}.joblib   # 5-seed ensemble (sklearn)
        ├── fp_2dgc_seed0.onnx          # ONNX export — seed 0 (2.4 MB)
        ├── fp_2dgc_meta.json           # Training metadata & metrics
        └── fp_2dgc_parity.png          # Validation parity plot
```

---

## Input Features (189 bins)

| Hydrocarbon class    | Carbon range | Bins |
|----------------------|-------------|------|
| n-paraffin           | C1–C30      | 30   |
| iso-paraffin         | C4–C30      | 27   |
| mono-naphthene       | C5–C30      | 26   |
| di-naphthene         | C10–C30     | 21   |
| tri-naphthene        | C14–C30     | 17   |
| mono-aromatic        | C6–C30      | 25   |
| naphtheno-aromatic   | C9–C30      | 22   |
| di-aromatic          | C10–C30     | 21   |

Column naming convention: `w_{class}_C{n}` (mass fraction, dimensionless, sum ≤ 1).

**Target:** `T_freeze_K` — freezing point in Kelvin (80–353 K in training data).

---

## Model

- **Algorithm:** `HistGradientBoostingRegressor` (sklearn) — 5-seed ensemble
- **Training data:** 149,418 samples from `model_training/freezing_point_2dgc/fp_2dgc_selfies_train.dat`
- **Validation split:** 90/10 random per seed

| Metric | Value |
|--------|-------|
| 5-Fold CV MAE | 4.30 ± 0.03 K |
| Ensemble MAE  | **3.63 K** |
| Ensemble R²   | **0.9748** |
| Within ±5 K   | 76.1% |

---

## Training

See `freezing_point/scripts/train_fp_2dgc_export.py`. Run from the repo root:

```bash
conda activate intensors
python freezing_point/scripts/train_fp_2dgc_export.py \
    --input   model_training/freezing_point_2dgc/fp_2dgc_selfies_train.dat \
    --out_dir freezing_point/models/fp_2dgc
```

Outputs all 5 seed checkpoints, ONNX export, and metadata JSON to `--out_dir`.

---

## Inference (Python)

```python
import joblib, numpy as np

# Load ensemble
models = [joblib.load(f"freezing_point/models/fp_2dgc/fp_2dgc_seed{i}.joblib") for i in range(5)]

# X: np.ndarray of shape (N, 189) — columns must match feature_cols in fp_2dgc_meta.json
T_freeze_K = np.mean([m.predict(X) for m in models], axis=0)
```

Or via ONNX (seed 0 only):

```python
import onnxruntime as ort, numpy as np

sess = ort.InferenceSession("freezing_point/models/fp_2dgc/fp_2dgc_seed0.onnx")
T_freeze_K = sess.run(None, {"float_input": X.astype(np.float32)})[0]
```
