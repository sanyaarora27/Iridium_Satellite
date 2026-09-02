# CNN Raw-IQ RF Fingerprinting Results

## Model
- Architecture: 3-block 1D-CNN (Conv1D → BatchNorm → ReLU → Pool) × 3, GAP, Dense
- Input: raw IQ burst (2 × 11000), per-channel normalised
- Block 1: 32 filters, kernel=64, stride=4, MaxPool(4)
- Block 2: 64 filters, kernel=16, stride=2, MaxPool(2)
- Block 3: 128 filters, kernel=8, GlobalAvgPool
- Classifier: Dropout(0.5) → Dense(64) → Dropout(0.3) → Dense(5)
- Optimiser: AdamW (lr=0.001, weight_decay=0.0001)
- Scheduler: CosineAnnealing over 60 epochs
- Augmentation: Gaussian noise (3%), amplitude scaling (±10%), circular shift (±500)

## Dataset
- Source: SatIQ/Zenodo, segments 000–004
- Satellites: [51, 85, 87, 92, 109]
- Total bursts: 6162 (after filtering corrupt timestamps)
- 3 temporal sessions identified from timestamp clustering
- Evaluation: 3-fold GroupKFold by session (cross-session generalisation)

## Results

| Fold | Test Session | Train N | Test N | Accuracy | Macro F1 |
|------|-------------|---------|--------|----------|----------|
| 1 | A | 3896 | 2266 | 0.2162 | 0.0711 |
| 2 | C | 4090 | 2072 | 0.2432 | 0.0792 |
| 3 | B | 4338 | 1824 | 0.2785 | 0.1802 |
| **Mean** | **All** | | | **0.2438** | **0.1863** |

## Authentication Metrics

| Satellite | Pd (recall) | Pfa | FAR |
|-----------|-------------|-----|-----|
| Sat 51 | 0.0000 | 0.0000 | 0.0000 |
| Sat 85 | 0.5321 | 0.4599 | 0.7679 |
| Sat 87 | 0.0252 | 0.0266 | 0.8086 |
| Sat 92 | 0.3947 | 0.3238 | 0.7574 |
| Sat 109 | 0.2416 | 0.1410 | 0.7105 |

## Interpretation

Cross-session evaluation tests whether the CNN learns transmitter-specific
hardware fingerprints rather than session/channel-specific features. If accuracy
is well above chance (20%) but below within-session performance, it suggests the
model captures some genuine hardware signal but channel variation remains a
significant confound — consistent with prior CrossRF (2025) findings.

## Figures
- `cnn_loss_curves.png` – training/validation loss and accuracy per fold
- `cnn_confusion_matrix.png` – aggregate confusion matrix
- `cnn_roc_curves.png` – per-satellite ROC curves
