# MLP Results

## Architecture
- Input: 20 extracted features
- Features: mean_I, std_I, mean_Q, std_Q, skew_I, skew_Q, mean_amp, std_amp, signal_power, PAPR, inst_freq_mean, inst_freq_std, inst_freq_median, inst_freq_IQR, fft_peak, fft_mean, fft_std, spectral_centroid, IQ_corr, kurtosis_I
- Layers: Linear(128) → BN → ReLU → Drop(0.3) → Linear(64) → BN → ReLU → Drop(0.3) → Linear(32) → ReLU → Drop(0.2) → Linear(5)
- Optimiser: AdamW (lr=0.001), cosine annealing, 80 epochs
- Balanced sampling via WeightedRandomSampler

## Results

| Evaluation | Accuracy | Macro F1 |
|------------|----------|----------|
| Chance | 0.2000 | 0.2000 |
| Stratified 5-fold | 0.2564 | 0.2429 |
| Cross-session | 0.2381 | 0.2210 |

## Interpretation
The MLP on extracted features shows similar performance to classical ML
baselines and the CNN, confirming that the lack of discrimination is
a data/signal issue, not a model capacity issue.
