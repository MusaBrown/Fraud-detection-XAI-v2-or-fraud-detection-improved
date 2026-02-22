# ULB Credit Card Fraud Detection - Real-Time XAI Demo

This demo uses the [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Dataset Information

- **Source**: Machine Learning Group, Université Libre de Bruxelles
- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1-V28)
- **Fraud Rate**: 0.172% (492 frauds out of 284,807 transactions)
- **File Size**: ~150 MB

## Setup Instructions

### Option 1: Using Kaggle API (Automated)

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Get API credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)

3. **Download dataset**:
   ```bash
   python download_ulb_data.py
   ```

### Option 2: Manual Download

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download"
3. Extract `creditcard.csv` to `data/raw/` folder

## Running the Demo

Once the data is downloaded:

```bash
python demo_ulb_creditcard.py
```

## Expected Output

The demo will:

1. **Load and validate** the ULB dataset
2. **Preprocess** the data (standardize Time and Amount)
3. **Train** two models:
   - Random Forest
   - Gradient Boosting (primary model)
4. **Evaluate** model performance (AUC, F1, Precision, Recall)
5. **Train** explainers:
   - TreeSHAP (approximation for comparison)
   - FastSHAP (fast neural surrogate)
6. **Benchmark** latency (P50, P95, P99)
7. **Evaluate** fidelity (Pearson correlation with TreeSHAP)
8. **Show** example explanations

## Target Metrics

| Metric | Target | Expected |
|--------|--------|----------|
| P95 Latency | < 50ms | ~5-10ms |
| P99 Latency | < 100ms | ~10-20ms |
| AUC-ROC | > 0.98 | ~0.97-0.98 |
| F1 Score | > 0.95 | ~0.80-0.85* |

*Note: F1 is lower due to extreme class imbalance (0.172% fraud rate). The model achieves high precision but lower recall due to many false negatives.

## Output Files

- `ulb_demo_results.json` - Complete results in JSON format

## Troubleshooting

### Dataset not found
```
FileNotFoundError: Dataset not found at data/raw/creditcard.csv
```

**Solution**: Download the dataset first using one of the methods above.

### Kaggle authentication error
```
401 - Unauthorized
```

**Solution**: Check your Kaggle API credentials in `~/.kaggle/kaggle.json`

### Memory error
The dataset loads entirely into memory (~150 MB). If you encounter memory issues:
- Close other applications
- Use a machine with at least 4GB RAM
