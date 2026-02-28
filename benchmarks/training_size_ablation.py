"""
Training Data Size Ablation Study
==================================
Analyze how FastSHAP fidelity changes with different training set sizes.
"""

import numpy as np
import sys
import torch
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, 'src')

from data.load_datasets import load_ulb_dataset
from data.preprocessing import FraudDataPreprocessor
from models.train_models import train_xgboost, load_trained_xgboost
from explainers.fastshap_implementation import FastSHAPTrainer, create_fastshap_from_model


def run_training_size_ablation(
    training_sizes=[100, 500, 1000, 5000, 10000, 20000],
    output_dir="thesis_results"
):
    """
    Train FastSHAP with different training set sizes and measure fidelity.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("TRAINING DATA SIZE ABLATION STUDY")
    logger.info("="*70)
    
    # Load full dataset
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = load_ulb_dataset()
    
    # Preprocess
    preprocessor = FraudDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    feature_cols = [c for c in X_train_raw.columns if c not in ['Class', 'Time']]
    n_features = len(feature_cols)
    
    # Load or train XGBoost
    try:
        xgb_model = load_trained_xgboost()
        logger.info("Loaded existing XGBoost model")
    except:
        logger.info("Training XGBoost model...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    results = {}
    
    for size in training_sizes:
        if size > len(X_train):
            logger.warning(f"Skipping size {size} (max available: {len(X_train)})")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training with {size} samples...")
        logger.info(f"{'='*70}")
        
        # Subsample training data
        indices = np.random.choice(len(X_train), size=min(size, len(X_train)), replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        try:
            # Train FastSHAP
            trainer = FastSHAPTrainer(
                n_features=n_features,
                hidden_layers=[256, 128, 64],
                lr=1e-3,
                batch_size=min(64, size // 4) if size > 256 else 32
            )
            
            # Train for fewer epochs with small data
            epochs = min(30, max(10, size // 500))
            
            history = trainer.fit(
                xgb_model, X_train_subset, X_val,
                epochs=epochs,
                validation_every=5
            )
            
            # Create explainer and measure fidelity
            fastshap = create_fastshap_from_model(
                trainer.surrogate, X_train_subset, feature_cols
            )
            
            fidelity = fastshap.compute_fidelity(xgb_model, X_val[:1000])
            
            results[f"n_{size}"] = {
                "training_size": size,
                "epochs_trained": epochs,
                "final_train_loss": float(history['train_losses'][-1]) if history['train_losses'] else None,
                "final_val_loss": float(history['val_losses'][-1]) if history['val_losses'] else None,
                "fidelity": float(fidelity),
                "convergence_epoch": int(history.get('convergence_epoch', epochs))
            }
            
            logger.info(f"  Fidelity: {fidelity:.4f}")
            logger.info(f"  Final val loss: {results[f'n_{size}']['final_val_loss']:.6f}" if results[f'n_{size}']['final_val_loss'] else "  N/A")
            
        except Exception as e:
            logger.error(f"Failed for size {size}: {e}")
            results[f"n_{size}"] = {
                "training_size": size,
                "error": str(e)
            }
    
    # Save results
    output_file = f"{output_dir}/training_size_ablation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING SIZE ABLATION SUMMARY")
    logger.info(f"{'='*70}")
    
    for key, data in results.items():
        if 'fidelity' in data:
            logger.info(f"Size {data['training_size']:>6}: Fidelity={data['fidelity']:.4f}")
    
    logger.info(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, 
                        default=[100, 500, 1000, 5000, 10000, 20000],
                        help='Training set sizes to test')
    args = parser.parse_args()
    
    results = run_training_size_ablation(args.sizes)
