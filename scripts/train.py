"""
scripts/train.py
CLI entry point — trains one or all bot-detection models.

Usage
-----
# Train default model on sample data
python scripts/train.py

# Train specific model with custom hyper-params
python scripts/train.py --model transformer --epochs 50 --batch_size 32

# Train on full TwiBot-20 dataset
python scripts/train.py --data_dir data/ --model deep_mlp --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config import (
    SAMPLE_FILE, TRAIN_FILE, DEV_FILE, TEST_FILE,
    DEFAULT_MODEL, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MODEL_TYPES, RANDOM_SEED,
)
from src.preprocessing import BotDataPreprocessor
from src.trainer import BotDetectionTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Bot Detection models")
    p.add_argument("--model",      default=DEFAULT_MODEL, choices=MODEL_TYPES)
    p.add_argument("--all",        action="store_true",
                   help="Train all model types sequentially")
    p.add_argument("--data_dir",   default=None,
                   help="Directory containing train.json, dev.json, test.json. "
                        "Defaults to sample data.")
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LEARNING_RATE)
    return p.parse_args()


def load_data(data_dir=None):
    """Load and preprocess data. Falls back to sample if data_dir not given."""
    preprocessor = BotDataPreprocessor()

    if data_dir:
        data_dir = Path(data_dir)
        print("Loading full TwiBot-20 dataset …")
        df_train   = preprocessor.process_file(str(data_dir / "train.json"))
        df_val     = preprocessor.process_file(str(data_dir / "dev.json"))
        df_test    = preprocessor.process_file(str(data_dir / "test.json"))
        df_all     = pd.concat([df_train, df_val, df_test], ignore_index=True)
        # Remove unlabelled rows (support set)
        df_all = df_all[df_all["label"] != -1].reset_index(drop=True)
    else:
        print(f"No --data_dir supplied — using sample: {SAMPLE_FILE}")
        df_all = preprocessor.process_file(str(SAMPLE_FILE))
        df_all = df_all[df_all["label"] != -1].reset_index(drop=True)

        if len(df_all) == 0:
            # Sample has no labels — assign synthetic ones so the pipeline can
            # be tested end-to-end. DO NOT use these for real evaluation.
            print("[INFO] Sample has no labels. Assigning synthetic labels "
                  "(alternating 0/1) for pipeline testing only.")
            df_all = preprocessor.process_file(str(SAMPLE_FILE))
            df_all["label"] = [i % 2 for i in range(len(df_all))]
            print(f"[INFO] {len(df_all)} users loaded with synthetic labels.")
        else:
            print(f"[INFO] {len(df_all)} labelled users found in sample.")

    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = \
        preprocessor.prepare_splits(df_all)

    preprocessor.save("models/preprocessor.pkl")

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Features: {len(feat_names)}")
    print(f"Bot ratio (train): {y_train.mean():.1%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_one(model_type, X_train, X_val, X_test, y_train, y_val, y_test,
            epochs, batch_size, lr):
    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()}")
    print(f"{'='*60}")

    trainer = BotDetectionTrainer(model_type)
    trainer.build(input_dim=X_train.shape[1], lr=lr)
    trainer.train(X_train, y_train, X_val, y_val,
                  epochs=epochs, batch_size=batch_size)
    trainer.evaluate(X_test, y_test)
    trainer.plot_training()
    trainer.plot_evaluation()
    trainer.save_metrics()

    return trainer.results


if __name__ == "__main__":
    args  = parse_args()
    data  = load_data(args.data_dir)
    X_tr, X_v, X_te, y_tr, y_v, y_te = data

    models_to_train = MODEL_TYPES if args.all else [args.model]
    all_results = {}

    for mt in models_to_train:
        res = run_one(mt, X_tr, X_v, X_te, y_tr, y_v, y_te,
                      args.epochs, args.batch_size, args.lr)
        all_results[mt] = res

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 50)
    for mt, r in all_results.items():
        print(f"{mt:<15} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f}")