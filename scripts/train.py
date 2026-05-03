#!/usr/bin/env python
"""
Train bot-detection models on Cresci-17 or TwiBot-20.
Usage:
    python scripts/train.py --data_dir data
    python scripts/train.py --data_dir data --model transformer --epochs 50
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.preprocessing import BotDataPreprocessor
from src.trainer import BotDetectionTrainer

SEED = config.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def split_data(records, train_ratio=0.70, dev_ratio=0.15):
    random.seed(SEED)
    records = list(records)
    random.shuffle(records)
    n = len(records)
    t = int(n * train_ratio)
    d = int(n * (train_ratio + dev_ratio))
    return records[:t], records[t:d], records[d:]


def ensure_splits(data_dir: Path):
    """
    Si train/dev/test.json existent → les utilise.
    Sinon → cherche n'importe quel .json et le split automatiquement.
    """
    train_path = data_dir / "train.json"
    dev_path   = data_dir / "dev.json"
    test_path  = data_dir / "test.json"

    if train_path.exists() and dev_path.exists() and test_path.exists():
        print("✅  Found existing splits (train/dev/test.json).")
        return str(train_path), str(dev_path), str(test_path)

    # Cherche le premier JSON disponible (cresci17.json, etc.)
    sample_file = None
    for c in sorted(data_dir.glob("*.json")):
        if c.name not in ("train.json", "dev.json", "test.json"):
            sample_file = c
            break

    if sample_file is None:
        raise FileNotFoundError(
            f"Aucun fichier JSON trouvé dans '{data_dir}'.\n"
            "Place cresci17.json (ou train/dev/test.json) dans ce dossier."
        )

    print(f"📂  Dataset trouvé : {sample_file.name}")
    records = load_json(str(sample_file))

    if isinstance(records, dict):
        records = list(records.values())

    print(f"   Total records : {len(records)}")

    # Afficher la distribution des labels
    labels = [r.get("label", "?") for r in records]
    from collections import Counter
    print(f"   Distribution  : {dict(Counter(labels))}")

    train_r, dev_r, test_r = split_data(records)

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    t_path = splits_dir / "train.json"
    d_path = splits_dir / "dev.json"
    e_path = splits_dir / "test.json"

    save_json(train_r, str(t_path))
    save_json(dev_r,   str(d_path))
    save_json(test_r,  str(e_path))

    print(f"   → Split : {len(train_r)} train / {len(dev_r)} dev / {len(test_r)} test")
    print(f"   → Sauvegardé dans {splits_dir}/")
    return str(t_path), str(d_path), str(e_path)


def load_data(data_dir: str):
    data_dir = Path(data_dir)
    print("\nChargement du dataset …")

    train_path, dev_path, test_path = ensure_splits(data_dir)

    preprocessor = BotDataPreprocessor()

    df_train = preprocessor.process_file(train_path)
    df_dev   = preprocessor.process_file(dev_path)
    df_test  = preprocessor.process_file(test_path)

    print(f"\n   Train : {len(df_train)} rows  | bots: {df_train['label'].sum()}")
    print(f"   Dev   : {len(df_dev)} rows  | bots: {df_dev['label'].sum()}")
    print(f"   Test  : {len(df_test)} rows  | bots: {df_test['label'].sum()}")

    # Fit scaler sur train uniquement
    preprocessor.fit(df_train)

    X_train, y_train = preprocessor.transform(df_train)
    X_dev,   y_dev   = preprocessor.transform(df_dev)
    X_test,  y_test  = preprocessor.transform(df_test)

    print(f"\n   Features : {X_train.shape[1]}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_dev":   X_dev,   "y_dev":   y_dev,
        "X_test":  X_test,  "y_test":  y_test,
        "preprocessor": preprocessor,
        "n_features": X_train.shape[1],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--model",       default=config.DEFAULT_MODEL,
                   choices=["mlp", "deep_mlp", "attention", "transformer"])
    p.add_argument("--epochs",      type=int, default=config.EPOCHS)
    p.add_argument("--batch_size",  type=int, default=config.BATCH_SIZE)
    p.add_argument("--output_dir",  default=str(config.MODELS_DIR))
    p.add_argument("--results_dir", default=str(config.RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    data = load_data(args.data_dir)

    output_dir  = Path(args.output_dir);  output_dir.mkdir(exist_ok=True)
    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)

    print(f"\n🏋️  Training '{args.model}'  |  epochs={args.epochs}  |  batch={args.batch_size}\n")

    trainer = BotDetectionTrainer(model_type=args.model)
    trainer.build(input_dim=data["n_features"], lr=config.LEARNING_RATE)

    trainer.train(
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_dev"],     y_val=data["y_dev"],
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    metrics = trainer.evaluate(data["X_test"], data["y_test"])

    print("\n📊  Résultats test :")
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"   {k:15s}: {metrics[k]:.4f}")

    trainer.plot_training(out_dir=str(results_dir))
    trainer.plot_evaluation(out_dir=str(results_dir))
    trainer.save_metrics(out_dir=str(results_dir))

    preprocessor_path = output_dir / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(data["preprocessor"], f)
    print(f"\n💾  Preprocessor sauvegardé → {preprocessor_path}")


if __name__ == "__main__":
    main()