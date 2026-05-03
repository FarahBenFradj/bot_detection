"""
scripts/predict.py
Run inference on one or more user records.

Usage
-----
python scripts/predict.py \
    --model   models/bot_detector_deep_mlp_best.h5 \
    --prep    models/preprocessor.pkl \
    --input   data/my_accounts.json
"""

import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tensorflow import keras
from src.preprocessing import BotDataPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--prep",   required=True)
    p.add_argument("--input",  required=True, help="JSON file: list of user records")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prep = BotDataPreprocessor()
    prep.load(args.prep)

    model = keras.models.load_model(args.model)

    with open(args.input) as f:
        users = json.load(f)

    if isinstance(users, dict):     # single user passed as object
        users = [users]

    print(f"\nRunning inference on {len(users)} account(s) …\n")
    print(f"{'UserID':<25} {'P(bot)':>8}  {'Label':>8}")
    print("-" * 45)

    for u in users:
        X    = prep.transform(u)
        prob = float(model.predict(X, verbose=0).flatten()[0])
        lbl  = "BOT ⚠️" if prob > args.threshold else "Human ✅"
        uid  = u.get("ID", u.get("id", "unknown"))
        print(f"{uid:<25} {prob:>8.4f}  {lbl:>8}")