"""
scripts/predict.py
Run inference on one or more user records.

Usage
-----
python scripts/predict.py --model models/bot_detector_deep_mlp_best.h5 --prep models/preprocessor.pkl --input data/cresci17.json
python scripts/predict.py --model models/bot_detector_transformer_best.h5 --prep models/preprocessor.pkl --input data/cresci17.json --limit 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tensorflow import keras
from src.preprocessing import BotDataPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True,  help="Path to .h5 model file")
    p.add_argument("--prep",      required=True,  help="Path to preprocessor.pkl")
    p.add_argument("--input",     required=True,  help="JSON file: list of user records")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit",     type=int,   default=None, help="Max users to process")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load preprocessor ─────────────────────────────────────────────────────
    prep = BotDataPreprocessor()
    prep.load(args.prep)

    # ── Load model ────────────────────────────────────────────────────────────
    model = keras.models.load_model(args.model)
    print(f"✅  Model loaded     : {args.model}")
    print(f"✅  Preprocessor     : {args.prep}")

    # ── Load users ────────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        users = json.load(f)

    if isinstance(users, dict):
        users = [users]

    if args.limit:
        users = users[:args.limit]

    print(f"\n📂  Running inference on {len(users)} account(s) …\n")
    print(f"{'UserID':<25} {'P(bot)':>8}  {'True Label':>11}  {'Prediction':>12}")
    print("-" * 65)

    bots = humans = correct = 0

    for u in users:
        # Use transform_single for inference on a raw user dict
        X    = prep.transform_single(u)
        prob = float(model.predict(X, verbose=0).flatten()[0])
        pred = "BOT ⚠️" if prob > args.threshold else "Human ✅"

        # Show true label if available
        true_label = u.get("label", None)
        if true_label is not None:
            true_str = "Bot" if str(true_label) == "1" else "Human"
            is_correct = (str(true_label) == "1") == (prob > args.threshold)
            correct += int(is_correct)
            mark = "✓" if is_correct else "✗"
        else:
            true_str = "N/A"
            mark = ""

        uid = u.get("ID", u.get("id", "unknown"))
        print(f"{str(uid):<25} {prob:>8.4f}  {true_str:>11}  {pred:>12}  {mark}")

        if prob > args.threshold:
            bots += 1
        else:
            humans += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 65)
    print(f"\n📊  Summary:")
    print(f"   Total   : {len(users)}")
    print(f"   Bots    : {bots}  ({bots/len(users)*100:.1f}%)")
    print(f"   Humans  : {humans}  ({humans/len(users)*100:.1f}%)")
    if users[0].get("label") is not None:
        print(f"   Accuracy: {correct/len(users)*100:.2f}%")