"""
scripts/generate_sample.py
Extract a small balanced sample from cresci17.json and save it as
data/sample_accounts.json — small enough to commit to GitHub and used
by the Streamlit app for the "Random Human / Random Bot" buttons.

Usage
-----
python scripts/generate_sample.py
python scripts/generate_sample.py --input data/cresci17.json --n 100 --output data/sample_accounts.json
"""

import argparse
import json
import random
from pathlib import Path


def strip_user(user: dict) -> dict:
    """Keep only the fields the app actually needs — shrinks file size significantly."""
    profile = user.get("profile", {})
    keep_profile = {
        "screen_name":           profile.get("screen_name", ""),
        "name":                  profile.get("name", ""),
        "description":           profile.get("description", ""),
        "location":              profile.get("location", ""),
        "url":                   profile.get("url", ""),
        "created_at":            profile.get("created_at", ""),
        "followers_count":       int(profile.get("followers_count") or 0),
        "friends_count":         int(profile.get("friends_count") or 0),
        "statuses_count":        int(profile.get("statuses_count") or 0),
        "favourites_count":      int(profile.get("favourites_count") or 0),
        "listed_count":          int(profile.get("listed_count") or 0),
        "verified":              bool(profile.get("verified", False)),
        "default_profile":       bool(profile.get("default_profile", False)),
        "default_profile_image": bool(profile.get("default_profile_image", False)),
        "geo_enabled":           bool(profile.get("geo_enabled", False)),
    }
    return {
        "ID":      user.get("ID", ""),
        "label":   str(user.get("label", "")),
        "profile": keep_profile,
        "tweet":   [],          # tweets are large — not needed for the demo buttons
    }


def generate(input_path: str, output_path: str, n_per_class: int, seed: int):
    src = Path(input_path)
    if not src.exists():
        print(f"[ERROR] Input file not found: {src}")
        print("Run convert_cresci17.py first to produce cresci17.json")
        raise SystemExit(1)

    print(f"Loading {src} …")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    humans = [u for u in data if str(u.get("label", "")) == "0"]
    bots   = [u for u in data if str(u.get("label", "")) == "1"]

    print(f"  Found {len(humans):,} humans and {len(bots):,} bots")

    random.seed(seed)
    sampled_humans = random.sample(humans, min(n_per_class, len(humans)))
    sampled_bots   = random.sample(bots,   min(n_per_class, len(bots)))

    sample = [strip_user(u) for u in sampled_humans + sampled_bots]
    random.shuffle(sample)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    size_kb = out.stat().st_size / 1024
    print(f"\n✅  Saved {len(sampled_humans)} humans + {len(sampled_bots)} bots")
    print(f"   → {out}  ({size_kb:.1f} KB)")
    print(f"\nNext steps:")
    print(f"  git add {output_path}")
    print(f"  git commit -m \"data: add sample accounts for Streamlit demo buttons\"")
    print(f"  git push")


def parse_args():
    p = argparse.ArgumentParser(description="Generate a small sample from cresci17.json")
    p.add_argument("--input",  default="data/cresci17.json",
                   help="Source JSON file (default: data/cresci17.json)")
    p.add_argument("--output", default="data/sample_accounts.json",
                   help="Output file (default: data/sample_accounts.json)")
    p.add_argument("--n",      type=int, default=100,
                   help="Number of accounts per class (default: 100 → 200 total)")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.input, args.output, args.n, args.seed)