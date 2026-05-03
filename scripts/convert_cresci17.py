"""
scripts/convert_cresci17.py
Convert the Cresci-2017 dataset into a single JSON file compatible with
this project's preprocessing pipeline.

This version handles the Indiana University Bot Repository format where
each category is a single .csv file that is actually a ZIP archive.

Dataset structure (what you downloaded)
----------------------------------------
datasets_full.csv/
├── genuine_accounts.csv       <- ZIP archive  -> label = 0  (human)
├── social_spambots_1.csv      <- ZIP archive  -> label = 1  (bot)
├── social_spambots_2.csv      <- ZIP archive  -> label = 1
├── social_spambots_3.csv      <- ZIP archive  -> label = 1
├── traditional_spambots_1.csv <- ZIP archive  -> label = 1
├── traditional_spambots_2.csv <- ZIP archive  -> label = 1
├── traditional_spambots_3.csv <- ZIP archive  -> label = 1
├── traditional_spambots_4.csv <- ZIP archive  -> label = 1
├── fake_followers.csv         <- ZIP archive  -> label = 1
└── crowdflower_results.csv    <- skipped (annotation metadata)

Each ZIP contains:
  users.csv   -- one row per Twitter user (profile fields)
  tweets.csv  -- one row per tweet (linked by user_id)

Usage
-----
python scripts/convert_cresci17.py --input_dir "path/to/datasets_full.csv" --output data/cresci17.json
python scripts/train.py --data_file data/cresci17.json --model transformer --epochs 100
"""

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


# Label mapping by filename stem
HUMAN_FILES = {"genuine_accounts"}
BOT_FILES = {
    "social_spambots_1", "social_spambots_2", "social_spambots_3",
    "traditional_spambots_1", "traditional_spambots_2",
    "traditional_spambots_3", "traditional_spambots_4",
    "fake_followers",
}
SKIP_FILES = {"crowdflower_results", "readme", "read.me"}


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode CSV")


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val)) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def _safe_str(val, default="") -> str:
    return str(val).strip() if pd.notna(val) else default


def _tweet_to_dict(row: pd.Series) -> dict:
    out = {"text": _safe_str(row.get("text", ""))}
    for src, dst, cast in [
        ("created_at",     "created_at",     _safe_str),
        ("retweet_count",  "retweet_count",  _safe_int),
        ("favorite_count", "favorite_count", _safe_int),
        ("source",         "source",         _safe_str),
    ]:
        if src in row.index:
            out[dst] = cast(row[src])
    return out


_STR_COLS  = ["screen_name", "name", "description", "location", "url", "created_at"]
_INT_COLS  = ["followers_count", "friends_count", "statuses_count",
              "favourites_count", "listed_count"]
_BOOL_COLS = ["verified", "default_profile", "default_profile_image",
              "protected", "geo_enabled"]


def _user_to_record(row: pd.Series, tweets: list, label: int) -> dict:
    uid_col = next((c for c in ("id", "user_id", "userid") if c in row.index), None)
    uid = _safe_str(row[uid_col]) if uid_col else ""

    profile: dict = {}
    for col in _STR_COLS:
        if col in row.index:
            profile[col] = _safe_str(row[col])
    for col in _INT_COLS:
        if col in row.index:
            profile[col] = _safe_int(row[col])
    for col in _BOOL_COLS:
        if col in row.index:
            v = row[col]
            if isinstance(v, bool):
                profile[col] = v
            elif isinstance(v, str):
                profile[col] = v.strip().lower() == "true"
            else:
                profile[col] = bool(_safe_int(v))

    return {"ID": uid, "profile": profile, "tweet": tweets, "label": str(label)}


def _build_from_single_csv(df: pd.DataFrame, label: int, max_tweets: int) -> list:
    id_col = next((c for c in ("id", "user_id", "userid") if c in df.columns), df.columns[0])
    records = []
    for _, row in df.iterrows():
        tweets = []
        if "text" in df.columns and pd.notna(row.get("text", None)):
            tweets = [{"text": _safe_str(row["text"])}]
        records.append(_user_to_record(row, tweets[:max_tweets], label))
    print(f"    -> {len(records):,} records (single-CSV mode)")
    return records


def _process_zip_csv(path: Path, label: int, max_tweets: int) -> list:
    tag = "human" if label == 0 else "bot"
    print(f"  Opening {path.name}  (label={tag}) ...")

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        print(f"    Not a ZIP -- trying as plain CSV ...")
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
            print(f"    Plain CSV: {len(df):,} rows")
            return _build_from_single_csv(df, label, max_tweets)
        except Exception as e:
            print(f"    [SKIP] Could not read: {e}")
            return []

    names = zf.namelist()
    print(f"    Contents: {names}")

    users_entry  = next((n for n in names if n.lower().endswith("users.csv")),  None)
    tweets_entry = next((n for n in names if n.lower().endswith("tweets.csv")), None)

    if users_entry is None:
        csv_entries = [n for n in names if n.lower().endswith(".csv")]
        if len(csv_entries) == 1:
            print(f"    Single CSV inside ZIP: {csv_entries[0]}")
            df = _read_csv_bytes(zf.read(csv_entries[0]))
            return _build_from_single_csv(df, label, max_tweets)
        print(f"    [SKIP] No users.csv found inside ZIP")
        return []

    users_df = _read_csv_bytes(zf.read(users_entry))
    print(f"    users.csv  : {len(users_df):,} rows")

    tweet_lookup = defaultdict(list)
    if tweets_entry:
        tweets_df = _read_csv_bytes(zf.read(tweets_entry))
        print(f"    tweets.csv : {len(tweets_df):,} rows")
        uid_col = next(
            (c for c in ("user_id", "author_id", "userid") if c in tweets_df.columns), None
        )
        if uid_col:
            for _, trow in tweets_df.iterrows():
                uid = _safe_str(trow[uid_col])
                tweet_lookup[uid].append(_tweet_to_dict(trow))
        else:
            print(f"    [WARN] No user_id column in tweets.csv -- tweets skipped")
    else:
        print(f"    [WARN] No tweets.csv inside ZIP -- proceeding without tweets")

    id_col = next(
        (c for c in ("id", "user_id", "userid") if c in users_df.columns),
        users_df.columns[0],
    )
    records = []
    for _, row in users_df.iterrows():
        uid    = _safe_str(row[id_col])
        tweets = tweet_lookup.get(uid, [])[:max_tweets]
        records.append(_user_to_record(row, tweets, label))

    print(f"    -> {len(records):,} records")
    return records


def convert(input_dir: str, output_path: str, max_tweets: int = 200):
    root = Path(input_dir)
    if not root.exists():
        print(f"[ERROR] Directory not found: {root}")
        sys.exit(1)

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No .csv files found in {root}")
        sys.exit(1)

    all_records = []
    skipped = []

    for csv_path in csv_files:
        stem = csv_path.stem.lower()
        if stem in SKIP_FILES or stem.startswith("read"):
            print(f"  [SKIP] {csv_path.name} (metadata)")
            continue
        if stem in HUMAN_FILES:
            label = 0
        elif stem in BOT_FILES:
            label = 1
        else:
            print(f"  [SKIP] {csv_path.name} -- unrecognised name")
            skipped.append(csv_path.name)
            continue

        all_records.extend(_process_zip_csv(csv_path, label, max_tweets))

    if not all_records:
        print("\n[ERROR] No records loaded.")
        sys.exit(1)

    n_humans = sum(1 for r in all_records if r["label"] == "0")
    n_bots   = sum(1 for r in all_records if r["label"] == "1")
    print(f"\n{'='*55}")
    print(f"  Total records : {len(all_records):,}")
    print(f"  Humans  (0)   : {n_humans:,}")
    print(f"  Bots    (1)   : {n_bots:,}")
    print(f"  Bot ratio     : {n_bots / len(all_records):.1%}")
    print(f"{'='*55}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    size_mb = out.stat().st_size / 1_048_576
    print(f"\n  Saved -> {out}  ({size_mb:.1f} MB)")
    print(f"\nNext -- train your models:")
    print(f"  python scripts/train.py --data_file {output_path}")
    print(f"  python scripts/train.py --data_file {output_path} --model transformer --epochs 100")
    print(f"  python scripts/train.py --data_file {output_path} --all")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Cresci-2017 (IU Bot Repository) to project JSON"
    )
    p.add_argument("--input_dir", required=True,
                   help='Folder containing the .csv ZIP archives')
    p.add_argument("--output", default="data/cresci17.json",
                   help="Output JSON file (default: data/cresci17.json)")
    p.add_argument("--max_tweets", type=int, default=200,
                   help="Max tweets per user (default: 200)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input_dir, args.output, args.max_tweets)