"""
src/preprocessing.py
Bot Detection — Data Preprocessing & Feature Extraction
Compatible with Cresci-17 and TwiBot-20 dataset schemas.
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class BotDataPreprocessor:
    """
    Extracts ~38 features from a user record and prepares
    train / val / test splits for deep learning.
    """

    # All possible date formats across datasets
    _DATE_FMTS = [
        "%a %b %d %H:%M:%S +0000 %Y",   # Cresci-17 + TwiBot-20 Twitter format
        "%Y-%m-%d %H:%M:%S",             # TwiBot-20 tweet format
        "%Y-%m-%d",                       # TwiBot-20 profile format
    ]

    def __init__(self):
        self.scaler        = StandardScaler()
        self.feature_names: list = []

    # ─────────────────────────────── public API ───────────────────────────────

    def extract_user_features(self, user: dict) -> dict:
        features = {}
        features.update(self._metadata_features(user))
        features.update(self._tweet_content_features(user))
        features.update(self._temporal_features(user))
        features.update(self._behavioural_features(features))
        return features

    def process_file(self, json_path: str, default_label: int = None) -> pd.DataFrame:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows, labels, user_ids = [], [], []

        for user in data:
            try:
                feat = self.extract_user_features(user)
                rows.append(feat)

                if default_label is not None:
                    labels.append(default_label)
                elif "label" in user:
                    labels.append(int(user["label"]))
                else:
                    labels.append(-1)

                user_ids.append(user.get("ID", user.get("id", "")))

            except Exception as exc:
                uid = user.get("ID", user.get("id", "unknown"))
                print(f"[WARN] Skipping user {uid}: {exc}")

        df = pd.DataFrame(rows)
        df["label"]   = labels
        df["user_id"] = user_ids
        return df

    def fit(self, df: pd.DataFrame):
        """Fit the scaler on training data."""
        meta_cols = {"label", "user_id"}
        self.feature_names = [c for c in df.columns if c not in meta_cols]
        X = self._to_array(df)
        self.scaler.fit(X)

    def transform(self, df: pd.DataFrame):
        """Transform a DataFrame into scaled (X, y) arrays."""
        X = self._to_array(df)
        y = df["label"].values.astype(int)
        return self.scaler.transform(X), y

    def prepare_splits(self, df: pd.DataFrame, test_size=0.20, val_size=0.10):
        """Fit+transform on the whole df, return stratified splits."""
        meta_cols = {"label", "user_id"}
        self.feature_names = [c for c in df.columns if c not in meta_cols]

        X = self._to_array(df)
        y = df["label"].values.astype(int)

        stratify = y if len(np.unique(y)) > 1 else None

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        adj_val = val_size / (1 - test_size)
        stratify_tmp = y_tmp if len(np.unique(y_tmp)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=adj_val, random_state=42, stratify=stratify_tmp
        )

        self.scaler.fit(X_train)
        return (
            self.scaler.transform(X_train), X_val, self.scaler.transform(X_test),
            y_train, y_val, y_test, self.feature_names
        )

    def save(self, path: str = "models/preprocessor.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_names": self.feature_names}, f)
        print(f"Preprocessor saved → {path}")

    def load(self, path: str = "models/preprocessor.pkl") -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Support both dict format and direct object format
        if isinstance(obj, dict):
            self.scaler        = obj["scaler"]
            self.feature_names = obj.get("feature_names", [])
        else:
            # obj is a BotDataPreprocessor instance saved directly
            self.scaler        = obj.scaler
            self.feature_names = obj.feature_names
        print(f"Preprocessor loaded ← {path}")

    def transform_single(self, user: dict) -> np.ndarray:
        """Transform a single raw user dict → scaled feature vector (inference)."""
        feat = self.extract_user_features(user)
        row  = np.array([feat.get(k, 0.0) for k in self.feature_names], dtype=float).reshape(1, -1)
        row  = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(row)

    # ─────────────────────────────── feature groups ───────────────────────────

    def _parse_date(self, s: str):
        for fmt in self._DATE_FMTS:
            try:
                return datetime.strptime(s, fmt)
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_int(val, default=False):
        if isinstance(val, str):
            return int(val.lower() not in ("", "false", "0", "none", "null"))
        return int(bool(val)) if val is not None else int(default)

    @staticmethod
    def _div(a, b):
        return a / b if b else 0.0

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].values.astype(float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _metadata_features(self, user: dict) -> dict:
        profile = user.get("profile", {})
        f = {}

        created = profile.get("created_at", "")
        dt = self._parse_date(created)
        age = (datetime.now() - dt).days if dt else 0

        f["account_age_days"]        = age
        f["followers_count"]         = self._int(profile.get("followers_count", 0))
        f["friends_count"]           = self._int(profile.get("friends_count", 0))
        f["statuses_count"]          = self._int(profile.get("statuses_count", 0))
        f["favourites_count"]        = self._int(profile.get("favourites_count", 0))
        f["listed_count"]            = self._int(profile.get("listed_count", 0))
        f["followers_friends_ratio"] = self._div(f["followers_count"], f["friends_count"])
        f["tweet_frequency"]         = self._div(f["statuses_count"], max(age, 1))
        f["has_description"]         = int(bool(profile.get("description", "")))
        f["has_url"]                 = int(bool(profile.get("url", "")))
        f["has_location"]            = int(bool(profile.get("location", "")))
        f["verified"]                = self._bool_int(profile.get("verified", False))
        f["default_profile"]         = self._bool_int(profile.get("default_profile", True))
        f["default_profile_image"]   = self._bool_int(profile.get("default_profile_image", True))
        f["geo_enabled"]             = self._bool_int(profile.get("geo_enabled", False))
        sname = profile.get("screen_name", "")
        f["screen_name_length"]      = len(sname)
        f["screen_name_has_digits"]  = int(bool(re.search(r"\d", sname)))
        f["name_length"]             = len(profile.get("name", ""))
        return f

    def _tweet_content_features(self, user: dict) -> dict:
        tweets = user.get("tweet", user.get("tweets", []))
        zero_keys = [
            "tweet_count", "avg_tweet_length", "std_tweet_length",
            "avg_urls_per_tweet", "avg_mentions_per_tweet",
            "avg_hashtags_per_tweet", "retweet_ratio",
            "lexical_diversity", "source_diversity",
            "avg_retweets", "avg_favorites", "engagement_rate",
        ]
        if not tweets:
            return dict.fromkeys(zero_keys, 0.0)

        texts = [t.get("text", "") for t in tweets]
        f = {}
        f["tweet_count"]             = len(tweets)
        lengths                      = [len(t) for t in texts]
        f["avg_tweet_length"]        = float(np.mean(lengths))
        f["std_tweet_length"]        = float(np.std(lengths))
        f["avg_urls_per_tweet"]      = float(np.mean([len(re.findall(r"https?://\S+", t)) for t in texts]))
        f["avg_mentions_per_tweet"]  = float(np.mean([len(re.findall(r"@\w+", t))         for t in texts]))
        f["avg_hashtags_per_tweet"]  = float(np.mean([len(re.findall(r"#\w+", t))         for t in texts]))
        f["retweet_ratio"]           = float(np.mean([1.0 if t.startswith("RT @") else 0.0 for t in texts]))
        all_words                    = " ".join(texts).lower().split()
        f["lexical_diversity"]       = self._div(len(set(all_words)), max(len(all_words), 1))
        sources                      = [t.get("source", "") for t in tweets]
        f["source_diversity"]        = self._div(len(set(sources)), len(sources))
        rt_vals                      = [self._int(t.get("retweet_count",  0)) for t in tweets]
        fav_vals                     = [self._int(t.get("favorite_count", 0)) for t in tweets]
        f["avg_retweets"]            = float(np.mean(rt_vals))
        f["avg_favorites"]           = float(np.mean(fav_vals))
        followers                    = max(self._int(user.get("profile", {}).get("followers_count", 1)), 1)
        f["engagement_rate"]         = self._div(f["avg_retweets"] + f["avg_favorites"], followers)
        return f

    def _temporal_features(self, user: dict) -> dict:
        tweets = user.get("tweet", user.get("tweets", []))
        zero_keys = [
            "avg_time_between_tweets", "std_time_between_tweets",
            "tweets_night_ratio", "tweets_weekend_ratio",
            "tweets_morning_ratio", "tweets_afternoon_ratio",
        ]
        if not tweets:
            return dict.fromkeys(zero_keys, 0.0)

        times = []
        for t in tweets:
            dt = self._parse_date(t.get("created_at", ""))
            if dt:
                times.append(dt)

        if not times:
            return dict.fromkeys(zero_keys, 0.0)

        f = {}
        if len(times) > 1:
            diffs = [(times[i] - times[i-1]).total_seconds() / 3600 for i in range(1, len(times))]
            f["avg_time_between_tweets"] = float(np.mean(diffs))
            f["std_time_between_tweets"] = float(np.std(diffs))
        else:
            f["avg_time_between_tweets"] = 0.0
            f["std_time_between_tweets"] = 0.0

        hours = [t.hour    for t in times]
        wdays = [t.weekday() for t in times]
        n = len(hours)
        f["tweets_night_ratio"]     = sum(1 for h in hours if h < 6 or h >= 22) / n
        f["tweets_morning_ratio"]   = sum(1 for h in hours if 6  <= h < 12)     / n
        f["tweets_afternoon_ratio"] = sum(1 for h in hours if 12 <= h < 18)     / n
        f["tweets_weekend_ratio"]   = sum(1 for w in wdays if w >= 5)           / n
        return f

    def _behavioural_features(self, f: dict) -> dict:
        age = max(f.get("account_age_days", 1), 1)
        return {
            "avg_daily_tweets":     self._div(f.get("statuses_count", 0), age),
            "follower_growth_rate": self._div(f.get("followers_count", 0), age),
            "reciprocity": self._div(
                min(f.get("followers_count", 0), f.get("friends_count", 0)),
                max(f.get("followers_count",  1), f.get("friends_count",  1)),
            ),
            "network_size": f.get("followers_count", 0) + f.get("friends_count", 0),
        }