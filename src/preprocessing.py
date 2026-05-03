"""
src/preprocessing.py
Social Media Bot Detection — Data Preprocessing & Feature Extraction
Based on TwiBot-20 dataset schema.

Key fix vs original: TwiBot-20 uses "tweet" (not "tweets") and "ID" (not "id").
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class BotDataPreprocessor:
    """
    Extracts ~60 features from TwiBot-20 user records and prepares
    train / val / test splits for deep learning.

    Feature categories
    ------------------
    1. User metadata          (17 features)
    2. Tweet content          (11 features)
    3. Temporal patterns       (6 features)
    4. Behavioural / derived   (4 features)
    """

    # TwiBot-20 date format inside profiles
    _PROFILE_DATE_FMT = "%Y-%m-%d"
    # TwiBot-20 date format inside tweets
    _TWEET_DATE_FMT   = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []

    # ─────────────────────────────── public API ───────────────────────────────

    def extract_user_features(self, user_data: dict) -> dict:
        """Return a flat feature dict for a single TwiBot-20 user record."""
        features: dict = {}
        features.update(self._metadata_features(user_data))
        features.update(self._tweet_content_features(user_data))
        features.update(self._temporal_features(user_data))
        features.update(self._behavioural_features(features))
        return features

    def process_file(self, json_path: str, default_label: int | None = None) -> pd.DataFrame:
        """
        Load a TwiBot-20 JSON file and return a DataFrame with features + label.

        Parameters
        ----------
        json_path      : path to train.json / dev.json / test.json
        default_label  : override label (use when file has no 'label' key)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows, labels, user_ids = [], [], []

        for user in data:
            try:
                feat = self.extract_user_features(user)
                rows.append(feat)

                # TwiBot-20 labels are strings "0" / "1"
                if default_label is not None:
                    labels.append(default_label)
                elif "label" in user:
                    labels.append(int(user["label"]))
                else:
                    labels.append(-1)   # unlabelled (support set)

                user_ids.append(user.get("ID", user.get("id", "")))

            except Exception as exc:
                uid = user.get("ID", user.get("id", "unknown"))
                print(f"[WARN] Skipping user {uid}: {exc}")

        df = pd.DataFrame(rows)
        df["label"]   = labels
        df["user_id"] = user_ids
        return df

    def prepare_splits(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        val_size:  float = 0.10,
    ):
        """
        Scale features and create stratified train / val / test splits.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        """
        meta_cols  = {"label", "user_id"}
        feat_cols  = [c for c in df.columns if c not in meta_cols]
        self.feature_names = feat_cols

        X = df[feat_cols].values.astype(float)
        y = df["label"].values

        # Replace NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X)

        # Stratify only when there are enough samples per class
        use_stratify = len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(int))) >= 2
        stratify_y   = y if use_stratify else None

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=stratify_y
        )
        adj_val       = val_size / (1 - test_size)
        stratify_tmp  = y_tmp if use_stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=adj_val, random_state=42, stratify=stratify_tmp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols

    def save(self, path: str = "models/preprocessor.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_names": self.feature_names}, f)
        print(f"Preprocessor saved → {path}")

    def load(self, path: str = "models/preprocessor.pkl") -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler        = obj["scaler"]
        self.feature_names = obj.get("feature_names", [])
        print(f"Preprocessor loaded ← {path}")

    def transform(self, user_data: dict) -> np.ndarray:
        """
        Transform a single raw user dict into a scaled feature vector.
        Used at inference time (e.g. in the Streamlit app).
        """
        feat = self.extract_user_features(user_data)
        row  = np.array([feat.get(k, 0.0) for k in self.feature_names],
                        dtype=float).reshape(1, -1)
        row  = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(row)

    # ─────────────────────────────── feature groups ───────────────────────────

    @staticmethod
    def _int(val, default: int = 0) -> int:
        """Safely cast a value to int — handles strings like '\"150\"'."""
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_int(val, default: bool = False) -> int:
        """Safely cast a truthy value to 0/1."""
        if isinstance(val, str):
            return int(val.lower() not in ("", "false", "0", "none", "null"))
        return int(bool(val)) if val is not None else int(default)

    def _metadata_features(self, user: dict) -> dict:
        profile = user.get("profile", {})
        f: dict = {}

        # Account age — created_at may be "2010-06-14" or "Mon Jun 14 00:00:00 +0000 2010"
        created = profile.get("created_at", "")
        age = 0
        if created:
            for fmt in (self._PROFILE_DATE_FMT, "%a %b %d %H:%M:%S +0000 %Y"):
                try:
                    age = (datetime.now() -
                           datetime.strptime(created, fmt)).days
                    break
                except ValueError:
                    continue

        f["account_age_days"]       = age
        # TwiBot-20 stores these as strings — cast explicitly
        f["followers_count"]        = self._int(profile.get("followers_count", 0))
        f["friends_count"]          = self._int(profile.get("friends_count", 0))
        f["statuses_count"]         = self._int(profile.get("statuses_count", 0))
        f["favourites_count"]       = self._int(profile.get("favourites_count", 0))
        f["listed_count"]           = self._int(profile.get("listed_count", 0))

        # Derived ratios
        f["followers_friends_ratio"] = self._div(f["followers_count"],
                                                  f["friends_count"])
        f["tweet_frequency"]         = self._div(f["statuses_count"],
                                                  max(age, 1))

        # Profile completeness
        f["has_description"]        = int(bool(profile.get("description", "")))
        f["has_url"]                = int(bool(profile.get("url", "")))
        f["has_location"]           = int(bool(profile.get("location", "")))
        f["verified"]               = self._bool_int(profile.get("verified", False))
        f["default_profile"]        = self._bool_int(profile.get("default_profile", True))
        f["default_profile_image"]  = self._bool_int(profile.get("default_profile_image", True))

        # Username
        sname = profile.get("screen_name", "")
        f["screen_name_length"]     = len(sname)
        f["screen_name_has_digits"] = int(bool(re.search(r"\d", sname)))
        f["name_length"]            = len(profile.get("name", ""))

        return f

    def _tweet_content_features(self, user: dict) -> dict:
        # TwiBot-20 uses "tweet" key (list of tweet objects)
        tweets = user.get("tweet", user.get("tweets", []))
        f: dict = {}

        zero_keys = [
            "avg_tweet_length", "std_tweet_length",
            "avg_urls_per_tweet", "avg_mentions_per_tweet",
            "avg_hashtags_per_tweet", "retweet_ratio",
            "lexical_diversity", "source_diversity",
            "avg_retweets", "avg_favorites", "engagement_rate",
        ]

        if not tweets:
            return dict.fromkeys(zero_keys, 0.0)

        texts = [t.get("text", "") for t in tweets]

        lengths = [len(t) for t in texts]
        f["avg_tweet_length"] = float(np.mean(lengths))
        f["std_tweet_length"] = float(np.std(lengths))

        f["avg_urls_per_tweet"]     = np.mean([len(re.findall(r"https?://\S+", t)) for t in texts])
        f["avg_mentions_per_tweet"] = np.mean([len(re.findall(r"@\w+", t))         for t in texts])
        f["avg_hashtags_per_tweet"] = np.mean([len(re.findall(r"#\w+", t))         for t in texts])
        f["retweet_ratio"]          = np.mean([1.0 if t.startswith("RT @") else 0.0 for t in texts])

        all_words   = " ".join(texts).lower().split()
        f["lexical_diversity"] = self._div(len(set(all_words)), max(len(all_words), 1))

        sources = [t.get("source", "") for t in tweets]
        f["source_diversity"] = self._div(len(set(sources)), len(sources))

        rt_vals  = [self._int(t.get("retweet_count",  0)) for t in tweets]
        fav_vals = [self._int(t.get("favorite_count", 0)) for t in tweets]
        f["avg_retweets"]  = float(np.mean(rt_vals))
        f["avg_favorites"] = float(np.mean(fav_vals))

        followers = self._int(user.get("profile", {}).get("followers_count", 1), default=1)
        f["engagement_rate"] = self._div(
            f["avg_retweets"] + f["avg_favorites"], max(followers, 1)
        )

        return f

    def _temporal_features(self, user: dict) -> dict:
        tweets = user.get("tweet", user.get("tweets", []))
        f: dict = {}

        zero_keys = [
            "avg_time_between_tweets", "std_time_between_tweets",
            "tweets_night_ratio", "tweets_weekend_ratio",
            "tweets_morning_ratio", "tweets_afternoon_ratio",
        ]

        if not tweets:
            return dict.fromkeys(zero_keys, 0.0)

        raw_ts = [t.get("created_at", "") for t in tweets if t.get("created_at")]
        if not raw_ts:
            return dict.fromkeys(zero_keys, 0.0)

        try:
            times = [datetime.strptime(ts, self._TWEET_DATE_FMT) for ts in raw_ts]
        except ValueError:
            return dict.fromkeys(zero_keys, 0.0)

        if len(times) > 1:
            diffs = [(times[i] - times[i - 1]).total_seconds() / 3600
                     for i in range(1, len(times))]
            f["avg_time_between_tweets"] = float(np.mean(diffs))
            f["std_time_between_tweets"] = float(np.std(diffs))
        else:
            f["avg_time_between_tweets"] = 0.0
            f["std_time_between_tweets"] = 0.0

        hours   = [t.hour    for t in times]
        wdays   = [t.weekday() for t in times]
        n = len(hours)

        f["tweets_night_ratio"]     = sum(1 for h in hours if h < 6 or h >= 22)  / n
        f["tweets_morning_ratio"]   = sum(1 for h in hours if 6 <= h < 12)        / n
        f["tweets_afternoon_ratio"] = sum(1 for h in hours if 12 <= h < 18)       / n
        f["tweets_weekend_ratio"]   = sum(1 for w in wdays  if w >= 5)            / n

        return f

    def _behavioural_features(self, f: dict) -> dict:
        age = max(f.get("account_age_days", 1), 1)
        return {
            "avg_daily_tweets":    self._div(f.get("statuses_count", 0), age),
            "follower_growth_rate": self._div(f.get("followers_count", 0), age),
            "reciprocity": self._div(
                min(f.get("followers_count", 0), f.get("friends_count", 0)),
                max(f.get("followers_count", 1), f.get("friends_count", 1)),
            ),
            "network_size": f.get("followers_count", 0) + f.get("friends_count", 0),
        }

    # ─────────────────────────────── helpers ──────────────────────────────────

    @staticmethod
    def _div(a: float, b: float) -> float:
        return a / b if b else 0.0