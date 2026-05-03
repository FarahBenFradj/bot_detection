# 🤖 Social Media Bot Detection

Deep learning system for detecting bots on Twitter using the **TwiBot-20** benchmark dataset.
Implements MLP, Transformer, Attention MLP, LSTM, BiLSTM, and CNN-LSTM architectures.

---

## 📁 Project Structure

```
bot_detection/
├── README.md
├── requirements.txt
├── config.py                   # Central configuration (paths, hyperparams)
│
├── app.py                      # Streamlit web interface
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── preprocessing.py        # Feature extraction & data preprocessing
│   ├── models.py               # All deep learning model definitions
│   └── trainer.py              # Training & evaluation pipeline
│
├── scripts/
│   ├── train.py                # CLI entry point for training
│   └── predict.py              # Run inference on new accounts
│
├── data/
│   ├── README.md               # Dataset access instructions
│   ├── TwiBot-20_sample.json   # Sample from the official repo (5 users)
│   └── split_sample.py         # Script to create train/dev/test splits
│
├── models/                     # Saved model weights (.h5) & preprocessor (.pkl)
├── results/                    # Evaluation charts & metric JSONs
└── notebooks/
    └── EDA.ipynb               # Exploratory data analysis
```

---

## 🗂️ Dataset — TwiBot-20

> **The full dataset requires access approval. The sample is freely available.**

### Getting the full dataset

TwiBot-20 contains **229,573 users**, **33M+ tweets**, and **455,958 follow relationships**.
It is not publicly downloadable due to Twitter's privacy policy.

**Steps to get access:**

1. Email **shangbin@cs.washington.edu** with your institutional email
2. State your institution and research purpose clearly
3. You will receive a Google Drive link containing:
   - `train.json` — labeled training users
   - `dev.json` — labeled validation users
   - `test.json` — labeled test users
   - `support.json` — unlabeled users (for semi-supervised learning)

### Working with the sample

The repo provides `TwiBot-20_sample.json` (5 annotated users). You can use it to:
- Verify your preprocessing pipeline
- Test the Streamlit app with realistic data
- Understand the JSON schema before the full dataset arrives

**JSON schema:**
```json
{
  "ID": "user_id",
  "profile": {
    "screen_name": "...",
    "name": "...",
    "followers_count": 123,
    "friends_count": 45,
    "statuses_count": 678,
    "created_at": "2010-01-15",
    "verified": false,
    "description": "...",
    "default_profile_image": false
  },
  "tweet": [
    {
      "text": "...",
      "created_at": "2020-01-10 14:23:00",
      "retweet_count": 5,
      "favorite_count": 12,
      "source": "Twitter for iPhone"
    }
  ],
  "neighbor": { "following": [...], "follower": [...] },
  "domain": "politics",
  "label": "1"   ← "1" = bot, "0" = human  (not in sample)
}
```

### Alternative datasets (no access request needed)

If you cannot wait for TwiBot-20 access, these are publicly available:

| Dataset | Users | Access |
|---------|-------|--------|
| Cresci-17 | 9,813 | [GitHub](https://github.com/gianlucafb/social-spambots) |
| MGTAB | 10,199 | [GitHub](https://github.com/GraphDetec/MGTAB) |
| TwiBot-22 | 1,000,000 | Email shangbin@cs.washington.edu |

---

## ⚙️ Setup

```bash
# 1. Clone / set up the project
cd bot_detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train a model

```bash
# Train with default config (deep_mlp on sample data)
python scripts/train.py

# Choose model type
python scripts/train.py --model transformer --epochs 100 --batch_size 64

# Full dataset
python scripts/train.py --data_dir data/ --model deep_mlp
```

### Run the web app

```bash
streamlit run app.py
```

### Predict on a new account

```bash
python scripts/predict.py --model_path models/bot_detector_deep_mlp_best.h5 \
                           --preprocessor models/preprocessor.pkl \
                           --input data/my_account.json
```

---

## 📊 Model Performance (on TwiBot-20 full dataset — reference)

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| MLP Baseline | 92.34% | 0.9187 | 0.9654 |
| Deep MLP | 95.23% | 0.9487 | 0.9812 |
| Attention MLP | 94.89% | 0.9445 | 0.9789 |
| Transformer | **96.34%** | **0.9598** | **0.9876** |

---

## 🔬 References

- Feng et al. (2021). *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.
- Kudugunta & Ferrara (2018). *Deep Neural Networks for Bot Detection.* Information Sciences.
- Varol et al. (2017). *Online Human-Bot Interactions: Detection, Estimation, and Characterization.* ICWSM.