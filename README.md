# 🤖 Social Media Bot Detection

Deep learning system for detecting bots on Twitter using the **Cresci-17** benchmark dataset.
Implements **MLP**, **Deep MLP**, **Attention MLP**, and **Transformer** architectures with a full training, evaluation, and inference pipeline.

---

## 📁 Project Structure

```
bot_detection/
├── README.md
├── requirements.txt
├── config.py                      # Central configuration (paths, hyperparams)
│
├── app.py                         # Streamlit web interface
│
├── src/                           # Core library
│   ├── __init__.py
│   ├── preprocessing.py           # Feature extraction & data preprocessing
│   ├── models.py                  # All deep learning model definitions
│   └── trainer.py                 # Training & evaluation pipeline
│
├── scripts/
│   ├── train.py                   # CLI entry point for training
│   ├── predict.py                 # Run inference on new accounts
│   └── convert_cresci17.py        # Convert Cresci-17 CSVs → project JSON format
│
├── data/
│   ├── cresci17.json              # Converted Cresci-17 dataset (after conversion)
│   └── TwiBot-20_sample.json      # Small sample for pipeline testing
│
├── models/                        # Saved model weights (.h5) & preprocessor (.pkl)
└── results/                       # Evaluation charts & metric JSONs
```

---

## 🗂️ Dataset — Cresci-17

The primary dataset used in this project is **Cresci-17** (Cresci et al., WWW 2017), a publicly available benchmark from the Indiana University Bot Repository containing **~11,500 labeled Twitter accounts** across multiple bot categories.

### Download

```
https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip
```

### Structure after unzipping

```
cresci-2017.csv/
└── datasets_full.csv/
    ├── genuine_accounts.csv       ← humans      (label = 0)
    ├── social_spambots_1.csv      ← bots         (label = 1)
    ├── social_spambots_2.csv      ← bots
    ├── social_spambots_3.csv      ← bots
    ├── traditional_spambots_1.csv ← bots
    ├── traditional_spambots_2.csv ← bots
    ├── traditional_spambots_3.csv ← bots
    ├── traditional_spambots_4.csv ← bots
    ├── fake_followers.csv         ← bots
    └── crowdflower_results.csv    ← skipped (annotation metadata)
```

> Each `.csv` file is a **ZIP archive** containing `users.csv` (account profiles) and `tweets.csv` (tweet history).

### Dataset statistics

| Category | Users | Label |
|---|---|---|
| Genuine accounts | 3,474 | 0 (human) |
| Social spambots (×3) | 5,301 | 1 (bot) |
| Traditional spambots (×4) | 1,947 | 1 (bot) |
| Fake followers | 1,169 | 1 (bot) |
| **Total** | **~11,891** | — |

### Convert to project format

```bash
python scripts/convert_cresci17.py \
    --input_dir "path/to/cresci-2017.csv/datasets_full.csv" \
    --output data/cresci17.json
```

### Alternative — TwiBot-20 (gated access)

For larger-scale experiments, TwiBot-20 contains 229,573 users and 33M+ tweets.
Email **shangbin@cs.washington.edu** with your institution and research purpose to request access.
Once you have `train.json`, `dev.json`, `test.json`, place them in `data/` and run:

```bash
python scripts/train.py --data_dir data/
```

---

## ⚙️ Setup

```bash
# 1. Clone the project
cd bot_detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Convert the dataset

```bash
python scripts/convert_cresci17.py \
    --input_dir "path/to/cresci-2017.csv/datasets_full.csv" \
    --output data/cresci17.json
```

### Step 2 — Train a model

```bash
# Default model (Deep MLP)
python scripts/train.py --data_file data/cresci17.json

# Choose a specific architecture
python scripts/train.py --data_file data/cresci17.json --model transformer --epochs 100

# Train all 4 architectures and compare
python scripts/train.py --data_file data/cresci17.json --all

# Custom hyperparameters
python scripts/train.py --data_file data/cresci17.json --model attention --epochs 50 --batch_size 32 --lr 0.0005
```

Available models: `mlp` | `deep_mlp` | `attention` | `transformer`

### Step 3 — Launch the web app

```bash
streamlit run app.py
```

### Inference on a new account

```bash
python scripts/predict.py \
    --model_path models/bot_detector_deep_mlp_best.h5 \
    --preprocessor models/preprocessor.pkl \
    --input data/my_account.json
```

---

## 🧠 Feature Engineering

The preprocessing pipeline (`src/preprocessing.py`) extracts **38 features** from raw user records:

| Group | Features | Count |
|---|---|---|
| User metadata | account age, followers, friends, tweet frequency, profile completeness, username stats | 17 |
| Tweet content | avg length, URL/mention/hashtag density, retweet ratio, lexical diversity, engagement | 11 |
| Temporal patterns | inter-tweet timing, night/morning/afternoon/weekend posting ratios | 6 |
| Behavioural | daily tweet rate, follower growth rate, reciprocity, network size | 4 |

---

## 🏗️ Model Architectures

| Model | Description |
|---|---|
| `mlp` | Baseline feedforward network (256 → 128 → 64), BatchNorm + Dropout |
| `deep_mlp` | Deep MLP with residual skip connections (ResNet-style) |
| `attention` | MLP with soft-attention gate — learns to weight important features |
| `transformer` | Multi-head self-attention encoder (4 heads, 2 blocks) |

---

## 📊 Model Performance (on Cresci-17)

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| MLP Baseline | ~97% | ~0.97 | ~0.98 |
| Deep MLP | ~98% | ~0.98 | ~0.99 |
| Attention MLP | ~98% | ~0.98 | ~0.99 |
| **Transformer** | **~99%** | **~0.99** | **~0.99** |

> Cresci-17 is a relatively separable dataset — high accuracy is expected. For a more challenging benchmark, use TwiBot-20 (requires access approval).

---

## 🔬 References

- Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). *The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race.* WWW 2017.
- Feng, S., Wan, H., Wang, N., Li, J., & Luo, M. (2021). *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.
- Kudugunta, S., & Ferrara, E. (2018). *Deep Neural Networks for Bot Detection.* Information Sciences, 467, 312–322.
- Varol, O., Ferrara, E., Davis, C. A., Menczer, F., & Flammini, A. (2017). *Online Human-Bot Interactions: Detection, Estimation, and Characterization.* ICWSM 2017.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.