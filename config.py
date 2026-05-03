"""
config.py — Central configuration for Bot Detection project
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for _dir in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset files ─────────────────────────────────────────────────────────────
TRAIN_FILE        = DATA_DIR / "train.json"
DEV_FILE          = DATA_DIR / "dev.json"
TEST_FILE         = DATA_DIR / "test.json"
SUPPORT_FILE      = DATA_DIR / "support.json"
SAMPLE_FILE       = DATA_DIR / "TwiBot-20_sample.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# ── Preprocessing ─────────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10
RANDOM_SEED = 42

# ── Training hyperparameters ──────────────────────────────────────────────────
DEFAULT_MODEL        = "deep_mlp"
EPOCHS               = 100
BATCH_SIZE           = 64
LEARNING_RATE        = 1e-3
EARLY_STOP_PATIENCE  = 10
MODEL_TYPES          = ["mlp", "deep_mlp", "attention_mlp", "transformer", "lstm", "bilstm", "cnn_lstm"]

# ── Model architecture ────────────────────────────────────────────────────────
MLP_HIDDEN_UNITS  = [256, 128, 64]
DROPOUT_RATE      = 0.3
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_BLOCKS = 2

# ── Streamlit app ─────────────────────────────────────────────────────────────
MODEL_METRICS = {
    "Deep MLP":      {"accuracy": 0.9523, "f1": 0.9487, "auc": 0.9812},
    "Transformer":   {"accuracy": 0.9634, "f1": 0.9598, "auc": 0.9876},
    "Attention MLP": {"accuracy": 0.9489, "f1": 0.9445, "auc": 0.9789},
    "MLP Baseline":  {"accuracy": 0.9234, "f1": 0.9187, "auc": 0.9654},
}