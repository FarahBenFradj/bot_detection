"""
src/models.py
Deep Learning model definitions for Bot Detection.

Models: MLP, DeepMLP, AttentionMLP, Transformer, LSTM, BiLSTM, CNN-LSTM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class BotDetectionModels:
    """Factory class — each static method returns a compiled-ready Keras model."""

    # ──────────────────────────────── MLP family ──────────────────────────────

    @staticmethod
    def build_mlp(input_dim: int,
                  hidden_units: list[int] = (256, 128, 64),
                  dropout_rate: float = 0.3) -> Model:
        """Baseline MLP — fast, solid tabular baseline."""
        model = keras.Sequential(name="MLP")
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        for units in hidden_units:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    @staticmethod
    def build_deep_mlp(input_dim: int) -> Model:
        """Deep MLP with skip (residual) connections."""
        inp = layers.Input(shape=(input_dim,))

        x = layers.Dense(256, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Residual block 1
        res = x
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])

        # Residual block 2
        x = layers.Dense(128, activation="relu")(x)
        res = x
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])

        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        return Model(inp, out, name="DeepMLP")

    @staticmethod
    def build_attention_mlp(input_dim: int) -> Model:
        """MLP with a soft-attention gate — learns to weight important features."""
        inp = layers.Input(shape=(input_dim,))

        x = layers.Dense(256, activation="relu")(inp)
        x = layers.BatchNormalization()(x)

        # Attention gate
        attn = layers.Dense(256, activation="tanh")(x)
        attn = layers.Dense(256, activation="softmax")(attn)
        x = layers.Multiply()([x, attn])

        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        return Model(inp, out, name="AttentionMLP")

    # ─────────────────────────────── Transformer ──────────────────────────────

    @staticmethod
    def build_transformer(input_dim: int,
                          num_heads: int = 4,
                          ff_dim: int = 128,
                          num_blocks: int = 2) -> Model:
        """
        Transformer encoder applied to a single-token feature vector.
        Multi-head self-attention learns inter-feature relationships.
        """
        inp = layers.Input(shape=(input_dim,))
        x   = layers.Reshape((1, input_dim))(inp)   # (batch, 1, dim)

        for _ in range(num_blocks):
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(input_dim // num_heads, 1)
            )(x, x)
            attn_out = layers.Dropout(0.1)(attn_out)
            x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

            ff = layers.Dense(ff_dim, activation="relu")(x1)
            ff = layers.Dense(input_dim)(ff)
            ff = layers.Dropout(0.1)(ff)
            x  = layers.LayerNormalization(epsilon=1e-6)(x1 + ff)

        x   = layers.GlobalAveragePooling1D()(x)
        x   = layers.Dense(128, activation="relu")(x)
        x   = layers.Dropout(0.3)(x)
        x   = layers.Dense(64,  activation="relu")(x)
        x   = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        return Model(inp, out, name="Transformer")

    # ─────────────────────────────── Sequence models ──────────────────────────

    @staticmethod
    def build_lstm(feature_dim: int,
                   seq_len: int = 10,
                   units: int = 128,
                   dropout: float = 0.3) -> Model:
        """
        Stacked LSTM.
        Input shape: (batch, seq_len, feature_dim)
        Use reshape_for_sequence() to prepare tabular data.
        """
        model = keras.Sequential([
            layers.Input(shape=(seq_len, feature_dim)),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.LSTM(units // 2),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1,  activation="sigmoid"),
        ], name="LSTM")
        return model

    @staticmethod
    def build_bilstm(feature_dim: int,
                     seq_len: int = 10,
                     units: int = 128,
                     dropout: float = 0.3) -> Model:
        """Bidirectional LSTM — processes sequences in both directions."""
        model = keras.Sequential([
            layers.Input(shape=(seq_len, feature_dim)),
            layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(units // 2)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1,  activation="sigmoid"),
        ], name="BiLSTM")
        return model

    @staticmethod
    def build_cnn_lstm(feature_dim: int,
                       seq_len: int = 10,
                       filters: int = 64,
                       units: int = 128) -> Model:
        """CNN extracts local patterns; LSTM captures sequential dependencies."""
        model = keras.Sequential([
            layers.Input(shape=(seq_len, feature_dim)),
            layers.Conv1D(filters,   kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv1D(filters*2, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(units // 2),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1,  activation="sigmoid"),
        ], name="CNN_LSTM")
        return model

    # ─────────────────────────────── utilities ────────────────────────────────

    @staticmethod
    def compile(model: Model, lr: float = 1e-3) -> Model:
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        return model

    @staticmethod
    def get_callbacks(model_name: str = "model", patience: int = 10):
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                f"models/{model_name}_best.h5",
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

    @staticmethod
    def reshape_for_sequence(X: "np.ndarray", seq_len: int = 10):
        """
        Reshape flat tabular features into (N, seq_len, feature_dim) for LSTM / CNN-LSTM.
        Pads or truncates feature dimension to be divisible by seq_len.
        """
        import numpy as np
        n, d = X.shape
        feature_dim = d // seq_len
        X_trimmed   = X[:, : feature_dim * seq_len]
        return X_trimmed.reshape(n, seq_len, feature_dim)

    @staticmethod
    def build_from_name(name: str, input_dim: int) -> Model:
        """Factory helper — returns a model by string name."""
        builders = {
            "mlp":       lambda: BotDetectionModels.build_mlp(input_dim),
            "deep_mlp":  lambda: BotDetectionModels.build_deep_mlp(input_dim),
            "attention": lambda: BotDetectionModels.build_attention_mlp(input_dim),
            "transformer": lambda: BotDetectionModels.build_transformer(input_dim),
        }
        if name not in builders:
            raise ValueError(f"Unknown model '{name}'. Choose from: {list(builders)}")
        return builders[name]()