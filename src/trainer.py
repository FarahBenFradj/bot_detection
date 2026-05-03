"""
src/trainer.py
Training and evaluation pipeline for Bot Detection.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — safe on servers without a display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)

from .models import BotDetectionModels


class BotDetectionTrainer:
    """End-to-end training, evaluation, and artefact saving."""

    def __init__(self, model_type: str = "deep_mlp"):
        if model_type not in ("mlp", "deep_mlp", "attention", "transformer"):
            raise ValueError(f"Unknown model_type '{model_type}'")
        self.model_type = model_type
        self.model      = None
        self.history    = None
        self.results: dict = {}

    # ──────────────────────────────── build ───────────────────────────────────

    def build(self, input_dim: int, lr: float = 1e-3):
        """Instantiate and compile the chosen model."""
        self.model = BotDetectionModels.build_from_name(self.model_type, input_dim)
        self.model = BotDetectionModels.compile(self.model, lr=lr)
        print(f"\n[{self.model_type.upper()}] Parameters: "
              f"{self.model.count_params():,}")
        self.model.summary()
        return self.model

    # ──────────────────────────────── train ───────────────────────────────────

    def train(self,
              X_train, y_train,
              X_val,   y_val,
              epochs:     int = 100,
              batch_size: int = 64):
        """Fit the model; saves best weights to models/."""
        Path("models").mkdir(exist_ok=True)
        callbacks = BotDetectionModels.get_callbacks(
            model_name=f"bot_detector_{self.model_type}"
        )
        print(f"\nTraining {self.model_type} …")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    # ──────────────────────────────── evaluate ────────────────────────────────

    def evaluate(self, X_test, y_test, threshold: float = 0.5) -> dict:
        """Compute and print all evaluation metrics."""
        y_proba = self.model.predict(X_test).flatten()
        y_pred  = (y_proba > threshold).astype(int)

        self.results = {
            "model":     self.model_type,
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred,    zero_division=0)),
            "f1_score":  float(f1_score(y_test, y_pred,        zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_true":    y_test,
            "y_pred":    y_pred,
            "y_proba":   y_proba,
        }

        print("\n" + "=" * 50)
        print(f"{'EVALUATION — ' + self.model_type.upper():^50}")
        print("=" * 50)
        for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            print(f"  {k:<12} {self.results[k]:.4f}")
        print("\nConfusion matrix:\n", self.results["confusion_matrix"])
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

        return self.results

    # ─────────────────────────────── plotting ─────────────────────────────────

    def plot_training(self, out_dir: str = "results") -> None:
        """Save a 2×2 grid of training curves."""
        if self.history is None:
            print("No training history — run train() first.")
            return
        Path(out_dir).mkdir(exist_ok=True)
        h = self.history.history

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        pairs = [
            ("accuracy", "Accuracy"),
            ("loss",     "Loss"),
            ("precision","Precision"),
            ("auc",      "AUC"),
        ]
        for ax, (key, title) in zip(axes.flat, pairs):
            ax.plot(h[key],         label="Train",      linewidth=2)
            ax.plot(h[f"val_{key}"],label="Validation", linewidth=2, linestyle="--")
            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.legend(); ax.grid(alpha=0.4)

        fig.suptitle(f"{self.model_type.upper()} — Training History", fontsize=14)
        plt.tight_layout()
        path = f"{out_dir}/training_{self.model_type}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved → {path}")

    def plot_evaluation(self, out_dir: str = "results") -> None:
        """Save confusion matrix, ROC, and PR curves."""
        if not self.results:
            print("No results — run evaluate() first.")
            return
        Path(out_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # Confusion matrix
        sns.heatmap(
            self.results["confusion_matrix"],
            annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Human", "Bot"],
            yticklabels=["Human", "Bot"],
        )
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

        # ROC
        fpr, tpr, _ = roc_curve(self.results["y_true"], self.results["y_proba"])
        axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {self.results['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(); axes[1].grid(alpha=0.4)

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(self.results["y_true"], self.results["y_proba"])
        axes[2].plot(rec, prec, lw=2)
        axes[2].set_title("Precision-Recall Curve")
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
        axes[2].grid(alpha=0.4)

        fig.suptitle(f"{self.model_type.upper()} — Evaluation", fontsize=14)
        plt.tight_layout()
        path = f"{out_dir}/evaluation_{self.model_type}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Evaluation plots saved → {path}")

    # ──────────────────────────────── save ────────────────────────────────────

    def save_metrics(self, out_dir: str = "results") -> None:
        """Persist scalar metrics to JSON."""
        Path(out_dir).mkdir(exist_ok=True)
        scalar_keys = ("model", "accuracy", "precision", "recall", "f1_score", "roc_auc")
        payload = {k: self.results[k] for k in scalar_keys if k in self.results}
        path = f"{out_dir}/metrics_{self.model_type}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics saved → {path}")