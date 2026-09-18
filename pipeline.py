"""Training pipeline for a League of Legends binary match-outcome predictor."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

DEFAULT_DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"
)
DEFAULT_TARGET = "win"
DEFAULT_SEED = 42


@dataclass
class PreparedData:
    """Scaled train/validation/test tensors and preprocessing metadata."""

    X_train: torch.Tensor
    X_val: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]
    scaler: StandardScaler


class LogisticRegressionModel(nn.Module):
    """Single-layer logistic regression model that returns logits."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Make model initialization and training reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(source: str | Path = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load the CSV dataset from a local path or URL and validate the target."""

    data = pd.read_csv(source)
    if DEFAULT_TARGET not in data.columns:
        raise ValueError(f"Dataset must include a '{DEFAULT_TARGET}' target column.")
    if data.empty:
        raise ValueError("Dataset is empty.")
    return data


def prepare_data(
    data: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = DEFAULT_SEED,
) -> PreparedData:
    """Split data into train/validation/test sets and standardize features.

    The validation split is used for hyperparameter selection; the test split is
    held back until final evaluation. The scaler is fitted only on training data.
    """

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to < 1.")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        non_numeric = X.columns[~X.dtypes.apply(pd.api.types.is_numeric_dtype)].tolist()
        raise ValueError(f"All features must be numeric. Non-numeric columns: {non_numeric}")

    stratify = y if y.nunique() > 1 else None
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    stratify_dev = y_dev if y_dev.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=stratify_dev,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def x_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32)

    def y_tensor(values: pd.Series) -> torch.Tensor:
        return torch.tensor(values.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    return PreparedData(
        X_train=x_tensor(X_train_scaled),
        X_val=x_tensor(X_val_scaled),
        X_test=x_tensor(X_test_scaled),
        y_train=y_tensor(y_train),
        y_val=y_tensor(y_val),
        y_test=y_tensor(y_test),
        feature_names=X.columns.tolist(),
        scaler=scaler,
    )


def train_model(
    model: LogisticRegressionModel,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> list[float]:
    """Train a model with SGD and numerically stable binary cross-entropy."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: list[float] = []
    report_every = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

        if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    return history


def predict_probabilities(model: LogisticRegressionModel, X: torch.Tensor) -> np.ndarray:
    """Return positive-class probabilities."""

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X)).cpu().numpy().ravel()
    return probabilities


def evaluate_model(
    model: LogisticRegressionModel,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict:
    """Evaluate accuracy, ROC-AUC, confusion matrix, and classification report."""

    y_true = y.cpu().numpy().ravel().astype(int)
    y_prob = predict_probabilities(model, X)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else None
    )
    return metrics


def tune_learning_rate(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    input_dim: int,
    learning_rates: Iterable[float] = (0.01, 0.05, 0.1),
    epochs: int = 100,
    weight_decay: float = 0.01,
    seed: int = DEFAULT_SEED,
) -> tuple[float, dict[float, float]]:
    """Choose a learning rate using validation accuracy, never test accuracy."""

    results: dict[float, float] = {}
    for learning_rate in learning_rates:
        set_seed(seed)
        candidate = LogisticRegressionModel(input_dim)
        train_model(
            candidate,
            X_train,
            y_train,
            learning_rate=float(learning_rate),
            epochs=epochs,
            weight_decay=weight_decay,
        )
        results[float(learning_rate)] = evaluate_model(candidate, X_val, y_val)["accuracy"]

    best_learning_rate = max(results, key=results.get)
    return best_learning_rate, results


def feature_importance(
    model: LogisticRegressionModel,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return coefficients sorted by absolute magnitude."""

    weights = model.linear.weight.detach().cpu().numpy().ravel()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": weights,
        }
    )
    importance["absolute_coefficient"] = importance["coefficient"].abs()
    return importance.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def save_artifacts(
    model: LogisticRegressionModel,
    scaler: StandardScaler,
    feature_names: list[str],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist both the model and scaler required for inference."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.pt"
    scaler_path = output_path / "scaler.joblib"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "feature_names": feature_names,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def load_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
) -> tuple[LogisticRegressionModel, StandardScaler, list[str]]:
    """Load a saved model, scaler, and feature ordering."""

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    feature_names = list(checkpoint["feature_names"])
    model = LogisticRegressionModel(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, feature_names


def save_evaluation_outputs(
    model: LogisticRegressionModel,
    metrics: dict,
    feature_names: list[str],
    output_dir: str | Path,
) -> None:
    """Save metrics, plots, and feature-importance data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"y_true", "y_prob", "y_pred"}
    }
    (output_path / "metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2),
        encoding="utf-8",
    )

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    y_prob = metrics["y_prob"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    if np.unique(y_true).size > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path / "roc_curve.png", dpi=160)
        plt.close(fig)

    importance = feature_importance(model, feature_names)
    importance.to_csv(output_path / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["feature"], ordered["coefficient"])
    ax.set_xlabel("Coefficient")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path / "feature_importance.png", dpi=160)
    plt.close(fig)


def run_pipeline(
    data_source: str | Path = DEFAULT_DATA_URL,
    *,
    epochs: int = 1000,
    tuning_epochs: int = 100,
    weight_decay: float = 0.01,
    learning_rates: Iterable[float] = (0.01, 0.05, 0.1),
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = "artifacts",
) -> dict:
    """Run data prep, validation-based tuning, final training, and test evaluation."""

    set_seed(seed)
    data = load_dataset(data_source)
    prepared = prepare_data(data, random_state=seed)
    input_dim = prepared.X_train.shape[1]

    best_lr, tuning_results = tune_learning_rate(
        prepared.X_train,
        prepared.y_train,
        prepared.X_val,
        prepared.y_val,
        input_dim=input_dim,
        learning_rates=learning_rates,
        epochs=tuning_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )

    # Hyperparameters are fixed now; validation data can be included in final training.
    X_final_train = torch.cat([prepared.X_train, prepared.X_val], dim=0)
    y_final_train = torch.cat([prepared.y_train, prepared.y_val], dim=0)

    set_seed(seed)
    final_model = LogisticRegressionModel(input_dim)
    train_model(
        final_model,
        X_final_train,
        y_final_train,
        learning_rate=best_lr,
        epochs=epochs,
        weight_decay=weight_decay,
        verbose=True,
    )

    metrics = evaluate_model(final_model, prepared.X_test, prepared.y_test)
    model_path, scaler_path = save_artifacts(
        final_model,
        prepared.scaler,
        prepared.feature_names,
        output_dir,
    )
    save_evaluation_outputs(final_model, metrics, prepared.feature_names, output_dir)

    summary = {
        "best_learning_rate": best_lr,
        "validation_accuracy_by_learning_rate": tuning_results,
        "test_accuracy": metrics["accuracy"],
        "test_roc_auc": metrics["roc_auc"],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the League of Legends match predictor."
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_URL,
        help="CSV path or URL. Defaults to the dataset used by the original notebook.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--tuning-epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default="artifacts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(
        data_source=args.data_source,
        epochs=args.epochs,
        tuning_epochs=args.tuning_epochs,
        weight_decay=args.weight_decay,
        learning_rates=args.learning_rates,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("\nRun summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
