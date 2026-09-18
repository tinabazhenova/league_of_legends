# League of Legends Match Predictor

A cleaned, reproducible version of the original course notebook for predicting League of Legends match outcomes with logistic regression in PyTorch.

## What changed

The original notebook worked as a step-by-step exercise, but several pieces were duplicated or inconsistent. This repository consolidates the reusable code into one Python module and keeps the notebook as a thin demonstration layer.

Key fixes include:

- Renamed the model class to `LogisticRegressionModel` and centralized training/evaluation logic.
- Added deterministic seeding for reproducible model initialization.
- Replaced `sigmoid + BCELoss` during training with the numerically stable `BCEWithLogitsLoss`; sigmoid is applied only when probabilities are needed.
- Added a validation split so learning-rate tuning no longer selects hyperparameters using the test set.
- Fixed the save/load mismatch from the original notebook, which saved the baseline model under an L2 filename and then compared it with the L2 model.
- Saves the fitted `StandardScaler` alongside the model so future inference uses the same preprocessing.
- Added a CLI, smoke tests, and GitHub Actions CI.
- Removed the notebook's package-install cell, repeated imports, course hints, and third-party course branding from the cleaned version.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/                  # generated metrics, plots, model, scaler
├── data/                       # optional local CSV files
├── notebooks/
│   └── league_of_legends_match_predictor.ipynb
├── src/lol_match_predictor/
│   ├── __init__.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the project

The CLI defaults to the dataset URL referenced by the original notebook:

```bash
lol-match-predictor
```

Or provide a local CSV:

```bash
lol-match-predictor --data-source data/league-of-legends-data-large.csv
```

Equivalent module invocation:

```bash
python -m lol_match_predictor.pipeline --data-source data/league-of-legends-data-large.csv
```

Useful options:

```bash
lol-match-predictor \
  --epochs 1000 \
  --tuning-epochs 100 \
  --learning-rates 0.01 0.05 0.1 \
  --weight-decay 0.01 \
  --seed 42
```

Generated outputs are written to `artifacts/` and intentionally ignored by Git. They include the trained model, fitted scaler, metrics JSON, confusion matrix, ROC curve, and feature-importance files.

## Modeling workflow

1. Load the CSV and separate the `win` target.
2. Create stratified train, validation, and test partitions.
3. Fit `StandardScaler` on training features only.
4. Tune the learning rate on validation accuracy.
5. Train the final L2-regularized logistic regression model using the selected rate.
6. Evaluate once on the held-out test set.
7. Save the model, scaler, metrics, plots, and coefficient-based feature importance.

The final test set is not used for hyperparameter selection.

## Tests

The test suite uses synthetic data, so CI does not depend on the remote course dataset being available.

```bash
pytest
```

## Dataset

The project is configured for the same League of Legends CSV referenced in the original notebook. You can also download that CSV separately and place it under `data/`; CSV data is ignored by default to keep the repository lightweight.

## Notes

The source notebook reported roughly 50% test accuracy in its saved outputs, so this should be treated primarily as a machine-learning workflow demonstration rather than a production match-prediction system. Re-running the cleaned pipeline may produce different metrics because model selection and reproducibility issues were corrected.
