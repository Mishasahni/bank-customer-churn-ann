"""Shared configuration and helper functions for the churn project.

Every notebook and the Streamlit dashboard import from this module so that
paths, feature lists, preprocessing, evaluation and risk tiers are defined in
exactly one place.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths (relative to the project root, so the project works from any clone)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "churn_modelling.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]  # identifiers, not predictors
CATEGORICAL_FEATURES = ["Geography", "Gender"]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]  # already 0/1
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Risk tiers used in the business analysis and the dashboard
# (cut-offs on the predicted churn probability; see notebook 05)
# ---------------------------------------------------------------------------
RISK_TIERS = [("Low", 0.0, 0.30), ("Medium", 0.30, 0.60), ("High", 0.60, 1.01)]

# ---------------------------------------------------------------------------
# Plot styling (one palette used across all notebooks)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#2a78d6"  # blue: churn-rate bars, retained customers
COLOR_CHURN = "#eb6834"  # orange: churned customers
COLOR_MUTED = "#898781"  # gray: reference lines, secondary text
COLOR_GRID = "#e1e0d9"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}


def set_plot_style():
    """Apply a clean, readable matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "legend.frameon": False,
        }
    )


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
def load_raw_data():
    """Load the original Kaggle-style churn dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def load_train_test():
    """Load the fixed train/test split created in notebook 02.

    Returns X_train, X_test, y_train, y_test and the test-set customer IDs.
    """
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return (
        train[FEATURES],
        test[FEATURES],
        train[TARGET],
        test[TARGET],
        test["CustomerId"],
    )


def build_preprocessor():
    """Column transformer shared by every model.

    * Geography and Gender -> one-hot encoded (Gender becomes a single 0/1 column)
    * Numeric features -> standardised (mean 0, std 1); needed for logistic
      regression and the neural network, harmless for tree models
    * HasCrCard / IsActiveMember -> already 0/1, passed through unchanged
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def classification_metrics(y_true, churn_probability, threshold=0.5):
    """Standard metrics for the churn (positive) class at a given threshold."""
    y_pred = (np.asarray(churn_probability) >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, churn_probability),
    }


def assign_risk_tier(churn_probability):
    """Map churn probabilities to Low / Medium / High risk labels."""
    probabilities = np.asarray(churn_probability)
    bins = [low for _, low, _ in RISK_TIERS] + [RISK_TIERS[-1][2]]
    labels = [name for name, _, _ in RISK_TIERS]
    return pd.cut(probabilities, bins=bins, labels=labels, right=False)


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def churn_rate_table(df, column):
    """Churn rate and customer count for each value of `column`."""
    table = (
        df.groupby(column, observed=True)[TARGET]
        .agg(customers="size", churned="sum", churn_rate="mean")
        .reset_index()
    )
    table["churn_rate"] = (table["churn_rate"] * 100).round(1)
    return table


def plot_churn_rate(df, column, ax=None, title=None, xlabel=None):
    """Bar chart of churn rate (%) per category with customer counts.

    The dashed line marks the overall churn rate so each segment can be read
    as "above" or "below" average.
    """
    ax = ax or plt.gca()
    table = churn_rate_table(df, column)
    overall = df[TARGET].mean() * 100
    labels = table[column].astype(str)

    bars = ax.bar(labels, table["churn_rate"], color=COLOR_PRIMARY, width=0.6)
    ax.axhline(overall, color=COLOR_MUTED, linestyle="--", linewidth=1.2,
               label=f"Overall churn rate ({overall:.1f}%)")
    ax.legend(loc="best", fontsize=9)
    for bar, rate, n in zip(bars, table["churn_rate"], table["customers"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0,
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
        )
    ax.set_ylim(0, max(table["churn_rate"].max(), overall) * 1.3)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(xlabel or column)
    ax.set_title(title or f"Churn rate by {column}")
    return ax


def save_figure(fig, name):
    """Save a figure to reports/figures for use in the README."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=130)
