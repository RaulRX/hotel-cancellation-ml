from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn import set_config
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

set_config(transform_output="pandas")

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

LEAKAGE_COLUMNS = [
    "company",
    "arrival_date_year",
    "arrival_date_week_number",
    "reservation_status",
    "reservation_status_date",
]

RANDOM_STATE = 11


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate and incoherent rows. Applied only during training."""
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df = df[df["adr"] >= 0].reset_index(drop=True)
    guests = (
        df["adults"].fillna(0)
        + df["children"].fillna(0)
        + df["babies"].fillna(0)
    )
    df = df[guests > 0].reset_index(drop=True)
    return df


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace rare categories with a replacement label.

    Parameters
    ----------
    columns : list[str]
        Columns to apply grouping to.
    threshold : float
        Frequency threshold.
    replacement : str
        Label used for rare / unseen categories.
    inclusive : bool
        If True, keeps categories with frequency > threshold (strict).
        If False, keeps categories with frequency >= threshold.
        In both cases, everything below is grouped into `replacement`.
    """

    def __init__(self, columns, threshold=0.01, replacement="Others", inclusive=False):
        self.columns = columns
        self.threshold = threshold
        self.replacement = replacement
        self.inclusive = inclusive

    def fit(self, X, y=None):
        self._frequent = {}
        for col in self.columns:
            freqs = X[col].value_counts(normalize=True)
            if self.inclusive:
                self._frequent[col] = set(freqs[freqs > self.threshold].index)
            else:
                self._frequent[col] = set(freqs[freqs >= self.threshold].index)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            frequent = self._frequent[col]
            X[col] = X[col].where(X[col].isin(frequent), other=self.replacement)
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip values at the q-th quantile learned from training data."""

    def __init__(self, columns, q=0.99):
        self.columns = columns
        self.q = q

    def fit(self, X, y=None):
        self._upper = {col: X[col].quantile(self.q) for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].clip(upper=self._upper[col])
        return X


class CommonPreprocessor(BaseEstimator, TransformerMixin):
    """Steps common to all models.

    - Impute children NaN -> 0
    - Binarise agent -> agent_known, drop agent
    - Drop leakage columns
    - Group rare country (<1%)
    - Group rare distribution_channel (<1%)
    - Group rare market_segment (<=1%)
    """

    def fit(self, X, y=None):
        self._rare_country = RareCategoryGrouper(
            columns=["country"], threshold=0.01, replacement="Others", inclusive=False
        ).fit(X)
        self._rare_dist = RareCategoryGrouper(
            columns=["distribution_channel"], threshold=0.01, replacement="others", inclusive=False
        ).fit(X)
        self._rare_market = RareCategoryGrouper(
            columns=["market_segment"], threshold=0.01, replacement="Others", inclusive=True
        ).fit(X)
        return self

    def transform(self, X):
        X = X.copy()

        X["children"] = X["children"].fillna(0)

        X["agent_known"] = X["agent"].notna().astype(int)
        X.drop(columns=["agent"], inplace=True, errors="ignore")

        X.drop(columns=LEAKAGE_COLUMNS, inplace=True, errors="ignore")

        X = self._rare_country.transform(X)
        X = self._rare_dist.transform(X)
        X = self._rare_market.transform(X)

        return X


class DecisionTreePreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering specific to Decision Tree."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["has_special_request"] = (X["total_of_special_requests"].astype(int) > 0).astype(int)
        X.drop(columns=["total_of_special_requests"], inplace=True, errors="ignore")

        X["has_reserved_parking"] = (X["required_car_parking_spaces"].astype(int) > 0).astype(int)
        X.drop(columns=["required_car_parking_spaces"], inplace=True, errors="ignore")

        X["arrival_date_month"] = X["arrival_date_month"].map(MONTHS)

        X.drop(columns=["arrival_date_day_of_month"], inplace=True, errors="ignore")

        return X


class AutoOrdinalEncoder(BaseEstimator, TransformerMixin):
    """OrdinalEncoder over all object/category columns, auto-detected at fit time."""

    def fit(self, X, y=None):
        self._cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._enc.fit(X[self._cat_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self._cat_cols] = self._enc.transform(X[self._cat_cols])
        return X


class LightGBMPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering specific to LightGBM.

    - Groups rare reserved_room_type and assigned_room_type (<=1%).
    - Converts arrival_date_month to int.
    - Keeps arrival_date_day_of_month (only LightGBM retains it).
    """

    def fit(self, X, y=None):
        self._rare_room_types = RareCategoryGrouper(
            columns=["reserved_room_type", "assigned_room_type"],
            threshold=0.01,
            replacement="Others",
            inclusive=True,
        ).fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = self._rare_room_types.transform(X)
        X["arrival_date_month"] = X["arrival_date_month"].map(MONTHS)
        return X


class LGBOrdinalEncoder(BaseEstimator, TransformerMixin):
    """OrdinalEncoder over explicit categorical columns for LightGBM.

    After encoding, casts columns to dtype 'category' for LightGBM native handling.
    """

    _CAT_COLS = [
        "hotel", "meal", "country", "market_segment", "distribution_channel",
        "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type",
    ]

    def fit(self, X, y=None):
        self._cols = [c for c in self._CAT_COLS if c in X.columns]
        self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._enc.fit(X[self._cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self._cols] = self._enc.transform(X[self._cols])
        for col in self._cols:
            X[col] = X[col].astype("category")
        return X


class LogisticRegressionPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering specific to Logistic Regression."""

    def fit(self, X, y=None):
        self._clipper = QuantileClipper(
            columns=["children", "adults", "babies", "adr"], q=0.99
        ).fit(X)
        return self

    def transform(self, X):
        X = X.copy()

        X["reserved_assigned_room"] = (
            X["reserved_room_type"] == X["assigned_room_type"]
        ).astype(int)
        X.drop(columns=["assigned_room_type"], inplace=True, errors="ignore")

        X["has_special_request"] = (X["total_of_special_requests"].astype(int) > 0).astype(int)
        X.drop(columns=["total_of_special_requests"], inplace=True, errors="ignore")

        X["has_reserved_parking"] = (X["required_car_parking_spaces"].astype(int) > 0).astype(int)
        X.drop(columns=["required_car_parking_spaces"], inplace=True, errors="ignore")

        X["lead_time_log"] = np.log1p(X["lead_time"])
        X.drop(columns=["lead_time"], inplace=True, errors="ignore")

        month_num = X["arrival_date_month"].map(MONTHS)
        X["month_sin"] = np.sin(2 * np.pi * month_num / 12)
        X["month_cos"] = np.cos(2 * np.pi * month_num / 12)
        X.drop(columns=["arrival_date_month"], inplace=True, errors="ignore")

        X.drop(columns=["arrival_date_day_of_month"], inplace=True, errors="ignore")

        X = self._clipper.transform(X)

        return X


class LROHEEncoder(BaseEstimator, TransformerMixin):
    """OneHotEncoder (drop first) over explicit categorical columns for LR."""

    _CAT_COLS = [
        "hotel", "meal", "country", "market_segment", "distribution_channel",
        "reserved_room_type", "deposit_type", "customer_type",
    ]

    def fit(self, X, y=None):
        self._cols = [c for c in self._CAT_COLS if c in X.columns]
        self._enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self._enc.fit(X[self._cols])
        return self

    def transform(self, X):
        X = X.copy()
        encoded: np.ndarray = np.asarray(self._enc.transform(X[self._cols]))
        ohe_df = pd.DataFrame(
            encoded,
            columns=self._enc.get_feature_names_out(self._cols),
            index=X.index,
        )
        return pd.concat([X.drop(columns=self._cols), ohe_df], axis=1)


def build_decision_tree_pipeline(
    criterion: Literal["gini", "entropy", "log_loss"] = "entropy",
    max_depth: int = 10,
    min_samples_leaf: int = 5,
    min_samples_split: int = 8,
    class_weight: Literal["balanced"] | None = "balanced",
) -> Pipeline:
    return Pipeline([
        ("common", CommonPreprocessor()),
        ("specific", DecisionTreePreprocessor()),
        ("encoder", AutoOrdinalEncoder()),
        ("model", DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )),
    ])


def build_lightgbm_pipeline(
    num_leaves: int = 60,
    learning_rate: float = 0.1,
    n_estimators: int = 1100,
) -> Pipeline:
    return Pipeline([
        ("common", CommonPreprocessor()),
        ("specific", LightGBMPreprocessor()),
        ("encoder", LGBOrdinalEncoder()),
        ("model", LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective="binary",
            random_state=RANDOM_STATE,
            verbose=-1,
        )),
    ])


def build_logistic_regression_pipeline(
    C: float = 0.05,
    max_iter: int = 800,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "liblinear",
    class_weight: Literal["balanced"] | None = "balanced",
) -> Pipeline:
    return Pipeline([
        ("common", CommonPreprocessor()),
        ("specific", LogisticRegressionPreprocessor()),
        ("encoder", LROHEEncoder()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty=penalty,
            solver=solver,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )),
    ])


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression_pipeline,
    "decision_tree": build_decision_tree_pipeline,
    "lightgbm": build_lightgbm_pipeline,
    # TODO: random_forest
    # TODO: neural_network
}


def available_models() -> list[str]:
    return list(MODEL_BUILDERS.keys())
