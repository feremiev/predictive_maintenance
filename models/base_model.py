from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from pathlib import Path
from typing import Callable, Sequence, Union

from sklearn.base import clone

from preprocessors.cmapss_preprocessor import CMapssPreprocessor

from utils.s_score_parameter import PHMMetrics

class TimeSeriesRegressionModel:
    """
    Regression class for datasets containing multiple independent time series.

    Example
    -------
    Each ``unique_motor_id`` represents one complete engine time series.

    The train/validation split is performed using complete groups. Therefore,
    observations from the same motor cannot appear in both training and
    validation datasets.

    This class treats every row as one tabular observation. It is appropriate
    for models such as Random Forest, Extra Trees, Gradient Boosting, Ridge,
    Linear Regression and SVR.

    For true sequence models such as LSTM, GRU or CNN, use a separate class
    that creates sliding windows.
    """

    SUPPORTED_MODELS = {
        "linear",
        "ridge",
        "elastic_net",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "svr",
        "xgboost"
    }

    CMAPSS_COLUMN_NAMES = (
        [
            "unit_number",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3",
        ]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        group_column: str,
        time_column: Optional[str] = None,
        model_name: str = "random_forest",
        model: Optional[RegressorMixin] = None,
        validation_group_count: Optional[int] = None,
        validation_group_size: float = 0.2,
        group_selection: str = "random",
        feature_columns: Optional[Sequence[str]] = None,
        columns_to_drop: Optional[list[str]] = None,
        random_state: int = 42,
        model_params: Optional[dict[str, Any]] = None,
        preprocessor=None,
    ) -> None:
        """
        Parameters
        ----------
        df:
            Complete source DataFrame.

        target_column:
            Column to predict, for example ``RUL``.

        group_column:
            Column identifying each independent series, for example
            ``unique_motor_id``.

        time_column:
            Column defining chronological order inside each series, for
            example ``cycle``.

        model_name:
            One of:

            - ``linear``
            - ``ridge``
            - ``elastic_net``
            - ``random_forest``
            - ``extra_trees``
            - ``hist_gradient_boosting``
            - ``svr``

        model:
            Optional custom scikit-learn compatible regression model.
            When provided, ``model_name`` is ignored.

        validation_group_count:
            Exact number of complete series to use for validation.

            Example: ``10`` reserves ten complete motors.

        validation_group_size:
            Fraction of complete series used for validation when
            ``validation_group_count`` is not provided.

        group_selection:
            Strategy used to choose validation groups:

            - ``random``: randomly select complete groups.
            - ``last``: use the last groups after sorting.
            - ``first``: use the first groups after sorting.

        columns_to_drop:
            Additional columns that should not be used as model features.

        random_state:
            Seed used for reproducible splits and compatible models.

        model_params:
            Optional parameters passed to the selected estimator.
        """
        self.df = df.copy()
        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column
        self.model_name = model_name.lower()
        self.custom_model = model
        self.validation_group_count = validation_group_count
        self.validation_group_size = validation_group_size
        self.group_selection = group_selection.lower()
        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else None
        )
        self.columns_to_drop = columns_to_drop or []
        self.random_state = random_state
        self.model_params = model_params or {}

        # Fitted pipeline
        self.pipeline: Optional[Pipeline] = None

        # Train/validation datasets
        self.X_train: Optional[pd.DataFrame] = None
        self.X_validation: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_validation: Optional[pd.Series] = None

        # Group information
        self.groups_train: Optional[pd.Series] = None
        self.groups_validation: Optional[pd.Series] = None
        self.train_group_ids: Optional[np.ndarray] = None
        self.validation_group_ids: Optional[np.ndarray] = None

        # Original DataFrame indexes
        self.train_index: Optional[pd.Index] = None
        self.validation_index: Optional[pd.Index] = None

        # Predictions
        self.y_train_pred: Optional[np.ndarray] = None
        self.y_validation_pred: Optional[np.ndarray] = None

        # Backwards-compatible alias for validation predictions
        self.y_pred: Optional[np.ndarray] = None

        # Metrics
        self.train_metrics: Optional[dict[str, float]] = None
        self.validation_metrics: Optional[dict[str, float]] = None

        # Backwards-compatible alias for validation metrics
        self.metrics: Optional[dict[str, float]] = None

        # Cross-validation
        self.cv_results: Optional[dict[str, np.ndarray]] = None
        self.cv_summary: Optional[dict[str, float]] = None

        # Group-aware learning-curve results.
        self.learning_curve_results: Optional[pd.DataFrame] = None

        self._validate_configuration()

        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else CMapssPreprocessor()
        )

        if (
            self.feature_columns is not None
            and hasattr(self.preprocessor, "set_params")
        ):
            self.preprocessor.set_params(
                feature_columns=self.feature_columns
            )

        self.fitted_preprocessor = None

    # ==========================================================
    # Validation and preparation
    # ==========================================================

    def _validate_configuration(self) -> None:
        required_columns = {
            self.target_column,
            self.group_column,
        }

        if self.time_column is not None:
            required_columns.add(self.time_column)

        missing_columns = required_columns.difference(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if (
            self.custom_model is None
            and self.model_name not in self.SUPPORTED_MODELS
        ):
            raise ValueError(
                f"Unsupported model '{self.model_name}'. "
                f"Available models: {sorted(self.SUPPORTED_MODELS)}"
            )

        if self.group_selection not in {"random", "first", "last"}:
            raise ValueError(
                "group_selection must be 'random', 'first', or 'last'."
            )

        if not 0 < self.validation_group_size < 1:
            raise ValueError(
                "validation_group_size must be greater than 0 and smaller than 1."
            )

        if (
            self.validation_group_count is not None
            and self.validation_group_count <= 0
        ):
            raise ValueError(
                "validation_group_count must be greater than zero."
            )

        if self.df[self.target_column].isna().any():
            raise ValueError(
                f"The target column '{self.target_column}' contains "
                "null values."
            )

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the DataFrame sorted by group and time.
        """
        data = self.df.copy()

        if self.time_column is not None:
            data = data.sort_values(
                by=[self.group_column, self.time_column]
            )

        return data

    def _feature_columns(self) -> list[str]:
        """
        Return the raw features expected by the preprocessor.
        """

        if self.preprocessor is not None:
            expected = list(
                self.preprocessor.correct_order_
            )

            condition_encoder = getattr(
                self.preprocessor,
                "operating_condition_encoder",
                None,
            )

            if condition_encoder is not None:
                condition_column = condition_encoder.column

                if condition_column not in expected:
                    expected.append(condition_column)

            missing = set(expected).difference(
                self.df.columns
            )

            if missing:
                raise ValueError(
                    "The DataFrame is missing required features: "
                    f"{sorted(missing)}"
                )

            return expected

        excluded_columns = {
            self.target_column,
            self.group_column,
            *self.columns_to_drop,
        }

        return [
            column
            for column in self.df.columns
            if column not in excluded_columns
        ]
    # ==========================================================
    # Model factory
    # ==========================================================

    def _create_estimator(self) -> RegressorMixin:
        """
        Create the selected regression estimator.
        """
        if self.custom_model is not None:
            return self.custom_model

        params = self.model_params.copy()

        if self.model_name == "linear":
            return LinearRegression(**params)

        if self.model_name == "ridge":
            default_params = {
                "alpha": 1.0,
            }
            default_params.update(params)

            return Ridge(**default_params)

        if self.model_name == "elastic_net":
            default_params = {
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "max_iter": 10_000,
                "random_state": self.random_state,
            }
            default_params.update(params)

            return ElasticNet(**default_params)

        if self.model_name == "random_forest":
            default_params = {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            default_params.update(params)

            return RandomForestRegressor(**default_params)

        if self.model_name == "extra_trees":
            default_params = {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            default_params.update(params)

            return ExtraTreesRegressor(**default_params)

        if self.model_name == "hist_gradient_boosting":
            default_params = {
                "learning_rate": 0.08,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "l2_regularization": 1.0,
                "random_state": self.random_state,
            }
            default_params.update(params)

            return HistGradientBoostingRegressor(**default_params)

        if self.model_name == "svr":
            default_params = {
                "kernel": "rbf",
                "C": 10.0,
                "epsilon": 0.1,
                "gamma": "scale",
            }
            default_params.update(params)

            return SVR(**default_params)

        if self.model_name == "xgboost":
            default_params = {
                "objective": "reg:squarederror",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "n_jobs": -1,
                "tree_method": "hist",
                "eval_metric": "mae",
            }

            default_params.update(params)

            return XGBRegressor(**default_params)

        raise ValueError(
            f"Could not create model '{self.model_name}'."
        )


    def _model_requires_scaling(self) -> bool:
        """
        Return True for models that normally benefit from scaling.
        """
        if self.custom_model is not None:
            # For custom models, scaling is enabled by default.
            # Modify this method if the custom estimator does not need it.
            return True

        return self.model_name in {
            "linear",
            "ridge",
            "elastic_net",
            "svr",
        }

    def _create_pipeline(self) -> Pipeline:
        """
        Create the model pipeline using the custom C-MAPSS preprocessor.
        """
        estimator = self._create_estimator()

        if self.preprocessor is None:
            return Pipeline([
                (
                    "model",
                    estimator,
                )
            ])

        return Pipeline([
            (
                "preprocessor",
                clone(self.preprocessor),
            ),
            (
                "model",
                estimator,
            ),
        ])
    # ==========================================================
    # Train/validation split by complete series
    # ==========================================================

    def split_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Split the dataset using complete groups.

        No group can appear in both train and validation.
        """
        data = self._prepare_dataframe()
        feature_columns = self._feature_columns()

        if not feature_columns:
            raise ValueError(
                "No model feature columns remain after exclusions."
            )

        group_ids = (
            data[self.group_column]
            .drop_duplicates()
            .to_numpy()
        )

        group_count = len(group_ids)

        if group_count < 2:
            raise ValueError(
                "At least two unique groups are required."
            )

        if self.validation_group_count is not None:
            number_of_validation_groups = self.validation_group_count
        else:
            number_of_validation_groups = max(
                1,
                int(np.ceil(group_count * self.validation_group_size)),
            )

        if number_of_validation_groups >= group_count:
            raise ValueError(
                "The number of validation groups must be smaller than "
                "the total number of groups."
            )

        if self.group_selection == "random":
            rng = np.random.default_rng(self.random_state)

            self.validation_group_ids = rng.choice(
                group_ids,
                size=number_of_validation_groups,
                replace=False,
            )

        elif self.group_selection == "last":
            self.validation_group_ids = group_ids[-number_of_validation_groups:]

        else:
            self.validation_group_ids = group_ids[:number_of_validation_groups]

        validation_mask = data[self.group_column].isin(
            self.validation_group_ids
        )
        train_mask = ~validation_mask

        self.train_group_ids = (
            data.loc[train_mask, self.group_column]
            .drop_duplicates()
            .to_numpy()
        )

        self.X_train = data.loc[
            train_mask,
            feature_columns,
        ].copy()

        self.X_validation = data.loc[
            validation_mask,
            feature_columns,
        ].copy()

        self.y_train = data.loc[
            train_mask,
            self.target_column,
        ].copy()

        self.y_validation = data.loc[
            validation_mask,
            self.target_column,
        ].copy()

        self.groups_train = data.loc[
            train_mask,
            self.group_column,
        ].copy()

        self.groups_validation = data.loc[
            validation_mask,
            self.group_column,
        ].copy()

        self.train_index = self.X_train.index
        self.validation_index = self.X_validation.index

        overlap = set(self.train_group_ids).intersection(
            set(self.validation_group_ids)
        )

        if overlap:
            raise RuntimeError(
                f"Data leakage detected. Shared groups: {overlap}"
            )

        print(
            f"Training series: {len(self.train_group_ids)} | "
            f"Validation series: {len(self.validation_group_ids)}"
        )

        print(
            f"Training rows: {len(self.X_train):,} | "
            f"Validation rows: {len(self.X_validation):,}"
        )

        return (
            self.X_train,
            self.X_validation,
            self.y_train,
            self.y_validation,
        )

    # ==========================================================
    # Training and evaluation
    # ==========================================================

    def train(self) -> dict[str, dict[str, float]]:
        """
        Fit the model and calculate both train and validation metrics.

        Returns
        -------
        dict
            Dictionary containing train and validation metrics.
        """
        if self.X_train is None:
            self.split_data()

        self.pipeline = self._create_pipeline()

        self.pipeline.fit(
            self.X_train,
            self.y_train,
        )

        self.y_train_pred = self.pipeline.predict(
            self.X_train
        )

        self.y_validation_pred = self.pipeline.predict(
            self.X_validation
        )

        # Backwards-compatible alias
        self.y_pred = self.y_validation_pred

        self.train_metrics = self._calculate_metrics(
            self.y_train,
            self.y_train_pred,
        )

        self.validation_metrics = self._calculate_metrics(
            self.y_validation,
            self.y_validation_pred,
        )

        # Backwards-compatible alias
        self.metrics = self.validation_metrics

        return {
            "train": self.train_metrics.copy(),
            "validation": self.validation_metrics.copy(),
        }

    def evaluate(
        self,
    ) -> dict[str, dict[str, float]]:
        """
        Return metrics for the fitted model.

        If predictions do not exist yet, they are generated.
        """
        self._ensure_fitted()

        if self.y_train_pred is None:
            self.y_train_pred = self.pipeline.predict(
                self.X_train
            )

        if self.y_validation_pred is None:
            self.y_validation_pred = self.pipeline.predict(
                self.X_validation
            )

        self.y_pred = self.y_validation_pred

        self.train_metrics = self._calculate_metrics(
            self.y_train,
            self.y_train_pred,
        )

        self.validation_metrics = self._calculate_metrics(
            self.y_validation,
            self.y_validation_pred,
        )

        self.metrics = self.validation_metrics

        return {
            "train": self.train_metrics.copy(),
            "validation": self.validation_metrics.copy(),
        }

    @staticmethod
    def _calculate_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate common regression metrics.
        """
        y_true_array = np.asarray(y_true, dtype=float)
        y_pred_array = np.asarray(y_pred, dtype=float)

        mae = mean_absolute_error(
            y_true_array,
            y_pred_array,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true_array,
                y_pred_array,
            )
        )

        r2 = r2_score(
            y_true_array,
            y_pred_array,
        )

        denominator = np.where(
            y_true_array == 0,
            np.nan,
            np.abs(y_true_array),
        )

        mape = (
            np.nanmean(
                np.abs(y_true_array - y_pred_array)
                / denominator
            )
            * 100
        )

        residuals = y_true_array - y_pred_array

        bias = np.mean(residuals)

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
            "Bias": float(bias),
            "NASA_SCORE": float(
                        PHMMetrics.nasa_score(
                                y_true_array,
                                y_pred_array
                            )
                        )
        }

    def get_overfitting_summary(self) -> dict[str, Any]:
        """
        Compare training and validation metrics.

        A large gap between training and validation performance may indicate
        overfitting.
        """
        self._ensure_trained()

        mae_gap = (
            self.validation_metrics["MAE"]
            - self.train_metrics["MAE"]
        )

        rmse_gap = (
            self.validation_metrics["RMSE"]
            - self.train_metrics["RMSE"]
        )

        r2_gap = (
            self.train_metrics["R2"]
            - self.validation_metrics["R2"]
        )

        train_mae = self.train_metrics["MAE"]

        relative_mae_increase = (
            mae_gap / train_mae
            if train_mae != 0
            else np.inf
        )

        if relative_mae_increase < 0.20:
            interpretation = "Low train/validation error gap."
        elif relative_mae_increase < 0.50:
            interpretation = (
                "Moderate train/validation error gap. "
                "Check for mild overfitting."
            )
        else:
            interpretation = (
                "Large train/validation error gap. "
                "The model may be overfitting."
            )

        return {
            "train_MAE": self.train_metrics["MAE"],
            "validation_MAE": self.validation_metrics["MAE"],
            "MAE_gap": float(mae_gap),
            "RMSE_gap": float(rmse_gap),
            "R2_gap": float(r2_gap),
            "relative_MAE_increase": float(
                relative_mae_increase
            ),
            "interpretation": interpretation,
        }

    # ==========================================================
    # Cross-validation by complete series
    # ==========================================================

    def cross_validation(
        self,
        cv: int = 5,
    ) -> dict[str, float]:
        """
        Perform GroupKFold cross-validation.

        Each complete motor is kept in one validation fold.
        """
        data = self._prepare_dataframe()
        feature_columns = self._feature_columns()

        X = data[feature_columns]
        y = data[self.target_column]
        groups = data[self.group_column]

        unique_groups = groups.nunique()

        if cv > unique_groups:
            raise ValueError(
                f"cv={cv}, but only {unique_groups} groups "
                "are available."
            )

        group_cv = GroupKFold(
            n_splits=cv
        )

        pipeline = self._create_pipeline()

        self.cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            groups=groups,
            cv=group_cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
                "nasa_score": make_scorer(
                    PHMMetrics.nasa_score,
                    greater_is_better=False,
                ),

            },
            return_train_score=True,
            n_jobs=-1,
            error_score="raise",
        )

        train_mae = -self.cv_results["train_mae"]
        validation_mae = -self.cv_results["validation_mae"]

        train_rmse = -self.cv_results["train_rmse"]
        validation_rmse = -self.cv_results["validation_rmse"]

        train_r2 = self.cv_results["train_r2"]
        validation_r2 = self.cv_results["validation_r2"]
        train_nasa_score = -self.cv_results["train_nasa_score"]
        validation_nasa_score = -self.cv_results["validation_nasa_score"]

        self.cv_summary = {
            "CV Train MAE mean": float(train_mae.mean()),
            "CV Train MAE std": float(train_mae.std()),
            "CV Validation MAE mean": float(validation_mae.mean()),
            "CV Validation MAE std": float(validation_mae.std()),
            "CV Train RMSE mean": float(train_rmse.mean()),
            "CV Train RMSE std": float(train_rmse.std()),
            "CV Validation RMSE mean": float(validation_rmse.mean()),
            "CV Validation RMSE std": float(validation_rmse.std()),
            "CV Train R2 mean": float(train_r2.mean()),
            "CV Train R2 std": float(train_r2.std()),
            "CV Validation R2 mean": float(validation_r2.mean()),
            "CV Validation R2 std": float(validation_r2.std()),
             "CV Train NASA Score mean": float(
                    train_nasa_score.mean()
                ),
                "CV Train NASA Score std": float(
                    train_nasa_score.std()
                ),
                "CV Validation NASA Score mean": float(
                    validation_nasa_score.mean()
                ),
                "CV Validation NASA Score std": float(
                    validation_nasa_score.std()
                ),
        }

        return self.cv_summary.copy()

    def calculate_learning_curve(
        self,
        train_fractions: Sequence[float] = (
            0.20,
            0.40,
            0.60,
            0.80,
            1.00,
        ),
    ) -> pd.DataFrame:
        """
        Fit the same pipeline using progressively more complete training motors.

        The validation motors remain fixed. A persistent gap between training
        and validation error indicates overfitting.
        """
        if self.X_train is None or self.X_validation is None:
            self.split_data()

        if self.groups_train is None:
            raise RuntimeError(
                "Training groups are unavailable. Run split_data() first."
            )

        fractions = sorted(
            {
                float(value)
                for value in train_fractions
                if 0 < float(value) <= 1
            }
        )

        if not fractions:
            raise ValueError(
                "train_fractions must contain values between 0 and 1."
            )

        available_groups = np.asarray(
            self.train_group_ids
        )

        if len(available_groups) == 0:
            raise RuntimeError(
                "No training motors are available."
            )

        rows: list[dict[str, float | int]] = []

        for fraction in fractions:
            group_count = max(
                1,
                int(np.ceil(
                    len(available_groups) * fraction
                )),
            )

            selected_groups = set(
                available_groups[:group_count]
            )

            train_mask = self.groups_train.isin(
                selected_groups
            )

            X_subset = self.X_train.loc[
                train_mask
            ]

            y_subset = self.y_train.loc[
                train_mask
            ]

            pipeline = self._create_pipeline()

            pipeline.fit(
                X_subset,
                y_subset,
            )

            train_predictions = pipeline.predict(
                X_subset
            )

            validation_predictions = pipeline.predict(
                self.X_validation
            )

            train_metrics = self._calculate_metrics(
                y_subset,
                train_predictions,
            )

            validation_metrics = self._calculate_metrics(
                self.y_validation,
                validation_predictions,
            )

            rows.append(
                {
                    "training_fraction": float(fraction),
                    "training_groups": int(group_count),
                    "training_rows": int(len(X_subset)),
                    "train_MAE": train_metrics["MAE"],
                    "validation_MAE": validation_metrics["MAE"],
                    "train_RMSE": train_metrics["RMSE"],
                    "validation_RMSE": validation_metrics["RMSE"],
                    "train_MEAN_NASA_SCORE": float(
                        PHMMetrics.mean_nasa_score(
                            y_subset,
                            train_predictions,
                        )
                    ),
                    "validation_MEAN_NASA_SCORE": float(
                        PHMMetrics.mean_nasa_score(
                            self.y_validation,
                            validation_predictions,
                        )
                    ),
                }
            )

        self.learning_curve_results = pd.DataFrame(
            rows
        )

        return self.learning_curve_results.copy()

    def get_cross_validation_folds(
        self,
    ) -> pd.DataFrame:
        """
        Return one row per cross-validation fold.
        """
        if self.cv_results is None:
            raise RuntimeError(
                "Cross-validation has not been executed. "
                "Call cross_validation() first."
            )

        return pd.DataFrame(
            {
                "fold": np.arange(
                    1,
                    len(self.cv_results["validation_mae"]) + 1,
                ),
                "train_MAE": -self.cv_results["train_mae"],
                "validation_MAE": -self.cv_results["validation_mae"],
                "train_RMSE": -self.cv_results["train_rmse"],
                "validation_RMSE": -self.cv_results["validation_rmse"],
                "train_R2": self.cv_results["train_r2"],
                "validation_R2": self.cv_results["validation_r2"],
                "train_NASA_SCORE": (
                    -self.cv_results["train_nasa_score"]
                ),
                "validation_NASA_SCORE": (
                    -self.cv_results["validation_nasa_score"]
                ),
            }
        )

    # ==========================================================
    # External predictions
    # ==========================================================

    def predict(
        self,
        new_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict target values for external data.
        """
        self._ensure_fitted()

        # The pipeline expects the raw columns used by its fitted
        # preprocessor. Processed feature names are outputs, not inputs.
        expected_columns = self._feature_columns()

        missing_columns = set(
            expected_columns
        ).difference(
            new_data.columns
        )

        if missing_columns:
            raise ValueError(
                f"New data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        X_new = new_data[
            expected_columns
        ].copy()

        return self.pipeline.predict(X_new)

    def predict_dataframe(
        self,
        new_data: pd.DataFrame,
        prediction_column: str = "predicted_value",
    ) -> pd.DataFrame:
        """
        Return external data with an additional prediction column.
        """
        result = new_data.copy()

        result[prediction_column] = self.predict(
            new_data
        )

        return result

    # ==========================================================
    # Results and model information
    # ==========================================================

    def get_results(self) -> dict[str, Any]:
        """
        Return model configuration, metrics and dataset information.
        """
        self._ensure_trained()

        return {
            "model_name": self.model_name,
            "train_metrics": self.train_metrics.copy(),
            "validation_metrics": self.validation_metrics.copy(),
            "overfitting_summary": self.get_overfitting_summary(),
            "cross_validation": (
                None
                if self.cv_summary is None
                else self.cv_summary.copy()
            ),
            "model_parameters": self.get_model_parameters(),
            "training_rows": len(self.X_train),
            "validation_rows": len(self.X_validation),
            "training_groups": len(self.train_group_ids),
            "validation_groups": len(self.validation_group_ids),
            "validation_group_ids": self.validation_group_ids.copy(),
        }

    def get_model_parameters(self) -> dict[str, Any]:
        """
        Return all pipeline and estimator parameters.
        """
        if self.pipeline is None:
            return self._create_pipeline().get_params()

        return self.pipeline.get_params()

    def get_prediction_results(
        self,
        dataset: str = "validation",
    ) -> pd.DataFrame:
        """
        Return row-level predictions and residuals.

        Parameters
        ----------
        dataset:
            ``train`` or ``validation``.
        """
        self._ensure_trained()

        dataset = dataset.lower()

        if dataset == "train":
            index = self.train_index
            y_true = np.asarray(self.y_train)
            y_pred = self.y_train_pred

        elif dataset == "validation":
            index = self.validation_index
            y_true = np.asarray(self.y_validation)
            y_pred = self.y_validation_pred

        else:
            raise ValueError(
                "dataset must be 'train' or 'validation'."
            )

        results = self.df.loc[
            index
        ].copy()

        results["actual"] = y_true
        results["predicted"] = y_pred
        results["residual"] = (
            results["actual"]
            - results["predicted"]
        )
        results["absolute_error"] = (
            results["residual"].abs()
        )
        results["squared_error"] = (
            results["residual"] ** 2
        )
        results["dataset_split"] = dataset

        return results

    def get_all_prediction_results(self) -> pd.DataFrame:
        """
        Return train and validation prediction results in one DataFrame.
        """
        train_results = self.get_prediction_results(
            dataset="train"
        )

        validation_results = self.get_prediction_results(
            dataset="validation"
        )

        return pd.concat(
            [
                train_results,
                validation_results,
            ],
            ignore_index=True,
        )

    def get_feature_importance(
        self,
    ) -> Optional[pd.DataFrame]:
        """
        Return feature importance or coefficients when supported.
        """
        self._ensure_trained()

        fitted_model = self.pipeline.named_steps[
            "model"
        ]

        feature_names = self.X_train.columns

        if hasattr(
            fitted_model,
            "feature_importances_",
        ):
            values = fitted_model.feature_importances_

        elif hasattr(
            fitted_model,
            "coef_",
        ):
            values = np.ravel(
                fitted_model.coef_
            )

        else:
            print(
                f"{type(fitted_model).__name__} does not "
                "expose feature_importances_ or coef_."
            )
            return None

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "absolute_importance": np.abs(values),
            }
        )

        return importance.sort_values(
            "absolute_importance",
            ascending=False,
        ).reset_index(drop=True)

    def get_metrics_by_target_range(
        self,
        bins: Optional[list[float]] = None,
        dataset: str = "validation",
    ) -> pd.DataFrame:
        """
        Calculate error metrics for different target/RUL ranges.

        Example ranges:
            0–30, 31–60, 61–90, 91–125.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        if bins is None:
            bins = [
                -np.inf,
                30,
                60,
                90,
                125,
                np.inf,
            ]

        results["target_range"] = pd.cut(
            results["actual"],
            bins=bins,
            include_lowest=True,
        )

        rows = []

        for target_range, group in results.groupby(
            "target_range",
            observed=True,
        ):
            if group.empty:
                continue

            metrics = self._calculate_metrics(
                group["actual"],
                group["predicted"].to_numpy(),
            )

            rows.append(
                {
                    "target_range": str(target_range),
                    "row_count": len(group),
                    **metrics,
                }
            )

        return pd.DataFrame(rows)

    def get_metrics_by_group(
        self,
        dataset: str = "validation",
    ) -> pd.DataFrame:
        """
        Calculate metrics independently for each motor or series.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        rows = []

        for group_id, group in results.groupby(
            self.group_column
        ):
            metrics = self._calculate_metrics(
                group["actual"],
                group["predicted"].to_numpy(),
            )

            rows.append(
                {
                    self.group_column: group_id,
                    "row_count": len(group),
                    **metrics,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("RMSE", ascending=False)
            .reset_index(drop=True)
        )

    # ==========================================================
    # Internal state validation
    # ==========================================================

    def _ensure_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(
                "The model has not been fitted. "
                "Call train() first."
            )

    def _ensure_trained(self) -> None:
        if (
            self.pipeline is None
            or self.y_train_pred is None
            or self.y_validation_pred is None
        ):
            raise RuntimeError(
                "The model has not been trained. "
                "Call train() first."
            )

    # ==========================================================
    # Visual analysis
    # ==========================================================

    def plot_predictions(
        self,
        dataset: str = "validation",
    ) -> None:
        """
        Plot actual versus predicted values for train or validation.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        minimum = min(
            float(results["actual"].min()),
            float(results["predicted"].min()),
        )

        maximum = max(
            float(results["actual"].max()),
            float(results["predicted"].max()),
        )

        plt.figure(figsize=(7, 5))

        plt.scatter(
            results["actual"],
            results["predicted"],
            alpha=0.5,
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")

        plt.title(
            f"Actual vs Predicted — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_train_vs_validation_predictions(self) -> None:
        """
        Plot train and validation actual-versus-predicted values together.
        """
        self._ensure_trained()

        all_actual = np.concatenate(
            [
                np.asarray(self.y_train),
                np.asarray(self.y_validation),
            ]
        )

        all_predicted = np.concatenate(
            [
                self.y_train_pred,
                self.y_validation_pred,
            ]
        )

        minimum = min(
            float(all_actual.min()),
            float(all_predicted.min()),
        )

        maximum = max(
            float(all_actual.max()),
            float(all_predicted.max()),
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.y_train,
            self.y_train_pred,
            alpha=0.25,
            label="Train",
        )

        plt.scatter(
            self.y_validation,
            self.y_validation_pred,
            alpha=0.55,
            label="Validation",
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")

        plt.title(
            f"Train vs Validation Predictions — {self.model_name}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals(
        self,
        dataset: str = "validation",
    ) -> None:
        """
        Plot residuals against predicted values.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        plt.figure(figsize=(7, 5))

        plt.scatter(
            results["predicted"],
            results["residual"],
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted values")
        plt.ylabel("Residual: actual - predicted")

        plt.title(
            f"Residual Plot — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_train_vs_validation_residuals(self) -> None:
        """
        Compare train and validation residuals on the same chart.
        """
        self._ensure_trained()

        train_residuals = (
            np.asarray(self.y_train)
            - self.y_train_pred
        )

        validation_residuals = (
            np.asarray(self.y_validation)
            - self.y_validation_pred
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.y_train_pred,
            train_residuals,
            alpha=0.25,
            label="Train",
        )

        plt.scatter(
            self.y_validation_pred,
            validation_residuals,
            alpha=0.55,
            label="Validation",
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted values")
        plt.ylabel("Residual: actual - predicted")

        plt.title(
            f"Train vs Validation Residuals — {self.model_name}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals_vs_target(
        self,
        dataset: str = "validation",
    ) -> None:
        """
        Plot residuals against actual target values.

        For RUL prediction this shows where the model performs poorly
        across the degradation range.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            results["actual"],
            results["residual"],
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel(
            f"Actual {self.target_column}"
        )
        plt.ylabel(
            "Residual: actual - predicted"
        )

        plt.title(
            f"Residuals vs {self.target_column} — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_train_vs_validation_residuals_by_target(self) -> None:
        """
        Compare train and validation residuals against actual target values.
        """
        self._ensure_trained()

        train_residuals = (
            np.asarray(self.y_train)
            - self.y_train_pred
        )

        validation_residuals = (
            np.asarray(self.y_validation)
            - self.y_validation_pred
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.y_train,
            train_residuals,
            alpha=0.25,
            label="Train",
        )

        plt.scatter(
            self.y_validation,
            validation_residuals,
            alpha=0.55,
            label="Validation",
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel(
            f"Actual {self.target_column}"
        )

        plt.ylabel(
            "Residual: actual - predicted"
        )

        plt.title(
            f"Residuals by {self.target_column} — "
            f"{self.model_name}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals_vs_time(
        self,
        dataset: str = "validation",
    ) -> None:
        """
        Plot residuals against the time/cycle column.
        """
        if self.time_column is None:
            raise ValueError(
                "time_column was not configured."
            )

        results = self.get_prediction_results(
            dataset=dataset
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            results[self.time_column],
            results["residual"],
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel(self.time_column)
        plt.ylabel(
            "Residual: actual - predicted"
        )

        plt.title(
            f"Residuals vs {self.time_column} — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_error_distribution(
        self,
        dataset: str = "validation",
        bins: int = 30,
    ) -> None:
        """
        Plot the residual distribution for train or validation.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        plt.figure(figsize=(7, 5))

        plt.hist(
            results["residual"],
            bins=bins,
            alpha=0.7,
        )

        plt.axvline(
            0,
            linestyle="--",
        )

        plt.xlabel(
            "Residual: actual - predicted"
        )
        plt.ylabel("Frequency")

        plt.title(
            f"Residual Distribution — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_train_vs_validation_error_distribution(
        self,
        bins: int = 30,
    ) -> None:
        """
        Compare train and validation residual distributions.
        """
        self._ensure_trained()

        train_residuals = (
            np.asarray(self.y_train)
            - self.y_train_pred
        )

        validation_residuals = (
            np.asarray(self.y_validation)
            - self.y_validation_pred
        )

        plt.figure(figsize=(8, 5))

        plt.hist(
            train_residuals,
            bins=bins,
            alpha=0.45,
            label="Train",
        )

        plt.hist(
            validation_residuals,
            bins=bins,
            alpha=0.45,
            label="Validation",
        )

        plt.axvline(
            0,
            linestyle="--",
        )

        plt.xlabel(
            "Residual: actual - predicted"
        )
        plt.ylabel("Frequency")

        plt.title(
            f"Train vs Validation Error Distribution — "
            f"{self.model_name}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_predictions_by_series(
        self,
        group_id: Any,
        dataset: str = "validation",
    ) -> None:
        """
        Plot actual and predicted target values for one series.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        series = results[
            results[self.group_column] == group_id
        ].copy()

        if series.empty:
            raise ValueError(
                f"Group '{group_id}' is not present "
                f"in the {dataset} dataset."
            )

        if self.time_column is not None:
            series = series.sort_values(
                self.time_column
            )

            x_values = series[
                self.time_column
            ]

            x_label = self.time_column

        else:
            x_values = np.arange(
                len(series)
            )

            x_label = "Observation"

        plt.figure(figsize=(10, 5))

        plt.plot(
            x_values,
            series["actual"],
            label="Actual",
        )

        plt.plot(
            x_values,
            series["predicted"],
            label="Predicted",
        )

        plt.xlabel(x_label)
        plt.ylabel(self.target_column)

        plt.title(
            f"Predictions for {group_id} — "
            f"{self.model_name} — {dataset}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_absolute_error_by_series(
        self,
        group_id: Any,
        dataset: str = "validation",
    ) -> None:
        """
        Plot absolute prediction error for one complete series.
        """
        results = self.get_prediction_results(
            dataset=dataset
        )

        series = results[
            results[self.group_column] == group_id
        ].copy()

        if series.empty:
            raise ValueError(
                f"Group '{group_id}' is not present "
                f"in the {dataset} dataset."
            )

        if self.time_column is not None:
            series = series.sort_values(
                self.time_column
            )

            x_values = series[
                self.time_column
            ]

            x_label = self.time_column

        else:
            x_values = np.arange(
                len(series)
            )

            x_label = "Observation"

        plt.figure(figsize=(10, 5))

        plt.plot(
            x_values,
            series["absolute_error"],
        )

        plt.xlabel(x_label)
        plt.ylabel("Absolute error")

        plt.title(
            f"Absolute Error for {group_id} — "
            f"{self.model_name} — {dataset}"
        )

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(
        self,
        top_n: int = 20,
    ) -> None:
        """
        Plot the most important features or coefficients.
        """
        importance = self.get_feature_importance()

        if importance is None:
            return

        plot_data = (
            importance.head(top_n)
            .sort_values("absolute_importance")
        )

        plt.figure(figsize=(8, 6))

        plt.barh(
            plot_data["feature"],
            plot_data["absolute_importance"],
        )

        plt.xlabel("Absolute importance")
        plt.ylabel("Feature")

        plt.title(
            f"Feature Importance — {self.model_name}"
        )

        plt.tight_layout()
        plt.show()



    def plot_external_residuals(
        self,
        results: pd.DataFrame,
        prediction_column: str = "predicted_RUL",
        actual_column: str = "actual",
    ) -> None:
        """
        Plot residuals for a previously evaluated external dataset.
        """
        required = {
            prediction_column,
            actual_column,
        }

        missing = required.difference(results.columns)

        if missing:
            raise ValueError(
                f"Results are missing columns: {sorted(missing)}"
            )

        residuals = (
            results[actual_column]
            - results[prediction_column]
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            results[prediction_column],
            residuals,
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted values")
        plt.ylabel("Residual: actual - predicted")

        plt.title(
            f"External Test Residuals — {self.model_name}"
        )

        plt.tight_layout()
        plt.show()

        # ==========================================================
    # C-MAPSS external evaluation
    # ==========================================================

    @staticmethod
    def _resolve_cmapss_datasets(
        datasets: Union[int, str, Sequence[str]]
    ) -> list[str]:
        """
        Convert the dataset configuration into a list of FD identifiers.

        Examples
        --------
        datasets=2
            -> ["FD001", "FD002"]

        datasets="FD003"
            -> ["FD003"]

        datasets=["FD001", "FD004"]
            -> ["FD001", "FD004"]
        """
        if isinstance(datasets, int):
            if datasets < 1 or datasets > 4:
                raise ValueError(
                    "When datasets is an integer, it must be between 1 and 4."
                )

            return [
                f"FD{number:03d}"
                for number in range(1, datasets + 1)
            ]

        if isinstance(datasets, str):
            datasets = [datasets]

        resolved = []

        for dataset in datasets:
            dataset = dataset.upper().strip()

            if dataset not in {
                "FD001",
                "FD002",
                "FD003",
                "FD004",
            }:
                raise ValueError(
                    f"Invalid C-MAPSS dataset: {dataset}. "
                    "Expected FD001, FD002, FD003 or FD004."
                )

            resolved.append(dataset)

        return resolved

    @classmethod
    def _load_cmapss_external_data(
        cls,
        data_folder: str | Path,
        datasets: Union[int, str, Sequence[str]] = 4,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load C-MAPSS test and official RUL files.

        Parameters
        ----------
        data_folder:
            Folder containing files such as:

            test_FD001.txt
            test_FD002.txt
            RUL_FD001.txt
            RUL_FD002.txt

        datasets:
            Integer, one dataset name, or a sequence of dataset names.

            Examples:
                2
                "FD001"
                ["FD001", "FD003"]

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            test_data:
                All test observations.

            official_rul:
                One official RUL value per motor.
        """
        folder = Path(data_folder)

        if not folder.exists():
            raise FileNotFoundError(
                f"The data folder does not exist: {folder}"
            )

        dataset_names = cls._resolve_cmapss_datasets(datasets)

        test_frames = []
        rul_frames = []

        for dataset_name in dataset_names:
            test_path = folder / f"test_{dataset_name}.txt"
            rul_path = folder / f"RUL_{dataset_name}.txt"

            if not test_path.is_file():
                raise FileNotFoundError(
                    f"Test file not found: {test_path}"
                )

            if not rul_path.is_file():
                raise FileNotFoundError(
                    f"RUL file not found: {rul_path}"
                )

            # Load the test observations.
            test_df = pd.read_csv(
                test_path,
                sep=r"\s+",
                header=None,
                names=cls.CMAPSS_COLUMN_NAMES,
            )

            test_df["dataset"] = dataset_name

            test_df["unique_motor_id"] = (
                test_df["dataset"]
                + "_"
                + test_df["unit_number"].astype(str)
            )

            # Load one official RUL value per motor.
            rul_df = pd.read_csv(
                rul_path,
                sep=r"\s+",
                header=None,
                names=["official_final_RUL"],
            )

            rul_df["dataset"] = dataset_name

            # RUL rows are ordered according to unit_number.
            rul_df["unit_number"] = (
                np.arange(len(rul_df)) + 1
            )

            rul_df["unique_motor_id"] = (
                rul_df["dataset"]
                + "_"
                + rul_df["unit_number"].astype(str)
            )

            expected_motor_count = (
                test_df["unit_number"].nunique()
            )

            if len(rul_df) != expected_motor_count:
                raise ValueError(
                    f"{dataset_name}: test data contains "
                    f"{expected_motor_count} motors, but the RUL file "
                    f"contains {len(rul_df)} values."
                )

            test_frames.append(test_df)
            rul_frames.append(rul_df)

        test_data = pd.concat(
            test_frames,
            ignore_index=True,
        )

        official_rul = pd.concat(
            rul_frames,
            ignore_index=True,
        )

        return test_data, official_rul

    def _prepare_external_features(
        self,
        external_df: pd.DataFrame,
        preprocess_fn: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = None,
    ) -> pd.DataFrame:
        """
        Apply an optional preprocessing or feature-engineering function.

        The callable must receive a DataFrame and return a DataFrame.

        This is useful when the model was trained with engineered features.
        """
        data = external_df.copy()

        if preprocess_fn is not None:
            data = preprocess_fn(data)

            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "preprocess_fn must return a pandas DataFrame."
                )

        return data

    def evaluate_external(
        self,
        external_df: pd.DataFrame,
        target_column: Optional[str] = None,
        prediction_column: str = "predicted_RUL",
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Predict and evaluate a labeled external DataFrame.
        """
        self._ensure_fitted()

        target_column = target_column or self.target_column

        if target_column not in external_df.columns:
            raise ValueError(
                f"The external DataFrame does not contain "
                f"'{target_column}'."
            )

        # The saved pipeline receives raw C-MAPSS columns and performs
        # feature selection/scaling internally. Do not use processed feature
        # names from feature_names.json as pipeline inputs.
        expected_columns = self._feature_columns()

        missing_columns = set(expected_columns).difference(
            external_df.columns
        )

        if missing_columns:
            raise ValueError(
                "The external dataset is missing model features: "
                f"{sorted(missing_columns)}"
            )

        results = external_df.copy()

        results[prediction_column] = self.pipeline.predict(
            results[expected_columns]
        )

        results["actual"] = results[target_column]

        results["residual"] = (
            results["actual"]
            - results[prediction_column]
        )

        results["absolute_error"] = (
            results["residual"].abs()
        )

        results["squared_error"] = (
            results["residual"] ** 2
        )

        metrics = self._calculate_metrics(
            y_true=results["actual"],
            y_pred=results[prediction_column].to_numpy(),
        )

        return results, metrics

    def evaluate_cmapss_final_cycles(
        self,
        data_folder: str | Path,
        datasets: Union[int, str, Sequence[str]] = 4,
        clip_rul: bool = True,
        rul_clip_value: int = 125,
        preprocess_fn: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = None,
        prediction_column: str = "predicted_RUL",
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Evaluate the model using only the last observed cycle of every motor.

        This is the standard C-MAPSS external evaluation method.

        The official RUL file contains the true RUL after the final observed
        cycle of each test engine.

        Parameters
        ----------
        data_folder:
            Folder containing test_FDxxx.txt and RUL_FDxxx.txt files.

        datasets:
            Number or identifiers of datasets to evaluate.

            Examples:
                1
                    Evaluate FD001.

                2
                    Evaluate FD001 and FD002.

                ["FD002", "FD004"]
                    Evaluate only FD002 and FD004.

        clip_rul:
            Apply the same RUL clipping used during training.

        rul_clip_value:
            Maximum RUL value when clipping is enabled.

        preprocess_fn:
            Optional feature-engineering function.

            Example:
                preprocess_fn=feature_engineer.transform

        prediction_column:
            Name of the prediction column.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, float]]
            External predictions and metrics.
        """
        test_data, official_rul = (
            self._load_cmapss_external_data(
                data_folder=data_folder,
                datasets=datasets,
            )
        )

        # Select only the final available observation for each motor.
        final_cycles = (
            test_data
            .sort_values(
                [
                    "dataset",
                    "unit_number",
                    "cycle",
                ]
            )
            .groupby(
                "unique_motor_id",
                as_index=False,
                group_keys=False,
            )
            .tail(1)
            .reset_index(drop=True)
        )

        # Attach the official RUL at the last observed cycle.
        final_cycles = final_cycles.merge(
            official_rul[
                [
                    "unique_motor_id",
                    "official_final_RUL",
                ]
            ],
            on="unique_motor_id",
            how="left",
            validate="one_to_one",
        )

        final_cycles["RUL"] = (
            final_cycles["official_final_RUL"]
        )

        if clip_rul:
            final_cycles["RUL"] = (
                final_cycles["RUL"]
                .clip(upper=rul_clip_value)
            )

        final_cycles = self._prepare_external_features(
            external_df=final_cycles,
            preprocess_fn=preprocess_fn,
        )

        results, metrics = self.evaluate_external(
            external_df=final_cycles,
            target_column="RUL",
            prediction_column=prediction_column,
        )

        metrics["evaluation_method"] = "final_cycle"
        metrics["motor_count"] = int(
            results["unique_motor_id"].nunique()
        )
        metrics["row_count"] = int(len(results))

        return results, metrics

    def evaluate_cmapss_all_cycles(
        self,
        data_folder: str | Path,
        datasets: Union[int, str, Sequence[str]] = 4,
        clip_rul: bool = True,
        rul_clip_value: int = 125,
        preprocess_fn: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = None,
        prediction_column: str = "predicted_RUL",
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Evaluate the model using every available cycle from the test files.

        The true row-level RUL is derived as:

            maximum observed cycle
            + official RUL at the final observed cycle
            - current cycle

        Parameters
        ----------
        data_folder:
            Folder containing test_FDxxx.txt and RUL_FDxxx.txt files.

        datasets:
            Number or identifiers of datasets to evaluate.

        clip_rul:
            Apply the same clipping used for the training target.

        rul_clip_value:
            Maximum RUL value when clipping is enabled.

        preprocess_fn:
            Optional feature-engineering function.

        prediction_column:
            Name of the prediction column.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, float]]
            External row-level predictions and metrics.
        """
        test_data, official_rul = (
            self._load_cmapss_external_data(
                data_folder=data_folder,
                datasets=datasets,
            )
        )

        # Add the official RUL at the final observed cycle.
        test_with_rul = test_data.merge(
            official_rul[
                [
                    "unique_motor_id",
                    "official_final_RUL",
                ]
            ],
            on="unique_motor_id",
            how="left",
            validate="many_to_one",
        )

        # Find the last observed cycle for each motor.
        test_with_rul["max_observed_cycle"] = (
            test_with_rul
            .groupby("unique_motor_id")["cycle"]
            .transform("max")
        )

        # Derive the true RUL for every recorded cycle.
        test_with_rul["RUL"] = (
            test_with_rul["max_observed_cycle"]
            + test_with_rul["official_final_RUL"]
            - test_with_rul["cycle"]
        )

        if clip_rul:
            test_with_rul["RUL"] = (
                test_with_rul["RUL"]
                .clip(upper=rul_clip_value)
            )

        test_with_rul = self._prepare_external_features(
            external_df=test_with_rul,
            preprocess_fn=preprocess_fn,
        )

        results, metrics = self.evaluate_external(
            external_df=test_with_rul,
            target_column="RUL",
            prediction_column=prediction_column,
        )

        metrics["evaluation_method"] = "all_cycles"
        metrics["motor_count"] = int(
            results["unique_motor_id"].nunique()
        )
        metrics["row_count"] = int(len(results))

        return results, metrics
