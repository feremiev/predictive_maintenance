from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.base import clone

from preprocessors.cmapss_preprocessor import CMapssPreprocessor


class SequenceRULModel:
    """
    Train sequence-based regression models for Remaining Useful Life prediction.

    Development workflow
    --------------------
    The DataFrame passed to this class must come from the C-MAPSS training files.

    Complete motors from those training files are divided into:

        1. Training motors
        2. Validation motors

    The validation motors are used during model development for:

        - early stopping
        - learning-rate reduction
        - hyperparameter selection
        - overfitting analysis
        - model comparison

    The official C-MAPSS test files are not used during training. They are
    evaluated afterward with:

        test_FD00X.txt
        RUL_FD00X.txt

    The official RUL value represents the remaining useful life at the final
    observed cycle of each test motor. For sequence models, the standard final
    evaluation uses the last valid sequence window from every test motor.

    Supported window types
    ----------------------
    sliding:
        Fixed-length windows.

        Example with window_size=4:

            cycles 1-4 -> predict RUL at cycle 4
            cycles 2-5 -> predict RUL at cycle 5
            cycles 3-6 -> predict RUL at cycle 6

    growing:
        Uses the available history up to the current cycle, starting at
        min_window_size. The sequence is left-padded to max_window_size.

        Example with min_window_size=3:

            cycles 1-3 -> predict RUL at cycle 3
            cycles 1-4 -> predict RUL at cycle 4
            cycles 1-5 -> predict RUL at cycle 5

    Supported model types
    ---------------------
    lstm
    gru
    cnn
    cnn_lstm
    """

    SUPPORTED_MODELS = {
        "lstm",
        "gru",
        "cnn",
        "cnn_lstm",
    }

    SUPPORTED_WINDOW_TYPES = {
        "sliding",
        "growing",
    }

    SUPPORTED_SCALERS = {
        "standard",
        "minmax",
        "robust",
        "none",
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
        time_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        columns_to_drop: Optional[Sequence[str]] = None,

        # Window configuration
        window_type: str = "sliding",
        window_size: int = 30,
        min_window_size: int = 10,
        max_window_size: Optional[int] = None,
        stride: int = 1,
        prediction_horizon: int = 0,
        padding_value: float = 0.0,

        # Train/validation split configuration
        validation_group_count: Optional[int] = None,
        validation_group_size: float = 0.15,
        group_selection: str = "random",
        random_state: int = 42,

        # Scaling
        scaler: str = "standard",

        # Model configuration
        model_type: str = "lstm",
        recurrent_units: Sequence[int] = (128, 64),
        dense_units: Sequence[int] = (64, 32),
        cnn_filters: Sequence[int] = (64, 128),
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        bidirectional: bool = False,

        # Training configuration
        learning_rate: float = 0.001,
        loss: str = "huber",
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 12,
        reduce_lr: bool = True,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        min_learning_rate: float = 1e-6,
        shuffle_windows: bool = True,
        verbose: int = 1,
        preprocessor=None,


    ) -> None:
        self.df = df.copy()

        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column

        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else None
        )
        self.columns_to_drop = list(columns_to_drop or [])

        self.window_type = window_type.lower()
        self.window_size = int(window_size)
        self.min_window_size = int(min_window_size)
        self.max_window_size = (
            int(max_window_size)
            if max_window_size is not None
            else int(window_size)
        )
        self.stride = int(stride)
        self.prediction_horizon = int(prediction_horizon)
        self.padding_value = float(padding_value)

        self.validation_group_count = validation_group_count
        self.validation_group_size = float(validation_group_size)
        self.group_selection = group_selection.lower()
        self.random_state = int(random_state)

        self.scaler_name = scaler.lower()
        self.scaler = None

        self.model_type = model_type.lower()
        self.recurrent_units = list(recurrent_units)
        self.dense_units = list(dense_units)
        self.cnn_filters = list(cnn_filters)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.bidirectional = bool(bidirectional)

        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.reduce_lr = bool(reduce_lr)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.shuffle_windows = bool(shuffle_windows)
        self.verbose = int(verbose)

        # Development split
        self.train_group_ids: Optional[np.ndarray] = None
        self.validation_group_ids: Optional[np.ndarray] = None

        self.train_df: Optional[pd.DataFrame] = None
        self.validation_df: Optional[pd.DataFrame] = None

        self.X_train: Optional[np.ndarray] = None
        self.X_validation: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_validation: Optional[np.ndarray] = None

        self.train_metadata: Optional[pd.DataFrame] = None
        self.validation_metadata: Optional[pd.DataFrame] = None

        self.model: Optional[keras.Model] = None
        self.history = None

        self.y_train_pred: Optional[np.ndarray] = None
        self.y_validation_pred: Optional[np.ndarray] = None

        self.train_metrics: Optional[dict[str, float]] = None
        self.validation_metrics: Optional[dict[str, float]] = None

        # Official external test results
        self.external_test_results: Optional[pd.DataFrame] = None
        self.external_test_metrics: Optional[dict[str, Any]] = None

        self._validate_configuration()
        self.preprocessor = (
                            preprocessor
                            if preprocessor is not None
                            else CMapssPreprocessor()
        )

        self.fitted_preprocessor = None
        self.processed_feature_columns_: Optional[list[str]] = None

    # ==========================================================
    # Validation and feature selection
    # ==========================================================

    def _validate_configuration(self) -> None:
        required_columns = {
            self.target_column,
            self.group_column,
            self.time_column,
        }

        missing = required_columns.difference(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Available values: {sorted(self.SUPPORTED_MODELS)}"
            )

        if self.window_type not in self.SUPPORTED_WINDOW_TYPES:
            raise ValueError(
                f"Unsupported window_type '{self.window_type}'. "
                f"Available values: {sorted(self.SUPPORTED_WINDOW_TYPES)}"
            )

        if self.scaler_name not in self.SUPPORTED_SCALERS:
            raise ValueError(
                f"Unsupported scaler '{self.scaler_name}'. "
                f"Available values: {sorted(self.SUPPORTED_SCALERS)}"
            )

        if self.group_selection not in {"random", "first", "last"}:
            raise ValueError(
                "group_selection must be 'random', 'first', or 'last'."
            )

        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")

        if self.min_window_size < 2:
            raise ValueError("min_window_size must be at least 2.")

        if self.max_window_size < self.min_window_size:
            raise ValueError(
                "max_window_size cannot be smaller than min_window_size."
            )

        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

        if self.prediction_horizon < 0:
            raise ValueError(
                "prediction_horizon cannot be negative."
            )

        if not 0 < self.validation_group_size < 1:
            raise ValueError(
                "validation_group_size must be between 0 and 1."
            )

        if (
            self.validation_group_count is not None
            and self.validation_group_count <= 0
        ):
            raise ValueError(
                "validation_group_count must be greater than zero."
            )

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1.")

        if self.df[self.target_column].isna().any():
            raise ValueError(
                f"Target column '{self.target_column}' contains null values."
            )

    def _resolve_feature_columns(
        self,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> list[str]:
        """
        Return the raw input columns required by the preprocessor.

        These are the columns before preprocessing. The preprocessor will
        transform them into the final numeric model features.
        """
        source = (
            dataframe
            if dataframe is not None
            else self.df
        )

        if self.preprocessor is not None:
            expected = list(
                self.preprocessor.correct_order_
            )

            # CMapssPreprocessor uses OperatingConditionEncoder, so the raw
            # operating-condition column must also be passed to it.
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
                source.columns
            )

            if missing:
                raise ValueError(
                    "Missing sequence input features: "
                    f"{sorted(missing)}"
                )

            return expected

        if self.feature_columns is not None:
            missing = set(
                self.feature_columns
            ).difference(source.columns)

            if missing:
                raise ValueError(
                    f"Missing configured features: {sorted(missing)}"
                )

            return list(self.feature_columns)

        excluded = {
            self.target_column,
            self.group_column,
            *self.columns_to_drop,
        }

        return [
            column
            for column in source.select_dtypes(
                include=np.number
            ).columns
            if column not in excluded
        ]

    # ==========================================================
    # Train/validation split by complete motors
    # ==========================================================

    def split_groups(self) -> None:
        """
        Split complete motors from the training files into train and validation.

        No observations from a validation motor can appear in training.
        """
        data = self.df.sort_values(
            [self.group_column, self.time_column]
        ).copy()

        group_ids = (
            data[self.group_column]
            .drop_duplicates()
            .to_numpy()
        )

        total_groups = len(group_ids)

        if total_groups < 2:
            raise ValueError(
                "At least two complete motors are required."
            )

        if self.validation_group_count is not None:
            validation_count = int(self.validation_group_count)
        else:
            validation_count = max(
                1,
                int(np.ceil(total_groups * self.validation_group_size)),
            )

        if validation_count >= total_groups:
            raise ValueError(
                "The number of validation motors must be smaller than "
                "the total number of motors."
            )

        if self.group_selection == "random":
            rng = np.random.default_rng(self.random_state)
            ordered_groups = rng.permutation(group_ids)
            self.validation_group_ids = ordered_groups[-validation_count:]
            self.train_group_ids = ordered_groups[:-validation_count]

        elif self.group_selection == "last":
            self.validation_group_ids = group_ids[-validation_count:]
            self.train_group_ids = group_ids[:-validation_count]

        else:
            self.validation_group_ids = group_ids[:validation_count]
            self.train_group_ids = group_ids[validation_count:]

        overlap = set(self.train_group_ids).intersection(
            set(self.validation_group_ids)
        )
        if overlap:
            raise RuntimeError(
                f"Data leakage detected. Shared motors: {sorted(overlap)}"
            )

        train_mask = data[self.group_column].isin(self.train_group_ids)
        validation_mask = data[self.group_column].isin(
            self.validation_group_ids
        )

        self.train_df = data.loc[train_mask].copy()
        self.validation_df = data.loc[validation_mask].copy()

        print(
            f"Train motors: {len(self.train_group_ids)} | "
            f"Validation motors: {len(self.validation_group_ids)}"
        )
        print(
            f"Train rows: {len(self.train_df):,} | "
            f"Validation rows: {len(self.validation_df):,}"
        )

    # ==========================================================
    # Scaling
    # ==========================================================

    def _create_scaler(self):
        if self.scaler_name == "standard":
            return StandardScaler()
        if self.scaler_name == "minmax":
            return MinMaxScaler()
        if self.scaler_name == "robust":
            return RobustScaler()
        if self.scaler_name == "none":
            return None

        raise ValueError(f"Unsupported scaler: {self.scaler_name}")

    def _replace_raw_features(
        self,
        original: pd.DataFrame,
        processed: pd.DataFrame,
        raw_feature_columns: list[str],
    ) -> pd.DataFrame:
        """
        Replace raw preprocessor inputs with the processed numeric features.

        This is necessary because preprocessing can change the number of columns,
        for example by converting operating_condition into one-hot columns.
        """
        result = original.drop(
            columns=raw_feature_columns,
            errors="ignore",
        ).copy()

        processed = processed.copy()
        processed.index = original.index

        return pd.concat(
            [
                result,
                processed,
            ],
            axis=1,
        )

    def _fit_and_apply_preprocessor(
        self,
    ) -> None:
        """
        Fit preprocessing using training motors only.

        The fitted preprocessing is then applied to validation motors without
        fitting again, preventing data leakage.
        """
        raw_feature_columns = (
            self._resolve_feature_columns(
                self.train_df
            )
        )

        if self.preprocessor is None:
            self.processed_feature_columns_ = (
                raw_feature_columns
            )
            return

        self.fitted_preprocessor = clone(
            self.preprocessor
        )

        # Fit only on training data.
        self.fitted_preprocessor.fit(
            self.train_df[
                raw_feature_columns
            ]
        )

        train_processed = (
            self.fitted_preprocessor.transform(
                self.train_df[
                    raw_feature_columns
                ]
            )
        )

        validation_processed = (
            self.fitted_preprocessor.transform(
                self.validation_df[
                    raw_feature_columns
                ]
            )
        )

        if not isinstance(
            train_processed,
            pd.DataFrame,
        ):
            train_processed = pd.DataFrame(
                train_processed,
                index=self.train_df.index,
                columns=(
                    self.fitted_preprocessor
                    .get_feature_names_out()
                ),
            )

        if not isinstance(
            validation_processed,
            pd.DataFrame,
        ):
            validation_processed = pd.DataFrame(
                validation_processed,
                index=self.validation_df.index,
                columns=(
                    self.fitted_preprocessor
                    .get_feature_names_out()
                ),
            )

        self.processed_feature_columns_ = list(
            train_processed.columns
        )

        self.train_df = self._replace_raw_features(
            original=self.train_df,
            processed=train_processed,
            raw_feature_columns=raw_feature_columns,
        )

        self.validation_df = (
            self._replace_raw_features(
                original=self.validation_df,
                processed=validation_processed,
                raw_feature_columns=raw_feature_columns,
            )
        )

    def _transform_external_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform validation-independent or official external data using the
        preprocessor fitted on training motors.

        The preprocessor is never fitted again here.
        """
        if self.fitted_preprocessor is None:
            raise RuntimeError(
                "The preprocessor has not been fitted. "
                "Train or prepare the model first."
            )

        raw_feature_columns = (
            self._resolve_feature_columns(df)
        )

        processed = (
            self.fitted_preprocessor.transform(
                df[raw_feature_columns]
            )
        )

        if not isinstance(
            processed,
            pd.DataFrame,
        ):
            processed = pd.DataFrame(
                processed,
                index=df.index,
                columns=(
                    self.fitted_preprocessor
                    .get_feature_names_out()
                ),
            )

        return self._replace_raw_features(
            original=df,
            processed=processed,
            raw_feature_columns=raw_feature_columns,
        )

    # ==========================================================
    # Window generation
    # ==========================================================

    def _sequence_length(self) -> int:
        if self.window_type == "sliding":
            return self.window_size
        return self.max_window_size

    def _left_pad_window(
        self,
        values: np.ndarray,
        desired_length: int,
    ) -> np.ndarray:
        current_length = len(values)

        if current_length > desired_length:
            values = values[-desired_length:]
            current_length = desired_length

        pad_length = desired_length - current_length

        if pad_length == 0:
            return values

        padding = np.full(
            shape=(pad_length, values.shape[1]),
            fill_value=self.padding_value,
            dtype=np.float32,
        )

        return np.vstack([padding, values])

    def _generate_windows(
        self,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate all valid training or validation windows independently
        for every motor.
        """
        if not self.processed_feature_columns_:
            raise RuntimeError(
                "Processed feature columns are unavailable. "
                "Run prepare_data() first."
            )

        feature_columns = (
            self.processed_feature_columns_
        )

        X_windows: list[np.ndarray] = []
        y_values: list[float] = []
        metadata_rows: list[dict[str, Any]] = []

        grouped = data.groupby(
            self.group_column,
            sort=False,
        )

        for group_id, group in grouped:
            group = group.sort_values(self.time_column)

            X_group = group[feature_columns].to_numpy(dtype=np.float32)
            y_group = group[self.target_column].to_numpy(dtype=np.float32)
            cycles = group[self.time_column].to_numpy()
            indexes = group.index.to_numpy()

            sequence_length = len(group)

            if self.window_type == "sliding":
                minimum_required = (
                    self.window_size + self.prediction_horizon
                )

                if sequence_length < minimum_required:
                    continue

                last_start = (
                    sequence_length
                    - self.window_size
                    - self.prediction_horizon
                    + 1
                )

                for start in range(0, last_start, self.stride):
                    end = start + self.window_size
                    target_position = (
                        end - 1 + self.prediction_horizon
                    )

                    X_windows.append(X_group[start:end])
                    y_values.append(float(y_group[target_position]))

                    metadata_rows.append(
                        {
                            self.group_column: group_id,
                            "window_start_cycle": cycles[start],
                            "window_end_cycle": cycles[end - 1],
                            "target_cycle": cycles[target_position],
                            "target_row_index": indexes[target_position],
                        }
                    )

            else:
                first_end = self.min_window_size
                last_end = (
                    sequence_length - self.prediction_horizon
                )

                for end in range(
                    first_end,
                    last_end + 1,
                    self.stride,
                ):
                    target_position = (
                        end - 1 + self.prediction_horizon
                    )
                    start = max(0, end - self.max_window_size)
                    observed = X_group[start:end]

                    X_windows.append(
                        self._left_pad_window(
                            observed,
                            desired_length=self.max_window_size,
                        )
                    )
                    y_values.append(float(y_group[target_position]))

                    metadata_rows.append(
                        {
                            self.group_column: group_id,
                            "window_start_cycle": cycles[start],
                            "window_end_cycle": cycles[end - 1],
                            "target_cycle": cycles[target_position],
                            "target_row_index": indexes[target_position],
                            "observed_window_length": len(observed),
                        }
                    )

        if not X_windows:
            raise ValueError(
                "No windows were created. Check the configured window sizes, "
                "prediction horizon, and motor history lengths."
            )

        return (
            np.asarray(X_windows, dtype=np.float32),
            np.asarray(y_values, dtype=np.float32),
            pd.DataFrame(metadata_rows),
        )

    def _generate_final_windows(
        self,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Generate exactly one final sequence per external test motor.

        The generated sequence ends at the last observed test cycle. This is
        the standard input used to compare predictions with RUL_FD00X.txt.
        """
        if not self.processed_feature_columns_:
            raise RuntimeError(
                "Processed feature columns are unavailable. "
                "Train the model before official test evaluation."
            )

        feature_columns = (
            self.processed_feature_columns_
        )

        X_windows: list[np.ndarray] = []
        metadata_rows: list[dict[str, Any]] = []

        for group_id, group in data.groupby(
            self.group_column,
            sort=False,
        ):
            group = group.sort_values(self.time_column)

            values = group[feature_columns].to_numpy(dtype=np.float32)
            cycles = group[self.time_column].to_numpy()

            if self.window_type == "sliding":
                if len(values) < self.window_size:
                    continue

                window = values[-self.window_size:]
                start_cycle = cycles[-self.window_size]
                observed_length = self.window_size

            else:
                if len(values) < self.min_window_size:
                    continue

                observed = values[-self.max_window_size:]
                window = self._left_pad_window(
                    observed,
                    desired_length=self.max_window_size,
                )
                start_cycle = cycles[-len(observed)]
                observed_length = len(observed)

            X_windows.append(window)

            metadata_rows.append(
                {
                    self.group_column: group_id,
                    "window_start_cycle": start_cycle,
                    "window_end_cycle": cycles[-1],
                    "target_cycle": cycles[-1],
                    "observed_window_length": observed_length,
                }
            )

        if not X_windows:
            raise ValueError(
                "No final external windows were created. Some motors may be "
                "shorter than the configured minimum window."
            )

        return (
            np.asarray(X_windows, dtype=np.float32),
            pd.DataFrame(metadata_rows),
        )


    def prepare_data(self) -> None:
        """
        Split complete motors, preprocess the train and validation sets,
        and create sequence windows.
        """
        self.split_groups()

        self._fit_and_apply_preprocessor()

        (
            self.X_train,
            self.y_train,
            self.train_metadata,
        ) = self._generate_windows(
            self.train_df
        )

        (
            self.X_validation,
            self.y_validation,
            self.validation_metadata,
        ) = self._generate_windows(
            self.validation_df
        )

        print(
            f"X_train: {self.X_train.shape} | "
            f"y_train: {self.y_train.shape}"
        )

        print(
            f"X_validation: {self.X_validation.shape} | "
            f"y_validation: {self.y_validation.shape}"
        )

    # ==========================================================
    # Model construction
    # ==========================================================

    def _add_dense_head(
        self,
        model: keras.Sequential,
    ) -> None:
        for units in self.dense_units:
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                )
            )

            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout))

        model.add(layers.Dense(1, activation="linear"))

    def _build_lstm_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        model = keras.Sequential(name="lstm_rul_model")
        model.add(layers.Input(shape=input_shape))

        if self.window_type == "growing":
            model.add(
                layers.Masking(
                    mask_value=self.padding_value
                )
            )

        for index, units in enumerate(self.recurrent_units):
            return_sequences = (
                index < len(self.recurrent_units) - 1
            )

            recurrent_layer = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )

            if self.bidirectional:
                recurrent_layer = layers.Bidirectional(
                    recurrent_layer
                )

            model.add(recurrent_layer)

        self._add_dense_head(model)
        return model

    def _build_gru_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        model = keras.Sequential(name="gru_rul_model")
        model.add(layers.Input(shape=input_shape))

        if self.window_type == "growing":
            model.add(
                layers.Masking(
                    mask_value=self.padding_value
                )
            )

        for index, units in enumerate(self.recurrent_units):
            return_sequences = (
                index < len(self.recurrent_units) - 1
            )

            recurrent_layer = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )

            if self.bidirectional:
                recurrent_layer = layers.Bidirectional(
                    recurrent_layer
                )

            model.add(recurrent_layer)

        self._add_dense_head(model)
        return model

    def _build_cnn_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        model = keras.Sequential(name="cnn_rul_model")
        model.add(layers.Input(shape=input_shape))

        for filters in self.cnn_filters:
            model.add(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(layers.BatchNormalization())
            model.add(
                layers.MaxPooling1D(
                    pool_size=self.pool_size,
                    padding="same",
                )
            )

            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout))

        model.add(layers.GlobalAveragePooling1D())
        self._add_dense_head(model)
        return model

    def _build_cnn_lstm_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        model = keras.Sequential(name="cnn_lstm_rul_model")
        model.add(layers.Input(shape=input_shape))

        for filters in self.cnn_filters:
            model.add(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(layers.BatchNormalization())

            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout))

        lstm_units = (
            self.recurrent_units[-1]
            if self.recurrent_units
            else 64
        )

        model.add(
            layers.LSTM(
                lstm_units,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        )

        self._add_dense_head(model)
        return model

    def build_model(self) -> keras.Model:
        if self.X_train is None:
            self.prepare_data()

        input_shape = self.X_train.shape[1:]

        if self.model_type == "lstm":
            self.model = self._build_lstm_model(input_shape)
        elif self.model_type == "gru":
            self.model = self._build_gru_model(input_shape)
        elif self.model_type == "cnn":
            self.model = self._build_cnn_model(input_shape)
        else:
            self.model = self._build_cnn_lstm_model(input_shape)

        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        selected_loss = (
            keras.losses.Huber()
            if self.loss == "huber"
            else self.loss
        )

        self.model.compile(
            optimizer=optimizer,
            loss=selected_loss,
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

        return self.model

    # ==========================================================
    # Training and development evaluation
    # ==========================================================

    def _create_callbacks(self) -> list:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        if self.reduce_lr:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                )
            )

        return callbacks

    def train(self) -> dict[str, dict[str, float]]:
        """
        Train using training motors and monitor validation motors.

        Returns only development metrics:

            train
            validation

        Official test metrics are produced separately by
        evaluate_cmapss_final_windows().
        """
        if self.X_train is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(
                self.X_validation,
                self.y_validation,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._create_callbacks(),
            shuffle=self.shuffle_windows,
            verbose=self.verbose,
        )

        self.y_train_pred = (
            self.model.predict(
                self.X_train,
                verbose=0,
            )
            .reshape(-1)
        )

        self.y_validation_pred = (
            self.model.predict(
                self.X_validation,
                verbose=0,
            )
            .reshape(-1)
        )

        self.train_metrics = self._calculate_metrics(
            self.y_train,
            self.y_train_pred,
        )

        self.validation_metrics = self._calculate_metrics(
            self.y_validation,
            self.y_validation_pred,
        )

        return self.get_metrics()

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        y_true_array = np.asarray(
            y_true,
            dtype=float,
        ).reshape(-1)

        y_pred_array = np.asarray(
            y_pred,
            dtype=float,
        ).reshape(-1)

        residuals = y_true_array - y_pred_array

        denominator = np.where(
            y_true_array == 0,
            np.nan,
            np.abs(y_true_array),
        )

        return {
            "MAE": float(
                mean_absolute_error(
                    y_true_array,
                    y_pred_array,
                )
            ),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        y_true_array,
                        y_pred_array,
                    )
                )
            ),
            "R2": float(
                r2_score(
                    y_true_array,
                    y_pred_array,
                )
            ),
            "MAPE": float(
                np.nanmean(
                    np.abs(residuals) / denominator
                )
                * 100
            ),
            "Bias": float(residuals.mean()),
        }

    def get_metrics(self) -> dict[str, dict[str, float]]:
        if (
            self.train_metrics is None
            or self.validation_metrics is None
        ):
            raise RuntimeError(
                "The model has not been trained."
            )

        return {
            "train": self.train_metrics.copy(),
            "validation": self.validation_metrics.copy(),
        }

    def get_prediction_results(
        self,
        dataset: str = "validation",
    ) -> pd.DataFrame:
        """
        Return development predictions for 'train' or 'validation'.
        """
        dataset = dataset.lower()

        if dataset == "train":
            metadata = self.train_metadata
            y_true = self.y_train
            y_pred = self.y_train_pred

        elif dataset == "validation":
            metadata = self.validation_metadata
            y_true = self.y_validation
            y_pred = self.y_validation_pred

        else:
            raise ValueError(
                "dataset must be 'train' or 'validation'. "
                "Use get_external_test_results() for the official test."
            )

        if metadata is None or y_pred is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        results = metadata.copy()
        results["actual"] = y_true
        results["predicted"] = y_pred
        results["residual"] = (
            results["actual"] - results["predicted"]
        )
        results["absolute_error"] = results["residual"].abs()
        results["squared_error"] = results["residual"] ** 2
        results["dataset_split"] = dataset

        return results

    # ==========================================================
    # General external DataFrame prediction
    # ==========================================================

    def prepare_external_data(
        self,
        external_df: pd.DataFrame,
        target_available: bool = True,
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        pd.DataFrame,
    ]:
        """
        Prepare any external DataFrame using the fitted scaler and the same
        window configuration.
        """
        if self.model is None:
            raise RuntimeError(
                "Train or build the model before external prediction."
            )

        feature_columns = self._resolve_feature_columns(external_df)

        required_columns = {
            self.group_column,
            self.time_column,
            *feature_columns,
        }

        if target_available:
            required_columns.add(self.target_column)

        missing = required_columns.difference(
            external_df.columns
        )
        if missing:
            raise ValueError(
                f"External data is missing columns: {sorted(missing)}"
            )

        transformed = self._transform_external_features(
            external_df.sort_values(
                [self.group_column, self.time_column]
            ).copy()
        )

        if target_available:
            return self._generate_windows(transformed)

        transformed[self.target_column] = 0.0
        X_external, _, metadata = self._generate_windows(
            transformed
        )

        return X_external, None, metadata

    def predict_external(
        self,
        external_df: pd.DataFrame,
    ) -> pd.DataFrame:
        X_external, _, metadata = self.prepare_external_data(
            external_df,
            target_available=False,
        )

        predictions = (
            self.model.predict(
                X_external,
                verbose=0,
            )
            .reshape(-1)
        )

        results = metadata.copy()
        results["predicted"] = predictions

        return results

    def evaluate_external(
        self,
        external_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        X_external, y_external, metadata = (
            self.prepare_external_data(
                external_df,
                target_available=True,
            )
        )

        predictions = (
            self.model.predict(
                X_external,
                verbose=0,
            )
            .reshape(-1)
        )

        results = metadata.copy()
        results["actual"] = y_external
        results["predicted"] = predictions
        results["residual"] = (
            results["actual"] - results["predicted"]
        )
        results["absolute_error"] = results["residual"].abs()
        results["squared_error"] = results["residual"] ** 2

        metrics = self._calculate_metrics(
            y_external,
            predictions,
        )

        return results, metrics

    # ==========================================================
    # Official C-MAPSS test + RUL evaluation
    # ==========================================================

    @staticmethod
    def _resolve_cmapss_datasets(
        datasets: Union[int, str, Sequence[str]],
    ) -> list[str]:
        if isinstance(datasets, int):
            if not 1 <= datasets <= 4:
                raise ValueError(
                    "When datasets is an integer, it must be between 1 and 4."
                )

            return [
                f"FD{number:03d}"
                for number in range(1, datasets + 1)
            ]

        if isinstance(datasets, str):
            datasets = [datasets]

        resolved: list[str] = []

        for dataset in datasets:
            normalized = dataset.upper().strip()

            if normalized not in {
                "FD001",
                "FD002",
                "FD003",
                "FD004",
            }:
                raise ValueError(
                    f"Invalid C-MAPSS dataset '{dataset}'."
                )

            if normalized not in resolved:
                resolved.append(normalized)

        if not resolved:
            raise ValueError(
                "At least one C-MAPSS dataset must be selected."
            )

        return resolved

    @classmethod
    def _load_cmapss_test_and_rul(
        cls,
        data_folder: str | Path,
        datasets: Union[int, str, Sequence[str]] = 4,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        folder = Path(data_folder)

        if not folder.exists():
            raise FileNotFoundError(
                f"The data folder does not exist: {folder}"
            )

        test_frames: list[pd.DataFrame] = []
        rul_frames: list[pd.DataFrame] = []

        for dataset_name in cls._resolve_cmapss_datasets(
            datasets
        ):
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

            rul_df = pd.read_csv(
                rul_path,
                sep=r"\s+",
                header=None,
                names=["official_RUL"],
            )

            rul_df["dataset"] = dataset_name
            rul_df["unit_number"] = (
                np.arange(len(rul_df)) + 1
            )
            rul_df["unique_motor_id"] = (
                rul_df["dataset"]
                + "_"
                + rul_df["unit_number"].astype(str)
            )

            motor_count = test_df[
                "unique_motor_id"
            ].nunique()

            if len(rul_df) != motor_count:
                raise ValueError(
                    f"{dataset_name}: test data contains {motor_count} motors, "
                    f"but the RUL file contains {len(rul_df)} values."
                )

            test_frames.append(test_df)
            rul_frames.append(rul_df)

        return (
            pd.concat(test_frames, ignore_index=True),
            pd.concat(rul_frames, ignore_index=True),
        )

    def evaluate_cmapss_final_windows(
        self,
        data_folder: str | Path,
        datasets: Union[int, str, Sequence[str]] = 4,
        clip_rul: bool = False,
        rul_clip_value: int = 125,
        preprocess_fn: Optional[Any] = None,
        prediction_column: str = "predicted_RUL",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Perform the standard official C-MAPSS sequence evaluation.

        One final sequence is generated for every test motor and compared with
        the corresponding value from RUL_FD00X.txt.

        Parameters
        ----------
        data_folder:
            Folder containing test_FD00X.txt and RUL_FD00X.txt.

        datasets:
            1, 2, 3, 4, one FD identifier, or a sequence of FD identifiers.

        clip_rul:
            Apply only when the model was trained with clipped RUL targets.

        rul_clip_value:
            RUL cap used when clip_rul=True.

        preprocess_fn:
            Optional feature-engineering function applied before scaling and
            window generation. It must return a DataFrame.

        Returns
        -------
        tuple
            Official external test prediction table and metrics.
        """
        if self.model is None:
            raise RuntimeError(
                "The model must be trained before official test evaluation."
            )

        test_data, official_rul = (
            self._load_cmapss_test_and_rul(
                data_folder=data_folder,
                datasets=datasets,
            )
        )

        if preprocess_fn is not None:
            test_data = preprocess_fn(test_data.copy())

            if not isinstance(test_data, pd.DataFrame):
                raise TypeError(
                    "preprocess_fn must return a pandas DataFrame."
                )

        transformed_test = self._transform_external_features(
            test_data
        )

        X_external, metadata = self._generate_final_windows(
            transformed_test
        )

        predictions = (
            self.model.predict(
                X_external,
                verbose=0,
            )
            .reshape(-1)
        )

        results = metadata.merge(
            official_rul[
                [
                    "unique_motor_id",
                    "dataset",
                    "unit_number",
                    "official_RUL",
                ]
            ],
            left_on=self.group_column,
            right_on="unique_motor_id",
            how="left",
            validate="one_to_one",
        )

        if self.group_column != "unique_motor_id":
            results.drop(
                columns=["unique_motor_id"],
                inplace=True,
            )

        results["actual"] = results["official_RUL"]

        if clip_rul:
            results["actual"] = results["actual"].clip(
                upper=rul_clip_value
            )

        results[prediction_column] = predictions
        results["predicted"] = predictions
        results["residual"] = (
            results["actual"] - results["predicted"]
        )
        results["absolute_error"] = results["residual"].abs()
        results["squared_error"] = results["residual"] ** 2
        results["dataset_split"] = "external_test"

        metrics: dict[str, Any] = self._calculate_metrics(
            results["actual"],
            results["predicted"],
        )

        selected_datasets = self._resolve_cmapss_datasets(
            datasets
        )

        metrics.update(
            {
                "evaluation_method": "final_window",
                "datasets": selected_datasets,
                "motor_count": int(len(results)),
                "clip_rul": bool(clip_rul),
                "rul_clip_value": (
                    int(rul_clip_value)
                    if clip_rul
                    else None
                ),
            }
        )

        self.external_test_results = results
        self.external_test_metrics = metrics

        return results.copy(), metrics.copy()

    def get_external_test_results(self) -> pd.DataFrame:
        if self.external_test_results is None:
            raise RuntimeError(
                "Official external test evaluation has not been executed. "
                "Call evaluate_cmapss_final_windows() first."
            )

        return self.external_test_results.copy()

    def get_all_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = self.get_metrics()

        if self.external_test_metrics is not None:
            metrics["external_test"] = (
                self.external_test_metrics.copy()
            )

        return metrics

    # ==========================================================
    # Visual diagnostics
    # ==========================================================

    def plot_training_history(
        self,
        metric: str = "loss",
    ) -> None:
        """
        Plot training and validation error by epoch.

        Examples:
            metric="loss"
            metric="mae"
            metric="rmse"
        """
        if self.history is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        if metric not in self.history.history:
            raise ValueError(
                f"Metric '{metric}' is not available. "
                f"Available values: {list(self.history.history)}"
            )

        validation_metric = f"val_{metric}"

        plt.figure(figsize=(8, 5))
        plt.plot(
            self.history.history[metric],
            label=f"Train {metric}",
        )

        if validation_metric in self.history.history:
            plt.plot(
                self.history.history[validation_metric],
                label=f"Validation {metric}",
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.title(
            f"Training and Validation {metric.upper()} — "
            f"{self.model_type}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        dataset: str = "validation",
    ) -> None:
        results = self.get_prediction_results(dataset)

        minimum = min(
            results["actual"].min(),
            results["predicted"].min(),
        )
        maximum = max(
            results["actual"].max(),
            results["predicted"].max(),
        )

        plt.figure(figsize=(7, 5))
        plt.scatter(
            results["actual"],
            results["predicted"],
            alpha=0.45,
        )
        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.title(
            f"Actual vs Predicted — {self.model_type} — {dataset}"
        )
        plt.tight_layout()
        plt.show()

    def plot_residuals(
        self,
        dataset: str = "validation",
    ) -> None:
        results = self.get_prediction_results(dataset)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            results["predicted"],
            results["residual"],
            alpha=0.45,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: actual - predicted")
        plt.title(
            f"Residuals — {self.model_type} — {dataset}"
        )
        plt.tight_layout()
        plt.show()

    def plot_train_validation_residuals(self) -> None:
        train = self.get_prediction_results("train")
        validation = self.get_prediction_results(
            "validation"
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            train["predicted"],
            train["residual"],
            alpha=0.2,
            label="Train",
        )
        plt.scatter(
            validation["predicted"],
            validation["residual"],
            alpha=0.5,
            label="Validation",
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: actual - predicted")
        plt.title(
            f"Train vs Validation Residuals — {self.model_type}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_external_test_predictions(self) -> None:
        results = self.get_external_test_results()

        minimum = min(
            results["actual"].min(),
            results["predicted"].min(),
        )
        maximum = max(
            results["actual"].max(),
            results["predicted"].max(),
        )

        plt.figure(figsize=(7, 5))
        plt.scatter(
            results["actual"],
            results["predicted"],
            alpha=0.6,
        )
        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )
        plt.xlabel("Official RUL")
        plt.ylabel("Predicted RUL")
        plt.title(
            f"Official C-MAPSS Test — {self.model_type}"
        )
        plt.tight_layout()
        plt.show()

    def plot_external_test_residuals(self) -> None:
        results = self.get_external_test_results()

        plt.figure(figsize=(8, 5))
        plt.scatter(
            results["predicted"],
            results["residual"],
            alpha=0.6,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: official - predicted")
        plt.title(
            f"Official Test Residuals — {self.model_type}"
        )
        plt.tight_layout()
        plt.show()

    def plot_validation_vs_external_test_residuals(self) -> None:
        """
        Compare development-validation residuals with the final official test
        residuals after evaluate_cmapss_final_windows() has been executed.
        """
        validation = self.get_prediction_results(
            "validation"
        )
        external = self.get_external_test_results()

        plt.figure(figsize=(9, 6))
        plt.scatter(
            validation["predicted"],
            validation["residual"],
            alpha=0.35,
            label="Validation motors",
        )
        plt.scatter(
            external["predicted"],
            external["residual"],
            alpha=0.7,
            label="Official external test",
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: actual - predicted")
        plt.title(
            f"Validation vs Official Test Residuals — "
            f"{self.model_type}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_error_metric_comparison(
        self,
        metric: str = "RMSE",
    ) -> None:
        """
        Compare one scalar metric across train, validation, and official test.
        """
        metric = metric.upper()

        if self.train_metrics is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        labels = ["Train", "Validation"]
        values = [
            self.train_metrics[metric],
            self.validation_metrics[metric],
        ]

        if self.external_test_metrics is not None:
            labels.append("Official test")
            values.append(
                self.external_test_metrics[metric]
            )

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.title(
            f"{metric} Comparison — {self.model_type}"
        )
        plt.tight_layout()
        plt.show()

    def plot_motor_predictions(
        self,
        group_id: Any,
        dataset: str = "validation",
    ) -> None:
        results = self.get_prediction_results(dataset)

        motor_results = results[
            results[self.group_column] == group_id
        ].copy()

        if motor_results.empty:
            raise ValueError(
                f"Motor '{group_id}' was not found in {dataset}."
            )

        motor_results = motor_results.sort_values(
            "target_cycle"
        )

        plt.figure(figsize=(10, 5))
        plt.plot(
            motor_results["target_cycle"],
            motor_results["actual"],
            label="Actual RUL",
        )
        plt.plot(
            motor_results["target_cycle"],
            motor_results["predicted"],
            label="Predicted RUL",
        )
        plt.xlabel("Cycle")
        plt.ylabel("RUL")
        plt.title(f"RUL Prediction for {group_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()
