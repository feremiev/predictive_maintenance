from __future__ import annotations

from typing import Any, Optional, Sequence

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


class SequenceRULModel:
    """
    Train sequence-based regression models for Remaining Useful Life prediction.

    This class is designed for datasets containing multiple independent time
    series, such as the NASA C-MAPSS turbofan dataset.

    Each motor is treated as one independent time series. Complete motors are
    assigned to train, validation, or test sets before windows are generated.
    This prevents overlapping cycles from the same motor from leaking across
    datasets.

    Supported window types
    ----------------------
    sliding:
        Uses a fixed number of cycles.

        Example with window_size=4:

            cycles 1-4 -> predict target
            cycles 2-5 -> predict target
            cycles 3-6 -> predict target

    growing:
        Uses all available history up to the current cycle, starting from
        min_window_size.

        Example with min_window_size=3:

            cycles 1-3   -> predict target
            cycles 1-4   -> predict target
            cycles 1-5   -> predict target

        Since neural networks require equal sequence dimensions, growing
        windows are left-padded to max_window_size.

    Supported models
    ----------------
    lstm:
        Stacked Long Short-Term Memory network.

    gru:
        Stacked Gated Recurrent Unit network.

    cnn:
        One-dimensional convolutional network.

    cnn_lstm:
        Convolutional feature extraction followed by an LSTM layer.

    Important
    ---------
    This class creates many-to-one models:

        sequence of sensor readings -> one RUL prediction

    The target can represent:

        - RUL at the final cycle inside the window
        - RUL some cycles into the future using prediction_horizon
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

        # Dataset split configuration
        test_group_count: Optional[int] = None,
        validation_group_count: Optional[int] = None,
        test_group_size: float = 0.15,
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
        verbose: int = 1,
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

        # Window configuration
        self.window_type = window_type.lower()
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = (
            max_window_size
            if max_window_size is not None
            else window_size
        )
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.padding_value = padding_value

        # Split configuration
        self.test_group_count = test_group_count
        self.validation_group_count = validation_group_count
        self.test_group_size = test_group_size
        self.validation_group_size = validation_group_size
        self.group_selection = group_selection.lower()
        self.random_state = random_state

        # Scaling configuration
        self.scaler_name = scaler.lower()
        self.scaler = None

        # Model configuration
        self.model_type = model_type.lower()
        self.recurrent_units = list(recurrent_units)
        self.dense_units = list(dense_units)
        self.cnn_filters = list(cnn_filters)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional

        # Training configuration
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_learning_rate = min_learning_rate
        self.verbose = verbose

        # Group identifiers
        self.train_group_ids = None
        self.validation_group_ids = None
        self.test_group_ids = None

        # Row-level split DataFrames
        self.train_df = None
        self.validation_df = None
        self.test_df = None

        # Sequence arrays
        self.X_train = None
        self.X_validation = None
        self.X_test = None

        self.y_train = None
        self.y_validation = None
        self.y_test = None

        # Metadata links each generated window to its motor and cycle.
        self.train_metadata = None
        self.validation_metadata = None
        self.test_metadata = None

        # TensorFlow model and history
        self.model = None
        self.history = None

        # Predictions
        self.y_train_pred = None
        self.y_validation_pred = None
        self.y_test_pred = None

        # Metrics
        self.train_metrics = None
        self.validation_metrics = None
        self.test_metrics = None

        self._validate_configuration()

    # ==========================================================
    # Configuration validation
    # ==========================================================

    def _validate_configuration(self) -> None:
        """Validate columns and constructor parameters."""

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

        if self.group_selection not in {
            "random",
            "first",
            "last",
        }:
            raise ValueError(
                "group_selection must be 'random', 'first', or 'last'."
            )

        if self.window_size < 2:
            raise ValueError(
                "window_size must be at least 2."
            )

        if self.min_window_size < 2:
            raise ValueError(
                "min_window_size must be at least 2."
            )

        if self.max_window_size < self.min_window_size:
            raise ValueError(
                "max_window_size cannot be smaller than min_window_size."
            )

        if self.stride < 1:
            raise ValueError(
                "stride must be at least 1."
            )

        if self.prediction_horizon < 0:
            raise ValueError(
                "prediction_horizon cannot be negative."
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        if self.df[self.target_column].isna().any():
            raise ValueError(
                f"Target column '{self.target_column}' contains null values."
            )

    # ==========================================================
    # Feature selection
    # ==========================================================

    def _resolve_feature_columns(self) -> list[str]:
        """
        Determine the columns used as sequence input features.

        If feature_columns was explicitly supplied, those columns are used.

        Otherwise, every numeric column is used except:
            - target column
            - group identifier
            - columns_to_drop

        The time column remains available by default. Add it to
        columns_to_drop if cycle should not be a model feature.
        """

        if self.feature_columns is not None:
            missing = set(self.feature_columns).difference(
                self.df.columns
            )

            if missing:
                raise ValueError(
                    f"Missing feature columns: {sorted(missing)}"
                )

            return self.feature_columns.copy()

        excluded = {
            self.target_column,
            self.group_column,
            *self.columns_to_drop,
        }

        columns = [
            column
            for column in self.df.select_dtypes(
                include=np.number
            ).columns
            if column not in excluded
        ]

        if not columns:
            raise ValueError(
                "No numeric feature columns are available."
            )

        return columns

    # ==========================================================
    # Group-aware splitting
    # ==========================================================

    def split_groups(self) -> None:
        """
        Divide complete motors into train, validation, and test sets.

        Motors are split before sequence windows are generated. This is
        essential because overlapping windows from the same motor must never
        appear in multiple datasets.
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

        if total_groups < 3:
            raise ValueError(
                "At least three complete groups are required for "
                "train, validation, and test splitting."
            )

        if self.test_group_count is not None:
            test_count = self.test_group_count
        else:
            test_count = max(
                1,
                int(np.ceil(total_groups * self.test_group_size)),
            )

        remaining_after_test = total_groups - test_count

        if self.validation_group_count is not None:
            validation_count = self.validation_group_count
        else:
            validation_count = max(
                1,
                int(
                    np.ceil(
                        total_groups
                        * self.validation_group_size
                    )
                ),
            )

        if test_count + validation_count >= total_groups:
            raise ValueError(
                "The combined validation and test group counts must be "
                "smaller than the total number of groups."
            )

        if self.group_selection == "random":
            rng = np.random.default_rng(
                self.random_state
            )

            shuffled_groups = rng.permutation(
                group_ids
            )

        elif self.group_selection == "last":
            shuffled_groups = group_ids

        else:
            # For "first", reverse the order so the first groups are
            # selected for test and validation.
            shuffled_groups = group_ids[::-1]

        if self.group_selection in {"random", "last"}:
            self.test_group_ids = shuffled_groups[-test_count:]

            self.validation_group_ids = shuffled_groups[
                -(test_count + validation_count):-test_count
            ]

            self.train_group_ids = shuffled_groups[
                :-(test_count + validation_count)
            ]

        else:
            self.test_group_ids = shuffled_groups[-test_count:]

            self.validation_group_ids = shuffled_groups[
                -(test_count + validation_count):-test_count
            ]

            self.train_group_ids = shuffled_groups[
                :-(test_count + validation_count)
            ]

        train_mask = data[self.group_column].isin(
            self.train_group_ids
        )

        validation_mask = data[self.group_column].isin(
            self.validation_group_ids
        )

        test_mask = data[self.group_column].isin(
            self.test_group_ids
        )

        self.train_df = data.loc[train_mask].copy()
        self.validation_df = data.loc[validation_mask].copy()
        self.test_df = data.loc[test_mask].copy()

        print(
            f"Train groups: {len(self.train_group_ids)} | "
            f"Validation groups: {len(self.validation_group_ids)} | "
            f"Test groups: {len(self.test_group_ids)}"
        )

        print(
            f"Train rows: {len(self.train_df):,} | "
            f"Validation rows: {len(self.validation_df):,} | "
            f"Test rows: {len(self.test_df):,}"
        )

    # ==========================================================
    # Scaling
    # ==========================================================

    def _create_scaler(self):
        """Create the selected sklearn scaler."""

        if self.scaler_name == "standard":
            return StandardScaler()

        if self.scaler_name == "minmax":
            return MinMaxScaler()

        if self.scaler_name == "robust":
            return RobustScaler()

        if self.scaler_name == "none":
            return None

        raise ValueError(
            f"Unsupported scaler: {self.scaler_name}"
        )

    def _fit_and_apply_scaler(self) -> None:
        """
        Fit the scaler only using training motors.

        Validation and test datasets are transformed using the statistics
        learned from training data. This prevents information leakage.
        """

        feature_columns = self._resolve_feature_columns()

        self.scaler = self._create_scaler()

        if self.scaler is None:
            return

        self.scaler.fit(
            self.train_df[feature_columns]
        )

        self.train_df.loc[:, feature_columns] = (
            self.scaler.transform(
                self.train_df[feature_columns]
            )
        )

        self.validation_df.loc[:, feature_columns] = (
            self.scaler.transform(
                self.validation_df[feature_columns]
            )
        )

        self.test_df.loc[:, feature_columns] = (
            self.scaler.transform(
                self.test_df[feature_columns]
            )
        )

    # ==========================================================
    # Window generation
    # ==========================================================

    def _left_pad_window(
        self,
        values: np.ndarray,
        desired_length: int,
    ) -> np.ndarray:
        """
        Left-pad a growing window to a fixed sequence length.

        Recent observations remain at the end of the sequence.

        Example:
            original length: 3
            desired length: 5

            [padding, padding, cycle1, cycle2, cycle3]
        """

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
            dtype=float,
        )

        return np.vstack(
            [padding, values]
        )

    def _generate_windows(
        self,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate sequence windows independently for each motor.

        Returns
        -------
        X:
            Shape:
                number of windows,
                sequence length,
                number of features

        y:
            One target value per generated window.

        metadata:
            Information about each window:
                group id
                starting cycle
                ending cycle
                target cycle
                source row index
        """

        feature_columns = self._resolve_feature_columns()

        X_windows = []
        y_values = []
        metadata_rows = []

        grouped = data.groupby(
            self.group_column,
            sort=False,
        )

        for group_id, group in grouped:
            group = group.sort_values(
                self.time_column
            )

            X_group = group[
                feature_columns
            ].to_numpy(dtype=np.float32)

            y_group = group[
                self.target_column
            ].to_numpy(dtype=np.float32)

            cycles = group[
                self.time_column
            ].to_numpy()

            indexes = group.index.to_numpy()

            sequence_length = len(group)

            if self.window_type == "sliding":
                minimum_required = (
                    self.window_size
                    + self.prediction_horizon
                )

                if sequence_length < minimum_required:
                    continue

                last_start = (
                    sequence_length
                    - self.window_size
                    - self.prediction_horizon
                    + 1
                )

                for start in range(
                    0,
                    last_start,
                    self.stride,
                ):
                    end = start + self.window_size

                    target_position = (
                        end - 1 + self.prediction_horizon
                    )

                    X_window = X_group[start:end]

                    X_windows.append(X_window)
                    y_values.append(
                        y_group[target_position]
                    )

                    metadata_rows.append({
                        self.group_column: group_id,
                        "window_start_cycle": cycles[start],
                        "window_end_cycle": cycles[end - 1],
                        "target_cycle": cycles[target_position],
                        "target_row_index": indexes[target_position],
                    })

            elif self.window_type == "growing":
                first_end = self.min_window_size

                last_end = (
                    sequence_length
                    - self.prediction_horizon
                )

                for end in range(
                    first_end,
                    last_end + 1,
                    self.stride,
                ):
                    target_position = (
                        end - 1 + self.prediction_horizon
                    )

                    start = max(
                        0,
                        end - self.max_window_size,
                    )

                    growing_values = X_group[start:end]

                    X_window = self._left_pad_window(
                        growing_values,
                        desired_length=self.max_window_size,
                    )

                    X_windows.append(X_window)
                    y_values.append(
                        y_group[target_position]
                    )

                    metadata_rows.append({
                        self.group_column: group_id,
                        "window_start_cycle": cycles[start],
                        "window_end_cycle": cycles[end - 1],
                        "target_cycle": cycles[target_position],
                        "target_row_index": indexes[target_position],
                        "observed_window_length": len(
                            growing_values
                        ),
                    })

        if not X_windows:
            raise ValueError(
                "No windows were created. Check window_size, "
                "min_window_size, max_window_size, prediction_horizon, "
                "and the lengths of the motor histories."
            )

        return (
            np.asarray(
                X_windows,
                dtype=np.float32,
            ),
            np.asarray(
                y_values,
                dtype=np.float32,
            ),
            pd.DataFrame(metadata_rows),
        )

    def prepare_data(self) -> None:
        """
        Perform the complete sequence-data preparation process.

        Steps
        -----
        1. Split complete motors.
        2. Fit the scaler using training motors only.
        3. Transform train, validation, and test data.
        4. Generate sequence windows independently in each split.
        """

        self.split_groups()
        self._fit_and_apply_scaler()

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

        (
            self.X_test,
            self.y_test,
            self.test_metadata,
        ) = self._generate_windows(
            self.test_df
        )

        print(
            "\nSequence shapes:"
        )

        print(
            f"X_train: {self.X_train.shape} | "
            f"y_train: {self.y_train.shape}"
        )

        print(
            f"X_validation: {self.X_validation.shape} | "
            f"y_validation: {self.y_validation.shape}"
        )

        print(
            f"X_test: {self.X_test.shape} | "
            f"y_test: {self.y_test.shape}"
        )

    # ==========================================================
    # Model creation
    # ==========================================================

    def _add_dense_head(
        self,
        model: keras.Sequential,
    ) -> None:
        """Add fully connected regression layers."""

        for units in self.dense_units:
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                )
            )

            if self.dropout > 0:
                model.add(
                    layers.Dropout(
                        self.dropout
                    )
                )

        model.add(
            layers.Dense(
                1,
                activation="linear",
            )
        )

    def _build_lstm_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        """Create an LSTM regression model."""

        model = keras.Sequential(
            name="lstm_rul_model"
        )

        model.add(
            layers.Input(
                shape=input_shape
            )
        )

        # Mask left-padding for growing windows.
        if self.window_type == "growing":
            model.add(
                layers.Masking(
                    mask_value=self.padding_value
                )
            )

        for index, units in enumerate(
            self.recurrent_units
        ):
            return_sequences = (
                index
                < len(self.recurrent_units) - 1
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
        """Create a GRU regression model."""

        model = keras.Sequential(
            name="gru_rul_model"
        )

        model.add(
            layers.Input(
                shape=input_shape
            )
        )

        if self.window_type == "growing":
            model.add(
                layers.Masking(
                    mask_value=self.padding_value
                )
            )

        for index, units in enumerate(
            self.recurrent_units
        ):
            return_sequences = (
                index
                < len(self.recurrent_units) - 1
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
        """
        Create a one-dimensional CNN regression model.

        Note:
            CNN layers do not use Keras Masking in the same way as recurrent
            layers. Sliding windows are generally preferable for this model.
        """

        model = keras.Sequential(
            name="cnn_rul_model"
        )

        model.add(
            layers.Input(
                shape=input_shape
            )
        )

        for filters in self.cnn_filters:
            model.add(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    activation="relu",
                )
            )

            model.add(
                layers.BatchNormalization()
            )

            model.add(
                layers.MaxPooling1D(
                    pool_size=self.pool_size,
                    padding="same",
                )
            )

            if self.dropout > 0:
                model.add(
                    layers.Dropout(
                        self.dropout
                    )
                )

        model.add(
            layers.GlobalAveragePooling1D()
        )

        self._add_dense_head(model)

        return model

    def _build_cnn_lstm_model(
        self,
        input_shape: tuple[int, int],
    ) -> keras.Model:
        """Create a CNN feature extractor followed by an LSTM."""

        model = keras.Sequential(
            name="cnn_lstm_rul_model"
        )

        model.add(
            layers.Input(
                shape=input_shape
            )
        )

        for filters in self.cnn_filters:
            model.add(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    activation="relu",
                )
            )

            model.add(
                layers.BatchNormalization()
            )

            if self.dropout > 0:
                model.add(
                    layers.Dropout(
                        self.dropout
                    )
                )

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
        """
        Build and compile the configured neural network.

        prepare_data() is called automatically when sequence arrays do not
        already exist.
        """

        if self.X_train is None:
            self.prepare_data()

        input_shape = self.X_train.shape[1:]

        if self.model_type == "lstm":
            self.model = self._build_lstm_model(
                input_shape
            )

        elif self.model_type == "gru":
            self.model = self._build_gru_model(
                input_shape
            )

        elif self.model_type == "cnn":
            self.model = self._build_cnn_model(
                input_shape
            )

        elif self.model_type == "cnn_lstm":
            self.model = self._build_cnn_lstm_model(
                input_shape
            )

        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        if self.loss == "huber":
            selected_loss = keras.losses.Huber()
        else:
            selected_loss = self.loss

        self.model.compile(
            optimizer=optimizer,
            loss=selected_loss,
            metrics=[
                keras.metrics.MeanAbsoluteError(
                    name="mae"
                ),
                keras.metrics.RootMeanSquaredError(
                    name="rmse"
                ),
            ],
        )

        return self.model

    # ==========================================================
    # Training
    # ==========================================================

    def _create_callbacks(self) -> list:
        """Create EarlyStopping and optional ReduceLROnPlateau callbacks."""

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
        Prepare data, build the model, train, predict, and calculate metrics.

        Returns
        -------
        dict
            Metrics for train, validation, and test sequence windows.
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
            shuffle=True,
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

        self.y_test_pred = (
            self.model.predict(
                self.X_test,
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

        self.test_metrics = self._calculate_metrics(
            self.y_test,
            self.y_test_pred,
        )

        return self.get_metrics()

    # ==========================================================
    # Metrics and results
    # ==========================================================

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate regression metrics."""

        y_true = np.asarray(
            y_true,
            dtype=float,
        ).reshape(-1)

        y_pred = np.asarray(
            y_pred,
            dtype=float,
        ).reshape(-1)

        residuals = y_true - y_pred

        denominator = np.where(
            y_true == 0,
            np.nan,
            np.abs(y_true),
        )

        return {
            "MAE": float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),
            "R2": float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
            "MAPE": float(
                np.nanmean(
                    np.abs(residuals)
                    / denominator
                )
                * 100
            ),
            "Bias": float(
                residuals.mean()
            ),
        }

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Return train, validation, and test metrics."""

        if self.test_metrics is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        return {
            "train": self.train_metrics.copy(),
            "validation": self.validation_metrics.copy(),
            "test": self.test_metrics.copy(),
        }

    def get_prediction_results(
        self,
        dataset: str = "test",
    ) -> pd.DataFrame:
        """
        Return metadata, actual values, predictions, and residuals.

        Parameters
        ----------
        dataset:
            train, validation, or test
        """

        dataset = dataset.lower()

        if dataset == "train":
            metadata = self.train_metadata.copy()
            y_true = self.y_train
            y_pred = self.y_train_pred

        elif dataset == "validation":
            metadata = self.validation_metadata.copy()
            y_true = self.y_validation
            y_pred = self.y_validation_pred

        elif dataset == "test":
            metadata = self.test_metadata.copy()
            y_true = self.y_test
            y_pred = self.y_test_pred

        else:
            raise ValueError(
                "dataset must be 'train', 'validation', or 'test'."
            )

        if y_pred is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        results = metadata.copy()

        results["actual"] = y_true
        results["predicted"] = y_pred

        results["residual"] = (
            results["actual"]
            - results["predicted"]
        )

        results["absolute_error"] = (
            results["residual"].abs()
        )

        return results

    # ==========================================================
    # External sequence prediction
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
        Prepare an external DataFrame using the already-fitted scaler and the
        same window configuration.

        The external DataFrame must contain the same input feature columns.

        Parameters
        ----------
        external_df:
            New motor histories not used during training.

        target_available:
            True when the external DataFrame contains the real target column.

        Returns
        -------
        X_external:
            Sequence windows.

        y_external:
            Target values, or None when target_available=False.

        metadata:
            Window-to-motor metadata.
        """

        if self.model is None:
            raise RuntimeError(
                "Train or build the model before external prediction."
            )

        feature_columns = self._resolve_feature_columns()

        required_columns = {
            self.group_column,
            self.time_column,
            *feature_columns,
        }

        if target_available:
            required_columns.add(
                self.target_column
            )

        missing = required_columns.difference(
            external_df.columns
        )

        if missing:
            raise ValueError(
                f"External data is missing columns: {sorted(missing)}"
            )

        external_data = external_df.copy()

        external_data = external_data.sort_values(
            [self.group_column, self.time_column]
        )

        if self.scaler is not None:
            external_data.loc[:, feature_columns] = (
                self.scaler.transform(
                    external_data[feature_columns]
                )
            )

        if target_available:
            return self._generate_windows(
                external_data
            )

        # A temporary target is required because the internal window generator
        # expects a target column. It is used only to locate target positions.
        external_data[self.target_column] = 0.0

        X_external, _, metadata = self._generate_windows(
            external_data
        )

        return X_external, None, metadata

    def predict_external(
        self,
        external_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict RUL for external unlabeled motor histories.
        """

        (
            X_external,
            _,
            metadata,
        ) = self.prepare_external_data(
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
        """
        Predict and evaluate an external labeled sequence dataset.
        """

        (
            X_external,
            y_external,
            metadata,
        ) = self.prepare_external_data(
            external_df,
            target_available=True,
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
            results["actual"]
            - results["predicted"]
        )

        results["absolute_error"] = (
            results["residual"].abs()
        )

        metrics = self._calculate_metrics(
            y_external,
            predictions,
        )

        return results, metrics

    # ==========================================================
    # Visual diagnostics
    # ==========================================================

    def plot_training_history(self) -> None:
        """Plot training and validation loss."""

        if self.history is None:
            raise RuntimeError(
                "The model has not been trained."
            )

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history.history["loss"],
            label="Train loss",
        )

        plt.plot(
            self.history.history["val_loss"],
            label="Validation loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Training History — {self.model_type}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        dataset: str = "test",
    ) -> None:
        """Plot actual versus predicted RUL."""

        results = self.get_prediction_results(
            dataset
        )

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
        dataset: str = "test",
    ) -> None:
        """Plot residuals against predicted values."""

        results = self.get_prediction_results(
            dataset
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            results["predicted"],
            results["residual"],
            alpha=0.45,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: actual - predicted")
        plt.title(
            f"Residuals — {self.model_type} — {dataset}"
        )
        plt.tight_layout()
        plt.show()

    def plot_train_validation_test_residuals(
        self,
    ) -> None:
        """Compare residuals for all three datasets."""

        train = self.get_prediction_results(
            "train"
        )

        validation = self.get_prediction_results(
            "validation"
        )

        test = self.get_prediction_results(
            "test"
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
            alpha=0.4,
            label="Validation",
        )

        plt.scatter(
            test["predicted"],
            test["residual"],
            alpha=0.55,
            label="Test",
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual: actual - predicted")
        plt.title(
            f"Train vs Validation vs Test Residuals — "
            f"{self.model_type}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_motor_predictions(
        self,
        group_id: Any,
        dataset: str = "test",
    ) -> None:
        """Plot predictions across cycles for one motor."""

        results = self.get_prediction_results(
            dataset
        )

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
        plt.title(
            f"RUL Prediction for {group_id}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
