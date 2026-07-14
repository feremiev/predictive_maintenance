import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from preprocessors.operating_condition_encoder import OperatingConditionEncoder

class CMapssPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns
        self.features_settings = [
            "setting_1",
            "setting_2",
            "setting_3",
        ]

        self.features_sensors = [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        self.correct_order_ = (
            self.features_settings
            + self.features_sensors
        )

        self.features_minmax = [
            "sensor_1",
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_5",
            "sensor_6",
            "sensor_7",
            "sensor_9",
            "sensor_10",
            "sensor_11",
            "sensor_12",
            "sensor_16",
            "sensor_17",
            "sensor_20",
            "sensor_21",
        ]

        self.features_robust = [
            "sensor_8",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_18",
            "sensor_19",
        ]

        self.scaling_pipeline = ColumnTransformer(
            transformers=[
                (
                    "num_minmax",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            MinMaxScaler(),
                        ),
                    ]),
                    self.features_minmax,
                ),
                (
                    "num_robust",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            RobustScaler(),
                        ),
                    ]),
                    self.features_robust,
                ),
            ],
            remainder="passthrough",
        )

        self.operating_condition_encoder = OperatingConditionEncoder()
        self.encoded_columns_ = []

    def _selected_output_columns(self) -> list[str]:
        """Return the processed columns exposed to the model."""
        all_columns = self.correct_order_ + self.encoded_columns_

        if self.feature_columns is None:
            return all_columns

        requested = list(self.feature_columns)
        selected: list[str] = []

        for column in requested:
            if column == "operating_condition":
                selected.extend(self.encoded_columns_)
            elif column in all_columns:
                selected.append(column)
            else:
                raise ValueError(
                    f"Unknown C-MAPSS feature column: {column!r}. "
                    f"Available raw features are: "
                    f"{self.correct_order_ + ['operating_condition']}"
                )

        # Preserve user order while removing duplicates.
        return list(dict.fromkeys(selected))

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        self._validate_columns(X)

        # Learn the operating-condition categories
        self.operating_condition_encoder.fit(X)

        self.encoded_columns_ = list(
            self.operating_condition_encoder.encoded_columns_
        )

        # Learn scaling statistics
        X_input = X[self.correct_order_]

        self.scaling_pipeline.fit(
            X_input,
            y,
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate_columns(X)

        # ----------------------------
        # Scale the numeric features
        # ----------------------------

        X_input = X[self.correct_order_]

        transformed = (
            self.scaling_pipeline.transform(
                X_input
            )
        )

        sklearn_order = (
            self.features_minmax
            + self.features_robust
            + self.features_settings
        )

        scaled_df = pd.DataFrame(
            transformed,
            columns=sklearn_order,
            index=X.index,
        )

        scaled_df = scaled_df[
            self.correct_order_
        ].astype(float)

        # ----------------------------
        # Encode operating condition
        # ----------------------------

        encoded = self.operating_condition_encoder.transform(X)

        encoded = encoded[
            self.encoded_columns_
        ]

        # ----------------------------
        # Join both
        # ----------------------------

        processed = pd.concat(
            [
                scaled_df,
                encoded,
            ],
            axis=1,
        )

        return processed[self._selected_output_columns()]

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return self._selected_output_columns()

    def _validate_columns(
        self,
        X: pd.DataFrame,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CMapssPreprocessor expects a pandas DataFrame."
            )

        required = (
            self.correct_order_
            + ["operating_condition"]
        )

        missing = set(
            required
        ).difference(X.columns)

        if missing:
            raise ValueError(
                f"Missing C-MAPSS columns: "
                f"{sorted(missing)}"
            )
