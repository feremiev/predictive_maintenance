import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class OperatingConditionEncoder(BaseEstimator, TransformerMixin):
    """
    Round the three operating settings, create an operating-condition
    category, and one-hot encode it.

    Default rounding for NASA C-MAPSS:
        setting_1 -> 0 decimal places
        setting_2 -> 2 decimal places
        setting_3 -> 0 decimal places
    """

    def __init__(
        self,
        setting_columns: tuple[str, str, str] = (
            "setting_1",
            "setting_2",
            "setting_3",
        ),
        rounding_decimals: tuple[int, int, int] = (0, 2, 0),
        column: str = "operating_condition",
        prefix: str = "operating_condition",
        drop_original: bool = True,
        drop_rounded_settings: bool = False,
    ):
        self.setting_columns = setting_columns
        self.rounding_decimals = rounding_decimals
        self.column = column
        self.prefix = prefix
        self.drop_original = drop_original
        self.drop_rounded_settings = drop_rounded_settings

        self.categories_: list[str] | None = None
        self.encoded_columns_: list[str] | None = None
        self.rounded_setting_columns_: list[str] | None = None

    def _create_operating_condition(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Round settings and create the operating-condition column."""

        result = X.copy()

        missing_columns = [
            column
            for column in self.setting_columns
            if column not in result.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing setting columns: {missing_columns}"
            )

        if len(self.setting_columns) != len(self.rounding_decimals):
            raise ValueError(
                "'setting_columns' and 'rounding_decimals' "
                "must have the same length."
            )

        self.rounded_setting_columns_ = []

        # Create rounded versions of the three setting columns
        for setting_column, decimals in zip(
            self.setting_columns,
            self.rounding_decimals,
        ):
            rounded_column = f"{setting_column}_round"

            result[rounded_column] = (
                result[setting_column]
                .round(decimals)
                .fillna(0)
            )

            self.rounded_setting_columns_.append(
                rounded_column
            )

        # Combine rounded settings into one operating-condition label
        result[self.column] = (
            result[self.rounded_setting_columns_]
            .astype(str)
            .agg("_".join, axis=1)
        )

        return result

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        result = self._create_operating_condition(X)

        self.categories_ = sorted(
            result[self.column]
            .dropna()
            .unique()
            .tolist()
        )

        self.encoded_columns_ = [
            f"{self.prefix}_{category}"
            for category in self.categories_
        ]

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if (
            self.categories_ is None
            or self.encoded_columns_ is None
        ):
            raise RuntimeError(
                "The encoder has not been fitted. "
                "Call fit() before transform()."
            )

        result = self._create_operating_condition(X)

        # Use categories learned during fit
        operating_condition = pd.Categorical(
            result[self.column],
            categories=self.categories_,
        )

        encoded = pd.get_dummies(
            operating_condition,
            prefix=self.prefix,
            dtype=float,
        )

        encoded.index = result.index

        encoded = encoded.reindex(
            columns=self.encoded_columns_,
            fill_value=0.0,
        )

        if self.drop_original:
            result = result.drop(
                columns=[self.column]
            )

        if self.drop_rounded_settings:
            result = result.drop(
                columns=self.rounded_setting_columns_,
            )

        return pd.concat(
            [result, encoded],
            axis=1,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        if self.encoded_columns_ is None:
            raise RuntimeError(
                "The encoder has not been fitted."
            )

        return self.encoded_columns_
