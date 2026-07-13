import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class OperatingConditionEncoder(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(
        self,
        column: str = "operating_condition",
        prefix: str = "operating_condition",
        drop_original: bool = True,
    ):
        self.column = column
        self.prefix = prefix
        self.drop_original = drop_original
        self.encoded_columns_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        if self.column not in X.columns:
            raise ValueError(
                f"Column '{self.column}' was not found."
            )

        categories = sorted(
            X[self.column]
            .dropna()
            .unique()
        )

        self.encoded_columns_ = [
            f"{self.prefix}_{category}"
            for category in categories
        ]

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.encoded_columns_ is None:
            raise RuntimeError(
                "The encoder has not been fitted."
            )

        if self.column not in X.columns:
            raise ValueError(
                f"Column '{self.column}' was not found."
            )

        result = X.copy()

        encoded = pd.get_dummies(
            result[self.column],
            prefix=self.prefix,
            dtype=float,
        )

        encoded = encoded.reindex(
            columns=self.encoded_columns_,
            fill_value=0.0,
        )

        if self.drop_original:
            result = result.drop(
                columns=[self.column]
            )

        return pd.concat(
            [result, encoded],
            axis=1,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return self.encoded_columns_
