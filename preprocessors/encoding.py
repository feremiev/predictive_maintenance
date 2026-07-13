import pandas as pd


class OperatingConditionEncoder:
    def __init__(
        self,
        column="operating_condition",
        prefix="operating_condition"
    ):
        self.column = column
        self.prefix = prefix
        self.encoded_columns = None

    def fit(self, dataframe):
        """
        Learn the possible operating-condition categories
        from the training dataset.
        """
        self.encoded_columns = [
            f"{self.prefix}_{category}"
            for category in sorted(dataframe[self.column].dropna().unique())
        ]

        return self

    def transform(self, dataframe):
        """
        Apply the learned encoding to another dataset.
        """
        if self.encoded_columns is None:
            raise ValueError(
                "The encoder has not been fitted. "
                "Run fit() or fit_transform() first."
            )

        dataframe = dataframe.copy()

        encoded = pd.get_dummies(
            dataframe[self.column],
            prefix=self.prefix,
            dtype=int
        )

        # Ensure all columns learned during fit are present
        encoded = encoded.reindex(
            columns=self.encoded_columns,
            fill_value=0
        )

        dataframe = pd.concat(
            [
                dataframe.drop(columns=[self.column]),
                encoded
            ],
            axis=1
        )

        return dataframe

    def fit_transform(self, dataframe):
        """
        Learn the categories and encode the same dataset.
        """
        print("")
        return self.fit(dataframe).transform(dataframe)
