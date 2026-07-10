import os
import pandas as pd


###################################################### how to use #################################################
#loader = CMAPSSDataset(
#     data_folder="raw_data",
#     remove_nulls=True,
#     create_rul=True,
#     clip_rul=True,
#     rul_clip_value=125
# )

# train = loader.load(
#     file_type="train"
# )

# test = loader.load(
#     file_type="test"
# )

# rul = loader.load(
#     file_type="RUL"
# )

# fd001 = loader.load(
#     file_type="train",
#     datasets=("FD001",)
# )

# fd12 = loader.load(
#     file_type="train",
#     datasets=("FD001", "FD002")
# )

###################################################### how to use #################################################


class CMAPSSDataset:

    COLUMN_NAMES = (
        ['unit_number', 'cycle', 'setting_1', 'setting_2', 'setting_3']
        + [f'sensor_{i}' for i in range(1, 22)]
    )

    def __init__(self,
                 data_folder="raw_data",
                 remove_nulls=True,
                 create_rul=True,
                 clip_rul=True,
                 rul_clip_value=125):

        self.data_folder = data_folder
        self.remove_nulls = remove_nulls
        self.create_rul = create_rul
        self.clip_rul = clip_rul
        self.rul_clip_value = rul_clip_value

    ####################################################################
    # Public methods
    ####################################################################

    def load(self,
             file_type="train",
             datasets=("FD001", "FD002", "FD003", "FD004"),
             concatenate=True):
        """
        Reads one or multiple CMAPSS datasets.

        Parameters
        ----------
        file_type : str
            "train", "test" or "RUL"

        datasets : iterable
            Example:
                ("FD001",)
                ("FD001","FD002")

        concatenate : bool
            True -> returns one DataFrame
            False -> returns a dictionary of DataFrames
        """

        loaded = {}

        for dataset in datasets:

            df = self._read_file(file_type, dataset)

            if file_type != "RUL":
                df["dataset"] = dataset

                if self.remove_nulls:
                    df = df.dropna()

                if file_type == "train" and self.create_rul:
                    df = self._create_rul(df)

                df["unique_motor_id"] = (
                    df["dataset"] + "_" +
                    df["unit_number"].astype(str)
                )

            loaded[dataset] = df

        if concatenate:

            if file_type == "RUL":
                return pd.concat(
                    loaded.values(),
                    ignore_index=True
                )

            return pd.concat(
                loaded.values(),
                ignore_index=True
            )

        return loaded

    ####################################################################
    # Private methods
    ####################################################################

    def _read_file(self, file_type, dataset):

        filename = f"{file_type}_{dataset}.txt"

        path = os.path.join(
            self.data_folder,
            filename
        )

        if file_type == "RUL":

            df = pd.read_csv(
                path,
                header=None,
                names=["RUL"]
            )

        else:

            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                names=self.COLUMN_NAMES
            )

        return df

    def _create_rul(self, df):

        max_cycles = (
            df.groupby("unit_number")["cycle"]
            .max()
            .reset_index(name="max_cycle")
        )

        df = df.merge(
            max_cycles,
            on="unit_number"
        )

        df["RUL"] = (
            df["max_cycle"] - df["cycle"]
        )

        if self.clip_rul:
            df["RUL"] = df["RUL"].clip(
                upper=self.rul_clip_value
            )

        df.drop(
            columns="max_cycle",
            inplace=True
        )

        return df
