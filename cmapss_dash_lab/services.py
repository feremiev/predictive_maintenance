from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from preprocessors.cmapss_preprocessor import CMapssPreprocessor



# =====================================================================
# Project paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Models exposed in the Dash interface
# =====================================================================

AVAILABLE_TABULAR_MODELS = [
    "linear",
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "svr",
]

AVAILABLE_SEQUENCE_MODELS = [
    "lstm",
    "gru",
    "cnn",
    "cnn_lstm",
]


# =====================================================================
# Project-class imports
# =====================================================================

def load_project_classes() -> tuple[dict[str, Any], str | None]:
    """
    Import the project classes from the current project structure.

    Expected structure
    ------------------
    project_root/
        models/
            base_model.py
            timeseries_model.py
        utils/
            data.py
            modelManager.py
        cmapss_dash_lab/
            app.py
            services.py
    """
    try:
        from utils.data import CMAPSSDataset
        from models.base_model import TimeSeriesRegressionModel
        from models.timeseries_model import SequenceRULModel
        from utils.modelManager import ExperimentManager

        return {
            "CMAPSSDataset": CMAPSSDataset,
            "TimeSeriesRegressionModel": TimeSeriesRegressionModel,
            "SequenceRULModel": SequenceRULModel,
            "ExperimentManager": ExperimentManager,
        }, None

    except Exception as exc:
        return {}, (
            f"{type(exc).__name__}: {exc}\n\n"
            "Check that models and utils are importable Python packages and "
            "that the class names match the imports in services.py."
        )


# =====================================================================
# Service layer
# =====================================================================

class ExperimentService:
    """
    Connect the Dash application with the dataset loader, model classes, and
    ExperimentManager.

    Workflow
    --------
    1. Load only train_FD00X files.
    2. Split complete training motors into train and validation groups.
    3. Train and save the experiment.
    4. Optionally load the saved experiment and evaluate it against:
           test_FD00X.txt + RUL_FD00X.txt
    5. Add the official external-test results to the same experiment folder.
    """

    def __init__(
        self,
        project_classes: dict[str, Any],
        experiments_folder: str | Path = "experiments",
    ) -> None:
        self.classes = project_classes

        experiments_path = Path(experiments_folder)

        if not experiments_path.is_absolute():
            experiments_path = PROJECT_ROOT / experiments_path

        self.experiments_folder = experiments_path

        self.manager = (
            self.classes["ExperimentManager"](
                base_folder=self.experiments_folder
            )
            if self.classes
            else None
        )

    @staticmethod
    def _validate_experiments_folder_name(
        folder_name: str | Path,
    ) -> str:
        raw_value = str(folder_name).strip() or "experiments"
        path = Path(raw_value)

        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError(
                "Use one project-local folder name, such as "
                "'experiments' or 'experiment_fd001'."
            )

        name = path.name

        if not name.lower().startswith("experiment"):
            raise ValueError(
                "Experiment folders must start with 'experiment'."
            )

        if not all(
            character.isalnum() or character in {"_", "-"}
            for character in name
        ):
            raise ValueError(
                "Use only letters, numbers, underscores, and hyphens."
            )

        return name

    def select_experiments_folder(
        self,
        folder_name: str | Path = "experiments",
    ) -> Path:
        name = self._validate_experiments_folder_name(
            folder_name
        )

        folder = PROJECT_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)

        if (
            self.manager is None
            or folder.resolve()
            != Path(self.manager.base_folder).resolve()
        ):
            self.experiments_folder = folder
            self.manager = self.classes["ExperimentManager"](
                base_folder=folder
            )

        return folder

    def list_experiment_folders(
        self,
    ) -> list[str]:
        (PROJECT_ROOT / "experiments").mkdir(
            parents=True,
            exist_ok=True,
        )

        folders = sorted(
            path.name
            for path in PROJECT_ROOT.iterdir()
            if path.is_dir()
            and path.name.lower().startswith("experiment")
        )

        if "experiments" not in folders:
            folders.insert(0, "experiments")

        return folders

    # =================================================================
    # Generic helpers
    # =================================================================

    @staticmethod
    def resolve_datasets(
        datasets: Sequence[str],
    ) -> tuple[str, ...]:
        """
        Validate and normalize selected C-MAPSS dataset identifiers.
        """
        if not datasets:
            raise ValueError(
                "Select at least one FD dataset."
            )

        allowed = {
            "FD001",
            "FD002",
            "FD003",
            "FD004",
        }

        resolved: list[str] = []

        for dataset in datasets:
            normalized = str(dataset).upper().strip()

            if normalized not in allowed:
                raise ValueError(
                    f"Invalid dataset '{dataset}'."
                )

            if normalized not in resolved:
                resolved.append(normalized)

        return tuple(resolved)

    @staticmethod
    def _construct_with_supported_kwargs(
        cls: type,
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Construct a class while passing only parameters accepted by its current
        constructor. This keeps the dashboard tolerant of small class-version
        differences.
        """
        signature = inspect.signature(cls.__init__)

        accepts_arbitrary_kwargs = any(
            parameter.kind
            == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if accepts_arbitrary_kwargs:
            return cls(**kwargs)

        supported = {
            name
            for name in signature.parameters
            if name != "self"
        }

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in supported
        }

        return cls(**filtered_kwargs)

    @staticmethod
    def _safe_prediction_table(
        experiment: Any,
        dataset: str,
    ) -> Optional[pd.DataFrame]:
        """
        Read a prediction table from a trained wrapper without failing when a
        split is unavailable.
        """
        if not hasattr(
            experiment,
            "get_prediction_results",
        ):
            return None

        try:
            result = experiment.get_prediction_results(
                dataset=dataset
            )
        except (
            AttributeError,
            FileNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

        if not isinstance(result, pd.DataFrame):
            return None

        return result.copy()

    @staticmethod
    def _history_dataframe(
        experiment: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Convert a Keras History object or stored history dictionary into a
        DataFrame.
        """
        history = getattr(
            experiment,
            "history",
            None,
        )

        if history is None:
            return None

        history_data = getattr(
            history,
            "history",
            history,
        )

        if isinstance(history_data, dict):
            return pd.DataFrame(history_data)

        if isinstance(history_data, pd.DataFrame):
            return history_data.copy()

        return None

    @staticmethod
    def _json_safe(
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, pd.Series):
            return value.tolist()

        if isinstance(value, pd.Index):
            return value.tolist()

        if isinstance(value, dict):
            return {
                str(key): ExperimentService._json_safe(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                ExperimentService._json_safe(item)
                for item in value
            ]

        return str(value)

    @staticmethod
    def _clean_dataframe(
        frame: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()

        cleaned = frame.copy()

        for column in cleaned.columns:
            cleaned[column] = cleaned[column].map(
                lambda value: (
                    None
                    if isinstance(value, float)
                    and np.isnan(value)
                    else value
                )
            )

        return cleaned

    @staticmethod
    def _resolve_data_folder(
        data_folder: str | Path,
    ) -> Path:
        """
        Resolve relative data paths from the project root rather than from the
        dashboard process working directory.
        """
        folder = Path(data_folder)

        if not folder.is_absolute():
            folder = PROJECT_ROOT / folder

        return folder.resolve()


    @staticmethod
    def _add_operating_condition(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a deterministic operating-condition category from the three
        C-MAPSS operating settings.

        The same transformation must be used for training, validation, and
        official test data.

        Rounding prevents tiny floating-point differences from creating
        unnecessarily different categories.
        """
        required_columns = {
            "setting_1",
            "setting_2",
            "setting_3",
        }

        missing = required_columns.difference(
            dataframe.columns
        )

        if missing:
            raise ValueError(
                "Cannot create operating_condition. "
                f"Missing settings: {sorted(missing)}"
            )

        result = dataframe.copy()

        setting_1 = (
            result["setting_1"]
            .astype(float)
            .round(1)
            .astype(str)
        )

        setting_2 = (
            result["setting_2"]
            .astype(float)
            .round(3)
            .astype(str)
        )

        setting_3 = (
            result["setting_3"]
            .astype(float)
            .round(0)
            .astype(int)
            .astype(str)
        )

        result["operating_condition"] = (
            setting_1
            + "_"
            + setting_2
            + "_"
            + setting_3
        )

        return result

    # =================================================================
    # Data loading
    # =================================================================

    def load_training_data(
        self,
        data_folder: str | Path,
        datasets: Sequence[str],
        remove_nulls: bool,
        clip_rul: bool,
        rul_cap: int,
    ) -> pd.DataFrame:
        """
        Load and combine selected C-MAPSS training files.

        The RUL target is calculated from the complete motor histories in the
        train files. Official test and RUL files are not read here.
        """
        if not self.classes:
            raise RuntimeError(
                "Project classes are not available."
            )

        resolved_folder = self._resolve_data_folder(
            data_folder
        )

        loader_class = self.classes[
            "CMAPSSDataset"
        ]

        loader_kwargs = {
            "data_folder": resolved_folder,
            "remove_nulls": remove_nulls,
            "create_rul": True,
            "clip_rul": clip_rul,
            "rul_clip_value": int(rul_cap),
        }

        loader = self._construct_with_supported_kwargs(
            loader_class,
            loader_kwargs,
        )

        frame = loader.load(
            file_type="train",
            datasets=self.resolve_datasets(
                datasets
            ),
            concatenate=True,
        )

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "CMAPSSDataset.load() must return a pandas DataFrame."
            )

        if frame.empty:
            raise ValueError(
                "The selected training files produced an empty DataFrame."
            )

        frame = self._add_operating_condition(
            frame
        )

        return frame


    def load_test_data_with_rul(
        self,
        data_folder: str | Path,
        datasets: list[str],
        remove_nulls: bool = True,
        clip_rul: bool = False,
        rul_cap: int = 125,
    ) -> pd.DataFrame:
        """
        Load C-MAPSS test files and attach the true RUL to every test row.

        The official RUL file contains one value per motor. That value represents
        the remaining useful life after the final cycle recorded in test_FDxxx.

        Row-level RUL is calculated as:

            max_observed_cycle
            + official_final_RUL
            - current_cycle

        Parameters
        ----------
        data_folder:
            Folder containing test_FDxxx.txt and RUL_FDxxx.txt.

        datasets:
            Dataset identifiers such as ["FD001", "FD002"].

        remove_nulls:
            Remove rows containing null values.

        clip_rul:
            Whether to cap the derived RUL.

        rul_cap:
            Maximum RUL when clipping is enabled.

        Returns
        -------
        pd.DataFrame
            Combined test dataset with official and row-level RUL information.
        """
        folder = Path(data_folder)

        if not folder.exists():
            raise FileNotFoundError(
                f"Data folder does not exist: {folder.resolve()}"
            )

        if not datasets:
            raise ValueError(
                "Select at least one test dataset."
            )

        column_names = [
            "unit_number",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3",
        ] + [
            f"sensor_{number}"
            for number in range(1, 22)
        ]

        all_test_frames = []

        for dataset_name in datasets:
            dataset_name = dataset_name.upper().strip()

            test_path = folder / f"test_{dataset_name}.txt"
            rul_path = folder / f"RUL_{dataset_name}.txt"

            if not test_path.is_file():
                raise FileNotFoundError(
                    f"Test file was not found: {test_path}"
                )

            if not rul_path.is_file():
                raise FileNotFoundError(
                    f"RUL file was not found: {rul_path}"
                )

            # ----------------------------------------------------------
            # Load test sensor history
            # ----------------------------------------------------------

            test_df = pd.read_csv(
                test_path,
                sep=r"\s+",
                header=None,
                names=column_names,
            )

            test_df["dataset"] = dataset_name

            test_df["unique_motor_id"] = (
                test_df["dataset"]
                + "_"
                + test_df["unit_number"].astype(str)
            )

            # ----------------------------------------------------------
            # Load official RUL values
            # ----------------------------------------------------------
            # The first RUL row corresponds to unit 1, the second to unit 2,
            # and so on inside each FD dataset.

            rul_df = pd.read_csv(
                rul_path,
                sep=r"\s+",
                header=None,
                names=["official_final_RUL"],
            )

            rul_df["unit_number"] = (
                np.arange(len(rul_df)) + 1
            )

            rul_df["dataset"] = dataset_name

            rul_df["unique_motor_id"] = (
                rul_df["dataset"]
                + "_"
                + rul_df["unit_number"].astype(str)
            )

            motor_count = test_df["unit_number"].nunique()

            if len(rul_df) != motor_count:
                raise ValueError(
                    f"{dataset_name} contains {motor_count} test motors, "
                    f"but {rul_path.name} contains {len(rul_df)} RUL values."
                )

            # ----------------------------------------------------------
            # Get the final observed cycle for every motor
            # ----------------------------------------------------------

            maximum_cycles = (
                test_df.groupby(
                    "unique_motor_id",
                    as_index=False,
                )["cycle"]
                .max()
                .rename(
                    columns={
                        "cycle": "max_observed_cycle",
                    }
                )
            )

            # ----------------------------------------------------------
            # Attach official RUL and maximum cycle to every test row
            # ----------------------------------------------------------

            test_df = test_df.merge(
                rul_df[
                    [
                        "unique_motor_id",
                        "official_final_RUL",
                    ]
                ],
                on="unique_motor_id",
                how="left",
                validate="many_to_one",
            )

            test_df = test_df.merge(
                maximum_cycles,
                on="unique_motor_id",
                how="left",
                validate="many_to_one",
            )

            # ----------------------------------------------------------
            # Calculate true RUL for every recorded test cycle
            # ----------------------------------------------------------

            test_df["RUL"] = (
                test_df["max_observed_cycle"]
                + test_df["official_final_RUL"]
                - test_df["cycle"]
            )

            if clip_rul:
                test_df["RUL"] = test_df["RUL"].clip(
                    upper=rul_cap
                )

            if remove_nulls:
                test_df = test_df.dropna().copy()

            all_test_frames.append(test_df)

        combined_test = pd.concat(
            all_test_frames,
            ignore_index=True,
        )

        combined_test = combined_test.sort_values(
            [
                "dataset",
                "unit_number",
                "cycle",
            ]
        ).reset_index(drop=True)

        combined_test = self._add_operating_condition(
            combined_test
        )

        return combined_test
    # =================================================================
    # Development experiment execution
    # =================================================================

    def run_and_save(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Train one tabular or sequence experiment using train and validation
        motors, then save all development artifacts.

        Returns
        -------
        dict
            experiment_name
            metrics
            train_predictions
            validation_predictions
            history
        """
        self.select_experiments_folder(
            config.get("experiments_folder", "experiments")
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        training_data = self.load_training_data(
            data_folder=config["data_folder"],
            datasets=config["datasets"],
            remove_nulls=config["remove_nulls"],
            clip_rul=config["clip_rul"],
            rul_cap=config["rul_cap"],
        )

        model_family = config["model_family"]

        if model_family == "tabular":
            experiment = self._create_tabular_model(
                training_data,
                config,
            )

        elif model_family == "sequence":
            experiment = self._create_sequence_model(
                training_data,
                config,
            )

        else:
            raise ValueError(
                f"Unsupported model family: {model_family}"
            )

        metrics = experiment.train()

        train_predictions = self._safe_prediction_table(
            experiment,
            "train",
        )

        validation_predictions = self._safe_prediction_table(
            experiment,
            "validation",
        )

        history = self._history_dataframe(
            experiment
        )

        learning_curve = None

        if model_family == "tabular":
            learning_curve = (
                experiment.calculate_learning_curve()
            )

        elif model_family == "sequence":
            learning_curve = (
                experiment.get_learning_curve()
            )

        resolved_data_folder = self._resolve_data_folder(
            config["data_folder"]
        )

        experiment_folder = self.manager.save_experiment(
            experiment=experiment,
            experiment_name=config.get(
                "experiment_name"
            ),
            notes=(
                "Created from the Dash interface. "
                f"Training datasets={config['datasets']}; "
                f"clip_rul={config['clip_rul']}; "
                f"rul_cap={config['rul_cap']}. "
                "Official external testing is stored separately."
            ),
            extra_config={
                "model_family": model_family,
                "experiments_folder": self.experiments_folder.name,
                "data_folder": str(
                    resolved_data_folder
                ),
                "datasets": list(
                    self.resolve_datasets(
                        config["datasets"]
                    )
                ),
                "remove_nulls": bool(
                    config["remove_nulls"]
                ),
                "clip_rul": bool(
                    config["clip_rul"]
                ),
                "rul_cap": int(
                    config["rul_cap"]
                ),
            },
            extra_tables=(
                {
                    "learning_curve": learning_curve,
                }
                if isinstance(
                    learning_curve,
                    pd.DataFrame,
                )
                and not learning_curve.empty
                else None
            ),
        )

        return {
            "experiment_name": experiment_folder.name,
            "metrics": self._json_safe(
                metrics
            ),
            "train_predictions": train_predictions,
            "validation_predictions": (
                validation_predictions
            ),
            "history": history,
            "learning_curve": learning_curve,
        }

    def _create_tabular_model(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> Any:
        """
        Build the tabular TimeSeriesRegressionModel using an internal
        train/validation motor split.
        """
        cls = self.classes[
            "TimeSeriesRegressionModel"
        ]

        kwargs = {
            "df": frame,
            "target_column": config[
                "target_column"
            ],
            "group_column": config[
                "group_column"
            ],
            "time_column": config[
                "time_column"
            ],
            "feature_columns": config.get(
                "feature_columns"
            ),
            "model_name": config[
                "model_name"
            ],
            "validation_group_count": config[
                "validation_group_count"
            ],
            "group_selection": config[
                "group_selection"
            ],
            "columns_to_drop": config[
                "columns_to_drop"
            ],
            "random_state": config[
                "random_state"
            ],
            "model_params": config[
                "model_params"
            ],
            "preprocessor": CMapssPreprocessor(),

        }

        return self._construct_with_supported_kwargs(
            cls,
            kwargs,
        )

    def _create_sequence_model(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> Any:
        """
        Build the sequence model using train and validation motors only.
        """
        cls = self.classes[
            "SequenceRULModel"
        ]

        architecture_parameters = dict(
            config.get(
                "model_params",
                {},
            )
        )

        kwargs = {
            "df": frame,
            "target_column": config[
                "target_column"
            ],
            "group_column": config[
                "group_column"
            ],
            "time_column": config[
                "time_column"
            ],
            "feature_columns": config.get(
                "feature_columns"
            ),
            "columns_to_drop": config.get(
                "columns_to_drop",
                [],
            ),
            "model_type": config[
                "model_name"
            ],
            "window_type": config[
                "window_type"
            ],
            "window_size": config[
                "window_size"
            ],
            "min_window_size": config[
                "min_window_size"
            ],
            "max_window_size": config[
                "max_window_size"
            ],
            "stride": config[
                "stride"
            ],
            "prediction_horizon": config[
                "prediction_horizon"
            ],
            "validation_group_count": config[
                "validation_group_count"
            ],
            "group_selection": config[
                "group_selection"
            ],
            "random_state": config[
                "random_state"
            ],
            "scaler": config[
                "scaler"
            ],
            "learning_rate": config[
                "learning_rate"
            ],
            "loss": config[
                "loss"
            ],
            "asymmetric_huber_late_weight": config.get(
                "asymmetric_huber_late_weight",
                2.5,
            ),
            "asymmetric_huber_delta": config.get(
                "asymmetric_huber_delta",
                10.0,
            ),
            "optimizer_clipnorm": config.get(
                "optimizer_clipnorm",
                1.0,
            ),
            "batch_size": config[
                "batch_size"
            ],
            "epochs": config[
                "epochs"
            ],
            "patience": config[
                "patience"
            ],
            **architecture_parameters,
            "preprocessor": CMapssPreprocessor(),
        }

        return self._construct_with_supported_kwargs(
            cls,
            kwargs,
        )

    # =================================================================
    # Official external test
    # =================================================================

    def run_external_test(
        self,
        experiment_name: str,
        data_folder: str | Path,
        datasets: Sequence[str],
        clip_rul: bool,
        rul_cap: int,
        experiments_folder: str | Path = "experiments",
    ) -> dict[str, Any]:
        """
        Load a saved model, reconstruct its wrapper configuration, evaluate it
        against official test + RUL files, and save those external results into
        the same experiment folder.

        Tabular models use:
            evaluate_cmapss_final_cycles()

        Sequence models use:
            evaluate_cmapss_final_windows()
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        loaded = self.manager.load_experiment(
            experiment_name
        )

        saved_config = dict(
            loaded.config or {}
        )

        model_family = (
            loaded.metadata.get(
                "model_family"
            )
            or saved_config.get(
                "model_family"
            )
        )

        training_data_folder = saved_config.get(
            "data_folder",
            data_folder,
        )

        training_datasets = saved_config.get(
            "datasets",
            datasets,
        )

        training_data = self.load_training_data(
            data_folder=training_data_folder,
            datasets=training_datasets,
            remove_nulls=bool(
                saved_config.get(
                    "remove_nulls",
                    True,
                )
            ),
            clip_rul=bool(
                saved_config.get(
                    "clip_rul",
                    False,
                )
            ),
            rul_cap=int(
                saved_config.get(
                    "rul_cap",
                    125,
                )
            ),
        )

        external_folder = self._resolve_data_folder(
            data_folder
        )

        if model_family == "sklearn":
            model_family = "tabular"

        if model_family == "tensorflow":
            model_family = "sequence"

        if model_family == "tabular":
            wrapper = self._rebuild_tabular_wrapper(
                training_data=training_data,
                saved_config=saved_config,
                loaded=loaded,
            )

            if not hasattr(
                wrapper,
                "evaluate_cmapss_final_cycles",
            ):
                raise AttributeError(
                    "TimeSeriesRegressionModel does not provide "
                    "evaluate_cmapss_final_cycles()."
                )

            results, metrics = (
                wrapper.evaluate_cmapss_final_cycles(
                    data_folder=external_folder,
                    datasets=list(
                        self.resolve_datasets(
                            datasets
                        )
                    ),
                    clip_rul=clip_rul,
                    rul_clip_value=int(
                        rul_cap
                    ),
                    preprocess_fn=self._add_operating_condition,
                )
            )

        elif model_family == "sequence":
            wrapper = self._rebuild_sequence_wrapper(
                training_data=training_data,
                saved_config=saved_config,
                loaded=loaded,
            )

            if not hasattr(
                wrapper,
                "evaluate_cmapss_final_windows",
            ):
                raise AttributeError(
                    "SequenceRULModel does not provide "
                    "evaluate_cmapss_final_windows()."
                )

            results, metrics = (
                wrapper.evaluate_cmapss_final_windows(
                    data_folder=external_folder,
                    datasets=list(
                        self.resolve_datasets(
                            datasets
                        )
                    ),
                    clip_rul=clip_rul,
                    rul_clip_value=int(
                        rul_cap
                    ),
                    preprocess_fn=self._add_operating_condition,
                )
            )

        else:
            raise ValueError(
                f"Unsupported saved model family: {model_family}"
            )

        self.manager.update_external_test(
            experiment_name_or_path=experiment_name,
            results=results,
            metrics=metrics,
        )

        return {
            "experiment_name": experiment_name,
            "metrics": self._json_safe(
                metrics
            ),
            "predictions": results,
        }

    @staticmethod
    def _production_status(
        predicted_rul: float,
        red_threshold: float,
        yellow_threshold: float,
    ) -> str:
        if predicted_rul <= red_threshold:
            return "Red"

        if predicted_rul <= yellow_threshold:
            return "Yellow"

        return "Green"

    def _load_production_wrapper(
        self,
        experiments_folder: str | Path,
        experiment_name: str,
        data_folder: str | Path,
    ) -> tuple[Any, Any, dict[str, Any], str]:
        """
        Load a saved experiment and reconstruct the correct prediction wrapper.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        loaded = self.manager.load_experiment(
            experiment_name
        )

        saved_config = dict(
            loaded.config or {}
        )

        model_family = (
            loaded.metadata.get(
                "model_family"
            )
            or saved_config.get(
                "model_family"
            )
        )

        if model_family == "sklearn":
            model_family = "tabular"

        if model_family == "tensorflow":
            model_family = "sequence"

        training_data = self.load_training_data(
            data_folder=saved_config.get(
                "data_folder",
                data_folder,
            ),
            datasets=saved_config.get(
                "datasets",
                ["FD001"],
            ),
            remove_nulls=bool(
                saved_config.get(
                    "remove_nulls",
                    True,
                )
            ),
            clip_rul=bool(
                saved_config.get(
                    "clip_rul",
                    False,
                )
            ),
            rul_cap=int(
                saved_config.get(
                    "rul_cap",
                    125,
                )
            ),
        )

        if model_family == "tabular":
            wrapper = self._rebuild_tabular_wrapper(
                training_data=training_data,
                saved_config=saved_config,
                loaded=loaded,
            )

        elif model_family == "sequence":
            wrapper = self._rebuild_sequence_wrapper(
                training_data=training_data,
                saved_config=saved_config,
                loaded=loaded,
            )

        else:
            raise ValueError(
                f"Unsupported saved model family: {model_family}"
            )

        return (
            wrapper,
            loaded,
            saved_config,
            model_family,
        )

    def production_fleet_snapshot(
        self,
        experiments_folder: str | Path,
        experiment_name: str,
        data_folder: str | Path,
        red_threshold: float = 25.0,
        yellow_threshold: float = 60.0,
    ) -> dict[str, Any]:
        """
        Predict one current RUL value per turbine from the official test files.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        loaded = self.manager.load_experiment(
            experiment_name
        )

        saved_config = dict(
            loaded.config or {}
        )

        datasets = self.resolve_datasets(
            saved_config.get(
                "datasets",
                ["FD001"],
            )
        )

        outcome = self.run_external_test(
            experiment_name=experiment_name,
            data_folder=data_folder,
            datasets=datasets,
            clip_rul=bool(
                saved_config.get(
                    "clip_rul",
                    False,
                )
            ),
            rul_cap=int(
                saved_config.get(
                    "rul_cap",
                    125,
                )
            ),
            experiments_folder=experiments_folder,
        )

        fleet = outcome["predictions"].copy()

        prediction_column = (
            "predicted_RUL"
            if "predicted_RUL" in fleet.columns
            else "predicted"
            if "predicted" in fleet.columns
            else None
        )

        if prediction_column is None:
            raise ValueError(
                "The production prediction table does not contain "
                "a predicted RUL column."
            )

        fleet["predicted_RUL"] = pd.to_numeric(
            fleet[prediction_column],
            errors="coerce",
        )

        fleet["status"] = fleet[
            "predicted_RUL"
        ].map(
            lambda value: self._production_status(
                float(value),
                float(red_threshold),
                float(yellow_threshold),
            )
        )

        keep_columns = [
            column
            for column in (
                "unique_motor_id",
                "dataset",
                "unit_number",
                "cycle",
                "predicted_RUL",
                "status",
                "official_RUL",
                "official_final_RUL",
                "actual",
                "absolute_error",
            )
            if column in fleet.columns
        ]

        fleet = (
            fleet[keep_columns]
            .sort_values(
                "predicted_RUL",
                ascending=True,
            )
            .reset_index(drop=True)
        )

        counts = (
            fleet["status"]
            .value_counts()
            .to_dict()
        )

        return {
            "experiment_name": experiment_name,
            "experiments_folder": str(
                experiments_folder
            ),
            "datasets": datasets,
            "fleet": fleet,
            "counts": {
                "Red": int(
                    counts.get("Red", 0)
                ),
                "Yellow": int(
                    counts.get("Yellow", 0)
                ),
                "Green": int(
                    counts.get("Green", 0)
                ),
            },
        }

    def production_turbine_editor(
        self,
        experiments_folder: str | Path,
        experiment_name: str,
        data_folder: str | Path,
        unique_motor_id: str,
    ) -> dict[str, Any]:
        """
        Return the latest editable parameter values for one test turbine.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        loaded = self.manager.load_experiment(
            experiment_name
        )

        saved_config = dict(
            loaded.config or {}
        )

        test_data = self.load_test_data_with_rul(
            data_folder=data_folder,
            datasets=self.resolve_datasets(
                saved_config.get(
                    "datasets",
                    ["FD001"],
                )
            ),
            remove_nulls=bool(
                saved_config.get(
                    "remove_nulls",
                    True,
                )
            ),
            clip_rul=False,
            rul_cap=int(
                saved_config.get(
                    "rul_cap",
                    125,
                )
            ),
        )

        turbine = test_data.loc[
            test_data["unique_motor_id"]
            == unique_motor_id
        ].sort_values(
            saved_config.get(
                "time_column",
                "cycle",
            )
        )

        if turbine.empty:
            raise ValueError(
                f"Turbine '{unique_motor_id}' was not found."
            )

        latest = turbine.iloc[-1]

        parameter_columns = [
            column
            for column in (
                [
                    "setting_1",
                    "setting_2",
                    "setting_3",
                ]
                + [
                    f"sensor_{number}"
                    for number in range(1, 22)
                ]
            )
            if column in turbine.columns
        ]

        values = {
            column: float(latest[column])
            for column in parameter_columns
        }

        return {
            "unique_motor_id": unique_motor_id,
            "dataset": latest.get(
                "dataset"
            ),
            "unit_number": int(
                latest.get(
                    "unit_number"
                )
            ),
            "cycle": int(
                latest.get(
                    "cycle"
                )
            ),
            "parameter_columns": (
                parameter_columns
            ),
            "values": values,
        }

    def predict_production_turbine(
        self,
        experiments_folder: str | Path,
        experiment_name: str,
        data_folder: str | Path,
        unique_motor_id: str,
        updated_values: dict[str, Any],
        red_threshold: float = 25.0,
        yellow_threshold: float = 60.0,
    ) -> dict[str, Any]:
        """
        Replace the latest telemetry values for one turbine and predict its RUL.
        """
        (
            wrapper,
            _loaded,
            saved_config,
            model_family,
        ) = self._load_production_wrapper(
            experiments_folder=experiments_folder,
            experiment_name=experiment_name,
            data_folder=data_folder,
        )

        test_data = self.load_test_data_with_rul(
            data_folder=data_folder,
            datasets=self.resolve_datasets(
                saved_config.get(
                    "datasets",
                    ["FD001"],
                )
            ),
            remove_nulls=bool(
                saved_config.get(
                    "remove_nulls",
                    True,
                )
            ),
            clip_rul=False,
            rul_cap=int(
                saved_config.get(
                    "rul_cap",
                    125,
                )
            ),
        )

        time_column = saved_config.get(
            "time_column",
            "cycle",
        )

        turbine = (
            test_data.loc[
                test_data["unique_motor_id"]
                == unique_motor_id
            ]
            .sort_values(time_column)
            .copy()
        )

        if turbine.empty:
            raise ValueError(
                f"Turbine '{unique_motor_id}' was not found."
            )

        latest_index = turbine.index[-1]

        editable_columns = {
            "setting_1",
            "setting_2",
            "setting_3",
            *{
                f"sensor_{number}"
                for number in range(1, 22)
            },
        }

        for column, value in (
            updated_values or {}
        ).items():
            if (
                column in editable_columns
                and column in turbine.columns
            ):
                turbine.loc[
                    latest_index,
                    column,
                ] = float(value)

        turbine = self._add_operating_condition(
            turbine
        )

        if model_family == "tabular":
            latest_row = turbine.tail(1)
            prediction = float(
                wrapper.predict(
                    latest_row
                )[0]
            )

        else:
            sequence_results = (
                wrapper.predict_external(
                    turbine
                )
            )

            if sequence_results.empty:
                raise ValueError(
                    "The selected turbine does not have enough "
                    "history to generate a sequence prediction."
                )

            prediction_column = (
                "predicted"
                if "predicted"
                in sequence_results.columns
                else "predicted_RUL"
            )

            prediction = float(
                sequence_results.iloc[-1][
                    prediction_column
                ]
            )

        prediction = max(
            0.0,
            prediction,
        )

        return {
            "unique_motor_id": unique_motor_id,
            "predicted_RUL": prediction,
            "status": self._production_status(
                prediction,
                float(red_threshold),
                float(yellow_threshold),
            ),
            "updated_values": self._json_safe(
                updated_values
            ),
        }

    def _rebuild_tabular_wrapper(
        self,
        training_data: pd.DataFrame,
        saved_config: dict[str, Any],
        loaded: Any,
    ) -> Any:
        """
        Recreate the tabular wrapper around the loaded fitted pipeline.

        Reconstructing the wrapper restores feature-selection and C-MAPSS
        external-evaluation behavior without retraining the estimator.
        """
        cls = self.classes[
            "TimeSeriesRegressionModel"
        ]

        kwargs = {
            "df": training_data,
            "target_column": saved_config.get(
                "target_column",
                "RUL",
            ),
            "group_column": saved_config.get(
                "group_column",
                "unique_motor_id",
            ),
            "time_column": saved_config.get(
                "time_column",
                "cycle",
            ),
            "feature_columns": saved_config.get(
                "feature_columns"
            ),
            "columns_to_drop": saved_config.get(
                "columns_to_drop",
                [],
            ),
            "model_name": saved_config.get(
                "model_name",
                saved_config.get(
                    "model_type",
                    "random_forest",
                ),
            ),
            "model_params": saved_config.get(
                "model_params",
                {},
            ),
            "validation_group_count": saved_config.get(
                "validation_group_count"
            ),
            "validation_group_size": saved_config.get(
                "validation_group_size",
                0.2,
            ),
            "group_selection": saved_config.get(
                "group_selection",
                "random",
            ),
            "random_state": saved_config.get(
                "random_state",
                42,
            ),
            "preprocessor": CMapssPreprocessor(),
        }

        wrapper = self._construct_with_supported_kwargs(
            cls,
            kwargs,
        )

        wrapper.pipeline = loaded.model

        if hasattr(
            loaded.model,
            "named_steps",
        ):
            named_steps = loaded.model.named_steps

            if "model" in named_steps:
                wrapper.model = named_steps[
                    "model"
                ]

            elif named_steps:
                wrapper.model = list(
                    named_steps.values()
                )[-1]

        else:
            wrapper.model = loaded.model

        if loaded.scaler is not None:
            wrapper.scaler = loaded.scaler

        # Reconstruct the raw train/validation inputs expected by the saved
        # pipeline. feature_names.json contains processed output features and
        # must never be used as the input DataFrame passed to the preprocessor.
        for method_name in (
            "split_data",
            "split_groups",
        ):
            method = getattr(
                wrapper,
                method_name,
                None,
            )

            if callable(method):
                method()
                break

        return wrapper

    def _rebuild_sequence_wrapper(
        self,
        training_data: pd.DataFrame,
        saved_config: dict[str, Any],
        loaded: Any,
    ) -> Any:
        """
        Recreate the sequence wrapper around the loaded Keras model and scaler.
        """
        cls = self.classes[
            "SequenceRULModel"
        ]

        kwargs = {
            "df": training_data,
            "target_column": saved_config.get(
                "target_column",
                "RUL",
            ),
            "group_column": saved_config.get(
                "group_column",
                "unique_motor_id",
            ),
            "time_column": saved_config.get(
                "time_column",
                "cycle",
            ),
            "feature_columns": saved_config.get(
                "feature_columns"
            ),
            "columns_to_drop": saved_config.get(
                "columns_to_drop",
                [],
            ),
            "model_type": saved_config.get(
                "model_type",
                "lstm",
            ),
            "window_type": saved_config.get(
                "window_type",
                "sliding",
            ),
            "window_size": saved_config.get(
                "window_size",
                30,
            ),
            "min_window_size": saved_config.get(
                "min_window_size",
                10,
            ),
            "max_window_size": saved_config.get(
                "max_window_size",
                saved_config.get(
                    "window_size",
                    30,
                ),
            ),
            "stride": saved_config.get(
                "stride",
                1,
            ),
            "prediction_horizon": saved_config.get(
                "prediction_horizon",
                0,
            ),
            "padding_value": saved_config.get(
                "padding_value",
                0.0,
            ),
            "validation_group_count": saved_config.get(
                "validation_group_count"
            ),
            "validation_group_size": saved_config.get(
                "validation_group_size",
                0.15,
            ),
            "group_selection": saved_config.get(
                "group_selection",
                "random",
            ),
            "random_state": saved_config.get(
                "random_state",
                42,
            ),
            "scaler": saved_config.get(
                "scaler_name",
                "standard",
            ),
            "recurrent_units": saved_config.get(
                "recurrent_units",
                [128, 64],
            ),
            "dense_units": saved_config.get(
                "dense_units",
                [64, 32],
            ),
            "cnn_filters": saved_config.get(
                "cnn_filters",
                [64, 128],
            ),
            "kernel_size": saved_config.get(
                "kernel_size",
                3,
            ),
            "pool_size": saved_config.get(
                "pool_size",
                2,
            ),
            "dropout": saved_config.get(
                "dropout",
                0.2,
            ),
            "recurrent_dropout": saved_config.get(
                "recurrent_dropout",
                0.0,
            ),
            "bidirectional": saved_config.get(
                "bidirectional",
                False,
            ),
            "learning_rate": saved_config.get(
                "learning_rate",
                0.001,
            ),
            "loss": saved_config.get(
                "loss",
                "huber",
            ),
            "asymmetric_huber_late_weight": (
                saved_config.get(
                    "asymmetric_huber_late_weight",
                    2.5,
                )
            ),
            "asymmetric_huber_delta": (
                saved_config.get(
                    "asymmetric_huber_delta",
                    10.0,
                )
            ),
            "optimizer_clipnorm": saved_config.get(
                "optimizer_clipnorm",
                1.0,
            ),
            "batch_size": saved_config.get(
                "batch_size",
                64,
            ),
            "epochs": saved_config.get(
                "epochs",
                100,
            ),
            "patience": saved_config.get(
                "patience",
                12,
            ),
            "reduce_lr": saved_config.get(
                "reduce_lr",
                True,
            ),
            "reduce_lr_patience": saved_config.get(
                "reduce_lr_patience",
                5,
            ),
            "reduce_lr_factor": saved_config.get(
                "reduce_lr_factor",
                0.5,
            ),
            "min_learning_rate": saved_config.get(
                "min_learning_rate",
                1e-6,
            ),
            "shuffle_windows": saved_config.get(
                "shuffle_windows",
                True,
            ),
            "verbose": 0,
            "preprocessor": CMapssPreprocessor(),
        }

        wrapper = self._construct_with_supported_kwargs(
            cls,
            kwargs,
        )

        wrapper.model = loaded.model
        wrapper.fitted_preprocessor = getattr(
            loaded,
            "preprocessor",
            None,
        )

        if wrapper.fitted_preprocessor is None:
            wrapper.fitted_preprocessor = getattr(
                loaded,
                "scaler",
                None,
            )

        if wrapper.fitted_preprocessor is None:
            raise RuntimeError(
                "The saved sequence experiment does not contain its fitted "
                "C-MAPSS preprocessor."
            )

        wrapper.processed_feature_columns_ = list(
            loaded.feature_names or []
        )

        if not wrapper.processed_feature_columns_:
            wrapper.processed_feature_columns_ = list(
                wrapper.fitted_preprocessor
                .get_feature_names_out()
            )
        return wrapper

    # =================================================================
    # Saved-experiment access
    # =================================================================

    def list_experiments(
        self,
        experiments_folder: str | Path = "experiments",
    ) -> pd.DataFrame:
        """
        Return saved experiments with train, validation, and external-test
        metrics kept in separate columns.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            return pd.DataFrame()

        frame = self.manager.list_experiments()

        return self._clean_dataframe(
            frame
        )

    def compare_experiments(
        self,
        experiment_names: Sequence[str],
        experiments_folder: str | Path = "experiments",
    ) -> pd.DataFrame:
        """
        Compare selected experiments using validation NASA score by default.

        External-test columns remain visible when those results exist.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            return pd.DataFrame()

        frame = self.manager.compare_experiments(
            experiments=list(
                experiment_names
            ),
            sort_by="validation_NASA_SCORE",
            ascending=True,
        )

        return self._clean_dataframe(
            frame
        )


    def load_saved(
        self,
        experiment_name: str,
        experiments_folder: str | Path = "experiments",
    ) -> dict[str, Any]:
        """
        Load all available artifacts required by the updated dashboard.
        """
        self.select_experiments_folder(
            experiments_folder
        )

        if self.manager is None:
            raise RuntimeError(
                "ExperimentManager is unavailable."
            )

        loaded = self.manager.load_experiment(
            experiment_name
        )

        train_predictions = self._load_saved_split(
            loaded,
            "train",
        )

        validation_predictions = self._load_saved_split(
            loaded,
            "validation",
        )

        external_test_predictions = (
            self._load_saved_split(
                loaded,
                "external_test",
            )
        )

        return {
            "metrics": loaded.metrics,
            "train_predictions": train_predictions,
            "validation_predictions": (
                validation_predictions
            ),
            "external_test_predictions": (
                external_test_predictions
            ),
            "history": loaded.history,
            "learning_curve": loaded.extra_tables.get(
                "learning_curve"
            ),
            "loaded": loaded,
        }

    @staticmethod
    def _load_saved_split(
        loaded: Any,
        split_name: str,
    ) -> Optional[pd.DataFrame]:
        try:
            predictions = loaded.get_predictions(
                split_name
            )
        except (
            AttributeError,
            FileNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

        if not isinstance(
            predictions,
            pd.DataFrame,
        ):
            return None

        return predictions.copy()

    # =================================================================
    # Metric formatting utilities
    # =================================================================

    @staticmethod
    def select_primary_metrics(
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Select validation metrics for development displays, then external test,
        then train metrics as fallbacks.
        """
        for split_name in (
            "validation",
            "external_test",
            "train",
        ):
            values = metrics.get(
                split_name
            )

            if isinstance(values, dict):
                return values

        if any(
            key in metrics
            for key in (
                "NASA_SCORE",
                "MEAN_NASA_SCORE",
                "MAE",
                "RMSE",
                "R2",
            )
        ):
            return metrics

        return {}

    @classmethod
    def metrics_to_dataframe(
        cls,
        metrics: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert train, validation, and official external-test metrics into a
        dashboard-friendly table.
        """
        rows = []

        display_names = {
            "train": "Train",
            "validation": "Validation",
            "external_test": "Official test",
        }

        for split_name in (
            "train",
            "validation",
            "external_test",
        ):
            values = metrics.get(
                split_name
            )

            if not isinstance(values, dict):
                continue

            row: dict[str, Any] = {
                "split": display_names[
                    split_name
                ]
            }

            for key, value in values.items():
                if isinstance(
                    value,
                    (int, float, np.number),
                ):
                    row[key] = float(value)

                elif key in {
                    "evaluation_method",
                    "datasets",
                }:
                    row[key] = value

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def format_metric(
        value: Any,
    ) -> str:
        if value is None:
            return "—"

        try:
            return f"{float(value):.4f}"
        except (
            TypeError,
            ValueError,
        ):
            return str(value)

    # =================================================================
    # Optional Plotly helpers retained for compatibility
    # =================================================================

    @staticmethod
    def prediction_figure(
        predictions: Optional[pd.DataFrame],
        title: str = "Actual vs predicted",
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(
                title=title
            )

        actual_column = (
            "actual"
            if "actual" in predictions.columns
            else None
        )

        predicted_column = (
            "predicted"
            if "predicted" in predictions.columns
            else "predicted_RUL"
            if "predicted_RUL"
            in predictions.columns
            else None
        )

        if (
            actual_column is None
            or predicted_column is None
        ):
            return go.Figure().update_layout(
                title=title
            )

        frame = predictions[
            [
                actual_column,
                predicted_column,
            ]
        ].dropna()

        figure = px.scatter(
            frame,
            x=actual_column,
            y=predicted_column,
            opacity=0.5,
            title=title,
        )

        if not frame.empty:
            lower = float(
                min(
                    frame[actual_column].min(),
                    frame[
                        predicted_column
                    ].min(),
                )
            )
            upper = float(
                max(
                    frame[actual_column].max(),
                    frame[
                        predicted_column
                    ].max(),
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    name="Ideal",
                    line={
                        "dash": "dash"
                    },
                )
            )

        return figure

    @staticmethod
    def residual_figure(
        predictions: Optional[pd.DataFrame],
        title: str = "Residuals",
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(
                title=title
            )

        frame = predictions.copy()

        predicted_column = (
            "predicted"
            if "predicted" in frame.columns
            else "predicted_RUL"
            if "predicted_RUL"
            in frame.columns
            else None
        )

        if predicted_column is None:
            return go.Figure().update_layout(
                title=title
            )

        if (
            "residual" not in frame.columns
            and "actual" in frame.columns
        ):
            frame["residual"] = (
                frame["actual"]
                - frame[predicted_column]
            )

        if "residual" not in frame.columns:
            return go.Figure().update_layout(
                title=title
            )

        figure = px.scatter(
            frame,
            x=predicted_column,
            y="residual",
            opacity=0.5,
            title=title,
        )

        figure.add_hline(
            y=0,
            line_dash="dash",
        )

        return figure

    @staticmethod
    def error_distribution_figure(
        predictions: Optional[pd.DataFrame],
        title: str = "Error distribution",
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(
                title=title
            )

        frame = predictions.copy()

        predicted_column = (
            "predicted"
            if "predicted" in frame.columns
            else "predicted_RUL"
            if "predicted_RUL"
            in frame.columns
            else None
        )

        if (
            "residual" not in frame.columns
            and predicted_column is not None
            and "actual" in frame.columns
        ):
            frame["residual"] = (
                frame["actual"]
                - frame[predicted_column]
            )

        if "residual" not in frame.columns:
            return go.Figure().update_layout(
                title=title
            )

        return px.histogram(
            frame,
            x="residual",
            nbins=50,
            title=title,
        )

    @staticmethod
    def history_figure(
        history: Optional[pd.DataFrame],
    ) -> go.Figure:
        if history is None or history.empty:
            return go.Figure().update_layout(
                title=(
                    "Training history — "
                    "available for sequence models"
                )
            )

        frame = history.reset_index(
            names="epoch"
        )

        figure = go.Figure()

        for column in (
            "loss",
            "val_loss",
            "mae",
            "val_mae",
            "rmse",
            "val_rmse",
            "val_nasa_score",
            "val_mean_nasa_score",
        ):
            if column in frame.columns:
                figure.add_trace(
                    go.Scatter(
                        x=frame["epoch"],
                        y=frame[column],
                        mode="lines",
                        name=column,
                    )
                )

        figure.update_layout(
            title="Training and validation history",
            xaxis_title="Epoch",
            yaxis_title="Metric value",
        )

        return figure

    @staticmethod
    def comparison_figure(
        comparison: pd.DataFrame,
    ) -> go.Figure:
        if comparison is None or comparison.empty:
            return go.Figure().update_layout(
                title="Experiment comparison"
            )

        metric = next(
            (
                column
                for column in (
                    "validation_NASA_SCORE",
                    "external_test_NASA_SCORE",
                    "validation_MEAN_NASA_SCORE",
                    "external_test_MEAN_NASA_SCORE",
                    "validation_RMSE",
                    "validation_MAE",
                    "external_test_RMSE",
                    "external_test_MAE",
                )
                if column in comparison.columns
            ),
            None,
        )

        if metric is None:
            return go.Figure().update_layout(
                title="Experiment comparison"
            )

        frame = comparison.dropna(
            subset=[metric]
        ).copy()

        return px.bar(
            frame,
            x="experiment_name",
            y=metric,
            color=(
                "model_type"
                if "model_type" in frame.columns
                else None
            ),
            title=f"Saved experiments by {metric}",
        )
