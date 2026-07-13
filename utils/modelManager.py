from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd


class ExperimentManager:
    """
    Save, load, list, compare, and manage machine-learning experiments.

    This manager supports:

    1. TensorFlow/Keras sequence models, such as SequenceRULModel.
    2. Scikit-learn pipelines, such as TimeSeriesRegressionModel.

    Each experiment is stored in its own folder.

    Example structure
    -----------------
    experiments/
        2026-07-12_143215_lstm_sliding/
            metadata.json
            config.json
            metrics.json
            model.keras
            scaler.joblib
            history.csv
            train_predictions.csv
            validation_predictions.csv
            test_predictions.csv
            feature_names.json
            group_ids.json
            notes.txt

    Important
    ---------
    Loading an experiment does not automatically recreate the complete original
    Python wrapper class. Instead, it returns a LoadedExperiment object
    containing:

        - fitted model
        - scaler
        - configuration
        - metrics
        - predictions
        - history
        - feature names
        - group IDs
        - notes

    This is normally enough to continue evaluating, plotting, comparing, and
    predicting without retraining.
    """

    def __init__(
        self,
        base_folder: str | Path = "experiments",
    ) -> None:
        """
        Parameters
        ----------
        base_folder:
            Root folder where experiments will be stored.
        """
        self.base_folder = Path(base_folder)
        self.base_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Public save method
    # ==========================================================

    def save_experiment(
        self,
        experiment: Any,
        experiment_name: Optional[str] = None,
        notes: Optional[str] = None,
        extra_config: Optional[dict[str, Any]] = None,
        extra_metrics: Optional[dict[str, Any]] = None,
        extra_tables: Optional[dict[str, pd.DataFrame]] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Save a trained experiment.

        Parameters
        ----------
        experiment:
            Trained SequenceRULModel, TimeSeriesRegressionModel, or another
            compatible object.

        experiment_name:
            Optional folder name. If omitted, a timestamped name is generated.

        notes:
            Optional human-readable notes.

        extra_config:
            Extra configuration values to include in config.json.

        extra_metrics:
            Extra metrics to include in metrics.json.

        extra_tables:
            Optional dictionary of DataFrames to save as CSV.

            Example:
                {
                    "metrics_by_motor": metrics_by_motor,
                    "feature_importance": feature_importance,
                }

        overwrite:
            Whether an existing experiment folder can be replaced.

        Returns
        -------
        Path
            Path to the saved experiment folder.
        """
        model_family = self._detect_model_family(
            experiment
        )

        if experiment_name is None:
            experiment_name = self._generate_experiment_name(
                experiment=experiment,
                model_family=model_family,
            )

        experiment_folder = (
            self.base_folder / experiment_name
        )

        if experiment_folder.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Experiment already exists: "
                    f"{experiment_folder}"
                )

            shutil.rmtree(
                experiment_folder
            )

        experiment_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = self._build_metadata(
            experiment=experiment,
            model_family=model_family,
            experiment_name=experiment_name,
        )

        config = self._extract_configuration(
            experiment
        )

        if extra_config:
            config.update(
                self._make_json_safe(extra_config)
            )

        metrics = self._extract_metrics(
            experiment
        )

        if extra_metrics:
            metrics.update(
                self._make_json_safe(extra_metrics)
            )

        self._write_json(
            experiment_folder / "metadata.json",
            metadata,
        )

        self._write_json(
            experiment_folder / "config.json",
            config,
        )

        self._write_json(
            experiment_folder / "metrics.json",
            metrics,
        )

        self._save_model(
            experiment=experiment,
            model_family=model_family,
            experiment_folder=experiment_folder,
        )

        self._save_scaler(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        self._save_training_history(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        self._save_predictions(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        self._save_feature_names(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        self._save_group_ids(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        self._save_feature_importance(
            experiment=experiment,
            experiment_folder=experiment_folder,
        )

        if notes is not None:
            notes_path = (
                experiment_folder / "notes.txt"
            )

            notes_path.write_text(
                notes,
                encoding="utf-8",
            )

        if extra_tables:
            self._save_extra_tables(
                extra_tables=extra_tables,
                experiment_folder=experiment_folder,
            )

        print(
            f"✅ Experiment saved at: "
            f"{experiment_folder}"
        )

        return experiment_folder

    # ==========================================================
    # Public load method
    # ==========================================================

    def load_experiment(
        self,
        experiment_name_or_path: str | Path,
    ) -> "LoadedExperiment":
        """
        Load a saved experiment.

        Parameters
        ----------
        experiment_name_or_path:
            Experiment folder name or complete folder path.

        Returns
        -------
        LoadedExperiment
            Object containing the saved model and experiment artifacts.
        """
        experiment_folder = self._resolve_experiment_path(
            experiment_name_or_path
        )

        metadata = self._read_json(
            experiment_folder / "metadata.json"
        )

        config = self._read_json(
            experiment_folder / "config.json"
        )

        metrics = self._read_json(
            experiment_folder / "metrics.json"
        )

        model_family = metadata[
            "model_family"
        ]

        model = self._load_model(
            experiment_folder=experiment_folder,
            model_family=model_family,
        )

        scaler = self._load_optional_joblib(
            experiment_folder / "scaler.joblib"
        )

        history = self._load_optional_csv(
            experiment_folder / "history.csv"
        )

        train_predictions = self._load_optional_csv(
            experiment_folder
            / "train_predictions.csv"
        )

        validation_predictions = self._load_optional_csv(
            experiment_folder
            / "validation_predictions.csv"
        )

        test_predictions = self._load_optional_csv(
            experiment_folder
            / "test_predictions.csv"
        )

        feature_importance = self._load_optional_csv(
            experiment_folder
            / "feature_importance.csv"
        )

        feature_names = self._read_optional_json(
            experiment_folder
            / "feature_names.json"
        )

        group_ids = self._read_optional_json(
            experiment_folder
            / "group_ids.json"
        )

        notes_path = (
            experiment_folder / "notes.txt"
        )

        notes = (
            notes_path.read_text(
                encoding="utf-8"
            )
            if notes_path.exists()
            else None
        )

        extra_tables = self._load_extra_tables(
            experiment_folder
        )

        return LoadedExperiment(
            folder=experiment_folder,
            metadata=metadata,
            config=config,
            metrics=metrics,
            model=model,
            scaler=scaler,
            history=history,
            train_predictions=train_predictions,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            feature_names=feature_names,
            group_ids=group_ids,
            feature_importance=feature_importance,
            notes=notes,
            extra_tables=extra_tables,
        )

    # ==========================================================
    # Experiment discovery
    # ==========================================================

    def list_experiments(
        self,
    ) -> pd.DataFrame:
        """
        Return a table containing all saved experiments.
        """
        rows = []

        for folder in sorted(
            self.base_folder.iterdir()
        ):
            if not folder.is_dir():
                continue

            metadata_path = (
                folder / "metadata.json"
            )

            metrics_path = (
                folder / "metrics.json"
            )

            if not metadata_path.exists():
                continue

            metadata = self._read_json(
                metadata_path
            )

            metrics = (
                self._read_json(metrics_path)
                if metrics_path.exists()
                else {}
            )

            test_metrics = self._find_test_metrics(
                metrics
            )

            rows.append({
                "experiment_name": folder.name,
                "created_at": metadata.get(
                    "created_at"
                ),
                "model_family": metadata.get(
                    "model_family"
                ),
                "model_type": metadata.get(
                    "model_type"
                ),
                "window_type": metadata.get(
                    "window_type"
                ),
                "test_MAE": test_metrics.get(
                    "MAE"
                ),
                "test_RMSE": test_metrics.get(
                    "RMSE"
                ),
                "test_R2": test_metrics.get(
                    "R2"
                ),
                "folder": str(folder),
            })

        return pd.DataFrame(rows)

    def compare_experiments(
        self,
        experiments: Optional[
            Sequence[str | Path]
        ] = None,
        sort_by: str = "test_RMSE",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Compare metrics from multiple experiments.

        Parameters
        ----------
        experiments:
            Experiment names or folders. If None, all experiments are used.

        sort_by:
            Column used for sorting.

        ascending:
            Sorting direction.
        """
        if experiments is None:
            comparison = self.list_experiments()

        else:
            rows = []

            for experiment_ref in experiments:
                loaded = self.load_experiment(
                    experiment_ref
                )

                test_metrics = self._find_test_metrics(
                    loaded.metrics
                )

                validation_metrics = (
                    loaded.metrics.get(
                        "validation",
                        {},
                    )
                )

                train_metrics = (
                    loaded.metrics.get(
                        "train",
                        {},
                    )
                )

                rows.append({
                    "experiment_name": (
                        loaded.folder.name
                    ),
                    "model_family": (
                        loaded.metadata.get(
                            "model_family"
                        )
                    ),
                    "model_type": (
                        loaded.metadata.get(
                            "model_type"
                        )
                    ),
                    "window_type": (
                        loaded.metadata.get(
                            "window_type"
                        )
                    ),
                    "train_MAE": train_metrics.get(
                        "MAE"
                    ),
                    "validation_MAE": (
                        validation_metrics.get(
                            "MAE"
                        )
                    ),
                    "test_MAE": test_metrics.get(
                        "MAE"
                    ),
                    "train_RMSE": train_metrics.get(
                        "RMSE"
                    ),
                    "validation_RMSE": (
                        validation_metrics.get(
                            "RMSE"
                        )
                    ),
                    "test_RMSE": test_metrics.get(
                        "RMSE"
                    ),
                    "train_R2": train_metrics.get(
                        "R2"
                    ),
                    "validation_R2": (
                        validation_metrics.get(
                            "R2"
                        )
                    ),
                    "test_R2": test_metrics.get(
                        "R2"
                    ),
                })

            comparison = pd.DataFrame(
                rows
            )

        if (
            not comparison.empty
            and sort_by in comparison.columns
        ):
            comparison = comparison.sort_values(
                sort_by,
                ascending=ascending,
                na_position="last",
            )

        return comparison.reset_index(
            drop=True
        )

    def delete_experiment(
        self,
        experiment_name_or_path: str | Path,
        confirm: bool = False,
    ) -> None:
        """
        Delete a saved experiment folder.

        confirm must be True to avoid accidental deletion.
        """
        if not confirm:
            raise ValueError(
                "Set confirm=True to delete an experiment."
            )

        experiment_folder = (
            self._resolve_experiment_path(
                experiment_name_or_path
            )
        )

        shutil.rmtree(
            experiment_folder
        )

        print(
            f"✅ Deleted experiment: "
            f"{experiment_folder}"
        )

    # ==========================================================
    # Metadata and configuration extraction
    # ==========================================================

    def _detect_model_family(
        self,
        experiment: Any,
    ) -> str:
        """
        Detect whether the experiment uses TensorFlow or scikit-learn.
        """
        if hasattr(
            experiment,
            "model",
        ):
            model = getattr(
                experiment,
                "model",
                None,
            )

            if (
                model is not None
                and hasattr(model, "save")
                and hasattr(model, "predict")
            ):
                return "tensorflow"

        if hasattr(
            experiment,
            "pipeline",
        ):
            pipeline = getattr(
                experiment,
                "pipeline",
                None,
            )

            if pipeline is not None:
                return "sklearn"

        raise TypeError(
            "Could not detect model family. "
            "Expected an object containing either a trained "
            "TensorFlow model in `.model` or a trained "
            "scikit-learn pipeline in `.pipeline`."
        )

    def _generate_experiment_name(
        self,
        experiment: Any,
        model_family: str,
    ) -> str:
        """
        Generate a timestamped experiment name.
        """
        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        model_type = getattr(
            experiment,
            "model_type",
            None,
        )

        if model_type is None:
            model_type = getattr(
                experiment,
                "model_name",
                model_family,
            )

        window_type = getattr(
            experiment,
            "window_type",
            None,
        )

        parts = [
            timestamp,
            str(model_type),
        ]

        if window_type:
            parts.append(
                str(window_type)
            )

        return "_".join(parts)

    def _build_metadata(
        self,
        experiment: Any,
        model_family: str,
        experiment_name: str,
    ) -> dict[str, Any]:
        """
        Build general experiment metadata.
        """
        return {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "model_family": model_family,
            "model_type": getattr(
                experiment,
                "model_type",
                getattr(
                    experiment,
                    "model_name",
                    None,
                ),
            ),
            "window_type": getattr(
                experiment,
                "window_type",
                None,
            ),
            "class_name": type(
                experiment
            ).__name__,
            "python_module": type(
                experiment
            ).__module__,
        }

    def _extract_configuration(
        self,
        experiment: Any,
    ) -> dict[str, Any]:
        """
        Extract relevant constructor and training configuration.
        """
        config_fields = [
            "target_column",
            "group_column",
            "time_column",
            "feature_columns",
            "columns_to_drop",
            "model_name",
            "model_type",
            "model_params",
            "window_type",
            "window_size",
            "min_window_size",
            "max_window_size",
            "stride",
            "prediction_horizon",
            "padding_value",
            "test_group_count",
            "validation_group_count",
            "test_group_size",
            "validation_group_size",
            "group_selection",
            "random_state",
            "scaler_name",
            "recurrent_units",
            "dense_units",
            "cnn_filters",
            "kernel_size",
            "pool_size",
            "dropout",
            "recurrent_dropout",
            "bidirectional",
            "learning_rate",
            "loss",
            "batch_size",
            "epochs",
            "patience",
            "reduce_lr",
            "reduce_lr_patience",
            "reduce_lr_factor",
            "min_learning_rate",
        ]

        config = {}

        for field in config_fields:
            if hasattr(
                experiment,
                field,
            ):
                config[field] = (
                    self._make_json_safe(
                        getattr(
                            experiment,
                            field,
                        )
                    )
                )

        return config

    def _extract_metrics(
        self,
        experiment: Any,
    ) -> dict[str, Any]:
        """
        Extract train, validation, test, and cross-validation metrics.
        """
        metrics = {}

        if getattr(
            experiment,
            "train_metrics",
            None,
        ) is not None:
            metrics["train"] = self._make_json_safe(
                experiment.train_metrics
            )

        if getattr(
            experiment,
            "validation_metrics",
            None,
        ) is not None:
            metrics["validation"] = (
                self._make_json_safe(
                    experiment.validation_metrics
                )
            )

        if getattr(
            experiment,
            "test_metrics",
            None,
        ) is not None:
            metrics["test"] = self._make_json_safe(
                experiment.test_metrics
            )

        if getattr(
            experiment,
            "cv_summary",
            None,
        ) is not None:
            metrics["cross_validation"] = (
                self._make_json_safe(
                    experiment.cv_summary
                )
            )

        if (
            not metrics
            and getattr(
                experiment,
                "metrics",
                None,
            ) is not None
        ):
            metrics["test"] = self._make_json_safe(
                experiment.metrics
            )

        return metrics

    # ==========================================================
    # Model persistence
    # ==========================================================

    def _save_model(
        self,
        experiment: Any,
        model_family: str,
        experiment_folder: Path,
    ) -> None:
        """
        Save the fitted model.
        """
        if model_family == "tensorflow":
            model_path = (
                experiment_folder / "model.keras"
            )

            experiment.model.save(
                model_path
            )

        elif model_family == "sklearn":
            pipeline_path = (
                experiment_folder
                / "pipeline.joblib"
            )

            joblib.dump(
                experiment.pipeline,
                pipeline_path,
            )

    def _load_model(
        self,
        experiment_folder: Path,
        model_family: str,
    ) -> Any:
        """
        Load a saved TensorFlow model or sklearn pipeline.
        """
        if model_family == "tensorflow":
            from tensorflow import keras

            model_path = (
                experiment_folder / "model.keras"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"TensorFlow model not found: "
                    f"{model_path}"
                )

            return keras.models.load_model(
                model_path
            )

        if model_family == "sklearn":
            pipeline_path = (
                experiment_folder
                / "pipeline.joblib"
            )

            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Scikit-learn pipeline not found: "
                    f"{pipeline_path}"
                )

            return joblib.load(
                pipeline_path
            )

        raise ValueError(
            f"Unsupported model family: "
            f"{model_family}"
        )

    # ==========================================================
    # Artifact persistence
    # ==========================================================

    def _save_scaler(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        scaler = getattr(
            experiment,
            "scaler",
            None,
        )

        if scaler is not None:
            joblib.dump(
                scaler,
                experiment_folder
                / "scaler.joblib",
            )

    def _save_training_history(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        history = getattr(
            experiment,
            "history",
            None,
        )

        if history is None:
            return

        history_dict = getattr(
            history,
            "history",
            history,
        )

        if isinstance(
            history_dict,
            dict,
        ):
            pd.DataFrame(
                history_dict
            ).to_csv(
                experiment_folder
                / "history.csv",
                index=False,
            )

    def _save_predictions(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        """
        Save prediction tables when get_prediction_results is available.
        """
        if not hasattr(
            experiment,
            "get_prediction_results",
        ):
            return

        datasets = [
            "train",
            "validation",
            "test",
        ]

        for dataset in datasets:
            try:
                results = (
                    experiment.get_prediction_results(
                        dataset=dataset
                    )
                )

            except (
                ValueError,
                RuntimeError,
                TypeError,
                AttributeError,
            ):
                continue

            if (
                isinstance(
                    results,
                    pd.DataFrame,
                )
                and not results.empty
            ):
                results.to_csv(
                    experiment_folder
                    / f"{dataset}_predictions.csv",
                    index=False,
                )

    def _save_feature_names(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        feature_names = None

        if getattr(
            experiment,
            "feature_columns",
            None,
        ) is not None:
            feature_names = list(
                experiment.feature_columns
            )

        elif getattr(
            experiment,
            "X_train",
            None,
        ) is not None:
            X_train = experiment.X_train

            if isinstance(
                X_train,
                pd.DataFrame,
            ):
                feature_names = list(
                    X_train.columns
                )

        if feature_names is not None:
            self._write_json(
                experiment_folder
                / "feature_names.json",
                feature_names,
            )

    def _save_group_ids(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        group_ids = {}

        for split_name in [
            "train",
            "validation",
            "test",
        ]:
            attribute_name = (
                f"{split_name}_group_ids"
            )

            values = getattr(
                experiment,
                attribute_name,
                None,
            )

            if values is not None:
                group_ids[split_name] = (
                    self._make_json_safe(
                        values
                    )
                )

        if group_ids:
            self._write_json(
                experiment_folder
                / "group_ids.json",
                group_ids,
            )

    def _save_feature_importance(
        self,
        experiment: Any,
        experiment_folder: Path,
    ) -> None:
        if not hasattr(
            experiment,
            "get_feature_importance",
        ):
            return

        try:
            importance = (
                experiment.get_feature_importance()
            )

        except (
            RuntimeError,
            ValueError,
            AttributeError,
        ):
            return

        if isinstance(
            importance,
            pd.DataFrame,
        ):
            importance.to_csv(
                experiment_folder
                / "feature_importance.csv",
                index=False,
            )

    def _save_extra_tables(
        self,
        extra_tables: dict[
            str,
            pd.DataFrame,
        ],
        experiment_folder: Path,
    ) -> None:
        extra_folder = (
            experiment_folder
            / "extra_tables"
        )

        extra_folder.mkdir(
            exist_ok=True
        )

        for table_name, table in extra_tables.items():
            if not isinstance(
                table,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"Extra table '{table_name}' "
                    "must be a pandas DataFrame."
                )

            safe_name = self._sanitize_filename(
                table_name
            )

            table.to_csv(
                extra_folder
                / f"{safe_name}.csv",
                index=False,
            )

    # ==========================================================
    # Generic helpers
    # ==========================================================

    def _resolve_experiment_path(
        self,
        experiment_name_or_path: str | Path,
    ) -> Path:
        supplied_path = Path(
            experiment_name_or_path
        )

        if supplied_path.exists():
            experiment_folder = supplied_path

        else:
            experiment_folder = (
                self.base_folder
                / supplied_path
            )

        if not experiment_folder.exists():
            raise FileNotFoundError(
                f"Experiment folder not found: "
                f"{experiment_folder}"
            )

        return experiment_folder

    @staticmethod
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _read_optional_json(
        self,
        path: Path,
    ) -> Any:
        if not path.exists():
            return None

        return self._read_json(
            path
        )

    @staticmethod
    def _load_optional_csv(
        path: Path,
    ) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None

        return pd.read_csv(
            path
        )

    @staticmethod
    def _load_optional_joblib(
        path: Path,
    ) -> Any:
        if not path.exists():
            return None

        return joblib.load(
            path
        )

    @staticmethod
    def _sanitize_filename(
        value: str,
    ) -> str:
        safe_characters = []

        for character in value:
            if (
                character.isalnum()
                or character in {
                    "_",
                    "-",
                }
            ):
                safe_characters.append(
                    character
                )
            else:
                safe_characters.append(
                    "_"
                )

        return "".join(
            safe_characters
        )

    def _load_extra_tables(
        self,
        experiment_folder: Path,
    ) -> dict[str, pd.DataFrame]:
        extra_folder = (
            experiment_folder
            / "extra_tables"
        )

        if not extra_folder.exists():
            return {}

        tables = {}

        for csv_path in extra_folder.glob(
            "*.csv"
        ):
            tables[
                csv_path.stem
            ] = pd.read_csv(
                csv_path
            )

        return tables

    @staticmethod
    def _find_test_metrics(
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        if "test" in metrics:
            return metrics["test"]

        if "external_test" in metrics:
            return metrics[
                "external_test"
            ]

        return {}

    @staticmethod
    def _make_json_safe(
        value: Any,
    ) -> Any:
        """
        Convert NumPy, pandas, Path, tuple, and other objects into
        JSON-compatible values.
        """
        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            Path,
        ):
            return str(
                value
            )

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            pd.Series,
        ):
            return value.tolist()

        if isinstance(
            value,
            pd.Index,
        ):
            return value.tolist()

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): ExperimentManager._make_json_safe(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                ExperimentManager._make_json_safe(
                    item
                )
                for item in value
            ]

        return str(
            value
        )

class LoadedExperiment:
    """
    Container for a previously saved experiment.

    This object provides convenient access to:

        - trained model
        - scaler
        - configuration
        - metrics
        - prediction tables
        - training history
        - group IDs
        - feature names
        - feature importance
        - notes
        - additional saved tables
    """

    def __init__(
        self,
        folder: Path,
        metadata: dict[str, Any],
        config: dict[str, Any],
        metrics: dict[str, Any],
        model: Any,
        scaler: Any = None,
        history: Optional[pd.DataFrame] = None,
        train_predictions: Optional[
            pd.DataFrame
        ] = None,
        validation_predictions: Optional[
            pd.DataFrame
        ] = None,
        test_predictions: Optional[
            pd.DataFrame
        ] = None,
        feature_names: Optional[list[str]] = None,
        group_ids: Optional[dict[str, Any]] = None,
        feature_importance: Optional[
            pd.DataFrame
        ] = None,
        notes: Optional[str] = None,
        extra_tables: Optional[
            dict[str, pd.DataFrame]
        ] = None,
    ) -> None:
        self.folder = folder
        self.metadata = metadata
        self.config = config
        self.metrics = metrics
        self.model = model
        self.scaler = scaler
        self.history = history
        self.train_predictions = train_predictions
        self.validation_predictions = (
            validation_predictions
        )
        self.test_predictions = test_predictions
        self.feature_names = feature_names
        self.group_ids = group_ids
        self.feature_importance = (
            feature_importance
        )
        self.notes = notes
        self.extra_tables = (
            extra_tables or {}
        )

    # ==========================================================
    # Information
    # ==========================================================

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact experiment summary.
        """
        return {
            "experiment_name": (
                self.metadata.get(
                    "experiment_name"
                )
            ),
            "created_at": self.metadata.get(
                "created_at"
            ),
            "model_family": self.metadata.get(
                "model_family"
            ),
            "model_type": self.metadata.get(
                "model_type"
            ),
            "window_type": self.metadata.get(
                "window_type"
            ),
            "metrics": self.metrics,
            "folder": str(
                self.folder
            ),
        }

    def get_predictions(
        self,
        dataset: str = "test",
    ) -> pd.DataFrame:
        """
        Return saved train, validation, or test predictions.
        """
        dataset = dataset.lower()

        if dataset == "train":
            predictions = (
                self.train_predictions
            )

        elif dataset == "validation":
            predictions = (
                self.validation_predictions
            )

        elif dataset == "test":
            predictions = (
                self.test_predictions
            )

        else:
            raise ValueError(
                "dataset must be 'train', "
                "'validation', or 'test'."
            )

        if predictions is None:
            raise FileNotFoundError(
                f"No saved {dataset} predictions "
                "were found."
            )

        return predictions.copy()

    # ==========================================================
    # Prediction
    # ==========================================================

    def predict_tabular(
        self,
        new_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict using a loaded scikit-learn pipeline.

        The new DataFrame must contain the same feature columns that were
        used during training.
        """
        if (
            self.metadata.get(
                "model_family"
            )
            != "sklearn"
        ):
            raise TypeError(
                "predict_tabular() can only be used "
                "with scikit-learn experiments."
            )

        if not self.feature_names:
            raise ValueError(
                "Feature names were not stored."
            )

        missing = set(
            self.feature_names
        ).difference(
            new_data.columns
        )

        if missing:
            raise ValueError(
                f"Missing input columns: "
                f"{sorted(missing)}"
            )

        return self.model.predict(
            new_data[self.feature_names]
        )

    def predict_sequences(
        self,
        X_sequences: np.ndarray,
    ) -> np.ndarray:
        """
        Predict using a loaded TensorFlow sequence model.

        X_sequences must already be transformed into the correct 3D shape:

            samples, timesteps, features

        It must also be scaled using the saved scaler when a scaler was used.
        """
        if (
            self.metadata.get(
                "model_family"
            )
            != "tensorflow"
        ):
            raise TypeError(
                "predict_sequences() can only be used "
                "with TensorFlow experiments."
            )

        predictions = self.model.predict(
            X_sequences,
            verbose=0,
        )

        return np.asarray(
            predictions
        ).reshape(-1)

    # ==========================================================
    # Plots
    # ==========================================================

    def plot_training_history(
        self,
        metric: str = "loss",
    ) -> None:
        """
        Plot a saved training metric and its validation equivalent.

        Examples
        --------
        plot_training_history("loss")
        plot_training_history("mae")
        plot_training_history("rmse")
        """
        import matplotlib.pyplot as plt

        if self.history is None:
            raise FileNotFoundError(
                "No training history was saved."
            )

        if metric not in self.history.columns:
            raise ValueError(
                f"Metric '{metric}' is not available. "
                f"Available columns: "
                f"{list(self.history.columns)}"
            )

        validation_metric = (
            f"val_{metric}"
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history[metric],
            label=f"Train {metric}",
        )

        if (
            validation_metric
            in self.history.columns
        ):
            plt.plot(
                self.history[
                    validation_metric
                ],
                label=(
                    f"Validation {metric}"
                ),
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(
            f"Training History — {metric}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        dataset: str = "test",
        actual_column: str = "actual",
        prediction_column: str = "predicted",
    ) -> None:
        """
        Plot actual versus predicted values from saved predictions.
        """
        import matplotlib.pyplot as plt

        results = self.get_predictions(
            dataset
        )

        required = {
            actual_column,
            prediction_column,
        }

        missing = required.difference(
            results.columns
        )

        if missing:
            raise ValueError(
                f"Missing prediction columns: "
                f"{sorted(missing)}"
            )

        minimum = min(
            results[actual_column].min(),
            results[prediction_column].min(),
        )

        maximum = max(
            results[actual_column].max(),
            results[prediction_column].max(),
        )

        plt.figure(figsize=(7, 5))

        plt.scatter(
            results[actual_column],
            results[prediction_column],
            alpha=0.5,
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"Actual vs Predicted — {dataset}"
        )
        plt.tight_layout()
        plt.show()

    def plot_residuals(
        self,
        dataset: str = "test",
        actual_column: str = "actual",
        prediction_column: str = "predicted",
    ) -> None:
        """
        Plot saved residuals against predictions.
        """
        import matplotlib.pyplot as plt

        results = self.get_predictions(
            dataset
        )

        required = {
            actual_column,
            prediction_column,
        }

        missing = required.difference(
            results.columns
        )

        if missing:
            raise ValueError(
                f"Missing prediction columns: "
                f"{sorted(missing)}"
            )

        residuals = (
            results[actual_column]
            - results[prediction_column]
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            results[prediction_column],
            residuals,
            alpha=0.5,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel("Predicted")
        plt.ylabel(
            "Residual: actual - predicted"
        )
        plt.title(
            f"Residuals — {dataset}"
        )
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(
        self,
        dataset: str = "test",
        actual_column: str = "actual",
        prediction_column: str = "predicted",
        bins: int = 30,
    ) -> None:
        """
        Plot the distribution of saved residuals.
        """
        import matplotlib.pyplot as plt

        results = self.get_predictions(
            dataset
        )

        residuals = (
            results[actual_column]
            - results[prediction_column]
        )

        plt.figure(figsize=(7, 5))

        plt.hist(
            residuals,
            bins=bins,
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
            f"Error Distribution — {dataset}"
        )
        plt.tight_layout()
        plt.show()
