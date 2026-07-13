
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVAILABLE_TABULAR_MODELS = [
    "linear",
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "svr",
]

AVAILABLE_SEQUENCE_MODELS = [
    "lstm",
    "gru",
    "cnn",
    "cnn_lstm",
]


def load_project_classes() -> tuple[dict[str, Any], str | None]:
    """
    Import the project classes.

    Expected project layout
    -----------------------
    cmapss_dataset.py
        CMAPSSDataset

    time_series_regression_model.py
        TimeSeriesRegressionModel

    sequence_rul_model.py
        SequenceRULModel

    experiment_manager.py
        ExperimentManager

    You can change these imports here if your module names are different.
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
            "Place your four class modules beside app.py, or update "
            "load_project_classes() in services.py."
        )


class ExperimentService:
    """
    Adapter between the Dash interface and the project model classes.

    Keeping orchestration outside callbacks makes the interface easier to test
    and prevents UI code from becoming tightly coupled to model internals.
    """

    def __init__(
        self,
        project_classes: dict[str, Any],
        experiments_folder: str | Path = "experiments",
    ) -> None:
        self.classes = project_classes
        self.experiments_folder = Path(experiments_folder)

        self.manager = (
            self.classes["ExperimentManager"](
                base_folder=self.experiments_folder
            )
            if self.classes
            else None
        )

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------

    @staticmethod
    def resolve_datasets(datasets: list[str]) -> tuple[str, ...]:
        if not datasets:
            raise ValueError("Select at least one FD dataset.")
        return tuple(datasets)

    def load_training_data(
        self,
        data_folder: str,
        datasets: list[str],
        remove_nulls: bool,
        clip_rul: bool,
        rul_cap: int,
    ) -> pd.DataFrame:
        loader = self.classes["CMAPSSDataset"](
            data_folder=data_folder,
            remove_nulls=remove_nulls,
            create_rul=True,
            clip_rul=clip_rul,
            rul_clip_value=rul_cap,
        )

        frame = loader.load(
            file_type="train",
            datasets=self.resolve_datasets(datasets),
            concatenate=True,
        )

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("CMAPSSDataset.load() must return a DataFrame.")

        return frame

    # -----------------------------------------------------------------
    # Experiment execution
    # -----------------------------------------------------------------

    def run_and_save(self, config: dict[str, Any]) -> dict[str, Any]:
        frame = self.load_training_data(
            data_folder=config["data_folder"],
            datasets=config["datasets"],
            remove_nulls=config["remove_nulls"],
            clip_rul=config["clip_rul"],
            rul_cap=config["rul_cap"],
        )

        if config["model_family"] == "tabular":
            experiment = self._create_tabular_model(frame, config)
        elif config["model_family"] == "sequence":
            experiment = self._create_sequence_model(frame, config)
        else:
            raise ValueError(
                f"Unsupported model family: {config['model_family']}"
            )

        metrics = experiment.train()

        experiment_folder = self.manager.save_experiment(
            experiment=experiment,
            experiment_name=config["experiment_name"],
            notes=(
                f"Created from Dash. Datasets={config['datasets']}; "
                f"clip_rul={config['clip_rul']}; rul_cap={config['rul_cap']}."
            ),
            extra_config={
                "data_folder": config["data_folder"],
                "datasets": config["datasets"],
                "remove_nulls": config["remove_nulls"],
                "clip_rul": config["clip_rul"],
                "rul_cap": config["rul_cap"],
            },
        )

        predictions = self._best_prediction_table(experiment)
        history = self._history_dataframe(experiment)

        return {
            "experiment_name": experiment_folder.name,
            "metrics": self._json_safe(metrics),
            "predictions": predictions,
            "history": history,
        }

    def _create_tabular_model(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> Any:
        cls = self.classes["TimeSeriesRegressionModel"]

        return cls(
            df=frame,
            target_column=config["target_column"],
            group_column=config["group_column"],
            time_column=config["time_column"],
            model_name=config["model_name"],
            test_group_count=config["test_group_count"],
            group_selection=config["group_selection"],
            columns_to_drop=config["columns_to_drop"],
            random_state=config["random_state"],
            model_params=config["model_params"],
        )

    def _create_sequence_model(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> Any:
        cls = self.classes["SequenceRULModel"]

        # model_params can override architecture defaults accepted by
        # SequenceRULModel, such as recurrent_units, dense_units, cnn_filters,
        # dropout, kernel_size, or pool_size.
        extra = dict(config["model_params"])

        return cls(
            df=frame,
            target_column=config["target_column"],
            group_column=config["group_column"],
            time_column=config["time_column"],
            feature_columns=config["feature_columns"],
            columns_to_drop=config["columns_to_drop"],
            model_type=config["model_name"],
            window_type=config["window_type"],
            window_size=config["window_size"],
            min_window_size=config["min_window_size"],
            max_window_size=config["max_window_size"],
            stride=config["stride"],
            prediction_horizon=config["prediction_horizon"],
            test_group_count=config["test_group_count"],
            validation_group_count=config["validation_group_count"],
            group_selection=config["group_selection"],
            random_state=config["random_state"],
            scaler=config["scaler"],
            learning_rate=config["learning_rate"],
            loss=config["loss"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            patience=config["patience"],
            **extra,
        )

    # -----------------------------------------------------------------
    # Saving and loading
    # -----------------------------------------------------------------

    def list_experiments(self) -> pd.DataFrame:
        frame = self.manager.list_experiments()
        return self._clean_dataframe(frame)

    def compare_experiments(
        self,
        experiment_names: list[str],
    ) -> pd.DataFrame:
        frame = self.manager.compare_experiments(
            experiments=experiment_names,
            sort_by="test_RMSE",
        )
        return self._clean_dataframe(frame)

    def load_saved(self, experiment_name: str) -> dict[str, Any]:
        loaded = self.manager.load_experiment(experiment_name)

        predictions = None
        for split in ("test", "validation", "train"):
            try:
                predictions = loaded.get_predictions(split)
                if predictions is not None and not predictions.empty:
                    break
            except Exception:
                continue

        return {
            "metrics": loaded.metrics,
            "predictions": predictions,
            "history": loaded.history,
            "loaded": loaded,
        }

    # -----------------------------------------------------------------
    # Result extraction
    # -----------------------------------------------------------------

    @staticmethod
    def _best_prediction_table(experiment: Any) -> pd.DataFrame | None:
        for split in ("test", "validation", "train"):
            try:
                result = experiment.get_prediction_results(dataset=split)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
            except Exception:
                continue
        return None

    @staticmethod
    def _history_dataframe(experiment: Any) -> pd.DataFrame | None:
        history = getattr(experiment, "history", None)
        if history is None:
            return None

        history_dict = getattr(history, "history", history)
        if isinstance(history_dict, dict):
            return pd.DataFrame(history_dict)

        if isinstance(history_dict, pd.DataFrame):
            return history_dict.copy()

        return None

    @staticmethod
    def select_primary_metrics(
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        for key in ("test", "external_test", "validation", "train"):
            value = metrics.get(key)
            if isinstance(value, dict):
                return value

        # Some train methods may return a flat metrics dictionary.
        if any(key in metrics for key in ("MAE", "RMSE", "R2")):
            return metrics

        return {}

    @classmethod
    def metrics_to_dataframe(
        cls,
        metrics: dict[str, Any],
    ) -> pd.DataFrame:
        rows = []

        for split, values in metrics.items():
            if not isinstance(values, dict):
                continue

            row = {"split": split}
            for key, value in values.items():
                if isinstance(value, (int, float, np.number)):
                    row[key] = float(value)
            if len(row) > 1:
                rows.append(row)

        if not rows and metrics:
            row = {"split": "test"}
            row.update(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.number))
                }
            )
            rows.append(row)

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Plotly figures
    # -----------------------------------------------------------------

    @staticmethod
    def prediction_figure(
        predictions: pd.DataFrame | None,
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(
                title="Actual vs predicted",
                annotations=[{"text": "No predictions available", "showarrow": False}],
            )

        actual_col = "actual" if "actual" in predictions else None
        predicted_col = (
            "predicted"
            if "predicted" in predictions
            else "predicted_RUL"
            if "predicted_RUL" in predictions
            else None
        )

        if not actual_col or not predicted_col:
            return go.Figure().update_layout(title="Actual vs predicted")

        frame = predictions[[actual_col, predicted_col]].dropna()
        fig = px.scatter(
            frame,
            x=actual_col,
            y=predicted_col,
            opacity=0.5,
            title="Actual vs predicted",
        )

        if not frame.empty:
            lower = float(min(frame.min()))
            upper = float(max(frame.max()))
            fig.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    name="Ideal",
                    line={"dash": "dash"},
                )
            )
        return fig

    @staticmethod
    def residual_figure(
        predictions: pd.DataFrame | None,
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(title="Residuals")

        predicted_col = (
            "predicted"
            if "predicted" in predictions
            else "predicted_RUL"
            if "predicted_RUL" in predictions
            else None
        )

        frame = predictions.copy()
        if "residual" not in frame and "actual" in frame and predicted_col:
            frame["residual"] = frame["actual"] - frame[predicted_col]

        if not predicted_col or "residual" not in frame:
            return go.Figure().update_layout(title="Residuals")

        fig = px.scatter(
            frame,
            x=predicted_col,
            y="residual",
            opacity=0.5,
            title="Residuals vs prediction",
        )
        fig.add_hline(y=0, line_dash="dash")
        return fig

    @staticmethod
    def error_distribution_figure(
        predictions: pd.DataFrame | None,
    ) -> go.Figure:
        if predictions is None or predictions.empty:
            return go.Figure().update_layout(title="Error distribution")

        frame = predictions.copy()
        predicted_col = (
            "predicted"
            if "predicted" in frame
            else "predicted_RUL"
            if "predicted_RUL" in frame
            else None
        )

        if "residual" not in frame and "actual" in frame and predicted_col:
            frame["residual"] = frame["actual"] - frame[predicted_col]

        if "residual" not in frame:
            return go.Figure().update_layout(title="Error distribution")

        return px.histogram(
            frame,
            x="residual",
            nbins=50,
            title="Residual distribution",
        )

    @staticmethod
    def history_figure(
        history: pd.DataFrame | None,
    ) -> go.Figure:
        if history is None or history.empty:
            return go.Figure().update_layout(
                title="Training history",
                annotations=[
                    {
                        "text": "Available for sequence models",
                        "showarrow": False,
                    }
                ],
            )

        frame = history.reset_index(names="epoch")
        fig = go.Figure()

        for column in ("loss", "val_loss", "mae", "val_mae"):
            if column in frame:
                fig.add_trace(
                    go.Scatter(
                        x=frame["epoch"],
                        y=frame[column],
                        mode="lines",
                        name=column,
                    )
                )

        fig.update_layout(
            title="Training history",
            xaxis_title="Epoch",
            yaxis_title="Metric",
        )
        return fig

    @staticmethod
    def comparison_figure(
        comparison: pd.DataFrame,
    ) -> go.Figure:
        if comparison is None or comparison.empty:
            return go.Figure().update_layout(title="Experiment comparison")

        metric = next(
            (
                column
                for column in ("test_RMSE", "test_MAE", "test_R2")
                if column in comparison.columns
            ),
            None,
        )

        if metric is None:
            return go.Figure().update_layout(title="Experiment comparison")

        frame = comparison.dropna(subset=[metric]).copy()
        return px.bar(
            frame,
            x="experiment_name",
            y=metric,
            color="model_type" if "model_type" in frame else None,
            title=f"Saved experiments by {metric}",
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def format_metric(value: Any) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()

        cleaned = frame.copy()
        for column in cleaned.columns:
            cleaned[column] = cleaned[column].map(
                lambda value: (
                    None
                    if isinstance(value, float) and np.isnan(value)
                    else value
                )
            )
        return cleaned

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): ExperimentService._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ExperimentService._json_safe(item) for item in value]
        return str(value)
