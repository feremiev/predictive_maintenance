
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from services import (
    AVAILABLE_SEQUENCE_MODELS,
    AVAILABLE_TABULAR_MODELS,
    ExperimentService,
    load_project_classes,
)

# ---------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="C-MAPSS Experiment Lab",
)
server = app.server

PROJECT_CLASSES, IMPORT_ERROR = load_project_classes()
SERVICE = ExperimentService(PROJECT_CLASSES)

DATASET_OPTIONS = [
    {"label": "FD001", "value": "FD001"},
    {"label": "FD002", "value": "FD002"},
    {"label": "FD003", "value": "FD003"},
    {"label": "FD004", "value": "FD004"},
]

METRIC_COLUMNS = ["MAE", "RMSE", "R2", "MAPE", "Bias"]


def card(title: str, body: Any, class_name: str = "") -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title, className="fw-semibold"),
            dbc.CardBody(body),
        ],
        className=f"shadow-sm h-100 {class_name}".strip(),
    )


def number_input(
    component_id: str,
    value: float | int,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    step: float | int = 1,
) -> dbc.Input:
    return dbc.Input(
        id=component_id,
        type="number",
        value=value,
        min=minimum,
        max=maximum,
        step=step,
    )


# ---------------------------------------------------------------------
# Layout sections
# ---------------------------------------------------------------------

sidebar = html.Div(
    [
        html.H3("C-MAPSS Lab", className="mb-1"),
        html.P(
            "Train, save, reload, evaluate, and compare RUL experiments.",
            className="text-secondary small",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Experiment setup", href="#setup", external_link=True),
                dbc.NavLink("Training results", href="#results", external_link=True),
                dbc.NavLink("Saved experiments", href="#saved", external_link=True),
                dbc.NavLink("Data explorer", href="#data", external_link=True),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Alert(
            "Training runs synchronously in this starter dashboard. "
            "For large neural networks, run the app behind a job queue.",
            color="info",
            className="small",
        ),
    ],
    className="sidebar",
)

data_config_card = card(
    "1. Data configuration",
    [
        dbc.Label("Raw data folder"),
        dbc.Input(
            id="data-folder",
            value="raw_data",
            placeholder="Path containing train_FD001.txt, test_FD001.txt, ...",
        ),
        dbc.Label("Training datasets", className="mt-3"),
        dcc.Dropdown(
            id="datasets",
            options=DATASET_OPTIONS,
            value=["FD001"],
            multi=True,
            clearable=False,
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Remove nulls", className="mt-3"),
                        dbc.Switch(id="remove-nulls", value=True),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Clip RUL", className="mt-3"),
                        dbc.Switch(id="clip-rul", value=False),
                    ]
                ),
            ]
        ),
        dbc.Label("RUL cap", className="mt-3"),
        number_input("rul-cap", 125, minimum=1, step=1),
        dbc.Label("Target", className="mt-3"),
        dbc.Input(id="target-column", value="RUL"),
        dbc.Label("Group identifier", className="mt-3"),
        dbc.Input(id="group-column", value="unique_motor_id"),
        dbc.Label("Time column", className="mt-3"),
        dbc.Input(id="time-column", value="cycle"),
    ],
)

model_config_card = card(
    "2. Model configuration",
    [
        dbc.Label("Model family"),
        dcc.RadioItems(
            id="model-family",
            options=[
                {"label": "Tabular / scikit-learn", "value": "tabular"},
                {"label": "Sequence / TensorFlow", "value": "sequence"},
            ],
            value="tabular",
            labelStyle={"display": "block", "marginBottom": "0.4rem"},
        ),
        dbc.Label("Model", className="mt-3"),
        dcc.Dropdown(
            id="model-name",
            options=[
                {"label": value.replace("_", " ").title(), "value": value}
                for value in AVAILABLE_TABULAR_MODELS
            ],
            value="hist_gradient_boosting",
            clearable=False,
        ),
        dbc.Label("Experiment name", className="mt-3"),
        dbc.Input(
            id="experiment-name",
            placeholder="Leave blank to generate a timestamped name",
        ),
        dbc.Label("Columns to drop", className="mt-3"),
        dbc.Input(
            id="columns-to-drop",
            value="dataset,unit_number",
            placeholder="Comma-separated names",
        ),
        dbc.Label("Feature columns", className="mt-3"),
        dbc.Textarea(
            id="feature-columns",
            placeholder=(
                "Optional comma-separated list. Leave blank to use the "
                "class defaults."
            ),
            rows=3,
        ),
        dbc.Label("Model parameters (JSON)", className="mt-3"),
        dbc.Textarea(
            id="model-params",
            value="{}",
            rows=5,
            className="font-monospace",
        ),
    ],
)

split_config_card = card(
    "3. Split and training",
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Test motors"),
                        number_input("test-group-count", 10, minimum=1),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Validation motors"),
                        number_input("validation-group-count", 10, minimum=1),
                    ],
                    id="validation-count-container",
                    style={"display": "none"},
                ),
            ]
        ),
        dbc.Label("Group selection", className="mt-3"),
        dcc.Dropdown(
            id="group-selection",
            options=[
                {"label": "Random", "value": "random"},
                {"label": "First groups", "value": "first"},
                {"label": "Last groups", "value": "last"},
            ],
            value="random",
            clearable=False,
        ),
        dbc.Label("Random seed", className="mt-3"),
        number_input("random-state", 42, minimum=0),
        html.Div(
            [
                html.Hr(),
                html.H6("Sequence settings"),
                dbc.Label("Window type"),
                dcc.Dropdown(
                    id="window-type",
                    options=[
                        {"label": "Sliding", "value": "sliding"},
                        {"label": "Growing", "value": "growing"},
                    ],
                    value="sliding",
                    clearable=False,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Window size", className="mt-3"),
                                number_input("window-size", 30, minimum=2),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Minimum window", className="mt-3"),
                                number_input("min-window-size", 10, minimum=2),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Maximum window", className="mt-3"),
                                number_input("max-window-size", 60, minimum=2),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Stride", className="mt-3"),
                                number_input("stride", 1, minimum=1),
                            ]
                        ),
                    ]
                ),
                dbc.Label("Prediction horizon", className="mt-3"),
                number_input("prediction-horizon", 0, minimum=0),
                dbc.Label("Scaler", className="mt-3"),
                dcc.Dropdown(
                    id="scaler",
                    options=[
                        {"label": "Standard", "value": "standard"},
                        {"label": "Min-Max", "value": "minmax"},
                        {"label": "Robust", "value": "robust"},
                        {"label": "None", "value": "none"},
                    ],
                    value="standard",
                    clearable=False,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Epochs", className="mt-3"),
                                number_input("epochs", 100, minimum=1),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Batch size", className="mt-3"),
                                number_input("batch-size", 64, minimum=1),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Learning rate", className="mt-3"),
                                number_input(
                                    "learning-rate",
                                    0.001,
                                    minimum=0.000001,
                                    step=0.0001,
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Patience", className="mt-3"),
                                number_input("patience", 12, minimum=1),
                            ]
                        ),
                    ]
                ),
                dbc.Label("Loss", className="mt-3"),
                dcc.Dropdown(
                    id="loss",
                    options=[
                        {"label": "Huber", "value": "huber"},
                        {"label": "Mean squared error", "value": "mse"},
                        {"label": "Mean absolute error", "value": "mae"},
                    ],
                    value="huber",
                    clearable=False,
                ),
            ],
            id="sequence-settings",
            style={"display": "none"},
        ),
        dbc.Button(
            "Run experiment",
            id="run-experiment",
            color="primary",
            className="mt-4 w-100",
        ),
        dbc.Button(
            "Preview data",
            id="preview-data",
            color="secondary",
            outline=True,
            className="mt-2 w-100",
        ),
    ],
)

metrics_cards = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [html.Div(metric, className="metric-label"),
                     html.Div("—", id=f"metric-{metric.lower()}", className="metric-value")]
                ),
                className="shadow-sm",
            ),
            md=True,
        )
        for metric in ["MAE", "RMSE", "R2", "Bias"]
    ],
    className="g-3 mb-3",
)

results_section = html.Div(
    [
        html.H3("Training results", id="results"),
        metrics_cards,
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="prediction-graph"), lg=6),
                dbc.Col(dcc.Graph(id="residual-graph"), lg=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="error-distribution"), lg=6),
                dbc.Col(dcc.Graph(id="history-graph"), lg=6),
            ]
        ),
        card(
            "Metrics by split",
            dash_table.DataTable(
                id="metrics-table",
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),
        ),
        html.Div(id="run-status", className="mt-3"),
    ],
    className="content-section",
)

saved_section = html.Div(
    [
        html.H3("Saved experiments", id="saved"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="saved-experiment-selector",
                        placeholder="Select one or more experiments",
                        multi=True,
                    ),
                    lg=8,
                ),
                dbc.Col(
                    dbc.Button(
                        "Refresh",
                        id="refresh-experiments",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    lg=2,
                ),
                dbc.Col(
                    dbc.Button(
                        "Load selected",
                        id="load-experiment",
                        color="primary",
                        className="w-100",
                    ),
                    lg=2,
                ),
            ],
            className="g-2",
        ),
        dcc.Graph(id="comparison-graph", className="mt-3"),
        card(
            "Experiment comparison",
            dash_table.DataTable(
                id="comparison-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),
        ),
        html.Div(id="load-status", className="mt-3"),
    ],
    className="content-section",
)

data_section = html.Div(
    [
        html.H3("Data explorer", id="data"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="rul-distribution"), lg=6),
                dbc.Col(dcc.Graph(id="motor-length-graph"), lg=6),
            ]
        ),
        card(
            "Dataset preview",
            dash_table.DataTable(
                id="data-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "padding": "7px",
                    "fontFamily": "monospace",
                    "fontSize": "0.82rem",
                    "minWidth": "85px",
                    "maxWidth": "180px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
            ),
        ),
        html.Div(id="preview-status", className="mt-3"),
    ],
    className="content-section",
)

app.layout = html.Div(
    [
        dcc.Store(id="latest-experiment-name"),
        dcc.Store(id="latest-results-store"),
        sidebar,
        html.Main(
            [
                html.Div(
                    [
                        html.H2("Predictive Maintenance Experiment Dashboard"),
                        html.P(
                            "Configure C-MAPSS data, train tabular or sequence "
                            "models, persist experiments, and inspect diagnostics.",
                            className="text-secondary",
                        ),
                    ],
                    className="page-header",
                ),
                html.Div(
                    [
                        html.H3("Experiment setup", id="setup"),
                        dbc.Row(
                            [
                                dbc.Col(data_config_card, lg=4),
                                dbc.Col(model_config_card, lg=4),
                                dbc.Col(split_config_card, lg=4),
                            ],
                            className="g-3",
                        ),
                    ],
                    className="content-section",
                ),
                html.Hr(),
                results_section,
                html.Hr(),
                saved_section,
                html.Hr(),
                data_section,
            ],
            className="main-content",
        ),
    ]
)


# ---------------------------------------------------------------------
# UI utility callbacks
# ---------------------------------------------------------------------

@callback(
    Output("model-name", "options"),
    Output("model-name", "value"),
    Output("sequence-settings", "style"),
    Output("validation-count-container", "style"),
    Input("model-family", "value"),
)
def update_model_family(model_family: str):
    if model_family == "sequence":
        options = [
            {"label": value.replace("_", " ").upper(), "value": value}
            for value in AVAILABLE_SEQUENCE_MODELS
        ]
        return options, "lstm", {"display": "block"}, {"display": "block"}

    options = [
        {"label": value.replace("_", " ").title(), "value": value}
        for value in AVAILABLE_TABULAR_MODELS
    ]
    return (
        options,
        "hist_gradient_boosting",
        {"display": "none"},
        {"display": "none"},
    )


@callback(
    Output("rul-cap", "disabled"),
    Input("clip-rul", "value"),
)
def toggle_rul_cap(clip_rul: bool):
    return not clip_rul


def parse_csv_names(value: str | None) -> list[str] | None:
    if not value or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_json(value: str | None) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Model parameters must be a JSON object.")
    return parsed


def create_run_config(state_values: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_folder": state_values["data_folder"],
        "datasets": state_values["datasets"],
        "remove_nulls": state_values["remove_nulls"],
        "clip_rul": state_values["clip_rul"],
        "rul_cap": int(state_values["rul_cap"]),
        "target_column": state_values["target_column"],
        "group_column": state_values["group_column"],
        "time_column": state_values["time_column"],
        "model_family": state_values["model_family"],
        "model_name": state_values["model_name"],
        "experiment_name": state_values["experiment_name"] or None,
        "columns_to_drop": parse_csv_names(state_values["columns_to_drop"]) or [],
        "feature_columns": parse_csv_names(state_values["feature_columns"]),
        "model_params": safe_json(state_values["model_params"]),
        "test_group_count": int(state_values["test_group_count"]),
        "validation_group_count": int(state_values["validation_group_count"]),
        "group_selection": state_values["group_selection"],
        "random_state": int(state_values["random_state"]),
        "window_type": state_values["window_type"],
        "window_size": int(state_values["window_size"]),
        "min_window_size": int(state_values["min_window_size"]),
        "max_window_size": int(state_values["max_window_size"]),
        "stride": int(state_values["stride"]),
        "prediction_horizon": int(state_values["prediction_horizon"]),
        "scaler": state_values["scaler"],
        "epochs": int(state_values["epochs"]),
        "batch_size": int(state_values["batch_size"]),
        "learning_rate": float(state_values["learning_rate"]),
        "patience": int(state_values["patience"]),
        "loss": state_values["loss"],
    }


RUN_STATES = [
    State("data-folder", "value"),
    State("datasets", "value"),
    State("remove-nulls", "value"),
    State("clip-rul", "value"),
    State("rul-cap", "value"),
    State("target-column", "value"),
    State("group-column", "value"),
    State("time-column", "value"),
    State("model-family", "value"),
    State("model-name", "value"),
    State("experiment-name", "value"),
    State("columns-to-drop", "value"),
    State("feature-columns", "value"),
    State("model-params", "value"),
    State("test-group-count", "value"),
    State("validation-group-count", "value"),
    State("group-selection", "value"),
    State("random-state", "value"),
    State("window-type", "value"),
    State("window-size", "value"),
    State("min-window-size", "value"),
    State("max-window-size", "value"),
    State("stride", "value"),
    State("prediction-horizon", "value"),
    State("scaler", "value"),
    State("epochs", "value"),
    State("batch-size", "value"),
    State("learning-rate", "value"),
    State("patience", "value"),
    State("loss", "value"),
]

RUN_KEYS = [
    "data_folder", "datasets", "remove_nulls", "clip_rul", "rul_cap",
    "target_column", "group_column", "time_column", "model_family",
    "model_name", "experiment_name", "columns_to_drop", "feature_columns",
    "model_params", "test_group_count", "validation_group_count",
    "group_selection", "random_state", "window_type", "window_size",
    "min_window_size", "max_window_size", "stride", "prediction_horizon",
    "scaler", "epochs", "batch_size", "learning_rate", "patience", "loss",
]


# ---------------------------------------------------------------------
# Main training callback
# ---------------------------------------------------------------------

@callback(
    Output("latest-experiment-name", "data"),
    Output("latest-results-store", "data"),
    Output("run-status", "children"),
    Output("metric-mae", "children"),
    Output("metric-rmse", "children"),
    Output("metric-r2", "children"),
    Output("metric-bias", "children"),
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Output("prediction-graph", "figure"),
    Output("residual-graph", "figure"),
    Output("error-distribution", "figure"),
    Output("history-graph", "figure"),
    Input("run-experiment", "n_clicks"),
    *RUN_STATES,
    prevent_initial_call=True,
    running=[
        (Output("run-experiment", "disabled"), True, False),
        (Output("run-experiment", "children"), "Training…", "Run experiment"),
    ],
)
def run_experiment(n_clicks: int, *values):
    if not n_clicks:
        raise PreventUpdate

    if IMPORT_ERROR:
        alert = dbc.Alert(
            [
                html.Strong("Project classes could not be imported. "),
                html.Pre(IMPORT_ERROR, className="small mb-0"),
            ],
            color="danger",
        )
        empty = go.Figure()
        return (
            no_update, no_update, alert, "—", "—", "—", "—",
            [], [], empty, empty, empty, empty,
        )

    try:
        config = create_run_config(dict(zip(RUN_KEYS, values)))
        outcome = SERVICE.run_and_save(config)

        metrics = outcome["metrics"]
        predictions = outcome["predictions"]
        history = outcome.get("history")
        experiment_name = outcome["experiment_name"]

        metric_df = SERVICE.metrics_to_dataframe(metrics)
        selected_metrics = SERVICE.select_primary_metrics(metrics)

        prediction_fig = SERVICE.prediction_figure(predictions)
        residual_fig = SERVICE.residual_figure(predictions)
        distribution_fig = SERVICE.error_distribution_figure(predictions)
        history_fig = SERVICE.history_figure(history)

        serialized = {
            "experiment_name": experiment_name,
            "metrics": metrics,
        }

        status = dbc.Alert(
            f"Experiment '{experiment_name}' trained and saved successfully.",
            color="success",
        )

        return (
            experiment_name,
            serialized,
            status,
            SERVICE.format_metric(selected_metrics.get("MAE")),
            SERVICE.format_metric(selected_metrics.get("RMSE")),
            SERVICE.format_metric(selected_metrics.get("R2")),
            SERVICE.format_metric(selected_metrics.get("Bias")),
            metric_df.to_dict("records"),
            [{"name": col, "id": col} for col in metric_df.columns],
            prediction_fig,
            residual_fig,
            distribution_fig,
            history_fig,
        )

    except Exception as exc:
        trace = traceback.format_exc()
        alert = dbc.Alert(
            [
                html.Strong(f"{type(exc).__name__}: {exc}"),
                html.Details(
                    [
                        html.Summary("Show traceback"),
                        html.Pre(trace, className="traceback"),
                    ]
                ),
            ],
            color="danger",
        )
        empty = go.Figure()
        return (
            no_update, no_update, alert, "—", "—", "—", "—",
            [], [], empty, empty, empty, empty,
        )


# ---------------------------------------------------------------------
# Data preview callback
# ---------------------------------------------------------------------

@callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Output("rul-distribution", "figure"),
    Output("motor-length-graph", "figure"),
    Output("preview-status", "children"),
    Input("preview-data", "n_clicks"),
    State("data-folder", "value"),
    State("datasets", "value"),
    State("remove-nulls", "value"),
    State("clip-rul", "value"),
    State("rul-cap", "value"),
    prevent_initial_call=True,
    running=[
        (Output("preview-data", "disabled"), True, False),
        (Output("preview-data", "children"), "Loading…", "Preview data"),
    ],
)
def preview_data(
    n_clicks: int,
    data_folder: str,
    datasets: list[str],
    remove_nulls: bool,
    clip_rul: bool,
    rul_cap: int,
):
    if not n_clicks:
        raise PreventUpdate

    if IMPORT_ERROR:
        return [], [], go.Figure(), go.Figure(), dbc.Alert(
            IMPORT_ERROR, color="danger"
        )

    try:
        frame = SERVICE.load_training_data(
            data_folder=data_folder,
            datasets=datasets,
            remove_nulls=remove_nulls,
            clip_rul=clip_rul,
            rul_cap=int(rul_cap),
        )

        preview = frame.head(500)
        columns = [{"name": col, "id": col} for col in preview.columns]

        rul_fig = px.histogram(
            frame,
            x="RUL",
            nbins=50,
            title="RUL distribution",
        )

        lengths = (
            frame.groupby("unique_motor_id")
            .size()
            .rename("cycles")
            .reset_index()
            .sort_values("cycles", ascending=False)
        )
        motor_fig = px.histogram(
            lengths,
            x="cycles",
            nbins=40,
            title="Distribution of motor history lengths",
        )

        status = dbc.Alert(
            f"Loaded {len(frame):,} rows and "
            f"{frame['unique_motor_id'].nunique():,} motors.",
            color="success",
        )
        return (
            preview.to_dict("records"),
            columns,
            rul_fig,
            motor_fig,
            status,
        )
    except Exception as exc:
        return [], [], go.Figure(), go.Figure(), dbc.Alert(
            f"{type(exc).__name__}: {exc}", color="danger"
        )


# ---------------------------------------------------------------------
# Saved experiment callbacks
# ---------------------------------------------------------------------

@callback(
    Output("saved-experiment-selector", "options"),
    Output("comparison-table", "data"),
    Output("comparison-table", "columns"),
    Output("comparison-graph", "figure"),
    Input("refresh-experiments", "n_clicks"),
    Input("latest-experiment-name", "data"),
)
def refresh_experiments(_n_clicks, _latest):
    if IMPORT_ERROR:
        return [], [], [], go.Figure()

    comparison = SERVICE.list_experiments()
    if comparison.empty:
        return [], [], [], go.Figure()

    options = [
        {"label": name, "value": name}
        for name in comparison["experiment_name"].tolist()
    ]

    figure = SERVICE.comparison_figure(comparison)
    return (
        options,
        comparison.to_dict("records"),
        [{"name": col, "id": col} for col in comparison.columns],
        figure,
    )


@callback(
    Output("comparison-table", "data", allow_duplicate=True),
    Output("comparison-table", "columns", allow_duplicate=True),
    Output("comparison-graph", "figure", allow_duplicate=True),
    Input("saved-experiment-selector", "value"),
    prevent_initial_call=True,
)
def compare_selected(experiment_names: list[str] | None):
    if not experiment_names:
        raise PreventUpdate

    comparison = SERVICE.compare_experiments(experiment_names)
    return (
        comparison.to_dict("records"),
        [{"name": col, "id": col} for col in comparison.columns],
        SERVICE.comparison_figure(comparison),
    )


@callback(
    Output("load-status", "children"),
    Output("metric-mae", "children", allow_duplicate=True),
    Output("metric-rmse", "children", allow_duplicate=True),
    Output("metric-r2", "children", allow_duplicate=True),
    Output("metric-bias", "children", allow_duplicate=True),
    Output("metrics-table", "data", allow_duplicate=True),
    Output("metrics-table", "columns", allow_duplicate=True),
    Output("prediction-graph", "figure", allow_duplicate=True),
    Output("residual-graph", "figure", allow_duplicate=True),
    Output("error-distribution", "figure", allow_duplicate=True),
    Output("history-graph", "figure", allow_duplicate=True),
    Input("load-experiment", "n_clicks"),
    State("saved-experiment-selector", "value"),
    prevent_initial_call=True,
)
def load_saved_experiment(n_clicks: int, selected: list[str] | None):
    if not n_clicks or not selected:
        raise PreventUpdate

    # Load the first selected experiment into the result views.
    try:
        loaded = SERVICE.load_saved(selected[0])
        metrics = loaded["metrics"]
        predictions = loaded["predictions"]
        history = loaded.get("history")

        selected_metrics = SERVICE.select_primary_metrics(metrics)
        metric_df = SERVICE.metrics_to_dataframe(metrics)

        return (
            dbc.Alert(f"Loaded '{selected[0]}'.", color="success"),
            SERVICE.format_metric(selected_metrics.get("MAE")),
            SERVICE.format_metric(selected_metrics.get("RMSE")),
            SERVICE.format_metric(selected_metrics.get("R2")),
            SERVICE.format_metric(selected_metrics.get("Bias")),
            metric_df.to_dict("records"),
            [{"name": col, "id": col} for col in metric_df.columns],
            SERVICE.prediction_figure(predictions),
            SERVICE.residual_figure(predictions),
            SERVICE.error_distribution_figure(predictions),
            SERVICE.history_figure(history),
        )
    except Exception as exc:
        empty = go.Figure()
        return (
            dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
            "—", "—", "—", "—", [], [], empty, empty, empty, empty,
        )


if __name__ == "__main__":
    app.run(debug=True)
