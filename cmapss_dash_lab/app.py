from __future__ import annotations

import json
import traceback
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Input,
    Output,
    State,
    callback,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from services import (
    AVAILABLE_SEQUENCE_MODELS,
    AVAILABLE_TABULAR_MODELS,
    ExperimentService,
    load_project_classes,
)


# =====================================================================
# Application setup
# =====================================================================

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


# =====================================================================
# Reusable UI helpers
# =====================================================================

def card(
    title: str,
    body: Any,
    class_name: str = "",
) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                title,
                className="fw-semibold",
            ),
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


def metric_card(
    title: str,
    component_id: str,
    description: str,
) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    title,
                    className="metric-label",
                ),
                html.Div(
                    "—",
                    id=component_id,
                    className="metric-value",
                ),
                html.Div(
                    description,
                    className="small text-secondary mt-1",
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


def empty_figure(title: str) -> go.Figure:
    return go.Figure().update_layout(
        title=title,
        annotations=[
            {
                "text": "No results available",
                "showarrow": False,
            }
        ],
    )


def parse_csv_names(
    value: str | None,
) -> list[str] | None:
    if not value or not value.strip():
        return None

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def safe_json(
    value: str | None,
) -> dict[str, Any]:
    if not value or not value.strip():
        return {}

    parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Model parameters must be a JSON object."
        )

    return parsed


def format_metric(value: Any) -> str:
    if value is None:
        return "—"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def split_metrics(
    metrics: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    values = metrics.get(split_name, {})

    return values if isinstance(values, dict) else {}


def metrics_dataframe(
    metrics: dict[str, Any],
) -> pd.DataFrame:
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
        values = metrics.get(split_name)

        if not isinstance(values, dict):
            continue

        row: dict[str, Any] = {
            "split": display_names[split_name],
        }

        for metric_name in (
            "NASA_SCORE",
            "MEAN_NASA_SCORE",
            "MAE",
            "RMSE",
            "R2",
            "MAPE",
            "Bias",
        ):
            if metric_name in values:
                row[metric_name] = values[metric_name]

        if "motor_count" in values:
            row["motor_count"] = values["motor_count"]

        if "evaluation_method" in values:
            row["evaluation_method"] = values[
                "evaluation_method"
            ]

        rows.append(row)

    return pd.DataFrame(rows)


def metric_comparison_figure(
    metrics: dict[str, Any],
    metric_name: str = "RMSE",
) -> go.Figure:
    frame = metrics_dataframe(metrics)

    if (
        frame.empty
        or metric_name not in frame.columns
    ):
        return empty_figure(
            f"{metric_name} by dataset split"
        )

    frame = frame.dropna(
        subset=[metric_name]
    )

    return px.bar(
        frame,
        x="split",
        y=metric_name,
        title=f"{metric_name}: train vs validation vs official test",
        text_auto=".3f",
    )


def prediction_figure(
    predictions: pd.DataFrame | None,
    title: str,
) -> go.Figure:
    if predictions is None or predictions.empty:
        return empty_figure(title)

    predicted_column = (
        "predicted"
        if "predicted" in predictions.columns
        else "predicted_RUL"
        if "predicted_RUL" in predictions.columns
        else None
    )

    if (
        "actual" not in predictions.columns
        or predicted_column is None
    ):
        return empty_figure(title)

    frame = predictions[
        ["actual", predicted_column]
    ].dropna()

    figure = px.scatter(
        frame,
        x="actual",
        y=predicted_column,
        opacity=0.5,
        title=title,
        labels={
            "actual": "Actual RUL",
            predicted_column: "Predicted RUL",
        },
    )

    if not frame.empty:
        lower = float(
            min(
                frame["actual"].min(),
                frame[predicted_column].min(),
            )
        )
        upper = float(
            max(
                frame["actual"].max(),
                frame[predicted_column].max(),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                name="Ideal prediction",
                line={"dash": "dash"},
            )
        )

    return figure


def residual_figure(
    predictions: pd.DataFrame | None,
    title: str,
) -> go.Figure:
    if predictions is None or predictions.empty:
        return empty_figure(title)

    frame = predictions.copy()

    predicted_column = (
        "predicted"
        if "predicted" in frame.columns
        else "predicted_RUL"
        if "predicted_RUL" in frame.columns
        else None
    )

    if predicted_column is None:
        return empty_figure(title)

    if (
        "residual" not in frame.columns
        and "actual" in frame.columns
    ):
        frame["residual"] = (
            frame["actual"]
            - frame[predicted_column]
        )

    if "residual" not in frame.columns:
        return empty_figure(title)

    figure = px.scatter(
        frame,
        x=predicted_column,
        y="residual",
        opacity=0.5,
        title=title,
        labels={
            predicted_column: "Predicted RUL",
            "residual": "Actual - predicted",
        },
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
    )

    return figure


def error_distribution_figure(
    predictions: pd.DataFrame | None,
    title: str,
) -> go.Figure:
    if predictions is None or predictions.empty:
        return empty_figure(title)

    frame = predictions.copy()

    predicted_column = (
        "predicted"
        if "predicted" in frame.columns
        else "predicted_RUL"
        if "predicted_RUL" in frame.columns
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
        return empty_figure(title)

    return px.histogram(
        frame,
        x="residual",
        nbins=50,
        title=title,
        labels={
            "residual": "Actual - predicted",
        },
    )


def history_figure(
    history: pd.DataFrame | None,
) -> go.Figure:
    if history is None or history.empty:
        return empty_figure(
            "Training history — available for sequence models"
        )

    frame = history.reset_index(
        names="epoch"
    )

    figure = go.Figure()

    if "val_nasa_score" in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame["epoch"],
                y=frame["val_nasa_score"],
                mode="lines",
                name="Validation NASA score",
                yaxis="y",
            )
        )

    for column in (
        "loss",
        "val_loss",
        "mae",
        "val_mae",
        "rmse",
        "val_rmse",
    ):
        if column in frame.columns:
            figure.add_trace(
                go.Scatter(
                    x=frame["epoch"],
                    y=frame[column],
                    mode="lines",
                    name=column,
                    yaxis="y2",
                )
            )

    if not figure.data:
        return empty_figure(
            "Training history — no supported metrics found"
        )

    figure.update_layout(
        title="Training history — NASA score and regression metrics",
        xaxis={
            "title": "Epoch",
        },
        yaxis={
            "title": "NASA score",
            "side": "left",
        },
        yaxis2={
            "title": "Loss / MAE / RMSE",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure

# =====================================================================
# Sidebar
# =====================================================================

sidebar = html.Div(
    [
        html.H3(
            "C-MAPSS Lab",
            className="mb-1",
        ),
        html.P(
            "Train, validate, externally test, save, and compare RUL models.",
            className="text-secondary small",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    "Experiment setup",
                    href="#setup",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Development results",
                    href="#development-results",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Official test",
                    href="#official-test",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Saved experiments",
                    href="#saved",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Data explorer",
                    href="#data",
                    external_link=True,
                ),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Alert(
            [
                html.Strong("Workflow"),
                html.Br(),
                "Train files are split into train and validation motors. "
                "The test and RUL files are used only in the official final test.",
            ],
            color="info",
            className="small",
        ),
    ],
    className="sidebar",
)


# =====================================================================
# Experiment configuration
# =====================================================================

data_config_card = card(
    "1. Training data",
    [
        dbc.Label("Raw data folder"),
        dbc.Input(
            id="data-folder",
            value="raw_data",
            placeholder=(
                "Folder containing train_FD001.txt, "
                "test_FD001.txt, and RUL_FD001.txt"
            ),
        ),
        dbc.Label(
            "Training files",
            className="mt-3",
        ),
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
                        dbc.Label(
                            "Remove nulls",
                            className="mt-3",
                        ),
                        dbc.Switch(
                            id="remove-nulls",
                            value=True,
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "Clip training RUL",
                            className="mt-3",
                        ),
                        dbc.Switch(
                            id="clip-rul",
                            value=False,
                        ),
                    ]
                ),
            ]
        ),
        dbc.Label(
            "Training RUL cap",
            className="mt-3",
        ),
        number_input(
            "rul-cap",
            125,
            minimum=1,
        ),
        dbc.Label(
            "Target column",
            className="mt-3",
        ),
        dbc.Input(
            id="target-column",
            value="RUL",
        ),
        dbc.Label(
            "Motor identifier",
            className="mt-3",
        ),
        dbc.Input(
            id="group-column",
            value="unique_motor_id",
        ),
        dbc.Label(
            "Cycle column",
            className="mt-3",
        ),
        dbc.Input(
            id="time-column",
            value="cycle",
        ),
    ],
)


model_config_card = card(
    "2. Model",
    [
        dbc.Label("Model family"),
        dcc.RadioItems(
            id="model-family",
            options=[
                {
                    "label": "Tabular / scikit-learn",
                    "value": "tabular",
                },
                {
                    "label": "Sequence / TensorFlow",
                    "value": "sequence",
                },
            ],
            value="tabular",
            labelStyle={
                "display": "block",
                "marginBottom": "0.4rem",
            },
        ),
        dbc.Label(
            "Model type",
            className="mt-3",
        ),
        dcc.Dropdown(
            id="model-name",
            options=[
                {
                    "label": value.replace(
                        "_",
                        " ",
                    ).title(),
                    "value": value,
                }
                for value in AVAILABLE_TABULAR_MODELS
            ],
            value="hist_gradient_boosting",
            clearable=False,
        ),
        dbc.Label(
            "Experiment name",
            className="mt-3",
        ),
        dbc.Input(
            id="experiment-name",
            placeholder=(
                "Leave blank to create a timestamped name"
            ),
        ),
        dbc.Label(
            "Columns to exclude",
            className="mt-3",
        ),
        dbc.Input(
            id="columns-to-drop",
            value="dataset,unit_number",
            placeholder="Comma-separated columns",
        ),
        dbc.Label(
            "Feature columns",
            className="mt-3",
        ),
        dbc.Textarea(
            id="feature-columns",
            placeholder=(
                "Optional comma-separated list. "
                "Leave blank to use numeric columns."
            ),
            rows=3,
        ),
        dbc.Label(
            "Model parameters (JSON)",
            className="mt-3",
        ),
        dbc.Textarea(
            id="model-params",
            value="{}",
            rows=5,
            className="font-monospace",
        ),
    ],
)


split_config_card = card(
    "3. Train and validation",
    [
        dbc.Alert(
            "Validation motors are selected only from the training files.",
            color="light",
            className="small",
        ),
        dbc.Label(
            "Validation motors",
        ),
        number_input(
            "validation-group-count",
            10,
            minimum=1,
        ),
        dbc.Label(
            "Motor selection",
            className="mt-3",
        ),
        dcc.Dropdown(
            id="group-selection",
            options=[
                {
                    "label": "Random motors",
                    "value": "random",
                },
                {
                    "label": "First motors",
                    "value": "first",
                },
                {
                    "label": "Last motors",
                    "value": "last",
                },
            ],
            value="random",
            clearable=False,
        ),
        dbc.Label(
            "Random seed",
            className="mt-3",
        ),
        number_input(
            "random-state",
            42,
            minimum=0,
        ),
        html.Div(
            [
                html.Hr(),
                html.H6("Sequence settings"),
                dbc.Label("Window type"),
                dcc.Dropdown(
                    id="window-type",
                    options=[
                        {
                            "label": "Sliding",
                            "value": "sliding",
                        },
                        {
                            "label": "Growing",
                            "value": "growing",
                        },
                    ],
                    value="sliding",
                    clearable=False,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Window size",
                                    className="mt-3",
                                ),
                                number_input(
                                    "window-size",
                                    30,
                                    minimum=2,
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Minimum window",
                                    className="mt-3",
                                ),
                                number_input(
                                    "min-window-size",
                                    10,
                                    minimum=2,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Maximum window",
                                    className="mt-3",
                                ),
                                number_input(
                                    "max-window-size",
                                    60,
                                    minimum=2,
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Stride",
                                    className="mt-3",
                                ),
                                number_input(
                                    "stride",
                                    1,
                                    minimum=1,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Label(
                    "Prediction horizon",
                    className="mt-3",
                ),
                number_input(
                    "prediction-horizon",
                    0,
                    minimum=0,
                ),
                dbc.Label(
                    "Scaler",
                    className="mt-3",
                ),
                dcc.Dropdown(
                    id="scaler",
                    options=[
                        {
                            "label": "Standard",
                            "value": "standard",
                        },
                        {
                            "label": "Min-Max",
                            "value": "minmax",
                        },
                        {
                            "label": "Robust",
                            "value": "robust",
                        },
                        {
                            "label": "None",
                            "value": "none",
                        },
                    ],
                    value="standard",
                    clearable=False,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Epochs",
                                    className="mt-3",
                                ),
                                number_input(
                                    "epochs",
                                    100,
                                    minimum=1,
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Batch size",
                                    className="mt-3",
                                ),
                                number_input(
                                    "batch-size",
                                    64,
                                    minimum=1,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Learning rate",
                                    className="mt-3",
                                ),
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
                                dbc.Label(
                                    "Patience",
                                    className="mt-3",
                                ),
                                number_input(
                                    "patience",
                                    12,
                                    minimum=1,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Label(
                    "Loss",
                    className="mt-3",
                ),
                dcc.Dropdown(
                    id="loss",
                    options=[
                        {
                            "label": "Huber",
                            "value": "huber",
                        },
                        {
                            "label": "Mean squared error",
                            "value": "mse",
                        },
                        {
                            "label": "Mean absolute error",
                            "value": "mae",
                        },
                    ],
                    value="huber",
                    clearable=False,
                ),
            ],
            id="sequence-settings",
            style={"display": "none"},
        ),
        dbc.Button(
            "Train experiment",
            id="run-experiment",
            color="primary",
            className="mt-4 w-100",
        ),
        dbc.Button(
            "Preview training data",
            id="preview-data",
            color="secondary",
            outline=True,
            className="mt-2 w-100",
        ),
    ],
)


# =====================================================================
# Development metrics
# =====================================================================

development_metric_cards = dbc.Row(
    [
        dbc.Col(
            metric_card(
                "Train NASA Score",
                "train-nasa-score",
                "NASA PHM score on the training motors (lower is better).",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Validation NASA Score",
                "validation-nasa-score",
                "Primary optimization metric on held-out validation motors.",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Train MAE",
                "train-mae",
                "Mean absolute error on the training motors.",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Validation MAE",
                "validation-mae",
                "Mean absolute error on the validation motors.",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Train RMSE",
                "train-rmse",
                "Training error with stronger penalty for large misses.",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Validation RMSE",
                "validation-rmse",
                "Validation RMSE for comparison with published results.",
            ),
            md=3,
        ),
    ],
    className="g-3 mb-3",
)

development_results_section = html.Div(
    [
        html.H3(
            "Development results",
            id="development-results",
        ),
        html.P(
            "These metrics use only train_FD00X files. "
            "Validation motors are unseen during fitting.",
            className="text-secondary",
        ),
        development_metric_cards,
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="development-metric-comparison",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="history-graph",
                    ),
                    lg=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="validation-prediction-graph",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="validation-residual-graph",
                    ),
                    lg=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="train-residual-graph",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="validation-error-distribution",
                    ),
                    lg=6,
                ),
            ]
        ),
        card(
            "Train and validation metrics",
            dash_table.DataTable(
                id="development-metrics-table",
                page_size=10,
                sort_action="native",
                style_table={
                    "overflowX": "auto",
                },
                style_cell={
                    "padding": "8px",
                    "textAlign": "left",
                },
            ),
        ),
        html.Div(
            id="run-status",
            className="mt-3",
        ),
    ],
    className="content-section",
)


# =====================================================================
# Official external test
# =====================================================================

external_test_configuration = card(
    "Official test configuration",
    [
        dbc.Alert(
            [
                html.Strong("Final evaluation only. "),
                "The model uses test_FD00X sensor histories and compares "
                "one final prediction per motor with RUL_FD00X.",
            ],
            color="warning",
            className="small",
        ),
        dbc.Label("Test datasets"),
        dcc.Dropdown(
            id="external-test-datasets",
            options=DATASET_OPTIONS,
            value=["FD001"],
            multi=True,
            clearable=False,
        ),
        dbc.Label(
            "External RUL treatment",
            className="mt-3",
        ),
        dcc.RadioItems(
            id="external-rul-mode",
            options=[
                {
                    "label": "Use the same clipping as training",
                    "value": "same",
                },
                {
                    "label": "Do not clip external RUL",
                    "value": "none",
                },
                {
                    "label": "Use a custom cap",
                    "value": "custom",
                },
            ],
            value="same",
            labelStyle={
                "display": "block",
                "marginBottom": "0.35rem",
            },
        ),
        dbc.Label(
            "Custom external RUL cap",
            className="mt-3",
        ),
        number_input(
            "external-rul-cap",
            125,
            minimum=1,
        ),
        dbc.Button(
            "Run official test",
            id="run-external-test",
            color="danger",
            className="mt-4 w-100",
        ),
    ],
)


external_test_metric_cards = dbc.Row(
    [
        dbc.Col(
            metric_card(
                "Official test NASA Score",
                "external-nasa-score",
                "Primary PHM Challenge evaluation metric (lower is better)",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Official test MAE",
                "external-mae",
                "Average final RUL error per external motor",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Official test RMSE",
                "external-rmse",
                "Final error with stronger penalty for large misses",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Official test R²",
                "external-r2",
                "Variance explained on the official test motors",
            ),
            md=3,
        ),
        dbc.Col(
            metric_card(
                "Official test Bias",
                "external-bias",
                "Positive means underprediction on average",
            ),
            md=3,
        ),
    ],
    className="g-3 mb-3",
)

official_test_section = html.Div(
    [
        html.H3(
            "Official final test",
            id="official-test",
        ),
        dbc.Row(
            [
                dbc.Col(
                    external_test_configuration,
                    lg=4,
                ),
                dbc.Col(
                    [
                        external_test_metric_cards,
                        dcc.Graph(
                            id="all-splits-metric-comparison",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="external-prediction-graph",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="external-residual-graph",
                    ),
                    lg=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="external-error-distribution",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="validation-vs-external-residuals",
                    ),
                    lg=6,
                ),
            ]
        ),
        card(
            "Official test metrics",
            dash_table.DataTable(
                id="external-metrics-table",
                page_size=10,
                sort_action="native",
                style_table={
                    "overflowX": "auto",
                },
                style_cell={
                    "padding": "8px",
                    "textAlign": "left",
                },
            ),
        ),
        html.Div(
            id="external-test-status",
            className="mt-3",
        ),
    ],
    className="content-section",
)


# =====================================================================
# Saved experiments
# =====================================================================

saved_section = html.Div(
    [
        html.H3(
            "Saved experiments",
            id="saved",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="saved-experiment-selector",
                        placeholder=(
                            "Select one or more experiments"
                        ),
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
        dcc.Graph(
            id="comparison-graph",
            className="mt-3",
        ),
        card(
            "Experiment comparison",
            dash_table.DataTable(
                id="comparison-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={
                    "overflowX": "auto",
                },
                style_cell={
                    "padding": "8px",
                    "textAlign": "left",
                },
            ),
        ),
        html.Div(
            id="load-status",
            className="mt-3",
        ),
    ],
    className="content-section",
)


# =====================================================================
# Data explorer
# =====================================================================

data_section = html.Div(
    [
        html.H3(
            "Data explorer",
            id="data",
        ),

        # ==========================================================
        # Training data
        # ==========================================================

        html.H4(
            "Training data",
            className="mt-4",
        ),

        html.P(
            "Training files contain complete motor histories until failure. "
            "RUL can therefore be calculated directly from each motor's "
            "maximum cycle.",
            className="text-secondary",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="rul-distribution",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="motor-length-graph",
                    ),
                    lg=6,
                ),
            ]
        ),

        card(
            "Training dataset preview",
            dash_table.DataTable(
                id="data-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={
                    "overflowX": "auto",
                },
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

        html.Div(
            id="preview-status",
            className="mt-3",
        ),

        # ==========================================================
        # Official test data
        # ==========================================================

        html.Hr(
            className="my-5",
        ),

        html.H4(
            "Official test data with derived RUL",
        ),

        html.P(
            "The official RUL file contains one value per motor at its "
            "last recorded test cycle. The dashboard derives the true RUL "
            "for every earlier test cycle.",
            className="text-secondary",
        ),

        dbc.Alert(
            [
                html.Strong("RUL calculation: "),
                html.Code(
                    "max_observed_cycle + official_final_RUL - current_cycle"
                ),
            ],
            color="info",
            className="small",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="test-rul-distribution",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="test-motor-length-graph",
                    ),
                    lg=6,
                ),
            ]
        ),

        card(
            "Test dataset preview with official RUL information",
            dash_table.DataTable(
                id="test-data-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={
                    "overflowX": "auto",
                },
                style_cell={
                    "padding": "7px",
                    "fontFamily": "monospace",
                    "fontSize": "0.82rem",
                    "minWidth": "85px",
                    "maxWidth": "180px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_data_conditional=[
                    {
                        "if": {
                            "column_id": "RUL",
                        },
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "column_id": "official_final_RUL",
                        },
                        "fontWeight": "bold",
                    },
                ],
            ),
        ),

        html.Div(
            id="test-preview-status",
            className="mt-3",
        ),
    ],
    className="content-section",
)

# =====================================================================
# Complete layout
# =====================================================================

app.layout = html.Div(
    [
        dcc.Store(
            id="latest-experiment-name",
        ),
        dcc.Store(
            id="latest-development-metrics",
        ),
        dcc.Store(
            id="latest-validation-predictions",
        ),
        sidebar,
        html.Main(
            [
                html.Div(
                    [
                        html.H2(
                            "Predictive Maintenance Experiment Dashboard"
                        ),
                        html.P(
                            "Develop with train and validation motors, then "
                            "run one separate official test using test + RUL files.",
                            className="text-secondary",
                        ),
                    ],
                    className="page-header",
                ),
                html.Div(
                    [
                        html.H3(
                            "Experiment setup",
                            id="setup",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    data_config_card,
                                    lg=4,
                                ),
                                dbc.Col(
                                    model_config_card,
                                    lg=4,
                                ),
                                dbc.Col(
                                    split_config_card,
                                    lg=4,
                                ),
                            ],
                            className="g-3",
                        ),
                    ],
                    className="content-section",
                ),
                html.Hr(),
                development_results_section,
                html.Hr(),
                official_test_section,
                html.Hr(),
                saved_section,
                html.Hr(),
                data_section,
            ],
            className="main-content",
        ),
    ]
)


# =====================================================================
# Configuration callbacks
# =====================================================================

@callback(
    Output(
        "model-name",
        "options",
    ),
    Output(
        "model-name",
        "value",
    ),
    Output(
        "sequence-settings",
        "style",
    ),
    Input(
        "model-family",
        "value",
    ),
)
def update_model_family(
    model_family: str,
):
    if model_family == "sequence":
        options = [
            {
                "label": value.replace(
                    "_",
                    " ",
                ).upper(),
                "value": value,
            }
            for value in AVAILABLE_SEQUENCE_MODELS
        ]

        return (
            options,
            "lstm",
            {"display": "block"},
        )

    options = [
        {
            "label": value.replace(
                "_",
                " ",
            ).title(),
            "value": value,
        }
        for value in AVAILABLE_TABULAR_MODELS
    ]

    return (
        options,
        "hist_gradient_boosting",
        {"display": "none"},
    )


@callback(
    Output(
        "rul-cap",
        "disabled",
    ),
    Input(
        "clip-rul",
        "value",
    ),
)
def toggle_rul_cap(
    clip_rul: bool,
):
    return not clip_rul


@callback(
    Output(
        "external-rul-cap",
        "disabled",
    ),
    Input(
        "external-rul-mode",
        "value",
    ),
)
def toggle_external_rul_cap(
    mode: str,
):
    return mode != "custom"


# =====================================================================
# Training configuration extraction
# =====================================================================

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
    "data_folder",
    "datasets",
    "remove_nulls",
    "clip_rul",
    "rul_cap",
    "target_column",
    "group_column",
    "time_column",
    "model_family",
    "model_name",
    "experiment_name",
    "columns_to_drop",
    "feature_columns",
    "model_params",
    "validation_group_count",
    "group_selection",
    "random_state",
    "window_type",
    "window_size",
    "min_window_size",
    "max_window_size",
    "stride",
    "prediction_horizon",
    "scaler",
    "epochs",
    "batch_size",
    "learning_rate",
    "patience",
    "loss",
]


def create_run_config(
    state_values: dict[str, Any],
) -> dict[str, Any]:
    model_family = state_values["model_family"]

    config = {
        "data_folder": state_values["data_folder"],
        "datasets": state_values["datasets"],
        "remove_nulls": bool(
            state_values["remove_nulls"]
        ),
        "clip_rul": bool(
            state_values["clip_rul"]
        ),
        "rul_cap": int(
            state_values["rul_cap"] or 125
        ),
        "target_column": state_values[
            "target_column"
        ],
        "group_column": state_values[
            "group_column"
        ],
        "time_column": state_values[
            "time_column"
        ],
        "model_family": model_family,
        "model_name": state_values[
            "model_name"
        ],
        "experiment_name": (
            state_values["experiment_name"]
            or None
        ),
        "columns_to_drop": (
            parse_csv_names(
                state_values["columns_to_drop"]
            )
            or []
        ),
        "feature_columns": parse_csv_names(
            state_values["feature_columns"]
        ),
        "model_params": safe_json(
            state_values["model_params"]
        ),
        "validation_group_count": int(
            state_values["validation_group_count"]
            or 1
        ),
        "group_selection": state_values[
            "group_selection"
        ],
        "random_state": int(
            state_values["random_state"]
            or 42
        ),
    }

    if model_family == "sequence":
        config.update(
            {
                "window_type": (
                    state_values["window_type"]
                    or "sliding"
                ),
                "window_size": int(
                    state_values["window_size"]
                    or 30
                ),
                "min_window_size": int(
                    state_values["min_window_size"]
                    or 10
                ),
                "max_window_size": int(
                    state_values["max_window_size"]
                    or 60
                ),
                "stride": int(
                    state_values["stride"]
                    or 1
                ),
                "prediction_horizon": int(
                    state_values["prediction_horizon"]
                    or 0
                ),
                "scaler": (
                    state_values["scaler"]
                    or "standard"
                ),
                "epochs": int(
                    state_values["epochs"]
                    or 100
                ),
                "batch_size": int(
                    state_values["batch_size"]
                    or 64
                ),
                "learning_rate": float(
                    state_values["learning_rate"]
                    if state_values["learning_rate"]
                    is not None
                    else 0.001
                ),
                "patience": int(
                    state_values["patience"]
                    or 12
                ),
                "loss": (
                    state_values["loss"]
                    or "huber"
                ),
            }
        )

    else:
        # These values are not used by tabular models, but keeping defaults
        # makes the configuration structure predictable.
        config.update(
            {
                "window_type": None,
                "window_size": None,
                "min_window_size": None,
                "max_window_size": None,
                "stride": None,
                "prediction_horizon": None,
                "scaler": None,
                "epochs": None,
                "batch_size": None,
                "learning_rate": None,
                "patience": None,
                "loss": None,
            }
        )

    return config

# =====================================================================
# Train callback
# =====================================================================

@callback(
    Output(
        "latest-experiment-name",
        "data",
    ),
    Output(
        "latest-development-metrics",
        "data",
    ),
    Output(
        "latest-validation-predictions",
        "data",
    ),
    Output(
        "run-status",
        "children",
    ),
    Output(
    "train-nasa-score",
    "children",
    ),
    Output(
        "validation-nasa-score",
        "children",
    ),
    Output(
        "train-mae",
        "children",
    ),
    Output(
        "validation-mae",
        "children",
    ),
    Output(
        "train-rmse",
        "children",
    ),
    Output(
        "validation-rmse",
        "children",
    ),
    Output(
        "development-metrics-table",
        "data",
    ),
    Output(
        "development-metrics-table",
        "columns",
    ),
    Output(
        "development-metric-comparison",
        "figure",
    ),
    Output(
        "history-graph",
        "figure",
    ),
    Output(
        "validation-prediction-graph",
        "figure",
    ),
    Output(
        "validation-residual-graph",
        "figure",
    ),
    Output(
        "train-residual-graph",
        "figure",
    ),
    Output(
        "validation-error-distribution",
        "figure",
    ),
    Input(
        "run-experiment",
        "n_clicks",
    ),
    *RUN_STATES,
    prevent_initial_call=True,
    running=[
        (
            Output(
                "run-experiment",
                "disabled",
            ),
            True,
            False,
        ),
        (
            Output(
                "run-experiment",
                "children",
            ),
            "Training…",
            "Train experiment",
        ),
    ],
)
def run_experiment(
    n_clicks: int,
    *values,
):
    if not n_clicks:
        raise PreventUpdate

    empty = empty_figure(
        "No results available"
    )

    if IMPORT_ERROR:
        alert = dbc.Alert(
            [
                html.Strong(
                    "Project classes could not be imported."
                ),
                html.Pre(
                    IMPORT_ERROR,
                    className="small mb-0",
                ),
            ],
            color="danger",
        )

        return (
            no_update,
            no_update,
            no_update,
            alert,
            "—",
            "—",
            "—",
            "—",
            [],
            [],
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    try:
        config = create_run_config(
            dict(
                zip(
                    RUN_KEYS,
                    values,
                )
            )
        )

        outcome = SERVICE.run_and_save(config)

        metrics = outcome["metrics"]
        train_predictions = outcome.get(
            "train_predictions"
        )
        validation_predictions = outcome.get(
            "validation_predictions"
        )
        history = outcome.get("history")
        experiment_name = outcome[
            "experiment_name"
        ]

        train_values = split_metrics(
            metrics,
            "train",
        )
        validation_values = split_metrics(
            metrics,
            "validation",
        )

        metric_frame = metrics_dataframe(
            metrics
        )

        status = dbc.Alert(
            (
                f"Experiment '{experiment_name}' trained and saved. "
                "The official test has not been run yet."
            ),
            color="success",
        )

        validation_records = (
            validation_predictions.to_dict(
                "records"
            )
            if isinstance(
                validation_predictions,
                pd.DataFrame,
            )
            else None
        )

        return (
            experiment_name,
            metrics,
            validation_records,
            status,
            format_metric(
                train_values.get("NASA_SCORE")
            ),
            format_metric(
                validation_values.get("NASA_SCORE")
            ),
            format_metric(
                train_values.get("MAE")
            ),
            format_metric(
                validation_values.get("MAE")
            ),
            format_metric(
                train_values.get("RMSE")
            ),
            format_metric(
                validation_values.get("RMSE")
            ),
            metric_frame.to_dict("records"),
            [
                {
                    "name": column,
                    "id": column,
                }
                for column in metric_frame.columns
            ],
            metric_comparison_figure(
                metrics,
                "NASA_SCORE",
            ),
            history_figure(history),
            prediction_figure(
                validation_predictions,
                "Validation: actual vs predicted RUL",
            ),
            residual_figure(
                validation_predictions,
                "Validation residuals",
            ),
            residual_figure(
                train_predictions,
                "Training residuals",
            ),
            error_distribution_figure(
                validation_predictions,
                "Validation error distribution",
            ),
        )

    except Exception as exc:
        trace = traceback.format_exc()

        alert = dbc.Alert(
            [
                html.Strong(
                    f"{type(exc).__name__}: {exc}"
                ),
                html.Details(
                    [
                        html.Summary(
                            "Show traceback"
                        ),
                        html.Pre(
                            trace,
                            className="traceback",
                        ),
                    ]
                ),
            ],
            color="danger",
        )

        return (
            no_update,
            no_update,
            no_update,
            alert,
            "—",  # Train NASA
            "—",  # Validation NASA
            "—",  # Train MAE
            "—",  # Validation MAE
            "—",  # Train RMSE
            "—",  # Validation RMSE
            [],
            [],
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )


# =====================================================================
# Official test callback
# =====================================================================

@callback(
    Output(
        "external-test-status",
        "children",
    ),
    Output(
        "external-mae",
        "children",
    ),
    Output("external-nasa-score", "children"),
    Output(
        "external-rmse",
        "children",
    ),
    Output(
        "external-r2",
        "children",
    ),
    Output(
        "external-bias",
        "children",
    ),
    Output(
        "external-metrics-table",
        "data",
    ),
    Output(
        "external-metrics-table",
        "columns",
    ),
    Output(
        "all-splits-metric-comparison",
        "figure",
    ),
    Output(
        "external-prediction-graph",
        "figure",
    ),
    Output(
        "external-residual-graph",
        "figure",
    ),
    Output(
        "external-error-distribution",
        "figure",
    ),
    Output(
        "validation-vs-external-residuals",
        "figure",
    ),
    Output(
        "latest-development-metrics",
        "data",
        allow_duplicate=True,
    ),
    Input(
        "run-external-test",
        "n_clicks",
    ),
    State(
        "latest-experiment-name",
        "data",
    ),
    State(
        "latest-development-metrics",
        "data",
    ),
    State(
        "latest-validation-predictions",
        "data",
    ),
    State(
        "data-folder",
        "value",
    ),
    State(
        "external-test-datasets",
        "value",
    ),
    State(
        "external-rul-mode",
        "value",
    ),
    State(
        "external-rul-cap",
        "value",
    ),
    State(
        "clip-rul",
        "value",
    ),
    State(
        "rul-cap",
        "value",
    ),
    prevent_initial_call=True,
    running=[
        (
            Output(
                "run-external-test",
                "disabled",
            ),
            True,
            False,
        ),
        (
            Output(
                "run-external-test",
                "children",
            ),
            "Testing…",
            "Run official test",
        ),
    ],
)
def run_external_test(
    n_clicks: int,
    experiment_name: str | None,
    development_metrics: dict[str, Any] | None,
    validation_prediction_records: list[dict[str, Any]] | None,
    data_folder: str,
    datasets: list[str],
    rul_mode: str,
    custom_cap: int,
    training_clip_rul: bool,
    training_rul_cap: int,
):
    if not n_clicks:
        raise PreventUpdate

    empty = empty_figure(
        "No official test results"
    )

    if not experiment_name:
        return (
            dbc.Alert(
                "Train or load an experiment before running the official test.",
                color="warning",
            ),
            "—",
            "—",
            "—",
            "—",
            [],
            [],
            empty,
            empty,
            empty,
            empty,
            empty,
            no_update,
        )

    try:
        if rul_mode == "same":
            clip_rul = bool(
                training_clip_rul
            )
            rul_cap = int(
                training_rul_cap
            )
        elif rul_mode == "custom":
            clip_rul = True
            rul_cap = int(custom_cap)
        else:
            clip_rul = False
            rul_cap = int(custom_cap)

        outcome = SERVICE.run_external_test(
            experiment_name=experiment_name,
            data_folder=data_folder,
            datasets=datasets,
            clip_rul=clip_rul,
            rul_cap=rul_cap,
        )

        external_metrics = outcome[
            "metrics"
        ]
        external_predictions = outcome[
            "predictions"
        ]

        all_metrics = dict(
            development_metrics or {}
        )
        all_metrics["external_test"] = (
            external_metrics
        )

        external_frame = pd.DataFrame(
            [
                {
                    "split": "Official test",
                    **{
                        key: value
                        for key, value
                        in external_metrics.items()
                        if key in {
                            "MAE",
                            "RMSE",
                            "R2",
                            "MAPE",
                            "Bias",
                            "motor_count",
                            "evaluation_method",
                        }
                    },
                }
            ]
        )

        validation_predictions = (
            pd.DataFrame(
                validation_prediction_records
            )
            if validation_prediction_records
            else None
        )

        combined_residual_figure = go.Figure()

        if (
            validation_predictions is not None
            and not validation_predictions.empty
        ):
            if (
                "residual"
                not in validation_predictions.columns
                and {
                    "actual",
                    "predicted",
                }.issubset(
                    validation_predictions.columns
                )
            ):
                validation_predictions[
                    "residual"
                ] = (
                    validation_predictions[
                        "actual"
                    ]
                    - validation_predictions[
                        "predicted"
                    ]
                )

            if {
                "predicted",
                "residual",
            }.issubset(
                validation_predictions.columns
            ):
                combined_residual_figure.add_trace(
                    go.Scatter(
                        x=validation_predictions[
                            "predicted"
                        ],
                        y=validation_predictions[
                            "residual"
                        ],
                        mode="markers",
                        name="Validation",
                        opacity=0.35,
                    )
                )

        if (
            isinstance(
                external_predictions,
                pd.DataFrame,
            )
            and not external_predictions.empty
        ):
            predicted_column = (
                "predicted"
                if "predicted"
                in external_predictions.columns
                else "predicted_RUL"
            )

            combined_residual_figure.add_trace(
                go.Scatter(
                    x=external_predictions[
                        predicted_column
                    ],
                    y=external_predictions[
                        "residual"
                    ],
                    mode="markers",
                    name="Official test",
                    opacity=0.7,
                )
            )

        combined_residual_figure.add_hline(
            y=0,
            line_dash="dash",
        )
        combined_residual_figure.update_layout(
            title=(
                "Validation vs official test residuals"
            ),
            xaxis_title="Predicted RUL",
            yaxis_title="Actual - predicted",
        )

        status = dbc.Alert(
            (
                f"Official test completed for '{experiment_name}' "
                f"using {', '.join(datasets)}. Results were added "
                "to the saved experiment."
            ),
            color="success",
        )

        return (
            status,
            format_metric(
                external_metrics.get("MAE")
            ),
            format_metric(
                external_metrics.get("RMSE")
            ),
            format_metric(
                external_metrics.get("R2")
            ),
            format_metric(
                external_metrics.get("Bias")
            ),
            format_metric(external_metrics.get("NASA_SCORE")),
            external_frame.to_dict(
                "records"
            ),
            [
                {
                    "name": column,
                    "id": column,
                }
                for column in external_frame.columns
            ],
            metric_comparison_figure(
                all_metrics,
                "RMSE",
            ),
            prediction_figure(
                external_predictions,
                "Official test: actual vs predicted RUL",
            ),
            residual_figure(
                external_predictions,
                "Official test residuals",
            ),
            error_distribution_figure(
                external_predictions,
                "Official test error distribution",
            ),
            combined_residual_figure,
            all_metrics,
        )

    except Exception as exc:
        trace = traceback.format_exc()

        return (
            dbc.Alert(
                [
                    html.Strong(
                        f"{type(exc).__name__}: {exc}"
                    ),
                    html.Details(
                        [
                            html.Summary(
                                "Show traceback"
                            ),
                            html.Pre(
                                trace,
                                className="traceback",
                            ),
                        ]
                    ),
                ],
                color="danger",
            ),
            "—",
            "—",
            "—",
            "—",
            [],
            [],
            empty,
            empty,
            empty,
            empty,
            empty,
            no_update,
        )


# =====================================================================
# Data preview callback
# =====================================================================

@callback(
    # Training data outputs
    Output(
        "data-table",
        "data",
    ),
    Output(
        "data-table",
        "columns",
    ),
    Output(
        "rul-distribution",
        "figure",
    ),
    Output(
        "motor-length-graph",
        "figure",
    ),
    Output(
        "preview-status",
        "children",
    ),

    # Test data outputs
    Output(
        "test-data-table",
        "data",
    ),
    Output(
        "test-data-table",
        "columns",
    ),
    Output(
        "test-rul-distribution",
        "figure",
    ),
    Output(
        "test-motor-length-graph",
        "figure",
    ),
    Output(
        "test-preview-status",
        "children",
    ),

    Input(
        "preview-data",
        "n_clicks",
    ),
    State(
        "data-folder",
        "value",
    ),
    State(
        "datasets",
        "value",
    ),
    State(
        "remove-nulls",
        "value",
    ),
    State(
        "clip-rul",
        "value",
    ),
    State(
        "rul-cap",
        "value",
    ),
    prevent_initial_call=True,
    running=[
        (
            Output(
                "preview-data",
                "disabled",
            ),
            True,
            False,
        ),
        (
            Output(
                "preview-data",
                "children",
            ),
            "Loading…",
            "Preview training data",
        ),
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

    empty_training_rul = empty_figure(
        "Training RUL distribution"
    )

    empty_training_lengths = empty_figure(
        "Training motor lengths"
    )

    empty_test_rul = empty_figure(
        "Test RUL distribution"
    )

    empty_test_lengths = empty_figure(
        "Test motor lengths"
    )

    if IMPORT_ERROR:
        error_alert = dbc.Alert(
            IMPORT_ERROR,
            color="danger",
        )

        return (
            [],
            [],
            empty_training_rul,
            empty_training_lengths,
            error_alert,
            [],
            [],
            empty_test_rul,
            empty_test_lengths,
            error_alert,
        )

    try:
        # ==========================================================
        # Load training data
        # ==========================================================

        training_frame = SERVICE.load_training_data(
            data_folder=data_folder,
            datasets=datasets,
            remove_nulls=remove_nulls,
            clip_rul=clip_rul,
            rul_cap=int(rul_cap),
        )

        training_preview = training_frame.head(500)

        training_columns = [
            {
                "name": column,
                "id": column,
            }
            for column in training_preview.columns
        ]

        training_rul_figure = px.histogram(
            training_frame,
            x="RUL",
            color="dataset",
            nbins=50,
            title="Training RUL distribution",
            marginal="box",
        )

        training_lengths = (
            training_frame.groupby(
                [
                    "dataset",
                    "unique_motor_id",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "cycles",
                }
            )
        )

        training_motor_figure = px.histogram(
            training_lengths,
            x="cycles",
            color="dataset",
            nbins=40,
            title="Training motor history lengths",
        )

        training_status = dbc.Alert(
            (
                f"Training: loaded {len(training_frame):,} rows and "
                f"{training_frame['unique_motor_id'].nunique():,} motors."
            ),
            color="success",
        )

        # ==========================================================
        # Load test data and attach RUL
        # ==========================================================

        test_frame = SERVICE.load_test_data_with_rul(
            data_folder=data_folder,
            datasets=datasets,
            remove_nulls=remove_nulls,
            clip_rul=clip_rul,
            rul_cap=int(rul_cap),
        )

        # Put the identification and RUL columns first so they are easier
        # to inspect in the horizontally scrollable table.
        preferred_columns = [
            "dataset",
            "unique_motor_id",
            "unit_number",
            "cycle",
            "max_observed_cycle",
            "official_final_RUL",
            "RUL",
        ]

        remaining_columns = [
            column
            for column in test_frame.columns
            if column not in preferred_columns
        ]

        test_frame = test_frame[
            preferred_columns + remaining_columns
        ]

        test_preview = test_frame.head(500)

        test_columns = [
            {
                "name": column,
                "id": column,
            }
            for column in test_preview.columns
        ]

        test_rul_figure = px.histogram(
            test_frame,
            x="RUL",
            color="dataset",
            nbins=50,
            title="Derived RUL distribution in official test histories",
            marginal="box",
        )

        test_lengths = (
            test_frame.groupby(
                [
                    "dataset",
                    "unique_motor_id",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "cycles",
                }
            )
        )

        test_motor_figure = px.histogram(
            test_lengths,
            x="cycles",
            color="dataset",
            nbins=40,
            title="Official test motor history lengths",
        )

        test_status = dbc.Alert(
            (
                f"Official test: loaded {len(test_frame):,} rows and "
                f"{test_frame['unique_motor_id'].nunique():,} motors. "
                "RUL was derived for every recorded cycle."
            ),
            color="success",
        )

        return (
            # Training
            training_preview.to_dict("records"),
            training_columns,
            training_rul_figure,
            training_motor_figure,
            training_status,

            # Test
            test_preview.to_dict("records"),
            test_columns,
            test_rul_figure,
            test_motor_figure,
            test_status,
        )

    except Exception as exc:
        trace = traceback.format_exc()

        error_alert = dbc.Alert(
            [
                html.Strong(
                    f"{type(exc).__name__}: {exc}"
                ),
                html.Details(
                    [
                        html.Summary(
                            "Show traceback"
                        ),
                        html.Pre(
                            trace,
                            className="traceback",
                        ),
                    ]
                ),
            ],
            color="danger",
        )

        return (
            [],
            [],
            empty_training_rul,
            empty_training_lengths,
            error_alert,
            [],
            [],
            empty_test_rul,
            empty_test_lengths,
            error_alert,
        )
# =====================================================================
# Saved experiments callbacks
# =====================================================================

@callback(
    Output(
        "saved-experiment-selector",
        "options",
    ),
    Output(
        "comparison-table",
        "data",
    ),
    Output(
        "comparison-table",
        "columns",
    ),
    Output(
        "comparison-graph",
        "figure",
    ),
    Input(
        "refresh-experiments",
        "n_clicks",
    ),
    Input(
        "latest-experiment-name",
        "data",
    ),
)
def refresh_experiments(
    _n_clicks,
    _latest,
):
    if IMPORT_ERROR:
        return (
            [],
            [],
            [],
            empty_figure(
                "Experiment comparison"
            ),
        )

    comparison = SERVICE.list_experiments()

    if comparison.empty:
        return (
            [],
            [],
            [],
            empty_figure(
                "Experiment comparison"
            ),
        )

    options = [
        {
            "label": name,
            "value": name,
        }
        for name in comparison[
            "experiment_name"
        ].tolist()
    ]

    metric_column = (
        "validation_RMSE"
        if "validation_RMSE"
        in comparison.columns
        else "external_test_RMSE"
        if "external_test_RMSE"
        in comparison.columns
        else None
    )

    if metric_column is None:
        figure = empty_figure(
            "Experiment comparison"
        )
    else:
        figure = px.bar(
            comparison.dropna(
                subset=[metric_column]
            ),
            x="experiment_name",
            y=metric_column,
            color=(
                "model_type"
                if "model_type"
                in comparison.columns
                else None
            ),
            title=(
                f"Saved experiments by {metric_column}"
            ),
        )

    return (
        options,
        comparison.to_dict("records"),
        [
            {
                "name": column,
                "id": column,
            }
            for column in comparison.columns
        ],
        figure,
    )


@callback(
    Output(
        "comparison-table",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "comparison-table",
        "columns",
        allow_duplicate=True,
    ),
    Output(
        "comparison-graph",
        "figure",
        allow_duplicate=True,
    ),
    Input(
        "saved-experiment-selector",
        "value",
    ),
    prevent_initial_call=True,
)
def compare_selected(
    experiment_names: list[str] | None,
):
    if not experiment_names:
        raise PreventUpdate

    comparison = SERVICE.compare_experiments(
        experiment_names
    )

    metric_column = (
        "validation_RMSE"
        if "validation_RMSE"
        in comparison.columns
        else "external_test_RMSE"
        if "external_test_RMSE"
        in comparison.columns
        else None
    )

    if metric_column is None:
        figure = empty_figure(
            "Experiment comparison"
        )
    else:
        figure = px.bar(
            comparison.dropna(
                subset=[metric_column]
            ),
            x="experiment_name",
            y=metric_column,
            color=(
                "model_type"
                if "model_type"
                in comparison.columns
                else None
            ),
            title=(
                f"Selected experiments by {metric_column}"
            ),
        )

    return (
        comparison.to_dict("records"),
        [
            {
                "name": column,
                "id": column,
            }
            for column in comparison.columns
        ],
        figure,
    )


@callback(
    Output(
        "load-status",
        "children",
    ),
    Output(
        "latest-experiment-name",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "latest-development-metrics",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "latest-validation-predictions",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "train-mae",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "validation-mae",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "train-rmse",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "validation-rmse",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "external-mae",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "external-rmse",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "external-r2",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "external-bias",
        "children",
        allow_duplicate=True,
    ),
    Output(
        "development-metrics-table",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "development-metrics-table",
        "columns",
        allow_duplicate=True,
    ),
    Output(
        "external-metrics-table",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "external-metrics-table",
        "columns",
        allow_duplicate=True,
    ),
    Output(
        "development-metric-comparison",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "all-splits-metric-comparison",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "history-graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "validation-prediction-graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "validation-residual-graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "external-prediction-graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "external-residual-graph",
        "figure",
        allow_duplicate=True,
    ),
    Input(
        "load-experiment",
        "n_clicks",
    ),
    State(
        "saved-experiment-selector",
        "value",
    ),
    prevent_initial_call=True,
)
def load_saved_experiment(
    n_clicks: int,
    selected: list[str] | None,
):
    if not n_clicks or not selected:
        raise PreventUpdate

    try:
        loaded = SERVICE.load_saved(
            selected[0]
        )

        metrics = loaded["metrics"]
        train_predictions = loaded.get(
            "train_predictions"
        )
        validation_predictions = loaded.get(
            "validation_predictions"
        )
        external_predictions = loaded.get(
            "external_test_predictions"
        )
        history = loaded.get("history")

        train_values = split_metrics(
            metrics,
            "train",
        )
        validation_values = split_metrics(
            metrics,
            "validation",
        )
        external_values = split_metrics(
            metrics,
            "external_test",
        )

        development_frame = metrics_dataframe(
            {
                "train": train_values,
                "validation": validation_values,
            }
        )

        external_frame = metrics_dataframe(
            {
                "external_test": external_values,
            }
        )

        validation_records = (
            validation_predictions.to_dict(
                "records"
            )
            if isinstance(
                validation_predictions,
                pd.DataFrame,
            )
            else None
        )

        return (
            dbc.Alert(
                f"Loaded '{selected[0]}'.",
                color="success",
            ),
            selected[0],
            metrics,
            validation_records,
            format_metric(
                train_values.get("MAE")
            ),
            format_metric(
                validation_values.get("MAE")
            ),
            format_metric(
                train_values.get("RMSE")
            ),
            format_metric(
                validation_values.get("RMSE")
            ),
            format_metric(
                external_values.get("MAE")
            ),
            format_metric(
                external_values.get("RMSE")
            ),
            format_metric(
                external_values.get("R2")
            ),
            format_metric(
                external_values.get("Bias")
            ),
            development_frame.to_dict(
                "records"
            ),
            [
                {
                    "name": column,
                    "id": column,
                }
                for column
                in development_frame.columns
            ],
            external_frame.to_dict(
                "records"
            ),
            [
                {
                    "name": column,
                    "id": column,
                }
                for column
                in external_frame.columns
            ],
            metric_comparison_figure(
                {
                    "train": train_values,
                    "validation": validation_values,
                },
                "RMSE",
            ),
            metric_comparison_figure(
                metrics,
                "RMSE",
            ),
            history_figure(history),
            prediction_figure(
                validation_predictions,
                "Validation: actual vs predicted RUL",
            ),
            residual_figure(
                validation_predictions,
                "Validation residuals",
            ),
            prediction_figure(
                external_predictions,
                "Official test: actual vs predicted RUL",
            ),
            residual_figure(
                external_predictions,
                "Official test residuals",
            ),
        )

    except Exception as exc:
        empty = empty_figure(
            "No results available"
        )

        return (
            dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            ),
            no_update,
            no_update,
            no_update,
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            [],
            [],
            [],
            [],
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8051,
        debug=True,
        use_reloader=False,
    )
