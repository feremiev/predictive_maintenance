from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
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

from .components import (
    card,
    metric_card,
    number_input,
)
from .context import SERVICE


# =====================================================================
# Production monitoring page
# =====================================================================

layout = html.Div(
    [
        dcc.Store(
            id="production-fleet-store",
        ),
        dcc.Store(
            id="production-selected-turbine",
        ),
        html.Div(
            [
                html.H2(
                    "Production turbine monitoring"
                ),
                html.P(
                    "Load a trained experiment, estimate the current "
                    "remaining useful life of each test turbine, and "
                    "simulate updated telemetry for an individual turbine.",
                    className="text-secondary",
                ),
            ],
            className="page-header",
        ),
        dbc.Row(
            [
                dbc.Col(
                    card(
                        "Production model",
                        [
                            dbc.Label(
                                "Experiment folder"
                            ),
                            dcc.Dropdown(
                                id=(
                                    "production-experiments-folder"
                                ),
                                options=[
                                    {
                                        "label": folder,
                                        "value": folder,
                                    }
                                    for folder
                                    in SERVICE.list_experiment_folders()
                                ],
                                value="experiments",
                                clearable=False,
                            ),
                            dbc.Label(
                                "Saved model",
                                className="mt-3",
                            ),
                            dcc.Dropdown(
                                id=(
                                    "production-experiment-name"
                                ),
                                placeholder=(
                                    "Select a trained experiment"
                                ),
                                clearable=False,
                            ),
                            dbc.Label(
                                "Raw data folder",
                                className="mt-3",
                            ),
                            dbc.Input(
                                id="production-data-folder",
                                value="raw_data",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Red at or below",
                                                className="mt-3",
                                            ),
                                            number_input(
                                                "production-red-threshold",
                                                25,
                                                minimum=0,
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Yellow at or below",
                                                className="mt-3",
                                            ),
                                            number_input(
                                                "production-yellow-threshold",
                                                60,
                                                minimum=1,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Button(
                                "Load turbine fleet",
                                id="load-production-fleet",
                                color="primary",
                                className="mt-4 w-100",
                            ),
                            html.Div(
                                id="production-status",
                                className="mt-3",
                            ),
                        ],
                    ),
                    lg=4,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    metric_card(
                                        "Critical turbines",
                                        "production-red-count",
                                        "Predicted RUL at or below the red threshold",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    metric_card(
                                        "Warning turbines",
                                        "production-yellow-count",
                                        "Predicted RUL between the red and yellow thresholds",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    metric_card(
                                        "Healthy turbines",
                                        "production-green-count",
                                        "Predicted RUL above the yellow threshold",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="g-3",
                        ),
                        card(
                            "Current turbine status",
                            html.Div(
                                [
                                    html.P(
                                        (
                                            "One row per turbine using the "
                                            "last observed test cycle."
                                        ),
                                        className="text-secondary small mb-2",
                                    ),
                                    dash_table.DataTable(
                                        id="production-fleet-table",
                                page_size=15,
                                sort_action="native",
                                filter_action="native",
                                row_selectable="single",
                                selected_rows=[],
                                style_table={
                                    "overflowX": "auto",
                                },
                                style_cell={
                                    "padding": "8px",
                                    "textAlign": "left",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {
                                            "filter_query": (
                                                '{status} = "Red"'
                                            ),
                                        },
                                        "backgroundColor": (
                                            "#f8d7da"
                                        ),
                                        "color": "#842029",
                                    },
                                    {
                                        "if": {
                                            "filter_query": (
                                                '{status} = "Yellow"'
                                            ),
                                        },
                                        "backgroundColor": (
                                            "#fff3cd"
                                        ),
                                        "color": "#664d03",
                                    },
                                    {
                                        "if": {
                                            "filter_query": (
                                                '{status} = "Green"'
                                            ),
                                        },
                                        "backgroundColor": (
                                            "#d1e7dd"
                                        ),
                                        "color": "#0f5132",
                                    },
                                ],
                                    ),
                                ]
                            ),
                            class_name="mt-3",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="g-3",
        ),
        html.Hr(
            className="my-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    card(
                        "Selected turbine",
                        [
                            html.H4(
                                "Select a turbine from the fleet table",
                                id="production-turbine-title",
                            ),
                            html.P(
                                "Edit the latest operating settings or sensor "
                                "values, then run a new prediction.",
                                className="text-secondary",
                            ),
                            dash_table.DataTable(
                                id="production-parameter-editor",
                                editable=True,
                                page_size=30,
                                style_table={
                                    "overflowX": "auto",
                                },
                                style_cell={
                                    "padding": "7px",
                                    "textAlign": "left",
                                },
                            ),
                            dbc.Button(
                                "Predict with updated values",
                                id="predict-production-turbine",
                                color="success",
                                className="mt-3",
                                disabled=True,
                            ),
                        ],
                    ),
                    lg=8,
                ),
                dbc.Col(
                    card(
                        "Updated prediction",
                        [
                            html.Div(
                                "—",
                                id=(
                                    "production-updated-rul"
                                ),
                                className="display-4 fw-bold",
                            ),
                            html.Div(
                                "Predicted remaining cycles",
                                className=(
                                    "text-secondary mb-3"
                                ),
                            ),
                            html.Div(
                                "—",
                                id=(
                                    "production-updated-status"
                                ),
                                className="h4",
                            ),
                            html.Div(
                                id=(
                                    "production-prediction-status"
                                ),
                                className="mt-3",
                            ),
                        ],
                    ),
                    lg=4,
                ),
            ],
            className="g-3",
        ),
    ],
    className="main-content",
)


# =====================================================================
# Production monitoring callbacks
# =====================================================================


@callback(
    Output(
        "production-experiments-folder",
        "options",
    ),
    Input(
        "url",
        "pathname",
    ),
    State(
        "production-experiments-folder",
        "value",
    ),
)
def refresh_production_experiment_folders(
    pathname,
    current_folder,
):
    folders = SERVICE.list_experiment_folders()

    options = [
        {
            "label": folder,
            "value": folder,
        }
        for folder in folders
    ]

    return options


@callback(
    Output(
        "production-experiment-name",
        "options",
    ),
    Output(
        "production-experiment-name",
        "value",
    ),
    Input(
        "production-experiments-folder",
        "value",
    ),
)
def refresh_production_models(
    experiments_folder,
):
    comparison = SERVICE.list_experiments(
        experiments_folder=(
            experiments_folder
            or "experiments"
        )
    )

    if comparison.empty:
        return [], None

    names = comparison[
        "experiment_name"
    ].dropna().astype(str).tolist()

    options = [
        {
            "label": name,
            "value": name,
        }
        for name in names
    ]

    return (
        options,
        names[0] if names else None,
    )


@callback(
    Output(
        "production-fleet-store",
        "data",
    ),
    Output(
        "production-fleet-table",
        "data",
    ),
    Output(
        "production-fleet-table",
        "columns",
    ),
    Output(
        "production-red-count",
        "children",
    ),
    Output(
        "production-yellow-count",
        "children",
    ),
    Output(
        "production-green-count",
        "children",
    ),
    Output(
        "production-status",
        "children",
    ),
    Input(
        "load-production-fleet",
        "n_clicks",
    ),
    State(
        "production-experiments-folder",
        "value",
    ),
    State(
        "production-experiment-name",
        "value",
    ),
    State(
        "production-data-folder",
        "value",
    ),
    State(
        "production-red-threshold",
        "value",
    ),
    State(
        "production-yellow-threshold",
        "value",
    ),
    prevent_initial_call=True,
    running=[
        (
            Output(
                "load-production-fleet",
                "disabled",
            ),
            True,
            False,
        ),
        (
            Output(
                "load-production-fleet",
                "children",
            ),
            "Loading fleet…",
            "Load turbine fleet",
        ),
    ],
)
def load_production_fleet(
    n_clicks,
    experiments_folder,
    experiment_name,
    data_folder,
    red_threshold,
    yellow_threshold,
):
    if not n_clicks:
        raise PreventUpdate

    if not experiment_name:
        return (
            no_update,
            [],
            [],
            "—",
            "—",
            "—",
            dbc.Alert(
                "Select a trained model.",
                color="warning",
            ),
        )

    try:
        outcome = (
            SERVICE.production_fleet_snapshot(
                experiments_folder=(
                    experiments_folder
                    or "experiments"
                ),
                experiment_name=experiment_name,
                data_folder=data_folder,
                red_threshold=float(
                    red_threshold or 25
                ),
                yellow_threshold=float(
                    yellow_threshold or 60
                ),
            )
        )

        fleet = outcome["fleet"].copy()

        # The production dashboard represents the latest known state of
        # each turbine. Official/actual RUL values belong to evaluation
        # and are intentionally not displayed here because they are not
        # available in a real production environment.
        visible_columns = [
            column
            for column in (
                "unique_motor_id",
                "cycle",
                "predicted_RUL",
                "status",
            )
            if column in fleet.columns
        ]

        fleet = fleet[
            visible_columns
        ].copy()

        if "predicted_RUL" in fleet.columns:
            fleet["predicted_RUL"] = (
                pd.to_numeric(
                    fleet["predicted_RUL"],
                    errors="coerce",
                )
                .round(2)
            )

        display_names = {
            "unique_motor_id": "Turbine",
            "cycle": "Last cycle",
            "predicted_RUL": "Predicted RUL",
            "status": "Status",
        }

        records = fleet.to_dict(
            "records"
        )

        columns = [
            {
                "name": display_names.get(
                    column,
                    column,
                ),
                "id": column,
            }
            for column in visible_columns
        ]

        return (
            {
                "records": records,
                "experiment_name": (
                    experiment_name
                ),
                "experiments_folder": (
                    experiments_folder
                    or "experiments"
                ),
            },
            records,
            columns,
            str(
                outcome["counts"]["Red"]
            ),
            str(
                outcome["counts"]["Yellow"]
            ),
            str(
                outcome["counts"]["Green"]
            ),
            dbc.Alert(
                (
                    f"Loaded {len(fleet)} turbines "
                    f"using '{experiment_name}'."
                ),
                color="success",
            ),
        )

    except Exception as exc:
        return (
            no_update,
            [],
            [],
            "—",
            "—",
            "—",
            dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            ),
        )


@callback(
    Output(
        "production-selected-turbine",
        "data",
    ),
    Output(
        "production-turbine-title",
        "children",
    ),
    Output(
        "production-parameter-editor",
        "data",
    ),
    Output(
        "production-parameter-editor",
        "columns",
    ),
    Output(
        "predict-production-turbine",
        "disabled",
    ),
    Input(
        "production-fleet-table",
        "selected_rows",
    ),
    State(
        "production-fleet-table",
        "derived_virtual_data",
    ),
    State(
        "production-fleet-table",
        "data",
    ),
    State(
        "production-experiments-folder",
        "value",
    ),
    State(
        "production-experiment-name",
        "value",
    ),
    State(
        "production-data-folder",
        "value",
    ),
    prevent_initial_call=True,
)
def select_production_turbine(
    selected_rows,
    visible_rows,
    all_rows,
    experiments_folder,
    experiment_name,
    data_folder,
):
    if not selected_rows:
        return (
            None,
            "Select a turbine from the fleet table",
            [],
            [],
            True,
        )

    rows = visible_rows or all_rows or []
    row_index = selected_rows[0]

    if row_index >= len(rows):
        raise PreventUpdate

    unique_motor_id = rows[
        row_index
    ].get("unique_motor_id")

    if not unique_motor_id:
        raise PreventUpdate

    try:
        editor = (
            SERVICE.production_turbine_editor(
                experiments_folder=(
                    experiments_folder
                    or "experiments"
                ),
                experiment_name=experiment_name,
                data_folder=data_folder,
                unique_motor_id=(
                    unique_motor_id
                ),
            )
        )

        editor_rows = [
            {
                "parameter": parameter,
                "value": value,
            }
            for parameter, value
            in editor["values"].items()
        ]

        return (
            editor,
            (
                f"Turbine {unique_motor_id} — "
                f"last cycle {editor['cycle']}"
            ),
            editor_rows,
            [
                {
                    "name": "Parameter",
                    "id": "parameter",
                    "editable": False,
                },
                {
                    "name": "Latest value",
                    "id": "value",
                    "type": "numeric",
                    "editable": True,
                },
            ],
            False,
        )

    except Exception as exc:
        return (
            None,
            f"Could not load {unique_motor_id}",
            [],
            [],
            True,
        )


@callback(
    Output(
        "production-updated-rul",
        "children",
    ),
    Output(
        "production-updated-status",
        "children",
    ),
    Output(
        "production-prediction-status",
        "children",
    ),
    Input(
        "predict-production-turbine",
        "n_clicks",
    ),
    State(
        "production-selected-turbine",
        "data",
    ),
    State(
        "production-parameter-editor",
        "data",
    ),
    State(
        "production-experiments-folder",
        "value",
    ),
    State(
        "production-experiment-name",
        "value",
    ),
    State(
        "production-data-folder",
        "value",
    ),
    State(
        "production-red-threshold",
        "value",
    ),
    State(
        "production-yellow-threshold",
        "value",
    ),
    prevent_initial_call=True,
    running=[
        (
            Output(
                "predict-production-turbine",
                "disabled",
            ),
            True,
            False,
        ),
        (
            Output(
                "predict-production-turbine",
                "children",
            ),
            "Predicting…",
            "Predict with updated values",
        ),
    ],
)
def predict_updated_turbine(
    n_clicks,
    selected_turbine,
    parameter_rows,
    experiments_folder,
    experiment_name,
    data_folder,
    red_threshold,
    yellow_threshold,
):
    if not n_clicks:
        raise PreventUpdate

    if not selected_turbine:
        return (
            "—",
            "—",
            dbc.Alert(
                "Select a turbine first.",
                color="warning",
            ),
        )

    updated_values = {
        row["parameter"]: row["value"]
        for row in (
            parameter_rows or []
        )
        if row.get("parameter")
        is not None
        and row.get("value")
        is not None
    }

    try:
        outcome = (
            SERVICE.predict_production_turbine(
                experiments_folder=(
                    experiments_folder
                    or "experiments"
                ),
                experiment_name=experiment_name,
                data_folder=data_folder,
                unique_motor_id=(
                    selected_turbine[
                        "unique_motor_id"
                    ]
                ),
                updated_values=updated_values,
                red_threshold=float(
                    red_threshold or 25
                ),
                yellow_threshold=float(
                    yellow_threshold or 60
                ),
            )
        )

        status = outcome["status"]

        color = {
            "Red": "danger",
            "Yellow": "warning",
            "Green": "success",
        }[status]

        return (
            f"{outcome['predicted_RUL']:.2f}",
            dbc.Badge(
                status,
                color=color,
                className="fs-5",
            ),
            dbc.Alert(
                (
                    "Prediction updated using the "
                    "edited latest telemetry values."
                ),
                color=color,
            ),
        )

    except Exception as exc:
        return (
            "—",
            "—",
            dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            ),
        )
