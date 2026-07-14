from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html


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
        className=(
            f"shadow-sm h-100 {class_name}"
        ).strip(),
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
                    className=(
                        "small text-secondary mt-1"
                    ),
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


def empty_figure(
    title: str,
) -> go.Figure:
    return go.Figure().update_layout(
        title=title,
        annotations=[
            {
                "text": "No results available",
                "showarrow": False,
            }
        ],
    )
