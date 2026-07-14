from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from pages import development
from pages import production


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title="SLB Experiment Lab",
)

server = app.server


HEADER_HEIGHT = "88px"
SIDEBAR_WIDTH = "300px"
SIDEBAR_COLLAPSED_WIDTH = "72px"

HEADER_STYLE = {
    "height": HEADER_HEIGHT,
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "zIndex": 1100,
    "backgroundColor": "#ffffff",
    "borderBottom": "1px solid #e5e7eb",
    "boxShadow": "0 1px 4px rgba(15, 23, 42, 0.06)",
}

SIDEBAR_BASE_STYLE = {
    "position": "fixed",
    "top": HEADER_HEIGHT,
    "left": 0,
    "bottom": 0,
    "backgroundColor": "#ffffff",
    "color": "#111827",
    "borderRight": "1px solid #e5e7eb",
    "overflowY": "auto",
    "overflowX": "hidden",
    "transition": "width 0.25s ease, padding 0.25s ease",
    "zIndex": 1000,
}

CONTENT_BASE_STYLE = {
    "marginTop": HEADER_HEIGHT,
    "minHeight": f"calc(100vh - {HEADER_HEIGHT})",
    "transition": "margin-left 0.25s ease",
    "backgroundColor": "#f6f8fb",
    "overflowX": "hidden",
}


header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Img(
                        src="/assets/logo.png",
                        alt="SLB logo",
                        style={
                            "height": "50px",
                            "width": "auto",
                            "objectFit": "contain",
                            "display": "block",
                        },
                    ),
                    dbc.NavbarBrand(
                        "SLB Predictive Maintenance",
                        href="/",
                        className="fw-semibold mb-0",
                        style={
                            "color": "#111827",
                            "fontSize": "1.35rem",
                        },
                    ),
                ],
                className="d-flex align-items-center gap-3",
            ),
        ],
        fluid=True,
        className="px-4",
    ),
    color="white",
    dark=False,
    style=HEADER_STYLE,
)


sidebar = html.Div(
    [
        html.Div(
            [
                html.Span(
                    "Navigation",
                    id="sidebar-title",
                    className="fw-semibold",
                    style={
                        "color": "#111827",
                    },
                ),
                dbc.Button(
                    html.I(
                        className="bi bi-chevron-left fs-4"
                    ),
                    id="sidebar-toggle",
                    color="link",
                    className="p-0 border-0 shadow-none",
                    style={
                        "color": "#374151",
                        "textDecoration": "none",
                    },
                    title="Collapse navigation",
                ),
            ],
            className=(
                "d-flex align-items-center "
                "justify-content-between mb-4"
            ),
        ),
        dbc.Nav(
            [
                dbc.NavLink(
    [
        html.I(
            className="bi bi-speedometer2 me-2",
            style={"color": "inherit"},
        ),
        html.Span(
            "Production monitoring",
            id={
                "type": "sidebar-label",
                "index": "production",
            },
            style={"color": "inherit"},
        ),
    ],
    href="/production",
    active="exact",
    className="sidebar-nav-link mb-2",
),

                dbc.NavLink(
    [
        html.I(
            className="bi bi-bar-chart-line me-2",
            style={"color": "inherit"},
        ),
        html.Span(
            "Model development",
            id={
                "type": "sidebar-label",
                "index": "development",
            },
            style={"color": "inherit"},
        ),
    ],
    href="/",
    active="exact",
    className="sidebar-nav-link mb-2",
),


            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style={
        **SIDEBAR_BASE_STYLE,
        "width": SIDEBAR_WIDTH,
        "padding": "1.5rem 1rem",
    },
)


content = html.Div(
    id="page-content",
    style={
        **CONTENT_BASE_STYLE,
        "marginLeft": SIDEBAR_WIDTH,
        "padding": 0,
    },
)


app.layout = html.Div(
    [
        dcc.Location(
            id="url",
            refresh=False,
        ),
        dcc.Store(
            id="sidebar-collapsed",
            data=False,
        ),
        header,
        sidebar,
        content,
    ],
    style={
        "minHeight": "100vh",
        "backgroundColor": "#f6f8fb",
    },
)


@callback(
    Output(
        "page-content",
        "children",
    ),
    Input(
        "url",
        "pathname",
    ),
)
def display_page(
    pathname: str | None,
):
    if pathname == "/production":
        return production.layout

    return development.layout


@callback(
    Output(
        "sidebar-collapsed",
        "data",
    ),
    Input(
        "sidebar-toggle",
        "n_clicks",
    ),
    State(
        "sidebar-collapsed",
        "data",
    ),
    prevent_initial_call=True,
)
def toggle_sidebar(
    _n_clicks: int,
    collapsed: bool,
) -> bool:
    return not bool(collapsed)


@callback(
    Output(
        "sidebar",
        "style",
    ),
    Output(
        "page-content",
        "style",
    ),
    Output(
        "sidebar-title",
        "style",
    ),
    Output(
        "sidebar-toggle",
        "children",
    ),
    Output(
        {
            "type": "sidebar-label",
            "index": dash.ALL,
        },
        "style",
    ),
    Input(
        "sidebar-collapsed",
        "data",
    ),
)
def update_sidebar_layout(
    collapsed: bool,
):
    sidebar_width = (
        SIDEBAR_COLLAPSED_WIDTH
        if collapsed
        else SIDEBAR_WIDTH
    )

    sidebar_style = {
        **SIDEBAR_BASE_STYLE,
        "width": sidebar_width,
        "padding": (
            "1.5rem 0.75rem"
            if collapsed
            else "1.5rem 1rem"
        ),
    }

    content_style = {
        **CONTENT_BASE_STYLE,
        "marginLeft": sidebar_width,
        "padding": 0,
    }

    title_style = {
        "display": (
            "none"
            if collapsed
            else "inline"
        ),
        "color": "#111827",
    }

    toggle_icon = html.I(
        className=(
            "bi bi-chevron-right fs-4"
            if collapsed
            else "bi bi-chevron-left fs-4"
        )
    )

    label_style = {
        "display": (
            "none"
            if collapsed
            else "inline"
        )
    }

    return (
        sidebar_style,
        content_style,
        title_style,
        toggle_icon,
        [
            label_style,
            label_style,
        ],
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8051,
        debug=True,
        use_reloader=False,
    )
