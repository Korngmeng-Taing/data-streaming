import json
import os
import signal
import sys

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dotenv import load_dotenv

load_dotenv()

from dash_app.callbacks import *  # noqa: E402, F403
from dash_app.data_utils import save_session_csv  # noqa: E402


def _shutdown_handler(sig, frame):
    save_session_csv()
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Crypto Pipeline",
    update_title=None,
)
server = app.server

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "260px",
    "padding": "20px 16px",
    "background-color": "#1a1a2e",
    "border-right": "1px solid #333",
    "overflow-y": "auto",
    "z-index": 1000,
}
CONTENT_STYLE = {
    "margin-left": "260px",
    "padding": "24px 32px",
}

PAGES = {
    "overview": "Overview",
    "technical": "Technical Analysis",
    "comparison": "Comparison",
    "predictions": "Price Prediction",
    "data": "Data Explorer",
    "pipeline": "Pipeline Status",
    "alerts": "Alerts",
}

NAV_ICONS = {
    "overview": "bi-graph-up",
    "technical": "bi-bar-chart",
    "comparison": "bi-arrow-left-right",
    "predictions": "bi-cpu",
    "data": "bi-table",
    "pipeline": "bi-diagram-3",
    "alerts": "bi-bell",
}

sidebar = html.Div(
    [
        html.Div(
            [
                html.H4(
                    "Crypto Pipeline", className="mb-0", style={"color": "#ffd700"}
                ),
                html.Small("Real-time streaming & ML", style={"color": "#888"}),
            ],
            className="mb-4",
        ),
        html.Hr(style={"border-color": "#333"}),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className=f"{NAV_ICONS[page_id]} me-2"), label],
                    href=f"/{page_id}" if page_id != "overview" else "/",
                    active="exact",
                    className="mb-1",
                    style={"border-radius": "8px"},
                )
                for page_id, label in PAGES.items()
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(style={"border-color": "#333"}),
        html.Label(
            "Coins",
            style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"},
        ),
        dcc.Dropdown(
            id="coin-select",
            multi=True,
            placeholder="Choose coins...",
            style={"color": "#000", "font-size": "0.85rem"},
        ),
        html.Div(
            [
                dbc.Button(
                    "All",
                    id="select-all-btn",
                    size="sm",
                    color="secondary",
                    className="me-1",
                    style={"font-size": "0.75rem", "padding": "2px 8px"},
                ),
                dbc.Button(
                    "Clear",
                    id="clear-btn",
                    size="sm",
                    color="secondary",
                    style={"font-size": "0.75rem", "padding": "2px 8px"},
                ),
            ],
            className="mt-1",
        ),
        html.Hr(style={"border-color": "#333"}),
        html.Label(
            "Timeframe",
            style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"},
        ),
        dcc.Dropdown(
            id="timeframe-select",
            options=[
                {"label": "1 min", "value": "1m"},
                {"label": "5 min", "value": "5m"},
                {"label": "15 min", "value": "15m"},
                {"label": "30 min", "value": "30m"},
                {"label": "1 hour", "value": "1h"},
            ],
            value="1m",
            clearable=False,
            style={"color": "#000", "font-size": "0.85rem"},
        ),
        html.Hr(style={"border-color": "#333"}),
        html.Label(
            "Chart Type",
            style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"},
        ),
        dcc.Dropdown(
            id="chart-type-select",
            options=[
                {"label": "Line", "value": "line"},
                {"label": "Candlestick", "value": "candlestick"},
            ],
            value="line",
            clearable=False,
            style={"color": "#000", "font-size": "0.85rem"},
        ),
        html.Label(
            "Time Range",
            style={
                "font-size": "0.85rem",
                "color": "#aaa",
                "margin-bottom": "4px",
                "margin-top": "8px",
            },
        ),
        dcc.Dropdown(
            id="time-range-select",
            options=[
                {"label": "This Session", "value": "session"},
                {"label": "All", "value": "all"},
                {"label": "Last 30 min", "value": "30m"},
                {"label": "Last 1 hour", "value": "1h"},
                {"label": "Last 6 hours", "value": "6h"},
                {"label": "Last 24 hours", "value": "24h"},
            ],
            value="session",
            clearable=False,
            style={"color": "#000", "font-size": "0.85rem"},
        ),
        html.Hr(style={"border-color": "#333"}),
        html.Label(
            "Model",
            style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"},
        ),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "ARIMA", "value": "arima"},
                {"label": "Prophet (seasonal)", "value": "prophet"},
            ],
            value="arima",
            clearable=False,
            style={"color": "#000", "font-size": "0.85rem"},
        ),
        html.Hr(style={"border-color": "#333"}),
        html.Div(
            [
                dbc.Checklist(
                    id="theme-toggle",
                    options=[{"label": " Light mode", "value": "light"}],
                    value=[],
                    switch=True,
                    style={"font-size": "0.85rem", "color": "#aaa"},
                ),
            ],
            className="mb-2",
        ),
        html.Div(id="sidebar-stats", style={"font-size": "0.8rem", "color": "#888"}),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        content,
        dcc.Interval(id="data-timer", interval=10_000),
        dcc.Interval(id="alert-timer", interval=10_000),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="toast-trigger", storage_type="memory", data=json.dumps([])),
        dcc.Store(id="theme-store", storage_type="memory", data="dark"),
        dcc.Store(id="client-tz", storage_type="memory", data=0),
        html.Div(id="toast-container"),
        html.Div(id="theme-dummy", style={"display": "none"}),
    ]
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)
