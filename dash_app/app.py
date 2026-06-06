import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from functools import lru_cache
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
from dash_extensions import WebSocket

from config.logging_config import setup_logger
from ml.features import build_features
from ml.model import load_model
from dash_app.alert_store import load_alerts, save_alerts, load_history, save_history
from dash_app.backtest import backtest
from viz.utils import sma, bollinger, rsi
from ws_gateway.client import get_last_update

logger = setup_logger("dash_app")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
GOLD_PATH = f"{OUTPUT_PATH}/gold"
SILVER_PATH = f"{OUTPUT_PATH}/silver"
MODEL_DIR = os.getenv("ML_MODEL_PATH", os.getenv("MODEL_PATH", "/tmp/crypto-model"))

# ─── Parquet cache ────────────────────────────────────────────────────

_parquet_cache: dict[str, tuple[pd.DataFrame, float]] = {}
CACHE_TTL = 8.0
_last_ws_ts: float = 0.0


def load_parquet(path: str) -> pd.DataFrame:
    now = time.time()
    cached = _parquet_cache.get(path)
    if cached is not None and (now - cached[1]) < CACHE_TTL:
        return cached[0]
    try:
        df = pd.read_parquet(path)
        _parquet_cache[path] = (df, now)
        return df
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return pd.DataFrame()


def load_model_safe(interval: str = "1m"):
    model_path = os.path.join(MODEL_DIR, f"model_{interval}.joblib")
    try:
        return load_model(model_path)
    except FileNotFoundError:
        pass
    # fallback to base model
    fallback = os.path.join(MODEL_DIR, "model.joblib")
    try:
        return load_model(fallback)
    except FileNotFoundError:
        return None, None, None


def prepare_df(df: pd.DataFrame, time_col: str, value_col: str, extra_cols: list[str]) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object"]):
        if c == time_col:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = [value_col] + [c for c in extra_cols if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[value_col, time_col]).sort_values(time_col)
    return df


def compute_predictions(gold_df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    if gold_df.empty or model is None:
        return pd.DataFrame()
    features_df, _ = build_features(gold_df)
    if features_df.empty:
        return pd.DataFrame()
    available = [c for c in feature_cols if c in features_df.columns]
    if len(available) < 5:
        logger.warning(f"Only {len(available)}/{len(feature_cols)} feature cols available")
        return pd.DataFrame()
    preds = model.predict(features_df[available])
    result = features_df[["coin_id", "window_start"]].copy()
    result["predicted_price"] = preds
    result["actual_price"] = features_df["avg_price"].values
    result["direction"] = np.where(
        result["predicted_price"] > result["actual_price"] * 1.005, "UP",
        np.where(result["predicted_price"] < result["actual_price"] * 0.995, "DOWN", "STABLE"),
    )
    result["pct_change_pred"] = ((result["predicted_price"] - result["actual_price"]) / result["actual_price"] * 100)
    return result


def compute_signal(row, r2):
    direction = row["direction"]
    pct_chg = row["pct_change_pred"]
    conf = row.get("confidence", 50)
    score = 0
    if direction == "UP":
        score = min(100, abs(pct_chg) * 10 + conf * 0.5)
    elif direction == "DOWN":
        score = -min(100, abs(pct_chg) * 10 + conf * 0.5)
    score = score * min(1, (r2 + 0.5) / 0.5) if r2 > 0 else score * 0.5
    if score > 60:
        signal, sig_color = "STRONG BUY", "#00E676"
    elif score > 20:
        signal, sig_color = "BUY", "#4CAF50"
    elif score < -60:
        signal, sig_color = "STRONG SELL", "#FF1744"
    elif score < -20:
        signal, sig_color = "SELL", "#F44336"
    else:
        signal, sig_color = "HOLD", "#FFC107"
    return signal, sig_color, round(score, 1)


# ─── Technical indicator helpers ──────────────────────────────────────


# ─── Timeframe resampling ─────────────────────────────────────────────

TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}

def resample_df(df: pd.DataFrame, interval: str, time_col: str) -> pd.DataFrame:
    if interval == "1m" or df.empty or time_col not in df.columns:
        return df
    rule = TIMEFRAMES.get(interval)
    if not rule:
        return df
    df = df.copy()
    # time_col comes as string after JSON roundtrip — restore datetime
    if time_col in df.columns and df[time_col].dtype == object:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    # Cast object columns that look numeric (PySpark writes Decimal as string)
    price_cols = ["avg_price", "min_price", "max_price", "avg_volume", "avg_change_pct", "price_volatility"]
    for c in price_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    agg = {}
    for col in num_cols:
        if col in ("coin_id", time_col):
            continue
        if col == "min_price":
            agg[col] = "min"
        elif col == "max_price":
            agg[col] = "max"
        elif col in ("avg_volume", "record_count"):
            agg[col] = "sum"
        else:
            agg[col] = "mean"
    df = df.set_index(time_col)
    resampled = df.groupby("coin_id").resample(rule).agg(agg)
    resampled = resampled.dropna(subset=[c for c in resampled.columns if c.startswith("avg_")][:1])
    resampled = resampled.reset_index()
    return resampled


# ─── Dash App ─────────────────────────────────────────────────────────

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
    "pipeline": "Pipeline Status",
    "predictions": "Predictions",
    "backtest": "Backtest",
    "alerts": "Alerts",
    "signals": "Signals",
}

NAV_ICONS = {
    "overview": "bi-graph-up",
    "technical": "bi-bar-chart",
    "comparison": "bi-arrow-left-right",
    "pipeline": "bi-diagram-3",
    "predictions": "bi-eye",
    "backtest": "bi-bar-chart-line",
    "alerts": "bi-bell",
    "signals": "bi-signal",
}

sidebar = html.Div(
    [
        html.Div([
            html.H4("Crypto Pipeline", className="mb-0", style={"color": "#ffd700"}),
            html.Small("Real-time streaming & ML", style={"color": "#888"}),
        ], className="mb-4"),
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
        html.Label("Coins", style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"}),
        dcc.Dropdown(id="coin-select", multi=True, placeholder="Choose coins...",
                     style={"color": "#000", "font-size": "0.85rem"}),
        html.Div([
            dbc.Button("All", id="select-all-btn", size="sm", color="secondary",
                       className="me-1", style={"font-size": "0.75rem", "padding": "2px 8px"}),
            dbc.Button("Clear", id="clear-btn", size="sm", color="secondary",
                       style={"font-size": "0.75rem", "padding": "2px 8px"}),
        ], className="mt-1"),
        html.Hr(style={"border-color": "#333"}),
        html.Label("Timeframe", style={"font-size": "0.85rem", "color": "#aaa", "margin-bottom": "4px"}),
        dcc.Dropdown(id="timeframe-select",
                     options=[
                         {"label": "1 min", "value": "1m"},
                         {"label": "5 min", "value": "5m"},
                         {"label": "15 min", "value": "15m"},
                         {"label": "30 min", "value": "30m"},
                         {"label": "1 hour", "value": "1h"},
                     ],
                     value="1m", clearable=False,
                     style={"color": "#000", "font-size": "0.85rem"}),
        html.Hr(style={"border-color": "#333"}),
        html.Div([
            dbc.Checklist(
                id="theme-toggle",
                options=[{"label": " Light mode", "value": "light"}],
                value=[],
                switch=True,
                style={"font-size": "0.85rem", "color": "#aaa"},
            ),
        ], className="mb-2"),
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
        WebSocket(id="ws", url=f"ws://{os.getenv('WS_GATEWAY_HOST', 'localhost')}:8765"),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="toast-trigger", storage_type="memory", data=json.dumps([])),
        dcc.Store(id="theme-store", storage_type="memory", data="dark"),
        dcc.Store(id="ws-last-update", storage_type="memory", data=0.0),
        html.Div(id="toast-container"),
        html.Div(id="theme-dummy", style={"display": "none"}),
    ]
)


# ─── Theme Toggle (clientside) ─────────────────────────────────────────

app.clientside_callback(
    """
    function(theme) {
        var links = document.querySelectorAll('link[rel="stylesheet"]');
        links.forEach(function(link) {
            var href = link.href || '';
            if (href.includes('darkly')) {
                link.disabled = theme !== 'dark';
            } else if (href.includes('flatly')) {
                link.disabled = theme !== 'light';
            }
        });
        return '';
    }
    """,
    Output("theme-dummy", "children"),
    Input("theme-store", "data"),
)


@callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_theme(val):
    return "light" if "light" in val else "dark"


# ─── Data Loading Callback ────────────────────────────────────────────

def _load_all_data(interval: str) -> str:
    global _last_data_ts
    gold = load_parquet(GOLD_PATH)
    silver = load_parquet(SILVER_PATH)
    model_obj, feature_cols, metrics = load_model_safe(interval)

    r2 = metrics.get("r2", 0) if metrics else 0
    result = {}

    if not gold.empty:
        df = prepare_df(gold, "window_start", "avg_price",
                        ["avg_volume", "avg_change_pct", "min_price", "max_price", "price_volatility", "record_count"])
        result["gold"] = df.to_dict("records") if not df.empty else []
        result["gold_time_col"] = "window_start"
        result["gold_value_col"] = "avg_price"
        result["gold_vol_col"] = "avg_volume"
        result["gold_chg_col"] = "avg_change_pct"
        result["gold_extra"] = ["min_price", "max_price", "price_volatility", "record_count"]

        preds = compute_predictions(df, model_obj, feature_cols or [])
        if not preds.empty:
            conf = max(0, min(100, (r2 * 50 + 50)))
            preds["confidence"] = conf
            result["predictions"] = preds.to_dict("records")
            result["pred_time_col"] = "window_start"
        else:
            result["predictions"] = []

    if not silver.empty:
        df = prepare_df(silver, "fetched_at", "price_usd",
                        ["volume_24h_usd", "change_24h_pct", "market_cap_usd"])
        result["silver"] = df.to_dict("records") if not df.empty else []
        result["silver_time_col"] = "fetched_at"
        result["silver_value_col"] = "price_usd"
        result["silver_vol_col"] = "volume_24h_usd"
        result["silver_chg_col"] = "change_24h_pct"
        result["silver_extra"] = ["market_cap_usd"]

    result["metrics"] = metrics or {}
    result["model_loaded"] = model_obj is not None

    coins = set()
    for key in ["gold", "silver"]:
        records = result.get(key, [])
        if isinstance(records, list):
            for r in records:
                if isinstance(r, dict) and "coin_id" in r:
                    coins.add(r["coin_id"])
    result["coins"] = sorted(coins)
    result["updated_at"] = datetime.now().isoformat()

    return json.dumps(result, default=str)


@callback(
    Output("data-store", "data"),
    Output("ws-last-update", "data"),
    Input("data-timer", "n_intervals"),
    State("timeframe-select", "value"),
)
def refresh_data(_, interval):
    global _last_ws_ts
    interval = interval or "1m"
    ws_ts = get_last_update()
    if ws_ts is not None and ws_ts <= _last_ws_ts:
        return no_update, no_update
    _last_ws_ts = ws_ts or 0
    return _load_all_data(interval), time.time()


@callback(
    Output("data-store", "data", allow_duplicate=True),
    Input("ws", "message"),
    State("timeframe-select", "value"),
    prevent_initial_call=True,
)
def ws_refresh(msg, interval):
    global _last_ws_ts
    if not msg:
        return no_update
    ws_ts = get_last_update()
    if ws_ts is not None and ws_ts <= _last_ws_ts:
        return no_update
    _last_ws_ts = ws_ts or 0
    return _load_all_data(interval or "1m")


# ─── Coin Selection Callbacks ─────────────────────────────────────────

@callback(
    Output("coin-select", "options"),
    Input("data-store", "data"),
)
def populate_coin_options(data_json):
    _, _, _, coins, _, _, _ = df_from_store(data_json)
    return [{"label": c.upper(), "value": c} for c in coins]


@callback(
    Output("coin-select", "value"),
    Input("select-all-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    State("coin-select", "options"),
    prevent_initial_call=True,
)
def handle_select_all_clear(all_clicks, clear_clicks, options):
    triggered = ctx.triggered_id
    if triggered == "select-all-btn":
        return [o["value"] for o in options]
    elif triggered == "clear-btn":
        return []
    return no_update


# ─── Page Router ─────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _parse_store_json(data_json: str) -> dict:
    return json.loads(data_json) if isinstance(data_json, str) else {}


def df_from_store(data_json: str) -> tuple:
    data = _parse_store_json(data_json)
    gold = pd.DataFrame(data.get("gold", []))
    silver = pd.DataFrame(data.get("silver", []))
    preds = pd.DataFrame(data.get("predictions", []))
    coins = data.get("coins", [])
    metrics = data.get("metrics", {})
    model_loaded = data.get("model_loaded", False)
    return gold, silver, preds, coins, metrics, model_loaded, data


def make_overview(gold, silver, preds, sel_coins, metrics, model_loaded, data, timeframe="1m"):
    use_gold = not gold.empty
    if use_gold:
        df = gold; tc = "window_start"; vc = "avg_price"
        voc = "avg_volume"; cc = "avg_change_pct"
    elif not silver.empty:
        df = silver; tc = "fetched_at"; vc = "price_usd"
        voc = "volume_24h_usd"; cc = "change_24h_pct"
    else:
        return html.Div(dbc.Alert("No data available yet.", color="warning"))
    if not sel_coins:
        return html.Div(dbc.Alert("Select coins from the sidebar to display.", color="info"))

    coin_filter = df[df["coin_id"].isin(sel_coins)]

    now_val = df[tc].max() if not df.empty else ""

    fig_main = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty:
            continue
        fig_main.add_trace(go.Scatter(
            x=cdf[tc], y=cdf[vc], mode="lines", name=coin.upper(),
            line=dict(width=2),
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:.4f}}<extra></extra>",
        ))
    fig_main.update_layout(
        title="Crypto Prices",
        xaxis_title="Time", yaxis_title="Price (USD)",
        template="plotly_dark", hovermode="x unified",
        height=400, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )

    fig_vol = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty or voc not in cdf.columns:
            continue
        fig_vol.add_trace(go.Bar(
            x=cdf[tc], y=cdf[voc], name=coin.upper(), opacity=0.75,
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig_vol.update_layout(
        template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20),
        barmode="group", hovermode="x unified", showlegend=False,
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )

    fig_chg = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty or cc not in cdf.columns:
            continue
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in cdf[cc]]
        fig_chg.add_trace(go.Bar(
            x=cdf[tc], y=cdf[cc], name=coin.upper(),
            marker_color=colors, opacity=0.75,
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
        ))
    fig_chg.update_layout(
        template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20),
        barmode="group", hovermode="x unified", showlegend=False,
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )

    # Metrics cards
    cards = []
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if len(cdf) < 2:
            continue
        last, prev = cdf.iloc[-1][vc], cdf.iloc[-2][vc]
        pct = ((last - prev) / prev * 100) if prev != 0 else 0
        delta_class = "text-success" if pct >= 0 else "text-danger"
        cards.append(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H6(coin.upper(), className="card-title text-muted"),
                    html.H3(f"${last:.4f}", className="card-text"),
                    html.Small(f"{pct:+.2f}%", className=delta_class),
                ])
            ], color="dark", inverse=True),
            xs=6, sm=6, md=3, lg=3,
        ))

    # Distribution & volatility
    fig_dist = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin][vc].dropna()
        if cdf.empty:
            continue
        fig_dist.add_trace(go.Box(y=cdf, name=coin.upper(), boxmean="sd"))
    fig_dist.update_layout(
        template="plotly_dark", height=250, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        showlegend=False,
    )

    fig_vola = go.Figure()
    if not use_gold:
        for coin in sel_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
            if cdf.empty:
                continue
            fig_vola.add_trace(go.Scatter(
                x=cdf[tc], y=cdf[voc] if voc in cdf.columns else [],
                mode="lines", name=coin.upper(),
            ))
    else:
        for coin in sel_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
            if cdf.empty or "price_volatility" not in cdf.columns:
                continue
            fig_vola.add_trace(go.Scatter(
                x=cdf[tc], y=cdf["price_volatility"], mode="lines", name=coin.upper(),
                line=dict(width=2),
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
            ))
    fig_vola.update_layout(
        template="plotly_dark", height=250, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        showlegend=False,
    )

    # Recent data table
    recent = coin_filter.sort_values(tc, ascending=False).head(25)
    if not recent.empty and tc in recent.columns:
        recent[tc] = recent[tc].astype(str)

    tf_label = {"1m": "1 min", "5m": "5 min", "15m": "15 min", "30m": "30 min", "1h": "1 hour"}.get(timeframe or "1m", timeframe)

    return html.Div([
        html.H3("Overview", className="mb-3"),
        html.Small(f"Layer: {'Gold' if use_gold else 'Silver'} ({tf_label}) · "
                   f"{len(sel_coins)} coins · {now_val}", className="text-muted"),
        html.Div(dbc.Row(cards), className="mt-3 mb-4"),
        dcc.Graph(figure=fig_main, className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_vol), xs=12, md=6),
            dbc.Col(dcc.Graph(figure=fig_chg), xs=12, md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_dist), xs=12, md=6),
            dbc.Col(dcc.Graph(figure=fig_vola), xs=12, md=6),
        ], className="mb-4"),
        html.H5("Recent Data", className="mt-4 mb-2"),
        dbc.Table.from_dataframe(recent.head(20), striped=True, bordered=False,
                                 class_name="table-dark", hover=True, responsive=True, size="sm"),
    ])


def make_technical(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    if gold.empty and silver.empty:
        return html.Div(dbc.Alert("No data available.", color="warning"))
    return html.Div([
        html.H3("Technical Analysis"),
        dcc.Dropdown(
            id="ta-coin-dropdown",
            options=[{"label": c.upper(), "value": c} for c in sel_coins],
            value=sel_coins[0] if sel_coins else None,
            clearable=False,
            className="mb-3",
            style={"color": "#000"},
        ),
        html.Div(id="ta-content"),
    ])


@callback(
    Output("ta-content", "children"),
    Input("ta-coin-dropdown", "value"),
    Input("data-store", "data"),
    Input("timeframe-select", "value"),
)
def update_technical(coin, data_json, timeframe):
    if not coin:
        return no_update
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)
    use_gold = not gold.empty
    if use_gold:
        try:
            gold = resample_df(gold, timeframe or "1m", "window_start")
        except Exception as e:
            logger.warning(f"Technical: resample failed for {timeframe}: {e}")
    df = gold if use_gold else silver
    if df.empty:
        return dbc.Alert("No data.", color="warning")

    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"
    voc = "avg_volume" if use_gold else "volume_24h_usd"
    cc = "avg_change_pct" if use_gold else "change_24h_pct"

    cdf = df[df["coin_id"] == coin].sort_values(tc).copy()
    if len(cdf) < 5:
        return dbc.Alert("Not enough data points for technical analysis.", color="info")

    price = cdf[vc]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price & Indicators", "RSI (14)", "Volume"],
    )

    fig.add_trace(
        go.Candlestick(
            x=cdf[tc],
            open=cdf.get("min_price", price),
            high=cdf.get("max_price", price),
            low=cdf.get("min_price", price * 0.99) if "min_price" not in cdf.columns else cdf["min_price"],
            close=price,
            name=coin.upper(),
            increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
        ),
        row=1, col=1,
    )
    for w in [10, 20]:
        if len(cdf) >= w:
            fig.add_trace(
                go.Scatter(x=cdf[tc], y=sma(price, w), mode="lines",
                           name=f"SMA({w})", line=dict(width=1.5)),
                row=1, col=1,
            )
    if len(cdf) >= 20:
        mid, upper, lower = bollinger(price, 20, 2)
        fig.add_trace(go.Scatter(x=cdf[tc], y=upper, mode="lines",
                                 name="BB Upper", line=dict(width=1, color="#888", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf[tc], y=lower, mode="lines",
                                 name="BB Lower", line=dict(width=1, color="#888", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

    if len(cdf) >= 14:
        rsi_vals = rsi(price, 14)
        fig.add_trace(go.Scatter(x=cdf[tc], y=rsi_vals, mode="lines",
                                 name="RSI (14)", line=dict(color="#FF9800", width=2)), row=2, col=1)
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#F44336", row=2, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#4CAF50", row=2, col=1)
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="#888", row=2, col=1)

    vol_colors = ["#4CAF50" if i >= 0 else "#F44336" for i in cdf[cc].fillna(0)]
    fig.add_trace(go.Bar(x=cdf[tc], y=cdf[voc], name="Volume",
                         marker_color=vol_colors, opacity=0.6), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=700, hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    # Metrics row
    last_price = price.iloc[-1]
    rsi_val = rsi(price, 14).iloc[-1] if len(cdf) >= 14 else 50
    bb_width = (bollinger(price)[1] - bollinger(price)[2]).iloc[-1] if len(cdf) >= 20 else 0
    pchg = ((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100) if len(price) > 1 else 0

    card_data = [
        ("Current", f"${last_price:.4f}", ""),
        ("RSI (14)", f"{rsi_val:.1f}",
         "Neutral" if 30 < rsi_val < 70 else "Overbought" if rsi_val >= 70 else "Oversold"),
        ("BB Width", f"${bb_width:.4f}", ""),
        ("Period Change", f"{pchg:+.2f}%", ""),
    ]
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(label, className="text-muted"),
            html.H4(val, className="card-text"),
            html.Small(note, className="text-muted") if note else "",
        ]), color="dark", inverse=True), xs=6, sm=3)
        for label, val, note in card_data
    ], className="mb-3")

    return html.Div([cards, dcc.Graph(figure=fig)])


def make_comparison(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    use_gold = not gold.empty
    df = gold if use_gold else silver
    if df.empty:
        return html.Div(dbc.Alert("No data available.", color="warning"))

    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"
    voc = "avg_volume" if use_gold else "avg_volume" if "avg_volume" in df.columns else "volume_24h_usd"
    coin_filter = df[df["coin_id"].isin(sel_coins)]

    if len(sel_coins) < 2:
        return html.Div(dbc.Alert("Select at least 2 coins in the sidebar for comparison.", color="info"))

    # Normalized
    fig_norm = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if len(cdf) < 2:
            continue
        base = cdf[vc].iloc[0]
        if base == 0:
            continue
        norm = cdf[vc] / base * 100
        fig_norm.add_trace(go.Scatter(
            x=cdf[tc], y=norm, mode="lines", name=coin.upper(),
            line=dict(width=2),
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig_norm.update_layout(
        title="Normalized Price (base=100)",
        template="plotly_dark", hovermode="x unified",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Normalized Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )

    # Correlation
    pivot = coin_filter.pivot_table(
        index=coin_filter.groupby("coin_id").cumcount(),
        columns="coin_id", values=vc,
    )
    fig_corr = go.Figure()
    if len(pivot.columns) >= 2 and len(pivot) >= 2:
        corr = pivot.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             range_color=[-1, 1], aspect="auto")
        fig_corr.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        )

    # Volume comparison
    vol_pivot = coin_filter.pivot_table(
        index=coin_filter.groupby("coin_id").cumcount(),
        columns="coin_id", values=voc,
    )
    fig_vc = go.Figure()
    if not vol_pivot.empty and len(vol_pivot.columns) >= 1:
        fig_vc = px.bar(vol_pivot.sum().reset_index(), x="coin_id", y=0, color="coin_id",
                        template="plotly_dark", labels={"coin_id": "", "0": "Total Volume"})
        fig_vc.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        )

    # Returns box
    returns_data = []
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if len(cdf) >= 5:
            ret = cdf[vc].pct_change().dropna().tail(50)
            for val in ret:
                returns_data.append({"coin": coin.upper(), "return_pct": val * 100})
    fig_ret = go.Figure()
    if returns_data:
        ret_df = pd.DataFrame(returns_data)
        fig_ret = px.box(ret_df, x="coin", y="return_pct", color="coin",
                         template="plotly_dark", points="all",
                         labels={"coin": "", "return_pct": "Return %"})
        fig_ret.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        )

    return html.Div([
        html.H3("Multi-Coin Comparison"),
        dcc.Graph(figure=fig_norm, className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_corr), xs=12, md=6),
            dbc.Col(dcc.Graph(figure=fig_vc), xs=12, md=6),
        ], className="mb-4"),
        dcc.Graph(figure=fig_ret, className="mb-4"),
    ])


def make_pipeline(gold, silver, preds, coins, metrics, model_loaded, data):
    use_gold = not gold.empty
    df = gold if use_gold else silver
    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"
    voc = "avg_volume" if use_gold else "volume_24h_usd"

    total_gold = len(gold)
    total_silver = len(silver)
    layer_label = "Gold" if use_gold else "Silver"
    now_val = df[tc].max() if not df.empty else ""

    stats_rows = []
    for coin in coins:
        cdf = df[df["coin_id"] == coin]
        if cdf.empty:
            continue
        stats_rows.append({
            "Coin": coin.upper(),
            "Records": len(cdf),
            "Latest": f"${cdf[vc].iloc[-1]:.4f}" if vc in cdf.columns else "-",
            "Avg": f"${cdf[vc].mean():.4f}",
            "Min": f"${cdf[vc].min():.4f}",
            "Max": f"${cdf[vc].max():.4f}",
            "Volume": f"${cdf[voc].sum():,.0f}" if voc in cdf.columns else "-",
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("Records", ascending=False) if stats_rows else pd.DataFrame()

    return html.Div([
        html.H3("Pipeline Status"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Gold Records", className="text-muted"),
                html.H3(f"{total_gold:,}", className="card-text"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Silver Records", className="text-muted"),
                html.H3(f"{total_silver:,}", className="card-text"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Coins Tracked", className="text-muted"),
                html.H3(str(len(coins)), className="card-text"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Data Layer", className="text-muted"),
                html.H3(layer_label, className="card-text"),
            ]), color="dark", inverse=True), xs=6, sm=3),
        ], className="mb-4"),
        html.H5("Per-Coin Stats"),
        dbc.Table.from_dataframe(stats_df, striped=True, bordered=False,
                                 class_name="table-dark", hover=True, responsive=True, size="sm"),

        html.H5("Data Freshness", className="mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(["Oldest: ", html.Br(), str(df[tc].min()) if not df.empty else "N/A"],
                       className="mb-0"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(["Newest: ", html.Br(), str(now_val) if not df.empty else "N/A"],
                       className="mb-0"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(["Model R²: ", html.Br(), f"{metrics.get('r2', 'N/A')}"],
                       className="mb-0"),
            ]), color="dark", inverse=True), xs=6, sm=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(["Model MAE: ", html.Br(), f"${metrics.get('mae', 0):.2f}"],
                       className="mb-0"),
            ]), color="dark", inverse=True), xs=6, sm=3),
        ]),
    ])


def make_predictions(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    if preds.empty:
        return html.Div(dbc.Alert("No predictions available. The model needs more data.", color="info"))
    use_gold = not gold.empty
    df = gold if use_gold else silver
    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"

    coin = sel_coins[0] if sel_coins else preds["coin_id"].iloc[0]

    return html.Div([
        html.H3("Price Prediction"),
        dcc.Dropdown(
            id="pred-coin-dropdown",
            options=[{"label": c.upper(), "value": c} for c in sorted(preds["coin_id"].unique())],
            value=coin, clearable=False,
            className="mb-3", style={"color": "#000"},
        ),
        html.Div(id="pred-content"),
    ])


@callback(
    Output("pred-content", "children"),
    Input("pred-coin-dropdown", "value"),
    Input("data-store", "data"),
)
def update_predictions(coin, data_json):
    if not coin:
        return no_update
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)
    if preds.empty:
        return dbc.Alert("No predictions yet.", color="info")

    r2 = metrics.get("r2", 0)
    cp = preds[preds["coin_id"] == coin].sort_values("window_start")
    if cp.empty:
        return dbc.Alert("No predictions for this coin.", color="info")

    latest = cp.iloc[-1]
    direction = latest["direction"]
    pred_price = latest["predicted_price"]
    actual_price = latest["actual_price"]
    pct_chg = latest["pct_change_pred"]
    conf = latest["confidence"]

    if direction == "UP":
        arrow, color = "\u2191", "#4CAF50"
    elif direction == "DOWN":
        arrow, color = "\u2193", "#F44336"
    else:
        arrow, color = "\u2192", "#FFC107"

    sig, sig_col, score = compute_signal(latest, r2)

    # Direction card
    card = dbc.Card([
        dbc.CardBody([
            html.H2(arrow, style={"font-size": "3rem", "color": color}, className="text-center"),
            html.H3(direction, className="text-center", style={"color": color, "font-weight": "800"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Small("Current", className="text-muted d-block"),
                    html.H4(f"${actual_price:.4f}"),
                ]), xs=4, className="text-center"),
                dbc.Col(html.Div([
                    html.Small("", className="d-block", style={"font-size": "2rem"}),
                    html.H4("\u2192"),
                ]), xs=4, className="text-center"),
                dbc.Col(html.Div([
                    html.Small("Predicted", className="text-muted d-block"),
                    html.H4(f"${pred_price:.4f}"),
                ]), xs=4, className="text-center"),
            ], className="mt-2"),
            html.H5(f"{pct_chg:+.2f}%", className="text-center", style={"color": color}),
        ])
    ], color="dark", inverse=True, className="mb-3")

    # Metrics cards
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Confidence", className="text-muted"),
            html.H4(f"{conf:.0f}%", style={"color": color}),
        ]), color="dark", inverse=True), xs=6, sm=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Signal", className="text-muted"),
            html.H4(sig, style={"color": sig_col}),
        ]), color="dark", inverse=True), xs=6, sm=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Score", className="text-muted"),
            html.H4(f"{score}", style={"color": sig_col}),
        ]), color="dark", inverse=True), xs=6, sm=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Model R²", className="text-muted"),
            html.H4(f"{r2:.3f}"),
        ]), color="dark", inverse=True), xs=6, sm=3),
    ], className="mb-3")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cp["window_start"], y=cp["actual_price"],
        mode="lines+markers", name="Actual",
        line=dict(color="#00BCD4", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=cp["window_start"], y=cp["predicted_price"],
        mode="lines+markers", name="Predicted",
        line=dict(color="#FF9800", width=2.5, dash="dot"),
    ))
    dir_colors = {"UP": "#4CAF50", "DOWN": "#F44336", "STABLE": "#FFC107"}
    fig.add_trace(go.Scatter(
        x=cp["window_start"], y=cp["predicted_price"],
        mode="markers",
        marker=dict(size=8, color=[dir_colors.get(d, "#888") for d in cp["direction"]],
                    symbol=["triangle-up" if d == "UP" else "triangle-down" if d == "DOWN" else "circle"
                            for d in cp["direction"]]),
        name="Direction",
        hovertemplate="%{text}<extra></extra>",
        text=[f"{d} ({c:+.1f}%)" for d, c in zip(cp["direction"], cp["pct_change_pred"])],
    ))
    fig.update_layout(
        title=f"{coin.upper()} \u2014 Actual vs Predicted",
        template="plotly_dark", hovermode="x unified", height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
    )

    # Direction breakdown
    dir_counts = cp["direction"].value_counts()
    dir_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{dir_counts.get(d, 0) / len(cp) * 100:.0f}%", style={"color": dc}),
            html.Small(f"{d} ({dir_counts.get(d, 0)})", className="text-muted"),
        ]), color="dark", inverse=True), xs=4)
        for d, dc in [("UP", "#4CAF50"), ("DOWN", "#F44336"), ("STABLE", "#FFC107")]
    ], className="mb-3")

    # Table
    display = cp.sort_values("window_start", ascending=False).head(20).copy()
    display["window_start"] = display["window_start"].astype(str)
    display["pct_change_pred"] = display["pct_change_pred"].round(2).apply(lambda x: f"{x:+.2f}%")
    display["actual_price"] = display["actual_price"].round(4).apply(lambda x: f"${x:.4f}")
    display["predicted_price"] = display["predicted_price"].round(4).apply(lambda x: f"${x:.4f}")
    display["confidence"] = display["confidence"].round(0).apply(lambda x: f"{x:.0f}%")

    return html.Div([
        card, cards, dcc.Graph(figure=fig), dir_cards,
        html.H5("Prediction History"),
        dbc.Table.from_dataframe(
            display[["window_start", "actual_price", "predicted_price", "direction", "pct_change_pred", "confidence"]],
            striped=True, bordered=False, class_name="table-dark", hover=True, responsive=True, size="sm",
        ),
    ])


def make_alerts(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    use_gold = not gold.empty
    df = gold if use_gold else silver
    vc = "avg_price" if use_gold else "price_usd"
    cc = "avg_change_pct" if use_gold else "change_24h_pct"

    return html.Div([
        html.H3("Alert System"),
        dbc.Row([
            dbc.Col([
                html.H5("Create Alert"),
                dbc.Card(dbc.CardBody([
                    dbc.Label("Coin"),
                    dcc.Dropdown(id="alert-coin", options=[{"label": c.upper(), "value": c} for c in sel_coins],
                                 value=sel_coins[0] if sel_coins else None, style={"color": "#000"}),
                    dbc.Label("Type", className="mt-2"),
                    dcc.Dropdown(id="alert-type", options=[
                        {"label": "Price threshold", "value": "price"},
                        {"label": "24h Change %", "value": "change_24h"},
                        {"label": "Predicted Direction", "value": "direction"},
                    ], value="price", style={"color": "#000"}),
                    html.Div(id="alert-config"),
                    dbc.Button("Add Alert", id="add-alert-btn", color="primary", className="mt-3", n_clicks=0),
                ]), color="dark", inverse=True),
            ], xs=12, md=6),
            dbc.Col([
                html.H5("Active Alerts"),
                html.Div(id="active-alerts-list"),
            ], xs=12, md=6),
        ]),
        html.H5("Alert History", className="mt-4"),
        html.Div(id="alert-history-list"),
    ])


def make_backtest(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    if preds.empty:
        return html.Div(dbc.Alert("No predictions yet. Wait for model training.", color="info"))
    if sel_coins:
        coin_preds = preds[preds["coin_id"].isin(sel_coins)]
    else:
        coin_preds = preds
    if coin_preds.empty:
        return html.Div(dbc.Alert("No data for selected coins.", color="info"))

    result = backtest(coin_preds)
    if "error" in result:
        return html.Div(dbc.Alert(result["error"], color="warning"))

    components = [html.H3("Backtesting Engine", className="mb-3"),
                  html.Small("Strategy: position = last predicted direction; return = position * actual pct_change",
                             className="text-muted d-block mb-3")]

    for entry in result["results"]:
        coin = entry["coin"]
        stats = entry["stats"]
        trades = entry["trades"]
        curve = entry["curve"]

        # Stats cards
        stat_cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Total Return", className="text-muted"),
                html.H4(f"{stats['total_return_pct']:+.2f}%", style={"color": "#4CAF50" if stats['total_return_pct'] >= 0 else "#F44336"}),
            ]), color="dark", inverse=True), xs=6, sm=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Buy & Hold", className="text-muted"),
                html.H4(f"{stats['buy_hold_return_pct']:+.2f}%"),
            ]), color="dark", inverse=True), xs=6, sm=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Win Rate", className="text-muted"),
                html.H4(f"{stats['win_rate_pct']:.0f}%", style={"color": "#4CAF50"}),
            ]), color="dark", inverse=True), xs=6, sm=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Max DD", className="text-muted"),
                html.H4(f"{stats['max_drawdown_pct']:.1f}%", style={"color": "#F44336"}),
            ]), color="dark", inverse=True), xs=6, sm=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Sharpe", className="text-muted"),
                html.H4(f"{stats['sharpe_ratio']:.2f}", style={"color": "#FF9800"}),
            ]), color="dark", inverse=True), xs=6, sm=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Trades", className="text-muted"),
                html.H4(str(stats['num_trades'])),
            ]), color="dark", inverse=True), xs=6, sm=2),
        ], className="mb-3")

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["window_start"], y=curve["cum_strategy"],
            mode="lines", name="Strategy", line=dict(color="#00BCD4", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=curve["window_start"], y=curve["buy_hold"],
            mode="lines", name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dash"),
        ))

        # Trade markers
        if trades:
            trade_times = []
            trade_returns = []
            trade_colors = []
            for t in trades:
                trade_times.append(t["entry_time"])
                trade_returns.append(1.0)
                trade_colors.append("#4CAF50" if t["pnl_pct"] > 0 else "#F44336")
            fig.add_trace(go.Scatter(
                x=trade_times, y=trade_returns, mode="markers",
                marker=dict(size=10, color=trade_colors, symbol="triangle-up"),
                name="Trades",
                hovertemplate="%{text}",
                text=[f"{t['direction']} PnL: {t['pnl_pct']:+.2f}%" for t in trades],
            ))

        fig.update_layout(
            title=f"{coin.upper()} Equity Curve",
            template="plotly_dark", hovermode="x unified", height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
            yaxis_title="Cumulative Return",
        )

        # Trade table
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl_pct"])
        if not trades_df.empty:
            trades_df["pnl_pct"] = trades_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")

        components.append(html.H5(f"{coin.upper()}", className="mt-3"))
        components.append(stat_cards)
        components.append(dcc.Graph(figure=fig))

        if not trades_df.empty:
            components.append(html.H6("Trades", className="mt-2"))
            components.append(dbc.Table.from_dataframe(
                trades_df, striped=True, bordered=False, class_name="table-dark", hover=True, responsive=True, size="sm",
            ))

    return html.Div(components)


def make_signals(gold, silver, preds, sel_coins, metrics, model_loaded, data):
    if preds.empty:
        return html.Div(dbc.Alert("No prediction data available yet.", color="info"))

    r2 = metrics.get("r2", 0)
    latest_per = preds.sort_values("window_start").groupby("coin_id").last().reset_index()

    rows = []
    for _, row in latest_per.iterrows():
        sig, sig_color, score = compute_signal(row, r2)
        rows.append({
            "coin": row["coin_id"].upper(),
            "signal": sig,
            "score": score,
            "direction": row["direction"],
            "actual": f"${row['actual_price']:.2f}",
            "predicted": f"${row['predicted_price']:.2f}",
            "change": f"{row['pct_change_pred']:+.2f}%",
            "color": sig_color,
        })

    signal_df = pd.DataFrame(rows).sort_values("score", ascending=False)

    cards = []
    for _, r in signal_df.iterrows():
        cards.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H5(r["coin"], className="card-title"),
                html.H3(r["signal"], style={"color": r["color"], "font-weight": "800"}),
                html.P(f"Score: {r['score']}", className="mb-1"),
                html.Small(f"{r['actual']} \u2192 {r['predicted']} ({r['change']})", className="text-muted"),
            ]), color="dark", inverse=True),
            xs=6, sm=6, md=3,
        ))
    signal_card_row = dbc.Row(cards, className="mb-4")

    # Table
    signal_df_display = signal_df[["coin", "signal", "score", "direction", "actual", "predicted", "change"]]

    # Signal history chart
    sel_coin = sel_coins[0] if sel_coins else preds["coin_id"].iloc[0]
    cp = preds[preds["coin_id"] == sel_coin].sort_values("window_start").copy()
    if len(cp) >= 3:
        cp["signal_score"] = cp.apply(
            lambda r: min(100, abs(r["pct_change_pred"]) * 10 + r["confidence"] * 0.5)
            if r["direction"] == "UP" else -min(100, abs(r["pct_change_pred"]) * 10 + r["confidence"] * 0.5),
            axis=1,
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=cp["window_start"], y=cp["actual_price"], mode="lines",
                                 name="Price", line=dict(color="#00BCD4", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=cp["window_start"], y=cp["signal_score"], mode="lines+markers",
                                 name="Signal Score", line=dict(color="#FF9800", width=2),
                                 fill="tozeroy", fillcolor="rgba(255,152,0,0.1)"), secondary_y=True)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="#4CAF50", secondary_y=True)
        fig.add_hline(y=-20, line_width=1, line_dash="dash", line_color="#F44336", secondary_y=True)
        fig.update_layout(
            template="plotly_dark", hovermode="x unified", height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        )
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Signal Score", secondary_y=True, range=[-100, 100])
        signal_chart = dcc.Graph(figure=fig)
    else:
        signal_chart = html.Div()

    return html.Div([
        html.H3("Trading Signals"),
        signal_card_row,
        html.H5("All Signals"),
        dbc.Table.from_dataframe(signal_df_display, striped=True, bordered=False,
                                 class_name="table-dark", hover=True, responsive=True, size="sm"),
        html.H5("Signal History"),
        dcc.Dropdown(
            id="sig-coin-dropdown",
            options=[{"label": c.upper(), "value": c} for c in sorted(preds["coin_id"].unique())],
            value=sel_coin, clearable=False,
            className="mb-2", style={"color": "#000"},
        ),
        html.Div(id="sig-chart-container", children=signal_chart),
    ])


# ─── Alert Management Callbacks ──────────────────────────────────────

@callback(
    Output("alert-config", "children"),
    Input("alert-type", "value"),
    Input("alert-coin", "value"),
    Input("data-store", "data"),
)
def update_alert_config(alert_type, coin, data_json):
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)
    use_gold = not gold.empty
    df = gold if use_gold else silver
    vc = "avg_price" if use_gold else "price_usd"

    current_price = 0
    if coin and not df.empty:
        cdf = df[df["coin_id"] == coin]
        if not cdf.empty:
            current_price = float(cdf[vc].iloc[-1])

    if alert_type == "price":
        return html.Div([
            dbc.Label("Condition", className="mt-2"),
            dcc.Dropdown(id="alert-condition", options=[
                {"label": "Above", "value": "above"},
                {"label": "Below", "value": "below"},
            ], value="above", style={"color": "#000"}),
            dbc.Label("Threshold ($)", className="mt-2"),
            dcc.Input(id="alert-threshold", type="number", value=round(current_price * 1.1, 2),
                      step=0.01, style={"width": "100%", "color": "#000"}),
        ])
    elif alert_type == "change_24h":
        return html.Div([
            dbc.Label("Condition", className="mt-2"),
            dcc.Dropdown(id="alert-condition", options=[
                {"label": "Above", "value": "above"},
                {"label": "Below", "value": "below"},
            ], value="above", style={"color": "#000"}),
            dbc.Label("Threshold (%)", className="mt-2"),
            dcc.Input(id="alert-threshold", type="number", value=5.0,
                      step=0.5, style={"width": "100%", "color": "#000"}),
        ])
    else:
        return html.Div([
            dbc.Label("Direction", className="mt-2"),
            dcc.Dropdown(id="alert-condition", options=[
                {"label": "UP", "value": "UP"},
                {"label": "DOWN", "value": "DOWN"},
            ], value="UP", style={"color": "#000"}),
            dcc.Input(id="alert-threshold", type="hidden", value=0),
        ])


@callback(
    Output("active-alerts-list", "children", allow_duplicate=True),
    Input("add-alert-btn", "n_clicks"),
    State("alert-coin", "value"),
    State("alert-type", "value"),
    State("alert-condition", "value"),
    State("alert-threshold", "value"),
    prevent_initial_call=True,
)
def add_alert(n_clicks, coin, alert_type, condition, threshold):
    if not coin or not condition:
        return no_update
    alerts = load_alerts()
    alerts.append({
        "id": int(time.time() * 1000),
        "coin": coin,
        "type": alert_type,
        "condition": condition,
        "threshold": float(threshold) if threshold else 0,
        "active": True,
    })
    save_alerts(alerts)
    return _render_active_alerts()


def _render_active_alerts():
    alerts = load_alerts()
    active = [a for a in alerts if a.get("active")]
    if not active:
        return dbc.Alert("No active alerts. Create one above.", color="info", className="mt-2")
    items = []
    for alert in active:
        items.append(dbc.ListGroupItem([
            html.Div([
                html.Strong(f"{alert['coin'].upper()}"),
                html.Span(f" — {alert['type']} {alert['condition']} {alert['threshold']}", className="ms-1"),
                dbc.Badge("Active", color="success", className="ms-2"),
                dbc.Button("Remove", id={"type": "remove-alert", "index": alert["id"]},
                           size="sm", color="danger", className="ms-auto float-end"),
            ], style={"display": "flex", "alignItems": "center"}),
        ]))
    return dbc.ListGroup(items, flush=True)


@callback(
    Output("active-alerts-list", "children"),
    Input("data-store", "data"),
)
def show_active_alerts(_):
    return _render_active_alerts()


@callback(
    Output("active-alerts-list", "children", allow_duplicate=True),
    Input({"type": "remove-alert", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def remove_alert(n_clicks):
    if not any(n for n in n_clicks if n):
        return no_update
    triggered_id = ctx.triggered_id
    if triggered_id and "index" in triggered_id:
        alert_id = triggered_id["index"]
        alerts = load_alerts()
        alerts = [a for a in alerts if a["id"] != alert_id]
        save_alerts(alerts)
    return _render_active_alerts()


@callback(
    Output("alert-history-list", "children"),
    Input("data-store", "data"),
)
def show_alert_history(_):
    history = load_history()
    if not history:
        return dbc.Alert("No alerts triggered yet.", color="info")
    hist_df = pd.DataFrame(reversed(history[-50:]))
    return dbc.Table.from_dataframe(
        hist_df, striped=True, bordered=False, class_name="table-dark", hover=True, responsive=True, size="sm",
    )


# ─── Alert Trigger Check ─────────────────────────────────────────────

@callback(
    Output("toast-trigger", "data"),
    Input("alert-timer", "n_intervals"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def check_alerts(_, data_json):
    alerts = load_alerts()
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)

    use_gold = not gold.empty
    df = gold if use_gold else silver
    vc = "avg_price" if use_gold else "price_usd"
    cc = "avg_change_pct" if use_gold else "change_24h_pct"

    triggered = []
    for alert in alerts:
        if not alert.get("active"):
            continue
        coin = alert["coin"]
        a_type = alert["type"]
        condition = alert["condition"]
        threshold = alert["threshold"]

        cdf = df[df["coin_id"] == coin]
        if cdf.empty:
            continue

        fire = False
        if a_type == "price":
            current = float(cdf[vc].iloc[-1])
            if condition == "above" and current > threshold:
                fire = True
            elif condition == "below" and current < threshold:
                fire = True
        elif a_type == "change_24h":
            if cc in cdf.columns:
                chg_val = float(cdf[cc].iloc[-1])
                if condition == "above" and chg_val > threshold:
                    fire = True
                elif condition == "below" and chg_val < threshold:
                    fire = True
        elif a_type == "direction":
            if not preds.empty:
                cp = preds[preds["coin_id"] == coin].sort_values("window_start")
                if not cp.empty:
                    dir_val = cp["direction"].iloc[-1]
                    if dir_val == condition:
                        fire = True

        if fire:
            msg = f"{coin.upper()} {condition} {threshold}"
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "coin": coin.upper(),
                "type": a_type,
                "message": msg,
            }
            triggered.append(entry)
            history = load_history()
            history.append(entry)
            save_history(history)

    return json.dumps(triggered)


@callback(
    Output("toast-container", "children"),
    Input("toast-trigger", "data"),
)
def show_toasts(triggered_json):
    triggered = json.loads(triggered_json) if triggered_json else []
    toasts = []
    for t in triggered[-3:]:
        toasts.append(dbc.Toast(
            t["message"],
            header=f"{t['coin']} — {t['type']}",
            icon="warning",
            duration=5000,
            style={"position": "fixed", "top": 80, "right": 20, "zIndex": 9999},
            is_open=True,
        ))
    return toasts


# ─── Page Routing Callback ───────────────────────────────────────────

@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("data-store", "data"),
    Input("coin-select", "value"),
    Input("timeframe-select", "value"),
)
def render_page(pathname, data_json, sel_coins, timeframe):
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)

    # Resample gold data for chart pages
    if not gold.empty:
        try:
            gold_rs = resample_df(gold, timeframe or "1m", "window_start")
        except Exception as e:
            logger.warning(f"Resample failed for {timeframe}: {e}")
            gold_rs = gold
    else:
        gold_rs = gold

    page = pathname.strip("/") if pathname else ""
    if page not in PAGES:
        page = "overview"

    if page == "overview":
        return make_overview(gold_rs, silver, preds, sel_coins or [], metrics, model_loaded, data, timeframe)
    elif page == "technical":
        return make_technical(gold_rs, silver, preds, sel_coins or [], metrics, model_loaded, data)
    elif page == "comparison":
        return make_comparison(gold_rs, silver, preds, sel_coins or [], metrics, model_loaded, data)
    elif page == "pipeline":
        return make_pipeline(gold, silver, preds, coins, metrics, model_loaded, data)
    elif page == "predictions":
        return make_predictions(gold_rs, silver, preds, sel_coins or [], metrics, model_loaded, data)
    elif page == "backtest":
        return make_backtest(gold_rs, silver, preds, sel_coins or [], metrics, model_loaded, data)
    elif page == "alerts":
        return make_alerts(gold, silver, preds, sel_coins or [], metrics, model_loaded, data)
    elif page == "signals":
        return make_signals(gold, silver, preds, sel_coins or [], metrics, model_loaded, data)

    return html.Div(dbc.Alert("Page not found", color="danger"))


# ─── Sidebar Stats ───────────────────────────────────────────────────

@callback(
    Output("sidebar-stats", "children"),
    Input("data-store", "data"),
)
def update_sidebar_stats(data_json):
    gold, silver, preds, coins, metrics, model_loaded, data = df_from_store(data_json)
    items = [
        html.P([html.Strong("Coins:"), f" {len(coins)}"], className="mb-1"),
        html.P([html.Strong("Gold:"), f" {len(gold)}"], className="mb-1"),
        html.P([html.Strong("Silver:"), f" {len(silver)}"], className="mb-1"),
        html.P([html.Strong("Model:"), " Loaded" if model_loaded else " Waiting"], className="mb-1",
               style={"color": "#4CAF50" if model_loaded else "#FF9800"}),
        html.P([html.Strong("Updated:"), f" {data.get('updated_at', '')[:19]}"],
               className="mb-1", style={"font-size": "0.75rem"}),
    ]
    return items


# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)
