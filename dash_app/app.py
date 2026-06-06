import os
import json
import time
from datetime import datetime

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
from dash_app.alert_store import load_alerts, save_alerts, load_history, save_history
from viz.utils import sma, bollinger, rsi
from ws_gateway.client import get_last_update

logger = setup_logger("dash_app")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
GOLD_PATH = f"{OUTPUT_PATH}/gold"
SILVER_PATH = f"{OUTPUT_PATH}/silver"

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


def prepare_df(df: pd.DataFrame, time_col: str, value_col: str, extra_cols: list[str]) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object"]):
        if c == time_col:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = [value_col] + [c for c in extra_cols if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[value_col, time_col]).sort_values(time_col)
    return df


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
    "alerts": "Alerts",
}

NAV_ICONS = {
    "overview": "bi-graph-up",
    "technical": "bi-bar-chart",
    "comparison": "bi-arrow-left-right",
    "pipeline": "bi-diagram-3",
    "alerts": "bi-bell",
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
    gold = load_parquet(GOLD_PATH)
    silver = load_parquet(SILVER_PATH)
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

    if not silver.empty:
        df = prepare_df(silver, "fetched_at", "price_usd",
                        ["volume_24h_usd", "change_24h_pct", "market_cap_usd"])
        result["silver"] = df.to_dict("records") if not df.empty else []
        result["silver_time_col"] = "fetched_at"
        result["silver_value_col"] = "price_usd"
        result["silver_vol_col"] = "volume_24h_usd"
        result["silver_chg_col"] = "change_24h_pct"
        result["silver_extra"] = ["market_cap_usd"]

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
def refresh_data(n_intervals, interval):
    global _last_ws_ts
    interval = interval or "1m"
    ws_ts = get_last_update()
    if n_intervals > 0 and ws_ts is not None and ws_ts <= _last_ws_ts:
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
    _, _, coins, _ = df_from_store(data_json)
    return [{"label": c.upper(), "value": c} for c in coins]


@callback(
    Output("coin-select", "value"),
    Input("coin-select", "options"),
    State("coin-select", "value"),
    prevent_initial_call=True,
)
def auto_select_coin(options, current_value):
    if current_value:
        return no_update
    for o in options:
        return [o["value"]]
    return no_update


@callback(
    Output("coin-select", "value", allow_duplicate=True),
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
    coins = data.get("coins", [])
    return gold, silver, coins, data


def make_overview(gold, silver, sel_coins, data, timeframe="1m"):
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


def make_technical(gold, silver, sel_coins, data):
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
    gold, silver, coins, data = df_from_store(data_json)
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


def make_comparison(gold, silver, sel_coins, data):
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


def make_pipeline(gold, silver, coins, data):
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
        ]),
    ])


def make_alerts(gold, silver, sel_coins, data):
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


# ─── Alert Management Callbacks ──────────────────────────────────────

@callback(
    Output("alert-config", "children"),
    Input("alert-type", "value"),
    Input("alert-coin", "value"),
    Input("data-store", "data"),
)
def update_alert_config(alert_type, coin, data_json):
    gold, silver, coins, data = df_from_store(data_json)
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
    else:
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
    gold, silver, coins, data = df_from_store(data_json)

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
    gold, silver, coins, data = df_from_store(data_json)

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
        return make_overview(gold_rs, silver, sel_coins or [], data, timeframe)
    elif page == "technical":
        return make_technical(gold_rs, silver, sel_coins or [], data)
    elif page == "comparison":
        return make_comparison(gold_rs, silver, sel_coins or [], data)
    elif page == "pipeline":
        return make_pipeline(gold, silver, coins, data)
    elif page == "alerts":
        return make_alerts(gold, silver, sel_coins or [], data)

    return html.Div(dbc.Alert("Page not found", color="danger"))


# ─── Sidebar Stats ───────────────────────────────────────────────────

@callback(
    Output("sidebar-stats", "children"),
    Input("data-store", "data"),
)
def update_sidebar_stats(data_json):
    gold, silver, coins, data = df_from_store(data_json)
    items = [
        html.P([html.Strong("Coins:"), f" {len(coins)}"], className="mb-1"),
        html.P([html.Strong("Gold:"), f" {len(gold)}"], className="mb-1"),
        html.P([html.Strong("Silver:"), f" {len(silver)}"], className="mb-1"),
        html.P([html.Strong("Updated:"), f" {data.get('updated_at', '')[:19]}"],
               className="mb-1", style={"font-size": "0.75rem"}),
    ]
    return items


# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)
