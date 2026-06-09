import json
import time
from datetime import datetime

import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update, clientside_callback
import dash_bootstrap_components as dbc

from config.logging_config import setup_logger
from config.timezone import PHNOM_PENH_TZ
from dash_app.data_utils import (
    _load_all_data, _last_ws_ts, df_from_store, resample_df,
)
from dash_app.pages import (
    make_overview, make_predictions, make_pipeline, make_comparison,
    make_alerts, make_technical, make_sessions, _update_technical_content,
)
from dash_app.alert_store import load_alerts, save_alerts, load_history, save_history
from dash_app.session_manager import record_data
from ws_gateway.client import get_last_update

logger = setup_logger("dash_app")


# ─── Theme Toggle ─────────────────────────────────────────────────────


clientside_callback(
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


# ─── Data Loading Callbacks ───────────────────────────────────────────


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
    if n_intervals is not None and n_intervals > 0 and ws_ts is not None and ws_ts <= _last_ws_ts:
        return no_update, no_update
    _last_ws_ts = ws_ts or 0
    data_json = _load_all_data(interval)
    record_data(data_json)
    return data_json, time.time()


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
    data_json = _load_all_data(interval or "1m")
    record_data(data_json)
    return data_json


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


# ─── Technical Analysis Callback ──────────────────────────────────────


@callback(
    Output("ta-content", "children"),
    Input("ta-coin-dropdown", "value"),
    Input("data-store", "data"),
    Input("timeframe-select", "value"),
)
def update_technical(coin, data_json, timeframe):
    if not coin:
        return no_update
    content = _update_technical_content(coin, data_json, timeframe, resample_df)
    if content is None:
        return dbc.Alert("Not enough data points for technical analysis.", color="info")
    return content


# ─── Alert Management Callbacks ───────────────────────────────────────


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
                "time": datetime.now(PHNOM_PENH_TZ).strftime("%H:%M:%S"),
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
    Input("chart-type-select", "value"),
    Input("time-range-select", "value"),
    Input("model-select", "value"),
)
def render_page(pathname, data_json, sel_coins, timeframe, chart_type, time_range, model):
    gold, silver, coins, data = df_from_store(data_json)

    if not gold.empty:
        try:
            gold_rs = resample_df(gold, timeframe or "1m", "window_start")
        except Exception as e:
            logger.warning(f"Resample failed for {timeframe}: {e}")
            gold_rs = gold
    else:
        gold_rs = gold

    page = pathname.strip("/") if pathname else ""
    if page not in ["overview", "technical", "comparison", "predictions", "pipeline", "alerts", "sessions"]:
        page = "overview"

    if page == "overview":
        return make_overview(gold_rs, silver, sel_coins or [], data, timeframe,
                             chart_type or "line", time_range or "all")
    elif page == "technical":
        return make_technical(gold_rs, silver, sel_coins or [], data)
    elif page == "comparison":
        return make_comparison(gold_rs, silver, sel_coins or [], data)
    elif page == "predictions":
        return make_predictions(gold_rs, sel_coins or [], timeframe, model or "arima")
    elif page == "pipeline":
        return make_pipeline(gold, silver, coins, data)
    elif page == "alerts":
        return make_alerts(gold, silver, sel_coins or [], data)
    elif page == "sessions":
        return make_sessions()

    return html.Div(dbc.Alert("Page not found", color="danger"))


# ─── CSV Export ──────────────────────────────────────────────────────


@callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("url", "pathname"),
    State("data-store", "data"),
    State("coin-select", "value"),
    State("timeframe-select", "value"),
    State("time-range-select", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, pathname, data_json, sel_coins, timeframe, time_range):
    if not n_clicks:
        return no_update
    gold, silver, coins, data = df_from_store(data_json)
    use_gold = not gold.empty
    df = gold if use_gold else silver
    if df.empty:
        return no_update

    tc = "window_start" if use_gold else "fetched_at"

    coin_filter = df[df["coin_id"].isin(sel_coins or [])].sort_values(tc)
    if coin_filter.empty:
        return no_update

    csv_str = coin_filter.to_csv(index=False)
    return dcc.send_string(csv_str, f"crypto_data_{datetime.now(PHNOM_PENH_TZ).strftime('%Y%m%d_%H%M%S')}.csv")


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
