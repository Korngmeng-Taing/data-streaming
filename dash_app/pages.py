from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from config.logging_config import setup_logger
from dash_app.data_utils import df_from_store
from dash_app.charts import (
    build_main_price_chart, build_volume_chart, build_change_chart,
    build_distribution_chart, build_volatility_chart,
    build_technical_chart, build_normalized_chart, build_correlation_chart,
    build_volume_comparison_chart, build_returns_chart, build_predictions_chart,
)
from prediction import predict_prices, ARIMA_ORDER

logger = setup_logger("dash_app")


def _filter_time_range(df, tc, time_range):
    if time_range and time_range != "all" and not df.empty:
        try:
            minutes = int(time_range.replace("h", "")) * 60 if "h" in time_range else int(time_range.replace("m", ""))
            now = df[tc].max()
            cutoff = now - pd.Timedelta(minutes=minutes)
            df = df[df[tc] >= cutoff]
        except Exception:
            pass
    return df


def make_overview(gold, silver, sel_coins, data, timeframe="1m", chart_type="line", time_range="all"):
    use_gold = not gold.empty
    if use_gold:
        df = gold
        tc = "window_start"
        vc = "avg_price"
        voc = "avg_volume"
        cc = "avg_change_pct"
    elif not silver.empty:
        df = silver
        tc = "fetched_at"
        vc = "price_usd"
        voc = "volume_24h_usd"
        cc = "change_24h_pct"
    else:
        return html.Div(dbc.Alert("No data available yet.", color="warning"))
    if not sel_coins:
        return html.Div(dbc.Alert("Select coins from the sidebar to display.", color="info"))

    df = _filter_time_range(df, tc, time_range)
    coin_filter = df[df["coin_id"].isin(sel_coins)]
    if coin_filter.empty:
        return html.Div(dbc.Alert("No data for the selected time range.", color="info"))
    now_val = df[tc].max() if not df.empty else ""

    fig_main = build_main_price_chart(coin_filter, sel_coins, tc, vc, chart_type)
    fig_vol = build_volume_chart(coin_filter, sel_coins, tc, voc)
    fig_chg = build_change_chart(coin_filter, sel_coins, tc, cc)

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

    fig_dist = build_distribution_chart(coin_filter, sel_coins, vc)
    fig_vola = build_volatility_chart(coin_filter, sel_coins, tc, voc, use_gold)

    recent = coin_filter.sort_values(tc, ascending=False).head(25)
    if not recent.empty and tc in recent.columns:
        recent[tc] = recent[tc].astype(str)

    tf_label = {"1m": "1 min", "5m": "5 min", "15m": "15 min", "30m": "30 min", "1h": "1 hour"}.get(timeframe or "1m", timeframe)

    export_id = "export-csv-btn"
    recent_json = recent.head(20).to_json(date_format="iso", orient="records") if not recent.empty else "[]"

    return html.Div([
        html.H3("Overview", className="mb-3"),
        html.Small(f"Layer: {'Gold' if use_gold else 'Silver'} ({tf_label}) · "
                   f"{len(sel_coins)} coins · {now_val}", className="text-muted"),
        dbc.Row([
            dbc.Col([
                html.Label("Export", style={"font-size": "0.85rem", "color": "#aaa"}),
                dbc.Button("Download CSV", id=export_id, color="secondary", size="sm",
                           style={"font-size": "0.8rem"}),
                dcc.Download(id="download-csv"),
            ], xs=12, sm=6, md=3),
        ], className="mb-3"),
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
        html.Div(id="recent-data-json", style={"display": "none"}, children=recent_json),
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


def _get_auto_interpretation(price, cdf, last_price):
    from viz.utils import rsi, bollinger, sma

    # 1. RSI Interpretation
    rsi_val = rsi(price, 14).iloc[-1] if len(cdf) >= 14 else 50
    if rsi_val >= 70:
        rsi_signal = "Bearish (Overbought)"
        rsi_color = "danger"
        rsi_score = -1
        rsi_desc = f"RSI is at {rsi_val:.1f}, indicating the asset is overbought. A price correction or consolidation may occur."
    elif rsi_val <= 30:
        rsi_signal = "Bullish (Oversold)"
        rsi_color = "success"
        rsi_score = 1
        rsi_desc = f"RSI is at {rsi_val:.1f}, indicating the asset is oversold. A price bounce or trend reversal may be near."
    else:
        rsi_signal = "Neutral"
        rsi_color = "secondary"
        rsi_score = 0
        rsi_desc = f"RSI is at {rsi_val:.1f}, indicating standard momentum without overbought or oversold conditions."

    # 2. Bollinger Bands Interpretation
    if len(cdf) >= 20:
        middle, upper, lower = bollinger(price, 20, 2)
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        middle_val = middle.iloc[-1]

        if last_price >= upper_val * 0.98:
            bb_signal = "Bearish (Overextended)"
            bb_color = "danger"
            bb_score = -1
            bb_desc = f"Price (${last_price:.4f}) is trading near or above the upper Bollinger Band (${upper_val:.4f}), indicating overextended territory."
        elif last_price <= lower_val * 1.02:
            bb_signal = "Bullish (Undervalued)"
            bb_color = "success"
            bb_score = 1
            bb_desc = f"Price (${last_price:.4f}) is trading near or below the lower Bollinger Band (${lower_val:.4f}), indicating support/undervalued conditions."
        else:
            bb_signal = "Neutral"
            bb_color = "secondary"
            bb_score = 0
            bb_desc = f"Price (${last_price:.4f}) is stable within the bands, trading near the middle line (${middle_val:.4f})."
    else:
        bb_signal = "Neutral (No Data)"
        bb_color = "secondary"
        bb_score = 0
        bb_desc = "Insufficient data points to calculate Bollinger Bands."

    # 3. SMAs Interpretation
    if len(cdf) >= 20:
        sma10_val = sma(price, 10).iloc[-1]
        sma20_val = sma(price, 20).iloc[-1]

        if last_price > sma10_val and sma10_val > sma20_val:
            ma_signal = "Bullish (Uptrend)"
            ma_color = "success"
            ma_score = 1
            ma_desc = f"Price (${last_price:.4f}) is above SMA(10) (${sma10_val:.4f}) and SMA(20) (${sma20_val:.4f}) in an uptrend, signaling upward momentum."
        elif last_price < sma10_val and sma10_val < sma20_val:
            ma_signal = "Bearish (Downtrend)"
            ma_color = "danger"
            ma_score = -1
            ma_desc = f"Price (${last_price:.4f}) is below SMA(10) (${sma10_val:.4f}) and SMA(20) (${sma20_val:.4f}) in a downtrend, signaling downward momentum."
        else:
            ma_signal = "Neutral"
            ma_color = "secondary"
            ma_score = 0
            ma_desc = f"Price is trading between moving averages. SMA(10) is at ${sma10_val:.4f} and SMA(20) is at ${sma20_val:.4f}."
    else:
        ma_signal = "Neutral (No Data)"
        ma_color = "secondary"
        ma_score = 0
        ma_desc = "Insufficient data points to calculate moving averages."

    # 4. Overall Interpretation Score
    total_score = rsi_score + bb_score + ma_score
    if total_score >= 2:
        recommendation = "Strong Buy"
        rec_style = {"color": "#4CAF50", "fontWeight": "bold", "fontSize": "1.5rem"}
    elif total_score == 1:
        recommendation = "Buy"
        rec_style = {"color": "#81C784", "fontWeight": "bold", "fontSize": "1.4rem"}
    elif total_score == 0:
        recommendation = "Neutral"
        rec_style = {"color": "#E0E0E0", "fontWeight": "bold", "fontSize": "1.4rem"}
    elif total_score == -1:
        recommendation = "Sell"
        rec_style = {"color": "#E57373", "fontWeight": "bold", "fontSize": "1.4rem"}
    else:
        recommendation = "Strong Sell"
        rec_style = {"color": "#F44336", "fontWeight": "bold", "fontSize": "1.5rem"}

    return dbc.Card([
        dbc.CardHeader(html.H5("📊 Auto-Interpretation & Recommendations", className="mb-0 text-white")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Overall Signal:", className="text-muted mb-1"),
                    html.Span(recommendation, style=rec_style),
                    html.P(f"Signal Score: {total_score:+d}", className="text-muted mt-2 mb-0 small"),
                    html.P("Based on RSI, Bollinger Bands, and SMA", className="text-muted small", style={"font-size": "0.75rem"}),
                ], xs=12, md=4, className="border-end border-secondary d-flex flex-column justify-content-center align-items-center text-center"),
                
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Strong("Relative Strength Index (RSI):"),
                            dbc.Badge(rsi_signal, color=rsi_color, className="ms-2"),
                        ], className="d-flex align-items-center justify-content-between mb-1"),
                        html.P(rsi_desc, className="small text-muted mb-3"),
                        
                        html.Div([
                            html.Strong("Bollinger Bands (BB):"),
                            dbc.Badge(bb_signal, color=bb_color, className="ms-2"),
                        ], className="d-flex align-items-center justify-content-between mb-1"),
                        html.P(bb_desc, className="small text-muted mb-3"),
                        
                        html.Div([
                            html.Strong("Moving Averages (SMA):"),
                            dbc.Badge(ma_signal, color=ma_color, className="ms-2"),
                        ], className="d-flex align-items-center justify-content-between mb-1"),
                        html.P(ma_desc, className="small text-muted mb-0"),
                    ])
                ], xs=12, md=8, className="ps-md-4 mt-3 mt-md-0"),
            ])
        ])
    ], color="dark", outline=True, className="mb-4 shadow")


def _update_technical_content(coin, data_json, timeframe, resample_df):
    if not coin:
        return None
    gold, silver, coins, data = df_from_store(data_json)
    use_gold = not gold.empty
    if use_gold:
        from dash_app.data_utils import resample_df as _resample
        try:
            gold = _resample(gold, timeframe or "1m", "window_start")
        except Exception as e:
            logger.warning(f"Technical: resample failed for {timeframe}: {e}")
    df = gold if use_gold else silver
    if df.empty:
        return None

    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"
    voc = "avg_volume" if use_gold else "volume_24h_usd"
    cc = "avg_change_pct" if use_gold else "change_24h_pct"

    cdf = df[df["coin_id"] == coin].sort_values(tc).copy()
    if len(cdf) < 5:
        return None

    from viz.utils import rsi, bollinger
    price = cdf[vc]
    fig = build_technical_chart(cdf, coin, tc, vc, voc, cc)

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

    interpretation_card = _get_auto_interpretation(price, cdf, last_price)

    return html.Div([cards, interpretation_card, dcc.Graph(figure=fig)])


def make_comparison(gold, silver, sel_coins, data):
    use_gold = not gold.empty
    df = gold if use_gold else silver
    if df.empty:
        return html.Div(dbc.Alert("No data available.", color="warning"))

    tc = "window_start" if use_gold else "fetched_at"
    vc = "avg_price" if use_gold else "price_usd"
    voc = "avg_volume" if use_gold else ("avg_volume" if "avg_volume" in df.columns else "volume_24h_usd")
    coin_filter = df[df["coin_id"].isin(sel_coins)]

    if len(sel_coins) < 2:
        return html.Div(dbc.Alert("Select at least 2 coins in the sidebar for comparison.", color="info"))

    fig_norm = build_normalized_chart(coin_filter, sel_coins, tc, vc)

    pivot = coin_filter.pivot_table(
        index=coin_filter.groupby("coin_id").cumcount(),
        columns="coin_id", values=vc,
    )
    fig_corr = build_correlation_chart(pivot)

    vol_pivot = coin_filter.pivot_table(
        index=coin_filter.groupby("coin_id").cumcount(),
        columns="coin_id", values=voc,
    )
    fig_vc = build_volume_comparison_chart(vol_pivot)

    returns_data = []
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if len(cdf) >= 5:
            ret = cdf[vc].pct_change().dropna().tail(50)
            for val in ret:
                returns_data.append({"coin": coin.upper(), "return_pct": val * 100})
    fig_ret = build_returns_chart(returns_data)

    return html.Div([
        html.H3("Multi-Coin Comparison"),
        dcc.Graph(figure=fig_norm, className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_corr), xs=12, md=6),
            dbc.Col(dcc.Graph(figure=fig_vc), xs=12, md=6),
        ], className="mb-4"),
        dcc.Graph(figure=fig_ret, className="mb-4"),
    ])


def make_predictions(gold, sel_coins, timeframe, model="arima"):
    if gold.empty:
        return html.Div(dbc.Alert("No historical data available for predictions.", color="warning"))

    coins = sel_coins if sel_coins else gold["coin_id"].unique().tolist()
    gold_filtered = gold[gold["coin_id"].isin(coins)].copy()

    forecasts = predict_prices(gold_filtered, model=model)

    if not forecasts:
        return html.Div(dbc.Alert(
            "Could not generate predictions. Please select at least one coin with data.",
            color="warning",
        ))

    fig = build_predictions_chart(forecasts, gold_filtered)

    stats = []

    for i, (coin, fcast) in enumerate(forecasts.items()):
        hist = gold_filtered[gold_filtered["coin_id"] == coin].sort_values("window_start")
        last_price = hist["avg_price"].iloc[-1]
        last_pred = fcast["predicted_price"].iloc[-1]
        change_pct = ((last_pred - last_price) / last_price) * 100
        stats.append(
            dbc.Card(dbc.CardBody([
                html.H6(coin, className="card-title"),
                html.P(f"Last: ${last_price:.2f}", className="mb-1"),
                html.P(f"Forecast: ${last_pred:.2f}", className="mb-1"),
                html.P(
                    f"Change: {change_pct:+.2f}%",
                    style={"color": "green" if change_pct >= 0 else "red", "font-weight": "bold"},
                ),
            ]), className="mb-2")
        )
        stats.append(dbc.Card(dbc.CardBody([
            html.H6(f"{coin} - Next {len(fcast)} Steps", className="card-title"),
            html.Table(
                [html.Tr([html.Th("Step"), html.Th("Timestamp"), html.Th("Price")])] +
                [
                    html.Tr([
                        html.Td(str(j + 1)),
                        html.Td(str(row["window_start"])),
                        html.Td(f"${row['predicted_price']:.2f}"),
                    ])
                    for j, (_, row) in enumerate(fcast.iterrows())
                ],
                className="table table-sm table-dark",
                style={"font-size": "0.8rem"},
            ),
        ]), className="mb-3"))

    model_label = "Prophet (seasonal)" if model == "prophet" else f"ARIMA{ARIMA_ORDER}"

    return html.Div([
        html.H3("Price Prediction", className="mb-3"),
        html.P(
            f"Using {model_label} per coin with {len(next(iter(forecasts.values())))}-step forecast.",
            style={"color": "#aaa"},
        ),
        dbc.Row([dbc.Col(card, xs=12, md=4) for card in stats], className="mb-4"),
        dcc.Graph(figure=fig, className="mb-4"),
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
