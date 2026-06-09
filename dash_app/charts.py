import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz.utils import bollinger, rsi, sma


def build_main_price_chart(coin_filter, sel_coins, tc, vc, chart_type="line"):
    fig = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty:
            continue
        if chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=cdf[tc],
                    y=cdf[vc],
                    name=coin.upper(),
                    opacity=0.75,
                    hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:.4f}}<extra></extra>",
                )
            )
        elif chart_type == "candlestick" and len(sel_coins) == 1:
            import numpy as np

            open_series = cdf[vc].shift(1).fillna(cdf[vc])
            close_series = cdf[vc]
            high_series = np.maximum(
                cdf.get("max_price", cdf[vc]), np.maximum(open_series, close_series)
            )
            low_series = np.minimum(
                cdf.get("min_price", cdf[vc]), np.minimum(open_series, close_series)
            )
            fig.add_trace(
                go.Candlestick(
                    x=cdf[tc],
                    open=open_series,
                    high=high_series,
                    low=low_series,
                    close=close_series,
                    name=coin.upper(),
                    increasing_line_color="#4CAF50",
                    decreasing_line_color="#F44336",
                    increasing_fillcolor="#4CAF50",
                    decreasing_fillcolor="#F44336",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=cdf[tc],
                    y=cdf[vc],
                    mode="lines",
                    name=coin.upper(),
                    line=dict(width=2),
                    hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:.4f}}<extra></extra>",
                )
            )
    fig.update_layout(
        title=f"Crypto Prices ({chart_type})",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )
    return fig


def build_volume_chart(coin_filter, sel_coins, tc, voc):
    fig = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty or voc not in cdf.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=cdf[tc],
                y=cdf[voc],
                name=coin.upper(),
                opacity=0.75,
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Volume",
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode="group",
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )
    return fig


def build_change_chart(coin_filter, sel_coins, tc, cc):
    fig = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if cdf.empty or cc not in cdf.columns:
            continue
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in cdf[cc]]
        fig.add_trace(
            go.Bar(
                x=cdf[tc],
                y=cdf[cc],
                name=coin.upper(),
                marker_color=colors,
                opacity=0.75,
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="24h Change %",
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode="group",
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )
    return fig


def build_distribution_chart(coin_filter, sel_coins, vc):
    fig = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin][vc].dropna()
        if cdf.empty:
            continue
        fig.add_trace(go.Box(y=cdf, name=coin.upper(), boxmean="sd"))
    fig.update_layout(
        title="Price Distribution",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        showlegend=False,
    )
    return fig


def build_volatility_chart(coin_filter, sel_coins, tc, voc, use_gold):
    fig = go.Figure()
    if not use_gold:
        for coin in sel_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
            if cdf.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=cdf[tc],
                    y=cdf[voc] if voc in cdf.columns else [],
                    mode="lines",
                    name=coin.upper(),
                )
            )
    else:
        for coin in sel_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
            if cdf.empty or "price_volatility" not in cdf.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=cdf[tc],
                    y=cdf["price_volatility"],
                    mode="lines",
                    name=coin.upper(),
                    line=dict(width=2),
                    hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
                )
            )
    fig.update_layout(
        title="Price Volatility",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        showlegend=False,
    )
    return fig


def build_technical_chart(cdf, coin, tc, vc, voc, cc):
    price = cdf[vc]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price & Indicators", "RSI (14)", "Volume"],
    )

    import numpy as np

    open_series = price.shift(1).fillna(price)
    close_series = price
    high_series = np.maximum(
        cdf.get("max_price", price), np.maximum(open_series, close_series)
    )
    low_series = np.minimum(
        cdf.get("min_price", price), np.minimum(open_series, close_series)
    )

    fig.add_trace(
        go.Candlestick(
            x=cdf[tc],
            open=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name=coin.upper(),
            increasing_line_color="#4CAF50",
            decreasing_line_color="#F44336",
            increasing_fillcolor="#4CAF50",
            decreasing_fillcolor="#F44336",
        ),
        row=1,
        col=1,
    )
    for w in [10, 20]:
        if len(cdf) >= w:
            fig.add_trace(
                go.Scatter(
                    x=cdf[tc],
                    y=sma(price, w),
                    mode="lines",
                    name=f"SMA({w})",
                    line=dict(width=1.5),
                ),
                row=1,
                col=1,
            )
    if len(cdf) >= 20:
        mid, upper, lower = bollinger(price, 20, 2)
        fig.add_trace(
            go.Scatter(
                x=cdf[tc],
                y=upper,
                mode="lines",
                name="BB Upper",
                line=dict(width=1, color="#888", dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cdf[tc],
                y=lower,
                mode="lines",
                name="BB Lower",
                line=dict(width=1, color="#888", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1,
            col=1,
        )

    if len(cdf) >= 14:
        rsi_vals = rsi(price, 14)
        fig.add_trace(
            go.Scatter(
                x=cdf[tc],
                y=rsi_vals,
                mode="lines",
                name="RSI (14)",
                line=dict(color="#FF9800", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=70, line_width=1, line_dash="dash", line_color="#F44336", row=2, col=1
        )
        fig.add_hline(
            y=30, line_width=1, line_dash="dash", line_color="#4CAF50", row=2, col=1
        )
        fig.add_hline(
            y=50, line_width=1, line_dash="dot", line_color="#888", row=2, col=1
        )

    vol_colors = ["#4CAF50" if i >= 0 else "#F44336" for i in cdf[cc].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=cdf[tc], y=cdf[voc], name="Volume", marker_color=vol_colors, opacity=0.6
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"{coin.upper()} Technical Analysis",
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    return fig


def build_normalized_chart(coin_filter, sel_coins, tc, vc):
    fig = go.Figure()
    for coin in sel_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(tc)
        if len(cdf) < 2:
            continue
        base = cdf[vc].iloc[0]
        if base == 0:
            continue
        norm = cdf[vc] / base * 100
        fig.add_trace(
            go.Scatter(
                x=cdf[tc],
                y=norm,
                mode="lines",
                name=coin.upper(),
                line=dict(width=2),
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Normalized Price (base=100)",
        template="plotly_dark",
        hovermode="x unified",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Normalized Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
    )
    return fig


def build_correlation_chart(pivot):
    fig = go.Figure()
    if len(pivot.columns) >= 2 and len(pivot) >= 2:
        corr = pivot.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            aspect="auto",
        )
        fig.update_layout(
            title="Correlation",
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
        )
    return fig


def build_volume_comparison_chart(vol_pivot):
    fig = go.Figure()
    if not vol_pivot.empty and len(vol_pivot.columns) >= 1:
        fig = px.bar(
            vol_pivot.sum().reset_index(),
            x="coin_id",
            y=0,
            color="coin_id",
            template="plotly_dark",
            labels={"coin_id": "", "0": "Total Volume"},
        )
        fig.update_layout(
            title="Volume Comparison",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
        )
    return fig


def build_returns_chart(returns_data):
    fig = go.Figure()
    if returns_data:
        ret_df = pd.DataFrame(returns_data)
        fig = px.box(
            ret_df,
            x="coin",
            y="return_pct",
            color="coin",
            template="plotly_dark",
            points="all",
            labels={"coin": "", "return_pct": "Return %"},
        )
        fig.update_layout(
            title="Returns Distribution",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
        )
    return fig


def build_predictions_chart(forecasts, gold_filtered):
    fig = go.Figure()
    color_cycle = px.colors.qualitative.Plotly

    for i, (coin, fcast) in enumerate(forecasts.items()):
        c = color_cycle[i % len(color_cycle)]
        hist = gold_filtered[gold_filtered["coin_id"] == coin].sort_values(
            "window_start"
        )
        fig.add_trace(
            go.Scatter(
                x=hist["window_start"],
                y=hist["avg_price"],
                mode="lines",
                name=f"{coin} (actual)",
                line=dict(color=c, width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fcast["window_start"],
                y=fcast["predicted_price"],
                mode="lines+markers",
                name=f"{coin} (forecast)",
                line=dict(color=c, width=2.5, dash="dash"),
                marker=dict(size=6, symbol="circle"),
            )
        )

    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=500,
    )
    return fig
