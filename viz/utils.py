import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def bollinger(series: pd.Series, window: int = 20, num_std: int = 2):
    middle = sma(series, window)
    std = series.rolling(window=window).std()
    return middle, middle + num_std * std, middle - num_std * std


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_candlestick_fig(df, time_col, value_col, extra_cols, template, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    price = df[value_col]
    n_secondary = 2
    fig = make_subplots(
        rows=1 + n_secondary, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.25] * n_secondary,
        subplot_titles=[title, "RSI", "Volume"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df[time_col], open=df.get("min_price", price),
            high=df.get("max_price", price),
            low=df.get("min_price", price * 0.99 if "min_price" not in df.columns else df["min_price"]),
            close=price, name="Price",
            increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
        ), row=1, col=1,
    )

    if len(df) >= 14:
        rsi_vals = rsi(price, 14)
        fig.add_trace(
            go.Scatter(x=df[time_col], y=rsi_vals, mode="lines", name="RSI (14)",
                       line=dict(color="#FF9800", width=2)), row=2, col=1,
        )
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#F44336", row=2, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#4CAF50", row=2, col=1)
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="#888", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    fig.add_trace(
        go.Bar(x=df[time_col], y=df.get("avg_volume", df.get("volume_24h_usd", pd.Series(0, index=df.index))),
               name="Volume", marker_color="#90CAF9", opacity=0.6), row=3, col=1,
    )
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    fig.update_layout(
        template=template, height=200 + 300 * (1 + n_secondary),
        hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
