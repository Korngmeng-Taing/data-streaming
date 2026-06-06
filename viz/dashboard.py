import os
import time
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.logging_config import setup_logger
from viz.utils import sma, bollinger, rsi
from ws_gateway.client import get_last_update

logger = setup_logger("dashboard")

st.set_page_config(page_title="Crypto Pipeline Dashboard", layout="wide", page_icon="\U0001F4CA")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
GOLD_PATH = f"{OUTPUT_PATH}/gold"
SILVER_PATH = f"{OUTPUT_PATH}/silver"

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return pd.DataFrame()


def load_pipeline_data() -> dict:
    if "last_ws_ts" not in st.session_state:
        st.session_state.last_ws_ts = 0.0
        st.session_state.gold = pd.DataFrame()
        st.session_state.silver = pd.DataFrame()

    ws_ts = get_last_update()
    if ws_ts is not None and ws_ts <= st.session_state.last_ws_ts:
        gold = st.session_state.gold
        silver = st.session_state.silver
    else:
        gold = load_data(GOLD_PATH)
        silver = load_data(SILVER_PATH)
        st.session_state.gold = gold
        st.session_state.silver = silver
        st.session_state.last_ws_ts = ws_ts or time.time()
    return {"gold": gold, "silver": silver}


data = load_pipeline_data()
gold = data["gold"]
silver = data["silver"]

if not gold.empty:
    df = gold
    is_silver = False
    time_col = "window_start"
    value_col = "avg_price"
    volume_col = "avg_volume"
    change_col = "avg_change_pct"
    extra_cols = ["min_price", "max_price", "price_volatility", "record_count"]
    layer_label = "Gold (5-min windows)"
elif not silver.empty:
    df = silver
    is_silver = True
    time_col = "fetched_at"
    value_col = "price_usd"
    volume_col = "volume_24h_usd"
    change_col = "change_24h_pct"
    extra_cols = ["market_cap_usd"]
    layer_label = "Silver (raw ticks)"
else:
    st.warning("No data yet. Pipeline is starting up — check back in 30 seconds.")
    st.stop()

coins = sorted(df["coin_id"].unique())

# ─── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/64px-Bitcoin.svg.png",
    width=32,
)
st.sidebar.title("Crypto Pipeline")
st.sidebar.caption("Real-time streaming analytics")

st.sidebar.divider()

st.sidebar.subheader("Data Controls")
data_source = st.sidebar.selectbox("Layer", ["Gold (aggregated)", "Silver (raw)"], index=0)
tab_selection = st.sidebar.radio(
    "View",
    ["Overview", "Technical Analysis", "Comparison", "Pipeline Status"],
    index=0,
)

if data_source == "Gold (aggregated)" and not gold.empty:
    df = gold
    is_silver = False
    time_col = "window_start"
    value_col = "avg_price"
    volume_col = "avg_volume"
    change_col = "avg_change_pct"
    extra_cols = ["min_price", "max_price", "price_volatility", "record_count"]
    layer_label = "Gold"
elif data_source == "Silver (raw)" and not silver.empty:
    df = silver
    is_silver = True
    time_col = "fetched_at"
    value_col = "price_usd"
    volume_col = "volume_24h_usd"
    change_col = "change_24h_pct"
    extra_cols = ["market_cap_usd"]
    layer_label = "Silver"
else:
    fallback_is_silver = gold.empty and not silver.empty
    df = silver if fallback_is_silver else gold
    is_silver = fallback_is_silver
    if is_silver:
        time_col = "fetched_at"; value_col = "price_usd"; volume_col = "volume_24h_usd"; change_col = "change_24h_pct"
        extra_cols = ["market_cap_usd"]; layer_label = "Silver"
    else:
        time_col = "window_start"; value_col = "avg_price"; volume_col = "avg_volume"; change_col = "avg_change_pct"
        extra_cols = ["min_price", "max_price", "price_volatility", "record_count"]; layer_label = "Gold"

for c in df.select_dtypes(include=["object"]):
    if c in (time_col,):
        df[c] = pd.to_datetime(df[c], errors="coerce")
for c in [value_col, volume_col, change_col] + extra_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[value_col])
df = df.sort_values(time_col)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (s)", 5, 120, 15, disabled=not auto_refresh)
if auto_refresh:
    st.sidebar.caption(f"Auto-refresh every {refresh_interval}s")

coins_in_data = sorted(df["coin_id"].unique())
selected_coins = st.sidebar.multiselect(
    "Coins", coins_in_data, default=coins_in_data[: min(3, len(coins_in_data))]
)

if not selected_coins:
    st.info("Select at least one coin in the sidebar.")
    st.stop()

coin_filter = df[df["coin_id"].isin(selected_coins)]
now = df[time_col].max()
time_range = st.sidebar.selectbox(
    "Time range",
    ["All", "Last 10 points", "Last 25 points", "Last 50 points", "Last 100 points"],
    index=2,
)
if time_range != "All":
    n = int(time_range.split()[1])
    coin_filter = coin_filter.groupby("coin_id").apply(lambda x: x.tail(n)).reset_index(drop=True)

st.sidebar.divider()
st.sidebar.caption(f"**{layer_label}** · {len(coin_filter)} rows · {len(selected_coins)} coins")

# ─── Theme ────────────────────────────────────────────────────────────
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
template = "plotly_dark" if theme == "Dark" else "plotly_white"
bg_color = "#0E1117" if theme == "Dark" else "#FFFFFF"
card_bg = "#1E1E1E" if theme == "Dark" else "#F0F2F6"
text_color = "#FAFAFA" if theme == "Dark" else "#31333F"

# ─── Helpers ──────────────────────────────────────────────────────────

def style_metric(label: str, value: str, delta: Optional[str] = None):
    st.markdown(
        f"""
        <div style="background:{card_bg};padding:16px;border-radius:10px;text-align:center;border:1px solid #333;">
            <div style="color:#888;font-size:0.8rem;margin-bottom:4px;">{label}</div>
            <div style="color:{text_color};font-size:1.6rem;font-weight:700;">{value}</div>
            {f'<div style="color:#4CAF50;font-size:0.85rem;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if tab_selection == "Overview":
    st.title("Crypto Pipeline Dashboard")
    st.caption(f"Layer: **{layer_label}** · {len(selected_coins)} coins tracked · Last updated: {now}")

    # Global metrics row
    metrics_cols = st.columns(len(selected_coins))
    for i, coin in enumerate(selected_coins):
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
        if cdf.empty:
            continue
        last = cdf.iloc[-1]
        prev = cdf.iloc[-2] if len(cdf) > 1 else last
        pct_change = ((last[value_col] - prev[value_col]) / prev[value_col] * 100) if prev[value_col] != 0 else 0
        delta_str = f"{pct_change:+.2f}%"
        with metrics_cols[i]:
            style_metric(
                coin.upper(),
                f"${last[value_col]:.4f}",
                delta_str,
            )

    st.divider()

    # Main price chart with multi-coin overlay
    fig_main = go.Figure()
    for coin in selected_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
        if cdf.empty:
            continue
        fig_main.add_trace(go.Scatter(
            x=cdf[time_col],
            y=cdf[value_col],
            mode="lines",
            name=coin.upper(),
            line=dict(width=2),
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:.4f}}<extra></extra>",
        ))

    fig_main.update_layout(
        title=f"Crypto Prices — {layer_label}",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template=template,
        hovermode="x unified",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Secondary charts row
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Volume")
        fig_vol = go.Figure()
        for coin in selected_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
            if cdf.empty:
                continue
            fig_vol.add_trace(go.Bar(
                x=cdf[time_col],
                y=cdf[volume_col],
                name=coin.upper(),
                opacity=0.75,
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>",
            ))
        fig_vol.update_layout(
            template=template,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            barmode="group",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_r:
        st.subheader("24h Change %")
        fig_chg = go.Figure()
        for coin in selected_coins:
            cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
            if cdf.empty:
                continue
            colors = ["#4CAF50" if v >= 0 else "#F44336" for v in cdf[change_col]]
            fig_chg.add_trace(go.Bar(
                x=cdf[time_col],
                y=cdf[change_col],
                name=coin.upper(),
                marker_color=colors,
                opacity=0.75,
                hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
            ))
        fig_chg.update_layout(
            template=template,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            barmode="group",
            hovermode="x unified",
            showlegend=False,
            yaxis_title="%",
        )
        st.plotly_chart(fig_chg, use_container_width=True)

    # Distribution and volatility
    if not is_silver:
        col3l, col3r = st.columns(2)
        with col3l:
            st.subheader("Price Distribution")
            fig_dist = go.Figure()
            for coin in selected_coins:
                cdf = coin_filter[coin_filter["coin_id"] == coin][value_col].dropna()
                if cdf.empty:
                    continue
                fig_dist.add_trace(go.Box(
                    y=cdf,
                    name=coin.upper(),
                    boxmean="sd",
                    hovertemplate=f"<b>{coin.upper()}</b><br>$%{{y:.4f}}<extra></extra>",
                ))
            fig_dist.update_layout(
                template=template,
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Price (USD)",
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col3r:
            st.subheader("Volatility Trend")
            fig_vola = go.Figure()
            for coin in selected_coins:
                cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
                if cdf.empty or "price_volatility" not in cdf.columns:
                    continue
                fig_vola.add_trace(go.Scatter(
                    x=cdf[time_col],
                    y=cdf["price_volatility"],
                    mode="lines",
                    name=coin.upper(),
                    line=dict(width=2),
                    hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
                ))
            fig_vola.update_layout(
                template=template,
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig_vola, use_container_width=True)

    st.divider()
    st.subheader("Recent Data")
    st.dataframe(
        coin_filter.sort_values(time_col, ascending=False).head(25),
        use_container_width=True,
        hide_index=True,
    )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif tab_selection == "Technical Analysis":
    st.title("Technical Analysis")
    ta_coin = st.selectbox("Coin", selected_coins, key="ta_coin")
    cdf = coin_filter[coin_filter["coin_id"] == ta_coin].sort_values(time_col).copy()

    if len(cdf) < 5:
        st.warning("Not enough data points for technical analysis.")
        st.stop()

    price = cdf[value_col]
    sma_windows = st.multiselect(
        "Moving Averages",
        [5, 10, 20, 50],
        default=[10, 20],
        key="sma_windows",
    )
    show_bollinger = st.checkbox("Bollinger Bands (20,2)", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_volume_ta = st.checkbox("Volume", value=True)

    n_secondary = sum([show_rsi, show_volume_ta])
    row_heights = [0.5] + [0.25] * n_secondary
    specs = [[{"secondary_y": False}]]
    for _ in range(n_secondary):
        specs.append([{"secondary_y": False}])

    fig_ta = make_subplots(
        rows=1 + n_secondary,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=["Price & Indicators"] + (
            ["RSI"] if show_rsi else []
        ) + (
            ["Volume"] if show_volume_ta else []
        ),
    )

    row = 1
    fig_ta.add_trace(
        go.Candlestick(
            x=cdf[time_col],
            open=cdf.get("min_price", price),
            high=cdf.get("max_price", price),
            low=cdf.get("min_price", price * 0.99 if "min_price" not in cdf.columns else cdf["min_price"]),
            close=price,
            name=ta_coin.upper(),
            increasing_line_color="#4CAF50",
            decreasing_line_color="#F44336",
        ),
        row=row, col=1,
    )

    for w in sma_windows:
        if len(cdf) >= w:
            fig_ta.add_trace(
                go.Scatter(
                    x=cdf[time_col],
                    y=sma(price, w),
                    mode="lines",
                    name=f"SMA({w})",
                    line=dict(width=1.5),
                ),
                row=row, col=1,
            )

    if show_bollinger and len(cdf) >= 20:
        middle, upper, lower = bollinger(price, 20, 2)
        fig_ta.add_trace(
            go.Scatter(x=cdf[time_col], y=upper, mode="lines", name="BB Upper",
                       line=dict(width=1, color="#888", dash="dash"), showlegend=True),
            row=row, col=1,
        )
        fig_ta.add_trace(
            go.Scatter(x=cdf[time_col], y=lower, mode="lines", name="BB Lower",
                       line=dict(width=1, color="#888", dash="dash"), fill="tonexty",
                       fillcolor="rgba(128,128,128,0.1)", showlegend=True),
            row=row, col=1,
        )

    fig_ta.update_yaxes(title_text="Price (USD)", row=row, col=1)

    row += 1
    if show_rsi and len(cdf) >= 14:
        rsi_vals = rsi(price, 14)
        fig_ta.add_trace(
            go.Scatter(x=cdf[time_col], y=rsi_vals, mode="lines", name="RSI (14)",
                       line=dict(color="#FF9800", width=2)),
            row=row, col=1,
        )
        fig_ta.add_hline(y=70, line_width=1, line_dash="dash", line_color="#F44336", row=row, col=1)
        fig_ta.add_hline(y=30, line_width=1, line_dash="dash", line_color="#4CAF50", row=row, col=1)
        fig_ta.add_hline(y=50, line_width=1, line_dash="dot", line_color="#888", row=row, col=1)
        fig_ta.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)

    row += 1
    if show_volume_ta:
        vol_colors = ["#4CAF50" if i >= 0 else "#F44336" for i in cdf[change_col].fillna(0)]
        fig_ta.add_trace(
            go.Bar(x=cdf[time_col], y=cdf[volume_col], name="Volume",
                   marker_color=vol_colors, opacity=0.6),
            row=row, col=1,
        )
        fig_ta.update_yaxes(title_text="Volume", row=row, col=1)

    fig_ta.update_layout(
        template=template,
        height=200 + 300 * (1 + n_secondary),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ta, use_container_width=True)

    # Metrics row for TA
    ta_cols = st.columns(4)
    last_price = price.iloc[-1]
    with ta_cols[0]:
        style_metric("Current", f"${last_price:.4f}")
    with ta_cols[1]:
        rsi_val = rsi(price, 14).iloc[-1] if len(cdf) >= 14 else 50
        rsi_color = "#4CAF50" if 30 < rsi_val < 70 else "#F44336"
        st.markdown(
            f"""<div style="background:{card_bg};padding:16px;border-radius:10px;text-align:center;border:1px solid #333;">
                <div style="color:#888;font-size:0.8rem;">RSI (14)</div>
                <div style="color:{rsi_color};font-size:1.6rem;font-weight:700;">{rsi_val:.1f}</div>
                <div style="color:#888;font-size:0.75rem;">{'Neutral' if 30 < rsi_val < 70 else 'Overbought' if rsi_val >= 70 else 'Oversold'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with ta_cols[2]:
        bb_width = (bollinger(price)[1] - bollinger(price)[2]).iloc[-1] if len(cdf) >= 20 else 0
        style_metric("BB Width", f"${bb_width:.4f}")
    with ta_cols[3]:
        price_chg_24h = ((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100) if len(cdf) > 1 else 0
        chg_color = "#4CAF50" if price_chg_24h >= 0 else "#F44336"
        st.markdown(
            f"""<div style="background:{card_bg};padding:16px;border-radius:10px;text-align:center;border:1px solid #333;">
                <div style="color:#888;font-size:0.8rem;">Period Change</div>
                <div style="color:{chg_color};font-size:1.6rem;font-weight:700;">{price_chg_24h:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: COMPARISON
# ═══════════════════════════════════════════════════════════════════════
elif tab_selection == "Comparison":
    st.title("Multi-Coin Comparison")

    if len(selected_coins) < 2:
        st.info("Select at least 2 coins in the sidebar for comparison.")
        st.stop()

    # Normalized price overlay
    st.subheader("Normalized Price (base=100)")
    fig_norm = go.Figure()
    for coin in selected_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
        if len(cdf) < 2:
            continue
        base = cdf[value_col].iloc[0]
        if base == 0:
            continue
        norm = cdf[value_col] / base * 100
        fig_norm.add_trace(go.Scatter(
            x=cdf[time_col],
            y=norm,
            mode="lines",
            name=coin.upper(),
            line=dict(width=2),
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig_norm.update_layout(
        template=template,
        height=400,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Normalized Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Correlation Matrix")
        pivot = coin_filter.pivot_table(
            index=coin_filter.groupby("coin_id").cumcount(),
            columns="coin_id",
            values=value_col,
        )
        if len(pivot.columns) >= 2 and len(pivot) >= 2:
            corr = pivot.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
                aspect="auto",
                template=template,
            )
            fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need more data for correlation.")

    with col_c2:
        st.subheader("Volume Comparison")
        vol_pivot = coin_filter.pivot_table(
            index=coin_filter.groupby("coin_id").cumcount(),
            columns="coin_id",
            values=volume_col,
        )
        if len(vol_pivot.columns) >= 2:
            fig_vc = px.bar(
                vol_pivot.sum().reset_index(),
                x="coin_id",
                y=0,
                color="coin_id",
                template=template,
                labels={"coin_id": "", "0": "Total Volume"},
            )
            fig_vc.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_vc, use_container_width=True)

    # Returns scatter
    st.subheader("Returns Comparison")
    returns_data = []
    for coin in selected_coins:
        cdf = coin_filter[coin_filter["coin_id"] == coin].sort_values(time_col)
        if len(cdf) >= 5:
            ret = cdf[value_col].pct_change().dropna().tail(50)
            for i, r in enumerate(ret):
                returns_data.append({"coin": coin.upper(), "return_pct": r * 100, "index": i})
    if returns_data:
        ret_df = pd.DataFrame(returns_data)
        fig_ret = px.box(
            ret_df, x="coin", y="return_pct", color="coin",
            template=template, points="all",
            labels={"coin": "", "return_pct": "Return %"},
        )
        fig_ret.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_ret, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: PIPELINE STATUS
# ═══════════════════════════════════════════════════════════════════════
elif tab_selection == "Pipeline Status":
    st.title("Pipeline Status")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        style_metric("Gold Records", f"{len(gold):,}" if not gold.empty else "0")
    with col_s2:
        style_metric("Silver Records", f"{len(silver):,}" if not silver.empty else "0")
    with col_s3:
        style_metric("Coins Tracked", str(len(coins)))
    with col_s4:
        style_metric("Data Layer", layer_label)

    st.divider()

    st.subheader("Per-Coin Stats")
    stats_rows = []
    for coin in coins:
        cdf = df[df["coin_id"] == coin]
        if cdf.empty:
            continue
        stats_rows.append({
            "coin": coin.upper(),
            "records": len(cdf),
            "latest_price": cdf[value_col].iloc[-1] if not cdf.empty else 0,
            "avg_price": cdf[value_col].mean(),
            "min_price": cdf[value_col].min(),
            "max_price": cdf[value_col].max(),
            "total_volume": cdf[volume_col].sum(),
            "price_range": cdf[value_col].max() - cdf[value_col].min(),
        })
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows).sort_values("records", ascending=False)
        stats_df = stats_df.round(4)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Data Freshness")
    freshness_cols = st.columns(2)
    with freshness_cols[0]:
        if not gold.empty:
            st.metric("Oldest Gold Record", gold[time_col].min() if time_col in gold.columns else "N/A")
        if not silver.empty:
            st.metric("Oldest Silver Record", silver[time_col].min() if time_col in silver.columns else "N/A")
    with freshness_cols[1]:
        if not gold.empty:
            st.metric("Newest Gold Record", gold[time_col].max() if time_col in gold.columns else "N/A")
        if not silver.empty:
            st.metric("Newest Silver Record", silver[time_col].max() if time_col in silver.columns else "N/A")

    if not is_silver:
        st.divider()
        st.subheader("Latest Gold Windows")
        latest_gold = gold.sort_values("window_start", ascending=False).head(10)
        st.dataframe(latest_gold, use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
