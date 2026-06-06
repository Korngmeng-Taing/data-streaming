import os
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.logging_config import setup_logger
from ml.features import build_features
from ml.model import load_model
from ws_gateway.client import get_last_update

logger = setup_logger("app")

st.set_page_config(page_title="Crypto Predictor & Alerts", page_icon="", layout="wide")

MODEL_PATH = os.getenv("ML_MODEL_PATH", os.getenv("MODEL_PATH", "/tmp/crypto-model")) + "/model_1m.joblib"
GOLD_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh") + "/gold"
SILVER_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh") + "/silver"

# ─── Model Loading ────────────────────────────────────────────────────

@st.cache_resource
def load_prediction_model():
    try:
        model, feature_cols, metrics = load_model(MODEL_PATH)
        return model, feature_cols, metrics
    except FileNotFoundError:
        return None, None, None


model, feature_cols, metrics = load_prediction_model()

if model is None:
    st.warning("No trained model found. The ML training container is still initializing — check back soon.")
    st.info("The model will be trained automatically once enough gold data accumulates (~2 minutes after pipeline start).")
    st.stop()

# ─── Data Loading ─────────────────────────────────────────────────────

if "last_ws_ts" not in st.session_state:
    st.session_state.last_ws_ts = 0.0
    st.session_state.gold = pd.DataFrame()
    st.session_state.silver = pd.DataFrame()


def load_gold_data():
    try:
        return pd.read_parquet(GOLD_PATH)
    except Exception as e:
        logger.warning(f"Cannot read gold: {e}")
        return pd.DataFrame()


def load_silver_data():
    try:
        return pd.read_parquet(SILVER_PATH)
    except Exception as e:
        return pd.DataFrame()


ws_ts = get_last_update()
if ws_ts is not None and ws_ts <= st.session_state.last_ws_ts:
    gold = st.session_state.gold
    silver = st.session_state.silver
else:
    gold = load_gold_data()
    silver = load_silver_data()
    st.session_state.gold = gold
    st.session_state.silver = silver
    st.session_state.last_ws_ts = ws_ts or time.time()

if gold.empty and silver.empty:
    st.warning("No data yet. Pipeline is starting up.")
    st.stop()

# Use gold as primary, fallback to silver
if not gold.empty:
    live_df = gold
    is_gold = True
    time_col = "window_start"
    price_col = "avg_price"
    volume_col = "avg_volume"
    change_col = "avg_change_pct"
else:
    live_df = silver
    is_gold = False
    time_col = "fetched_at"
    price_col = "price_usd"
    volume_col = "volume_24h_usd"
    change_col = "change_24h_pct"

for c in [time_col, price_col, volume_col, change_col]:
    if c in live_df.columns:
        live_df[c] = pd.to_numeric(live_df[c], errors="coerce")

live_df[time_col] = pd.to_datetime(live_df[time_col], errors="coerce")
live_df = live_df.dropna(subset=[price_col, time_col]).sort_values(time_col)

# ─── Predictions ──────────────────────────────────────────────────────

@st.cache_data(ttl=15, show_spinner="Computing predictions...")
def compute_predictions():
    if gold.empty:
        return pd.DataFrame()
    features_df, _ = build_features(gold)
    if features_df.empty:
        return pd.DataFrame()
    preds = model.predict(features_df[feature_cols])
    result = features_df[["coin_id", "window_start"]].copy()
    result["predicted_price"] = preds
    result["actual_price"] = features_df["avg_price"].values
    result["direction"] = np.where(
        result["predicted_price"] > result["actual_price"] * 1.005, "UP",
        np.where(result["predicted_price"] < result["actual_price"] * 0.995, "DOWN", "STABLE"),
    )
    result["pct_change_pred"] = ((result["predicted_price"] - result["actual_price"]) / result["actual_price"] * 100)
    # Confidence proxy based on model R2
    r2 = metrics.get("r2", 0) if metrics else 0
    result["confidence"] = max(0, min(100, (r2 * 50 + 50)))
    return result


preds = compute_predictions()

# ─── Alert Engine ─────────────────────────────────────────────────────

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []
if "next_alert_id" not in st.session_state:
    st.session_state.next_alert_id = 1


def add_alert(coin: str, alert_type: str, condition: str, threshold: float):
    st.session_state.alerts.append({
        "id": st.session_state.next_alert_id,
        "coin": coin,
        "type": alert_type,
        "condition": condition,
        "threshold": threshold,
        "active": True,
    })
    st.session_state.next_alert_id += 1


def remove_alert(alert_id: int):
    st.session_state.alerts = [a for a in st.session_state.alerts if a["id"] != alert_id]


def check_alerts():
    triggered = []
    for alert in st.session_state.alerts:
        if not alert["active"]:
            continue
        coin = alert["coin"]
        alert_type = alert["type"]
        condition = alert["condition"]
        threshold = alert["threshold"]

        cdf = live_df[live_df["coin_id"] == coin]
        if cdf.empty:
            continue
        current_price = cdf[price_col].iloc[-1]

        fire = False
        if alert_type == "price":
            if condition == "above" and current_price > threshold:
                fire = True
            elif condition == "below" and current_price < threshold:
                fire = True
        elif alert_type == "change_24h":
            change_val = cdf[change_col].iloc[-1] if change_col in cdf.columns else 0
            if condition == "above" and change_val > threshold:
                fire = True
            elif condition == "below" and change_val < threshold:
                fire = True
        elif alert_type == "direction":
            if not preds.empty:
                cp = preds[(preds["coin_id"] == coin)].sort_values("window_start")
                if not cp.empty:
                    direction = cp["direction"].iloc[-1]
                    if direction == condition:
                        fire = True

        if fire:
            triggered.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "coin": coin.upper(),
                "type": alert_type,
                "message": f"{coin.upper()} {condition} {threshold} — current: ${current_price:.2f}",
            })
            st.session_state.alert_history.append(triggered[-1])

    return triggered


triggered_alerts = check_alerts()

# ─── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/64px-Bitcoin.svg.png",
    width=28,
)
st.sidebar.title("Predictor & Alerts")

if metrics:
    st.sidebar.subheader("Model Health")
    col_m1, col_m2 = st.sidebar.columns(2)
    col_m1.metric("R²", f"{metrics.get('r2', 0):.3f}")
    col_m2.metric("MAE", f"${metrics.get('mae', 0):.2f}")
    col_m1.metric("RMSE", f"${metrics.get('rmse', 0):.2f}")
    col_m2.metric("Samples", f"{len(preds) if not preds.empty else 0}")

coins_in_data = sorted(live_df["coin_id"].unique())
selected_coin = st.sidebar.selectbox("Coin", coins_in_data, index=0)

tab = st.sidebar.radio("View", ["Predictions", "Alerts", "Signals"], index=0)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_int = st.sidebar.slider("Interval (s)", 5, 120, 15, disabled=not auto_refresh)

st.sidebar.divider()
st.sidebar.caption(f"Data: {'Gold (5-min)' if is_gold else 'Silver (raw)'}")

# ─── Triggered Alert Toast ────────────────────────────────────────────
for ta in triggered_alerts:
    st.toast(f"{ta['message']}", icon="")

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════
if tab == "Predictions":
    st.title("Price Prediction")
    st.caption(f"Model predicts next price direction using RandomForest — refreshed every {refresh_int}s")

    coin_data = live_df[live_df["coin_id"] == selected_coin].sort_values(time_col)
    if coin_data.empty:
        st.warning(f"No data for {selected_coin}")
        st.stop()

    current_price = coin_data[price_col].iloc[-1]

    # Latest prediction for this coin
    latest_pred = None
    if not preds.empty:
        cp = preds[preds["coin_id"] == selected_coin].sort_values("window_start")
        if not cp.empty:
            latest_pred = cp.iloc[-1]

    # ─── Big Direction Card ──────────────────────────────────────────
    if latest_pred is not None:
        direction = latest_pred["direction"]
        pred_price = latest_pred["predicted_price"]
        pct_chg = latest_pred["pct_change_pred"]
        conf = latest_pred["confidence"]

        if direction == "UP":
            arrow = ""
            color = "#4CAF50"
            bg = "rgba(76,175,80,0.15)"
        elif direction == "DOWN":
            arrow = ""
            color = "#F44336"
            bg = "rgba(244,67,54,0.15)"
        else:
            arrow = "→"
            color = "#FFC107"
            bg = "rgba(255,193,7,0.15)"

        col_card, col_metrics = st.columns([2, 1])

        with col_card:
            st.markdown(
                f"""
                <div style="background:{bg};border-radius:16px;padding:24px;text-align:center;
                            border:2px solid {color};">
                    <div style="font-size:3rem;">{arrow}</div>
                    <div style="font-size:2.5rem;font-weight:800;color:{color};">
                        {direction}
                    </div>
                    <div style="display:flex;justify-content:center;gap:40px;margin-top:12px;">
                        <div>
                            <div style="color:#888;font-size:0.8rem;">Current</div>
                            <div style="font-size:1.4rem;font-weight:700;">${current_price:.4f}</div>
                        </div>
                        <div style="color:#555;font-size:2rem;">→</div>
                        <div>
                            <div style="color:#888;font-size:0.8rem;">Predicted</div>
                            <div style="font-size:1.4rem;font-weight:700;">${pred_price:.4f}</div>
                        </div>
                    </div>
                    <div style="margin-top:8px;font-size:1.1rem;font-weight:600;color:{color};">
                        {pct_chg:+.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_metrics:
            st.markdown(
                f"""
                <div style="background:#1E1E1E;border-radius:12px;padding:20px;border:1px solid #333;height:100%;">
                    <div style="margin-bottom:16px;">
                        <div style="color:#888;font-size:0.8rem;">Confidence</div>
                        <div style="font-size:1.8rem;font-weight:700;color:{color};">{conf:.0f}%</div>
                    </div>
                    <div style="margin-bottom:16px;">
                        <div style="color:#888;font-size:0.8rem;">Signal</div>
                        <div style="font-size:1.4rem;font-weight:700;color:{color};">
                            {"STRONG BUY" if direction == "UP" and conf > 70 else "BUY" if direction == "UP" else "STRONG SELL" if direction == "DOWN" and conf > 70 else "SELL" if direction == "DOWN" else "HOLD"}
                        </div>
                    </div>
                    <div>
                        <div style="color:#888;font-size:0.8rem;">Model R²</div>
                        <div style="font-size:1.4rem;font-weight:700;">{metrics.get('r2', 0):.3f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough historical data for prediction yet.")

    st.divider()

    # ─── Actual vs Predicted Chart ───────────────────────────────────
    st.subheader("Actual vs Predicted Price")
    if not preds.empty and selected_coin in preds["coin_id"].values:
        cp = preds[preds["coin_id"] == selected_coin].sort_values("window_start")
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
        # Shade between
        fig.add_trace(go.Scatter(
            x=cp["window_start"], y=cp["actual_price"],
            mode="lines", line=dict(width=0), showlegend=False,
            fillcolor="rgba(0,188,212,0.1)",
        ))

        # Direction markers
        colors = {"UP": "#4CAF50", "DOWN": "#F44336", "STABLE": "#FFC107"}
        fig.add_trace(go.Scatter(
            x=cp["window_start"], y=cp["predicted_price"],
            mode="markers",
            marker=dict(
                size=8,
                color=[colors.get(d, "#888") for d in cp["direction"]],
                symbol=["triangle-up" if d == "UP" else "triangle-down" if d == "DOWN" else "circle" for d in cp["direction"]],
            ),
            name="Direction",
            hovertemplate="%{text}<extra></extra>",
            text=[f"{d} ({c:+.1f}%)" for d, c in zip(cp["direction"], cp["pct_change_pred"])],
        ))

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence gauge
        st.subheader("Direction Breakdown")
        dir_counts = cp["direction"].value_counts()
        cols = st.columns(3)
        for i, (dir_name, dir_color) in enumerate([("UP", "#4CAF50"), ("DOWN", "#F44336"), ("STABLE", "#FFC107")]):
            count = dir_counts.get(dir_name, 0)
            pct = count / len(cp) * 100 if len(cp) > 0 else 0
            with cols[i]:
                st.markdown(
                    f"""<div style="background:#1E1E1E;border-radius:10px;padding:12px;text-align:center;border:1px solid #333;">
                        <div style="font-size:1.5rem;color:{dir_color};font-weight:700;">{pct:.0f}%</div>
                        <div style="color:#888;font-size:0.8rem;">{dir_name} ({count})</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Not enough data for prediction chart.")

    # ─── Prediction Table ────────────────────────────────────────────
    st.divider()
    st.subheader("All Predictions")
    if not preds.empty:
        display = preds[preds["coin_id"] == selected_coin].sort_values("window_start", ascending=False)
        display["window_start"] = display["window_start"].astype(str)
        display["pct_change_pred"] = display["pct_change_pred"].round(2)
        display["actual_price"] = display["actual_price"].round(4)
        display["predicted_price"] = display["predicted_price"].round(4)
        st.dataframe(
            display[["window_start", "actual_price", "predicted_price", "direction", "pct_change_pred", "confidence"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "window_start": "Time",
                "actual_price": st.column_config.NumberColumn("Actual", format="$%.4f"),
                "predicted_price": st.column_config.NumberColumn("Predicted", format="$%.4f"),
                "direction": st.column_config.TextColumn("Direction"),
                "pct_change_pred": st.column_config.NumberColumn("Change %", format="%+.2f%%"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
            },
        )
    else:
        st.info("No predictions yet.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: ALERTS
# ═══════════════════════════════════════════════════════════════════════
elif tab == "Alerts":
    st.title("Alert System")
    st.caption("Configure price alerts and get notified when conditions are met.")

    col_config, col_active = st.columns([1, 1])

    with col_config:
        st.subheader("Create Alert")
        with st.form("alert_form", clear_on_submit=True):
            a_coin = st.selectbox("Coin", coins_in_data, key="a_coin")
            a_type = st.selectbox("Alert Type", ["price", "change_24h", "direction"], key="a_type")
            if a_type == "direction":
                a_condition = st.selectbox("Direction", ["UP", "DOWN"], key="a_dir")
                a_threshold = 0
            elif a_type == "price":
                a_condition = st.selectbox("Condition", ["above", "below"], key="a_price_cond")
                default_price = float(live_df[live_df["coin_id"] == a_coin][price_col].iloc[-1]) if not live_df.empty else 50000
                a_threshold = st.number_input("Threshold ($)", min_value=0.0, value=default_price * 1.1, step=100.0, format="%.2f")
            else:
                a_condition = st.selectbox("Condition", ["above", "below"], key="a_chg_cond")
                a_threshold = st.number_input("Threshold (%)", min_value=-100.0, max_value=1000.0, value=5.0, step=0.5)

            submitted = st.form_submit_button("Add Alert", use_container_width=True)
            if submitted:
                add_alert(a_coin, a_type, a_condition, a_threshold)
                st.success(f"Alert added for {a_coin.upper()} — {a_condition} {a_threshold}")
                st.rerun()

    with col_active:
        st.subheader(f"Active Alerts ({len([a for a in st.session_state.alerts if a['active']])})")
        if not st.session_state.alerts:
            st.info("No alerts configured.")
        else:
            for alert in st.session_state.alerts:
                if not alert["active"]:
                    continue
                with st.container():
                    col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
                    with col_a1:
                        st.markdown(f"**{alert['coin'].upper()}** — {alert['type']} {alert['condition']} {alert['threshold']}")
                    with col_a2:
                        st.markdown(f"<span style='color:#4CAF50;'>● Active</span>", unsafe_allow_html=True)
                    with col_a3:
                        if st.button("Remove", key=f"rm_{alert['id']}"):
                            remove_alert(alert["id"])
                            st.rerun()

    st.divider()

    # Alert History
    st.subheader("Alert History")
    if st.session_state.alert_history:
        hist_df = pd.DataFrame(reversed(st.session_state.alert_history[-50:]))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts triggered yet. Configure alerts above.")

    # Multi-coin price table for quick reference
    st.divider()
    st.subheader("Current Prices")
    price_rows = []
    for coin in coins_in_data:
        cdf = live_df[live_df["coin_id"] == coin]
        if cdf.empty:
            continue
        last = cdf.iloc[-1]
        price_rows.append({
            "coin": coin.upper(),
            "price": f"${last[price_col]:.4f}",
            "change": f"{last[change_col]:+.2f}%" if change_col in cdf.columns else "-",
            "volume": f"${last[volume_col]:,.0f}" if volume_col in cdf.columns else "-",
        })
    if price_rows:
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: SIGNALS
# ═══════════════════════════════════════════════════════════════════════
elif tab == "Signals":
    st.title("Trading Signals")
    st.caption("Aggregated BUY/SELL/HOLD signals based on ML predictions and technical indicators.")

    if preds.empty:
        st.info("Not enough prediction data yet.")
        st.stop()

    # Get latest prediction per coin
    latest_per_coin = preds.sort_values("window_start").groupby("coin_id").last().reset_index()

    # Build signal with score
    def compute_signal(row):
        direction = row["direction"]
        pct_chg = row["pct_change_pred"]
        conf = row["confidence"]
        r2 = metrics.get("r2", 0)

        # Score: -100 (strong sell) to +100 (strong buy)
        score = 0
        if direction == "UP":
            score = min(100, abs(pct_chg) * 10 + conf * 0.5)
        elif direction == "DOWN":
            score = -min(100, abs(pct_chg) * 10 + conf * 0.5)

        # Adjust by model confidence
        score = score * min(1, (r2 + 0.5) / 0.5) if r2 > 0 else score * 0.5

        if score > 60:
            signal = "STRONG BUY"
            sig_color = "#00E676"
        elif score > 20:
            signal = "BUY"
            sig_color = "#4CAF50"
        elif score < -60:
            signal = "STRONG SELL"
            sig_color = "#FF1744"
        elif score < -20:
            signal = "SELL"
            sig_color = "#F44336"
        else:
            signal = "HOLD"
            sig_color = "#FFC107"

        return signal, sig_color, round(score, 1)

    signals = []
    for _, row in latest_per_coin.iterrows():
        signal, sig_color, score = compute_signal(row)
        signals.append({
            "coin": row["coin_id"].upper(),
            "signal": signal,
            "score": score,
            "direction": row["direction"],
            "predicted": row["predicted_price"],
            "actual": row["actual_price"],
            "change_pct": round(row["pct_change_pred"], 2),
            "color": sig_color,
        })

    signals_df = pd.DataFrame(signals)
    signals_df = signals_df.sort_values("score", ascending=False)

    # Signal cards
    cols = st.columns(min(len(signals_df), 4))
    for i, (_, row) in enumerate(signals_df.iterrows()):
        with cols[i % len(cols)]:
            st.markdown(
                f"""<div style="background:#1E1E1E;border-radius:12px;padding:16px;text-align:center;
                            border:2px solid {row['color']};margin-bottom:12px;">
                    <div style="font-size:1.1rem;font-weight:700;">{row['coin']}</div>
                    <div style="font-size:1.8rem;font-weight:800;color:{row['color']};margin:8px 0;">
                        {row['signal']}
                    </div>
                    <div style="font-size:1rem;color:#888;">
                        Score: {row['score']}
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:0.8rem;">
                        <span>${row['actual']:.2f}</span>
                        <span style="color:{row['color']};">{row['change_pct']:+.2f}%</span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # Signal Table
    st.subheader("All Signals")
    signals_display = signals_df.copy()
    signals_display["actual"] = signals_display["actual"].apply(lambda x: f"${x:.4f}")
    signals_display["predicted"] = signals_display["predicted"].apply(lambda x: f"${x:.4f}")
    signals_display["change_pct"] = signals_display["change_pct"].apply(lambda x: f"{x:+.2f}%")
    st.dataframe(
        signals_display[["coin", "signal", "score", "direction", "actual", "predicted", "change_pct"]],
        use_container_width=True,
        hide_index=True,
    )

    # Signal history across time
    st.divider()
    st.subheader("Signal History")
    sel_coin_sig = st.selectbox("Coin", sorted(preds["coin_id"].unique()), key="sig_coin")
    cp = preds[preds["coin_id"] == sel_coin_sig].sort_values("window_start").copy()
    if len(cp) >= 3:
        cp["signal_score"] = cp.apply(
            lambda r: min(100, abs(r["pct_change_pred"]) * 10 + r["confidence"] * 0.5)
            if r["direction"] == "UP" else -min(100, abs(r["pct_change_pred"]) * 10 + r["confidence"] * 0.5),
            axis=1,
        )
        fig_sig = make_subplots(specs=[[{"secondary_y": True}]])

        fig_sig.add_trace(
            go.Scatter(x=cp["window_start"], y=cp["actual_price"], mode="lines",
                       name="Price", line=dict(color="#00BCD4", width=2)),
            secondary_y=False,
        )
        fig_sig.add_trace(
            go.Scatter(x=cp["window_start"], y=cp["signal_score"], mode="lines+markers",
                       name="Signal Score", line=dict(color="#FF9800", width=2),
                       fill="tozeroy", fillcolor="rgba(255,152,0,0.1)"),
            secondary_y=True,
        )
        fig_sig.add_hline(y=20, line_width=1, line_dash="dash", line_color="#4CAF50", secondary_y=True)
        fig_sig.add_hline(y=-20, line_width=1, line_dash="dash", line_color="#F44336", secondary_y=True)

        fig_sig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_sig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig_sig.update_yaxes(title_text="Signal Score", secondary_y=True, range=[-100, 100])

        st.plotly_chart(fig_sig, use_container_width=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_int)
    st.rerun()
