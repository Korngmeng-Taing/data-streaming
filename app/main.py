import os
import time
from datetime import datetime

import streamlit as st
import pandas as pd

from config.logging_config import setup_logger
from ws_gateway.client import get_last_update

logger = setup_logger("app")

st.set_page_config(page_title="Crypto Alerts", page_icon="", layout="wide")

GOLD_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh") + "/gold"
SILVER_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh") + "/silver"

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
st.sidebar.title("Crypto Alerts")

coins_in_data = sorted(live_df["coin_id"].unique())
selected_coin = st.sidebar.selectbox("Coin", coins_in_data, index=0)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_int = st.sidebar.slider("Interval (s)", 5, 120, 15, disabled=not auto_refresh)

st.sidebar.divider()
st.sidebar.caption(f"Data: {'Gold (5-min)' if is_gold else 'Silver (raw)'}")

# ─── Triggered Alert Toast ────────────────────────────────────────────
for ta in triggered_alerts:
    st.toast(f"{ta['message']}", icon="")

# ═══════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════
st.title("Alert System")
st.caption("Configure price alerts and get notified when conditions are met.")

col_config, col_active = st.columns([1, 1])

with col_config:
    st.subheader("Create Alert")
    with st.form("alert_form", clear_on_submit=True):
        a_coin = st.selectbox("Coin", coins_in_data, key="a_coin")
        a_type = st.selectbox("Alert Type", ["price", "change_24h"], key="a_type")
        if a_type == "price":
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

# ─── Auto-refresh ─────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_int)
    st.rerun()
