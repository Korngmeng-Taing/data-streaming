import pandas as pd
import numpy as np


def backtest(preds: pd.DataFrame, initial_capital: float = 1000.0) -> dict:
    if preds.empty or "direction" not in preds.columns:
        return {"error": "No prediction data"}

    preds = preds.sort_values(["coin_id", "window_start"]).reset_index(drop=True)

    results = []
    for coin in preds["coin_id"].unique():
        cp = preds[preds["coin_id"] == coin].copy()
        cp["actual_pct"] = cp["actual_price"].pct_change().fillna(0)
        cp["position"] = cp["direction"].map({"UP": 1, "DOWN": -1, "STABLE": 0})
        cp["strategy_return"] = cp["position"].shift(1).fillna(0) * cp["actual_pct"]
        cp["cum_strategy"] = (1 + cp["strategy_return"]).cumprod()
        cp["buy_hold"] = (1 + cp["actual_pct"]).cumprod()

        trades = _extract_trades(cp)
        stats = _compute_stats(cp, trades, initial_capital)
        results.append({"coin": coin, "stats": stats, "trades": trades, "curve": cp[["window_start", "cum_strategy", "buy_hold", "strategy_return", "position", "actual_price"]].copy()})

    return {"results": results}


def _extract_trades(df: pd.DataFrame) -> list:
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    pos_direction = 0

    for _, row in df.iterrows():
        pos = row["position"]
        if not in_position and pos != 0:
            in_position = True
            entry_price = row["actual_price"]
            entry_time = row["window_start"]
            pos_direction = pos
        elif in_position and (pos == 0 or pos != pos_direction):
            exit_price = row["actual_price"]
            exit_time = row["window_start"]
            pnl_pct = ((exit_price - entry_price) / entry_price) * pos_direction
            trades.append({
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "direction": "LONG" if pos_direction == 1 else "SHORT",
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
            })
            in_position = False

    return trades


def _compute_stats(df: pd.DataFrame, trades: list, initial_capital: float) -> dict:
    total_return = float(df["cum_strategy"].iloc[-1] - 1) * 100
    buy_hold_return = float(df["buy_hold"].iloc[-1] - 1) * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    equity = initial_capital * df["cum_strategy"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) * 100

    returns = df["strategy_return"]
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 24 * 6)) if returns.std() > 0 else 0

    return {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold_return, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
    }
