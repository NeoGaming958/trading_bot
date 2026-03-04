#!/usr/bin/env python3
"""
Master Research Runner V2.0
Runs: backtest + walk-forward + monte carlo + analytics + correlation
"""
import os, sys, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AppConfig
from research.backtester import Backtester, BacktestConfig
from research.optimizer import (
    WalkForwardValidator, MonteCarloSimulator,
    TradeAnalytics, CorrelationAnalyzer, ParamSpace
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_data(symbols, days=60):
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    data = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute, start=start)
            bars = client.get_stock_bars(request)
            bar_list = None
            if hasattr(bars, "data") and isinstance(bars.data, dict):
                bar_list = bars.data.get(symbol)
            if bar_list is None:
                try:
                    bar_list = bars[symbol]
                except (KeyError, TypeError):
                    pass
            if bar_list and len(bar_list) > 0:
                rows = []
                for b in bar_list:
                    rows.append({
                        "timestamp": b.timestamp,
                        "open": float(b.open), "high": float(b.high),
                        "low": float(b.low), "close": float(b.close),
                        "volume": float(b.volume),
                        "vwap": float(b.vwap) if hasattr(b, "vwap") and b.vwap else float(b.close),
                    })
                df = pd.DataFrame(rows)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                df["spread"] = np.maximum((df["high"] - df["low"]) * 0.1, df["close"] * 0.0005)
                df["bid"] = df["close"] - df["spread"] / 2
                df["ask"] = df["close"] + df["spread"] / 2
                data[symbol] = df
                print(f"{len(df)} bars")
            else:
                print("no data")
        except Exception as e:
            print(f"FAILED: {e}")
    return data


def screen_volatility(data):
    print("\nScreening for best mean-reversion candidates...")
    scores = {}
    for symbol, df in data.items():
        daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
        daily = daily.dropna()
        if len(daily) < 10:
            continue
        intraday_range = ((daily["high"] - daily["low"]) / daily["close"]).mean()
        avg_dollar_vol = (daily["close"] * daily["volume"]).mean()
        scores[symbol] = {
            "range_pct": round(intraday_range * 100, 2),
            "dollar_vol": round(avg_dollar_vol, 0),
        }
    ranked = sorted(scores.items(), key=lambda x: x[1]["range_pct"], reverse=True)
    print(f"  {'Symbol':<8} {'Range':>8} {'$ Volume':>15}")
    for sym, info in ranked:
        print(f"  {sym:<8} {info['range_pct']:>7.2f}% ${info['dollar_vol']:>14,.0f}")
    return ranked


def main():
    print("=" * 60)
    print("  RESEARCH FRAMEWORK V2.0")
    print("  MTF VWAP Mean Reversion - Full Analysis")
    print("=" * 60)

    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "SPY", "QQQ", "AMD",
        "COIN", "MARA", "RIOT", "SOFI", "PLTR",
        "SNAP", "ROKU", "SHOP", "ENPH",
        "SMCI", "ARM", "MSTR", "AFRM", "HOOD",
    ]

    print("\nFetching historical data (60 days, 1-min bars)...")
    data = fetch_data(symbols, days=60)
    if not data:
        print("ERROR: No data fetched")
        return

    # Always include SPY for regime filter even if not in trading universe
    ranked = screen_volatility(data)
    top_symbols = [sym for sym, _ in ranked[:10]]
    top_data = {s: data[s] for s in top_symbols if s in data}
    if "SPY" not in top_data and "SPY" in data:
        top_data["SPY"] = data["SPY"]

    print(f"\nTrading universe: {top_symbols}")

    # ── 1. PRIMARY BACKTEST ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: PRIMARY BACKTEST")
    print("=" * 60)

    bt = Backtester()
    results = bt.run(top_data, top_symbols)
    bt.print_report(results)

    # ── 2. TRADE ANALYTICS ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: TRADE ANALYTICS")
    print("=" * 60)

    analytics = TradeAnalytics()
    analysis = analytics.analyze(results["trades"])
    analytics.print_report(analysis)

    # ── 3. MONTE CARLO ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: MONTE CARLO SIMULATION")
    print("=" * 60)

    mc = MonteCarloSimulator(n_simulations=1000)
    mc_results = mc.run(results["trades"], bt.bt_config.starting_capital)
    mc.print_report(mc_results)

    # ── 4. CORRELATION ANALYSIS ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4: CORRELATION ANALYSIS")
    print("=" * 60)

    corr = CorrelationAnalyzer()
    corr_results = corr.analyze(top_data)
    corr.print_report(corr_results)

    # ── 5. WALK-FORWARD VALIDATION ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 5: WALK-FORWARD VALIDATION")
    print("=" * 60)

    wf = WalkForwardValidator(train_days=30, test_days=10)
    wf_results = wf.run(top_data, top_symbols)

    # ── 6. STRESSED SCENARIOS ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 6: STRESS TESTS")
    print("=" * 60)

    print("\n  --- 2x Slippage ---")
    bt2 = Backtester(bt_config=BacktestConfig(slippage_pct=0.001))
    r2 = bt2.run(top_data, top_symbols)
    bt2.print_report(r2)

    print("\n  --- $1,000 Account ---")
    bt3 = Backtester(bt_config=BacktestConfig(starting_capital=1000.0))
    r3 = bt3.run(top_data, top_symbols)
    bt3.print_report(r3)

    print("\n  --- Low Fill Rate (70%) ---")
    bt4 = Backtester(bt_config=BacktestConfig(fill_probability=0.70))
    r4 = bt4.run(top_data, top_symbols)
    bt4.print_report(r4)

    # ── SAVE RESULTS ───────────────────────────────────────────────
    if results["trades"]:
        trades_data = []
        for t in results["trades"]:
            trades_data.append({
                "symbol": t.symbol, "entry_time": t.entry_time,
                "exit_time": t.exit_time, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "shares": t.shares,
                "gross_pnl": t.gross_pnl, "net_pnl": t.net_pnl,
                "spread_cost": t.spread_cost, "slippage_cost": t.slippage_cost,
                "hold_bars": t.hold_bars, "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            })
        pd.DataFrame(trades_data).to_csv("logs/backtest_trades.csv", index=False)
        print("\nTrades saved to logs/backtest_trades.csv")

    if results["equity_curve"]:
        pd.Series(results["equity_curve"]).to_csv("logs/backtest_equity.csv")
        print("Equity curve saved to logs/backtest_equity.csv")

    print("\n" + "=" * 60)
    print("  RESEARCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
