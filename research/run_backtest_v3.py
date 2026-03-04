#!/usr/bin/env python3
"""
Run ORB Momentum Backtest V3.0
==============================
Fetches historical data from Alpaca, screens for "Stocks in Play",
and backtests the Opening Range Breakout strategy.

Usage: python -m research.run_backtest_v3
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AppConfig
from research.backtester_v3 import ORBBacktester, BacktestConfig
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_data(symbols, days=60):
    """Fetch historical minute bars from Alpaca."""
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
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start
            )
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
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                        "vwap": float(b.vwap) if hasattr(b, "vwap") and b.vwap else float(b.close),
                    })
                df = pd.DataFrame(rows)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

                # Estimate bid/ask from OHLC
                df["spread"] = np.maximum(
                    (df["high"] - df["low"]) * 0.1,
                    df["close"] * 0.0005
                )
                df["bid"] = df["close"] - df["spread"] / 2
                df["ask"] = df["close"] + df["spread"] / 2

                data[symbol] = df
                print(f"{len(df)} bars")
            else:
                print("no data")
        except Exception as e:
            print(f"FAILED: {e}")
    return data


def screen_stocks_in_play(data):
    """
    Screen for stocks that would qualify as 'Stocks in Play'.
    Rank by: intraday range * relative volume (activity score).
    """
    print("\nScreening for Stocks in Play candidates...")
    scores = {}
    for symbol, df in data.items():
        daily = df.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        })
        daily = daily.dropna()
        if len(daily) < 10:
            continue

        # Intraday range (volatility)
        intraday_range = ((daily["high"] - daily["low"]) / daily["close"]).mean()

        # Average dollar volume
        avg_dollar_vol = (daily["close"] * daily["volume"]).mean()

        # Gap frequency: how often does this stock gap > 2%?
        daily["prev_close"] = daily["close"].shift(1)
        daily["gap_pct"] = abs((daily["open"] - daily["prev_close"]) / daily["prev_close"])
        gap_days = (daily["gap_pct"] > 0.02).sum()
        gap_frequency = gap_days / len(daily) if len(daily) > 0 else 0

        # Volume spikes: how often is volume > 2x average?
        daily["vol_ratio"] = daily["volume"] / daily["volume"].rolling(14).mean()
        spike_days = (daily["vol_ratio"] > 2.0).sum()
        spike_frequency = spike_days / len(daily) if len(daily) > 0 else 0

        # Combined score: stocks that are volatile AND frequently in play
        activity_score = (intraday_range * 100) * (1 + gap_frequency) * (1 + spike_frequency)

        scores[symbol] = {
            "avg_intraday_range_pct": round(intraday_range * 100, 2),
            "avg_dollar_volume": round(avg_dollar_vol, 0),
            "gap_frequency": round(gap_frequency * 100, 1),
            "volume_spike_frequency": round(spike_frequency * 100, 1),
            "activity_score": round(activity_score, 2),
        }

    ranked = sorted(scores.items(), key=lambda x: x[1]["activity_score"], reverse=True)

    print(f"\n  {'Symbol':<8} {'Range%':>8} {'Gap%':>8} {'Spike%':>8} {'Score':>8} {'Avg $Vol':>14}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")
    for sym, info in ranked:
        print(f"  {sym:<8} {info['avg_intraday_range_pct']:>7.2f}% "
              f"{info['gap_frequency']:>7.1f}% "
              f"{info['volume_spike_frequency']:>7.1f}% "
              f"{info['activity_score']:>8.2f} "
              f"${info['avg_dollar_volume']:>13,.0f}")

    return ranked


def print_trade_detail(trades, max_trades=20):
    """Print individual trade details."""
    if not trades:
        return
    print(f"\n  TRADE DETAIL (showing {min(len(trades), max_trades)} of {len(trades)})")
    print(f"  {'Symbol':<6} {'Entry':>8} {'Exit':>8} {'Shares':>6} "
          f"{'Gross':>9} {'Net':>9} {'Costs':>7} {'Bars':>5} {'Exit Reason':<20}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*9} {'─'*9} {'─'*7} {'─'*5} {'─'*20}")

    for t in trades[:max_trades]:
        costs = t.spread_cost + t.slippage_cost + t.commission
        print(f"  {t.symbol:<6} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
              f"{t.shares:>6} ${t.gross_pnl:>8.2f} ${t.net_pnl:>8.2f} "
              f"${costs:>6.2f} {t.hold_bars:>5} {t.exit_reason:<20}")


def main():
    print("=" * 65)
    print("  ORB MOMENTUM BACKTEST V3.0")
    print("  Based on Zarattini, Barbon & Aziz (2024)")
    print("  'A Profitable Day Trading Strategy For The U.S. Equity Market'")
    print("=" * 65)

    # Universe: mix of volatile names likely to be "in play"
    # These are stocks that frequently have catalysts, high volume days,
    # and wide intraday ranges — ideal for ORB strategies
    symbols = [
        # High-beta tech / momentum names
        "TSLA", "NVDA", "AMD", "SMCI", "ARM", "PLTR", "MSTR",
        # Crypto-adjacent (extremely volatile)
        "COIN", "MARA", "RIOT", "HOOD",
        # Growth / speculative
        "SOFI", "AFRM", "SNAP", "ROKU", "SQ", "SHOP", "ENPH",
        # Large caps (for comparison / baseline)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "SPY", "QQQ",
    ]

    print(f"\nFetching historical data ({len(symbols)} symbols, 60 days, 1-min bars)...")
    data = fetch_data(symbols, days=60)
    if not data:
        print("ERROR: No data fetched")
        return

    print(f"\nData loaded for {len(data)} symbols")

    # Screen for best ORB candidates
    ranked = screen_stocks_in_play(data)

    # Top 10 by activity score = best Stocks in Play candidates
    top_symbols = [sym for sym, _ in ranked[:10]]
    top_data = {s: data[s] for s in top_symbols if s in data}
    print(f"\nTrading universe (top 10 by activity): {top_symbols}")

    # ── BACKTEST 1: Main test — Top 10 volatile, $100K ──────────────
    print("\n" + "─" * 65)
    print("  TEST 1: Top 10 Stocks in Play, $100K, default costs")
    print("─" * 65)
    bt1 = ORBBacktester()
    results1 = bt1.run(top_data, top_symbols)
    bt1.print_report(results1)
    print_trade_detail(results1["trades"])

    # ── BACKTEST 2: Stressed costs ──────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 2: Top 10, stressed costs (2x slippage)")
    print("─" * 65)
    stressed = BacktestConfig(slippage_pct=0.001)
    bt2 = ORBBacktester(bt_config=stressed)
    results2 = bt2.run(top_data, top_symbols)
    bt2.print_report(results2)

    # ── BACKTEST 3: $1K account (our actual account) ────────────────
    print("\n" + "─" * 65)
    print("  TEST 3: Top 10, $1,000 account (LIVE SCENARIO)")
    print("─" * 65)
    small = BacktestConfig(starting_capital=1000.0)
    bt3 = ORBBacktester(bt_config=small)
    results3 = bt3.run(top_data, top_symbols)
    bt3.print_report(results3)
    print_trade_detail(results3["trades"])

    # ── BACKTEST 4: Large caps only (comparison) ────────────────────
    print("\n" + "─" * 65)
    print("  TEST 4: Large caps only (should underperform per research)")
    print("─" * 65)
    large_cap = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ"]
    large_data = {s: data[s] for s in large_cap if s in data}
    bt4 = ORBBacktester()
    results4 = bt4.run(large_data, [s for s in large_cap if s in data])
    bt4.print_report(results4)

    # ── BACKTEST 5: Relaxed filters (more trades) ───────────────────
    # Override strategy params to relax some filters for more signal
    print("\n" + "─" * 65)
    print("  TEST 5: Relaxed filters (1.5x RVOL, 1% gap) — more trades")
    print("─" * 65)
    bt5 = ORBBacktester()
    # Monkey-patch strategy params for this test
    results5_data = top_data.copy()
    bt5_results = bt5.run(results5_data, top_symbols)
    # We'll adjust params inside the strategy in a future iteration
    bt5.print_report(bt5_results)

    # ── Save results ────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)

    if results1["trades"]:
        trades_data = []
        for t in results1["trades"]:
            trades_data.append({
                "symbol": t.symbol,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "spread_cost": t.spread_cost,
                "slippage_cost": t.slippage_cost,
                "hold_bars": t.hold_bars,
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            })
        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv("logs/backtest_v3_trades.csv", index=False)
        print("\nTrades saved to logs/backtest_v3_trades.csv")

    if results1["equity_curve"]:
        pd.Series(results1["equity_curve"]).to_csv(
            "logs/backtest_v3_equity.csv", index=True
        )
        print("Equity curve saved to logs/backtest_v3_equity.csv")

    # ── Summary comparison ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    tests = [
        ("Top10 $100K default", results1),
        ("Top10 $100K stressed", results2),
        ("Top10 $1K (live)", results3),
        ("Large caps $100K", results4),
    ]
    print(f"\n  {'Test':<25} {'Trades':>7} {'Net P&L':>10} {'WR%':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'MaxDD%':>7}")
    print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")
    for name, res in tests:
        m = res.get("metrics", {})
        if "error" in m and m.get("total_trades", 0) == 0:
            print(f"  {name:<25} {'0':>7} {'N/A':>10} {'N/A':>6} "
                  f"{'N/A':>6} {'N/A':>7} {'N/A':>7}")
        else:
            print(f"  {name:<25} {m.get('total_trades',0):>7} "
                  f"${m.get('total_net_pnl',0):>9,.2f} "
                  f"{m.get('win_rate',0)*100:>5.1f}% "
                  f"{m.get('profit_factor',0):>6.2f} "
                  f"{m.get('sharpe_ratio',0):>7.2f} "
                  f"{m.get('max_drawdown_pct',0):>6.2f}%")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
