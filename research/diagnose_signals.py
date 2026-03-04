#!/usr/bin/env python3
"""
Diagnostic V2: Fix ATR calculation (use daily bars, not minute bars)
and check all ORB filter rejections.
"""
import os, sys, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_bars(client, symbol, days=60, timeframe=TimeFrame.Minute):
    start = datetime.now(timezone.utc) - timedelta(days=days)
    request = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, start=start)
    bars = client.get_stock_bars(request)
    bar_list = None
    if hasattr(bars, "data") and isinstance(bars.data, dict):
        bar_list = bars.data.get(symbol)
    if bar_list is None:
        try: bar_list = bars[symbol]
        except: pass
    if not bar_list: return None
    rows = [{"timestamp": b.timestamp, "open": float(b.open), "high": float(b.high),
             "low": float(b.low), "close": float(b.close), "volume": float(b.volume),
             "vwap": float(b.vwap) if hasattr(b, "vwap") and b.vwap else float(b.close)} for b in bar_list]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def compute_daily_atr(daily_df, period=14):
    """ATR from DAILY bars (the correct way per Zarattini paper)."""
    if len(daily_df) < period + 1:
        return None
    h = daily_df["high"].values.astype(float)
    l = daily_df["low"].values.astype(float)
    c = daily_df["close"].values.astype(float)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def main():
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    symbols = ["MARA", "COIN", "MSTR", "HOOD", "SMCI"]

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  DIAGNOSING: {symbol}")
        print(f"{'='*60}")

        # Fetch DAILY bars for ATR and volume
        print(f"  Fetching daily bars...", end=" ")
        daily_df = fetch_bars(client, symbol, days=90, timeframe=TimeFrame.Day)
        if daily_df is None or len(daily_df) < 15:
            print("not enough daily data")
            continue
        print(f"{len(daily_df)} bars")

        # Compute DAILY ATR
        atr_values = {}
        daily_dates = sorted(daily_df.index)
        for i in range(len(daily_dates)):
            if i >= 14:
                window = daily_df.iloc[i-14:i+1]
                atr_values[daily_dates[i].date() if hasattr(daily_dates[i], 'date') else daily_dates[i]] = compute_daily_atr(window, 14)

        # Compute 14-day avg daily volume
        vol_values = {}
        for i in range(len(daily_dates)):
            if i >= 14:
                d = daily_dates[i].date() if hasattr(daily_dates[i], 'date') else daily_dates[i]
                vol_values[d] = float(daily_df.iloc[i-13:i+1]["volume"].mean())

        # Print recent daily stats
        print(f"\n  DAILY STATS (last 10 days with ATR):")
        recent_keys = sorted(atr_values.keys())[-10:]
        for d in recent_keys:
            atr = atr_values.get(d)
            avg_v = vol_values.get(d, 0)
            row = daily_df[daily_df.index.date == d].iloc[-1] if any(daily_df.index.date == d) else None
            if row is not None:
                print(f"    {d}  close=${float(row['close']):>8.2f}  "
                      f"ATR=${atr:>6.2f}  avgVol={avg_v:>12,.0f}")

        # Fetch minute bars
        print(f"\n  Fetching minute bars...", end=" ")
        minute_df = fetch_bars(client, symbol, days=60, timeframe=TimeFrame.Minute)
        if minute_df is None:
            print("no data")
            continue
        print(f"{len(minute_df)} bars")

        # Simulate ORB on minute data using DAILY ATR
        minute_df["date"] = minute_df.index.date
        days_grouped = minute_df.groupby("date")

        rejection_counts = Counter()
        total_bars = 0
        prev_close = None

        for day_date, day_df in days_grouped:
            day_df = day_df.sort_index()
            if len(day_df) < 10:
                continue

            # Get DAILY ATR and avg volume for this day
            atr_val = None
            avg_vol_daily = None
            for d in sorted(atr_values.keys(), reverse=True):
                if d <= day_date:
                    atr_val = atr_values[d]
                    break
            for d in sorted(vol_values.keys(), reverse=True):
                if d <= day_date:
                    avg_vol_daily = vol_values[d]
                    break

            # Build opening range (first 5 bars)
            orb_bars = day_df.iloc[:5]
            orb_high = float(orb_bars["high"].max())
            orb_low = float(orb_bars["low"].min())
            orb_open = float(orb_bars.iloc[0]["open"])
            orb_close = float(orb_bars.iloc[-1]["close"])
            orb_bullish = orb_close > orb_open
            orb_range = orb_high - orb_low
            orb_range_pct = orb_range / orb_high if orb_high > 0 else 0

            # Gap from previous close
            gap_pct = 0
            if prev_close and prev_close > 0:
                gap_pct = abs(float(day_df.iloc[0]["open"]) - prev_close) / prev_close
            prev_close = float(day_df.iloc[-1]["close"])

            # Scan post-ORB bars
            post_orb = day_df.iloc[5:]
            for bar_idx, (ts, row) in enumerate(post_orb.iterrows()):
                total_bars += 1
                price = float(row["close"])
                cur_vol = float(row["volume"])

                # Relative volume: current bar vol vs avg bar vol
                avg_bar_vol = avg_vol_daily / 390 if avg_vol_daily and avg_vol_daily > 0 else 0
                rel_vol = cur_vol / avg_bar_vol if avg_bar_vol > 0 else 0

                # Filter 1: Price
                if price < 5.0:
                    rejection_counts["01_price<$5"] += 1
                    continue

                # Filter 2: DAILY ATR > $0.50
                if atr_val is None or atr_val < 0.50:
                    rejection_counts["02_daily_atr<$0.50"] += 1
                    continue

                # Filter 3: Avg daily volume > 500K
                if avg_vol_daily is None or avg_vol_daily < 500000:
                    rejection_counts["03_avg_vol<500K"] += 1
                    continue

                # Filter 4: Relative volume > 2x
                if rel_vol < 2.0:
                    rejection_counts["04_rvol<2x"] += 1
                    continue

                # Filter 5: Gap > 2%
                if gap_pct < 0.02:
                    rejection_counts["05_gap<2%"] += 1
                    continue

                # Filter 6: Bullish ORB
                if not orb_bullish:
                    rejection_counts["06_bearish_orb"] += 1
                    continue

                # Filter 7: Price > ORB high (breakout)
                if price <= orb_high:
                    rejection_counts["07_no_breakout"] += 1
                    continue

                # Filter 8: Above VWAP
                day_up_to = day_df.loc[:ts]
                tp = (day_up_to["high"] + day_up_to["low"] + day_up_to["close"]) / 3
                vol_sum = day_up_to["volume"].sum()
                vwap = float((tp * day_up_to["volume"]).sum() / vol_sum) if vol_sum > 0 else price
                if price < vwap:
                    rejection_counts["08_below_vwap"] += 1
                    continue

                # Filter 9: Breakout volume (1.5x avg bar vol)
                if rel_vol < 1.5:
                    rejection_counts["09_weak_breakout_vol"] += 1
                    continue

                # Filter 10: Range size
                if orb_range_pct > 0.05:
                    rejection_counts["10_range_too_wide"] += 1
                    continue
                if orb_range_pct < 0.002:
                    rejection_counts["10_range_too_narrow"] += 1
                    continue

                # Filter 11: Time (first 60 bars = first hour)
                if bar_idx + 5 > 60:
                    rejection_counts["11_too_late"] += 1
                    continue

                # SIGNAL!
                rejection_counts["99_ENTRY_SIGNAL"] += 1
                print(f"    SIGNAL: {day_date} bar#{bar_idx+5:>3} ${price:.2f} "
                      f"ORB_hi=${orb_high:.2f} RVOL={rel_vol:.1f}x "
                      f"gap={gap_pct:.1%} range={orb_range_pct:.2%} "
                      f"ATR=${atr_val:.2f} VWAP=${vwap:.2f}")

        print(f"\n  FILTER FUNNEL ({total_bars} bars checked):")
        for reason, count in sorted(rejection_counts.items()):
            pct = count / total_bars * 100 if total_bars > 0 else 0
            marker = ">>>" if "SIGNAL" in reason else "   "
            print(f"    {marker} {reason:<30} {count:>8} ({pct:>5.1f}%)")

        signals = rejection_counts.get("99_ENTRY_SIGNAL", 0)
        print(f"\n  RESULT: {signals} entry signals in 60 days")

    print(f"\n{'='*60}")
    print("  Look at which filter kills the most signals per symbol.")
    print("  That tells us what to fix in the strategy + backtester.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
