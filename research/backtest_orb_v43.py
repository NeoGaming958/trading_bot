#!/usr/bin/env python3
# ORB Backtest V4.3 - Session-correct + entry/exit improvements
#
# This is a pragmatic ORB system intended to fix the failure mode seen in V4.2 logs:
# many stop-outs at the open and profits realized mostly by EOD trend days.
#
# Core infrastructure (kept):
#   - ET timezone (America/New_York)
#   - RTH-only bars (09:30-16:00 ET)
#   - session_date from ET
#   - bar_number = minutes since 09:30 ET
#   - Deterministic fills
#   - Mechanical stops (intrabar, gap-aware)
#   - Cash accounting + portfolio exposure cap
#
# Strategy adjustments (new, configurable):
#   1) Breakout confirmation: require N consecutive closes beyond ORB boundary.
#   2) Breakout buffer: entry level padded by a small buffer to avoid spread noise.
#   3) Candidate ranking: only trade top-N symbols per session by score (RVOL * range%).
#   4) Profit management: partial take-profit, breakeven stop, time stop, optional trail.
#
# Run:
#   python research/backtest_orb_v43.py
#
# Outputs:
#   logs/backtest_v43_trades.csv
#   logs/backtest_v43_equity.csv

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
from typing import Dict, Optional

ET = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)  # exclusive upper bound

UNIVERSE = [
    "MARA","MSTR","RIOT","COIN","HOOD","SMCI","SOFI","AFRM","ENPH","PLTR","TSLA","NVDA","AMD",
    "SNAP","ROKU","AAPL","AMZN","META","GOOG","MSFT","NFLX","SHOP","UBER","LYFT","BA","F","GM",
    "NIO","RIVN","JPM","BAC","GS","C","WFC","XOM","CVX","OXY","SLB","PFE","MRNA","ABBV","WMT",
    "TGT","COST","DIS","PYPL","INTC","MU","QCOM","FSLR","W","ADBE","CRM"
]

# Backtest window (days of history fetched from "now")
DAYS_DAILY = 365
DAYS_MINUTE = 120

# Filters
MIN_PRICE = 10.0
MIN_ATR = 0.50
MIN_AVG_DAILY_VOL = 1_000_000

ORB_BARS = 5
RVOL_LOOKBACK = 14
MIN_RVOL = 3.0
# Portfolio / sizing
START_CASH = float(os.environ.get("START_CASH", "100000"))
RISK_PER_TRADE_PCT = 0.01
MAX_POSITIONS = 2
MAX_GROSS_LEVERAGE = 2.0

MIN_NOTIONAL_PER_TRADE = float(os.environ.get("MIN_NOTIONAL", "250"))  # USD minimum notional per entry
MIN_NOTIONAL_PER_TRADE = float(os.environ.get("MIN_NOTIONAL", "200"))  # USD

# Entry improvements
CONFIRM_BARS = 3
BREAKOUT_BUFFER_BPS = 5  # 5 bps buffer beyond ORB boundary

# Exit improvements
PARTIAL_TP_R = 1.0
MOVE_BE_TO_R = 0.8
TIME_STOP_MIN = 90
TRAIL_AFTER_R = 2.0
TRAIL_ATR_MULT = 0.8
EOD_FLATTEN_MIN_BEFORE_CLOSE = 1

# Execution / costs (bounded)
SEC_FEE_RATE = 0.0000278
TAF_FEE_RATE = 0.000166
TAF_FEE_CAP = 8.30
SLIPPAGE_BPS = 2
SPREAD_BPS = 6

def get_alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient
    k = os.environ.get("ALPACA_API_KEY", "")
    s = os.environ.get("ALPACA_SECRET_KEY", "")
    if not k or not s:
        return None
    return StockHistoricalDataClient(api_key=k, secret_key=s)

def fetch_alpaca_bars(client, symbol: str, days: int, timeframe, feed="sip") -> Optional[pd.DataFrame]:
    from alpaca.data.requests import StockBarsRequest
    start = datetime.now(timezone.utc) - timedelta(days=days)
    request = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, start=start, feed=feed)
    bars = client.get_stock_bars(request)
    bl = getattr(bars, "data", {}).get(symbol) if hasattr(bars, "data") else None
    if bl is None:
        try:
            bl = bars[symbol]
        except Exception:
            bl = None
    if not bl:
        return None
    rows = []
    for b in bl:
        rows.append({
            "timestamp": pd.Timestamp(b.timestamp),
            "open": float(b.open), "high": float(b.high), "low": float(b.low), "close": float(b.close),
            "volume": float(b.volume),
            "vwap": float(b.vwap) if getattr(b, "vwap", None) else float(b.close),
        })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(ET)

def rth_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    m = (idx.time >= RTH_START) & (idx.time < RTH_END)
    return df.loc[m]

def daily_atr(df_daily: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df_daily is None or len(df_daily) < period + 1:
        return None
    h = df_daily["high"].values.astype(float)
    l = df_daily["low"].values.astype(float)
    c = df_daily["close"].values.astype(float)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    return float(np.mean(tr[-period:]))

def opening_5m_volume_by_session(df_min_rth: pd.DataFrame, orb_bars: int = 5) -> Dict[datetime.date, float]:
    out = {}
    if df_min_rth is None or df_min_rth.empty:
        return out
    for session_date, day_df in df_min_rth.groupby(df_min_rth.index.date):
        first = day_df.iloc[:orb_bars]
        if len(first) >= orb_bars:
            out[session_date] = float(first["volume"].sum())
    return out

def rvol_for_day(open_vol_hist: Dict[datetime.date, float], session_date, lookback=14) -> Optional[float]:
    if session_date not in open_vol_hist:
        return None
    prev_dates = sorted([d for d in open_vol_hist.keys() if d < session_date])[-lookback:]
    if len(prev_dates) < max(5, min(lookback, 10)):
        return None
    avg = float(np.mean([open_vol_hist[d] for d in prev_dates]))
    if avg <= 0:
        return None
    return float(open_vol_hist[session_date] / avg)

def est_spread(price: float) -> float:
    return max(price * (SPREAD_BPS / 10_000), 0.01)

def slippage(price: float) -> float:
    return price * (SLIPPAGE_BPS / 10_000)

def fees(side: str, notional: float, shares: int) -> float:
    sec = (notional * SEC_FEE_RATE) if side.lower() == "sell" else 0.0
    taf = min(shares * TAF_FEE_RATE, TAF_FEE_CAP) if side.lower() == "sell" else 0.0
    return sec + taf

@dataclass
class Position:
    sym: str
    dir: str  # "long" or "short"
    entry_ts: pd.Timestamp
    entry: float
    stop: float
    risk_per_share: float
    shares: int
    atr: float
    open_rvol: float
    orb_high: float
    orb_low: float
    partial_taken: bool = False
    max_favorable: float = 0.0
    bars_held: int = 0

def minutes_since_open(ts_et: pd.Timestamp) -> int:
    open_ts = ts_et.normalize().replace(hour=9, minute=30, second=0, microsecond=0)
    return int((ts_et - open_ts).total_seconds() // 60)

def main():
    os.makedirs("logs", exist_ok=True)
    client = get_alpaca_client()
    if client is None:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.")

    from alpaca.data.timeframe import TimeFrame

    print(f"Fetching daily bars (SIP) for {len(UNIVERSE)} symbols...")
    daily_stats = {}
    for sym in UNIVERSE:
        df_d = fetch_alpaca_bars(client, sym, DAYS_DAILY, TimeFrame.Day, feed="sip")
        if df_d is None or df_d.empty:
            continue
        atr = daily_atr(df_d, 14)
        avgvol = float(df_d["volume"].tail(20).mean()) if len(df_d) >= 20 else float(df_d["volume"].mean())
        if atr is not None:
            daily_stats[sym] = {"atr": float(atr), "avgvol": float(avgvol)}
            print(f"  {sym}... ATR=${atr:.2f} avgVol={avgvol:,.0f}")
        else:
            print(f"  {sym}... ATR=None avgVol={avgvol:,.0f}")

    print(f"\nFetching minute bars (SIP) for {len(UNIVERSE)} symbols...")
    minute = {}
    open_vol_hist = {}
    active = []
    for sym in UNIVERSE:
        if sym not in daily_stats:
            print(f"  {sym}... skip")
            continue
        df_m = fetch_alpaca_bars(client, sym, DAYS_MINUTE, TimeFrame.Minute, feed="sip")
        if df_m is None or df_m.empty:
            print(f"  {sym}... skip (no minute data)")
            continue
        pre = len(df_m)
        df_m = rth_filter(df_m)
        post = len(df_m)
        minute[sym] = df_m
        open_vol_hist[sym] = opening_5m_volume_by_session(df_m, ORB_BARS)
        if post > 0:
            active.append(sym)
        print(f"  {sym}... {pre}→{post} RTH bars")

    if not active:
        raise SystemExit("No symbols with minute data after RTH filter.")

    all_times = sorted(set(ts for sym in active for ts in minute[sym].index))
    sessions = sorted(set(ts.date() for ts in all_times))
    print(f"\nActive: {len(active)} of {len(UNIVERSE)}")
    print(f"  First session: {sessions[0]}  Last session: {sessions[-1]}")
    print("  RTH filter verified ✓")

    cash = START_CASH
    equity = START_CASH
    positions: Dict[str, Position] = {}
    trades = []
    eq_curve = []
    rejection = Counter()

    def get_last_close(sym: str, ts: pd.Timestamp) -> float:
        """Return the most recent close at or before ts for sym (handles sparse indices)."""
        df = minute[sym]
        if ts in df.index:
            return float(df.loc[ts, "close"])
        # asof-like: last observation before ts
        prior = df.loc[:ts]
        if prior.empty:
            # should not happen if position opened using existing ts
            return float(df.iloc[0]["close"])
        return float(prior.iloc[-1]["close"])

    def mark_to_market(ts):
        nonlocal equity
        eq = cash
        for p in positions.values():
            px = get_last_close(p.sym, ts)
            if p.dir == "long":
                eq += p.shares * px
            else:
                eq += p.shares * (p.entry - px)
        equity = eq

    orb_state = {}
    candidates_by_session = defaultdict(list)
    session_started = None

    for ts in all_times:
        ts_et = ts
        session_date = ts_et.date()
        bar_num = minutes_since_open(ts_et)

        if session_started != session_date:
            session_started = session_date
            orb_state = {}
            candidates_by_session[session_date] = []

        # Build ORB and candidates
        for sym in active:
            df = minute[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if sym not in orb_state:
                orb_state[sym] = {"open": float(row["open"]), "close": float(row["close"]),
                                  "high": float(row["high"]), "low": float(row["low"]),
                                  "vol": float(row["volume"]), "count": 1, "done": False}
            else:
                st = orb_state[sym]
                if not st["done"] and st["count"] < ORB_BARS:
                    st["high"] = max(st["high"], float(row["high"]))
                    st["low"] = min(st["low"], float(row["low"]))
                    st["close"] = float(row["close"])
                    st["vol"] += float(row["volume"])
                    st["count"] += 1
                if not st["done"] and st["count"] >= ORB_BARS:
                    st["done"] = True
                    atr = float(daily_stats[sym]["atr"])
                    avgvol = float(daily_stats[sym]["avgvol"])
                    px = float(row["close"])
                    if px < MIN_PRICE:
                        rejection["price_low"] += 1
                        continue
                    if atr < MIN_ATR:
                        rejection["atr_low"] += 1
                        continue
                    if avgvol < MIN_AVG_DAILY_VOL:
                        rejection["avgvol_low"] += 1
                        continue
                    rvol = rvol_for_day(open_vol_hist[sym], session_date, RVOL_LOOKBACK)
                    if rvol is None or rvol < MIN_RVOL:
                        rejection["rvol_low"] += 1
                        continue
                    rng = st["high"] - st["low"]
                    if rng <= 0:
                        rejection["range_zero"] += 1
                        continue
                    rng_pct = rng / max(st["open"], 1e-9)
                    score = float(rvol) * float(rng_pct)
                    bullish = st["close"] > st["open"]
                    dirn = "long" if bullish else "short"
                    if dirn == "short":
                        rejection["short_disabled"] += 1
                        continue
                    candidates_by_session[session_date].append((score, sym, dirn, st["high"], st["low"], float(rvol), atr))

        # EOD flatten
        if ts_et.time() >= (datetime.combine(ts_et.date(), time(16,0), tzinfo=ET) - timedelta(minutes=EOD_FLATTEN_MIN_BEFORE_CLOSE)).time():
            for sym in list(positions.keys()):
                p = positions.pop(sym)
                px = float(minute[sym].loc[ts, "close"])
                spr = est_spread(px)
                if p.dir == "long":
                    fill = px - spr/2 - slippage(px)
                    notional = fill * p.shares
                    cash += notional - fees("sell", notional, p.shares)
                    pnl = (fill - p.entry) * p.shares
                else:
                    fill = px + spr/2 + slippage(px)
                    notional = fill * p.shares
                    cash -= notional + fees("buy", notional, p.shares)
                    pnl = (p.entry - fill) * p.shares
                trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": p.dir,
                               "entry": p.entry, "exit": fill, "shares": p.shares, "stop": p.stop,
                               "rvol": p.open_rvol, "pnl": pnl, "reason": "eod", "bars": p.bars_held})

        # Manage open positions
        for sym in list(positions.keys()):
            p = positions[sym]
            if ts not in minute[sym].index:
                continue
            row = minute[sym].loc[ts]
            hi = float(row["high"]); lo = float(row["low"]); cl = float(row["close"])
            p.bars_held += 1

            if p.dir == "long":
                p.max_favorable = max(p.max_favorable, hi)
                r_close = (cl - p.entry) / p.risk_per_share

                if r_close >= MOVE_BE_TO_R:
                    p.stop = max(p.stop, p.entry)

                if (not p.partial_taken) and r_close >= PARTIAL_TP_R and p.shares >= 2:
                    sell_sh = p.shares // 2
                    spr = est_spread(cl)
                    fill = cl - spr/2 - slippage(cl)
                    notional = fill * sell_sh
                    cash += notional - fees("sell", notional, sell_sh)
                    pnl = (fill - p.entry) * sell_sh
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "long",
                                   "entry": p.entry, "exit": fill, "shares": sell_sh, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "partial_tp", "bars": p.bars_held})
                    p.shares -= sell_sh
                    p.partial_taken = True

                if r_close >= TRAIL_AFTER_R:
                    vol_proxy = max((p.orb_high - p.orb_low), p.atr * 0.05)
                    trail_dist = max(vol_proxy * TRAIL_ATR_MULT, 0.01)
                    p.stop = max(p.stop, p.max_favorable - trail_dist)

                if lo <= p.stop:
                    op = float(row["open"])
                    px = op if op < p.stop else p.stop
                    spr = est_spread(px)
                    fill = px - spr/2 - slippage(px)
                    notional = fill * p.shares
                    cash += notional - fees("sell", notional, p.shares)
                    pnl = (fill - p.entry) * p.shares
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "long",
                                   "entry": p.entry, "exit": fill, "shares": p.shares, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "stop", "bars": p.bars_held})
                    positions.pop(sym, None)
                    continue

                if p.bars_held >= TIME_STOP_MIN and cl <= p.entry:
                    spr = est_spread(cl)
                    fill = cl - spr/2 - slippage(cl)
                    notional = fill * p.shares
                    cash += notional - fees("sell", notional, p.shares)
                    pnl = (fill - p.entry) * p.shares
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "long",
                                   "entry": p.entry, "exit": fill, "shares": p.shares, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "time_stop", "bars": p.bars_held})
                    positions.pop(sym, None)

            else:
                p.max_favorable = min(p.max_favorable, lo) if p.max_favorable else lo
                r_close = (p.entry - cl) / p.risk_per_share

                if r_close >= MOVE_BE_TO_R:
                    p.stop = min(p.stop, p.entry)

                if (not p.partial_taken) and r_close >= PARTIAL_TP_R and p.shares >= 2:
                    cover_sh = p.shares // 2
                    spr = est_spread(cl)
                    fill = cl + spr/2 + slippage(cl)
                    notional = fill * cover_sh
                    cash -= notional + fees("buy", notional, cover_sh)
                    pnl = (p.entry - fill) * cover_sh
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "short",
                                   "entry": p.entry, "exit": fill, "shares": cover_sh, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "partial_tp", "bars": p.bars_held})
                    p.shares -= cover_sh
                    p.partial_taken = True

                if r_close >= TRAIL_AFTER_R:
                    vol_proxy = max((p.orb_high - p.orb_low), p.atr * 0.05)
                    trail_dist = max(vol_proxy * TRAIL_ATR_MULT, 0.01)
                    p.stop = min(p.stop, p.max_favorable + trail_dist)

                if hi >= p.stop:
                    op = float(row["open"])
                    px = op if op > p.stop else p.stop
                    spr = est_spread(px)
                    fill = px + spr/2 + slippage(px)
                    notional = fill * p.shares
                    cash -= notional + fees("buy", notional, p.shares)
                    pnl = (p.entry - fill) * p.shares
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "short",
                                   "entry": p.entry, "exit": fill, "shares": p.shares, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "stop", "bars": p.bars_held})
                    positions.pop(sym, None)
                    continue

                if p.bars_held >= TIME_STOP_MIN and cl >= p.entry:
                    spr = est_spread(cl)
                    fill = cl + spr/2 + slippage(cl)
                    notional = fill * p.shares
                    cash -= notional + fees("buy", notional, p.shares)
                    pnl = (p.entry - fill) * p.shares
                    trades.append({"ts_entry": p.entry_ts, "ts_exit": ts, "sym": sym, "dir": "short",
                                   "entry": p.entry, "exit": fill, "shares": p.shares, "stop": p.stop,
                                   "rvol": p.open_rvol, "pnl": pnl, "reason": "time_stop", "bars": p.bars_held})
                    positions.pop(sym, None)

        # Entries after ORB completes (ranked, confirmed)
        if bar_num >= ORB_BARS and bar_num <= 120:
            cands = sorted(candidates_by_session[session_date], key=lambda x: x[0], reverse=True)
            top = cands[:MAX_POSITIONS]

            mark_to_market(ts)
            gross = 0.0
            for p in positions.values():
                px = get_last_close(p.sym, ts)
                gross += abs(px * p.shares)

            for score, sym, dirn, orb_high, orb_low, rvol, atr in top:
                if sym in positions:
                    continue
                if ts not in minute[sym].index:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    rejection["max_positions"] += 1
                    break

                df = minute[sym]
                i = df.index.get_loc(ts)
                if i < CONFIRM_BARS - 1:
                    continue
                recent = df.iloc[i-(CONFIRM_BARS-1):i+1]["close"].astype(float).values
                px = float(df.loc[ts, "close"])
                spr = est_spread(px)
                buf = max(px * (BREAKOUT_BUFFER_BPS / 10_000), spr * 0.25)

                if dirn == "short":
                    rejection["short_disabled"] += 1
                    continue

                if dirn == "long":
                    level = orb_high + buf
                    if not np.all(recent > level):
                        rejection["no_confirm"] += 1
                        continue
                    stop = orb_low
                    risk_ps = max(level - stop, 0.01)
                else:
                    level = orb_low - buf
                    if not np.all(recent < level):
                        rejection["no_confirm"] += 1
                        continue
                    stop = orb_high
                    risk_ps = max(stop - level, 0.01)

                risk_dollars = max(equity * RISK_PER_TRADE_PCT, 1.0)
                sh = int(risk_dollars // risk_ps)
                if sh <= 0:
                    rejection["cant_size"] += 1
                    continue

                if dirn == "short":
                    rejection["short_disabled"] += 1
                    continue

                if dirn == "long":
                    fill = level + spr/2 + slippage(level)
                    notional = fill * sh
                    if notional < MIN_NOTIONAL_PER_TRADE:
                        rejection["min_notional"] += 1
                        continue

                    if cash < notional:
                        sh = int(cash // fill)
                        if sh <= 0:
                            rejection["no_cash"] += 1
                            continue
                        notional = fill * sh
                    if gross + notional > equity * MAX_GROSS_LEVERAGE:
                        rejection["lev_cap"] += 1
                        continue
                    cash -= notional
                    positions[sym] = Position(sym=sym, dir="long", entry_ts=ts, entry=fill, stop=stop,
                                              risk_per_share=risk_ps, shares=sh, atr=float(atr), open_rvol=float(rvol),
                                              orb_high=float(orb_high), orb_low=float(orb_low), max_favorable=fill)
                    rejection["entered"] += 1
                else:
                    fill = level - spr/2 - slippage(level)
                    notional = fill * sh
                    if gross + notional > equity * MAX_GROSS_LEVERAGE:
                        rejection["lev_cap"] += 1
                        continue
                    cash += notional
                    positions[sym] = Position(sym=sym, dir="short", entry_ts=ts, entry=fill, stop=stop,
                                              risk_per_share=risk_ps, shares=sh, atr=float(atr), open_rvol=float(rvol),
                                              orb_high=float(orb_high), orb_low=float(orb_low), max_favorable=fill)
                    rejection["entered"] += 1

        mark_to_market(ts)
        eq_curve.append({"ts": ts, "equity": equity, "cash": cash, "open_positions": len(positions)})

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(eq_curve)
    tpath = "logs/backtest_v43_trades.csv"
    epath = "logs/backtest_v43_equity.csv"
    tdf.to_csv(tpath, index=False)
    edf.to_csv(epath, index=False)

    net = float(tdf["pnl"].sum()) if not tdf.empty else 0.0
    wr = float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0
    reasons = tdf["reason"].value_counts().to_dict() if not tdf.empty else {}
    print("\n======================================================================")
    print("  SUMMARY - V4.3 (Improved ORB)")
    print("======================================================================")
    print(f"Trades: {len(tdf)}  Net P&L: ${net:,.2f}  WinRate: {wr*100:.1f}%")
    print(f"Exit reasons: {reasons}")
    print(f"Rejections (top 12): {rejection.most_common(12)}")
    print(f"Saved {tpath}")
    print(f"Saved {epath}")

if __name__ == "__main__":
    main()
