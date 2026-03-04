#!/usr/bin/env python3
"""
ORB Backtest V4.2 — Deterministic, Session-Correct
====================================================
Zarattini, Barbon & Aziz (2024)

Infrastructure fixes (V4.2 over V4.1):
  1. All timestamps converted to America/New_York BEFORE any logic
  2. Hard RTH filter: only 09:30–16:00 ET bars enter the engine
  3. session_date derived from ET time, not UTC .date()
  4. day_bar = minutes since 09:30 ET (not global counter)
     → bars_in_day=390, ORB=bars 1-5 (09:30-09:35), EOD=bar 390 (15:59)
  5. ORB built from first 5 RTH minutes (09:30–09:34), not "first 5 timestamps"
  6. RVOL history uses RTH-filtered, ET-session-dated data
  7. Deterministic fills: no randomness. Stop order triggers → fill at trigger.
     Stop loss hit intrabar → fill at stop price (or open if gapped through).
  8. Cash/buying-power accounting: opening a position reserves notional.
     Portfolio exposure cap prevents sizing all positions off full equity.
  9. Transaction costs: half-spread + SEC fee. No double-counting.
     Spread = max(0.01, price * 0.0002) per share per side, bounded.
     No (high-low)*0.1 nonsense.
  10. No Monte Carlo / randomness in baseline. Pure deterministic.

Usage: python research/backtest_orb.py
"""
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime, timezone
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)

# ── Timezone & RTH Utilities ───────────────────────────────────────

def to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any DataFrame index to America/New_York."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df

def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular trading hours: 09:30 <= time < 16:00 ET."""
    t = df.index.time
    mask = (t >= RTH_OPEN) & (t < RTH_CLOSE)
    return df[mask].copy()

def session_date(ts) -> "datetime.date":
    """Get the trading session date from an ET-localized timestamp."""
    if hasattr(ts, 'tz_convert'):
        et_ts = ts.tz_convert(ET)
    elif hasattr(ts, 'astimezone'):
        et_ts = ts.astimezone(ET)
    else:
        et_ts = ts
    return et_ts.date() if hasattr(et_ts, 'date') and callable(et_ts.date) else et_ts

def minute_of_day(ts) -> int:
    """
    Minutes since 09:30 ET. Bar 1 = 09:30, bar 5 = 09:34, bar 390 = 15:59.
    Returns 0 if before open, >390 if after close.
    Assumes ts is already ET-localized (via to_et + filter_rth).
    """
    if hasattr(ts, 'tz_convert'):
        et_ts = ts.tz_convert(ET)
    elif hasattr(ts, 'astimezone'):
        et_ts = ts.astimezone(ET)
    else:
        et_ts = ts
    h, m = et_ts.hour, et_ts.minute
    minutes_since_930 = (h - 9) * 60 + (m - 30)
    return minutes_since_930 + 1  # bar 1 = 09:30

# ── Data fetching (SIP feed) ──────────────────────────────────────

def get_alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient
    k = os.environ.get("ALPACA_API_KEY", "")
    s = os.environ.get("ALPACA_SECRET_KEY", "")
    if not k or not s:
        return None
    return StockHistoricalDataClient(api_key=k, secret_key=s)

def fetch_alpaca(client, symbol, days, tf, feed="sip"):
    from alpaca.data.requests import StockBarsRequest
    start = datetime.now(timezone.utc) - timedelta(days=days)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf,
        start=start,
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    bl = getattr(bars, 'data', {}).get(symbol) if hasattr(bars, 'data') else None
    if bl is None:
        try:
            bl = bars[symbol]
        except Exception:
            pass
    if not bl:
        return None
    rows = [{
        "timestamp": b.timestamp,
        "open": float(b.open), "high": float(b.high),
        "low": float(b.low), "close": float(b.close),
        "volume": float(b.volume),
        "vwap": float(b.vwap) if hasattr(b, "vwap") and b.vwap else float(b.close)
    } for b in bl]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df

def daily_atr_series(df, period=14):
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(h)
    out = [None] * n
    for i in range(period, n):
        wh, wl = h[i - period + 1:i + 1], l[i - period + 1:i + 1]
        wc = c[i - period:i]
        tr = np.maximum(
            wh[1:] - wl[1:],
            np.maximum(np.abs(wh[1:] - wc[1:]), np.abs(wl[1:] - wc[1:]))
        )
        out[i] = float(np.mean(tr))
    return out

# ── RVOL: uses RTH-filtered, ET-session-dated data ────────────────

def build_opening_vol_history(minute_data: Dict[str, pd.DataFrame], orb_bars=5):
    """Build {symbol: {session_date: first_5_RTH_minutes_volume}}."""
    history = {}
    for sym, df in minute_data.items():
        vol_by_day = {}
        # df is already ET-converted and RTH-filtered
        dates = sorted(set(d.date() for d in df.index))
        for dt in dates:
            day_bars = df[df.index.date == dt].head(orb_bars)
            if len(day_bars) >= orb_bars:
                vol_by_day[dt] = float(day_bars["volume"].sum())
        history[sym] = vol_by_day
    return history

def get_rvol(opening_vols, symbol, today, lookback=14):
    if symbol not in opening_vols:
        return None
    hist = opening_vols[symbol]
    past_dates = sorted([d for d in hist if d < today])[-lookback:]
    if len(past_dates) < 5:
        return None
    avg_vol = np.mean([hist[d] for d in past_dates])
    if avg_vol <= 0:
        return None
    today_vol = hist.get(today, 0)
    if today_vol <= 0:
        return None
    return today_vol / avg_vol

def get_avg_daily_vol(daily_stats, symbol, date, lookback=14):
    ds = daily_stats.get(symbol)
    if ds is None:
        return 0
    past = ds[ds.index.date < date] if hasattr(ds.index[0], 'date') else ds[ds.index < date]
    past = past.tail(lookback)
    if len(past) < lookback:
        return 0
    return float(past["volume"].mean())

# ── Transaction costs: realistic, bounded, no double-counting ──────

def calc_half_spread(price: float) -> float:
    """
    Half the bid-ask spread per share.
    Floor: $0.01 (one penny — minimum tick for stocks > $1).
    Estimate: price x 2 bps for liquid large-caps.
    Cap: 10 bps (for very expensive stocks).
    """
    estimate = price * 0.0002   # 2 bps
    capped = min(estimate, price * 0.001)  # cap at 10 bps
    return max(0.01, capped)    # floor at $0.01 always wins

def calc_costs(price: float, shares: int) -> float:
    """
    Total round-trip transaction costs per trade.
    - Half-spread on entry + half-spread on exit = 1 full spread
    - SEC fee: $0.0000278 per dollar sold (exit only)
    - FINRA TAF: $0.000166 per share sold (exit only)
    No slippage on top — the spread IS the slippage model.
    """
    hs = calc_half_spread(price)
    spread_cost = hs * shares * 2          # entry + exit
    sec_fee = price * shares * 0.0000278   # on sell side
    taf_fee = shares * 0.000166            # on sell side
    return spread_cost + sec_fee + taf_fee

# ── Trade dataclass ────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str; direction: str
    entry_time: object; exit_time: object
    entry_price: float; exit_price: float; shares: int
    gross_pnl: float; costs: float; net_pnl: float
    hold_bars: int; exit_reason: str; rvol: float
    r_multiple: float

# ── Per-session bar schedule builder ───────────────────────────────

def build_session_schedule(minute_data: Dict[str, pd.DataFrame], symbols: list):
    """
    Build a per-session, per-bar schedule from RTH-filtered minute data.
    Returns: {session_date: [list of (bar_number, timestamp, {sym: row})]}
    where bar_number = minutes since 09:30 (1-indexed).
    """
    # Collect all unique RTH timestamps across all symbols
    all_ts = set()
    for s in symbols:
        if s in minute_data:
            all_ts.update(minute_data[s].index.tolist())
    all_ts = sorted(all_ts)

    sessions = {}
    for ts in all_ts:
        sd = ts.date()  # Already ET
        bar = minute_of_day(ts)
        if bar < 1 or bar > 390:
            continue
        if sd not in sessions:
            sessions[sd] = {}
        if bar not in sessions[sd]:
            sessions[sd][bar] = {"ts": ts, "syms": {}}
        # Collect data for each symbol at this timestamp
        for sym in symbols:
            if sym in minute_data and ts in minute_data[sym].index:
                sessions[sd][bar]["syms"][sym] = minute_data[sym].loc[ts]

    # Convert to sorted list per session
    result = {}
    for sd in sorted(sessions.keys()):
        bars_list = []
        for bar_num in sorted(sessions[sd].keys()):
            bars_list.append((bar_num, sessions[sd][bar_num]["ts"],
                              sessions[sd][bar_num]["syms"]))
        result[sd] = bars_list
    return result

# ── Core Backtester ────────────────────────────────────────────────

def run(minute_data, daily_stats, opening_vols, symbols,
        capital=100000.0, verbose=False,
        rvol_threshold=2.0, max_positions=20,
        stop_atr_pct=0.10, max_leverage=4.0,
        min_avg_daily_vol=1_000_000):
    """
    Deterministic, session-correct ORB backtest.

    All timestamps are ET. All bars are RTH (09:30–16:00).
    Bar numbers are minutes since 09:30 (bar 1 = 09:30, bar 390 = 15:59).
    ORB = bars 1–5 (09:30–09:34 ET).
    EOD flatten = bar 386+ (~15:55 ET, giving 5 min to close cleanly).
    
    Cash accounting: equity tracks realized P&L. Buying power = equity
    minus notional of open positions. New positions sized off buying power
    for risk calc but capped by exposure limits.

    Fills: deterministic. Entry stop order fills at trigger price.
    Stop loss fills at stop price, or at open if price gaps through.
    """
    ORB_BARS = 5
    MIN_ATR = 0.50
    MIN_PRICE = 5.0
    EOD_BAR = 386         # 15:55 ET — flatten
    RISK_PCT = 0.01

    equity = capital       # Realized equity (cash + closed P&L)
    cash = capital         # Available cash (reduced by open positions)
    eq_curve = [equity]
    trades = []
    rej = Counter()

    # Build session schedule
    schedule = build_session_schedule(minute_data, symbols)
    n_sessions = len(schedule)
    n_bars = sum(len(bars) for bars in schedule.values())
    print(f"  {n_bars} RTH bars across {n_sessions} sessions, "
          f"{len(symbols)} syms, ${capital:,.0f}, "
          f"RVOL>={rvol_threshold}x, stop={stop_atr_pct}×ATR, "
          f"max_pos={max_positions}, lev={max_leverage}x")

    def get_atr(sym, dt):
        ds = daily_stats.get(sym)
        if ds is None:
            return None
        # daily_stats index is also ET-converted
        for t in sorted(ds.index, reverse=True):
            d = t.date() if hasattr(t, 'date') else t
            if d <= dt:
                val = ds.loc[t, "atr"]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return val
        return None

    def close_pos(sym, ep, et, reason, positions_dict):
        nonlocal equity, cash
        if sym not in positions_dict:
            return
        p = positions_dict[sym]
        costs = calc_costs(ep, p["sh"])
        if p["dir"] == "long":
            gross = (ep - p["en"]) * p["sh"]
        else:
            gross = (p["en"] - ep) * p["sh"]
        net = gross - costs - p["entry_cost"]
        risk_per_share = p["atr"] * stop_atr_pct
        r_mult = (gross / p["sh"]) / risk_per_share if risk_per_share > 0 and p["sh"] > 0 else 0
        trades.append(Trade(
            sym, p["dir"], p["et"], et, p["en"], ep, p["sh"],
            gross, costs + p["entry_cost"], net, p["bh"], reason,
            p.get("rvol", 0), r_mult
        ))
        equity += net
        # Return notional to cash pool
        cash += p["notional"] + net
        del positions_dict[sym]

    # ── Session loop ─────────────────────────────────────────────────
    for sd, bars_list in schedule.items():
        # Day-level state
        orb = {}                 # {symbol: {hi, lo, op, cl, n, done, ...}}
        rvol_today = {}
        traded_today = set()
        pending_orders = {}
        positions = {}           # carried within session only
        daily_pnl = 0.0
        disabled = False

        for bar_num, ts, sym_data in bars_list:

            # ── Process each symbol ─────────────────────────────
            for sym, row in sym_data.items():
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row["volume"])
                atr = get_atr(sym, sd)

                # ── Build ORB (bars 1-5 = 09:30-09:34) ──────────
                if bar_num <= ORB_BARS:
                    if sym not in orb:
                        orb[sym] = {"hi": h, "lo": l, "op": o, "cl": c,
                                    "n": 1, "done": False, "v": v}
                    else:
                        d = orb[sym]
                        if not d["done"]:
                            d["hi"] = max(d["hi"], h)
                            d["lo"] = min(d["lo"], l)
                            d["cl"] = c
                            d["n"] += 1
                            d["v"] += v

                    if bar_num == ORB_BARS and sym in orb:
                        d = orb[sym]
                        d["done"] = True
                        d["bull"] = d["cl"] > d["op"]
                        d["bear"] = d["cl"] < d["op"]
                        d["doji"] = d["cl"] == d["op"]

                        rv = get_rvol(opening_vols, sym, sd)
                        rvol_today[sym] = rv

                        # Filters
                        if d["doji"]:
                            rej["doji"] += 1; continue
                        if atr is None or atr < MIN_ATR:
                            rej["atr_low"] += 1; continue
                        if (d["hi"] + d["lo"]) / 2 < MIN_PRICE:
                            rej["price_low"] += 1; continue
                        avg_dv = get_avg_daily_vol(daily_stats, sym, sd)
                        if min_avg_daily_vol > 0 and avg_dv < min_avg_daily_vol:
                            rej["avg_vol_low"] += 1; continue
                        if rvol_threshold > 0 and (rv is None or rv < rvol_threshold):
                            rej["rvol_low"] += 1; continue

                        # Place pending stop order
                        if d["bull"]:
                            pending_orders[sym] = {
                                "dir": "long", "trigger": d["hi"],
                                "atr": atr, "rvol": rv if rv else 0,
                            }
                        elif d["bear"]:
                            pending_orders[sym] = {
                                "dir": "short", "trigger": d["lo"],
                                "atr": atr, "rvol": rv if rv else 0,
                            }

                        if verbose and rv and rv >= rvol_threshold:
                            dr = "BULL→buy" if d["bull"] else "BEAR→sell"
                            trig = d["hi"] if d["bull"] else d["lo"]
                            print(f"  ORB: {sym} {sd} {d['lo']:.2f}-{d['hi']:.2f} "
                                  f"RVOL={rv:.1f}x {dr}@{trig:.2f} "
                                  f"ATR=${atr:.2f} stop_d=${atr*stop_atr_pct:.2f}")
                    continue  # Don't trade during ORB formation

                # ── Check pending stop orders ────────────────────
                if (sym in pending_orders and sym not in positions
                        and sym not in traded_today):
                    po = pending_orders[sym]
                    triggered = False

                    if po["dir"] == "long" and h >= po["trigger"]:
                        triggered = True
                    elif po["dir"] == "short" and l <= po["trigger"]:
                        triggered = True

                    if triggered:
                        if disabled:
                            rej["daily_loss_limit"] += 1
                            del pending_orders[sym]; continue
                        if len(positions) >= max_positions:
                            rej["max_positions"] += 1; continue

                        direction = po["dir"]
                        entry_price = po["trigger"]  # Deterministic fill at trigger
                        atr_val = po["atr"]
                        stop_dist = atr_val * stop_atr_pct

                        if direction == "long":
                            stop_price = entry_price - stop_dist
                        else:
                            stop_price = entry_price + stop_dist

                        if stop_dist <= 0.001:
                            rej["zero_risk"] += 1
                            del pending_orders[sym]; continue

                        # Position sizing: 1% risk
                        sh = int(equity * RISK_PCT / stop_dist)

                        # Leverage cap: 4×
                        max_lev = int(equity * max_leverage / entry_price) \
                            if entry_price > 0 else 0

                        # Equal-weight cap
                        max_wt = int(equity / max_positions / entry_price) \
                            if entry_price > 0 else 0

                        # Buying power cap: can't spend more cash than available
                        notional = entry_price * min(sh, max_lev, max_wt)
                        if cash <= 0:
                            rej["no_cash"] += 1
                            del pending_orders[sym]; continue
                        max_bp = int(cash / entry_price)

                        sh = min(sh, max_lev, max_wt, max_bp)
                        if sh <= 0:
                            rej["cant_size"] += 1
                            del pending_orders[sym]; continue

                        notional = entry_price * sh
                        entry_cost = calc_costs(entry_price, sh)

                        # Reserve cash
                        cash -= notional

                        positions[sym] = {
                            "dir": direction, "en": entry_price,
                            "sh": sh, "st": stop_price, "atr": atr_val,
                            "entry_cost": entry_cost, "et": ts, "bh": 0,
                            "rvol": po["rvol"], "notional": notional,
                        }
                        traded_today.add(sym)
                        del pending_orders[sym]
                        rej["ENTERED"] += 1

                        if verbose and rej["ENTERED"] <= 50:
                            print(f"  {'LONG' if direction=='long' else 'SHORT'}: "
                                  f"{sym} {sd} bar={bar_num} ${entry_price:.2f} "
                                  f"stop=${stop_price:.2f} sh={sh} "
                                  f"RVOL={po['rvol']:.1f}x cash_left=${cash:,.0f}")

                # ── Mechanical stop execution ────────────────────
                if sym in positions:
                    p = positions[sym]
                    p["bh"] += 1

                    stopped = False
                    if p["dir"] == "long" and l <= p["st"]:
                        # Stop hit. If open gapped below stop, fill at open (worse).
                        # Otherwise fill at stop price (conservative: assumes
                        # intrabar price crossed stop at some point).
                        if o <= p["st"]:
                            exit_price = o   # Gapped through → fill at open
                        else:
                            exit_price = p["st"]  # Normal → fill at stop
                        close_pos(sym, exit_price, ts, "stop_loss", positions)
                        stopped = True

                    elif p["dir"] == "short" and h >= p["st"]:
                        if o >= p["st"]:
                            exit_price = o
                        else:
                            exit_price = p["st"]
                        close_pos(sym, exit_price, ts, "stop_loss", positions)
                        stopped = True

                    if stopped:
                        continue

                    # EOD flatten (bar 386 = ~15:55 ET)
                    if bar_num >= EOD_BAR:
                        won = ((p["dir"] == "long" and c > p["en"]) or
                               (p["dir"] == "short" and c < p["en"]))
                        close_pos(sym, c, ts, "eod_win" if won else "eod_loss",
                                  positions)
                        continue

            # Daily loss limit
            daily_realized = sum(t.net_pnl for t in trades
                                 if hasattr(t.entry_time, 'date')
                                 and t.entry_time.date() == sd)
            if daily_realized <= -(equity * 0.03):
                disabled = True

            # Mark-to-market equity curve (realized + unrealized)
            mtm = equity
            for sym, p in positions.items():
                if sym in sym_data:
                    cp = float(sym_data[sym]["close"])
                    if p["dir"] == "long":
                        mtm += (cp - p["en"]) * p["sh"]
                    else:
                        mtm += (p["en"] - cp) * p["sh"]
            eq_curve.append(mtm)

        # ── End of session: flatten any remaining positions ──────
        for sym in list(positions.keys()):
            # Use last bar's close for this symbol
            last_price = positions[sym]["en"]
            for bn, bts, sd2 in reversed(bars_list):
                if sym in sd2:
                    last_price = float(sd2[sym]["close"])
                    break
            close_pos(sym, last_price, bars_list[-1][1] if bars_list else ts,
                      "session_end", positions)

    return trades, eq_curve, rej

# ── Report ─────────────────────────────────────────────────────────

def report(trades, eq, rej, cap, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    if not trades:
        print(f"  NO TRADES. Rejections:")
        for r, c in rej.most_common(15):
            print(f"    {r:<25} {c:>8}")
        return

    pnls = [t.net_pnl for t in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    tot = sum(pnls)
    cs = sum(t.costs for t in trades)
    gr = sum(t.gross_pnl for t in trades)
    wr = len(w) / len(pnls)
    pf = sum(w) / abs(sum(l)) if l and sum(l) != 0 else float('inf')
    e = np.array(eq)
    pk = np.maximum.accumulate(e)
    dd = (pk - e) / pk
    mdd = float(np.max(dd)) * 100

    # Daily returns for Sharpe (group equity curve by session boundaries)
    # Approximate: use every 390th point as session boundary
    daily_eq = [e[0]]
    step = max(1, len(e) // max(1, len(set(t.entry_time.date() for t in trades if hasattr(t.entry_time, 'date')))))
    for i in range(step, len(e), step):
        daily_eq.append(e[i])
    daily_eq.append(e[-1])
    daily_eq = np.array(daily_eq)
    daily_ret = np.diff(daily_eq) / daily_eq[:-1]
    sh = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) \
        if len(daily_ret) > 1 and np.std(daily_ret) > 0 else 0

    long_t = [t for t in trades if t.direction == "long"]
    short_t = [t for t in trades if t.direction == "short"]
    long_w = len([t for t in long_t if t.net_pnl > 0])
    short_w = len([t for t in short_t if t.net_pnl > 0])

    r_mults = [t.r_multiple for t in trades]
    r_wins = [t.r_multiple for t in trades if t.r_multiple > 0]
    r_losses = [t.r_multiple for t in trades if t.r_multiple <= 0]

    print(f"\n  Net P&L: ${tot:,.2f} ({tot / cap * 100:.1f}%)")
    print(f"  Gross P&L: ${gr:,.2f}  Costs: ${cs:,.2f}")
    print(f"  Sharpe (daily): {sh:.2f}  Profit Factor: {pf:.2f}")
    print(f"  Trades: {len(trades)}  Win Rate: {wr * 100:.1f}%  ({len(w)}W / {len(l)}L)")
    print(f"    Long: {len(long_t)} ({long_w}W)  Short: {len(short_t)} ({short_w}W)")
    print(f"  Avg Win: ${np.mean(w) if w else 0:,.2f}  Avg Loss: ${np.mean(l) if l else 0:,.2f}")
    print(f"  Max DD: {mdd:.2f}%  Avg Hold: {np.mean([t.hold_bars for t in trades]):.0f} bars")
    print(f"  Avg RVOL: {np.mean([t.rvol for t in trades]):.1f}x")
    print(f"\n  R-Multiple Stats:")
    print(f"    Avg R: {np.mean(r_mults):.2f}R  "
          f"Avg Win: {np.mean(r_wins) if r_wins else 0:.2f}R  "
          f"Avg Loss: {np.mean(r_losses) if r_losses else 0:.2f}R")
    print(f"    Best: {max(r_mults):.1f}R  Worst: {min(r_mults):.1f}R  "
          f"Expectancy: {np.mean(r_mults):.3f}R/trade")

    # Cost analysis
    avg_cost_pct = np.mean([t.costs / (t.entry_price * t.shares) * 100
                            for t in trades if t.shares > 0])
    print(f"\n  Cost Analysis:")
    print(f"    Total costs: ${cs:,.2f}  Avg cost/trade: ${cs/len(trades):,.2f}")
    print(f"    Avg cost as % of notional: {avg_cost_pct:.3f}%")

    print(f"\n  Exit Reasons:")
    rc = Counter()
    rp = Counter()
    rr = {}
    for t in trades:
        rc[t.exit_reason] += 1
        rp[t.exit_reason] += t.net_pnl
        rr.setdefault(t.exit_reason, []).append(t.r_multiple)
    for r in sorted(rc):
        ww = len([t for t in trades if t.exit_reason == r and t.net_pnl > 0])
        avg_r = np.mean(rr[r])
        print(f"    {r:<22} {rc[r]:>3}  ${rp[r]:>10,.2f}  "
              f"WR:{ww / rc[r] * 100 if rc[r] else 0:.0f}%  avgR:{avg_r:+.2f}")

    print(f"\n  Trades (first 35):")
    print(f"  {'Dir':<5}{'Sym':<6}{'Entry':>8}{'Exit':>8}{'Stop':>8}"
          f"{'Sh':>7}{'Gross':>9}{'Net':>9}{'Bars':>5}{'R':>6}{'RV':>5} Reason")
    for t in trades[:35]:
        print(f"  {t.direction:<5}{t.symbol:<6}${t.entry_price:>7.2f}${t.exit_price:>7.2f}"
              f"{'':>8}{t.shares:>7}${t.gross_pnl:>8.2f}${t.net_pnl:>8.2f}"
              f"{t.hold_bars:>5}{t.r_multiple:>+5.1f}R{t.rvol:>4.1f}x {t.exit_reason}")

    print(f"\n  Rejections:")
    for r, c in rej.most_common(15):
        print(f"    {r:<25} {c:>8}")
    print(f"{'=' * 70}")

# ── Main ───────────────────────────────────────────────────────────

def main():
    from alpaca.data.timeframe import TimeFrame

    print("=" * 70)
    print("  ORB V4.2 — Deterministic, Session-Correct")
    print("  ET timezone | RTH only | Mechanical stops | Cash accounting")
    print("=" * 70)

    client = get_alpaca_client()
    if client is None:
        print("No Alpaca keys found. Exiting.")
        return

    syms = [
        "MARA", "MSTR", "RIOT", "COIN", "HOOD", "SMCI", "SOFI", "AFRM",
        "ENPH", "PLTR", "TSLA", "NVDA", "AMD", "SNAP", "ROKU",
        "AAPL", "AMZN", "META", "GOOG", "MSFT",
        "NFLX", "SQ", "SHOP", "UBER", "LYFT",
        "BA", "F", "GM", "NIO", "RIVN",
        "JPM", "BAC", "GS", "C", "WFC",
        "XOM", "CVX", "OXY", "SLB",
        "PFE", "MRNA", "ABBV",
        "WMT", "TGT", "COST",
        "DIS", "PYPL", "INTC", "MU", "QCOM",
        "FSLR", "W", "ADBE", "CRM",
    ]
    syms = list(dict.fromkeys(syms))

    # ── Daily bars ──────────────────────────────────────────────────
    print(f"\nFetching daily bars (SIP) for {len(syms)} symbols...")
    dstats = {}
    for s in syms:
        print(f"  {s}...", end=" ", flush=True)
        df = None
        try:
            df = fetch_alpaca(client, s, 90, TimeFrame.Day, feed="sip")
        except Exception as e:
            print(f"SIP fail ({e}), IEX...", end=" ")
            try:
                df = fetch_alpaca(client, s, 90, TimeFrame.Day, feed="iex")
            except Exception:
                pass
        if df is not None and len(df) >= 15:
            df = to_et(df)   # Convert to ET
            df["atr"] = daily_atr_series(df, 14)
            dstats[s] = df
            a = [x for x in df["atr"] if x]
            v = df["volume"].mean()
            print(f"ATR=${a[-1]:.2f} avgVol={v:,.0f}" if a else "ok")
        else:
            print("skip")

    # ── Minute bars ─────────────────────────────────────────────────
    print(f"\nFetching minute bars (SIP) for {len(syms)} symbols...")
    mdata = {}
    for s in syms:
        print(f"  {s}...", end=" ", flush=True)
        df = None
        try:
            df = fetch_alpaca(client, s, 60, TimeFrame.Minute, feed="sip")
        except Exception as e:
            print(f"SIP fail ({e}), IEX...", end=" ")
            try:
                df = fetch_alpaca(client, s, 60, TimeFrame.Minute, feed="iex")
            except Exception:
                pass
        if df is not None and len(df) > 100:
            df = to_et(df)           # Convert to ET
            pre_len = len(df)
            df = filter_rth(df)      # Keep only 09:30–16:00
            mdata[s] = df
            print(f"{pre_len}→{len(df)} RTH bars")
        else:
            print("skip")

    act = [s for s in syms if s in mdata and s in dstats]
    print(f"\nActive: {len(act)} of {len(syms)}")
    if not act:
        print("No symbols. Exiting.")
        return

    # Verify RTH filtering
    for s in act[:3]:
        df = mdata[s]
        times = df.index.time
        print(f"  {s}: first_bar={df.index[0]} last_bar={df.index[-1]}")
        assert all(t >= RTH_OPEN for t in times), f"{s} has pre-market bars!"
        assert all(t < RTH_CLOSE for t in times), f"{s} has after-hours bars!"
    print("  RTH filter verified ✓")

    # ── RVOL history (from RTH-filtered data) ────────────────────
    print("\nBuilding opening volume history for RVOL (RTH-filtered)...")
    opening_vols = build_opening_vol_history(mdata, orb_bars=5)
    for s in act[:5]:
        hist = opening_vols.get(s, {})
        vals = list(hist.values())
        if vals:
            print(f"  {s}: {len(vals)} sessions, avg_5min_vol={np.mean(vals):,.0f}")

    # ── Tests ───────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  TEST 1: Paper-matched (stop=0.10×ATR, RVOL>=2x, vol>1M, lev=4x)")
    print("=" * 70)
    t1, e1, r1 = run(mdata, dstats, opening_vols, act,
                      capital=100000, verbose=True,
                      rvol_threshold=2.0, max_positions=20,
                      stop_atr_pct=0.10, max_leverage=4.0,
                      min_avg_daily_vol=1_000_000)
    report(t1, e1, r1, 100000, "TEST 1: Paper-matched")

    print("\n" + "=" * 70)
    print("  TEST 2: Relaxed RVOL>=1.5x")
    print("=" * 70)
    t2, e2, r2 = run(mdata, dstats, opening_vols, act,
                      capital=100000,
                      rvol_threshold=1.5, max_positions=20,
                      stop_atr_pct=0.10, max_leverage=4.0,
                      min_avg_daily_vol=1_000_000)
    report(t2, e2, r2, 100000, "TEST 2: RVOL>=1.5x")

    print("\n" + "=" * 70)
    print("  TEST 3: No RVOL (isolate mechanical fixes)")
    print("=" * 70)
    t3, e3, r3 = run(mdata, dstats, opening_vols, act,
                      capital=100000,
                      rvol_threshold=0.0, max_positions=20,
                      stop_atr_pct=0.10, max_leverage=4.0,
                      min_avg_daily_vol=1_000_000)
    report(t3, e3, r3, 100000, "TEST 3: No RVOL")

    print("\n" + "=" * 70)
    print("  TEST 4: $1K, paper-matched")
    print("=" * 70)
    t4, e4, r4 = run(mdata, dstats, opening_vols, act,
                      capital=1000,
                      rvol_threshold=2.0, max_positions=20,
                      stop_atr_pct=0.10, max_leverage=4.0,
                      min_avg_daily_vol=1_000_000)
    report(t4, e4, r4, 1000, "TEST 4: $1K paper-matched")

    print("\n" + "=" * 70)
    print("  TEST 5: Wider stop 0.20×ATR (sensitivity)")
    print("=" * 70)
    t5, e5, r5 = run(mdata, dstats, opening_vols, act,
                      capital=100000,
                      rvol_threshold=2.0, max_positions=20,
                      stop_atr_pct=0.20, max_leverage=4.0,
                      min_avg_daily_vol=1_000_000)
    report(t5, e5, r5, 100000, "TEST 5: Stop 0.20×ATR")

    # Save trades
    for name, tl in [("v42_paper", t1), ("v42_relaxed", t2), ("v42_norvol", t3)]:
        if tl:
            os.makedirs("logs", exist_ok=True)
            rows = [{
                "sym": t.symbol, "dir": t.direction,
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "entry": t.entry_price, "exit": t.exit_price,
                "shares": t.shares, "gross": t.gross_pnl,
                "costs": t.costs, "net": t.net_pnl,
                "bars": t.hold_bars, "rvol": t.rvol,
                "r_mult": t.r_multiple, "reason": t.exit_reason
            } for t in tl]
            pd.DataFrame(rows).to_csv(f"logs/backtest_{name}_trades.csv", index=False)
            print(f"  Saved logs/backtest_{name}_trades.csv")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — V4.2 (Session-Correct)")
    print(f"{'=' * 70}")
    for nm, tl, cp in [
        ("Paper-match", t1, 100000),
        ("RVOL>=1.5x", t2, 100000),
        ("No RVOL", t3, 100000),
        ("$1K paper", t4, 1000),
        ("Stop=0.20×ATR", t5, 100000),
    ]:
        n = len(tl)
        p = sum(t.net_pnl for t in tl)
        w = len([t for t in tl if t.net_pnl > 0]) / n * 100 if n else 0
        avgr = np.mean([t.r_multiple for t in tl]) if tl else 0
        cs = sum(t.costs for t in tl)
        longs = len([t for t in tl if t.direction == "long"])
        shorts = len([t for t in tl if t.direction == "short"])
        print(f"  {nm:<15} {n:>4} trades  ${p:>10,.2f}  "
              f"WR:{w:>5.1f}%  avgR:{avgr:>+.2f}  "
              f"costs:${cs:>8,.2f}  L:{longs} S:{shorts}")

    print(f"\n  Infrastructure fixes in V4.2:")
    print(f"    1. All timestamps ET (America/New_York)")
    print(f"    2. RTH-only bars (09:30–16:00 ET)")
    print(f"    3. session_date from ET, not UTC")
    print(f"    4. bar_number = minutes since 09:30 ET")
    print(f"    5. ORB = bars 1-5 (09:30-09:34 ET)")
    print(f"    6. Deterministic fills (no randomness)")
    print(f"    7. Mechanical stop: fill at stop or open if gapped")
    print(f"    8. Cash accounting with buying power tracking")
    print(f"    9. Bounded transaction costs (spread + SEC + TAF)")
    print(f"   10. RVOL from RTH-filtered data")

if __name__ == "__main__":
    main()
