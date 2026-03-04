"""
Backtester V2.0 - Realistic simulation with multi-timeframe support.
Executes at bid/ask, models spreads, slippage, commissions, partial fills.
Feeds SPY to regime filter. No lookahead bias.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from decimal import Decimal
from typing import Dict, List
from dataclasses import dataclass

from config.settings import AppConfig, TradeSignal
from strategy.mean_reversion import MeanReversionStrategy
from utils.logger import StructuredLogger


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    side: str
    gross_pnl: float
    commission: float
    slippage_cost: float
    spread_cost: float
    net_pnl: float
    hold_bars: int
    entry_reason: str
    exit_reason: str


@dataclass
class BacktestConfig:
    starting_capital: float = 100000.0
    commission_per_share: float = 0.0
    commission_per_order: float = 0.0
    sec_fee_per_dollar: float = 0.0000278
    taf_fee_per_share: float = 0.000166
    slippage_model: str = "fixed_pct"
    slippage_pct: float = 0.0005
    slippage_vol_multiplier: float = 0.1
    fill_probability: float = 0.95
    partial_fill_pct: float = 0.80
    max_position_pct: float = 0.40
    max_risk_per_trade_pct: float = 0.03
    max_daily_loss_pct: float = 0.03
    max_simultaneous_positions: int = 3
    no_trade_first_bars: int = 5
    force_exit_last_bars: int = 5


class CostModel:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def entry_cost(self, price, shares, spread, volatility=0):
        spread_cost = (spread / 2) * shares
        if self.config.slippage_model == "volatility" and volatility > 0:
            slip_pct = volatility * self.config.slippage_vol_multiplier
        else:
            slip_pct = self.config.slippage_pct
        slippage = price * slip_pct * shares
        commission = max(self.config.commission_per_order,
                         self.config.commission_per_share * shares)
        return spread_cost + slippage + commission, {
            "spread_cost": spread_cost, "slippage": slippage,
            "commission": commission, "slip_pct": slip_pct}

    def exit_cost(self, price, shares, spread, volatility=0):
        spread_cost = (spread / 2) * shares
        if self.config.slippage_model == "volatility" and volatility > 0:
            slip_pct = volatility * self.config.slippage_vol_multiplier
        else:
            slip_pct = self.config.slippage_pct
        slippage = price * slip_pct * shares
        commission = max(self.config.commission_per_order,
                         self.config.commission_per_share * shares)
        sec_fee = price * shares * self.config.sec_fee_per_dollar
        taf_fee = shares * self.config.taf_fee_per_share
        total = spread_cost + slippage + commission + sec_fee + taf_fee
        return total, {"spread_cost": spread_cost, "slippage": slippage,
                        "commission": commission}


class Backtester:
    def __init__(self, bt_config=None, app_config=None):
        self.bt_config = bt_config or BacktestConfig()
        self.app_config = app_config or AppConfig()
        self.log = StructuredLogger("backtest", level="WARNING")
        self.cost_model = CostModel(self.bt_config)

    def run(self, data: Dict[str, pd.DataFrame],
            symbols: List[str] = None) -> dict:
        if symbols is None:
            symbols = list(data.keys())

        capital = self.bt_config.starting_capital
        equity_curve = [capital]
        all_trades = []
        positions = {}
        daily_pnl = 0.0
        current_day = None
        trading_disabled_today = False
        consecutive_losses = 0

        strategy = MeanReversionStrategy(self.app_config, self.log)

        # Build unified timeline
        all_times = set()
        for sym in symbols:
            if sym in data:
                all_times.update(data[sym].index.tolist())
        # Also include SPY times for regime filter
        if "SPY" not in symbols and "SPY" in data:
            all_times.update(data["SPY"].index.tolist())
        all_times = sorted(all_times)

        if len(all_times) == 0:
            return {"error": "No data", "trades": [], "equity_curve": [],
                    "metrics": {"error": "No data"}}

        print(f"Backtesting {len(symbols)} symbols over "
              f"{len(all_times)} bars...")
        print(f"Date range: {all_times[0]} to {all_times[-1]}")
        print(f"Starting capital: ${capital:,.2f}")
        print("-" * 60)

        day_bar_count = 0
        bars_in_day = 390

        for bar_idx, timestamp in enumerate(all_times):
            if hasattr(timestamp, 'date'):
                bar_date = timestamp.date()
            else:
                bar_date = pd.Timestamp(timestamp).date()

            # ── New day ────────────────────────────────────────────
            if bar_date != current_day:
                if positions:
                    for sym in list(positions.keys()):
                        self._force_close(sym, positions, capital,
                                          all_trades, data, timestamp,
                                          "end_of_day")
                        capital += all_trades[-1].net_pnl
                        daily_pnl += all_trades[-1].net_pnl
                        if all_trades[-1].net_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                current_day = bar_date
                daily_pnl = 0.0
                day_bar_count = 0
                trading_disabled_today = False

            day_bar_count += 1

            # ── Regime filter: feed SPY data ───────────────────────
            if "SPY" in data and timestamp in data["SPY"].index:
                spy_hist = data["SPY"].loc[:timestamp]
                if len(spy_hist) >= 50:
                    spy_c = spy_hist["close"].tolist()[-200:]
                    spy_h = spy_hist["high"].tolist()[-200:]
                    spy_l = spy_hist["low"].tolist()[-200:]
                    spy_v = spy_hist["volume"].tolist()[-200:]
                    strategy.update_regime(spy_c, spy_h, spy_l, spy_v)

            # Skip first N bars
            if day_bar_count <= self.bt_config.no_trade_first_bars:
                continue

            # Force exit at end of day
            is_eod = day_bar_count >= (bars_in_day
                                       - self.bt_config.force_exit_last_bars)
            if is_eod and positions:
                for sym in list(positions.keys()):
                    self._force_close(sym, positions, capital,
                                      all_trades, data, timestamp,
                                      "eod_flatten")
                    capital += all_trades[-1].net_pnl
                    daily_pnl += all_trades[-1].net_pnl
                    if all_trades[-1].net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                continue

            # Daily loss limit
            if daily_pnl <= -(capital * self.bt_config.max_daily_loss_pct):
                if not trading_disabled_today:
                    trading_disabled_today = True
                continue

            if consecutive_losses >= 5:
                continue

            # ── Process each symbol ────────────────────────────────
            for symbol in symbols:
                if symbol not in data:
                    continue
                df = data[symbol]
                if timestamp not in df.index:
                    continue

                row = df.loc[timestamp]
                hist = df.loc[:timestamp]
                if len(hist) < 30:
                    continue

                # Limit lookback for performance
                lookback = min(200, len(hist))
                recent = hist.tail(lookback)

                closes = recent["close"].tolist()
                highs = recent["high"].tolist()
                lows = recent["low"].tolist()
                volumes = recent["volume"].tolist()

                # Update strategy
                strategy.update_bars(symbol, closes, highs, lows, volumes)

                has_position = symbol in positions
                signal, meta = strategy.generate_signal(
                    symbol, has_position, current_bar=bar_idx)

                # Price data for execution
                price = float(row["close"])
                if "spread" in row and not pd.isna(row["spread"]):
                    spread = float(row["spread"])
                else:
                    spread = price * 0.001
                if "bid" in row and not pd.isna(row["bid"]):
                    bid = float(row["bid"])
                    ask = float(row["ask"])
                else:
                    bid = price - spread / 2
                    ask = price + spread / 2

                vol = float(np.std(closes[-20:])) / price \
                    if len(closes) >= 20 else 0.01

                # ── EXITS ──────────────────────────────────────────
                if has_position and signal == TradeSignal.LONG_EXIT:
                    pos = positions[symbol]
                    exit_cost, cd = self.cost_model.exit_cost(
                        bid, pos["shares"], spread, vol)
                    gross = (bid - pos["entry_price"]) * pos["shares"]
                    net = gross - exit_cost - pos["entry_cost"]

                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        entry_price=pos["entry_price"],
                        exit_price=bid,
                        shares=pos["shares"],
                        side="long",
                        gross_pnl=gross,
                        commission=cd["commission"],
                        slippage_cost=cd["slippage"],
                        spread_cost=cd["spread_cost"] + pos["entry_spread"],
                        net_pnl=net,
                        hold_bars=bar_idx - pos["entry_bar"],
                        entry_reason=pos["entry_reason"],
                        exit_reason=meta.get("reason", "signal"),
                    )
                    all_trades.append(trade)
                    capital += net
                    daily_pnl += net
                    del positions[symbol]
                    if net < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                # ── ENTRIES ─────────────────────────────────────────
                elif (not has_position
                      and signal == TradeSignal.LONG_ENTRY
                      and not trading_disabled_today):

                    if len(positions) >= self.bt_config.max_simultaneous_positions:
                        continue

                    if np.random.random() > self.bt_config.fill_probability:
                        continue

                    stop_price = strategy.get_stop_price(symbol)
                    if stop_price is None:
                        continue
                    stop_f = float(stop_price)
                    if stop_f >= ask:
                        continue

                    risk_ps = ask - stop_f
                    max_risk = capital * self.bt_config.max_risk_per_trade_pct
                    sh_risk = int(max_risk / risk_ps) if risk_ps > 0 else 0
                    max_pos = capital * self.bt_config.max_position_pct
                    sh_pos = int(max_pos / ask)
                    sh_cap = int(capital / ask)
                    shares = min(sh_risk, sh_pos, sh_cap)
                    if shares <= 0:
                        continue

                    if np.random.random() > 0.5:
                        shares = max(1, int(
                            shares * self.bt_config.partial_fill_pct))

                    entry_cost, ecd = self.cost_model.entry_cost(
                        ask, shares, spread, vol)

                    ev, _ = strategy.estimate_expected_value(
                        entry_price=ask,
                        vwap=strategy.get_state(symbol)["vwap"],
                        stop_price=stop_f,
                        spread_cost=spread,
                        slippage_est=ask * self.bt_config.slippage_pct,
                        win_rate=0.50)

                    if float(ev) * shares < float(
                            self.app_config.strategy
                            .min_expected_value_per_trade):
                        continue

                    positions[symbol] = {
                        "entry_price": ask, "shares": shares,
                        "entry_time": timestamp, "entry_bar": bar_idx,
                        "entry_cost": entry_cost,
                        "entry_spread": ecd["spread_cost"],
                        "stop": stop_f,
                        "entry_reason": meta.get("reason", "signal"),
                    }

            # Equity mark-to-market
            mtm = capital
            for sym, pos in positions.items():
                if sym in data and timestamp in data[sym].index:
                    cur = float(data[sym].loc[timestamp, "close"])
                    mtm += (cur - pos["entry_price"]) * pos["shares"]
            equity_curve.append(mtm)

        # Close remaining
        if positions:
            last_ts = all_times[-1]
            for sym in list(positions.keys()):
                self._force_close(sym, positions, capital,
                                  all_trades, data, last_ts, "backtest_end")
                capital += all_trades[-1].net_pnl

        metrics = self._calc_metrics(all_trades, equity_curve,
                                     self.bt_config.starting_capital)
        return {"trades": all_trades, "equity_curve": equity_curve,
                "metrics": metrics, "config": self.bt_config}

    def _force_close(self, symbol, positions, capital, all_trades,
                     data, timestamp, reason):
        if symbol not in positions:
            return
        pos = positions[symbol]
        df = data.get(symbol)
        if df is not None and timestamp in df.index:
            row = df.loc[timestamp]
            price = float(row["close"])
            spread = float(row["spread"]) if "spread" in row and \
                not pd.isna(row["spread"]) else price * 0.001
        else:
            price = pos["entry_price"]
            spread = price * 0.001

        exit_cost, cd = self.cost_model.exit_cost(
            price, pos["shares"], spread)
        gross = (price - pos["entry_price"]) * pos["shares"]
        net = gross - exit_cost - pos["entry_cost"]

        trade = BacktestTrade(
            symbol=symbol, entry_time=pos["entry_time"],
            exit_time=timestamp, entry_price=pos["entry_price"],
            exit_price=price, shares=pos["shares"], side="long",
            gross_pnl=gross, commission=cd["commission"],
            slippage_cost=cd["slippage"],
            spread_cost=cd["spread_cost"],
            net_pnl=net, hold_bars=0,
            entry_reason=pos["entry_reason"], exit_reason=reason)
        all_trades.append(trade)
        del positions[symbol]

    def _calc_metrics(self, trades, equity_curve, starting_capital):
        if not trades:
            return {"error": "No trades executed"}

        net_pnls = [t.net_pnl for t in trades]
        gross_pnls = [t.gross_pnl for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]

        total_net = sum(net_pnls)
        total_gross = sum(gross_pnls)
        total_costs = total_gross - total_net
        win_rate = len(wins) / len(net_pnls) if net_pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 \
            else float('inf')

        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        max_dd_dollar = float(np.max(peak - eq)) if len(peak) > 0 else 0

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns)
                           * np.sqrt(252 * 390))
        else:
            sharpe = 0.0

        downside = returns[returns < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside)
                            * np.sqrt(252 * 390))
        else:
            sortino = 0.0

        max_cl = 0
        streak = 0
        for p in net_pnls:
            if p <= 0:
                streak += 1
                max_cl = max(max_cl, streak)
            else:
                streak = 0

        return {
            "total_trades": len(trades),
            "total_net_pnl": round(total_net, 2),
            "total_gross_pnl": round(total_gross, 2),
            "total_costs": round(total_costs, 2),
            "return_pct": round(total_net / starting_capital * 100, 2),
            "win_rate": round(win_rate, 4),
            "wins": len(wins), "losses": len(losses),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "expectancy": round(np.mean(net_pnls), 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "max_drawdown_dollar": round(max_dd_dollar, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_consecutive_losses": max_cl,
            "total_spread_cost": round(sum(t.spread_cost for t in trades), 2),
            "total_slippage_cost": round(
                sum(t.slippage_cost for t in trades), 2),
            "total_commissions": round(
                sum(t.commission for t in trades), 2),
            "cost_per_trade": round(total_costs / len(trades), 2),
            "cost_pct_of_gross": round(
                abs(total_costs / total_gross * 100), 2
            ) if total_gross != 0 else 0,
            "avg_hold_bars": round(
                np.mean([t.hold_bars for t in trades]), 1),
            "avg_shares": round(
                np.mean([t.shares for t in trades]), 1),
        }

    def print_report(self, results):
        if "error" in results.get("metrics", {}):
            print(f"\nBACKTEST FAILED: {results['metrics']['error']}")
            return

        m = results["metrics"]
        print("\n" + "=" * 60)
        print("  BACKTEST REPORT - V2.0 MTF VWAP Mean Reversion")
        print("=" * 60)

        print(f"\n  PERFORMANCE")
        print(f"  {'Total Net P&L:':<30} ${m['total_net_pnl']:>12,.2f}")
        print(f"  {'Return:':<30} {m['return_pct']:>12.2f}%")
        print(f"  {'Sharpe Ratio:':<30} {m['sharpe_ratio']:>12.2f}")
        print(f"  {'Sortino Ratio:':<30} {m['sortino_ratio']:>12.2f}")
        print(f"  {'Profit Factor:':<30} {m['profit_factor']:>12.2f}")
        print(f"  {'Expectancy ($/trade):':<30} ${m['expectancy']:>12.2f}")

        print(f"\n  TRADES")
        print(f"  {'Total Trades:':<30} {m['total_trades']:>12}")
        print(f"  {'Win Rate:':<30} {m['win_rate']*100:>12.1f}%")
        print(f"  {'Wins / Losses:':<30} "
              f"{m['wins']:>5} / {m['losses']}")
        print(f"  {'Avg Win:':<30} ${m['avg_win']:>12.2f}")
        print(f"  {'Avg Loss:':<30} ${m['avg_loss']:>12.2f}")
        print(f"  {'Avg Hold (bars):':<30} {m['avg_hold_bars']:>12.1f}")

        print(f"\n  RISK")
        print(f"  {'Max Drawdown:':<30} "
              f"{m['max_drawdown_pct']:>12.2f}%")
        print(f"  {'Max Drawdown ($):':<30} "
              f"${m['max_drawdown_dollar']:>12,.2f}")
        print(f"  {'Max Consec Losses:':<30} "
              f"{m['max_consecutive_losses']:>12}")

        print(f"\n  COSTS")
        print(f"  {'Total Costs:':<30} ${m['total_costs']:>12,.2f}")
        print(f"  {'Spread Cost:':<30} "
              f"${m['total_spread_cost']:>12,.2f}")
        print(f"  {'Slippage Cost:':<30} "
              f"${m['total_slippage_cost']:>12,.2f}")
        print(f"  {'Commissions:':<30} "
              f"${m['total_commissions']:>12,.2f}")
        print(f"  {'Cost/Trade:':<30} ${m['cost_per_trade']:>12.2f}")
        print(f"  {'Costs % of Gross:':<30} "
              f"{m['cost_pct_of_gross']:>12.1f}%")

        print(f"\n  VERDICT")
        for line in self._grade(m):
            print(f"  {line}")
        print("=" * 60)

    def _grade(self, m):
        v = []
        if m["total_trades"] < 30:
            v.append("WARNING: <30 trades, low statistical significance")
        if m["win_rate"] < 0.40:
            v.append("FAIL: Win rate below 40%")
        elif m["win_rate"] >= 0.50:
            v.append("PASS: Win rate above 50%")
        if m["profit_factor"] < 1.0:
            v.append("FAIL: Profit factor < 1.0 (losing money)")
        elif m["profit_factor"] < 1.3:
            v.append("WARNING: Profit factor < 1.3 (thin edge)")
        elif m["profit_factor"] >= 1.5:
            v.append("PASS: Profit factor > 1.5")
        if m["max_drawdown_pct"] > 10:
            v.append("FAIL: Max drawdown > 10%")
        elif m["max_drawdown_pct"] > 5:
            v.append("WARNING: Max drawdown > 5%")
        if m["sharpe_ratio"] < 0.5:
            v.append("FAIL: Sharpe < 0.5")
        elif m["sharpe_ratio"] >= 1.5:
            v.append("PASS: Sharpe > 1.5 (strong)")
        if m["cost_pct_of_gross"] > 50:
            v.append("FAIL: Costs > 50% of gross")
        if m["max_consecutive_losses"] >= 8:
            v.append("WARNING: 8+ consecutive losses")
        if not any("FAIL" in x for x in v):
            v.append("OVERALL: Strategy shows potential")
        else:
            v.append("OVERALL: Needs improvement")
        return v
