"""
Backtester V3.0 - ORB Momentum Strategy
========================================
Realistic simulation for the Opening Range Breakout strategy.

Key features:
- Proper day boundary detection and opening range construction
- Bid/ask execution (not last price)
- Variable spreads, slippage, commissions
- Intraday bar counting for time-based rules
- Position sizing with 1% risk per trade
- Full cost modeling including SEC/TAF fees
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import AppConfig, TradeSignal
from strategy.orb_momentum import ORBMomentumStrategy, Indicators
from utils.logger import StructuredLogger


@dataclass
class BacktestTrade:
    """Record of a single completed trade."""
    symbol: str
    entry_time: object
    exit_time: object
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
    entry_score: dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtester-specific settings."""
    starting_capital: float = 100000.0

    # Cost modeling
    commission_per_share: float = 0.0
    commission_per_order: float = 0.0
    sec_fee_per_dollar: float = 0.0000278
    taf_fee_per_share: float = 0.000166
    slippage_model: str = "fixed_pct"
    slippage_pct: float = 0.0005
    slippage_vol_multiplier: float = 0.1

    # Execution
    fill_probability: float = 0.95
    partial_fill_pct: float = 0.80

    # Risk
    max_position_pct: float = 0.95     # Can use almost all capital (1 pos)
    max_risk_per_trade_pct: float = 0.01  # 1% risk per trade
    max_daily_loss_pct: float = 0.03
    max_simultaneous_positions: int = 1

    # Time rules
    no_trade_first_bars: int = 5       # Opening range period
    force_exit_last_bars: int = 5      # Force exit last 5 bars


class CostModel:
    """Realistic execution cost calculator."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_entry_cost(self, price, shares, spread, volatility=0):
        spread_cost = (spread / 2) * shares
        if self.config.slippage_model == "volatility" and volatility > 0:
            slip_pct = volatility * self.config.slippage_vol_multiplier
        else:
            slip_pct = self.config.slippage_pct
        slippage = price * slip_pct * shares
        commission = max(self.config.commission_per_order,
                         self.config.commission_per_share * shares)
        return spread_cost + slippage + commission, {
            "spread_cost": spread_cost,
            "slippage": slippage,
            "commission": commission,
        }

    def calculate_exit_cost(self, price, shares, spread, volatility=0):
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
        return total, {
            "spread_cost": spread_cost,
            "slippage": slippage,
            "commission": commission,
            "sec_fee": sec_fee,
            "taf_fee": taf_fee,
        }


class ORBBacktester:
    """
    Backtester specifically designed for the ORB Momentum strategy.

    Handles:
    - Day boundary detection
    - Opening range construction from first 5 bars
    - Intraday bar counting
    - Proper time-based rules
    """

    def __init__(self, bt_config: BacktestConfig = None,
                 app_config: AppConfig = None):
        self.bt_config = bt_config or BacktestConfig()
        self.app_config = app_config or AppConfig()
        self.log = StructuredLogger("backtest", level="WARNING")
        self.cost_model = CostModel(self.bt_config)

    def run(self, data: Dict[str, pd.DataFrame],
            symbols: List[str] = None) -> dict:
        """
        Run backtest on historical data.

        data: dict of symbol -> DataFrame with columns:
              [open, high, low, close, volume, vwap]
              Optional: [bid, ask, spread]

        Returns: dict with trades, metrics, equity curve
        """
        if symbols is None:
            symbols = list(data.keys())

        # Initialize
        capital = self.bt_config.starting_capital
        equity_curve = [capital]
        all_trades: List[BacktestTrade] = []
        bt_positions: Dict[str, dict] = {}  # Backtester's position tracking
        daily_pnl = 0.0
        current_day = None
        trading_disabled_today = False
        consecutive_losses = 0

        # Strategy instance
        strategy = ORBMomentumStrategy(self.app_config, self.log)

        # Build unified timeline
        all_times = set()
        for sym in symbols:
            if sym in data:
                all_times.update(data[sym].index.tolist())
        all_times = sorted(all_times)

        if len(all_times) == 0:
            return {"error": "No data", "trades": [], "equity_curve": [capital],
                    "metrics": {"error": "No data"}}

        print(f"Backtesting {len(symbols)} symbols over "
              f"{len(all_times)} bars...")
        print(f"Date range: {all_times[0]} to {all_times[-1]}")
        print(f"Starting capital: ${capital:,.2f}")
        print("-" * 60)

        day_bar_count = 0
        bars_in_day = 390

        for bar_idx, timestamp in enumerate(all_times):
            # ── Day Boundary Detection ──────────────────────────────
            if hasattr(timestamp, 'date'):
                bar_date = timestamp.date()
            else:
                bar_date = pd.Timestamp(timestamp).date()

            if bar_date != current_day:
                # End of previous day — force close ALL positions
                if bt_positions:
                    for sym in list(bt_positions.keys()):
                        self._force_close(
                            sym, bt_positions, capital, all_trades,
                            data, timestamp, "end_of_day", strategy
                        )
                        trade = all_trades[-1]
                        capital += trade.net_pnl
                        daily_pnl += trade.net_pnl
                        if trade.net_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                # New day setup
                strategy.new_day(bar_date)
                current_day = bar_date
                daily_pnl = 0.0
                day_bar_count = 0
                trading_disabled_today = False

            day_bar_count += 1
            strategy.day_bar_count = day_bar_count
            strategy.increment_bar()

            # ── Build Opening Ranges ────────────────────────────────
            if day_bar_count <= strategy.OPENING_RANGE_BARS:
                for symbol in symbols:
                    if symbol not in data or timestamp not in data[symbol].index:
                        continue
                    row = data[symbol].loc[timestamp]
                    strategy.update_opening_range(
                        symbol,
                        high=float(row["high"]),
                        low=float(row["low"]),
                        open_price=float(row["open"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )

            # ── Force Exit Last Bars ────────────────────────────────
            is_eod = day_bar_count >= (bars_in_day
                                       - self.bt_config.force_exit_last_bars)
            if is_eod and bt_positions:
                for sym in list(bt_positions.keys()):
                    self._force_close(
                        sym, bt_positions, capital, all_trades,
                        data, timestamp, "eod_flatten", strategy
                    )
                    trade = all_trades[-1]
                    capital += trade.net_pnl
                    daily_pnl += trade.net_pnl
                    if trade.net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                continue

            # ── Daily Loss Limit ────────────────────────────────────
            if daily_pnl <= -(capital * self.bt_config.max_daily_loss_pct):
                if not trading_disabled_today:
                    trading_disabled_today = True
                continue

            if consecutive_losses >= 5:
                continue

            # ── Process Each Symbol ─────────────────────────────────
            for symbol in symbols:
                if symbol not in data or timestamp not in data[symbol].index:
                    continue

                df = data[symbol]
                row = df.loc[timestamp]

                # Build history up to this point (no lookahead)
                hist = df.loc[:timestamp]
                if len(hist) < 20:
                    continue

                # Limit lookback for performance
                lookback = min(200, len(hist))
                recent = hist.tail(lookback)

                closes = recent["close"].tolist()
                highs = recent["high"].tolist()
                lows = recent["low"].tolist()
                volumes = recent["volume"].tolist()
                opens = recent["open"].tolist() if "open" in recent else None
                vwaps = recent["vwap"].tolist() if "vwap" in recent else None

                # Update strategy indicators
                strategy.update_bars(symbol, closes, highs, lows, volumes,
                                     opens=opens, vwaps=vwaps)

                has_position = symbol in bt_positions

                # Generate signal
                signal, meta = strategy.generate_signal(
                    symbol, has_position, current_bar=bar_idx
                )

                # Get execution prices
                price = float(row["close"])
                if "spread" in row and not pd.isna(row.get("spread", float('nan'))):
                    spread = float(row["spread"])
                else:
                    spread = price * 0.001

                if "bid" in row and not pd.isna(row.get("bid", float('nan'))):
                    bid = float(row["bid"])
                    ask = float(row["ask"])
                else:
                    bid = price - spread / 2
                    ask = price + spread / 2

                volatility = (float(np.std(closes[-20:])) / price
                              if len(closes) >= 20 else 0.01)

                # ── EXITS ───────────────────────────────────────────
                if has_position and signal == TradeSignal.LONG_EXIT:
                    pos = bt_positions[symbol]
                    exit_cost, cost_detail = self.cost_model.calculate_exit_cost(
                        bid, pos["shares"], spread, volatility
                    )
                    gross_pnl = (bid - pos["entry_price"]) * pos["shares"]
                    net_pnl = gross_pnl - exit_cost - pos["entry_cost"]

                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        entry_price=pos["entry_price"],
                        exit_price=bid,
                        shares=pos["shares"],
                        side="long",
                        gross_pnl=gross_pnl,
                        commission=cost_detail["commission"],
                        slippage_cost=cost_detail["slippage"],
                        spread_cost=cost_detail["spread_cost"] + pos["entry_spread_cost"],
                        net_pnl=net_pnl,
                        hold_bars=bar_idx - pos["entry_bar"],
                        entry_reason=pos["entry_reason"],
                        exit_reason=meta.get("reason", "signal"),
                    )
                    all_trades.append(trade)
                    capital += net_pnl
                    daily_pnl += net_pnl
                    del bt_positions[symbol]

                    if net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                # ── ENTRIES ─────────────────────────────────────────
                elif (not has_position
                      and signal == TradeSignal.LONG_ENTRY
                      and not trading_disabled_today):

                    if len(bt_positions) >= self.bt_config.max_simultaneous_positions:
                        continue

                    # Fill probability
                    if np.random.random() > self.bt_config.fill_probability:
                        continue

                    # Position sizing
                    stop_price = strategy.get_stop_price(symbol)
                    if stop_price is None:
                        continue

                    stop_f = float(stop_price)
                    if stop_f >= ask:
                        continue

                    risk_per_share = ask - stop_f
                    if risk_per_share <= 0:
                        continue

                    max_risk = capital * self.bt_config.max_risk_per_trade_pct
                    shares = int(max_risk / risk_per_share)

                    max_pos = capital * self.bt_config.max_position_pct
                    shares_from_pos = int(max_pos / ask)
                    shares_from_capital = int(capital / ask)
                    shares = min(shares, shares_from_pos, shares_from_capital)

                    if shares <= 0:
                        continue

                    # Entry cost
                    entry_cost, cost_detail = self.cost_model.calculate_entry_cost(
                        ask, shares, spread, volatility
                    )

                    bt_positions[symbol] = {
                        "entry_price": ask,
                        "shares": shares,
                        "entry_time": timestamp,
                        "entry_bar": bar_idx,
                        "entry_cost": entry_cost,
                        "entry_spread_cost": cost_detail["spread_cost"],
                        "stop_price": stop_f,
                        "entry_reason": meta.get("reason", "signal"),
                    }

            # Track equity (mark-to-market)
            mtm = capital
            for sym, pos in bt_positions.items():
                if sym in data and timestamp in data[sym].index:
                    current = float(data[sym].loc[timestamp, "close"])
                    mtm += (current - pos["entry_price"]) * pos["shares"]
            equity_curve.append(mtm)

        # Close remaining positions
        if bt_positions and len(all_times) > 0:
            last_ts = all_times[-1]
            for sym in list(bt_positions.keys()):
                self._force_close(
                    sym, bt_positions, capital, all_trades,
                    data, last_ts, "backtest_end", strategy
                )
                capital += all_trades[-1].net_pnl

        metrics = self._calculate_metrics(
            all_trades, equity_curve, self.bt_config.starting_capital
        )

        return {
            "trades": all_trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "config": self.bt_config,
        }

    def _force_close(self, symbol, positions, capital, all_trades,
                     data, timestamp, reason, strategy):
        """Force close a position."""
        if symbol not in positions:
            return
        pos = positions[symbol]
        df = data.get(symbol)

        if df is not None and timestamp in df.index:
            row = df.loc[timestamp]
            price = float(row["close"])
            spread = (float(row["spread"]) if "spread" in row
                      and not pd.isna(row.get("spread", float('nan')))
                      else price * 0.001)
        else:
            price = pos["entry_price"]
            spread = price * 0.001

        exit_cost, cost_detail = self.cost_model.calculate_exit_cost(
            price, pos["shares"], spread
        )
        gross_pnl = (price - pos["entry_price"]) * pos["shares"]
        net_pnl = gross_pnl - exit_cost - pos["entry_cost"]

        trade = BacktestTrade(
            symbol=symbol,
            entry_time=pos["entry_time"],
            exit_time=timestamp,
            entry_price=pos["entry_price"],
            exit_price=price,
            shares=pos["shares"],
            side="long",
            gross_pnl=gross_pnl,
            commission=cost_detail["commission"],
            slippage_cost=cost_detail["slippage"],
            spread_cost=cost_detail["spread_cost"],
            net_pnl=net_pnl,
            hold_bars=0,
            entry_reason=pos["entry_reason"],
            exit_reason=reason,
        )
        all_trades.append(trade)
        del positions[symbol]
        strategy.force_close(symbol)

    def _calculate_metrics(self, trades, equity_curve, starting_capital):
        """Comprehensive performance metrics."""
        if not trades:
            return {"error": "No trades executed", "total_trades": 0}

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
        profit_factor = (sum(wins) / abs(sum(losses))
                         if losses and sum(losses) != 0 else float('inf'))

        # Equity metrics
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])

        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0
        max_dd_dollar = float(np.max(peak - eq)) if len(peak) > 0 else 0

        # Sharpe (annualized from 1-min bars)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns)
                           * np.sqrt(252 * 390))
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside)
                            * np.sqrt(252 * 390))
        else:
            sortino = 0.0

        # Cost breakdown
        total_spread = sum(t.spread_cost for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        total_commission = sum(t.commission for t in trades)
        avg_hold = np.mean([t.hold_bars for t in trades])

        # Consecutive losses
        max_consec = 0
        streak = 0
        for pnl in net_pnls:
            if pnl <= 0:
                streak += 1
                max_consec = max(max_consec, streak)
            else:
                streak = 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t.exit_reason
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "total_pnl": 0, "wins": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["total_pnl"] += t.net_pnl
            if t.net_pnl > 0:
                exit_reasons[r]["wins"] += 1

        # R:R achieved
        actual_rr_list = []
        for t in trades:
            if t.net_pnl > 0 and avg_loss != 0:
                actual_rr_list.append(t.net_pnl / abs(avg_loss))
        avg_rr = np.mean(actual_rr_list) if actual_rr_list else 0

        return {
            "total_trades": len(trades),
            "total_net_pnl": round(total_net, 2),
            "total_gross_pnl": round(total_gross, 2),
            "total_costs": round(total_costs, 2),
            "return_pct": round(total_net / starting_capital * 100, 2),

            "win_rate": round(win_rate, 4),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(np.mean(net_pnls), 2),
            "avg_rr_achieved": round(avg_rr, 2),

            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "max_drawdown_dollar": round(max_dd_dollar, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_consecutive_losses": max_consec,

            "total_spread_cost": round(total_spread, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "total_commissions": round(total_commission, 2),
            "cost_per_trade": round(total_costs / len(trades), 2) if trades else 0,
            "cost_pct_of_gross": round(
                abs(total_costs / total_gross * 100), 2
            ) if total_gross != 0 else 0,

            "avg_hold_bars": round(avg_hold, 1),
            "avg_shares": round(np.mean([t.shares for t in trades]), 1),
            "exit_reasons": exit_reasons,
        }

    def print_report(self, results):
        """Formatted backtest report."""
        m = results.get("metrics", {})
        if "error" in m:
            print(f"\n  BACKTEST: {m.get('error', 'Unknown error')}")
            print(f"  Total trades: {m.get('total_trades', 0)}")
            return

        print("\n" + "=" * 65)
        print("  BACKTEST REPORT - V3.0 ORB Momentum Strategy")
        print("=" * 65)

        print(f"\n  PERFORMANCE")
        print(f"  {'Total Net P&L:':<30} ${m['total_net_pnl']:>12,.2f}")
        print(f"  {'Return:':<30} {m['return_pct']:>12.2f}%")
        print(f"  {'Sharpe Ratio:':<30} {m['sharpe_ratio']:>12.2f}")
        print(f"  {'Sortino Ratio:':<30} {m['sortino_ratio']:>12.2f}")
        print(f"  {'Profit Factor:':<30} {m['profit_factor']:>12.2f}")
        print(f"  {'Expectancy ($/trade):':<30} ${m['expectancy']:>12.2f}")
        print(f"  {'Avg R:R Achieved:':<30} {m['avg_rr_achieved']:>12.2f}")

        print(f"\n  TRADES")
        print(f"  {'Total Trades:':<30} {m['total_trades']:>12}")
        print(f"  {'Win Rate:':<30} {m['win_rate']*100:>12.1f}%")
        print(f"  {'Wins / Losses:':<30} {m['wins']:>5} / {m['losses']}")
        print(f"  {'Avg Win:':<30} ${m['avg_win']:>12.2f}")
        print(f"  {'Avg Loss:':<30} ${m['avg_loss']:>12.2f}")
        print(f"  {'Avg Hold (bars):':<30} {m['avg_hold_bars']:>12.1f}")

        print(f"\n  RISK")
        print(f"  {'Max Drawdown:':<30} {m['max_drawdown_pct']:>12.2f}%")
        print(f"  {'Max Drawdown ($):':<30} ${m['max_drawdown_dollar']:>12,.2f}")
        print(f"  {'Max Consec Losses:':<30} {m['max_consecutive_losses']:>12}")

        print(f"\n  COSTS")
        print(f"  {'Total Costs:':<30} ${m['total_costs']:>12,.2f}")
        print(f"  {'Spread Cost:':<30} ${m['total_spread_cost']:>12,.2f}")
        print(f"  {'Slippage Cost:':<30} ${m['total_slippage_cost']:>12,.2f}")
        print(f"  {'Cost/Trade:':<30} ${m['cost_per_trade']:>12.2f}")
        print(f"  {'Costs % of Gross:':<30} {m['cost_pct_of_gross']:>12.1f}%")

        # Exit reason breakdown
        print(f"\n  EXIT REASONS")
        for reason, info in sorted(m.get("exit_reasons", {}).items(),
                                    key=lambda x: x[1]["count"], reverse=True):
            wr = info["wins"] / info["count"] * 100 if info["count"] > 0 else 0
            print(f"  {reason:<25} {info['count']:>3} trades  "
                  f"${info['total_pnl']:>10,.2f}  WR:{wr:>5.1f}%")

        # Verdict
        print(f"\n  VERDICT")
        for line in self._grade(m):
            print(f"  {line}")
        print("=" * 65)

    def _grade(self, m):
        """Honest strategy assessment."""
        v = []
        if m["total_trades"] < 20:
            v.append("⚠ Too few trades for statistical significance (<20)")
        elif m["total_trades"] < 50:
            v.append("⚠ Marginal sample size (20-50 trades)")
        else:
            v.append("✓ Sufficient trade count for analysis")

        if m["win_rate"] < 0.35:
            v.append("✗ Win rate below 35%")
        elif m["win_rate"] >= 0.45:
            v.append("✓ Win rate above 45%")

        if m["profit_factor"] < 1.0:
            v.append("✗ Profit factor < 1.0 (LOSING MONEY)")
        elif m["profit_factor"] < 1.3:
            v.append("⚠ Profit factor < 1.3 (thin edge)")
        elif m["profit_factor"] >= 1.5:
            v.append("✓ Profit factor > 1.5 (healthy edge)")

        if m["max_drawdown_pct"] > 10:
            v.append("✗ Max drawdown > 10%")
        elif m["max_drawdown_pct"] > 5:
            v.append("⚠ Max drawdown > 5%")
        else:
            v.append("✓ Max drawdown contained")

        if m["sharpe_ratio"] < 0.5:
            v.append("✗ Sharpe < 0.5")
        elif m["sharpe_ratio"] >= 1.5:
            v.append("✓ Sharpe > 1.5 (strong)")

        if m["cost_pct_of_gross"] > 50:
            v.append("✗ Costs > 50% of gross (edge too thin)")

        if m["avg_rr_achieved"] < 1.0:
            v.append("⚠ Avg R:R < 1.0 (winners too small)")
        elif m["avg_rr_achieved"] >= 2.0:
            v.append("✓ Avg R:R ≥ 2.0 (good risk/reward)")

        fails = sum(1 for x in v if x.startswith("✗"))
        if fails == 0:
            v.append("\n  OVERALL: Strategy shows potential → paper trade next")
        else:
            v.append(f"\n  OVERALL: {fails} critical issues → keep iterating")

        return v
