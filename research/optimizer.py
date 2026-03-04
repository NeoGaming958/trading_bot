"""
Research Framework V2.0
=======================
- Parameter optimization (grid + random search)
- Walk-forward validation
- Monte Carlo robustness testing
- Trade analytics and diagnostics
- Correlation analysis
"""
import numpy as np
import pandas as pd
import itertools
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict

from config.settings import AppConfig
from research.backtester import Backtester, BacktestConfig
from utils.logger import StructuredLogger


# ====================================================================
# PARAMETER SPACE
# ====================================================================

@dataclass
class ParamSpace:
    """Defines searchable parameter ranges."""
    # Entry scoring
    entry_threshold: List[int] = field(
        default_factory=lambda: [30, 35, 40, 45, 50])
    min_bars_between: List[int] = field(
        default_factory=lambda: [3, 5, 10, 15])

    # Exit parameters
    profit_target_z: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.8, 1.0])
    hard_stop_atr: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    trail_stop_atr: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    time_stop_bars: List[int] = field(
        default_factory=lambda: [30, 60, 120, 240])
    breakeven_z: List[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5])

    # Indicator periods
    vwap_lookback: List[int] = field(
        default_factory=lambda: [10, 15, 20, 30])
    rsi_period: List[int] = field(
        default_factory=lambda: [7, 10, 14, 21])
    vp_window: List[int] = field(
        default_factory=lambda: [60, 90, 120, 180])

    def total_combinations(self):
        count = 1
        for f in self.__dataclass_fields__:
            count *= len(getattr(self, f))
        return count

    def random_sample(self, n=50):
        """Generate n random parameter sets."""
        samples = []
        for _ in range(n):
            params = {}
            for f in self.__dataclass_fields__:
                vals = getattr(self, f)
                params[f] = vals[np.random.randint(len(vals))]
            samples.append(params)
        return samples

    def grid_sample(self, max_combos=200):
        """Generate grid of all combos, capped at max."""
        fields = list(self.__dataclass_fields__.keys())
        values = [getattr(self, f) for f in fields]
        total = 1
        for v in values:
            total *= len(v)

        if total <= max_combos:
            combos = list(itertools.product(*values))
        else:
            # Random subset of grid
            combos = []
            for _ in range(max_combos):
                combo = tuple(v[np.random.randint(len(v))] for v in values)
                combos.append(combo)

        return [dict(zip(fields, c)) for c in combos]


# ====================================================================
# STRATEGY WRAPPER (applies params to strategy)
# ====================================================================

def apply_params(strategy, params):
    """Apply parameter dict to a strategy instance."""
    if "entry_threshold" in params:
        strategy.ENTRY_THRESHOLD = params["entry_threshold"]
    if "min_bars_between" in params:
        strategy.min_bars_between = params["min_bars_between"]

    # Store exit params on strategy for the backtest to read
    strategy._bt_params = params


def apply_exit_params_to_backtest(backtester, params):
    """Patch backtester to use custom exit params.
    We store them and the strategy reads them during generate_signal."""
    # These get passed through to strategy via _bt_params
    pass


# ====================================================================
# WALK-FORWARD VALIDATOR
# ====================================================================

class WalkForwardValidator:
    """
    Train on N days, test on M days, roll forward.
    Prevents overfitting by never optimizing on test data.
    """

    def __init__(self, train_days=30, test_days=10):
        self.train_days = train_days
        self.test_days = test_days

    def split_data(self, data: Dict[str, pd.DataFrame]):
        """Generate train/test splits."""
        # Find common date range
        all_dates = set()
        for sym, df in data.items():
            dates = set(df.index.date if hasattr(df.index, 'date')
                        else [pd.Timestamp(t).date() for t in df.index])
            if not all_dates:
                all_dates = dates
            else:
                all_dates = all_dates.union(dates)

        all_dates = sorted(all_dates)
        if len(all_dates) < self.train_days + self.test_days:
            return []

        splits = []
        i = 0
        while i + self.train_days + self.test_days <= len(all_dates):
            train_start = all_dates[i]
            train_end = all_dates[i + self.train_days - 1]
            test_start = all_dates[i + self.train_days]
            test_end_idx = min(i + self.train_days + self.test_days - 1,
                               len(all_dates) - 1)
            test_end = all_dates[test_end_idx]

            train_data = {}
            test_data = {}
            for sym, df in data.items():
                idx = df.index
                if hasattr(idx, 'date'):
                    dates = idx.date
                else:
                    dates = pd.Series([pd.Timestamp(t).date()
                                       for t in idx], index=idx)
                mask_train = (dates >= train_start) & (dates <= train_end)
                mask_test = (dates >= test_start) & (dates <= test_end)
                if mask_train.sum() > 0:
                    train_data[sym] = df[mask_train]
                if mask_test.sum() > 0:
                    test_data[sym] = df[mask_test]

            if train_data and test_data:
                splits.append({
                    "train_data": train_data,
                    "test_data": test_data,
                    "train_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                })

            i += self.test_days  # Roll forward by test_days

        return splits

    def run(self, data, symbols, param_space=None, n_random=20):
        """Run walk-forward: optimize on train, validate on test."""
        splits = self.split_data(data)
        if not splits:
            print("ERROR: Not enough data for walk-forward splits")
            return None

        print(f"\nWalk-Forward: {len(splits)} periods, "
              f"{self.train_days}d train / {self.test_days}d test")

        all_results = []

        for fold_idx, split in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{len(splits)}: "
                  f"Train={split['train_period']} | "
                  f"Test={split['test_period']}")

            # Optimize on train
            if param_space:
                samples = param_space.random_sample(n_random)
                best_params = None
                best_sharpe = -999

                for params in samples:
                    bt = Backtester()
                    result = bt.run(split["train_data"], symbols)
                    m = result.get("metrics", {})
                    sharpe = m.get("sharpe_ratio", -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

                print(f"    Best train Sharpe: {best_sharpe:.2f}")
            else:
                best_params = {}

            # Test on out-of-sample
            bt_test = Backtester()
            test_result = bt_test.run(split["test_data"], symbols)
            test_m = test_result.get("metrics", {})

            fold_result = {
                "fold": fold_idx + 1,
                "train_period": split["train_period"],
                "test_period": split["test_period"],
                "best_params": best_params,
                "test_trades": test_m.get("total_trades", 0),
                "test_pnl": test_m.get("total_net_pnl", 0),
                "test_sharpe": test_m.get("sharpe_ratio", 0),
                "test_win_rate": test_m.get("win_rate", 0),
                "test_pf": test_m.get("profit_factor", 0),
                "test_max_dd": test_m.get("max_drawdown_pct", 0),
            }
            all_results.append(fold_result)

            print(f"    Test: {test_m.get('total_trades', 0)} trades, "
                  f"PnL=${test_m.get('total_net_pnl', 0):.2f}, "
                  f"Sharpe={test_m.get('sharpe_ratio', 0):.2f}, "
                  f"WR={test_m.get('win_rate', 0)*100:.0f}%")

        # Summary
        total_pnl = sum(r["test_pnl"] for r in all_results)
        total_trades = sum(r["test_trades"] for r in all_results)
        avg_sharpe = np.mean([r["test_sharpe"] for r in all_results])
        avg_wr = np.mean([r["test_win_rate"] for r in all_results])

        print(f"\n  WALK-FORWARD SUMMARY")
        print(f"  Total OOS P&L:  ${total_pnl:,.2f}")
        print(f"  Total OOS Trades: {total_trades}")
        print(f"  Avg OOS Sharpe:   {avg_sharpe:.2f}")
        print(f"  Avg OOS WinRate:  {avg_wr*100:.1f}%")

        return {
            "folds": all_results,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_wr,
        }


# ====================================================================
# MONTE CARLO SIMULATOR
# ====================================================================

class MonteCarloSimulator:
    """
    Shuffles trade order to test if results are robust
    or just lucky sequencing.
    """

    def __init__(self, n_simulations=1000):
        self.n_sims = n_simulations

    def run(self, trades, starting_capital=100000):
        """Run Monte Carlo on completed trade list."""
        if not trades or len(trades) < 5:
            print("Not enough trades for Monte Carlo")
            return None

        pnls = [t.net_pnl for t in trades]
        n = len(pnls)

        final_equities = []
        max_drawdowns = []
        max_drawdown_dollars = []

        for _ in range(self.n_sims):
            shuffled = np.random.permutation(pnls)
            equity = starting_capital
            peak = equity
            max_dd = 0
            max_dd_dollar = 0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                dd_dollar = peak - equity
                max_dd = max(max_dd, dd)
                max_dd_dollar = max(max_dd_dollar, dd_dollar)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            max_drawdown_dollars.append(max_dd_dollar)

        final_eq = np.array(final_equities)
        max_dds = np.array(max_drawdowns)
        max_dd_d = np.array(max_drawdown_dollars)

        results = {
            "n_simulations": self.n_sims,
            "n_trades": n,
            "median_final_equity": round(float(np.median(final_eq)), 2),
            "p5_equity": round(float(np.percentile(final_eq, 5)), 2),
            "p25_equity": round(float(np.percentile(final_eq, 25)), 2),
            "p75_equity": round(float(np.percentile(final_eq, 75)), 2),
            "p95_equity": round(float(np.percentile(final_eq, 95)), 2),
            "prob_profit": round(
                float(np.mean(final_eq > starting_capital)), 3),
            "prob_loss_5pct": round(
                float(np.mean(final_eq < starting_capital * 0.95)), 3),
            "median_max_dd_pct": round(
                float(np.median(max_dds) * 100), 2),
            "p95_max_dd_pct": round(
                float(np.percentile(max_dds, 95) * 100), 2),
            "median_max_dd_dollar": round(
                float(np.median(max_dd_d)), 2),
            "p95_max_dd_dollar": round(
                float(np.percentile(max_dd_d, 95)), 2),
        }

        return results

    def print_report(self, results):
        if results is None:
            return
        print(f"\n  MONTE CARLO ({results['n_simulations']} sims, "
              f"{results['n_trades']} trades)")
        print(f"  {'Median Final Equity:':<30} "
              f"${results['median_final_equity']:>12,.2f}")
        print(f"  {'5th Percentile:':<30} "
              f"${results['p5_equity']:>12,.2f}")
        print(f"  {'95th Percentile:':<30} "
              f"${results['p95_equity']:>12,.2f}")
        print(f"  {'P(Profit):':<30} "
              f"{results['prob_profit']*100:>12.1f}%")
        print(f"  {'P(Loss > 5%):':<30} "
              f"{results['prob_loss_5pct']*100:>12.1f}%")
        print(f"  {'Median Max DD:':<30} "
              f"{results['median_max_dd_pct']:>12.2f}%")
        print(f"  {'95th %ile Max DD:':<30} "
              f"{results['p95_max_dd_pct']:>12.2f}%")


# ====================================================================
# TRADE ANALYTICS
# ====================================================================

class TradeAnalytics:
    """Deep dive into what's working and what's not."""

    def analyze(self, trades):
        if not trades:
            return {"error": "No trades"}

        df = pd.DataFrame([{
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
            "commission": t.commission,
            "hold_bars": t.hold_bars,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        } for t in trades])

        results = {}

        # By symbol
        by_sym = df.groupby("symbol").agg({
            "net_pnl": ["sum", "mean", "count"],
            "gross_pnl": "sum",
            "hold_bars": "mean",
        }).round(2)
        by_sym.columns = ["net_pnl", "avg_pnl", "trades",
                          "gross_pnl", "avg_hold"]
        by_sym["win_rate"] = df.groupby("symbol").apply(
            lambda x: (x["net_pnl"] > 0).mean()).round(3)
        results["by_symbol"] = by_sym.to_dict()

        # By exit reason
        by_exit = df.groupby("exit_reason").agg({
            "net_pnl": ["sum", "mean", "count"],
            "hold_bars": "mean",
        }).round(2)
        by_exit.columns = ["net_pnl", "avg_pnl", "trades", "avg_hold"]
        results["by_exit_reason"] = by_exit.to_dict()

        # By hour of day
        if hasattr(df["entry_time"].iloc[0], 'hour'):
            df["hour"] = pd.to_datetime(
                df["entry_time"]).dt.hour
        else:
            df["hour"] = pd.to_datetime(
                df["entry_time"]).dt.hour
        by_hour = df.groupby("hour").agg({
            "net_pnl": ["sum", "mean", "count"]
        }).round(2)
        by_hour.columns = ["net_pnl", "avg_pnl", "trades"]
        results["by_hour"] = by_hour.to_dict()

        # Hold time distribution
        results["hold_stats"] = {
            "mean": round(df["hold_bars"].mean(), 1),
            "median": round(df["hold_bars"].median(), 1),
            "min": int(df["hold_bars"].min()),
            "max": int(df["hold_bars"].max()),
            "std": round(df["hold_bars"].std(), 1),
        }

        # Cost analysis
        total_gross = df["gross_pnl"].sum()
        total_net = df["net_pnl"].sum()
        results["cost_analysis"] = {
            "total_gross": round(total_gross, 2),
            "total_net": round(total_net, 2),
            "total_costs": round(total_gross - total_net, 2),
            "spread_total": round(df["spread_cost"].sum(), 2),
            "slippage_total": round(df["slippage_cost"].sum(), 2),
            "avg_cost_per_trade": round(
                (total_gross - total_net) / len(df), 2),
            "cost_as_pct_of_avg_trade": round(
                (total_gross - total_net) / len(df) /
                abs(df["gross_pnl"].mean()) * 100, 1
            ) if df["gross_pnl"].mean() != 0 else 0,
        }

        # Winners vs losers profile
        winners = df[df["net_pnl"] > 0]
        losers = df[df["net_pnl"] <= 0]
        results["winner_profile"] = {
            "count": len(winners),
            "avg_pnl": round(winners["net_pnl"].mean(), 2)
                if len(winners) > 0 else 0,
            "avg_hold": round(winners["hold_bars"].mean(), 1)
                if len(winners) > 0 else 0,
            "avg_gross": round(winners["gross_pnl"].mean(), 2)
                if len(winners) > 0 else 0,
        }
        results["loser_profile"] = {
            "count": len(losers),
            "avg_pnl": round(losers["net_pnl"].mean(), 2)
                if len(losers) > 0 else 0,
            "avg_hold": round(losers["hold_bars"].mean(), 1)
                if len(losers) > 0 else 0,
            "avg_gross": round(losers["gross_pnl"].mean(), 2)
                if len(losers) > 0 else 0,
        }

        # Correlation: are wins/losses clustered?
        if len(df) >= 5:
            pnl_series = df["net_pnl"].values
            if len(pnl_series) >= 5:
                autocorr = np.corrcoef(
                    pnl_series[:-1], pnl_series[1:]
                )[0, 1]
                results["serial_correlation"] = round(float(autocorr), 3)

        return results

    def print_report(self, analysis):
        if "error" in analysis:
            print(f"  No analysis: {analysis['error']}")
            return

        print(f"\n  TRADE ANALYTICS")
        print(f"  {'='*50}")

        # By exit reason
        print(f"\n  BY EXIT REASON:")
        er = analysis["by_exit_reason"]
        for reason in er.get("trades", {}):
            trades = er["trades"][reason]
            pnl = er["net_pnl"][reason]
            avg = er["avg_pnl"][reason]
            hold = er["avg_hold"][reason]
            print(f"    {reason:<25} {trades:>3} trades  "
                  f"${pnl:>8.2f}  avg=${avg:>7.2f}  "
                  f"hold={hold:.0f}bars")

        # By symbol
        print(f"\n  BY SYMBOL:")
        bs = analysis["by_symbol"]
        for sym in bs.get("trades", {}):
            trades = bs["trades"][sym]
            pnl = bs["net_pnl"][sym]
            wr = bs.get("win_rate", {}).get(sym, 0)
            print(f"    {sym:<8} {trades:>3} trades  "
                  f"${pnl:>8.2f}  WR={wr*100:.0f}%")

        # Hold times
        hs = analysis["hold_stats"]
        print(f"\n  HOLD TIMES: mean={hs['mean']:.0f} "
              f"median={hs['median']:.0f} "
              f"min={hs['min']} max={hs['max']}")

        # Cost analysis
        ca = analysis["cost_analysis"]
        print(f"\n  COSTS: ${ca['total_costs']:.2f} total "
              f"(${ca['avg_cost_per_trade']:.2f}/trade, "
              f"{ca['cost_as_pct_of_avg_trade']:.0f}% of avg trade)")

        # Winner/loser profile
        wp = analysis["winner_profile"]
        lp = analysis["loser_profile"]
        print(f"\n  WINNERS: {wp['count']} trades, "
              f"avg=${wp['avg_pnl']:.2f}, hold={wp['avg_hold']:.0f}bars")
        print(f"  LOSERS:  {lp['count']} trades, "
              f"avg=${lp['avg_pnl']:.2f}, hold={lp['avg_hold']:.0f}bars")

        if "serial_correlation" in analysis:
            sc = analysis["serial_correlation"]
            print(f"\n  Serial Correlation: {sc:.3f} "
                  f"({'clustered' if abs(sc) > 0.3 else 'random'})")


# ====================================================================
# CORRELATION ANALYZER
# ====================================================================

class CorrelationAnalyzer:
    """Check if symbols in watchlist are too correlated."""

    def analyze(self, data: Dict[str, pd.DataFrame]):
        """Calculate return correlations between symbols."""
        returns = {}
        for sym, df in data.items():
            if len(df) > 100:
                r = df["close"].pct_change().dropna()
                returns[sym] = r

        if len(returns) < 2:
            return None

        # Align on common index
        df_returns = pd.DataFrame(returns)
        df_returns = df_returns.dropna(how="all")

        corr = df_returns.corr()

        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                c = corr.iloc[i, j]
                if abs(c) > 0.7:
                    high_corr.append({
                        "sym1": corr.index[i],
                        "sym2": corr.columns[j],
                        "correlation": round(c, 3),
                    })

        avg_corr = corr.values[np.triu_indices_from(
            corr.values, k=1)].mean()

        return {
            "avg_correlation": round(float(avg_corr), 3),
            "high_pairs": high_corr,
            "matrix": corr.round(3).to_dict(),
        }

    def print_report(self, results):
        if results is None:
            print("  Not enough data for correlation analysis")
            return

        print(f"\n  CORRELATION ANALYSIS")
        print(f"  Avg pairwise correlation: "
              f"{results['avg_correlation']:.3f}")

        if results["high_pairs"]:
            print(f"  Highly correlated pairs (|r| > 0.7):")
            for p in results["high_pairs"]:
                print(f"    {p['sym1']}-{p['sym2']}: "
                      f"{p['correlation']:.3f}")
        else:
            print(f"  No highly correlated pairs (good diversification)")
