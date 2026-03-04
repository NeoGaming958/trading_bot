"""
ORB Strategy V4.1 — Paper-Matched Live Trading
================================================
Zarattini, Barbon & Aziz (2024) "A Profitable Day Trading Strategy
For The U.S. Equity Market" - Swiss Finance Institute Paper No. 24-98

This is the LIVE TRADING strategy class. It generates signals and
order parameters that the bot runner submits to Alpaca.

PAPER RULES (exact):
  1. 5-minute ORB (first 5 one-minute bars)
  2. Bullish ORB (close > open) → stop buy order at ORB high
     Bearish ORB (close < open) → stop sell order at ORB low
     Doji (close = open) → no trade
  3. Stop loss = 0.10 × 14-day ATR from entry
  4. Exit: hold to 4:00 PM ET close (no profit target, no trailing)
  5. Position size: risk 1% of capital per trade
  6. Max leverage: 4×
  7. Equal-weight cap: max position ≤ capital / max_positions
  8. One trade per stock per day
  9. Filters: price > $5, ATR > $0.50, avg daily vol ≥ 1M, RVOL ≥ 2×

FIXES FROM V3.x (9 errors corrected):
  1. Stop: 0.10×ATR from entry (was 0.15×ATR in V3.1, ORB low in V4.0)
  2. Entry: stop order at ORB high/low (was market order on bar close)
  3. One trade/stock/day (was unlimited re-entry)
  4. No VWAP filter (was hard requirement — not in paper)
  5. Avg daily vol ≥ 1M (was 500K or missing)
  6. Long + Short (was long-only)
  7. Hold to EOD only (was trailing stop + profit target + time stop)
  8. 4× leverage (was 1× cap)
  9. Up to 20 positions (was 1-3)

ARCHITECTURE:
  This class is a pure strategy engine. It does NOT submit orders.
  The bot runner calls:
    1. strategy.new_day()           — at market open
    2. strategy.feed_bar(...)       — each 1-min bar
    3. strategy.get_pending_orders() — after ORB complete, returns stop orders
    4. strategy.on_fill(...)        — when Alpaca confirms fill
    5. strategy.check_stops(...)    — each bar, returns stop-out exits
    6. strategy.get_eod_exits()     — at 3:55 PM, returns all positions to close
"""
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class TradeSignal(Enum):
    NO_TRADE = auto()
    LONG_ENTRY = auto()
    LONG_EXIT = auto()
    SHORT_ENTRY = auto()
    SHORT_EXIT = auto()


@dataclass
class OpeningRange:
    """Tracks the 5-minute opening range for one symbol."""
    symbol: str
    high: float = 0.0
    low: float = float('inf')
    open_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0
    bars: int = 0
    is_complete: bool = False
    is_bullish: bool = False    # close > open → long setup
    is_bearish: bool = False    # close < open → short setup
    is_doji: bool = False       # close == open → no trade
    size: float = 0.0


@dataclass
class PendingStopOrder:
    """
    Represents a stop order to be placed at Alpaca AFTER the ORB completes.
    
    Paper: "we placed a stop order at a level equal to the high/low
    of the 5-minute range"
    
    For Alpaca: submit as stop order, type='stop', stop_price=trigger_price.
    This fills at the trigger price (or next available), matching the paper's
    methodology better than any simulation we can do in code.
    """
    symbol: str
    direction: str              # "long" or "short"
    trigger_price: float        # ORB high (long) or ORB low (short)
    stop_loss_price: float      # 0.10 × ATR from trigger
    shares: int                 # Position size
    atr: float                  # For reference
    rvol: float                 # For logging
    risk_per_share: float       # stop distance in $


@dataclass
class LivePosition:
    """Tracks an open position for exit management."""
    symbol: str
    direction: str              # "long" or "short"
    entry_price: float
    shares: int
    stop_price: float           # 0.10 × ATR from entry (ONLY stop)
    atr: float
    rvol: float
    entry_bar: int = 0
    bars_held: int = 0


class ORBStrategyV41:
    """
    Opening Range Breakout V4.1 — Paper-Matched

    ENTRY: Stop order at ORB high (long) or ORB low (short).
    STOP:  0.10 × ATR from entry. Nothing else.
    EXIT:  Hold to 4:00 PM close. No profit target. No trailing.
    SIZE:  1% risk, 4× max leverage, equal-weight cap.
    FILTER: Price>$5, ATR>$0.50, AvgVol>1M, RVOL>=2×.
    ONCE:  One trade per stock per day.
    """

    # ── Paper Constants ──────────────────────────────────────────────
    ORB_BARS = 5                    # 5-minute opening range
    MIN_PRICE = 5.0                 # Paper: "opening price > $5"
    MIN_ATR = 0.50                  # Paper: "ATR > $0.50"
    MIN_AVG_DAILY_VOL = 1_000_000   # Paper: "avg volume ≥ 1,000,000"
    MIN_RVOL = 2.0                  # Paper: "Relative Volume ≥ 100%" (2×)
    STOP_ATR_PCT = 0.10             # Paper: "stop loss at 10% of the ATR"
    RISK_PCT = 0.01                 # Paper: "1% of capital"
    MAX_LEVERAGE = 4.0              # Paper: "maximum leverage constraint at 4x"
    MAX_POSITIONS = 20              # Paper: "top 20 stocks"
    EOD_BAR = 385                   # ~3:55 PM ET (leave 5 min for clean exit)

    def __init__(self, max_positions=20, max_leverage=4.0,
                 risk_pct=0.01, stop_atr_pct=0.10, min_rvol=2.0,
                 min_avg_daily_vol=1_000_000):
        """
        All parameters default to paper spec. Override for experimentation.
        """
        self.MAX_POSITIONS = max_positions
        self.MAX_LEVERAGE = max_leverage
        self.RISK_PCT = risk_pct
        self.STOP_ATR_PCT = stop_atr_pct
        self.MIN_RVOL = min_rvol
        self.MIN_AVG_DAILY_VOL = min_avg_daily_vol

        # Per-day state (reset in new_day)
        self.opening_ranges: Dict[str, OpeningRange] = {}
        self.positions: Dict[str, LivePosition] = {}
        self.pending_orders: Dict[str, PendingStopOrder] = {}
        self.traded_today: set = set()      # FIX #3: one trade per stock/day
        self.day_bar: int = 0
        self.orb_phase_complete: bool = False

    # ── Day Lifecycle ────────────────────────────────────────────────

    def new_day(self):
        """Call at market open (9:30 AM ET). Resets all daily state."""
        self.opening_ranges.clear()
        self.positions.clear()
        self.pending_orders.clear()
        self.traded_today.clear()
        self.day_bar = 0
        self.orb_phase_complete = False

    # ── Bar Processing ───────────────────────────────────────────────

    def increment_bar(self):
        """Call once per minute bar (not per symbol). Advances the day clock."""
        self.day_bar += 1

    def feed_bar(self, symbol: str, o: float, h: float, l: float,
                 c: float, v: float, daily_atr: float,
                 avg_daily_vol: float, rvol: float):
        """
        Feed one 1-minute bar for one symbol. Call for each symbol each minute.
        Call increment_bar() ONCE before feeding all symbols for that bar.

        During bars 1-5: builds the opening range.
        At bar 5: evaluates filters and creates pending stop orders.
        After bar 5: the bot runner checks get_pending_orders() and
                     submits them to Alpaca.

        Args:
            symbol: Ticker
            o, h, l, c, v: OHLCV for this 1-min bar
            daily_atr: 14-day ATR from daily bars (pre-computed)
            avg_daily_vol: 14-day average daily volume
            rvol: Relative volume (today's 5min vol / avg 5min vol)
        """

        # ── Build ORB (bars 1-5) ────────────────────────────────────
        if self.day_bar <= self.ORB_BARS:
            if symbol not in self.opening_ranges:
                self.opening_ranges[symbol] = OpeningRange(
                    symbol=symbol, high=h, low=l,
                    open_price=o, close_price=c, volume=v, bars=1
                )
            else:
                orb = self.opening_ranges[symbol]
                if not orb.is_complete:
                    orb.high = max(orb.high, h)
                    orb.low = min(orb.low, l)
                    orb.close_price = c
                    orb.volume += v
                    orb.bars += 1

            # Finalize ORB at bar 5
            if self.day_bar == self.ORB_BARS and symbol in self.opening_ranges:
                orb = self.opening_ranges[symbol]
                orb.is_complete = True
                orb.is_bullish = (orb.close_price > orb.open_price)
                orb.is_bearish = (orb.close_price < orb.open_price)
                orb.is_doji = (orb.close_price == orb.open_price)
                orb.size = orb.high - orb.low

                # Evaluate filters and create pending order
                self._evaluate_setup(symbol, orb, daily_atr,
                                     avg_daily_vol, rvol)

    def _evaluate_setup(self, symbol: str, orb: OpeningRange,
                        daily_atr: float, avg_daily_vol: float,
                        rvol: float) -> Optional[str]:
        """
        After ORB completes, check all paper filters.
        If passed, create a PendingStopOrder.
        Returns rejection reason or None if order created.
        """
        # Doji → no trade
        if orb.is_doji:
            return "doji"

        # Price filter
        mid = (orb.high + orb.low) / 2
        if mid < self.MIN_PRICE:
            return "price_low"

        # ATR filter
        if daily_atr is None or daily_atr < self.MIN_ATR:
            return "atr_low"

        # Avg daily volume filter (FIX #5)
        if avg_daily_vol < self.MIN_AVG_DAILY_VOL:
            return "avg_vol_low"

        # RVOL filter
        if rvol is None or rvol < self.MIN_RVOL:
            return "rvol_low"

        # Direction and trigger price
        if orb.is_bullish:
            direction = "long"
            trigger = orb.high
        elif orb.is_bearish:
            direction = "short"
            trigger = orb.low
        else:
            return "no_direction"

        # Stop distance: 0.10 × ATR (FIX #1 — paper spec)
        stop_dist = daily_atr * self.STOP_ATR_PCT
        if stop_dist <= 0.001:
            return "zero_stop_dist"

        if direction == "long":
            stop_price = trigger - stop_dist
        else:
            stop_price = trigger + stop_dist

        # Position sizing will be calculated when equity is known
        # Store the order template — shares computed at submission time
        self.pending_orders[symbol] = PendingStopOrder(
            symbol=symbol,
            direction=direction,
            trigger_price=round(trigger, 2),
            stop_loss_price=round(stop_price, 2),
            shares=0,  # Computed by calc_shares() at submission
            atr=daily_atr,
            rvol=rvol if rvol else 0,
            risk_per_share=round(stop_dist, 4),
        )
        return None

    # ── Order Management ─────────────────────────────────────────────

    def get_pending_orders(self, equity: float) -> List[PendingStopOrder]:
        """
        Returns list of stop orders to submit to Alpaca.

        Call after bar 5 (ORB complete). The bot runner should:
        1. Get this list
        2. For each order, submit to Alpaca as:
           - Long: side='buy', type='stop', stop_price=order.trigger_price
           - Short: side='sell', type='stop', stop_price=order.trigger_price
        3. Also submit a separate stop-loss order (OCA/bracket if supported)

        The bot runner should NOT submit orders for symbols in traded_today.
        """
        orders = []
        for sym, order in list(self.pending_orders.items()):
            # Skip if already traded today (FIX #3)
            if sym in self.traded_today:
                continue
            # Skip if already in position
            if sym in self.positions:
                continue
            # Skip if at max positions
            if len(self.positions) >= self.MAX_POSITIONS:
                continue

            # Calculate shares with current equity
            shares = self.calc_shares(
                equity=equity,
                entry_price=order.trigger_price,
                risk_per_share=order.risk_per_share,
            )
            if shares <= 0:
                continue

            order.shares = shares
            orders.append(order)

        return orders

    def calc_shares(self, equity: float, entry_price: float,
                    risk_per_share: float) -> int:
        """
        Paper position sizing:
        1. shares = (equity × 1%) / risk_per_share
        2. Cap at 4× leverage: shares ≤ (equity × 4) / price
        3. Cap at equal-weight: shares ≤ (equity / max_positions) / price
        """
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        # 1% risk
        risk_dollars = equity * self.RISK_PCT
        shares = int(risk_dollars / risk_per_share)

        # 4× leverage cap (FIX #8)
        max_leverage_shares = int(equity * self.MAX_LEVERAGE / entry_price)
        shares = min(shares, max_leverage_shares)

        # Equal-weight cap
        max_weight_shares = int(
            equity / self.MAX_POSITIONS / entry_price
        ) if entry_price > 0 else 0
        shares = min(shares, max_weight_shares)

        return max(0, shares)

    # ── Fill Handling ────────────────────────────────────────────────

    def on_fill(self, symbol: str, fill_price: float, shares: int,
                direction: str):
        """
        Called when Alpaca confirms a stop order fill.

        Creates the position and marks symbol as traded today.
        The bot runner should also submit the stop-loss order at this point.
        """
        order = self.pending_orders.pop(symbol, None)
        if order is None:
            # Fill without pending order — use provided params
            atr = 0
            rvol = 0
            stop_dist = 0
        else:
            atr = order.atr
            rvol = order.rvol
            stop_dist = order.risk_per_share

        # Calculate stop from actual fill price (may differ from trigger)
        if stop_dist > 0:
            if direction == "long":
                stop_price = fill_price - stop_dist
            else:
                stop_price = fill_price + stop_dist
        elif order:
            stop_price = order.stop_loss_price
        else:
            stop_price = fill_price  # Fallback, should not happen

        self.positions[symbol] = LivePosition(
            symbol=symbol,
            direction=direction,
            entry_price=fill_price,
            shares=shares,
            stop_price=round(stop_price, 2),
            atr=atr,
            rvol=rvol,
            entry_bar=self.day_bar,
        )
        self.traded_today.add(symbol)  # FIX #3: no re-entry

    # ── Stop Loss Checking ───────────────────────────────────────────

    def check_stops(self, symbol: str, bar_high: float,
                    bar_low: float, bar_close: float
                    ) -> Optional[Tuple[TradeSignal, dict]]:
        """
        Check if stop loss was hit this bar.

        Paper exit rules (FIX #7 — ONLY these two):
        1. Stop loss hit → exit at stop price
        2. EOD → exit at close

        NO trailing stop. NO profit target. NO time stop. NO breakeven.
        Winners run all day. That's the entire edge.

        Returns (signal, meta) if exit triggered, None if holding.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.bars_held += 1

        # Stop loss check
        if pos.direction == "long" and bar_low <= pos.stop_price:
            # Stopped out — exit at stop price (or open if gapped through)
            exit_price = pos.stop_price
            signal = TradeSignal.LONG_EXIT
            meta = self._build_exit_meta(pos, exit_price, "stop_loss")
            del self.positions[symbol]
            return signal, meta

        elif pos.direction == "short" and bar_high >= pos.stop_price:
            exit_price = pos.stop_price
            signal = TradeSignal.SHORT_EXIT
            meta = self._build_exit_meta(pos, exit_price, "stop_loss")
            del self.positions[symbol]
            return signal, meta

        # EOD check
        if self.day_bar >= self.EOD_BAR:
            exit_price = bar_close
            signal = (TradeSignal.LONG_EXIT if pos.direction == "long"
                      else TradeSignal.SHORT_EXIT)
            won = ((pos.direction == "long" and bar_close > pos.entry_price) or
                   (pos.direction == "short" and bar_close < pos.entry_price))
            reason = "eod_win" if won else "eod_loss"
            meta = self._build_exit_meta(pos, exit_price, reason)
            del self.positions[symbol]
            return signal, meta

        return None  # Holding — no action

    def _build_exit_meta(self, pos: LivePosition, exit_price: float,
                         reason: str) -> dict:
        """Build metadata dict for exit signal."""
        if pos.direction == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.shares

        risk_per_share = pos.atr * self.STOP_ATR_PCT
        r_multiple = ((exit_price - pos.entry_price) / risk_per_share
                      if pos.direction == "long" and risk_per_share > 0
                      else (pos.entry_price - exit_price) / risk_per_share
                      if risk_per_share > 0 else 0)

        return {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": round(exit_price, 2),
            "shares": pos.shares,
            "gross_pnl": round(gross_pnl, 2),
            "r_multiple": round(r_multiple, 2),
            "bars_held": pos.bars_held,
            "rvol": pos.rvol,
            "reason": reason,
        }

    # ── EOD Flatten ──────────────────────────────────────────────────

    def get_eod_exits(self, current_prices: Dict[str, float]
                      ) -> List[Tuple[TradeSignal, dict]]:
        """
        At EOD, return exit signals for ALL open positions.
        Call at ~3:55 PM ET. Bot runner submits market orders to close.
        """
        exits = []
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            price = current_prices.get(sym, pos.entry_price)
            signal = (TradeSignal.LONG_EXIT if pos.direction == "long"
                      else TradeSignal.SHORT_EXIT)
            won = ((pos.direction == "long" and price > pos.entry_price) or
                   (pos.direction == "short" and price < pos.entry_price))
            reason = "eod_win" if won else "eod_loss"
            meta = self._build_exit_meta(pos, price, reason)
            exits.append((signal, meta))
            del self.positions[sym]
        return exits

    # ── Cancel Unfilled Orders ───────────────────────────────────────

    def get_symbols_to_cancel(self) -> List[str]:
        """
        At EOD, return symbols with unfilled pending orders to cancel.
        Bot runner cancels the stop orders at Alpaca.
        """
        syms = list(self.pending_orders.keys())
        self.pending_orders.clear()
        return syms

    # ── Query Methods ────────────────────────────────────────────────

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        return self.positions.get(symbol)

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Get current stop price for a position."""
        pos = self.positions.get(symbol)
        if pos:
            return pos.stop_price
        return None

    def position_count(self) -> int:
        return len(self.positions)

    def get_all_positions(self) -> Dict[str, LivePosition]:
        return dict(self.positions)

    def was_traded_today(self, symbol: str) -> bool:
        return symbol in self.traded_today

    # ── Alpaca Order Helpers ─────────────────────────────────────────

    @staticmethod
    def to_alpaca_entry_order(order: PendingStopOrder) -> dict:
        """
        Convert PendingStopOrder to Alpaca order params.

        Usage:
            params = ORBStrategyV41.to_alpaca_entry_order(order)
            client.submit_order(**params)
        """
        return {
            "symbol": order.symbol,
            "qty": order.shares,
            "side": "buy" if order.direction == "long" else "sell",
            "type": "stop",
            "time_in_force": "day",  # Auto-cancels at EOD
            "stop_price": str(order.trigger_price),
        }

    @staticmethod
    def to_alpaca_stop_loss(pos: LivePosition) -> dict:
        """
        Create stop-loss order for an open position.

        Usage (after fill):
            params = ORBStrategyV41.to_alpaca_stop_loss(position)
            client.submit_order(**params)
        """
        return {
            "symbol": pos.symbol,
            "qty": pos.shares,
            "side": "sell" if pos.direction == "long" else "buy",
            "type": "stop",
            "time_in_force": "day",
            "stop_price": str(pos.stop_price),
        }

    @staticmethod
    def to_alpaca_eod_exit(pos: LivePosition) -> dict:
        """
        Create market order to close position at EOD.
        """
        return {
            "symbol": pos.symbol,
            "qty": pos.shares,
            "side": "sell" if pos.direction == "long" else "buy",
            "type": "market",
            "time_in_force": "day",
        }
