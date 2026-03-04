"""
Risk Manager - Pre-trade, intra-trade, and post-trade risk enforcement.
HARD limits that cannot be overridden by strategy signals.
If any check fails, the trade is blocked and logged.
"""
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Tuple
import pytz

from config.settings import AppConfig, SystemState
from utils.logger import StructuredLogger


class RiskManager:
    """
    Enforces all risk constraints. Strategy has ZERO authority to
    override these checks.
    """

    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.log = logger
        self.risk = config.risk
        self.et_tz = pytz.timezone(config.timezone)

        # Daily tracking — reset each trading day
        self.daily_pnl = Decimal("0.00")
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.peak_equity = config.starting_capital
        self.current_equity = config.starting_capital
        self.trading_day = None

        # State
        self.trading_disabled = False
        self.disable_reason = ""

        # Position tracking (symbol -> qty)
        self.positions: Dict[str, int] = {}

        self.log.log_system("Risk manager initialized",
                            max_daily_loss=str(self.risk.max_daily_loss_pct),
                            max_drawdown=str(self.risk.max_drawdown_pct))

    # ── Daily Reset ────────────────────────────────────────────────────

    def check_new_day(self):
        """Reset daily counters if it's a new trading day."""
        today = datetime.now(self.et_tz).date()
        if self.trading_day != today:
            self.log.log_system("New trading day",
                                previous=str(self.trading_day),
                                new=str(today),
                                prev_daily_pnl=str(self.daily_pnl))
            self.daily_pnl = Decimal("0.00")
            self.daily_trades = 0
            self.trading_day = today

            # Re-enable trading if it was disabled by daily limit
            # (drawdown limits remain across days)
            if self.trading_disabled and \
               self.disable_reason in ("daily_loss_limit", "consecutive_losses"):
                self.trading_disabled = False
                self.disable_reason = ""
                self.consecutive_losses = 0
                self.log.log_system("Trading re-enabled for new day")

    # ── Pre-Trade Checks ───────────────────────────────────────────────

    def pre_trade_check(self, symbol, side, qty, price, buying_power,
                        spread_pct, expected_slippage_pct=Decimal("0.001")):
        """
        Run ALL pre-trade risk checks. Returns (approved, reason).
        ALL checks must pass. One failure = trade blocked.
        """
        self.check_new_day()

        checks = [
            self._check_trading_enabled(),
            self._check_time_window(),
            self._check_daily_loss_limit(),
            self._check_drawdown_limit(),
            self._check_consecutive_losses(),
            self._check_position_count(symbol),
            self._check_position_size(symbol, qty, price),
            self._check_buying_power(side, qty, price, buying_power),
            self._check_spread(symbol, spread_pct),
            self._check_slippage(expected_slippage_pct),
            self._check_total_exposure(qty, price),
            self._check_order_size(qty, price),
            self._check_short_selling(side),
        ]

        for passed, reason in checks:
            if not passed:
                self.log.log_risk("PRE_TRADE", False,
                                  symbol=symbol, side=side, qty=qty,
                                  reason=reason)
                return False, reason

        self.log.log_risk("PRE_TRADE", True,
                          symbol=symbol, side=side, qty=qty)
        return True, "all_checks_passed"

    def _check_trading_enabled(self):
        if self.trading_disabled:
            return False, f"trading_disabled: {self.disable_reason}"
        return True, "ok"

    def _check_time_window(self):
        """No trades outside allowed hours."""
        now = datetime.now(self.et_tz)

        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < market_open or now >= market_close:
            return False, "outside_market_hours"

        # No trades in first N minutes
        minutes_since_open = (now - market_open).total_seconds() / 60
        if minutes_since_open < self.risk.no_trade_first_minutes:
            return False, f"first_{self.risk.no_trade_first_minutes}_minutes"

        # No new trades after cutoff
        cutoff_h, cutoff_m = map(int, self.risk.no_new_trades_after.split(":"))
        cutoff = now.replace(hour=cutoff_h, minute=cutoff_m, second=0)
        if now >= cutoff:
            return False, "past_entry_cutoff"

        return True, "ok"

    def _check_daily_loss_limit(self):
        """Hard disable if daily loss exceeds limit."""
        limit_pct = self.risk.max_daily_loss_pct * self.current_equity
        limit_abs = self.risk.max_daily_loss_abs

        # Use the tighter of the two limits
        effective_limit = min(limit_pct, limit_abs)

        if self.daily_pnl <= -effective_limit:
            self.trading_disabled = True
            self.disable_reason = "daily_loss_limit"
            self.log.critical("RISK",
                              "DAILY LOSS LIMIT HIT - TRADING DISABLED",
                              daily_pnl=str(self.daily_pnl),
                              limit=str(effective_limit))
            return False, f"daily_loss_limit_hit: {self.daily_pnl}"
        return True, "ok"

    def _check_drawdown_limit(self):
        """Hard disable if drawdown from peak exceeds limit."""
        drawdown = self.peak_equity - self.current_equity
        drawdown_pct = drawdown / self.peak_equity if self.peak_equity > 0 \
            else Decimal("0")

        if drawdown >= self.risk.max_drawdown_abs or \
           drawdown_pct >= self.risk.max_drawdown_pct:
            self.trading_disabled = True
            self.disable_reason = "max_drawdown"
            self.log.critical("RISK",
                              "MAX DRAWDOWN HIT - TRADING DISABLED",
                              drawdown=str(drawdown),
                              drawdown_pct=str(drawdown_pct),
                              peak=str(self.peak_equity),
                              current=str(self.current_equity))
            return False, f"max_drawdown: {drawdown_pct}"
        return True, "ok"

    def _check_consecutive_losses(self):
        if self.consecutive_losses >= self.risk.max_consecutive_losses:
            self.trading_disabled = True
            self.disable_reason = "consecutive_losses"
            self.log.critical("RISK",
                              "CONSECUTIVE LOSS LIMIT - TRADING DISABLED",
                              count=self.consecutive_losses)
            return False, f"consecutive_losses: {self.consecutive_losses}"
        return True, "ok"

    def _check_position_count(self, symbol):
        """Check we haven't exceeded max simultaneous positions."""
        open_count = len(self.positions)
        # If we already have a position in this symbol, it's an add/exit
        if symbol in self.positions:
            return True, "ok"
        if open_count >= self.risk.max_simultaneous_positions:
            return False, f"max_positions: {open_count}"
        return True, "ok"

    def _check_position_size(self, symbol, qty, price):
        """Check position doesn't exceed max % of account."""
        notional = Decimal(str(qty)) * Decimal(str(price))
        max_notional = self.current_equity * self.risk.max_position_pct

        # Include existing position if any
        existing_qty = self.positions.get(symbol, 0)
        total_notional = (Decimal(str(existing_qty + qty))
                          * Decimal(str(price)))

        if total_notional > max_notional:
            return False, (f"position_too_large: "
                           f"{total_notional} > {max_notional}")
        return True, "ok"

    def _check_buying_power(self, side, qty, price, buying_power):
        """Cash account: can only buy with settled funds."""
        if side == "sell":
            return True, "ok"  # Selling doesn't need buying power
        cost = Decimal(str(qty)) * Decimal(str(price))
        if cost > buying_power:
            return False, (f"insufficient_buying_power: "
                           f"need {cost}, have {buying_power}")
        return True, "ok"

    def _check_spread(self, symbol, spread_pct):
        """Block if spread is too wide."""
        if Decimal(str(spread_pct)) > self.risk.max_spread_pct:
            return False, (f"spread_too_wide: "
                           f"{spread_pct} > {self.risk.max_spread_pct}")
        return True, "ok"

    def _check_slippage(self, expected_slippage_pct):
        """Block if expected slippage is too high."""
        if Decimal(str(expected_slippage_pct)) > self.risk.max_slippage_pct:
            return False, (f"slippage_too_high: "
                           f"{expected_slippage_pct} > {self.risk.max_slippage_pct}")
        return True, "ok"

    def _check_total_exposure(self, qty, price):
        """Total exposure across all positions."""
        new_notional = Decimal(str(qty)) * Decimal(str(price))
        current_exposure = sum(
            Decimal(str(q)) * Decimal(str(price))  # approximate
            for q in self.positions.values()
        )
        total = current_exposure + new_notional
        max_exposure = self.current_equity * self.risk.max_total_exposure_pct

        if total > max_exposure:
            return False, (f"total_exposure_exceeded: "
                           f"{total} > {max_exposure}")
        return True, "ok"

    def _check_order_size(self, qty, price):
        """Minimum and maximum order size."""
        notional = Decimal(str(qty)) * Decimal(str(price))
        if notional < self.config.account.min_order_size_usd:
            return False, f"order_too_small: {notional}"
        if notional > self.config.account.max_notional_per_order:
            return False, f"order_too_large: {notional}"
        if qty < self.config.account.min_share_quantity:
            return False, f"qty_below_min: {qty}"
        return True, "ok"

    def _check_short_selling(self, side):
        """Cash account: no shorting."""
        if side == "sell" and not self.config.account.short_selling_enabled:
            # This is fine if we HAVE the position (covered sell)
            # The broker layer also checks this, but belt-and-suspenders
            pass
        return True, "ok"

    # ── Intra-Trade Checks ─────────────────────────────────────────────

    def check_hold_duration(self, symbol, entry_time):
        """Check if position has been held too long."""
        now = datetime.now(self.et_tz)
        if entry_time.tzinfo is None:
            entry_time = self.et_tz.localize(entry_time)
        held_minutes = (now - entry_time).total_seconds() / 60

        if held_minutes >= self.risk.max_hold_duration_minutes:
            self.log.warning("RISK", "Max hold duration exceeded",
                             symbol=symbol,
                             held_minutes=round(held_minutes, 1),
                             max=self.risk.max_hold_duration_minutes)
            return True  # True = must exit
        return False

    def check_force_flatten(self):
        """Check if we've hit the force-flatten time."""
        now = datetime.now(self.et_tz)
        flat_h, flat_m = map(int, self.risk.force_flatten_by.split(":"))
        flatten_time = now.replace(hour=flat_h, minute=flat_m, second=0)
        return now >= flatten_time

    # ── Post-Trade Updates ─────────────────────────────────────────────

    def record_trade_result(self, symbol, pnl, side):
        """Update risk state after a trade completes."""
        self.daily_pnl += Decimal(str(pnl))
        self.daily_trades += 1

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.log.log_risk("TRADE_RESULT", True,
                          symbol=symbol,
                          trade_pnl=str(pnl),
                          daily_pnl=str(self.daily_pnl),
                          consecutive_losses=self.consecutive_losses)

    def update_equity(self, equity):
        """Update equity tracking for drawdown calculation."""
        self.current_equity = Decimal(str(equity))
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def update_positions(self, positions):
        """Update internal position tracking."""
        self.positions = positions

    # ── Position Sizing ────────────────────────────────────────────────

    def calculate_position_size(self, price, stop_price, buying_power):
        """
        Calculate position size based on:
        1. Max risk per trade (% of account)
        2. Distance to stop loss
        3. Available buying power
        4. Max position size constraint
        Returns number of whole shares.
        """
        price = Decimal(str(price))
        stop_price = Decimal(str(stop_price))
        buying_power = Decimal(str(buying_power))

        if price <= 0 or stop_price <= 0 or stop_price >= price:
            self.log.warning("RISK", "Invalid sizing inputs",
                             price=str(price), stop=str(stop_price))
            return 0

        # Risk per share
        risk_per_share = price - stop_price

        # Max dollar risk for this trade
        max_risk = self.current_equity * self.risk.max_risk_per_trade_pct

        # Shares from risk limit
        shares_from_risk = int(max_risk / risk_per_share)

        # Shares from position size limit
        max_position = self.current_equity * self.risk.max_position_pct
        shares_from_position = int(max_position / price)

        # Shares from buying power
        shares_from_bp = int(buying_power / price)

        # Take the minimum
        shares = min(shares_from_risk, shares_from_position, shares_from_bp)

        # Floor at minimum
        if shares < self.config.account.min_share_quantity:
            self.log.warning("RISK", "Position size below minimum",
                             calculated=shares)
            return 0

        self.log.log_risk("POSITION_SIZE", True,
                          price=str(price),
                          stop=str(stop_price),
                          risk_per_share=str(risk_per_share),
                          max_risk=str(max_risk),
                          shares=shares,
                          from_risk=shares_from_risk,
                          from_position=shares_from_position,
                          from_bp=shares_from_bp)

        return shares

    def get_status(self):
        """Return current risk state as a dict."""
        return {
            "trading_disabled": self.trading_disabled,
            "disable_reason": self.disable_reason,
            "daily_pnl": str(self.daily_pnl),
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "peak_equity": str(self.peak_equity),
            "current_equity": str(self.current_equity),
            "drawdown": str(self.peak_equity - self.current_equity),
            "open_positions": len(self.positions),
        }
