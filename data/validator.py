"""
Data Validator - Ensures data integrity before any trading decision.
Blocks trading when inputs fall outside validated bounds.
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Tuple

from config.settings import AppConfig, DataConfig
from utils.logger import StructuredLogger


class QuoteData:
    """Validated quote container. Immutable once created."""

    def __init__(self, symbol, bid, ask, bid_size, ask_size, timestamp):
        self.symbol = symbol
        self.bid = Decimal(str(bid))
        self.ask = Decimal(str(ask))
        self.bid_size = int(bid_size)
        self.ask_size = int(ask_size)
        self.timestamp = timestamp

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def spread(self):
        return self.ask - self.bid

    @property
    def spread_pct(self):
        if self.mid > 0:
            return self.spread / self.mid
        return Decimal("999")  # Invalid

    def age_ms(self):
        now = datetime.now(timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (now - ts).total_seconds() * 1000
        return delta

    def __repr__(self):
        return (f"Quote({self.symbol} {self.bid}x{self.ask} "
                f"spread={self.spread_pct:.4f})")


class DataValidator:
    """
    Validates all market data before it reaches strategy or execution.
    Any validation failure = block trading for that symbol.
    """

    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.data_cfg = config.data
        self.log = logger

        # Track last known good quotes for deviation checks
        self.last_quotes: dict = {}

        self.log.log_system("Data validator initialized")

    def validate_quote(self, symbol, bid, ask, bid_size, ask_size,
                       timestamp) -> Tuple[bool, Optional[QuoteData], str]:
        """
        Validate a raw quote. Returns (valid, QuoteData|None, reason).
        """
        # ── Basic sanity ───────────────────────────────────────────
        if bid is None or ask is None:
            return False, None, "null_bid_or_ask"

        try:
            bid = Decimal(str(bid))
            ask = Decimal(str(ask))
        except Exception:
            return False, None, "invalid_price_format"

        if bid <= 0 or ask <= 0:
            return False, None, "non_positive_price"

        if bid >= ask:
            return False, None, f"crossed_market: bid={bid} >= ask={ask}"

        # ── Bid/Ask sanity ratio ───────────────────────────────────
        ratio = bid / ask if ask > 0 else Decimal("0")
        if ratio < self.data_cfg.min_bid_ask_sanity_ratio:
            return False, None, (f"insane_spread: ratio={ratio} "
                                 f"< {self.data_cfg.min_bid_ask_sanity_ratio}")

        # ── Size validation ────────────────────────────────────────
        if bid_size <= 0 or ask_size <= 0:
            return False, None, "zero_quote_size"

        # ── Timestamp validation ───────────────────────────────────
        if timestamp is None:
            return False, None, "null_timestamp"

        quote = QuoteData(symbol, bid, ask, bid_size, ask_size, timestamp)

        # ── Staleness check ────────────────────────────────────────
        age = quote.age_ms()
        if age > self.data_cfg.max_quote_age_ms:
            return False, None, f"stale_quote: {age:.0f}ms old"

        # ── Price deviation check ──────────────────────────────────
        if symbol in self.last_quotes:
            last_mid = self.last_quotes[symbol].mid
            if last_mid > 0:
                deviation = abs(quote.mid - last_mid) / last_mid
                if deviation > self.data_cfg.max_price_deviation_pct:
                    self.log.warning("DATA",
                                     "Large price deviation",
                                     symbol=symbol,
                                     deviation=str(deviation),
                                     new_mid=str(quote.mid),
                                     last_mid=str(last_mid))
                    return False, None, (f"price_deviation: "
                                         f"{deviation} > "
                                         f"{self.data_cfg.max_price_deviation_pct}")

        # ── All checks passed ──────────────────────────────────────
        self.last_quotes[symbol] = quote
        return True, quote, "ok"

    def validate_bar(self, symbol, open_p, high, low, close, volume,
                     timestamp) -> Tuple[bool, str]:
        """Validate a price bar for indicator calculation."""
        try:
            o = Decimal(str(open_p))
            h = Decimal(str(high))
            l = Decimal(str(low))
            c = Decimal(str(close))
            v = int(volume)
        except (ValueError, InvalidOperation):
            return False, "invalid_bar_format"

        if any(x <= 0 for x in [o, h, l, c]):
            return False, "non_positive_bar_price"

        if v < 0:
            return False, "negative_volume"

        # High must be highest, low must be lowest
        if h < max(o, c) or l > min(o, c):
            return False, f"inconsistent_ohlc: O={o} H={h} L={l} C={c}"

        if h < l:
            return False, f"high_below_low: H={h} L={l}"

        return True, "ok"

    def check_universe_eligibility(self, symbol, price, avg_dollar_volume,
                                   spread_pct) -> Tuple[bool, str]:
        """
        Check if a symbol meets universe requirements.
        Returns (eligible, reason).
        """
        uni = self.config.universe

        if Decimal(str(price)) < uni.min_price:
            return False, f"price_too_low: {price} < {uni.min_price}"

        if Decimal(str(price)) > uni.max_price:
            return False, f"price_too_high: {price} > {uni.max_price}"

        if Decimal(str(avg_dollar_volume)) < uni.min_avg_dollar_volume:
            return False, (f"volume_too_low: {avg_dollar_volume} "
                           f"< {uni.min_avg_dollar_volume}")

        if Decimal(str(spread_pct)) > uni.max_spread_pct:
            return False, (f"spread_too_wide: {spread_pct} "
                           f"> {uni.max_spread_pct}")

        return True, "ok"

    def is_data_healthy(self, symbol) -> bool:
        """Quick check: do we have recent valid data for this symbol?"""
        if symbol not in self.last_quotes:
            return False
        quote = self.last_quotes[symbol]
        return quote.age_ms() <= self.data_cfg.max_quote_age_ms
