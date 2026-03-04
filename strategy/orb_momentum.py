# ORBMomentumStrategy V3.1 - entry stabilization + payout shaping
#
# Drop-in replacement for your existing orb_momentum.py.
# Keeps your indicator stack and cash constraints, adds:
#   - breakout buffer
#   - consecutive-close confirmation
#   - candidate ranking hook
#   - partial TP + breakeven + time stop consistency

import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass

from config.settings import AppConfig, TradeSignal
from utils.logger import StructuredLogger


class Indicators:
    @staticmethod
    def vwap(prices, volumes):
        p = np.array(prices, dtype=float)
        v = np.array(volumes, dtype=float)
        if len(p) == 0 or v.sum() == 0:
            return None
        return float(np.sum(p * v) / np.sum(v))

    @staticmethod
    def atr(highs, lows, closes, period=14):
        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        c = np.array(closes, dtype=float)
        if len(h) < period + 1:
            return None
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
        )
        return float(np.mean(tr[-period:]))

    @staticmethod
    def relative_volume(current_vol, avg_vol):
        if avg_vol is None or avg_vol <= 0:
            return 0.0
        return float(current_vol) / float(avg_vol)


@dataclass
class OpeningRange:
    symbol: str
    range_high: float = 0.0
    range_low: float = float("inf")
    range_close: float = 0.0
    range_open: float = 0.0
    range_volume: float = 0.0
    is_bullish: bool = False
    is_complete: bool = False
    bars_counted: int = 0
    range_size: float = 0.0
    range_size_pct: float = 0.0


@dataclass
class PositionState:
    symbol: str
    entry_price: float
    entry_time: int = 0
    entry_atr: float = 0.0
    initial_stop: float = 0.0
    trailing_stop: float = 0.0
    profit_target: float = 0.0
    highest_since_entry: float = 0.0
    bars_held: int = 0
    breakeven_activated: bool = False
    partial_taken: bool = False


class ORBMomentumStrategy:
    OPENING_RANGE_BARS = 5
    MIN_RELATIVE_VOLUME = 3.0
    MIN_PRICE = 10.0
    MIN_ATR = 0.50
    MIN_AVG_VOLUME = 500_000

    CONFIRM_BARS = 3
    BREAKOUT_BUFFER_BPS = 5
    MAX_CANDIDATES_PER_DAY = 5

    ATR_STOP_MULTIPLIER = 0.10
    PROFIT_TARGET_RR = 2.0
    PARTIAL_TP_R = 1.0
    BREAKEVEN_ACTIVATION_R = 0.8
    TIME_STOP_BARS = 90

    MAX_ENTRY_BAR = 60
    NO_ENTRY_LAST_BARS = 30

    RISK_PER_TRADE_PCT = 0.01
    MAX_POSITIONS = 1

    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.cfg = config
        self.log = logger
        self.orb: Dict[str, OpeningRange] = {}
        self.pos: Optional[PositionState] = None
        self.candidates_ranked: List[str] = []
        self.session_locked: bool = False

    def _buffer(self, price: float) -> float:
        return max(price * (self.BREAKOUT_BUFFER_BPS / 10_000), 0.01)

    def on_new_session(self):
        self.orb = {}
        self.pos = None
        self.candidates_ranked = []
        self.session_locked = False

    def update_opening_range(self, symbol: str, o: float, h: float, l: float, c: float, v: float):
        st = self.orb.get(symbol)
        if st is None:
            st = OpeningRange(symbol=symbol, range_open=o, range_close=c, range_high=h, range_low=l, range_volume=v, bars_counted=1)
            self.orb[symbol] = st
            return st
        if st.is_complete:
            return st
        st.range_high = max(st.range_high, h)
        st.range_low = min(st.range_low, l)
        st.range_close = c
        st.range_volume += v
        st.bars_counted += 1
        if st.bars_counted >= self.OPENING_RANGE_BARS:
            st.is_complete = True
            st.is_bullish = st.range_close > st.range_open
            st.range_size = st.range_high - st.range_low
            st.range_size_pct = (st.range_size / st.range_open) if st.range_open else 0.0
        return st

    def rank_candidates(self, stats: Dict[str, Dict[str, float]]):
        scored = []
        for sym, s in stats.items():
            if s.get("price", 0) < self.MIN_PRICE:
                continue
            if s.get("atr", 0) < self.MIN_ATR:
                continue
            if s.get("avg_volume", 0) < self.MIN_AVG_VOLUME:
                continue
            if s.get("rvol", 0) < self.MIN_RELATIVE_VOLUME:
                continue
            score = float(s["rvol"]) * float(s.get("orb_range_pct", 0))
            scored.append((score, sym))
        scored.sort(reverse=True)
        self.candidates_ranked = [sym for _, sym in scored[: self.MAX_CANDIDATES_PER_DAY]]
        self.session_locked = True

    def generate_signal(
        self,
        bar_index: int,
        bars_in_day: int,
        symbol: str,
        ohlcv_window: Dict[str, List[float]],
        account_equity: float,
        rvol: float,
        avg_1m_vol: float,
        vwap_value: Optional[float] = None,
    ) -> Optional[TradeSignal]:
        closes = ohlcv_window["close"]
        highs = ohlcv_window["high"]
        lows = ohlcv_window["low"]
        vols = ohlcv_window["volume"]

        st = self.update_opening_range(symbol, ohlcv_window["open"][-1], highs[-1], lows[-1], closes[-1], vols[-1])
        if not st.is_complete:
            return None

        if bar_index > self.MAX_ENTRY_BAR or (bars_in_day - bar_index) <= self.NO_ENTRY_LAST_BARS:
            return None

        if self.pos is not None:
            return self._manage_position(bar_index, closes[-1])

        if self.session_locked and self.candidates_ranked and symbol not in self.candidates_ranked:
            return None

        buf = self._buffer(closes[-1])
        if len(closes) < self.CONFIRM_BARS:
            return None
        recent = closes[-self.CONFIRM_BARS:]

        # Long-only default (cash account). Shorts should be implemented under margin rules.
        level = st.range_high + buf
        if not st.is_bullish:
            return None
        if not all(px > level for px in recent):
            return None
        if vwap_value is not None and closes[-1] < vwap_value:
            return None

        atr = Indicators.atr(highs, lows, closes, 14) or 0.0
        if atr < self.MIN_ATR:
            return None

        stop = level - (self.ATR_STOP_MULTIPLIER * atr)
        risk_ps = max(level - stop, 0.01)
        risk_dollars = max(account_equity * self.RISK_PER_TRADE_PCT, 1.0)
        shares = int(risk_dollars // risk_ps)
        if shares <= 0:
            return None

        self.pos = PositionState(
            symbol=symbol,
            entry_price=level,
            entry_time=bar_index,
            entry_atr=atr,
            initial_stop=stop,
            trailing_stop=stop,
            profit_target=level + self.PROFIT_TARGET_RR * risk_ps,
            highest_since_entry=level,
        )
        return TradeSignal(action="BUY", symbol=symbol, qty=Decimal(shares), limit_price=Decimal(str(level)))

    def _manage_position(self, bar_index: int, last_price: float) -> Optional[TradeSignal]:
        p = self.pos
        if p is None:
            return None
        p.bars_held += 1
        p.highest_since_entry = max(p.highest_since_entry, last_price)
        risk_ps = max(p.entry_price - p.initial_stop, 0.01)
        r = (last_price - p.entry_price) / risk_ps

        if (not p.breakeven_activated) and r >= self.BREAKEVEN_ACTIVATION_R:
            p.trailing_stop = max(p.trailing_stop, p.entry_price)
            p.breakeven_activated = True

        if (not p.partial_taken) and r >= self.PARTIAL_TP_R:
            p.partial_taken = True
            return TradeSignal(action="SELL_PARTIAL", symbol=p.symbol, qty=Decimal("0"), limit_price=Decimal(str(last_price)))

        if last_price <= p.trailing_stop:
            self.pos = None
            return TradeSignal(action="SELL", symbol=p.symbol, qty=Decimal("0"), limit_price=Decimal(str(last_price)))

        if p.bars_held >= self.TIME_STOP_BARS and last_price <= p.entry_price:
            self.pos = None
            return TradeSignal(action="SELL", symbol=p.symbol, qty=Decimal("0"), limit_price=Decimal(str(last_price)))

        return None

    # --- Engine compatibility helpers ---
    def get_stop_price(self, symbol: str):
        """Return current stop for active position (engine uses this for sizing)."""
        if self.pos is None:
            return None
        if self.pos.symbol != symbol:
            return None
        return float(self.pos.trailing_stop or self.pos.initial_stop)

