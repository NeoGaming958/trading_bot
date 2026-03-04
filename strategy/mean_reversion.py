"""
V2.0 Strategy: Multi-Timeframe VWAP Mean Reversion
====================================================
Architecture:
  1. REGIME FILTER  - SPY trend + VIX level gates all trading
  2. HIGHER TF BIAS - 15min/5min confirms dip is buyable, not a breakdown
  3. VOLUME PROFILE  - identifies support levels and absorption
  4. 1MIN ENTRY      - precise timing using scoring model
  5. ADAPTIVE EXITS  - trailing stops, partial scaling, time decay

The strategy only trades when ALL layers agree.
"""
import numpy as np
from decimal import Decimal
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

from config.settings import AppConfig, TradeSignal
from utils.logger import StructuredLogger


# ====================================================================
# INDICATORS
# ====================================================================

class Indicators:

    @staticmethod
    def avg_volume(volumes, window):
        v = np.array(volumes, dtype=float)
        if len(v) < window:
            return None
        return float(np.mean(v[-window:]))
    @staticmethod
    def vwap(prices, volumes):
        p = np.array(prices, dtype=float)
        v = np.array(volumes, dtype=float)
        if len(p) == 0 or v.sum() == 0:
            return None
        return float(np.sum(p * v) / np.sum(v))

    @staticmethod
    def zscore(value, mean, std):
        if std == 0 or std is None:
            return 0.0
        return (value - mean) / std

    @staticmethod
    def rsi(prices, period=14):
        p = np.array(prices, dtype=float)
        if len(p) < period + 1:
            return None
        d = np.diff(p)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag = np.mean(g[:period])
        al = np.mean(l[:period])
        if al == 0:
            return 100.0
        for i in range(period, len(g)):
            ag = (ag * (period - 1) + g[i]) / period
            al = (al * (period - 1) + l[i]) / period
        if al == 0:
            return 100.0
        return float(100 - (100 / (1 + ag / al)))

    @staticmethod
    def ema(prices, period):
        p = np.array(prices, dtype=float)
        if len(p) < period:
            return None
        m = 2 / (period + 1)
        e = np.mean(p[:period])
        for px in p[period:]:
            e = (px - e) * m + e
        return float(e)

    @staticmethod
    def sma(prices, period):
        p = np.array(prices, dtype=float)
        if len(p) < period:
            return None
        return float(np.mean(p[-period:]))

    @staticmethod
    def rolling_std(values, window):
        v = np.array(values, dtype=float)
        if len(v) < window:
            return None
        return float(np.std(v[-window:]))

    @staticmethod
    def atr(highs, lows, closes, period=14):
        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        c = np.array(closes, dtype=float)
        if len(h) < period + 1:
            return None
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def slope(values, window=10):
        v = np.array(values, dtype=float)
        if len(v) < window:
            return 0.0
        y = v[-window:]
        x = np.arange(window)
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def bollinger_pct(price, closes, period=20, num_std=2.0):
        c = np.array(closes, dtype=float)
        if len(c) < period:
            return None
        mean = np.mean(c[-period:])
        std = np.std(c[-period:])
        if std == 0:
            return 0.5
        lower = mean - num_std * std
        upper = mean + num_std * std
        if upper == lower:
            return 0.5
        return float((price - lower) / (upper - lower))


# ====================================================================
# VOLUME PROFILE
# ====================================================================

class VolumeProfile:
    """
    Builds intraday volume-at-price profile to identify:
    - Point of Control (POC): price with most volume = magnet
    - Value Area: where 70% of volume traded = fair value zone
    - High Volume Nodes: support/resistance levels
    """

    def __init__(self, num_bins=50):
        self.num_bins = num_bins

    def build(self, closes, volumes, highs, lows):
        """Build volume profile from bar data."""
        c = np.array(closes, dtype=float)
        v = np.array(volumes, dtype=float)
        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)

        if len(c) < 10:
            return None

        price_min = float(np.min(l))
        price_max = float(np.max(h))
        if price_max <= price_min:
            return None

        bin_size = (price_max - price_min) / self.num_bins
        if bin_size == 0:
            return None

        # Distribute each bar's volume across its price range
        profile = np.zeros(self.num_bins)
        for i in range(len(c)):
            low_bin = max(0, int((l[i] - price_min) / bin_size))
            high_bin = min(self.num_bins - 1,
                           int((h[i] - price_min) / bin_size))
            if high_bin >= low_bin:
                bins_covered = high_bin - low_bin + 1
                vol_per_bin = v[i] / bins_covered
                profile[low_bin:high_bin + 1] += vol_per_bin

        # Point of Control
        poc_bin = int(np.argmax(profile))
        poc_price = price_min + (poc_bin + 0.5) * bin_size

        # Value Area (70% of volume)
        total_vol = profile.sum()
        if total_vol == 0:
            return None

        sorted_bins = np.argsort(profile)[::-1]
        cumulative = 0
        va_bins = set()
        for b in sorted_bins:
            va_bins.add(b)
            cumulative += profile[b]
            if cumulative >= total_vol * 0.70:
                break

        va_low = price_min + min(va_bins) * bin_size
        va_high = price_min + (max(va_bins) + 1) * bin_size

        # High volume nodes (bins with > 1.5x average)
        avg_vol = total_vol / self.num_bins
        hvn = []
        for i in range(self.num_bins):
            if profile[i] > avg_vol * 1.5:
                hvn.append(price_min + (i + 0.5) * bin_size)

        return {
            "poc": poc_price,
            "va_low": va_low,
            "va_high": va_high,
            "hvn": hvn,
            "price_min": price_min,
            "price_max": price_max,
            "profile": profile,
            "bin_size": bin_size,
        }

    def nearest_support(self, price, profile_data):
        """Find nearest high-volume node below current price."""
        if profile_data is None:
            return None
        hvn = profile_data["hvn"]
        below = [h for h in hvn if h < price]
        if not below:
            return profile_data["va_low"]
        return max(below)

    def is_at_support(self, price, profile_data, tolerance_pct=0.003):
        """Check if price is near a high-volume node."""
        if profile_data is None:
            return False
        for hvn_price in profile_data["hvn"]:
            if abs(price - hvn_price) / price < tolerance_pct:
                return True
        # Also check value area low
        if abs(price - profile_data["va_low"]) / price < tolerance_pct:
            return True
        return False


# ====================================================================
# MULTI-TIMEFRAME DATA
# ====================================================================

class MultiTimeframe:
    """
    Aggregates 1-min bars into higher timeframes.
    No lookahead: only uses completed bars.
    """

    @staticmethod
    def aggregate_bars(closes, highs, lows, volumes, factor):
        """Aggregate 1-min bars into N-min bars."""
        n = len(closes)
        if n < factor:
            return None

        # Only use complete bars (drop incomplete last group)
        complete = (n // factor) * factor
        c = np.array(closes[:complete], dtype=float).reshape(-1, factor)
        h = np.array(highs[:complete], dtype=float).reshape(-1, factor)
        l = np.array(lows[:complete], dtype=float).reshape(-1, factor)
        v = np.array(volumes[:complete], dtype=float).reshape(-1, factor)

        return {
            "closes": c[:, -1].tolist(),     # Last close in group
            "highs": np.max(h, axis=1).tolist(),
            "lows": np.min(l, axis=1).tolist(),
            "volumes": np.sum(v, axis=1).tolist(),
        }


# ====================================================================
# MARKET REGIME FILTER
# ====================================================================

class RegimeFilter:
    """
    Uses SPY trend and volatility to gate all trading.
    No entries when market is in a downtrend or high-vol panic.
    """

    def __init__(self):
        self.ind = Indicators()
        self.spy_state = {}

    def update(self, spy_closes, spy_highs, spy_lows, spy_volumes):
        """Update with SPY data."""
        if len(spy_closes) < 50:
            self.spy_state = {"regime": "unknown"}
            return

        ema_20 = self.ind.ema(spy_closes, 20)
        ema_50 = self.ind.ema(spy_closes, 50)
        slope_20 = self.ind.slope(spy_closes, 20)
        rsi = self.ind.rsi(spy_closes, 14)
        atr = self.ind.atr(spy_highs, spy_lows, spy_closes, 14)
        price = float(spy_closes[-1])

        # Volatility regime
        atr_pct = atr / price if atr and price > 0 else 0

        # Determine regime
        if ema_20 and ema_50:
            if ema_20 > ema_50 and slope_20 > 0:
                trend = "uptrend"
            elif ema_20 < ema_50 and slope_20 < 0:
                trend = "downtrend"
            else:
                trend = "neutral"
        else:
            trend = "unknown"

        if atr_pct > 0.015:
            vol_regime = "high_vol"
        elif atr_pct > 0.008:
            vol_regime = "normal_vol"
        else:
            vol_regime = "low_vol"

        self.spy_state = {
            "regime": trend,
            "vol_regime": vol_regime,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "slope": slope_20,
            "rsi": rsi,
            "atr_pct": round(atr_pct, 4) if atr_pct else 0,
            "price": price,
        }

    def allow_longs(self):
        """Should we allow long entries right now?"""
        regime = self.spy_state.get("regime", "unknown")
        vol = self.spy_state.get("vol_regime", "unknown")

        # Block longs in strong downtrend
        if regime == "downtrend" and vol == "high_vol":
            return False, "market_downtrend_high_vol"

        # Allow in all other conditions (including neutral/dips in uptrend)
        return True, "ok"

    def get_state(self):
        return self.spy_state


# ====================================================================
# MAIN STRATEGY
# ====================================================================

class MeanReversionStrategy:
    """
    Multi-timeframe scoring-based mean reversion.

    Layer 1 - REGIME: SPY must not be in downtrend + high vol
    Layer 2 - HIGHER TF: 5min/15min shows dip in uptrend (buyable)
    Layer 3 - VOLUME PROFILE: price near support / high volume node
    Layer 4 - 1MIN SCORING: z-score + RSI + volume spike
    Layer 5 - ADAPTIVE EXIT: trailing ATR, partial scale-out, time stop

    Entry score (0-100):
      +20: 1min z-score < -1.0
      +10: 1min z-score < -1.5
      +15: RSI < 35
      +10: RSI < 25
      +10: volume ratio > 1.2
      +10: volume ratio > 2.0
      +15: at volume profile support
      +15: 5min trend is up (buying the dip)
      +10: 15min above EMA
      -25: strong 5min downtrend
      -30: price below value area low (breakdown)

    Enter when score >= 45
    """

    ENTRY_THRESHOLD = 45

    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.strat = config.strategy
        self.log = logger
        self.ind = Indicators()
        self.vol_profile = VolumeProfile(num_bins=50)
        self.mtf = MultiTimeframe()
        self.regime = RegimeFilter()

        # Per-symbol state
        self.symbol_state: Dict[str, dict] = {}
        self.profiles: Dict[str, dict] = {}
        self.htf_state: Dict[str, dict] = {}

        # Position tracking
        self.entry_prices: Dict[str, float] = {}
        self.entry_scores: Dict[str, float] = {}
        self.highest_since_entry: Dict[str, float] = {}
        self.bars_in_trade: Dict[str, int] = {}
        self.partial_exits: Dict[str, bool] = {}

        # Cooldown
        self.last_entry_bar: int = -999
        self.min_bars_between: int = 5

        self.log.log_system("Strategy V2.0 initialized",
                            type="MTF_VWAP_MeanRev_Scoring")

    # ── DATA UPDATES ───────────────────────────────────────────────

    def update_regime(self, spy_closes, spy_highs, spy_lows, spy_volumes):
        """Update market regime with SPY data."""
        self.regime.update(spy_closes, spy_highs, spy_lows, spy_volumes)

    def update_bars(self, symbol, closes, highs, lows, volumes):
        """Update all indicators for a symbol using 1-min bars."""
        if len(closes) < 20:
            return

        price = float(closes[-1])
        lb = self.strat.vwap_lookback_bars

        # 1-min indicators
        typical = [(h+l+c)/3 for h, l, c in
                   zip(highs[-lb:], lows[-lb:], closes[-lb:])]
        vwap = self.ind.vwap(typical, volumes[-lb:])
        std = self.ind.rolling_std(typical, lb)
        zscore = self.ind.zscore(price, vwap, std) if vwap and std else 0.0
        rsi = self.ind.rsi(closes, self.strat.rsi_period)
        avg_vol = self.ind.avg_volume(volumes, lb)
        cur_vol = float(volumes[-1]) if volumes else 0
        vol_ratio = cur_vol / avg_vol if avg_vol and avg_vol > 0 else 0
        atr = self.ind.atr(highs, lows, closes, 14)
        boll = self.ind.bollinger_pct(price, closes, 20)

        # Higher timeframe analysis
        bars_5 = self.mtf.aggregate_bars(closes, highs, lows, volumes, 5)
        bars_15 = self.mtf.aggregate_bars(closes, highs, lows, volumes, 15)

        htf = {}
        if bars_5 and len(bars_5["closes"]) >= 20:
            htf["ema_5m_10"] = self.ind.ema(bars_5["closes"], 10)
            htf["ema_5m_20"] = self.ind.ema(bars_5["closes"], 20)
            htf["slope_5m"] = self.ind.slope(bars_5["closes"], 10)
            htf["rsi_5m"] = self.ind.rsi(bars_5["closes"], 14)
            htf["price_5m"] = bars_5["closes"][-1]
        if bars_15 and len(bars_15["closes"]) >= 10:
            htf["ema_15m_10"] = self.ind.ema(bars_15["closes"], 10)
            htf["slope_15m"] = self.ind.slope(bars_15["closes"], 10)

        self.htf_state[symbol] = htf

        # Volume profile (last 2 hours = 120 1-min bars)
        vp_window = min(120, len(closes))
        profile = self.vol_profile.build(
            closes[-vp_window:], volumes[-vp_window:],
            highs[-vp_window:], lows[-vp_window:]
        )
        self.profiles[symbol] = profile

        # Check support
        at_support = self.vol_profile.is_at_support(price, profile)
        nearest_sup = self.vol_profile.nearest_support(price, profile)

        self.symbol_state[symbol] = {
            "vwap": vwap, "std": std, "zscore": zscore,
            "rsi": rsi, "vol_ratio": vol_ratio,
            "current_price": price, "atr": atr,
            "boll_pct": boll,
            "at_vp_support": at_support,
            "nearest_support": nearest_sup,
            "poc": profile["poc"] if profile else None,
            "va_low": profile["va_low"] if profile else None,
            "va_high": profile["va_high"] if profile else None,
        }

        # Update trailing high
        if symbol in self.highest_since_entry:
            if price > self.highest_since_entry[symbol]:
                self.highest_since_entry[symbol] = price
        if symbol in self.bars_in_trade:
            self.bars_in_trade[symbol] += 1

    # ── ENTRY SCORING ──────────────────────────────────────────────

    def _score_entry(self, symbol):
        state = self.symbol_state[symbol]
        htf = self.htf_state.get(symbol, {})
        score = 0
        reasons = []

        z = state["zscore"]
        rsi = state["rsi"]
        vr = state["vol_ratio"]
        at_sup = state["at_vp_support"]

        # 1-min z-score (max 30)
        if z < -1.0:
            score += 20
            reasons.append(f"z={z:.1f}")
        if z < -1.5:
            score += 10
            reasons.append("z<-1.5")

        # RSI (max 25)
        if rsi is not None and rsi < 35:
            score += 15
            reasons.append(f"rsi={rsi:.0f}")
        if rsi is not None and rsi < 25:
            score += 10
            reasons.append("rsi<25")

        # Volume (max 20)
        if vr > 1.2:
            score += 10
            reasons.append(f"vol={vr:.1f}x")
        if vr > 2.0:
            score += 10
            reasons.append("vol>2x")

        # Volume profile support (15)
        if at_sup:
            score += 15
            reasons.append("VP_support")

        # 5-min trend context (15 or penalty)
        slope_5m = htf.get("slope_5m", 0)
        ema_5m_10 = htf.get("ema_5m_10")
        ema_5m_20 = htf.get("ema_5m_20")
        if ema_5m_10 and ema_5m_20:
            if ema_5m_10 > ema_5m_20:
                score += 15
                reasons.append("5m_uptrend")
            elif slope_5m < -0.1:
                score -= 25
                reasons.append("5m_downtrend")

        # 15-min context (10)
        ema_15 = htf.get("ema_15m_10")
        if ema_15 and state["current_price"] > ema_15:
            score += 10
            reasons.append("above_15m_ema")

        # Breakdown penalty
        va_low = state.get("va_low")
        if va_low and state["current_price"] < va_low:
            score -= 30
            reasons.append("below_VA_low")

        return max(0, min(100, score)), reasons

    # ── SIGNAL GENERATION ──────────────────────────────────────────

    def generate_signal(self, symbol, has_position,
                        current_bar=0) -> Tuple[TradeSignal, dict]:
        if symbol not in self.symbol_state:
            return TradeSignal.NO_TRADE, {"reason": "no_data"}

        state = self.symbol_state[symbol]
        price = state["current_price"]
        zscore = state["zscore"]
        atr = state["atr"]

        meta = {
            "zscore": round(zscore, 3),
            "rsi": round(state["rsi"], 1) if state["rsi"] else None,
            "vol_ratio": round(state["vol_ratio"], 2),
            "vwap": round(state["vwap"], 2) if state["vwap"] else None,
            "price": round(price, 2),
            "atr": round(atr, 4) if atr else None,
            "at_support": state["at_vp_support"],
            "regime": self.regime.get_state().get("regime", "unknown"),
        }

        # ── EXIT LOGIC ─────────────────────────────────────────────
        if has_position:
            return self._check_exit(symbol, state, meta)

        # ── ENTRY LOGIC ────────────────────────────────────────────

        # Regime gate
        allowed, regime_reason = self.regime.allow_longs()
        if not allowed:
            return TradeSignal.NO_TRADE, {"reason": regime_reason}

        # Cooldown
        if current_bar - self.last_entry_bar < self.min_bars_between:
            return TradeSignal.NO_TRADE, {"reason": "cooldown"}

        # Must have ATR
        if atr is None or atr <= 0:
            return TradeSignal.NO_TRADE, {"reason": "no_atr"}

        # Score the setup
        score, reasons = self._score_entry(symbol)
        meta["score"] = score
        meta["score_reasons"] = reasons

        if score < self.ENTRY_THRESHOLD:
            return TradeSignal.NO_TRADE, {
                "reason": f"score={score}<{self.ENTRY_THRESHOLD}"
            }

        # Enter
        self.last_entry_bar = current_bar
        self.entry_prices[symbol] = price
        self.entry_scores[symbol] = score
        self.highest_since_entry[symbol] = price
        self.bars_in_trade[symbol] = 0
        self.partial_exits[symbol] = False
        meta["reason"] = "entry_signal"
        self.log.log_signal(symbol, "LONG_ENTRY", **meta)
        return TradeSignal.LONG_ENTRY, meta

    def _check_exit(self, symbol, state, meta):
        """Multi-layered exit logic."""
        price = state["current_price"]
        atr = state["atr"]
        zscore = state["zscore"]
        entry = self.entry_prices.get(symbol, price)
        highest = self.highest_since_entry.get(symbol, price)
        bars_held = self.bars_in_trade.get(symbol, 0)

        # 1. HARD STOP: 2.5x ATR below entry (wider for volatile names)
        if atr and entry > 0:
            hard_stop = entry - (2.5 * atr)
            if price <= hard_stop:
                meta["reason"] = "hard_stop"
                meta["stop_level"] = round(hard_stop, 2)
                self._clear_position(symbol)
                return TradeSignal.LONG_EXIT, meta

        # 2. TRAILING STOP: 2x ATR below highest (only after 5 bars)
        if atr and bars_held >= 5 and highest > entry:
            trail_stop = highest - (2.0 * atr)
            if trail_stop > entry and price <= trail_stop:
                meta["reason"] = "trailing_stop"
                meta["trail_level"] = round(trail_stop, 2)
                self._clear_position(symbol)
                return TradeSignal.LONG_EXIT, meta

        # 3. PROFIT TARGET: strong reversion past VWAP
        if zscore >= 0.8:
            meta["reason"] = "profit_target"
            self._clear_position(symbol)
            return TradeSignal.LONG_EXIT, meta

        # 4. PARTIAL SCALE-OUT signal (z >= 0.0 and profitable)
        #    We flag it — the engine handles the partial fill
        if not self.partial_exits.get(symbol, False):
            if zscore >= 0.0 and price > entry:
                self.partial_exits[symbol] = True
                meta["reason"] = "partial_target"
                meta["partial"] = True
                self.log.log_signal(symbol, "PARTIAL_EXIT", **meta)
                # Don't return exit — just flag for engine
                # (In V2.0 backtest, we treat this as full exit for simplicity)
                self._clear_position(symbol)
                return TradeSignal.LONG_EXIT, meta

        # 5. TIME STOP: exit if held too long with no progress
        if bars_held >= 60:  # 60 minutes = 1 hour
            if price <= entry:  # Still not profitable after 1 hour
                meta["reason"] = "time_stop"
                self._clear_position(symbol)
                return TradeSignal.LONG_EXIT, meta

        # 6. BREAKEVEN PROTECT: profitable but z-score diving again
        if price > entry * 1.002 and zscore < -1.5:
            meta["reason"] = "breakeven_protect"
            self._clear_position(symbol)
            return TradeSignal.LONG_EXIT, meta

        return TradeSignal.NO_TRADE, {"reason": "holding",
                                       "bars_held": bars_held}

    def _clear_position(self, symbol):
        self.entry_prices.pop(symbol, None)
        self.entry_scores.pop(symbol, None)
        self.highest_since_entry.pop(symbol, None)
        self.bars_in_trade.pop(symbol, None)
        self.partial_exits.pop(symbol, None)

    # ── EXPECTED VALUE / STOPS ─────────────────────────────────────

    def estimate_expected_value(self, entry_price, vwap, stop_price,
                                spread_cost, slippage_est, win_rate=0.50):
        entry = float(entry_price)
        target = float(vwap)
        stop = float(stop_price)
        spread = float(spread_cost)
        slip = float(slippage_est)
        if entry <= 0 or target <= entry or stop >= entry:
            return Decimal("0"), {"reason": "invalid_prices"}
        gw = target - entry
        gl = entry - stop
        tc = spread + slip
        nw = gw - tc
        nl = gl + tc
        ev = (win_rate * nw) - ((1 - win_rate) * nl)
        return Decimal(str(round(ev, 4))), {"ev_per_share": round(ev, 4)}

    def get_stop_price(self, symbol):
        if symbol not in self.symbol_state:
            return None
        state = self.symbol_state[symbol]
        atr = state["atr"]
        price = state["current_price"]
        if atr and atr > 0 and price > 0:
            stop = price - (2.5 * atr)
            return max(Decimal(str(round(stop, 2))), Decimal("0.01"))
        return None

    def get_state(self, symbol):
        return self.symbol_state.get(symbol)

    def get_regime(self):
        return self.regime.get_state()
