"""
Core Engine - Main trading loop that orchestrates all subsystems.
This is the brain. It connects data -> strategy -> risk -> execution.
Runs as an event loop during market hours.
"""
import time
import signal as sig
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Optional
from queue import Queue, Empty
import pytz

from config.settings import (
    AppConfig, SystemState, TradeSignal, OrderState
)
from utils.logger import StructuredLogger
from core.broker import BrokerInterface
from core.cash_ledger import CashLedger
from risk.manager import RiskManager
from data.validator import DataValidator, QuoteData
from strategy.orb_momentum import ORBMomentumStrategy
from core.state_store import StateStore, EngineSnapshot
from execution.alpaca_stream import AlpacaTradeUpdatesStream, StreamConfig


class Position:
    """Tracks a single open position."""

    def __init__(self, symbol, qty, entry_price, entry_time, order_id):
        self.symbol = symbol
        self.qty = int(qty)
        self.entry_price = Decimal(str(entry_price))
        self.entry_time = entry_time
        self.order_id = order_id
        self.unrealized_pnl = Decimal("0.00")

    def update_pnl(self, current_price):
        price = Decimal(str(current_price))
        self.unrealized_pnl = (price - self.entry_price) * self.qty

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "unrealized_pnl": str(self.unrealized_pnl),
        }


class TradingEngine:
    """
    Main event loop. Responsibilities:
    1. Fetch data and validate it
    2. Run strategy to generate signals
    3. Pass signals through risk checks
    4. Execute approved trades via broker
    5. Monitor and reconcile continuously
    6. Handle all error states safely
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.log = StructuredLogger("engine", config.log_dir, config.log_level)

        # Initialize subsystems
        self.broker = BrokerInterface(config, self.log)
        self.cash_ledger = CashLedger(config)
        self.risk = RiskManager(config, self.log)
        self.validator = DataValidator(config, self.log)
        self.strategy = ORBMomentumStrategy(config, self.log)

        # State
        self.state = SystemState.INITIALIZING
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, dict] = {}  # order_id -> metadata
        self.daily_trades: list = []
        self.et_tz = pytz.timezone(config.timezone)

        # Watchlist (from config.universe.watchlist)
        self.watchlist = list(getattr(config.universe, "watchlist", []))
        if not self.watchlist:
            raise ValueError("Universe watchlist is empty: set UniverseConfig.watchlist")


        # Bar data cache: symbol -> {closes, highs, lows, volumes}
        self.bar_cache: Dict[str, dict] = {}

        # Cycle timing
        self.cycle_interval = 60  # seconds between main loop cycles
        self.last_reconcile = 0
        self.reconcile_interval = 300  # reconcile every 5 minutes

        self._orb_ranked_session = None  # YYYY-MM-DD when candidates were ranked

        # Shutdown handling
        self.shutdown_requested = False
        sig.signal(sig.SIGINT, self._handle_shutdown)
        sig.signal(sig.SIGTERM, self._handle_shutdown)

        self.log.log_system("Engine initialized",
                            watchlist=self.watchlist,
                            cycle_interval=self.cycle_interval)

        # Durable state (crash-safe)
        self.state_store = StateStore(
            path=(getattr(config, 'state_path', None) or str((self.log.log_dir / 'state.json')))
        )
        self._last_persist_ts = 0.0
        self._persist_interval = 15.0  # seconds

        # Trade updates stream (event-driven order/fill updates)
        self.trade_updates_q: Queue = Queue(maxsize=10_000)
        self.trade_stream = AlpacaTradeUpdatesStream(
            StreamConfig(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                paper=config.account.environment.name == 'PAPER',
            ),
            out_queue=self.trade_updates_q,
            logger=self.log,
        )
        self._processed_exec_ids = set()


    def _session_key(self, dt):
        return dt.strftime("%Y-%m-%d")

    def _compute_open5_volume_today(self, symbol: str) -> float:
        bars = self.bar_cache.get(symbol)
        if not bars:
            return 0.0
        vols = bars.get("volumes", [])
        # We assume bar_cache is in chronological order; use the last session's first 5 bars after open
        # For live/paper at open, this will become correct as bars accumulate.
        # For preloaded history, we approximate by using the first 5 bars of the most recent day in cache.
        return float(sum(vols[-390:][:5])) if len(vols) >= 395 else float(sum(vols[:5]))

    def _compute_avg_open5_volume(self, symbol: str, sessions: int = 14) -> float:
        bars = self.bar_cache.get(symbol)
        if not bars:
            return 0.0
        vols = bars.get("volumes", [])
        if len(vols) < 390 * 2:
            return float(sum(vols[:5]) / 5.0) if len(vols) >= 5 else 0.0
        # Partition into days of 390 bars from the end
        days = []
        i = len(vols)
        while i >= 390 and len(days) < sessions:
            day = vols[i-390:i]
            days.append(sum(day[:5]))
            i -= 390
        if not days:
            return 0.0
        return float(sum(days) / len(days))

    def _compute_rvol(self, symbol: str, sessions: int = 14) -> float:
        today = self._compute_open5_volume_today(symbol)
        avg = self._compute_avg_open5_volume(symbol, sessions=sessions)
        if avg <= 0:
            return 0.0
        return float(today / avg)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown on Ctrl+C."""
        self.log.log_system("Shutdown signal received",
                            signal=signum)
        self.shutdown_requested = True

    # ── Main Loop ──────────────────────────────────────────────────────

    def run(self):
        """Main entry point. Runs until shutdown or market close."""
        self.log.log_system("=" * 50)
        self.log.log_system("TRADING BOT V1.0 STARTING")
        self.log.log_system("=" * 50)

        try:
            # Step 1: Validate everything
            if not self._startup_validation():
                self.log.critical("SYSTEM", "Startup validation FAILED")
                return

            # Step 2: Load historical data for indicators
            self._load_historical_data()

            # Step 2.5: Restore durable state (if present)
            self._restore_state_from_disk()

            # Step 3: Sync with broker state
            self._sync_broker_state()

            # Step 3.5: Start trade_updates stream (best-effort)
            self.trade_stream.start()

            # Step 4: Main trading loop
            self.state = SystemState.WAITING_FOR_MARKET
            self._main_loop()

        except Exception as e:
            self.log.critical("SYSTEM", f"Unhandled exception: {e}")
            self.state = SystemState.ERROR
        finally:
            self._shutdown()

    def _main_loop(self):
        """Core loop: runs every cycle_interval seconds."""
        while not self.shutdown_requested:
            cycle_start = time.monotonic()

            try:
                # Check market status
                if not self.broker.is_market_open():
                    if self.state == SystemState.TRADING:
                        self.log.log_system("Market closed")
                        self._end_of_day_report()
                        self.state = SystemState.WAITING_FOR_MARKET
                    self._wait_for_market()
                    continue

                # Market is open
                if self.state == SystemState.WAITING_FOR_MARKET:
                    self.log.log_system("Market OPEN - starting trading")
                    self.state = SystemState.TRADING

                if self.state == SystemState.RISK_DISABLED:
                    # Still monitor positions but don't enter new ones
                    self._check_existing_positions()
                    self._check_force_flatten()
                elif self.state == SystemState.TRADING:
                    self._trading_cycle()

                # Periodic reconciliation
                now = time.monotonic()
                if now - self.last_reconcile > self.reconcile_interval:
                    self._reconcile()
                    self.last_reconcile = now

                # Periodic persistence
                if now - self._last_persist_ts > self._persist_interval:
                    self._persist_state()
                    self._last_persist_ts = now

            except KeyboardInterrupt:
                self.shutdown_requested = True
            except Exception as e:
                self.log.error("SYSTEM", f"Cycle error: {e}")
                # Don't crash on single cycle failure
                time.sleep(5)

            # Pace the loop
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, self.cycle_interval - elapsed)
            if sleep_time > 0 and not self.shutdown_requested:
                time.sleep(sleep_time)

    def _trading_cycle(self):
        """One complete trading cycle."""
        # Update settlements and order states
        self.cash_ledger.process_settlements()

        # Apply real-time trade_updates first (lower latency than polling)
        self._drain_trade_updates()
        self._update_orders_and_ledger()

        self.log.log_system("--- Trading cycle start ---",
                            positions=len(self.positions))

        # Check force flatten time
        if self._check_force_flatten():
            return

        # Process each symbol
        for symbol in self.watchlist:
            if self.shutdown_requested:
                break

            try:
                self._process_symbol(symbol)
            except Exception as e:
                self.log.error("SYSTEM",
                               f"Error processing {symbol}: {e}")

        # Check existing positions for exits
        self._check_existing_positions()

        # Log cycle summary
        equity = self.broker.get_equity()
        self.risk.update_equity(equity)
        self.log.log_system("--- Trading cycle end ---",
                            equity=str(equity),
                            risk_status=self.risk.get_status())

    def _process_symbol(self, symbol):
        """Fetch data, validate, generate signal, execute if approved."""

        # 1. Get latest quote
        raw_quote = self.broker.get_latest_quote(symbol)
        if raw_quote is None:
            return

        # 2. Validate quote
        valid, quote, reason = self.validator.validate_quote(
            symbol,
            bid=raw_quote.bid_price,
            ask=raw_quote.ask_price,
            bid_size=raw_quote.bid_size,
            ask_size=raw_quote.ask_size,
            timestamp=raw_quote.timestamp
        )

        if not valid:
            self.log.log_data(f"{symbol}: quote invalid - {reason}")
            return

        # 3. Update bar cache with latest data
        bars = self.broker.get_historical_bars(symbol, days=1, timeframe=TimeFrame.Minute)
        if isinstance(bars, tuple):
            bars = bars[0]
        if bars and len(bars) > 0:
            self._update_bar_cache(symbol, bars[-1])

        # 4. Check universe eligibility
        if symbol in self.bar_cache:
            bars = self.bar_cache[symbol]
            if len(bars["volumes"]) > 0:
                avg_dv = float(
                    sum(c * v for c, v in
                        zip(bars["closes"][-20:], bars["volumes"][-20:]))
                ) / min(len(bars["closes"]), 20)
            else:
                avg_dv = 0

            eligible, ereason = self.validator.check_universe_eligibility(
                symbol, float(quote.mid), avg_dv, float(quote.spread_pct)
            )
            if not eligible:
                return

                # 5. Feed strategy (ORB)
        if symbol not in self.bar_cache:
            return

        bars = self.bar_cache[symbol]
        if len(bars.get("closes", [])) < 20:
            return

        # Build OHLCV window
        ohlcv_window = {
            "open": bars.get("opens", []),
            "high": bars.get("highs", []),
            "low": bars.get("lows", []),
            "close": bars.get("closes", []),
            "volume": bars.get("volumes", []),
        }

        # Minutes since 09:30 ET
        now_et = datetime.now(self.et_tz)
        bar_index = max(0, int((now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)))
        bars_in_day = 390

        # Equity (fallback)
        try:
            account_equity = float(self.risk.equity)
        except Exception:
            account_equity = 100000.0

        # RVOL (opening 5-min volume vs avg opening 5-min volume)
        rvol = float(self._compute_rvol(symbol, sessions=14))

        # Rank candidates once per session after opening range completes
        session = now_et.strftime("%Y-%m-%d")
        if self._orb_ranked_session != session and bar_index >= 5:
            stats = {}
            for sym in self.watchlist:
                b = self.bar_cache.get(sym)
                if not b or len(b.get("closes", [])) < 6:
                    continue
                # opening range percent using first 5 bars of the most recent day slice
                closes = b.get("closes", [])
                highs = b.get("highs", [])
                lows = b.get("lows", [])
                vols = b.get("volumes", [])
                # Use last day slice if we have it, else use leading bars
                day_closes = closes[-390:] if len(closes) >= 390 else closes
                day_highs = highs[-390:] if len(highs) >= 390 else highs
                day_lows = lows[-390:] if len(lows) >= 390 else lows
                day_vols = vols[-390:] if len(vols) >= 390 else vols

                if len(day_closes) < 6:
                    continue
                o = day_closes[0]
                hi = max(day_highs[:5]) if len(day_highs) >= 5 else max(day_highs)
                lo = min(day_lows[:5]) if len(day_lows) >= 5 else min(day_lows)
                rng_pct = (hi - lo) / max(o, 1e-6)

                sym_rvol = float(self._compute_rvol(sym, sessions=14))
                stats[sym] = {
                    "price": float(day_closes[5]),
                    "rvol": sym_rvol,
                    "orb_range_pct": float(rng_pct),
                }
            try:
                self.strategy.rank_candidates(stats)
                self._orb_ranked_session = session
                self.log.log_system("ORB candidates ranked", count=len(getattr(self.strategy, "candidates_ranked", [])))
            except Exception as e:
                self.log.error("SYSTEM", f"ORB ranking failed: {e}")

        vv20 = ohlcv_window["volume"][-20:]
        avg_1m_vol = float(sum(vv20) / max(1, len(vv20)))

        # VWAP over last 20 bars
        cv = ohlcv_window["close"][-20:]
        vv = ohlcv_window["volume"][-20:]
        vwap_value = None
        if len(cv) == len(vv) and sum(vv) > 0:
            vwap_value = float(sum(c * v for c, v in zip(cv, vv)) / sum(vv))

        sig = self.strategy.generate_signal(
            bar_index=bar_index,
            bars_in_day=bars_in_day,
            symbol=symbol,
            ohlcv_window=ohlcv_window,
            account_equity=account_equity,
            rvol=rvol,
            avg_1m_vol=avg_1m_vol,
            vwap_value=vwap_value,
        )

        if sig is None:
            return

        # 6. Process signal
        action = getattr(sig, "action", "")
        if action == "BUY":
            self._handle_entry(symbol, quote, {"limit_price": float(sig.limit_price) if sig.limit_price else None})
        elif action == "SELL":
            self._handle_exit(symbol, quote, {"limit_price": float(sig.limit_price) if sig.limit_price else None})
        elif action == "SELL_PARTIAL":
            # partial exit not yet supported -> full exit for now
            self._handle_exit(symbol, quote, {"limit_price": float(sig.limit_price) if sig.limit_price else None})
            self._handle_exit(symbol, quote, meta)

    def _handle_entry(self, symbol, quote: QuoteData, signal_meta):
        """Process a long entry signal through risk checks and execution."""

        # Get buying power
        buying_power = self.cash_ledger.available_settled_cash()

        # Calculate stop price
        stop_price = self.strategy.get_stop_price(symbol)
        if stop_price is None:
            self.log.warning("EXECUTION",
                             "Cannot calculate stop - skipping",
                             symbol=symbol)
            return

        # Calculate position size
        shares = self.risk.calculate_position_size(
            price=float(quote.ask),  # Assume we pay the ask
            stop_price=float(stop_price),
            buying_power=buying_power
        )

        if shares <= 0:
            return

        # Estimate expected value
        state = self.strategy.get_state(symbol)
        ev, ev_details = self.strategy.estimate_expected_value(
            entry_price=float(quote.ask),
            vwap=state["vwap"],
            stop_price=float(stop_price),
            spread_cost=float(quote.spread),
            slippage_est=float(quote.mid * Decimal("0.001")),
            win_rate=0.50
        )

        # Check EV meets minimum
        total_ev = ev * shares
        if total_ev < self.config.strategy.min_expected_value_per_trade:
            self.log.log_signal(
                symbol, "ENTRY_REJECTED",
                reason="ev_too_low",
                ev_per_share=str(ev),
                total_ev=str(total_ev),
                min_required=str(
                    self.config.strategy.min_expected_value_per_trade)
            )
            return

        # Pre-trade risk check
        approved, reason = self.risk.pre_trade_check(
            symbol=symbol,
            side="buy",
            qty=shares,
            price=float(quote.ask),
            buying_power=buying_power,
            spread_pct=float(quote.spread_pct),
            expected_slippage_pct=Decimal("0.001")
        )

        if not approved:
            self.log.log_signal(symbol, "ENTRY_BLOCKED",
                                reason=reason, shares=shares)
            return

        # Execute
        if self.config.dry_run:
            self.log.log_order("DRY_RUN_BUY", symbol=symbol,
                               qty=shares, price=str(quote.ask))
            return

        # Reserve settled cash for this BUY (prevents over-commit and unsettled reuse)
        est_cost = Decimal(str(quote.ask)) * Decimal(str(shares))
        if not self.cash_ledger.can_reserve(est_cost):
            self.log.log_signal(symbol, "ENTRY_BLOCKED", reason="unsettled_or_reserved_cash", shares=shares)
            return

        # Use limit order at the ask for controlled entry
        order = self.broker.submit_limit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            limit_price=quote.ask,
            time_in_force="day"
        )

        if order:
            oid = str(order.id)
            self.cash_ledger.reserve_buy(oid, est_cost)
            self.open_orders[oid] = {"symbol": symbol, "side": "buy", "qty": shares, "est_price": str(quote.ask)}
            # Position will be added on fill by _update_orders_and_ledger

            self.log.log_order("ENTRY_SUBMITTED",
                               symbol=symbol,
                               qty=shares,
                               price=str(quote.ask),
                               stop=str(stop_price),
                               ev_per_share=str(ev),
                               total_ev=str(total_ev),
                               order_id=str(order.id))

    def _handle_exit(self, symbol, quote: QuoteData, signal_meta):
        """Process a long exit signal."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        if self.config.dry_run:
            pos.update_pnl(float(quote.bid))
            self.log.log_order("DRY_RUN_SELL", symbol=symbol,
                               qty=pos.qty, pnl=str(pos.unrealized_pnl))
            return

        # Market order for exits — we want OUT
        # Submit exit (sell). Proceeds are treated as UNSETTLED until settlement.
        order = self.broker.submit_market_order(
            symbol=symbol,
            qty=pos.qty,
            side="sell"
        )

        if order:
            oid = str(order.id)
            self.open_orders[oid] = {"symbol": symbol, "side": "sell", "qty": pos.qty, "est_price": str(quote.bid)}
            # Remove internal position immediately to avoid duplicate exits; ledger updates on fill.
            del self.positions[symbol]
            pos_map = {s: {"qty": p.qty} for s, p in self.positions.items()}
            self.risk.update_positions(pos_map)

            self.log.log_order("EXIT_EXECUTED",
                               symbol=symbol,
                               qty=pos.qty,
                               entry=str(pos.entry_price),
                               exit=str(exit_price),
                               pnl=str(pnl),
                               reason=signal_meta.get("reason"))

            self.log.log_pnl(symbol=symbol,
                             trade_pnl=str(pnl),
                             daily_pnl=str(self.risk.daily_pnl))

    # ── Position Management ────────────────────────────────────────────

    def _check_existing_positions(self):
        """Check all positions for time-based exits."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]

            # Update P&L with latest quote
            quote = self.broker.get_latest_quote(symbol)
            if quote:
                pos.update_pnl(float(quote.bid_price))

            # Check hold duration
            if self.risk.check_hold_duration(symbol, pos.entry_time):
                self.log.warning("RISK",
                                 "Max hold duration — forcing exit",
                                 symbol=symbol)
                self._force_exit(symbol, "max_hold_duration")

    def _check_force_flatten(self):
        """Flatten all positions if past force-flatten time."""
        if self.risk.check_force_flatten():
            if self.positions:
                self.log.critical("RISK",
                                  "FORCE FLATTEN TIME - closing all",
                                  positions=list(self.positions.keys()))
                self._flatten_all("force_flatten_time")
            return True
        return False

    def _force_exit(self, symbol, reason):
        """Force exit a single position."""
        if symbol not in self.positions:
            return

        if self.config.dry_run:
            self.log.log_order("DRY_RUN_FORCE_EXIT",
                               symbol=symbol, reason=reason)
            return

        success = self.broker.close_position(symbol)
        if success:
            pos = self.positions[symbol]
            self.log.log_order("FORCE_EXIT", symbol=symbol,
                               qty=pos.qty, reason=reason)
            del self.positions[symbol]

    def _flatten_all(self, reason):
        """Emergency: close everything."""
        self.log.critical("SYSTEM",
                          f"FLATTEN ALL: {reason}")

        if not self.config.dry_run:
            self.broker.cancel_all_orders()
            self.open_orders.clear()  # start clean
            self.cash_ledger.reserved.clear()
            self.broker.close_all_positions()

        self.positions.clear()
        pos_map = {}
        self.risk.update_positions(pos_map)
        self.state = SystemState.FLATTENING

    # ── Data Management ────────────────────────────────────────────────

    def _load_historical_data(self):
        """Load historical bars for all watchlist symbols."""
        self.log.log_system("Loading historical data...")
        from alpaca.data.timeframe import TimeFrame

        for symbol in self.watchlist:
            try:
                bars = self.broker.get_historical_bars(
                    symbol,
                    days=self.config.data.history_lookback_days,
                    timeframe=TimeFrame.Minute
                )

                if isinstance(bars, tuple):
                    bars = bars[0]

                if bars and len(bars) > 0:
                    opens = [float(getattr(b, "open", getattr(b, "o", 0.0))) for b in bars]
                    closes = [float(b.close) for b in bars]
                    highs = [float(b.high) for b in bars]
                    lows = [float(b.low) for b in bars]
                    volumes = [float(b.volume) for b in bars]

                    self.bar_cache[symbol] = {
                        "opens": opens,
                        "closes": closes,
                        "highs": highs,
                        "lows": lows,
                        "volumes": volumes,
                    }

                    self.log.log_data(
                        f"{symbol}: loaded {len(closes)} bars")
                else:
                    self.log.warning("DATA",
                                     f"{symbol}: no historical data")

            except Exception as e:
                self.log.error("DATA",
                               f"{symbol}: history load failed: {e}")

    def _update_bar_cache(self, symbol, bar):
        """Append latest 1-min OHLCV bar to cache (ORB-compatible)."""
        if symbol not in self.bar_cache:
            self.bar_cache[symbol] = {"closes": [], "highs": [], "lows": [], "volumes": [], "opens": []}

        cache = self.bar_cache[symbol]
        cache["opens"].append(float(getattr(bar, "open", getattr(bar, "o", 0.0))))
        cache["highs"].append(float(getattr(bar, "high", getattr(bar, "h", 0.0))))
        cache["lows"].append(float(getattr(bar, "low", getattr(bar, "l", 0.0))))
        cache["closes"].append(float(getattr(bar, "close", getattr(bar, "c", 0.0))))
        cache["volumes"].append(float(getattr(bar, "volume", getattr(bar, "v", 0.0))))

        # Keep only last 500 bars


        max_bars = 500
        for key in cache:
            if len(cache[key]) > max_bars:
                cache[key] = cache[key][-max_bars:]

    # ── Startup & Validation ───────────────────────────────────────────

    def _startup_validation(self):
        """Validate everything before trading."""
        self.state = SystemState.VALIDATING
        self.log.log_system("Running startup validation...")

        # Config validation
        errors = self.config.validate()
        if errors:
            for e in errors:
                self.log.error("VALIDATION", e)
            return False

        # Broker connectivity
        try:
            account = self.broker.get_account()
            equity = Decimal(str(account.equity))
            self.log.log_system("Broker connected",
                                equity=str(equity),
                                status=account.status)

            # Initialize settled cash ledger (cash-account conservative)
            try:
                self.cash_ledger.initialize_from_cash(Decimal(str(account.cash)))
            except Exception:
                pass

            # Update risk manager with actual equity
            self.risk.update_equity(equity)

        except Exception as e:
            self.log.critical("VALIDATION",
                              f"Broker connection failed: {e}")
            return False

        # Market clock
        clock = self.broker.get_clock()
        if clock:
            self.log.log_system("Market clock OK",
                                is_open=clock.is_open,
                                next_open=str(clock.next_open),
                                next_close=str(clock.next_close))

        self.log.log_system("Startup validation PASSED")
        return True

    def _sync_broker_state(self):
        """Sync internal state with broker on startup."""
        # Check for any existing positions
        positions = self.broker.get_positions()
        for pos in positions:
            self.log.warning("SYSTEM",
                             f"Found existing position: {pos.symbol} "
                             f"x{pos.qty}")
            self.positions[pos.symbol] = Position(
                symbol=pos.symbol,
                qty=int(pos.qty),
                entry_price=Decimal(str(pos.avg_entry_price)),
                entry_time=datetime.now(self.et_tz),  # Approximate
                order_id="synced_from_broker"
            )

        # Check for open orders
        open_orders = self.broker.get_open_orders()
        if open_orders:
            self.log.warning("SYSTEM",
                             f"Found {len(open_orders)} open orders "
                             f"- canceling for clean start")
            self.broker.cancel_all_orders()
            self.open_orders.clear()  # start clean
            self.cash_ledger.reserved.clear()


    def _update_orders_and_ledger(self):
        """Poll broker for order status changes; update positions and cash ledger."""
        if not self.open_orders:
            return

        # Get current open orders from broker
        broker_open = {str(o.id): o for o in self.broker.get_open_orders()}

        done_ids = []
        for oid, meta in list(self.open_orders.items()):
            if oid in broker_open:
                continue  # still working

            # Order is no longer open -> fetch final status
            order = self.broker.get_order(oid)
            if order is None:
                continue

            status = str(getattr(order, "status", "")).lower()
            filled_qty = int(float(getattr(order, "filled_qty", 0) or 0))
            filled_avg = getattr(order, "filled_avg_price", None)
            side = meta["side"]
            symbol = meta["symbol"]

            if status in ("canceled", "rejected", "expired"):
                # release reservation if it was a buy
                if side == "buy":
                    self.cash_ledger.release_reservation(oid)
                done_ids.append(oid)
                continue

            if status == "filled" and filled_qty > 0:
                # Determine fill price; fall back to submitted estimate
                fill_price = Decimal(str(filled_avg)) if filled_avg else Decimal(str(meta.get("est_price", "0")))
                notional = fill_price * Decimal(str(filled_qty))

                if side == "buy":
                    self.cash_ledger.confirm_buy_fill(oid, notional)
                    # create/refresh position only on fill
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        qty=filled_qty,
                        entry_price=fill_price,
                        entry_time=datetime.now(self.et_tz),
                        order_id=oid
                    )
                else:
                    # sell fill: proceeds are unsettled until settlement
                    self.cash_ledger.record_sell_fill(notional, datetime.now(self.et_tz))
                    # remove position on filled sell
                    if symbol in self.positions:
                        del self.positions[symbol]

                    # record trade result conservatively (no exact entry tracking here)
                    # Risk manager pnl is already tracked elsewhere; keep ledger primary.

                done_ids.append(oid)

        for oid in done_ids:
            self.open_orders.pop(oid, None)

        # update risk manager positions map
        pos_map = {s: {"qty": p.qty} for s, p in self.positions.items()}
        self.risk.update_positions(pos_map)

    def _reconcile(self):
        """Periodic reconciliation against broker."""
        internal = {s: {"qty": p.qty}
                    for s, p in self.positions.items()}
        matched, discrepancies = self.broker.reconcile_positions(internal)

        if not matched:
            self.log.critical("RECONCILIATION",
                              "STATE MISMATCH - disabling trading",
                              discrepancies=discrepancies)
            self.state = SystemState.RISK_DISABLED
            self.risk.trading_disabled = True
            self.risk.disable_reason = "reconciliation_mismatch"

    # ── Market Waiting ─────────────────────────────────────────────────

    def _wait_for_market(self):
        """Wait for market to open."""
        clock = self.broker.get_clock()
        if clock:
            next_open = clock.next_open
            self.log.log_system(f"Market closed. Next open: {next_open}")

        # Wait 60 seconds then check again
        for _ in range(60):
            if self.shutdown_requested:
                return
            time.sleep(1)

    # ── Reporting ──────────────────────────────────────────────────────

    def _end_of_day_report(self):
        """Generate end-of-day summary."""
        self.log.log_system("=" * 40)
        self.log.log_system("END OF DAY REPORT")
        self.log.log_system("=" * 40)

        self.log.log_pnl(
            daily_pnl=str(self.risk.daily_pnl),
            total_trades=len(self.daily_trades),
            consecutive_losses=self.risk.consecutive_losses,
            peak_equity=str(self.risk.peak_equity),
            current_equity=str(self.risk.current_equity),
        )

        for trade in self.daily_trades:
            self.log.log_system(f"  Trade: {trade}")

        self.daily_trades.clear()
        self.log.log_system("=" * 40)

    def _shutdown(self):
        """Clean shutdown."""
        self.log.log_system("Initiating shutdown...")

        # Stop stream
        try:
            self.trade_stream.stop()
        except Exception:
            pass

        # Flatten if we have positions and market is open
        if self.positions and self.broker.is_market_open():
            self.log.warning("SYSTEM",
                             "Shutting down with open positions "
                             "- flattening")
            self._flatten_all("shutdown")

        self.state = SystemState.SHUTDOWN
        self.log.log_system("Shutdown complete")

    # ── Status ─────────────────────────────────────────────────────────

    
    # ── Durable state ─────────────────────────────────────────────────

    def _restore_state_from_disk(self):
        """Best-effort restore. Broker reconciliation still determines truth."""
        snap = self.state_store.load()
        if not snap:
            return
        try:
            self.cash_ledger.load_from_dict(snap.get("cash_ledger") or {})
            self.open_orders = snap.get("open_orders") or {}
            restored_positions = snap.get("positions") or {}
            for sym, pd in restored_positions.items():
                try:
                    self.positions[sym] = Position(
                        symbol=sym,
                        qty=int(pd.get("qty", 0)),
                        entry_price=pd.get("entry_price", "0"),
                        entry_time=datetime.fromisoformat(pd["entry_time"]),
                        order_id=str(pd.get("order_id", "")),
                    )
                except Exception:
                    continue
            self.daily_trades = snap.get("daily_trades") or []
            self.log.log_system(
                "State restored from disk",
                state_path=str(self.state_store.path),
                positions=len(self.positions),
                open_orders=len(self.open_orders),
            )
        except Exception as e:
            self.log.error("SYSTEM", f"State restore failed: {e}")

    def _persist_state(self):
        """Persist minimal runtime state for crash-safe restarts."""
        try:
            snap = EngineSnapshot(
                schema_version=1,
                ts_utc=datetime.now(timezone.utc).isoformat(),
                state=str(self.state),
                positions={
                    sym: {**pos.to_dict(), "order_id": getattr(pos, "order_id", "")}
                    for sym, pos in self.positions.items()
                },
                open_orders=self.open_orders,
                daily_trades=self.daily_trades,
                cash_ledger=self.cash_ledger.to_dict(),
            )
            self.state_store.save(snap)
        except Exception as e:
            self.log.error("SYSTEM", f"State persist failed: {e}")

    def _drain_trade_updates(self, max_events: int = 500) -> None:
        """Consume queued trade_updates events and update internal state."""
        processed = 0
        while processed < max_events:
            try:
                evt = self.trade_updates_q.get_nowait()
            except Empty:
                break
            processed += 1
            try:
                if evt.get("stream") != "trade_updates":
                    continue
                data = evt.get("data") or {}
                event = data.get("event")
                order = data.get("order") or {}
                oid = str(order.get("id") or "")
                if not oid:
                    continue

                meta = self.open_orders.get(oid) or {"symbol": order.get("symbol"), "side": order.get("side")}
                meta["status"] = order.get("status")
                meta["filled_qty"] = order.get("filled_qty")
                meta["filled_avg_price"] = order.get("filled_avg_price")
                meta["last_event"] = event
                self.open_orders[oid] = meta

                exec_id = data.get("execution_id")
                if exec_id and exec_id in self._processed_exec_ids:
                    continue
                if exec_id:
                    self._processed_exec_ids.add(exec_id)

                if event in ("partial_fill", "fill"):
                    sym = str(order.get("symbol") or meta.get("symbol") or "")
                    side = str(order.get("side") or meta.get("side") or "").lower()
                    price = Decimal(str(data.get("price") or order.get("filled_avg_price") or meta.get("est_price") or "0"))
                    qty = int(Decimal(str(data.get("qty") or 0)))
                    notional = price * Decimal(str(qty))

                    if qty > 0 and sym:
                        if side == "buy":
                            self.cash_ledger.confirm_buy_fill(oid, notional)
                            try:
                                pos_qty = int(Decimal(str(data.get("position_qty") or qty)))
                            except Exception:
                                pos_qty = qty
                            self.positions[sym] = Position(sym, pos_qty, price, datetime.now(self.et_tz), oid)
                        elif side == "sell":
                            self.cash_ledger.record_sell_fill(notional, datetime.now(self.et_tz))
                            if sym in self.positions:
                                self.positions[sym].qty = max(0, self.positions[sym].qty - qty)
                                if self.positions[sym].qty == 0:
                                    del self.positions[sym]
                        self._persist_state()

                if event in ("canceled", "rejected", "expired"):
                    side = str(order.get("side") or meta.get("side") or "").lower()
                    if side == "buy":
                        self.cash_ledger.release_reservation(oid)
                    self.open_orders.pop(oid, None)
                    self._persist_state()

            except Exception:
                continue


def get_status(self):
        """Full system status."""
        return {
            "state": self.state.name,
            "positions": {s: p.to_dict()
                          for s, p in self.positions.items()},
            "risk": self.risk.get_status(),
            "watchlist": self.watchlist,
            "daily_trades": len(self.daily_trades),
        }
