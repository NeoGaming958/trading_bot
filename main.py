#!/usr/bin/env python3
"""
Trading Bot V1.0 - Entry Point
Usage:
    python main.py              # Run live paper trading
    python main.py --status     # Check account status
    python main.py --validate   # Validate config and connectivity
    python main.py --dry-run    # Signals only, no orders
"""
import sys
import json
from config.settings import AppConfig
from utils.logger import StructuredLogger
from core.engine import TradingEngine
from core.broker import BrokerInterface


def main():
    config = AppConfig()

    # Load credentials from environment
    try:
        config.load_credentials()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Run:")
        print('  export ALPACA_API_KEY="your_key"')
        print('  export ALPACA_SECRET_KEY="your_secret"')
        sys.exit(1)

    # Parse command line args
    args = set(sys.argv[1:])

    if "--status" in args:
        show_status(config)
        return

    if "--validate" in args:
        validate(config)
        return

    if "--dry-run" in args:
        config.dry_run = True
        print("DRY RUN MODE — signals calculated, no orders submitted")

    # Run the engine
    engine = TradingEngine(config)
    engine.run()


def show_status(config):
    """Show current account and position status."""
    log = StructuredLogger("status", level="WARNING")
    broker = BrokerInterface(config, log)

    account = broker.get_account()
    positions = broker.get_positions()
    open_orders = broker.get_open_orders()
    clock = broker.get_clock()

    print("\n" + "=" * 50)
    print("  TRADING BOT V1.0 - STATUS")
    print("=" * 50)
    print(f"  Account Type:   CASH (Paper)")
    print(f"  Equity:         ${account.equity}")
    print(f"  Cash:           ${account.cash}")
    print(f"  Buying Power:   ${account.buying_power}")
    print(f"  Market Open:    {clock.is_open if clock else 'Unknown'}")
    if clock:
        print(f"  Next Open:      {clock.next_open}")
        print(f"  Next Close:     {clock.next_close}")
    print(f"  Positions:      {len(positions)}")
    for pos in positions:
        print(f"    {pos.symbol}: {pos.qty} shares "
              f"@ ${pos.avg_entry_price} "
              f"(P&L: ${pos.unrealized_pl})")
    print(f"  Open Orders:    {len(open_orders)}")
    for order in open_orders:
        print(f"    {order.symbol}: {order.side} {order.qty} "
              f"({order.status})")
    print("=" * 50 + "\n")


def validate(config):
    """Validate configuration and connectivity."""
    print("\nValidating...")

    # Config
    errors = config.validate()
    if errors:
        print("CONFIG ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Config: OK")

    # Broker
    try:
        log = StructuredLogger("validate", level="WARNING")
        broker = BrokerInterface(config, log)
        account = broker.get_account()
        print(f"Broker: OK (equity=${account.equity})")
    except Exception as e:
        print(f"Broker: FAILED ({e})")

    # Market data
    try:
        quote = broker.get_latest_quote("SPY")
        if quote:
            print(f"Data Feed: OK (SPY bid={quote.bid_price})")
        else:
            print("Data Feed: FAILED (no quote)")
    except Exception as e:
        print(f"Data Feed: FAILED ({e})")

    print("Validation complete.\n")


if __name__ == "__main__":
    main()
