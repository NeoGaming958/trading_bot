"""
Immutable structured logging.
Every order, fill, signal, and risk check gets logged with timestamps.
Logs are append-only JSON lines — one event per line, easy to parse.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


class DecimalEncoder(json.JSONEncoder):
    """Handles Decimal serialization in JSON logs."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        if hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)


class StructuredLogger:
    """
    Dual-output logger:
    - Human-readable to console
    - Machine-readable JSON lines to file (immutable audit log)
    """

    def __init__(self, name, log_dir="logs", level="INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        # Console handler — human readable
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        console.setFormatter(console_fmt)
        self.logger.addHandler(console)

        # File handler — JSON lines, append only
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"bot_{today}.jsonl"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler

    def _log_structured(self, level, event_type, message, **data):
        """Write structured JSON event to file, readable text to console."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event_type,
            "msg": message,
        }
        record.update(data)

        # JSON to file
        json_line = json.dumps(record, cls=DecimalEncoder)
        self._file_handler.stream.write(json_line + "\n")
        self._file_handler.stream.flush()

        # Readable to console
        detail = ""
        if data:
            detail = " | " + " ".join(f"{k}={v}" for k, v in data.items())
        getattr(self.logger, level.lower())(
            f"[{event_type}] {message}{detail}"
        )

    # ── Convenience methods ────────────────────────────────────────────

    def info(self, event_type, message, **data):
        self._log_structured("INFO", event_type, message, **data)

    def warning(self, event_type, message, **data):
        self._log_structured("WARNING", event_type, message, **data)

    def error(self, event_type, message, **data):
        self._log_structured("ERROR", event_type, message, **data)

    def critical(self, event_type, message, **data):
        self._log_structured("CRITICAL", event_type, message, **data)

    # ── Domain-specific log methods ────────────────────────────────────

    def log_order(self, action, **order_data):
        """Log order lifecycle events."""
        self.info("ORDER", action, **order_data)

    def log_fill(self, **fill_data):
        """Log fill events."""
        self.info("FILL", "Order filled", **fill_data)

    def log_signal(self, symbol, signal, **signal_data):
        """Log strategy signals."""
        self.info("SIGNAL", f"{symbol}: {signal}", **signal_data)

    def log_risk(self, check_name, passed, **risk_data):
        """Log risk check results."""
        level = "INFO" if passed else "WARNING"
        self._log_structured(
            level, "RISK", f"{check_name}: {'PASS' if passed else 'FAIL'}",
            **risk_data
        )

    def log_reconciliation(self, matched, **recon_data):
        """Log broker reconciliation results."""
        level = "INFO" if matched else "ERROR"
        self._log_structured(
            level, "RECONCILIATION",
            f"{'MATCHED' if matched else 'MISMATCH'}", **recon_data
        )

    def log_system(self, message, **data):
        """Log system-level events."""
        self.info("SYSTEM", message, **data)

    def log_data(self, message, **data):
        """Log data validation events."""
        self.info("DATA", message, **data)

    def log_pnl(self, **pnl_data):
        """Log P&L updates."""
        self.info("PNL", "P&L update", **pnl_data)
