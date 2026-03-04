"""StateStore - Durable bot state for crash-safe operation.

Live trading bots must survive restarts without "forgetting" open orders,
positions, or reserved/unsettled cash. This module persists a minimal,
auditable snapshot to disk.

Design goals:
  - Small, human-readable JSON.
  - Atomic writes (write temp + replace).
  - Forward-compatible via schema_version.
  - No broker secrets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

def _json_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

@dataclass
class EngineSnapshot:
    schema_version: int
    ts_utc: str
    state: str
    positions: Dict[str, Dict[str, Any]]
    open_orders: Dict[str, Dict[str, Any]]
    daily_trades: list
    cash_ledger: Dict[str, Any]

class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, snapshot: EngineSnapshot) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(asdict(snapshot), indent=2, default=_json_default)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)
