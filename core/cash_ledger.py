"""Cash Ledger - Cash-account settlement (T+1) and reserved funds tracking.

This ledger is conservative: it prevents re-using sale proceeds until settlement.
It also reserves cash for pending BUY orders so you cannot over-commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import pytz

from config.settings import AppConfig


@dataclass
class UnsettledItem:
    amount: Decimal
    release_at: datetime
    reason: str


class CashLedger:
    def __init__(self, config: AppConfig):
        self.config = config
        self.et_tz = pytz.timezone(config.timezone)

        self.settled_cash: Decimal = Decimal("0")
        # order_id -> reserved amount (estimated, then adjusted)
        self.reserved: Dict[str, Decimal] = {}
        self.unsettled: List[UnsettledItem] = []

    def initialize_from_cash(self, cash: Decimal):
        self.settled_cash = Decimal(str(cash))

    def to_dict(self) -> dict:
        return {
            "settled_cash": str(self.settled_cash),
            "reserved": {k: str(v) for k, v in self.reserved.items()},
            "unsettled": [
                {"amount": str(u.amount), "release_at": u.release_at.isoformat(), "reason": u.reason}
                for u in self.unsettled
            ],
        }

    def load_from_dict(self, d: dict) -> None:
        if not d:
            return
        self.settled_cash = Decimal(str(d.get("settled_cash", "0")))
        self.reserved = {k: Decimal(str(v)) for k, v in (d.get("reserved") or {}).items()}
        self.unsettled = []
        for u in d.get("unsettled") or []:
            try:
                self.unsettled.append(
                    UnsettledItem(
                        amount=Decimal(str(u.get("amount", "0"))),
                        release_at=datetime.fromisoformat(u["release_at"]),
                        reason=str(u.get("reason", "")),
                    )
                )
            except Exception:
                continue

    def _add_trading_days(self, dt: datetime, days: int) -> datetime:
        # Simple trading-day add: skip weekends. (Holiday calendar not modeled.)
        cur = dt
        added = 0
        while added < days:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                added += 1
        return cur

    def process_settlements(self, now: Optional[datetime] = None):
        if now is None:
            now = datetime.now(self.et_tz)
        still = []
        for item in self.unsettled:
            if now >= item.release_at:
                self.settled_cash += item.amount
            else:
                still.append(item)
        self.unsettled = still

    def available_settled_cash(self) -> Decimal:
        # spendable = settled - reserved
        reserved_total = sum(self.reserved.values(), Decimal("0"))
        return max(Decimal("0"), self.settled_cash - reserved_total)

    def can_reserve(self, amount: Decimal) -> bool:
        return amount <= self.available_settled_cash()

    def reserve_buy(self, order_id: str, est_cost: Decimal) -> None:
        self.reserved[order_id] = Decimal(str(est_cost))

    def release_reservation(self, order_id: str) -> None:
        self.reserved.pop(order_id, None)

    def confirm_buy_fill(self, order_id: str, actual_cost: Decimal) -> None:
        # On fill, spend settled cash. Release reservation first.
        self.release_reservation(order_id)
        self.settled_cash -= Decimal(str(actual_cost))
        if self.settled_cash < 0:
            # clamp; engine should prevent this
            self.settled_cash = Decimal("0")

    def record_sell_fill(self, proceeds: Decimal, fill_time: datetime, reason: str = "sell_proceeds") -> None:
        # Proceeds settle after settlement_days trading days (T+1 in US equities now).
        days = int(self.config.account.settlement_days)
        release = self._add_trading_days(fill_time, days)
        self.unsettled.append(UnsettledItem(amount=Decimal(str(proceeds)), release_at=release, reason=reason))
