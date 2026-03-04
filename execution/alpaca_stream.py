"""AlpacaStream - trade_updates WebSocket consumer.

Docs: https://docs.alpaca.markets/docs/websocket-streaming

Notes:
  - Paper trading trade_updates may arrive as binary frames; we handle both
    text (JSON) and binary (MessagePack or JSON bytes).
  - Runs in a background thread and pushes decoded messages into a Queue.
"""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, Optional

@dataclass
class StreamConfig:
    api_key: str
    secret_key: str
    paper: bool = True

class AlpacaTradeUpdatesStream:
    def __init__(self, cfg: StreamConfig, out_queue: Queue, logger=None):
        self.cfg = cfg
        self.out_queue = out_queue
        self.log = logger
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def url(self) -> str:
        return "wss://paper-api.alpaca.markets/stream" if self.cfg.paper else "wss://api.alpaca.markets/stream"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alpaca_stream", daemon=True)
        self._thread.start()
        if self.log:
            self.log.log_system("Trade updates stream started", url=self.url)

    def stop(self) -> None:
        self._stop.set()
        if self.log:
            self.log.log_system("Trade updates stream stopping")

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            if self.log:
                self.log.error("STREAM", f"Trade updates stream crashed: {e}")

    async def _run_async(self) -> None:
        import websockets  # type: ignore

        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key}))
            await ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))

            while not self._stop.is_set():
                msg = await ws.recv()
                decoded = self._decode_message(msg)
                if decoded is not None:
                    self.out_queue.put(decoded)

    def _decode_message(self, msg: Any) -> Optional[Dict[str, Any]]:
        if isinstance(msg, str):
            try:
                return json.loads(msg)
            except Exception:
                return None

        if isinstance(msg, (bytes, bytearray)):
            b = bytes(msg)
            try:
                import msgpack  # type: ignore
                return msgpack.unpackb(b, raw=False)
            except Exception:
                pass
            try:
                return json.loads(b.decode("utf-8"))
            except Exception:
                return None

        return None
