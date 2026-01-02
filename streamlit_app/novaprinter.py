# Minimal qBittorrent-style printer hooks for aggregation.

from __future__ import annotations

import re
import threading
from typing import Any, Dict, Optional


_thread_ctx = threading.local()
_global_collector = None
_patched_threading = False
_original_thread_init = None
_original_thread_run = None


def set_thread_context(
    engine_name: Optional[str] = None,
    engine_url: Optional[str] = None,
    collector: Any = None,
) -> None:
    _thread_ctx.engine_name = engine_name
    _thread_ctx.engine_url = engine_url
    _thread_ctx.collector = collector
    global _global_collector
    if collector is not None:
        _global_collector = collector


def clear_thread_context() -> None:
    _thread_ctx.engine_name = None
    _thread_ctx.engine_url = None
    _thread_ctx.collector = None


def get_thread_context() -> Dict[str, Any]:
    return {
        "engine_name": getattr(_thread_ctx, "engine_name", None),
        "engine_url": getattr(_thread_ctx, "engine_url", None),
        "collector": getattr(_thread_ctx, "collector", None),
    }


def install_thread_propagation() -> None:
    global _patched_threading, _original_thread_init, _original_thread_run
    if _patched_threading:
        return

    _original_thread_init = threading.Thread.__init__
    _original_thread_run = threading.Thread.run

    def _patched_init(self, *args, **kwargs):
        _original_thread_init(self, *args, **kwargs)
        self._novaprinter_ctx = get_thread_context()

    def _patched_run(self, *args, **kwargs):
        ctx = getattr(self, "_novaprinter_ctx", None)
        should_set = ctx and any(v is not None for v in ctx.values())
        if should_set:
            set_thread_context(**ctx)
        try:
            return _original_thread_run(self, *args, **kwargs)
        finally:
            if should_set:
                clear_thread_context()

    threading.Thread.__init__ = _patched_init
    threading.Thread.run = _patched_run
    _patched_threading = True


def prettyPrinter(data: Dict[str, Any]) -> None:
    item = dict(data or {})

    engine_name = getattr(_thread_ctx, "engine_name", None)
    engine_url = getattr(_thread_ctx, "engine_url", None)
    if engine_name and "engine_name" not in item:
        item["engine_name"] = engine_name
    if engine_url and "engine_url" not in item:
        item["engine_url"] = engine_url

    collector = getattr(_thread_ctx, "collector", None) or _global_collector
    if collector is not None:
        collector.add(item)
        return

    # Fallback to a simple print for standalone runs.
    print(item)


_SIZE_RE = re.compile(r"^\s*([0-9.,]+)\s*([kmgtpe]?i?b)?\s*$", re.I)


def anySizeToBytes(size: Any) -> int:
    if size is None:
        return -1
    if isinstance(size, (int, float)):
        return int(size)

    text = str(size).strip()
    if not text:
        return -1

    if text.isdigit():
        return int(text)

    match = _SIZE_RE.match(text)
    if not match:
        return -1

    value = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "B").upper()

    if unit.endswith("IB"):
        scale = 1024
        unit = unit[0] + "B"
    else:
        scale = 1000

    multipliers = {
        "B": 1,
        "KB": scale,
        "MB": scale ** 2,
        "GB": scale ** 3,
        "TB": scale ** 4,
        "PB": scale ** 5,
        "EB": scale ** 6,
    }
    return int(value * multipliers.get(unit, 1))
