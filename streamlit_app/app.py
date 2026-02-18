from __future__ import annotations

import csv
import importlib.util
import io
import queue
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

import helpers
import novaprinter


ROOT = Path(__file__).resolve().parent
PLUGIN_DIR = ROOT.parent / "plugins"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

novaprinter.install_thread_propagation()
helpers.install_urllib_patches()


EXCLUDE_FILES = {"__init__.py"}
NAME_RE = re.compile(r"^\s*name\s*=\s*['\"]([^'\"]+)['\"]", re.M)

MAX_HISTORY = 20

# Size presets for the size filter (in bytes).
SIZE_PRESETS = {
    "Any": (0, 0),
    "< 500 MB": (0, 500 * 1024**2),
    "500 MB - 2 GB": (500 * 1024**2, 2 * 1024**3),
    "2 GB - 10 GB": (2 * 1024**3, 10 * 1024**3),
    "10 GB - 50 GB": (10 * 1024**3, 50 * 1024**3),
    "> 50 GB": (50 * 1024**3, 0),
}


def extract_display_name(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.stem
    match = NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    return path.stem


@st.cache_data
def list_plugins(plugin_dir: Path) -> List[Dict[str, str]]:
    plugins: List[Dict[str, str]] = []
    if not plugin_dir.exists():
        return plugins
    for path in sorted(plugin_dir.glob("*.py")):
        if path.name in EXCLUDE_FILES:
            continue
        plugins.append(
            {
                "path": str(path),
                "stem": path.stem,
                "label": extract_display_name(path),
            }
        )
    return plugins


def load_module(path: Path):
    module_name = f"plugin_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_engine_class(module: Any):
    candidates = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if not isinstance(obj, type):
            continue
        if not hasattr(obj, "search"):
            continue
        score = 0
        if hasattr(obj, "supported_categories"):
            score += 1
        if hasattr(obj, "name"):
            score += 1
        if hasattr(obj, "url"):
            score += 1
        candidates.append((score, obj))
    if not candidates:
        raise RuntimeError("No engine class with search() found")
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def pick_category(engine: Any, category: str) -> str:
    supported = getattr(engine, "supported_categories", None)
    if not isinstance(supported, dict):
        return category
    if category in supported:
        return category
    if "all" in supported:
        return "all"
    return next(iter(supported.keys()), category)


def parse_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return default


def format_bytes(size: int) -> str:
    if size < 0:
        return ""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(item or {})
    data["name"] = str(data.get("name") or "").strip()
    data["link"] = str(data.get("link") or "").strip()
    data["desc_link"] = str(data.get("desc_link") or "").strip()
    data["engine_name"] = str(data.get("engine_name") or "").strip()
    data["engine_url"] = str(data.get("engine_url") or "").strip()

    data["seeds"] = parse_int(data.get("seeds"))
    data["leech"] = parse_int(data.get("leech"))

    size_raw = data.get("size")
    size_bytes = novaprinter.anySizeToBytes(size_raw)
    data["size_bytes"] = size_bytes
    if size_bytes >= 0:
        data["size"] = format_bytes(size_bytes)
    else:
        data["size"] = str(size_raw or "").strip()
    return data


def dedupe_key(item: Dict[str, Any]) -> str:
    """Generate a key for detecting duplicate results across engines."""
    name = item.get("name", "").strip().lower()
    size_bytes = item.get("size_bytes", -1)
    # Two results are duplicates if they share the same name and size.
    return f"{name}|{size_bytes}"


class ResultCollector:
    def __init__(self, stop_event: Optional[threading.Event], out_queue: Optional[queue.Queue]) -> None:
        self._lock = threading.Lock()
        self.items: List[Dict[str, Any]] = []
        self._stop_event = stop_event
        self._queue = out_queue
        # Per-engine counters keyed by thread id, avoids racy len(items) diffs.
        self._engine_counts: Dict[int, int] = {}

    def add(self, item: Dict[str, Any]) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            return
        normalized = normalize_result(item)
        tid = threading.get_ident()
        with self._lock:
            self.items.append(normalized)
            self._engine_counts[tid] = self._engine_counts.get(tid, 0) + 1
        if self._queue is not None:
            self._queue.put(normalized)

    def add_normalized(self, item: Dict[str, Any]) -> None:
        """Add an already-normalized item (used when draining per-engine collectors)."""
        if self._stop_event is not None and self._stop_event.is_set():
            return
        with self._lock:
            self.items.append(item)
        if self._queue is not None:
            self._queue.put(item)

    def count_for_thread(self, tid: int) -> int:
        with self._lock:
            return self._engine_counts.get(tid, 0)


def run_engine(
    path: Path,
    query: str,
    category: str,
    collector: ResultCollector,
    stop_event: Optional[threading.Event],
    engine_timeout: int = 0,
) -> Dict[str, Any]:
    """Run a single engine and return stats about its execution."""
    t0 = time.monotonic()

    if stop_event is not None and stop_event.is_set():
        return {"name": path.stem, "status": "stopped", "results": 0, "elapsed": 0.0}

    module = load_module(path)
    engine_class = find_engine_class(module)
    engine = engine_class()

    engine_name = getattr(engine_class, "name", engine_class.__name__)
    engine_url = getattr(engine_class, "url", "")

    cat = pick_category(engine_class, category)

    if engine_timeout > 0:
        # Run search in a sub-thread with a timeout.
        # Use a per-engine collector so timed-out orphan threads don't
        # leak results into the shared collector after we've moved on.
        engine_queue = queue.Queue()
        engine_collector = ResultCollector(stop_event, engine_queue)
        result_exc = [None]

        def _search():
            novaprinter.set_thread_context(
                engine_name=engine_name,
                engine_url=engine_url,
                collector=engine_collector,
            )
            try:
                engine.search(query, cat)
            except Exception as e:
                result_exc[0] = e
            finally:
                novaprinter.clear_thread_context()

        t = threading.Thread(target=_search, daemon=True)
        t.start()
        t.join(timeout=engine_timeout)

        # Drain whatever the engine found so far into the shared collector.
        count = 0
        while True:
            try:
                item = engine_queue.get_nowait()
                collector.add_normalized(item)
                count += 1
            except queue.Empty:
                break

        if t.is_alive():
            elapsed = time.monotonic() - t0
            return {
                "name": engine_name,
                "status": "timeout",
                "results": count,
                "elapsed": elapsed,
            }
        if result_exc[0] is not None:
            raise result_exc[0]
        elapsed = time.monotonic() - t0
        return {
            "name": engine_name,
            "status": "ok",
            "results": count,
            "elapsed": elapsed,
        }
    else:
        novaprinter.set_thread_context(
            engine_name=engine_name,
            engine_url=engine_url,
            collector=collector,
        )
        caller_tid = threading.get_ident()
        try:
            engine.search(query, cat)
        finally:
            novaprinter.clear_thread_context()
        elapsed = time.monotonic() - t0
        return {
            "name": engine_name,
            "status": "ok",
            "results": collector.count_for_thread(caller_tid),
            "elapsed": elapsed,
        }


def run_search(
    query: str,
    category: str,
    selected_paths: List[Path],
    max_workers: int,
    stop_event: Optional[threading.Event],
    out_queue: Optional[queue.Queue],
    engine_stats_queue: Optional[queue.Queue] = None,
    engine_timeout: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    collector = ResultCollector(stop_event, out_queue)
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for path in selected_paths:
            if stop_event is not None and stop_event.is_set():
                break
            futures[executor.submit(
                run_engine, path, query, category, collector, stop_event, engine_timeout
            )] = path

        for future in as_completed(futures):
            path = futures[future]
            try:
                stats = future.result()
                if engine_stats_queue is not None:
                    engine_stats_queue.put(stats)
            except Exception as exc:
                errors[path.name] = str(exc)
                if engine_stats_queue is not None:
                    engine_stats_queue.put({
                        "name": path.stem,
                        "status": "error",
                        "results": 0,
                        "elapsed": 0.0,
                        "error": str(exc),
                    })
            if stop_event is not None and stop_event.is_set():
                executor.shutdown(cancel_futures=True)
                break

    if stop_event is not None and stop_event.is_set():
        errors["__stopped__"] = "Stopped by user"

    return collector.items, errors


_CSV_FORMULA_CHARS = {"=", "+", "-", "@", "\t", "\r"}


def _sanitize_csv_value(value: Any) -> Any:
    """Prefix values that start with formula-triggering characters to prevent CSV injection."""
    if isinstance(value, str) and value and value[0] in _CSV_FORMULA_CHARS:
        return "'" + value
    return value


def _sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


def results_to_csv(results: List[Dict[str, Any]]) -> str:
    """Convert results to a CSV string with formula-injection protection."""
    buf = io.StringIO()
    fields = ["name", "size", "seeds", "leech", "engine_name", "link", "desc_link"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        safe_row = {k: _sanitize_csv_value(v) for k, v in row.items()}
        writer.writerow(safe_row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page config & layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="P2P Aggregator", layout="wide")
st.title("P2P Aggregator")
st.caption(
    "Search across multiple sources in parallel. "
    "Some engines may require accounts or configuration."
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "search_state" not in st.session_state:
    st.session_state["search_state"] = {
        "running": False,
        "results": [],
        "errors": {},
        "thread": None,
        "stop_event": None,
        "last_query": "",
        "queue": None,
        "engine_stats_queue": None,
        "engine_stats": [],
    }

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []  # list of {"query", "category", "time", "count"}

state = st.session_state["search_state"]

# Clear finished background thread status on rerun.
thread = state.get("thread")
if thread is not None and not thread.is_alive():
    state["running"] = False
    state["thread"] = None
    state["stop_event"] = None


def drain_queue() -> None:
    q = state.get("queue")
    if q is None:
        return
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        state["results"].append(item)


def drain_engine_stats() -> None:
    q = state.get("engine_stats_queue")
    if q is None:
        return
    while True:
        try:
            stat = q.get_nowait()
        except queue.Empty:
            break
        state["engine_stats"].append(stat)


drain_queue()
drain_engine_stats()

plugins = list_plugins(PLUGIN_DIR)
labels = [p["label"] for p in plugins]
label_counts = Counter(labels)

display_labels: List[str] = []
label_to_path: Dict[str, str] = {}
for plugin in plugins:
    label = plugin["label"]
    if label_counts[label] > 1:
        label = f"{label} ({plugin['stem']})"
    display_labels.append(label)
    label_to_path[label] = plugin["path"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Engines")
    st.caption(f"{len(display_labels)} plugins available")

    # Quick select/deselect all - write directly to the widget's key
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        if st.button("Select all", use_container_width=True):
            st.session_state["plugin_multiselect"] = display_labels
    with sb_col2:
        if st.button("Deselect all", use_container_width=True):
            st.session_state["plugin_multiselect"] = []

    selected_labels = st.multiselect(
        "Plugins",
        options=display_labels,
        default=display_labels,
        key="plugin_multiselect",
    )

    st.divider()
    st.header("Search Settings")

    max_workers = st.slider(
        "Parallel engines",
        min_value=1,
        max_value=min(16, max(1, len(display_labels))),
        value=min(6, max(1, len(display_labels))),
    )

    engine_timeout = st.number_input(
        "Engine timeout (seconds, 0 = no limit)",
        min_value=0,
        value=30,
        step=5,
        help="Max time each engine gets before being cut off. Set to 0 to wait indefinitely.",
    )

    min_seeds = st.number_input("Min seeds", min_value=0, value=0, step=1)
    max_results = st.number_input("Max results", min_value=0, value=200, step=25)

    sort_by = st.selectbox(
        "Sort by",
        options=["Seeds (desc)", "Size (desc)", "Leechers (desc)", "Name (asc)"],
        index=0,
    )

    st.divider()
    st.header("Filters")

    name_filter = st.text_input(
        "Name filter",
        value="",
        help="Filter results by name. Supports regex (e.g. `1080p.*remux`). Case-insensitive.",
    )

    size_preset = st.selectbox(
        "Size range",
        options=list(SIZE_PRESETS.keys()),
        index=0,
    )

    hide_duplicates = st.checkbox(
        "Hide duplicates",
        value=False,
        help="Hide results that appear from multiple engines (keeps the one with most seeds).",
    )

    # --- Search History ---
    st.divider()
    st.header("History")
    history = st.session_state.get("search_history", [])
    if history:
        for i, entry in enumerate(reversed(history[-10:])):
            label_text = f"{entry['query']} [{entry['category']}] ({entry['count']} results)"
            if st.button(label_text, key=f"hist_{i}", use_container_width=True):
                st.session_state["_rerun_query"] = entry["query"]
                st.session_state["_rerun_category"] = entry["category"]
                st.rerun()
    else:
        st.caption("No searches yet.")

# ---------------------------------------------------------------------------
# Main area - search bar
# ---------------------------------------------------------------------------

# Handle history re-run
default_query = st.session_state.pop("_rerun_query", "")
default_category_name = st.session_state.pop("_rerun_category", None)

CATEGORIES = ["all", "movies", "tv", "music", "games", "anime", "software", "books"]
default_cat_index = 0
if default_category_name and default_category_name in CATEGORIES:
    default_cat_index = CATEGORIES.index(default_category_name)

query = st.text_input("Search query", value=default_query)
category = st.selectbox(
    "Category",
    options=CATEGORIES,
    index=default_cat_index,
)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    start = st.button("Search", type="primary", disabled=state["running"])
with col2:
    stop = st.button("Stop", disabled=not state["running"])
with col3:
    clear = st.button("Clear results")
with col4:
    # CSV export button (only when there are results)
    results_for_export = state.get("results", [])
    if results_for_export:
        csv_data = results_to_csv(results_for_export)
        st.download_button(
            "Export CSV",
            data=csv_data,
            file_name=f"p2p_{_sanitize_filename(state.get('last_query', 'results'))}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

if clear:
    state["results"] = []
    state["errors"] = {}
    state["engine_stats"] = []
    if not state["running"]:
        state["queue"] = None
        state["engine_stats_queue"] = None


def launch_search() -> None:
    stop_event = threading.Event()
    state["stop_event"] = stop_event
    state["running"] = True
    state["results"] = []
    state["errors"] = {}
    state["engine_stats"] = []
    state["last_query"] = query.strip()
    state["queue"] = queue.Queue()
    state["engine_stats_queue"] = queue.Queue()

    _query = query.strip()
    _category = category
    _paths = [Path(label_to_path[label]) for label in selected_labels]
    _max_workers = max_workers
    _timeout = engine_timeout

    def _worker():
        items, errors = run_search(
            query=_query,
            category=_category,
            selected_paths=_paths,
            max_workers=_max_workers,
            stop_event=stop_event,
            out_queue=state["queue"],
            engine_stats_queue=state["engine_stats_queue"],
            engine_timeout=_timeout,
        )
        state["errors"] = errors

        # Save to history. Use `items` (the collector's authoritative list)
        # rather than state["results"] which depends on queue draining.
        hist = st.session_state.get("search_history", [])
        hist.append({
            "query": _query,
            "category": _category,
            "time": datetime.now().isoformat(),
            "count": len(items),
        })
        # Keep history bounded
        if len(hist) > MAX_HISTORY:
            hist[:] = hist[-MAX_HISTORY:]

        state["running"] = False
        state["thread"] = None
        state["stop_event"] = None

    search_thread = threading.Thread(target=_worker, daemon=True)
    state["thread"] = search_thread
    search_thread.start()


if stop and state.get("stop_event") is not None:
    state["stop_event"].set()

if start:
    if state["running"]:
        st.info("Search already running.")
    elif not query.strip():
        st.warning("Enter a search query.")
    elif not selected_labels:
        st.warning("Select at least one plugin.")
    elif not plugins:
        st.warning(f"No plugins found in {PLUGIN_DIR}")
    else:
        launch_search()

if state["running"]:
    st.info("Searching... results update live. Click Stop to cancel.")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

results = state.get("results", [])
errors = state.get("errors", {})

if results:
    filtered = list(results)

    # --- Seed filter ---
    if min_seeds > 0:
        filtered = [r for r in filtered if parse_int(r.get("seeds"), 0) >= int(min_seeds)]

    # --- Name filter (regex supported, capped at 200 chars to limit ReDoS risk) ---
    if name_filter.strip():
        raw_filter = name_filter.strip()[:200]
        try:
            pattern = re.compile(raw_filter, re.IGNORECASE)
            filtered = [r for r in filtered if pattern.search(r.get("name", ""))]
        except re.error:
            # Fall back to plain substring match on bad regex.
            lower_filter = raw_filter.lower()
            filtered = [r for r in filtered if lower_filter in r.get("name", "").lower()]

    # --- Size range filter ---
    if size_preset != "Any":
        lo, hi = SIZE_PRESETS[size_preset]
        filtered = [
            r for r in filtered
            if (r.get("size_bytes", -1) >= 0
                and (lo == 0 or r["size_bytes"] >= lo)
                and (hi == 0 or r["size_bytes"] <= hi))
        ]

    # --- Duplicate handling ---
    if hide_duplicates:
        seen: Dict[str, Dict[str, Any]] = {}
        for r in filtered:
            key = dedupe_key(r)
            existing = seen.get(key)
            if existing is None or parse_int(r.get("seeds"), -1) > parse_int(existing.get("seeds"), -1):
                seen[key] = r
        filtered = list(seen.values())

    # --- Sorting ---
    if sort_by == "Seeds (desc)":
        filtered.sort(key=lambda r: parse_int(r.get("seeds"), -1), reverse=True)
    elif sort_by == "Size (desc)":
        filtered.sort(key=lambda r: parse_int(r.get("size_bytes"), -1), reverse=True)
    elif sort_by == "Leechers (desc)":
        filtered.sort(key=lambda r: parse_int(r.get("leech"), -1), reverse=True)
    else:
        filtered.sort(key=lambda r: r.get("name", ""))

    # --- Limit ---
    if max_results > 0:
        filtered = filtered[: int(max_results)]

    st.caption(f"Showing {len(filtered)} of {len(results)} results")
    st.dataframe(
        filtered,
        use_container_width=True,
        column_order=[
            "name",
            "size",
            "seeds",
            "leech",
            "engine_name",
            "link",
            "desc_link",
        ],
        column_config={
            "name": st.column_config.TextColumn("Name", width="large"),
            "size": st.column_config.TextColumn("Size"),
            "seeds": st.column_config.NumberColumn("Seeds"),
            "leech": st.column_config.NumberColumn("Leechers"),
            "engine_name": st.column_config.TextColumn("Engine"),
            "link": st.column_config.LinkColumn("Download/Magnet", width="small"),
            "desc_link": st.column_config.LinkColumn("Details", width="small"),
        },
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Engine stats panel
# ---------------------------------------------------------------------------

engine_stats = state.get("engine_stats", [])
if engine_stats:
    with st.expander(f"Engine stats ({len(engine_stats)} engines completed)"):
        ok = [s for s in engine_stats if s["status"] == "ok"]
        timed_out = [s for s in engine_stats if s["status"] == "timeout"]
        errored = [s for s in engine_stats if s["status"] == "error"]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Succeeded", len(ok))
        stat_col2.metric("Timed out", len(timed_out))
        stat_col3.metric("Errors", len(errored))
        stat_col4.metric("Total results", sum(s["results"] for s in engine_stats))

        # Per-engine breakdown sorted by result count
        sorted_stats = sorted(engine_stats, key=lambda s: s["results"], reverse=True)
        for s in sorted_stats:
            status_icon = {"ok": "[OK]", "timeout": "[TIMEOUT]", "error": "[ERR]"}.get(s["status"], "?")
            result_str = f"{s['results']} results" if s["results"] != 1 else "1 result"
            elapsed_str = f"{s['elapsed']:.1f}s" if s["elapsed"] > 0 else ""
            st.text(f"  {status_icon} {s['name']:30s} {result_str:>14s}  {elapsed_str:>6s}")

# ---------------------------------------------------------------------------
# Errors panel
# ---------------------------------------------------------------------------

if errors:
    display_errors = {k: v for k, v in errors.items() if k != "__stopped__"}
    if display_errors:
        with st.expander(f"Errors ({len(display_errors)})"):
            for name, err in sorted(display_errors.items()):
                st.text(f"  {name}: {err}")

if state["running"]:
    time.sleep(0.8)
    st.rerun()
