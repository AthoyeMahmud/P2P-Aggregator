from __future__ import annotations

import importlib.util
import queue
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class ResultCollector:
    def __init__(self, stop_event: Optional[threading.Event], out_queue: Optional[queue.Queue]) -> None:
        self._lock = threading.Lock()
        self.items: List[Dict[str, Any]] = []
        self._stop_event = stop_event
        self._queue = out_queue

    def add(self, item: Dict[str, Any]) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            return
        normalized = normalize_result(item)
        with self._lock:
            self.items.append(normalized)
        if self._queue is not None:
            self._queue.put(normalized)


def run_engine(
    path: Path,
    query: str,
    category: str,
    collector: ResultCollector,
    stop_event: Optional[threading.Event],
) -> str:
    if stop_event is not None and stop_event.is_set():
        return "stopped"

    module = load_module(path)
    engine_class = find_engine_class(module)
    engine = engine_class()

    engine_name = getattr(engine_class, "name", engine_class.__name__)
    engine_url = getattr(engine_class, "url", "")

    cat = pick_category(engine_class, category)
    novaprinter.set_thread_context(
        engine_name=engine_name,
        engine_url=engine_url,
        collector=collector,
    )
    try:
        engine.search(query, cat)
    finally:
        novaprinter.clear_thread_context()

    return str(engine_name)


def run_search(
    query: str,
    category: str,
    selected_paths: List[Path],
    max_workers: int,
    stop_event: Optional[threading.Event],
    out_queue: Optional[queue.Queue],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    collector = ResultCollector(stop_event, out_queue)
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for path in selected_paths:
            if stop_event is not None and stop_event.is_set():
                break
            futures[executor.submit(run_engine, path, query, category, collector, stop_event)] = path

        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors[path.name] = str(exc)
            if stop_event is not None and stop_event.is_set():
                executor.shutdown(cancel_futures=True)
                break

    if stop_event is not None and stop_event.is_set():
        errors["__stopped__"] = "Stopped by user"

    return collector.items, errors


st.set_page_config(page_title="qBit Plugins Search", layout="wide")
st.title("qBit Plugins Search Aggregator")
st.write(
    "Runs local qBittorrent search plugins and aggregates results in one place. "
    "Some engines may require accounts or configuration and will error if missing."
)

if "search_state" not in st.session_state:
    st.session_state["search_state"] = {
        "running": False,
        "results": [],
        "errors": {},
        "thread": None,
        "stop_event": None,
        "last_query": "",
        "queue": None,
    }
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


drain_queue()

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

with st.sidebar:
    st.header("Engines")
    st.caption(f"Plugin folder: {PLUGIN_DIR}")
    selected_labels = st.multiselect(
        "Plugins",
        options=display_labels,
        default=display_labels,
    )

    st.header("Search Settings")
    max_workers = st.slider(
        "Parallel engines",
        min_value=1,
        max_value=min(16, max(1, len(display_labels))),
        value=min(6, max(1, len(display_labels))),
    )
    min_seeds = st.number_input("Min seeds", min_value=0, value=0, step=1)
    max_results = st.number_input("Max results", min_value=0, value=200, step=25)
    sort_by = st.selectbox(
        "Sort by",
        options=["Seeds (desc)", "Size (desc)", "Name (asc)"],
        index=0,
    )

query = st.text_input("Search query", value="")
category = st.selectbox(
    "Category",
    options=["all", "movies", "tv", "music", "games", "anime", "software", "books"],
    index=0,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    start = st.button("Search", type="primary", disabled=state["running"])
with col2:
    stop = st.button("Stop", disabled=not state["running"])
with col3:
    clear = st.button("Clear results")

if clear:
    state["results"] = []
    state["errors"] = {}
    if not state["running"]:
        state["queue"] = None


def launch_search() -> None:
    stop_event = threading.Event()
    state["stop_event"] = stop_event
    state["running"] = True
    state["results"] = []
    state["errors"] = {}
    state["last_query"] = query.strip()
    state["queue"] = queue.Queue()

    def _worker():
        _, errors = run_search(
            query=query.strip(),
            category=category,
            selected_paths=[Path(label_to_path[label]) for label in selected_labels],
            max_workers=max_workers,
            stop_event=stop_event,
            out_queue=state["queue"],
        )
        state["errors"] = errors
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
    st.info("Search in progress... Click Stop to cancel queued engines. Results update as they arrive.")

results = state.get("results", [])
errors = state.get("errors", {})

if results:
    filtered = [
        r
        for r in results
        if parse_int(r.get("seeds"), 0) >= int(min_seeds)
    ]

    if sort_by == "Seeds (desc)":
        filtered.sort(key=lambda r: parse_int(r.get("seeds"), -1), reverse=True)
    elif sort_by == "Size (desc)":
        filtered.sort(key=lambda r: parse_int(r.get("size_bytes"), -1), reverse=True)
    else:
        filtered.sort(key=lambda r: r.get("name", ""))

    if max_results > 0:
        filtered = filtered[: int(max_results)]

    st.caption(f"Results: {len(filtered)} (raw: {len(results)})")
    st.dataframe(
        filtered,
        width="stretch",
        column_order=[
            "name",
            "size",
            "seeds",
            "leech",
            "engine_name",
            "engine_url",
            "link",
            "desc_link",
        ],
        column_config={
            "engine_name": st.column_config.TextColumn("Engine"),
            "engine_url": st.column_config.LinkColumn("Engine URL"),
            "link": st.column_config.LinkColumn("Download/Magnet"),
            "desc_link": st.column_config.LinkColumn("Details"),
        },
    )

if errors:
    with st.expander(f"Errors ({len(errors)})"):
        for name, err in sorted(errors.items()):
            st.write(f"{name}: {err}")

if state["running"]:
    time.sleep(0.8)
    st.rerun()
