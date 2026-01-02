# Minimal helper utilities for qBittorrent-style search plugins.
# These are intentionally lightweight for Streamlit aggregation.

from __future__ import annotations

import gzip
import html
import ssl
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}

# Some plugins import this symbol directly.
headers: Dict[str, str] = dict(DEFAULT_HEADERS)

_SSL_CONTEXT = ssl._create_unverified_context()
_PATCHED_URLLIB = False
_ORIGINAL_REQUEST_INIT = None


def _merge_headers(extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(DEFAULT_HEADERS)
    if extra:
        try:
            merged.update(extra)
        except Exception:
            pass
    return merged


def retrieve_url(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Any = None,
    method: Optional[str] = None,
    timeout: int = 20,
) -> str:
    url = sanitize_url(url)
    req_headers = _merge_headers(headers)

    if data is not None and isinstance(data, (dict, list, tuple)):
        data = urllib.parse.urlencode(data).encode("utf-8")
    elif data is not None and isinstance(data, str):
        data = data.encode("utf-8")

    if method is None:
        method = "POST" if data else "GET"

    request = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def download_file(
    url: str,
    referer: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> str:
    # qBittorrent plugins expect to print a link or file payload; for aggregation
    # purposes, returning the URL is sufficient and avoids saving files.
    _ = (referer, headers, timeout)
    return url


def htmlentitydecode(text: str) -> str:
    return html.unescape(text or "")


def sanitize_url(url: str) -> str:
    if not url:
        return url
    if not any(ch == " " or ord(ch) < 32 for ch in url):
        return url
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        return url.replace(" ", "%20")
    path = urllib.parse.quote(parts.path, safe="/%+")
    query = urllib.parse.quote(parts.query, safe="=&%+")
    fragment = urllib.parse.quote(parts.fragment, safe="%+")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def install_urllib_patches() -> None:
    global _PATCHED_URLLIB, _ORIGINAL_REQUEST_INIT
    if _PATCHED_URLLIB:
        return
    _ORIGINAL_REQUEST_INIT = urllib.request.Request.__init__

    def _patched_request_init(self, url, *args, **kwargs):
        if isinstance(url, str):
            url = sanitize_url(url)
        return _ORIGINAL_REQUEST_INIT(self, url, *args, **kwargs)

    urllib.request.Request.__init__ = _patched_request_init
    ssl._create_default_https_context = ssl._create_unverified_context
    _PATCHED_URLLIB = True


install_urllib_patches()
