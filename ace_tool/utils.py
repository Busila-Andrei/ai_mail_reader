"""General utilities used across ace_tool modules.

Includes HTML/text helpers, path helpers, boolean parsing, HTML escaping,
and human-friendly time window parsing.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from bs4 import BeautifulSoup
from typing import Any, Optional


def html_to_text(s: str) -> str:
    if not s:
        return ""
    soup = BeautifulSoup(s, "lxml")
    for x in soup(["script", "style"]):
        x.decompose()
    t = soup.get_text("\n")
    return re.sub(r"[ \t]+\n", "\n", t).strip()


def basename_or_none(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path.replace("\\", "/").rstrip("/").split("/")[-1] or None


def to_bool_or_none(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def html_escape(s: str) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def parse_since(s: str) -> dt.timedelta:
    m = re.match(r"(?i)^\s*(\d+)\s*([dhw])\s*$", s or "")
    if not m:
        return dt.timedelta(days=30)
    n = int(m.group(1))
    u = m.group(2).lower()
    if u == "h":
        return dt.timedelta(hours=n)
    if u == "w":
        return dt.timedelta(weeks=n)
    return dt.timedelta(days=n)


def bar_basename(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.splitext(os.path.basename(path))[0]


