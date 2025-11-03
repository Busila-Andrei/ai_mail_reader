"""Helpers that infer data from ACE: BAR↔(EG,App) lookups and corrections."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .ace_client import ACEClient
from .utils import bar_basename


def get_barfile_by_app(ace: ACEClient, app: str, eg: Optional[str] = None) -> Optional[str]:
    if not app:
        return None
    try:
        if eg:
            bar = ace.get_deploy_barfile(eg, app)
            return os.path.basename(bar) if bar else None
        for eg_name in ace.list_execution_groups():
            for a in ace.list_applications(eg_name):
                if a.strip().lower() == app.strip().lower():
                    bar = ace.get_deploy_barfile(eg_name, a)
                    return os.path.basename(bar) if bar else None
    except Exception:
        pass
    return None


def infer_eg_app_from_bar(ace: ACEClient, bar_file: str) -> Optional[Tuple[str, str]]:
    if not bar_file:
        return None
    try:
        target = (bar_basename(bar_file) or "").lower()
        if not target:
            return None
        for eg in ace.list_execution_groups():
            for app in ace.list_applications(eg):
                deployed = ace.get_deploy_barfile(eg, app) or ""
                base = (bar_basename(deployed) or "").lower()
                if base == target:
                    return eg, app
    except Exception:
        pass
    return None


def refine_from_ace_by_app(ace: ACEClient, app: str) -> Optional[Tuple[str, Optional[str]]]:
    if not app:
        return None
    try:
        app_lc = app.strip().lower()
        matches: List[Tuple[str, Optional[str]]] = []
        for eg in ace.list_execution_groups():
            for a in ace.list_applications(eg):
                if a.strip().lower() == app_lc:
                    bar = ace.get_deploy_barfile(eg, a)
                    matches.append((eg, os.path.basename(bar) if bar else None))
        if len(matches) == 1:
            return matches[0]
    except Exception:
        pass
    return None


