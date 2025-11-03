"""Validation of extracted fields against ACE and optional CFG inventory."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .ace_client import ACEClient
from .utils import bar_basename, basename_or_none


def infer_app_from_bar(bar_file: str) -> Optional[str]:
    if not bar_file:
        return None
    base = os.path.splitext(basename_or_none(bar_file) or "")[0]
    return base or None


def validate_against_ace(ace: ACEClient, extracted: Dict[str, Any], cfg_files: Optional[List[str]] = None) -> Dict[str, Any]:
    ret = {"ok": True, "checks": []}

    def add(ok: bool, msg: str, **extra):
        ret["checks"].append({"ok": ok, "msg": msg, **extra})
        if not ok:
            ret["ok"] = False

    eg = extracted.get("execution_group") or ""
    bar = basename_or_none(extracted.get("bar_file")) or ""
    app = extracted.get("application") or ""
    want_stopped = bool(extracted.get("status_note") and "oprit" in extracted["status_note"].lower())

    egs = set(ace.list_execution_groups())
    add(eg in egs, f"EG '{eg}' " + ("exists" if eg in egs else "NOT found"), kind="eg", eg=eg)

    if eg in egs:
        app_to_check = app or infer_app_from_bar(bar) or ""
        apps = set(ace.list_applications(eg))
        if app_to_check:
            add(app_to_check in apps, f"Application '{app_to_check}' in EG '{eg}' " + ("exists" if app_to_check in apps else "NOT found"),
                kind="app", app=app_to_check, eg=eg)
        else:
            add(False, "No application provided and cannot infer from BAR.", kind="app", eg=eg)

        if app_to_check in apps:
            details = ace.get_application(eg, app_to_check)
            deployed_bar = details.get("descriptiveProperties", {}).get("deployBarfile") or "N/A"
            is_running = details.get("active", {}).get("isRunning")
            state = details.get("active", {}).get("state")
            start_mode = (details.get("properties", {}) or {}).get("startMode")

            if bar:
                add((deployed_bar or "").lower() == bar.lower(),
                    f"Deployed BAR '{deployed_bar}' vs email '{bar}'",
                    kind="bar", deployed=deployed_bar, email=bar)
            else:
                add(False, "Email had no BAR to compare.", kind="bar")

            if want_stopped:
                add(is_running in (False, None),
                    f"Email says keep STOPPED; app is_running={is_running}, state={state}",
                    kind="runtime", expected="stopped", actual=is_running)
                if start_mode and str(start_mode).lower() in {"maintained", "automatic"}:
                    add(False, f"startMode is '{start_mode}' (may auto-start). Consider 'manual'.",
                        kind="startMode", startMode=start_mode)
                else:
                    add(True, f"startMode: {start_mode or 'N/A'}", kind="startMode")
        else:
            add(False, "Cannot verify BAR/runtime (missing or unknown application).", kind="details")

    if cfg_files:
        m = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in cfg_files}
        key = os.path.splitext(bar or "")[0].lower()
        cfg_hit = m.get(key)
        add(bool(cfg_hit), f"CFG matching BAR '{bar}': " + (cfg_hit or "NOT found"), kind="cfg", cfg=cfg_hit or "")
    else:
        add(True, "CFG check skipped (no CFG list supplied).", kind="cfg")

    return ret


