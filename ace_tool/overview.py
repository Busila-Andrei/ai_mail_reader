"""Overview HTML builder and ingestion into local memory store."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

from .ace_client import ACEClient
from .utils import html_escape
from rag_memory import add_memory


def build_overview_table(ace: ACEClient, execution_groups: Optional[List[str]] = None,
                         cfg_files: Optional[List[str]] = None, link_base: str = "/cfg-editor") -> str:
    cfg_files = cfg_files or []
    cfg_map = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in cfg_files}
    if execution_groups is None:
        execution_groups = ace.list_execution_groups()

    out: List[str] = []
    out.append("<h3>Aplicatii ACE</h3>")
    out.append("<h4>Apasa pe numele cfg-ului pentru a modifica valorile</h4>")
    out.append("<table border='1' cellpadding='5' cellspacing='0'>")
    out.append("<tr><th>Execution Group</th><th>Application</th><th>State</th><th>isRunning</th><th>Deploy BAR</th><th>Fisier config</th></tr>")

    for eg in execution_groups:
        app_names = ace.list_applications(eg)
        for app in app_names:
            details = ace.get_application(eg, app)
            deployed_bar = details.get("descriptiveProperties", {}).get("deployBarfile") or "N/A"
            state = details.get("active", {}).get("state") or ""
            is_running = details.get("active", {}).get("isRunning")
            bar_base = os.path.splitext(deployed_bar)[0].lower() if deployed_bar != "N/A" else None

            if bar_base and bar_base in cfg_map:
                from urllib.parse import urlencode, quote
                cfg_full = cfg_map[bar_base]
                q = urlencode({"file": cfg_full, "eg": eg, "bar": deployed_bar, "app": app}, quote_via=quote)
                cfg_cell = f"<a href='{link_base}?{q}'>{html_escape(cfg_full)}</a>"
            else:
                cfg_cell = "N/A"

            out.append(
                "<tr>"
                f"<td>{html_escape(eg)}</td>"
                f"<td>{html_escape(app)}</td>"
                f"<td>{html_escape(str(state))}</td>"
                f"<td>{html_escape(str(is_running))}</td>"
                f"<td>{html_escape(deployed_bar)}</td>"
                f"<td>{cfg_cell}</td>"
                "</tr>"
            )
    out.append("</table>")
    return "\n".join(out)


def ingest_overview_html_to_memory(html_path: pathlib.Path) -> int:
    """
    Parse the overview HTML table (EG, App, Deploy BAR, CFG) and add rows into local memory.
    Returns the number of new rows added (dedup by EG|App|BAR|CFG).
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        print("[WARN] BeautifulSoup not available, skipping HTML ingestion.")
        return 0

    if not html_path.exists():
        print(f"[WARN] Overview HTML not found: {html_path}")
        return 0

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        print("[WARN] Overview HTML: no table found.")
        return 0

    # Load existing docs to avoid duplicates
    existing_keys = set()
    try:
        mem_dir = os.getenv("MEM_STORE_DIR", "memstore")
        docs_path = pathlib.Path(mem_dir) / "docs.jsonl"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    ex = (d.get("extracted") or {})
                    key = (
                        (ex.get("execution_group") or "").strip(),
                        (ex.get("application") or "").strip(),
                        (ex.get("bar_file") or "").strip(),
                        (ex.get("cfg_name") or "").strip(),
                    )
                    existing_keys.add(key)
    except Exception as e:
        print(f"[WARN] Could not read existing memory for dedup: {e}")

    added = 0
    rows = table.find_all("tr")
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        eg = (tds[0].get_text(strip=True) or "").strip()
        app = (tds[1].get_text(strip=True) or "").strip()
        bar = (tds[4].get_text(strip=True) or "").strip()
        cfg_cell = tds[5]
        cfg = None
        a = cfg_cell.find("a")
        if a and a.get_text(strip=True):
            cfg = a.get_text(strip=True)
        elif cfg_cell and cfg_cell.get_text(strip=True).upper() != "N/A":
            cfg = cfg_cell.get_text(strip=True)

        bar_file = os.path.basename(bar) if bar else ""
        cfg_name = os.path.basename(cfg) if cfg else ""

        key = (eg, app, bar_file, cfg_name or "")
        if key in existing_keys:
            continue

        doc = {
            "subject": f"OVERVIEW ROW · EG={eg} · APP={app}",
            "body": f"Row imported from overview HTML.\nDeploy BAR: {bar_file}\nCFG: {cfg_name}",
            "extracted": {
                "execution_group": eg,
                "application": app,
                "bar_file": bar_file,
                "cfg_name": cfg_name,
                "environment": "ACE TEST",
                "path": "",
                "status_note": ""
            }
        }
        try:
            add_memory(doc)
            existing_keys.add(key)
            added += 1
        except Exception as e:
            print(f"[WARN] Could not add overview row to memory: {e}")

    return added


