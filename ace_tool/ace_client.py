"""ACE REST client with simple retries, connection pooling, and small caches.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ACE_REQUEST_TIMEOUT
from .utils import basename_or_none, to_bool_or_none


class ACEClient:
    def __init__(self, host: str, port: int = 4414,
                 user: Optional[str] = None, password: Optional[str] = None,
                 verify_ssl: bool | str = False):
        self.base = f"https://{host}:{port}/apiv2"
        self.s = requests.Session()
        if user and password:
            self.s.auth = (user, password)
        self.s.verify = verify_ssl

        retries = Retry(total=3, backoff_factor=0.3,
                        status_forcelist=[502, 503, 504, 520, 522, 524],
                        allowed_methods=frozenset(["GET"]))
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=50)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)

        self._cache_egs: Optional[List[str]] = None
        self._cache_apps: Dict[str, List[str]] = {}
        self._cache_app_details: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _get_json(self, path: str):
        r = self.s.get(self.base + path, headers={"Accept": "application/json"},
                       timeout=ACE_REQUEST_TIMEOUT)
        if r.status_code >= 400:
            r.raise_for_status()
        if "json" not in (r.headers.get("Content-Type") or "").lower():
            return None
        try:
            return r.json()
        except Exception:
            return None

    def _get_xml(self, path: str):
        r = self.s.get(self.base + path, headers={"Accept": "application/xml"},
                       timeout=ACE_REQUEST_TIMEOUT)
        r.raise_for_status()
        return ET.fromstring(r.content)

    def _get_best(self, path: str):
        j = self._get_json(path)
        if j is not None:
            return ("json", j)
        return ("xml", self._get_xml(path))

    def list_execution_groups(self) -> List[str]:
        if self._cache_egs is not None:
            return self._cache_egs
        kind, data = self._get_best("/servers")
        if kind == "json":
            out = [c["name"] for c in data.get("children", []) if "name" in c]
        else:
            names: List[str] = []
            for el in data.iter():
                if el.tag.lower() in {"server", "integrationserver", "child"} and "name" in el.attrib:
                    names.append(el.attrib["name"])
            seen = set(); out = []
            for n in names:
                if n not in seen:
                    seen.add(n); out.append(n)
        self._cache_egs = out
        return out

    def list_applications(self, eg: str) -> List[str]:
        if eg in self._cache_apps:
            return self._cache_apps[eg]
        from urllib.parse import quote
        eg_q = quote(eg, safe="")
        kind, data = self._get_best(f"/servers/{eg_q}/applications")
        if kind == "json":
            children = data.get("children", []) or []
            names: List[str] = []
            for c in children:
                if c.get("type", "").lower() == "application" and "name" in c:
                    names.append(c["name"])
                elif "name" in c and "uri" in c:
                    names.append(c["name"])
        else:
            root: ET.Element = data
            names = []
            ch = root.find(".//children")
            if ch is not None:
                for app_el in ch.findall("./application"):
                    n = app_el.attrib.get("name")
                    if n:
                        names.append(n)
            if not names:
                for app_el in root.iter():
                    if app_el.tag.lower() == "application" and "name" in app_el.attrib:
                        names.append(app_el.attrib["name"])
        seen = set(); out: List[str] = []
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
        self._cache_apps[eg] = out
        return out

    def get_application(self, eg: str, app: str) -> Dict[str, Any]:
        key = (eg, app)
        if key in self._cache_app_details:
            return self._cache_app_details[key]
        from urllib.parse import quote
        eg_q = quote(eg, safe=""); app_q = quote(app, safe="")
        kind, data = self._get_best(f"/servers/{eg_q}/applications/{app_q}")
        if kind == "json":
            dp = (data.get("descriptiveProperties") or {})
            pr = (data.get("properties") or {})
            ac = (data.get("active") or {})
            out = {
                "name": data.get("name") or pr.get("name") or app,
                "descriptiveProperties": {
                    "deployBarfile": basename_or_none(dp.get("deployBarfile")),
                    "deployTimestamp": dp.get("deployTimestamp"),
                    "lastModified": dp.get("lastModified"),
                    "locationOnDisk": dp.get("locationOnDisk"),
                },
                "properties": {"startMode": pr.get("startMode"), "type": pr.get("type")},
                "active": {"isRunning": to_bool_or_none(ac.get("isRunning")),
                           "state": ac.get("state") or ac.get("resourceState")},
            }
        else:
            root: ET.Element = data
            out: Dict[str, Any] = {"name": app, "descriptiveProperties": {}, "properties": {}, "active": {}}
            if "name" in root.attrib:
                out["name"] = root.attrib["name"]
            dp = root.find(".//descriptiveProperties")
            if dp is not None:
                out["descriptiveProperties"] = {
                    "deployBarfile": basename_or_none(dp.attrib.get("deployBarfile")),
                    "deployTimestamp": dp.attrib.get("deployTimestamp"),
                    "lastModified": dp.attrib.get("lastModified"),
                    "locationOnDisk": dp.attrib.get("locationOnDisk"),
                }
            pr = root.find(".//properties")
            if pr is not None:
                out["properties"] = {
                    "startMode": pr.attrib.get("startMode"),
                    "type": pr.attrib.get("type"),
                    "name": pr.attrib.get("name"),
                }
            ac = root.find(".//active")
            if ac is not None:
                out["active"] = {
                    "isRunning": to_bool_or_none(ac.attrib.get("isRunning")),
                    "state": ac.attrib.get("state") or ac.get("resourceState"),
                }
        self._cache_app_details[key] = out
        return out

    def get_deploy_barfile(self, eg: str, app: str) -> Optional[str]:
        try:
            details = self.get_application(eg, app)
            bar = (details.get("descriptiveProperties") or {}).get("deployBarfile")
            return bar or None
        except Exception:
            return None


