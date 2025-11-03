"""Extraction pipeline: regex-first with optional LLM fallback and memory fill."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from .config import OLLAMA_MODEL, ACE_REQUEST_TIMEOUT, USE_OPENAI
from .utils import basename_or_none
from rag_memory import add_memory, recall, memory_guess  # local RAG helpers


REQUIRED = ["execution_group", "bar_file"]
PATTERNS = [
    ("execution_group", [
        (r"(?i)\bdin\s+EG\s*[-–]?\s*([A-Za-z0-9._-]+)", 0),
        (r"(?i)\bEG\s*[-–]\s*([A-Za-z0-9._-]+)", 0),
        (r"(?i)\bin\s+eg\s+([A-Za-z0-9._-]+)", 0),
        (r"(?i)\bgrup(?:ul)?\s+de\s+executie\s*[:\-]\s*([^\r\n]+)", 0),
        (r"(?i)\bIntegration\s*Server\s*[:\-]\s*([^\r\n]+)", 0),
    ]),
    ("bar_file", [
        (r"(?i)\b([A-Za-z0-9._-]+\.bar)\b", 0),
    ]),
    ("cfg_name", [
        (r"\b([A-Za-z0-9._-]+\.cfg)\b", 0),
        (r"(?i)\bCFG\s*Name\s*[:\-]\s*([^\r\n]+)", 0),
    ]),
    ("application", [
        (r"(?i)\baplicat(?:ia|iei)\s+([A-Za-z0-9._-]+)", 0),
        (r"(?i)\bredeploy\s+la\s+aplicat(?:ia|iei)\s+([A-Za-z0-9._-]+)", 0),
        (r"\(([^()\r\n]+)\)\s*$", 0),
    ]),
    ("path", [
        (r"([A-Za-z]:\\[^\r\n]+)", 0),
        (r"(/[^ \r\n]+(?:/[^ \r\n]+)*)", 0),
    ]),
    ("environment", [
        (r"(?i)\bACE\s*TEST\b", 0),
    ]),
    ("status_note", [
        (r"(?i)\bram[aâ]ne\s+OPRIT[ĂA]!?", 0),
    ]),
]


def extract_rules(text: str, subject: str = "") -> dict:
    t = re.sub(r"[ \t]+\n", "\n", text or "")
    out: Dict[str, Any] = {}
    if re.search(r"(?i)\bACE\s*TEST\b", subject) or re.search(r"(?i)\bACE\s*TEST\b", t):
        out["environment"] = "ACE TEST"
    for field, variants in PATTERNS:
        for rx, flags in variants:
            m = re.search(rx, t, flags)
            if m:
                val = m.group(1).strip() if m.lastindex else m.group(0).strip()
                if field in {"execution_group", "path"}:
                    val = val.splitlines()[0].strip()
                out.setdefault(field, val)
                break
    if "application" in out:
        out["application"] = re.sub(r"\.(bar|cfg)$", "", out["application"], flags=re.I)
    return out


def needs_fallback(d: dict) -> bool:
    return any((not d.get(k)) for k in REQUIRED)


def llm_fallback(text: str) -> dict:
    hits = recall(query=f"Body:\n{text}", k=3)
    fewshot_parts: List[str] = []
    for h in hits:
        ex = h.get("extracted", {})
        fewshot_parts.append(
            "EXAMPLE\n"
            f"Subject: {h.get('subject','')}\n"
            f"Body:\n{(h.get('body','')[:600]).strip()}\n"
            "Extracted:\n"
            f"  execution_group={ex.get('execution_group','')}\n"
            f"  application={ex.get('application','')}\n"
            f"  bar_file={ex.get('bar_file','')}\n"
            f"  cfg_name={ex.get('cfg_name','')}\n"
            f"  environment={ex.get('environment','')}\n"
            f"  path={ex.get('path','')}\n"
            f"  status_note={ex.get('status_note','')}\n"
        )
    fewshot = "\n\n".join(fewshot_parts)

    schema_note = (
        "Return ONLY JSON with keys: execution_group, bar_file, cfg_name, "
        "application, environment, path, status_note. Required: execution_group, bar_file."
    )

    if USE_OPENAI:
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            system = "You extract structured deployment info from emails. Only output JSON."
            user = f"{fewshot}\n\nNEW EMAIL:\n{text}\n\n{schema_note}"
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            out = resp.output_text.strip()
            s, e = out.find("{"), out.rfind("}")
            if s != -1 and e != -1:
                return json.loads(out[s:e + 1])
            return {}
        except Exception:
            return {}
    else:
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL,
                      "prompt": f"{fewshot}\n\nNEW EMAIL:\n{text}\n\n{schema_note}",
                      "stream": False},
                timeout=ACE_REQUEST_TIMEOUT
            )
            if not r.ok:
                return {}
            t = r.json().get("response", "")
            s, e = t.find("{"), t.rfind("}")
            return json.loads(t[s:e + 1]) if s != -1 and e != -1 else {}
        except Exception:
            return {}


def extract_one(subject: str, body: str) -> dict:
    data = extract_rules(body, subject=subject)
    if needs_fallback(data):
        llm = llm_fallback(body)
        for k, v in llm.items():
            data.setdefault(k, v)
    return data


