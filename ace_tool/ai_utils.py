"""AI helpers: JSON completion and ACE candidate-aware resolution."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

from .config import AI_PROVIDER, OPENAI_MODEL, AI_MODEL, AI_TIMEOUT


def _ai_complete_json(prompt: str) -> Optional[dict]:
    try:
        if AI_PROVIDER == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role": "system", "content": "Return ONLY JSON."},
                       {"role": "user", "content": prompt}],
            )
            text = resp.output_text
        else:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": AI_MODEL, "prompt": prompt, "stream": False},
                timeout=AI_TIMEOUT
            )
            if not r.ok:
                return None
            text = r.json().get("response", "")
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def ace_candidate_sets(ace, eg_hint: Optional[str] = None) -> Dict[str, List[str]]:
    egs = ace.list_execution_groups()
    apps: List[str] = []
    bars: List[str] = []
    scan_egs = [eg_hint] if eg_hint and eg_hint in egs else egs
    for eg in scan_egs:
        for app in ace.list_applications(eg):
            apps.append(app)
            bar = ace.get_deploy_barfile(eg, app)
            if bar:
                base = os.path.basename(bar)
                if base:
                    bars.append(base)
    return {"egs": sorted(set(egs)), "apps": sorted(set(apps)), "bars": sorted(set(bars))}


def ai_resolve_eg_app_bar(email_subject: str,
                          email_body: str,
                          suggested: Dict[str, str],
                          ace_candidates: Dict[str, List[str]]) -> Optional[dict]:
    schema = """
Return ONLY JSON with:
{
  "execution_group": string | null,
  "application": string | null,
  "bar_file": string | null,
  "confidence": number
}
Rules:
- Prefer values that appear in ace_candidates.
- If 'bar_file' not present in ace_candidates, leave it null (do NOT invent).
- Use email content to decide the most plausible pair (EG, application).
- If uncertain, leave fields null and set confidence low.
"""
    prompt = f"""
Email Subject:
{email_subject}

Email Body:
{email_body[:4000]}

Current Suggested (may be incomplete or noisy):
{json.dumps(suggested, ensure_ascii=False)}

ACE Candidates (authoritative choices):
{json.dumps(ace_candidates, ensure_ascii=False)}

{schema}
"""
    return _ai_complete_json(prompt)


