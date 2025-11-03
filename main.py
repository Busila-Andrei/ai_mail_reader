# -*- coding: utf-8 -*-


from __future__ import annotations
import re, argparse, pathlib  # Removed unused: os, json, datetime as dt
from typing import List  # Removed unused: Dict

# import pandas as pd  # Commented out - not needed for email reading only

from ace_tool.config import (
    # ACE_HOST, ACE_PORT, ACE_USER, ACE_PASS, VERIFY_SSL,  # Commented out - not needed for email reading
    PROCESS_ONLY_UNREAD, MARK_AS_READ,
    # SSH_HOST, SSH_USER, SSH_CFG_PATH, SSH_PORT, SSH_KEY_PATH, SSH_KEY_PASS, SSH_KNOWN,  # Commented out - CFG discovery
    # CFG_ROOT, SKIP_ACE, USE_AI_RESOLVER,  # Commented out - not needed for email reading
    ACTION_PATTERNS, DEBUG_EMAIL_SCAN  # Keep - used by email reader
)
# from ace_tool.ace_client import ACEClient  # Commented out - not needed for email reading
# from ace_tool.extraction import extract_one  # Commented out - not needed for email reading
# from ace_tool.cfg_discovery import discover_cfg_files, discover_cfg_files_remote_key  # Commented out - not needed for email reading
# from ace_tool.validator import validate_against_ace  # Commented out - not needed for email reading
# from ace_tool.ai_utils import ace_candidate_sets, ai_resolve_eg_app_bar  # Commented out - not needed for email reading
from ace_tool.outlook_reader import read_outlook_ace_emails
# from ace_tool.ace_helpers import get_barfile_by_app, infer_eg_app_from_bar, refine_from_ace_by_app  # Commented out - not needed for email reading
# from rag_memory import add_memory, memory_guess  # Commented out - not needed for email reading

ACTION_REGEXES: List[re.Pattern] = []
for pat in ACTION_PATTERNS.split(","):
    pat = pat.strip()
    if not pat:
        continue
    try:
        ACTION_REGEXES.append(re.compile(pat, re.IGNORECASE))
    except re.error as e:
        print(f"[WARN] Invalid regex pattern {pat!r}: {e}")

# Maintain compiled action regexes (for optional custom filters in main loop)

# (extraction moved to ace_tool.extraction)

# (ACE helpers moved to ace_tool.ace_helpers)

# (validation moved to ace_tool.validator)

# (cfg discovery moved to ace_tool.cfg_discovery)

# (AI helpers moved to ace_tool.ai_utils)

# (Outlook reader moved to ace_tool.outlook_reader)

# from ace_tool.overview import build_overview_table, ingest_overview_html_to_memory  # Commented out - not needed for email reading


# (overview helpers moved to ace_tool.overview)


# ---------------------------------
# Main
# ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Read 'ACE TEST' emails from Outlook inbox.")
    ap.add_argument("--since", default="300d", help="e.g. 30d, 7d, 48h")
    ap.add_argument("--limit", type=int, default=200, help="max messages to scan")
    ap.add_argument("--outdir", default="out", help="output directory")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    PROCESSED_DB = outdir / "processed_ids.txt"

    # Load processed EntryIDs to track which emails have been read
    processed_ids: set[str] = set()
    if PROCESSED_DB.exists():
        with open(PROCESSED_DB, "r", encoding="utf-8") as f:
            processed_ids = {line.strip() for line in f if line.strip()}
        print(f"[INFO] Loaded {len(processed_ids)} processed EntryID(s).")

    # Commented out - ACE client initialization (not needed for email reading only)
    # ace = ACEClient(ACE_HOST, ACE_PORT, ACE_USER, ACE_PASS, VERIFY_SSL)

    # Commented out - CFG discovery (not needed for email reading only)
    # cfg_files: List[str] = []
    # if SSH_HOST and SSH_USER and SSH_CFG_PATH:
    #     try:
    #         from paramiko import SSHClient  # noqa
    #         cfg_files = discover_cfg_files_remote_key(
    #             host=SSH_HOST, user=SSH_USER, remote_path=SSH_CFG_PATH,
    #             port=SSH_PORT, key_path=SSH_KEY_PATH, passphrase=SSH_KEY_PASS, known_hosts=SSH_KNOWN
    #         )
    #         print(f"[INFO] Discovered {len(cfg_files)} *.cfg via SSH.")
    #     except ImportError:
    #         print("[WARN] Paramiko not installed; falling back to local CFG discovery.")
    # if not cfg_files and CFG_ROOT:
    #     cfg_files = discover_cfg_files(CFG_ROOT)
    #     if cfg_files:
    #         print(f"[INFO] Discovered {len(cfg_files)} *.cfg locally.")

    # Read emails
    mails = read_outlook_ace_emails(
        limit=args.limit,
        since=args.since,
        processed_ids=processed_ids,
        process_only_unread=PROCESS_ONLY_UNREAD,
        mark_as_read=MARK_AS_READ,
        action_regexes=ACTION_REGEXES
    )

    # Commented out - ACE cache warming (not needed for email reading only)
    # if not SKIP_ACE:
    #     eg_list = ace.list_execution_groups()
    #     _ = {eg: ace.list_applications(eg) for eg in eg_list}
    #     print(f"[INFO] Prefetched EGs/apps: {len(eg_list)} EG(s).")

    # Commented out - All processing logic (extraction, validation, AI, etc.)
    # rows = []
    # for idx, m in enumerate(mails, 1):
    #     if idx % 10 == 0:
    #         print(f"[INFO] Validating email {idx}/{len(mails)} ...")
    #
    #     field_src: Dict[str,str] = {}
    #
    #     # 1) rules + LLM fallback
    #     extracted = extract_one(m["subject"], m["body"])
    #     for k in ["execution_group", "application", "bar_file", "cfg_name"]:
    #         if extracted.get(k):
    #             field_src[k] = "rules/llm"
    #
    #     # 2) conservative memory fill (don't invent BAR unless .bar appears)
    #     email_mentions_bar = bool(re.search(r"\.bar\b", m["body"], re.IGNORECASE)) or bool(re.search(r"\.bar\b", m["subject"], re.IGNORECASE))
    #     try:
    #         fill = memory_guess(extracted)
    #         for k, v in fill.items():
    #             if not extracted.get(k):
    #                 if k == "bar_file" and not email_mentions_bar and not SKIP_ACE:
    #                     continue  # let ACE decide
    #                 extracted[k] = v
    #                 field_src[k] = "memory"
    #         if DEBUG_EMAIL_SCAN and fill:
    #             print(f"[DEBUG] memory filled: {fill}")
    #     except Exception as e:
    #         if DEBUG_EMAIL_SCAN:
    #             print(f"[DEBUG] memory_guess error: {e}")
    #
    #     # 3) ACE-authoritative inference/corrections
    #     if not SKIP_ACE:
    #         # From APP → EG & BAR
    #         if extracted.get("application"):
    #             res = refine_from_ace_by_app(ace, extracted["application"])
    #             if res:
    #                 eg_hit, bar_hit = res
    #                 if eg_hit and (not extracted.get("execution_group") or extracted["execution_group"] != eg_hit):
    #                     extracted["execution_group"] = eg_hit
    #                     field_src["execution_group"] = "ace_by_app"
    #                 if bar_hit and not extracted.get("bar_file"):
    #                     extracted["bar_file"] = bar_hit
    #                     field_src["bar_file"] = "ace_by_app"
    #
    #         # From BAR → EG & APP
    #         if extracted.get("bar_file"):
    #             hit = infer_eg_app_from_bar(ace, extracted["bar_file"])
    #             if hit:
    #                 eg2, app2 = hit
    #                 if eg2 and (not extracted.get("execution_group") or extracted["execution_group"] != eg2):
    #                     extracted["execution_group"] = eg2
    #                     field_src["execution_group"] = "ace_by_bar"
    #                 if app2 and (not extracted.get("application") or extracted["application"].strip().lower() != app2.strip().lower()):
    #                     extracted["application"] = app2
    #                     field_src["application"] = "ace_by_bar"
    #
    #         # Prefer ACE BAR if APP+EG known
    #         if extracted.get("application") and extracted.get("execution_group"):
    #             bar_from_ace = get_barfile_by_app(ace, extracted["application"], extracted["execution_group"])
    #             if bar_from_ace:
    #                 extracted["bar_file"] = bar_from_ace
    #                 field_src["bar_file"] = field_src.get("bar_file","ace_by_app_eg")
    #
    #     # 4) AI resolver (optional)
    #     if USE_AI_RESOLVER and not SKIP_ACE:
    #         cand = ace_candidate_sets(ace, eg_hint=extracted.get("execution_group"))
    #         suggested = {
    #             "execution_group": extracted.get("execution_group"),
    #             "application": extracted.get("application"),
    #             "bar_file": extracted.get("bar_file"),
    #         }
    #         need_ai = (
    #             not suggested.get("execution_group") or
    #             not suggested.get("application") or
    #             (suggested.get("execution_group") and suggested["execution_group"] not in cand["egs"]) or
    #             (suggested.get("application") and suggested["application"] not in cand["apps"]) or
    #             (suggested.get("bar_file") and suggested["bar_file"] not in cand["bars"])
    #         )
    #         if need_ai:
    #             ai_choice = ai_resolve_eg_app_bar(m["subject"], m["body"], suggested, cand) or {}
    #             conf = float(ai_choice.get("confidence") or 0)
    #             if conf >= 0.55:
    #                 eg_ai  = ai_choice.get("execution_group")
    #                 app_ai = ai_choice.get("application")
    #                 bar_ai = ai_choice.get("bar_file")
    #                 if eg_ai and eg_ai in cand["egs"]:
    #                     extracted["execution_group"] = eg_ai
    #                     field_src["execution_group"] = "ai_resolver"
    #                 if app_ai and app_ai in cand["apps"]:
    #                     extracted["application"] = app_ai
    #                     field_src["application"] = "ai_resolver"
    #                 if bar_ai and bar_ai in cand["bars"]:
    #                     extracted["bar_file"] = bar_ai
    #                     field_src["bar_file"] = "ai_resolver"
    #
    #     # 5) Save to memory (so future emails benefit)
    #     try:
    #         add_memory({"subject": m["subject"], "body": m["body"], "extracted": extracted})
    #     except Exception as e:
    #         print(f"[WARN] Could not add to memory: {e}")
    #
    #     # 6) Validate against ACE (or skip)
    #     if SKIP_ACE:
    #         validation = {"ok": True, "checks": [{"ok": True, "msg": "ACE validation skipped"}]}
    #     else:
    #         validation = validate_against_ace(ace, extracted, cfg_files)
    #
    #     rows.append({
    #         "received": m["received"],
    #         "subject": m["subject"],
    #         "execution_group": extracted.get("execution_group",""),
    #         "application": extracted.get("application",""),
    #         "bar_file": extracted.get("bar_file",""),
    #         "cfg_name": extracted.get("cfg_name",""),
    #         "path": extracted.get("path",""),
    #         "status_note": extracted.get("status_note",""),
    #         "ok": validation["ok"],
    #         "checks": json.dumps(validation["checks"], ensure_ascii=False),
    #         "source_execution_group": field_src.get("execution_group", ""),
    #         "source_application": field_src.get("application", ""),
    #         "source_bar_file": field_src.get("bar_file", ""),
    #         "source_cfg_name": field_src.get("cfg_name", ""),
    #         "entry_id": m.get("entry_id",""),
    #     })
    #
    # if not rows:
    #     print("[WARN] No rows to write.")
    #     return
    #
    # ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_path  = outdir / f"ace_mail_validation_{ts}.csv"
    # json_path = outdir / f"ace_mail_validation_{ts}.json"
    # pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(rows, f, ensure_ascii=False, indent=2)
    # print(f"[OK] wrote {csv_path}")
    # print(f"[OK] wrote {json_path}")
    #
    # # Append processed EntryIDs
    # with open(PROCESSED_DB, "a", encoding="utf-8") as f:
    #     new_ids = 0
    #     for r in rows:
    #         eid = r.get("entry_id")
    #         if eid and eid not in processed_ids:
    #             f.write(eid + "\n")
    #             new_ids += 1
    #     print(f"[OK] appended {new_ids} new EntryID(s) to {PROCESSED_DB.name}")

    # Commented out - Overview HTML generation (not needed for email reading only)
    # html_table = "<p>(ACE validation skipped)</p>" if SKIP_ACE else build_overview_table(ace, cfg_files=cfg_files, link_base="/cfg-editor")
    # html_path = outdir / f"ace_applications_overview_{ts}.html"
    # with open(html_path, "w", encoding="utf-8") as f:
    #     f.write("<!doctype html><html><head><meta charset='utf-8'>"
    #             "<title>Aplicatii ACE</title>"
    #             "<style>body{font-family:Segoe UI,Arial,sans-serif}table{border-collapse:collapse}"
    #             "th,td{border:1px solid #ddd;padding:6px}th{background:#f3f3f3;text-align:left}</style>"
    #             "</head><body>")
    #     f.write(html_table)
    #     f.write("</body></html>")
    # print(f"[OK] wrote {html_path}")
    #
    # # Ingest the just-generated HTML into memory so AI can learn EG/App/BAR/CFG mapping
    # try:
    #     added_rows = ingest_overview_html_to_memory(html_path)
    #     print(f"[OK] learned {added_rows} overview row(s) into memory.")
    # except Exception as e:
    #     print(f"[WARN] Overview HTML ingestion failed: {e}")

    # Display results - simple email reading output
    if not mails:
        print("[INFO] No emails found.")
        return

    print(f"\n[INFO] Found {len(mails)} email(s):")
    for idx, m in enumerate(mails, 1):
        print(f"\n--- Email {idx} ---")
        print(f"Subject: {m.get('subject', 'N/A')}")
        print(f"Received: {m.get('received', 'N/A')}")
        print(f"Entry ID: {m.get('entry_id', 'N/A')}")
        print(f"Body preview: {m.get('body', '')[:200]}...")

    # Save EntryIDs of read emails
    with open(PROCESSED_DB, "a", encoding="utf-8") as f:
        new_ids = 0
        for m in mails:
            eid = m.get("entry_id")
            if eid and eid not in processed_ids:
                f.write(eid + "\n")
                new_ids += 1
        if new_ids > 0:
            print(f"\n[OK] Recorded {new_ids} new EntryID(s) to {PROCESSED_DB.name}")



if __name__ == "__main__":
    main()
