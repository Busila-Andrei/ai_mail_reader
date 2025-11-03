"""Outlook inbox reader with filters for ACE-related deployment emails."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import win32com.client as win32

from .config import PROCESS_ONLY_SINGLE_CONVERSATION, DEBUG_EMAIL_SCAN
from .utils import html_to_text, parse_since


def _is_single_message_mail(item, folder=None, outlook=None, debug=False):
    try:
        subject = (getattr(item, "Subject", "") or "").strip()
        subj_u = subject.upper()
        if subj_u.startswith(("RE:", "FW:", "FWD:")):
            if debug:
                print(f"[DEBUG] single-check: RE/FW subject → not single: {subject!r}")
            return False
    except Exception:
        subject = ""
    try:
        cidx = getattr(item, "ConversationIndex", "") or ""
        cidx_len = len(str(cidx))
        if cidx_len == 44:
            if debug:
                print(f"[DEBUG] single-check: ConversationIndex length=44 → single: {subject!r}")
            return True
        if cidx_len > 44:
            if debug:
                print(f"[DEBUG] single-check: ConversationIndex length={cidx_len} (>44) → not single: {subject!r}")
            return False
    except Exception as e:
        if debug:
            print(f"[DEBUG] single-check: ConversationIndex unavailable: {e}")
    try:
        conv = item.GetConversation()
        if conv is not None and outlook is not None and folder is not None:
            try:
                table = conv.GetTable()
                cnt_same_folder = 0
                row = table.GetNextRow()
                while row:
                    try:
                        conv_item = outlook.GetItemFromID(row["EntryID"])
                        if getattr(conv_item, "Parent", None) and conv_item.Parent.FolderPath == folder.FolderPath:
                            cnt_same_folder += 1
                    except Exception:
                        pass
                    row = table.GetNextRow()
                if debug:
                    print(f"[DEBUG] single-check: conversation items in folder = {cnt_same_folder} for {subject!r}")
                return cnt_same_folder == 1
            except Exception as e:
                if debug:
                    print(f"[DEBUG] single-check: conv table error: {e}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] single-check: GetConversation error: {e}")
    return False


def read_outlook_ace_emails(limit: int = 200, since: str = "30d",
                            processed_ids: Optional[set[str]] = None,
                            process_only_unread: bool = True,
                            mark_as_read: bool = True,
                            action_regexes: Optional[List] = None) -> List[Dict[str, str]]:
    import pywintypes
    processed_ids = processed_ids or set()

    outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # Inbox
    items = inbox.Items
    items.Sort("[ReceivedTime]", True)

    if process_only_unread:
        try:
            items = items.Restrict("[Unread] = True")
        except Exception as e:
            print(f"[WARN] Unread Restrict failed: {e}")

    start_dt = dt.datetime.now() - parse_since(since)
    start_str = start_dt.strftime("%m/%d/%Y %I:%M %p")

    subject_filter = "@SQL=\"urn:schemas:httpmail:subject\" like '%ACE TEST%'"
    date_filter = f"[ReceivedTime] >= '{start_str}'"

    filtered = items
    try:
        filtered = filtered.Restrict(date_filter)
    except Exception as e:
        print(f"[WARN] Date Restrict failed ({e}); continuing without date filter.")
    try:
        filtered = filtered.Restrict(subject_filter)
    except Exception as e:
        print(f"[WARN] Subject Restrict failed ({e}); will filter in Python.")

    results: List[Dict[str, str]] = []
    count = 0

    for item in filtered:
        try:
            entry_id = getattr(item, "EntryID", None) or ""
            if entry_id and entry_id in processed_ids:
                continue

            subject = (getattr(item, "Subject", "") or "").strip()
            if "ACE TEST" not in subject.upper():
                continue

            recv = getattr(item, "ReceivedTime", None)
            if isinstance(recv, pywintypes.TimeType):
                received_dt = dt.datetime(recv.year, recv.month, recv.day, recv.hour, recv.minute, recv.second)
            else:
                received_dt = None
            if received_dt and received_dt < start_dt:
                continue

            body_text = (getattr(item, "Body", "") or "").strip()
            if not body_text:
                body_text = html_to_text(getattr(item, "HTMLBody", "") or "")
            if not body_text and not subject:
                continue

            # Action word regex filter (if provided)
            if action_regexes:
                text_to_search = f"{subject}\n{body_text}"
                try:
                    if not any(r.search(text_to_search) for r in action_regexes):
                        if DEBUG_EMAIL_SCAN:
                            print(f"[DEBUG] skip mail: no deploy-action word in {subject!r}")
                        continue
                except Exception:
                    pass

            if PROCESS_ONLY_SINGLE_CONVERSATION:
                if not _is_single_message_mail(item, folder=inbox, outlook=outlook, debug=DEBUG_EMAIL_SCAN):
                    if DEBUG_EMAIL_SCAN:
                        print(f"[DEBUG] skip non-single mail: {subject!r}")
                    continue

            results.append({
                "subject": subject,
                "body": body_text,
                "received": received_dt.isoformat(sep=" ") if received_dt else str(getattr(item, "ReceivedTime", "")),
                "entry_id": entry_id,
            })

            if mark_as_read and getattr(item, "UnRead", False):
                try:
                    item.UnRead = False
                    item.Save()
                except Exception as e:
                    print(f"[WARN] Could not mark item as read: {e}")

            count += 1
            if count % 10 == 0:
                print(f"[INFO] Collected {count} email(s) so far...")
            if count >= limit:
                break
        except Exception:
            continue

    if not results:
        print("[INFO] No 'ACE TEST' emails found with current filters.")
    else:
        print(f"[OK] Collected {len(results)} email(s) for extraction/validation.")
    return results


