from __future__ import annotations

"""
Check Outlook desktop (COM) access and list top 5 subjects from Inbox.
Requires: Outlook installed and a signed-in profile.
"""

import win32com.client as win32


def main() -> None:
    outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # Inbox
    items = inbox.Items
    items.Sort("[ReceivedTime]", True)
    count = 0
    for item in items:
        try:
            print(f"Subject: {(getattr(item, 'Subject', '') or '').strip()}")
            count += 1
            if count >= 5:
                break
        except Exception:
            continue
    if count == 0:
        print("No items found or access issue.")


if __name__ == "__main__":
    main()


