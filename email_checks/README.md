# email_checks

Standalone scripts to test different ways of reading email without modifying your main app.

Important:
- Do NOT hardcode credentials in code. Use environment variables.
- For Microsoft 365, prefer Microsoft Graph (device code) — no password needed.

## Scripts
- check_outlook_com.py: Reads top 5 messages using Outlook desktop (COM).
- check_graph_device_code.py: Reads top 5 messages via Microsoft Graph using device code login.

## Quick usage (Python-only)

### On a machine with internet (to download packages)
```bash
python prepare_packages.py
```

### On your offline work PC
```bash
python bootstrap_offline.py

# Then run a checker using the venv python
.venv/Scripts/python.exe check_outlook_com.py
.venv/Scripts/python.exe check_graph_device_code.py
```

## Notes:
- pywin32 packages are platform-specific; download on a machine with the same Windows/architecture as your work PC.
- For Graph, use device-code login; no secrets/files needed.

