"""Configuration and environment flags for ace_tool.

Loads .env and exposes constants used across modules.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv


# Load .env once at import time
load_dotenv()

# ACE connection
ACE_HOST = os.getenv("ACE_HOST", "localhost")
ACE_PORT = int(os.getenv("ACE_PORT", "4414"))
ACE_USER = os.getenv("ACE_USER")
ACE_PASS = os.getenv("ACE_PASS")
VERIFY_SSL_ENV = os.getenv("VERIFY_SSL", "0")
VERIFY_SSL = False if str(VERIFY_SSL_ENV).strip().lower() in {"0", "false", "no", ""} else True

# Timeouts
ACE_REQUEST_TIMEOUT = float(os.getenv("ACE_REQUEST_TIMEOUT", "8"))

# App behavior toggles
SKIP_ACE = os.getenv("SKIP_ACE", "0").lower() in {"1", "true", "yes", "y"}
PROCESS_ONLY_UNREAD = os.getenv("PROCESS_ONLY_UNREAD", "1").lower() in {"1", "true", "yes", "y"}
MARK_AS_READ = os.getenv("MARK_AS_READ", "1").lower() in {"1", "true", "yes", "y"}
PROCESS_ONLY_SINGLE_CONVERSATION = os.getenv("PROCESS_ONLY_SINGLE_CONVERSATION", "1").lower() in {"1", "true", "yes", "y"}
DEBUG_EMAIL_SCAN = os.getenv("DEBUG_EMAIL_SCAN", "0").lower() in {"1", "true", "yes", "y"}

# SSH/CFG discovery
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USER = os.getenv("SSH_USER")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")
SSH_KEY_PASS = os.getenv("SSH_KEY_PASSPHRASE") or None
SSH_KNOWN = os.getenv("SSH_KNOWN_HOSTS")
SSH_CFG_PATH = os.getenv("SSH_CFG_PATH")
CFG_ROOT = os.getenv("CFG_ROOT")

# AI toggles
USE_AI_RESOLVER = os.getenv("USE_AI_RESOLVER", "1").lower() in {"1", "true", "yes", "y"}
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
AI_MODEL = os.getenv("AI_MODEL", "llama3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "12"))

# Regex patterns for action words
ACTION_PATTERNS = os.getenv("ACTION_PATTERNS", r"\bdeploy\b,\bdeployati\b,\baplicati\b")


