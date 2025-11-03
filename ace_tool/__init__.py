"""
ace_tool: Tools for extracting, validating, and reporting IBM ACE deployment
information from Outlook emails, with optional AI assistance and local memory.

Modules:
- config: Environment-driven configuration flags and constants
- utils: Small helpers for text/HTML/booleans/paths/time parsing
- ace_client: Thin client for ACE REST APIs with retries + caches
- ace_helpers: ACE-derived inference helpers
- extraction: Rule-based + LLM fallback extraction from email content
- validator: Cross-check extracted info against ACE and CFG lists
- cfg_discovery: Local/SSH discovery of .cfg files
- ai_utils: AI helper calls and candidate set collection
- outlook_reader: Read/scan Outlook inbox with filters
- overview: Build HTML overview and ingest to local memory
"""

__all__ = [
    "config",
    "utils",
    "ace_client",
    "ace_helpers",
    "extraction",
    "validator",
    "cfg_discovery",
    "ai_utils",
    "outlook_reader",
    "overview",
]


