from __future__ import annotations

"""
Configuration loader for email extraction and profiling scripts.
Loads settings from config.json with fallback to defaults.
"""

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "extraction": {
        "email_count": 100,
        "output_directory": "data/raw",
        "source_folder": "Inbox",
        "sort_by": "ReceivedTime",
        "sort_descending": True,
        "include_attachments": True,
        "include_headers": True,
        "encoding": "utf-8"
    },
    "profiling": {
        "input_pattern": "data/raw/emails_*.json",
        "output_directory": "data/processed",
        "use_most_recent": True,
        "encoding": "utf-8"
    },
    "outlook": {
        "profile_name": None,
        "namespace": "MAPI",
        "folder_id": 6,
        "folder_name": "Inbox"
    },
    "processing": {
        "max_skipped_errors": 5,
        "progress_interval": 10,
        "json_indent": 2,
        "ensure_ascii": False
    }
}


def load_config(config_path: str | Path = None) -> dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file (default: config/config.json relative to project root)
        
    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        # Default to config/config.json relative to project root
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.json"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_file}. Using defaults.")
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Merge with defaults to ensure all keys exist
        merged_config = merge_config(DEFAULT_CONFIG, config)
        return merged_config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        print("Using default configuration.")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration.")
        return DEFAULT_CONFIG.copy()


def merge_config(default: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Merge user config into default config, preserving nested structure."""
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot-notation keys.
    
    Example:
        get_config_value(config, 'extraction', 'email_count') -> 100
        get_config_value(config, 'outlook', 'profile_name') -> None
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value if value is not None else default

