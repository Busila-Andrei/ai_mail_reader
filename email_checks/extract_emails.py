from __future__ import annotations

"""
Extract emails from Outlook using COM interface.
Extracts comprehensive email data including body, headers, attachments, and metadata.
Saves to JSON format for ML analysis.

Usage:
    python extract_emails.py [--count 100] [--output data/raw/] [--config config.json]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import win32com.client as win32

from config_loader import load_config, get_config_value


def get_email_property(item: Any, prop_name: str, default: Any = None) -> Any:
    """Safely get a property from an Outlook item."""
    try:
        value = getattr(item, prop_name, default)
        if value is None:
            return default
        # Convert COM objects to strings/standard types
        if hasattr(value, '__class__'):
            # Handle datetime objects
            if 'datetime' in str(type(value)).lower():
                return value.isoformat() if hasattr(value, 'isoformat') else str(value)
            # Handle other COM objects
            if 'com' in str(type(value)).lower():
                return str(value)
        return value
    except Exception:
        return default


def extract_email_data(item: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Extract comprehensive data from a single email item."""
    email_data: dict[str, Any] = {}
    
    include_attachments = get_config_value(config, 'extraction', 'include_attachments', default=True)
    include_headers = get_config_value(config, 'extraction', 'include_headers', default=True)
    
    # Basic properties
    email_data['subject'] = get_email_property(item, 'Subject', '')
    email_data['sender'] = get_email_property(item, 'SenderName', '')
    email_data['sender_email'] = get_email_property(item, 'SenderEmailAddress', '')
    email_data['received_time'] = get_email_property(item, 'ReceivedTime', '')
    email_data['sent_on'] = get_email_property(item, 'SentOn', '')
    email_data['creation_time'] = get_email_property(item, 'CreationTime', '')
    
    # Recipients
    try:
        recipients = []
        recips = item.Recipients
        if recips:
            for i in range(recips.Count):
                recip = recips.Item(i + 1)
                recipients.append({
                    'name': get_email_property(recip, 'Name', ''),
                    'email': get_email_property(recip, 'Address', ''),
                    'type': get_email_property(recip, 'Type', '')
                })
        email_data['recipients'] = recipients
    except Exception:
        email_data['recipients'] = []
    
    # Body content
    try:
        body = get_email_property(item, 'Body', '')
        email_data['body_plain'] = body
        email_data['body_length'] = len(body) if body else 0
    except Exception:
        email_data['body_plain'] = ''
        email_data['body_length'] = 0
    
    # HTML body
    try:
        html_body = get_email_property(item, 'HTMLBody', '')
        email_data['body_html'] = html_body
        email_data['body_html_length'] = len(html_body) if html_body else 0
    except Exception:
        email_data['body_html'] = ''
        email_data['body_html_length'] = 0
    
    # Flags and status
    email_data['unread'] = get_email_property(item, 'UnRead', False)
    email_data['importance'] = get_email_property(item, 'Importance', 1)  # 0=Low, 1=Normal, 2=High
    email_data['flag_status'] = get_email_property(item, 'FlagStatus', 0)  # 0=No flag, 1=Flagged
    
    # Categories
    try:
        categories = get_email_property(item, 'Categories', '')
        email_data['categories'] = categories.split(';') if categories else []
    except Exception:
        email_data['categories'] = []
    
    # Attachments
    attachments = []
    if include_attachments:
        try:
            atts = item.Attachments
            if atts:
                for i in range(atts.Count):
                    att = atts.Item(i + 1)
                    att_info = {
                        'filename': get_email_property(att, 'FileName', ''),
                        'size': get_email_property(att, 'Size', 0),
                        'display_name': get_email_property(att, 'DisplayName', '')
                    }
                    # Try to get content type
                    try:
                        att_info['content_type'] = get_email_property(att, 'PropertyAccessor').GetProperty('http://schemas.microsoft.com/mapi/proptag/0x370E001E')
                    except Exception:
                        att_info['content_type'] = ''
                    attachments.append(att_info)
            email_data['attachments'] = attachments
            email_data['attachment_count'] = len(attachments)
        except Exception:
            email_data['attachments'] = []
            email_data['attachment_count'] = 0
    else:
        email_data['attachments'] = []
        email_data['attachment_count'] = 0
    
    # Headers (if accessible)
    if include_headers:
        try:
            property_accessor = item.PropertyAccessor
            headers = property_accessor.GetProperty('http://schemas.microsoft.com/mapi/proptag/0x007D001F')
            email_data['headers'] = headers if headers else ''
        except Exception:
            email_data['headers'] = ''
    else:
        email_data['headers'] = ''
    
    # Additional metadata
    email_data['conversation_id'] = get_email_property(item, 'ConversationID', '')
    email_data['conversation_topic'] = get_email_property(item, 'ConversationTopic', '')
    
    # Word count (approximate)
    body_text = email_data.get('body_plain', '')
    email_data['word_count'] = len(body_text.split()) if body_text else 0
    
    # Extraction metadata
    email_data['extracted_at'] = datetime.now().isoformat()
    
    return email_data


def extract_emails(count: int | None = None, output_dir: str | None = None, config: dict[str, Any] | None = None) -> None:
    """Extract emails from Outlook and save to JSON."""
    if config is None:
        config = load_config()
    
    # Use config values, allow override via parameters
    email_count = count if count is not None else get_config_value(config, 'extraction', 'email_count', default=100)
    output_directory = output_dir if output_dir is not None else get_config_value(config, 'extraction', 'output_directory', default='data/raw')
    source_folder = get_config_value(config, 'extraction', 'source_folder', default='Inbox')
    sort_by = get_config_value(config, 'extraction', 'sort_by', default='ReceivedTime')
    sort_descending = get_config_value(config, 'extraction', 'sort_descending', default=True)
    encoding = get_config_value(config, 'extraction', 'encoding', default='utf-8')
    max_skipped_errors = get_config_value(config, 'processing', 'max_skipped_errors', default=5)
    progress_interval = get_config_value(config, 'processing', 'progress_interval', default=10)
    json_indent = get_config_value(config, 'processing', 'json_indent', default=2)
    ensure_ascii = get_config_value(config, 'processing', 'ensure_ascii', default=False)
    
    # Outlook connection settings
    profile_name = get_config_value(config, 'outlook', 'profile_name', default=None)
    namespace = get_config_value(config, 'outlook', 'namespace', default='MAPI')
    folder_id = get_config_value(config, 'outlook', 'folder_id', default=6)
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to Outlook...")
    try:
        outlook_app = win32.Dispatch("Outlook.Application")
        if profile_name:
            outlook = outlook_app.GetNamespace(namespace)
            outlook.Logon(profile_name)
        else:
            outlook = outlook_app.GetNamespace(namespace)
        
        inbox = outlook.GetDefaultFolder(folder_id)
        items = inbox.Items
        items.Sort(f"[{sort_by}]", sort_descending)
    except Exception as e:
        print(f"Error connecting to Outlook: {e}")
        return
    
    print(f"Extracting {email_count} emails from {source_folder}...")
    emails = []
    extracted_count = 0
    skipped_count = 0
    
    for item in items:
        try:
            # Check if it's a mail item
            if item.Class != 43:  # olMail = 43
                continue
            
            email_data = extract_email_data(item, config)
            emails.append(email_data)
            extracted_count += 1
            
            if extracted_count % progress_interval == 0:
                print(f"  Extracted {extracted_count}/{email_count} emails...")
            
            if extracted_count >= email_count:
                break
                
        except Exception as e:
            skipped_count += 1
            if skipped_count <= max_skipped_errors:
                print(f"  Warning: Skipped email due to error: {e}")
            continue
    
    if not emails:
        print("No emails extracted.")
        return
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"emails_{timestamp}.json"
    
    # Create metadata
    metadata = {
        'extraction_date': datetime.now().isoformat(),
        'total_emails': len(emails),
        'count_requested': email_count,
        'skipped': skipped_count,
        'source_folder': source_folder,
        'date_range': {
            'earliest': min(e.get('received_time', '') for e in emails if e.get('received_time')),
            'latest': max(e.get('received_time', '') for e in emails if e.get('received_time'))
        }
    }
    
    output_data = {
        'metadata': metadata,
        'emails': emails
    }
    
    with open(output_file, 'w', encoding=encoding) as f:
        json.dump(output_data, f, indent=json_indent, ensure_ascii=ensure_ascii)
    
    print(f"\n✓ Successfully extracted {extracted_count} emails")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Skipped {skipped_count} items")
    print(f"✓ File size: {output_file.stat().st_size / 1024:.2f} KB")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract emails from Outlook for ML analysis')
    parser.add_argument('--count', type=int, default=None, help='Number of emails to extract (overrides config)')
    parser.add_argument('--output', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    extract_emails(count=args.count, output_dir=args.output, config=config)


if __name__ == "__main__":
    main()

