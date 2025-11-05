"""
Email parser module for extracting text from .eml, .msg files and plain text.
"""

from __future__ import annotations

import email
import re
from pathlib import Path
from typing import Any

import html2text
from bs4 import BeautifulSoup


def parse_eml_file(file_path: str | Path) -> dict[str, Any]:
    """
    Parse an .eml file and extract email content.
    
    Args:
        file_path: Path to .eml file
        
    Returns:
        Dictionary with subject, sender, body, date, and metadata
    """
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as f:
        msg = email.message_from_bytes(f.read())
    
    # Extract basic fields
    subject = msg.get('Subject', '')
    sender = msg.get('From', '')
    date = msg.get('Date', '')
    
    # Extract body
    body_text = ''
    body_html = ''
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition', ''))
            
            # Skip attachments
            if 'attachment' in content_disposition:
                continue
            
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body_text = payload.decode(charset, errors='ignore')
                except Exception:
                    pass
            
            elif content_type == 'text/html':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body_html = payload.decode(charset, errors='ignore')
                except Exception:
                    pass
    else:
        # Single part message
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                content = payload.decode(charset, errors='ignore')
                
                if msg.get_content_type() == 'text/html':
                    body_html = content
                else:
                    body_text = content
        except Exception:
            pass
    
    # Convert HTML to text if needed
    if body_html and not body_text:
        body_text = html_to_text(body_html)
    
    # Normalize text
    normalized_text = normalize_email_text(body_text)
    
    return {
        'subject': subject,
        'sender': sender,
        'date': date,
        'body_text': normalized_text,
        'body_html': body_html,
        'file_path': str(file_path),
        'file_name': file_path.name
    }


def parse_msg_file(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a .msg file and extract email content.
    
    Args:
        file_path: Path to .msg file
        
    Returns:
        Dictionary with subject, sender, body, date, and metadata
    """
    try:
        import extract_msg
    except ImportError:
        raise ImportError("extract-msg package is required for .msg files. Install with: pip install extract-msg")
    
    file_path = Path(file_path)
    msg = extract_msg.Message(file_path)
    
    # Extract basic fields
    subject = msg.subject or ''
    sender = msg.sender or ''
    date = str(msg.date) if msg.date else ''
    
    # Extract body
    body_text = msg.body or ''
    body_html = msg.htmlBody or ''
    
    # Convert HTML to text if needed
    if body_html and not body_text:
        body_text = html_to_text(body_html)
    
    # Normalize text
    normalized_text = normalize_email_text(body_text)
    
    msg.close()
    
    return {
        'subject': subject,
        'sender': sender,
        'date': date,
        'body_text': normalized_text,
        'body_html': body_html,
        'file_path': str(file_path),
        'file_name': file_path.name
    }


def parse_text_input(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Parse plain text input (copied from email).
    
    Args:
        text: Plain text content
        metadata: Optional metadata (subject, sender, date, etc.)
        
    Returns:
        Dictionary with extracted email data
    """
    metadata = metadata or {}
    
    normalized_text = normalize_email_text(text)
    
    return {
        'subject': metadata.get('subject', ''),
        'sender': metadata.get('sender', ''),
        'date': metadata.get('date', ''),
        'body_text': normalized_text,
        'body_html': '',
        'file_path': metadata.get('file_path', ''),
        'file_name': metadata.get('file_name', 'text_input')
    }


def html_to_text(html_content: str) -> str:
    """Convert HTML content to plain text."""
    if not html_content:
        return ''
    
    # Use html2text for better conversion
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # Don't wrap lines
    text = h.handle(html_content)
    
    return text.strip()


def normalize_email_text(text: str) -> str:
    """
    Normalize email text by removing signatures, excessive whitespace, etc.
    
    Args:
        text: Raw email text
        
    Returns:
        Normalized text
    """
    if not text:
        return ''
    
    # Remove common email signatures (lines starting with --)
    lines = text.split('\n')
    normalized_lines = []
    in_signature = False
    
    for line in lines:
        # Check for signature markers
        if line.strip().startswith('--') or line.strip().startswith('___'):
            in_signature = True
        
        # Common signature patterns
        if re.match(r'^Sent from (my|a)', line, re.IGNORECASE):
            in_signature = True
        
        if not in_signature:
            normalized_lines.append(line)
    
    text = '\n'.join(normalized_lines)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    
    # Remove HTML entities if any remain
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    
    return text.strip()


def detect_email_file_type(file_path: str | Path) -> str | None:
    """
    Detect the type of email file.
    
    Args:
        file_path: Path to email file
        
    Returns:
        'eml', 'msg', or None
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext == '.eml':
        return 'eml'
    elif ext == '.msg':
        return 'msg'
    
    return None

