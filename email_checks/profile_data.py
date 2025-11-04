from __future__ import annotations

"""
Profile and analyze extracted email data.
Generates statistics, structure analysis, and feature summaries for ML preparation.

Usage:
    python profile_data.py [--input data/raw/emails_*.json] [--output data/processed/] [--config config.json]
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import dateutil.parser

from config_loader import load_config, get_config_value


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text (approximate)."""
    if not text:
        return 0
    # Simple sentence counting
    return len(re.split(r'[.!?]+', text))


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text, re.IGNORECASE)


def extract_emails_from_text(text: str) -> list[str]:
    """Extract email addresses from text."""
    if not text:
        return []
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_phone_numbers(text: str) -> list[str]:
    """Extract phone numbers from text (basic pattern)."""
    if not text:
        return []
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    return re.findall(phone_pattern, text)


def get_sender_domain(email: str) -> str:
    """Extract domain from email address."""
    if not email or '@' not in email:
        return ''
    return email.split('@')[-1].lower()


def parse_datetime(date_str: str) -> datetime | None:
    """Parse datetime string safely."""
    if not date_str:
        return None
    try:
        return dateutil.parser.parse(date_str)
    except Exception:
        return None


def analyze_email(email: dict[str, Any]) -> dict[str, Any]:
    """Analyze a single email and extract features."""
    analysis: dict[str, Any] = {}
    
    body = email.get('body_plain', '')
    
    # Text features
    analysis['word_count'] = count_words(body)
    analysis['sentence_count'] = count_sentences(body)
    analysis['character_count'] = len(body)
    analysis['avg_words_per_sentence'] = (
        analysis['word_count'] / analysis['sentence_count']
        if analysis['sentence_count'] > 0 else 0
    )
    
    # Content features
    analysis['urls'] = extract_urls(body)
    analysis['url_count'] = len(analysis['urls'])
    analysis['emails_in_body'] = extract_emails_from_text(body)
    analysis['email_count_in_body'] = len(analysis['emails_in_body'])
    analysis['phone_numbers'] = extract_phone_numbers(body)
    analysis['phone_count'] = len(analysis['phone_numbers'])
    
    # Structural features
    analysis['has_attachments'] = email.get('attachment_count', 0) > 0
    analysis['attachment_count'] = email.get('attachment_count', 0)
    analysis['recipient_count'] = len(email.get('recipients', []))
    
    # Metadata features
    sender_email = email.get('sender_email', '')
    analysis['sender_domain'] = get_sender_domain(sender_email)
    analysis['has_categories'] = len(email.get('categories', [])) > 0
    analysis['category_count'] = len(email.get('categories', []))
    analysis['is_unread'] = email.get('unread', False)
    analysis['importance'] = email.get('importance', 1)
    analysis['is_flagged'] = email.get('flag_status', 0) > 0
    
    # HTML vs plain text
    html_length = email.get('body_html_length', 0)
    plain_length = email.get('body_length', 0)
    analysis['has_html'] = html_length > 0
    analysis['html_ratio'] = html_length / plain_length if plain_length > 0 else 0
    
    # Temporal features
    received_time = parse_datetime(email.get('received_time', ''))
    if received_time:
        analysis['hour_of_day'] = received_time.hour
        analysis['day_of_week'] = received_time.weekday()  # 0=Monday, 6=Sunday
        analysis['is_weekend'] = analysis['day_of_week'] >= 5
        analysis['month'] = received_time.month
        analysis['year'] = received_time.year
    
    return analysis


def profile_emails(input_file: str, output_dir: str | None = None, config: dict[str, Any] | None = None) -> None:
    """Profile email data and generate statistics."""
    if config is None:
        config = load_config()
    
    output_directory = output_dir if output_dir is not None else get_config_value(config, 'profiling', 'output_directory', default='data/processed')
    encoding = get_config_value(config, 'profiling', 'encoding', default='utf-8')
    json_indent = get_config_value(config, 'processing', 'json_indent', default=2)
    ensure_ascii = get_config_value(config, 'processing', 'ensure_ascii', default=False)
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Loading email data from {input_file}...")
    with open(input_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    
    emails = data.get('emails', [])
    if not emails:
        print("No emails found in file.")
        return
    
    print(f"Analyzing {len(emails)} emails...")
    
    # Analyze each email
    analyzed_emails = []
    all_features = []
    
    for email in emails:
        analysis = analyze_email(email)
        email['features'] = analysis
        analyzed_emails.append(email)
        all_features.append(analysis)
    
    # Generate statistics
    stats: dict[str, Any] = {
        'profile_date': datetime.now().isoformat(),
        'total_emails': len(emails),
        'source_file': str(input_path),
    }
    
    # Text statistics
    word_counts = [f['word_count'] for f in all_features]
    stats['text_stats'] = {
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_word_count': min(word_counts) if word_counts else 0,
        'max_word_count': max(word_counts) if word_counts else 0,
        'median_word_count': sorted(word_counts)[len(word_counts) // 2] if word_counts else 0,
    }
    
    # Attachment statistics
    attachment_counts = [f['attachment_count'] for f in all_features]
    stats['attachment_stats'] = {
        'emails_with_attachments': sum(1 for f in all_features if f['has_attachments']),
        'total_attachments': sum(attachment_counts),
        'avg_attachments': sum(attachment_counts) / len(attachment_counts) if attachment_counts else 0,
    }
    
    # Sender statistics
    sender_domains = [f['sender_domain'] for f in all_features if f['sender_domain']]
    domain_counts = Counter(sender_domains)
    stats['sender_stats'] = {
        'unique_senders': len(set(sender_domains)),
        'top_sender_domains': dict(domain_counts.most_common(10)),
    }
    
    # Content statistics
    stats['content_stats'] = {
        'emails_with_urls': sum(1 for f in all_features if f['url_count'] > 0),
        'total_urls': sum(f['url_count'] for f in all_features),
        'emails_with_phones': sum(1 for f in all_features if f['phone_count'] > 0),
        'emails_with_email_addresses': sum(1 for f in all_features if f['email_count_in_body'] > 0),
    }
    
    # Status statistics
    stats['status_stats'] = {
        'unread_count': sum(1 for f in all_features if f['is_unread']),
        'flagged_count': sum(1 for f in all_features if f['is_flagged']),
        'high_importance_count': sum(1 for f in all_features if f['importance'] == 2),
        'low_importance_count': sum(1 for f in all_features if f['importance'] == 0),
    }
    
    # Temporal statistics
    hours = [f['hour_of_day'] for f in all_features if 'hour_of_day' in f]
    if hours:
        stats['temporal_stats'] = {
            'most_active_hour': Counter(hours).most_common(1)[0][0] if hours else None,
            'weekend_emails': sum(1 for f in all_features if f.get('is_weekend', False)),
            'weekday_emails': sum(1 for f in all_features if not f.get('is_weekend', True)),
        }
    
    # HTML statistics
    html_ratios = [f['html_ratio'] for f in all_features]
    stats['html_stats'] = {
        'emails_with_html': sum(1 for f in all_features if f['has_html']),
        'avg_html_ratio': sum(html_ratios) / len(html_ratios) if html_ratios else 0,
    }
    
    # Save profiled data
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save analyzed emails with features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiled_file = output_path / f"emails_profiled_{timestamp}.json"
    
    profiled_data = {
        'metadata': data.get('metadata', {}),
        'statistics': stats,
        'emails': analyzed_emails
    }
    
    with open(profiled_file, 'w', encoding=encoding) as f:
        json.dump(profiled_data, f, indent=json_indent, ensure_ascii=ensure_ascii)
    
    # Save statistics summary (separate file)
    stats_file = output_path / f"statistics_{timestamp}.json"
    with open(stats_file, 'w', encoding=encoding) as f:
        json.dump(stats, f, indent=json_indent, ensure_ascii=ensure_ascii)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA PROFILE SUMMARY")
    print("="*60)
    print(f"Total emails analyzed: {stats['total_emails']}")
    print(f"\nText Statistics:")
    print(f"  Average word count: {stats['text_stats']['avg_word_count']:.1f}")
    print(f"  Word count range: {stats['text_stats']['min_word_count']} - {stats['text_stats']['max_word_count']}")
    print(f"\nAttachment Statistics:")
    print(f"  Emails with attachments: {stats['attachment_stats']['emails_with_attachments']}")
    print(f"  Total attachments: {stats['attachment_stats']['total_attachments']}")
    print(f"\nSender Statistics:")
    print(f"  Unique sender domains: {stats['sender_stats']['unique_senders']}")
    print(f"  Top domains: {', '.join(list(stats['sender_stats']['top_sender_domains'].keys())[:5])}")
    print(f"\nContent Statistics:")
    print(f"  Emails with URLs: {stats['content_stats']['emails_with_urls']}")
    print(f"  Emails with phone numbers: {stats['content_stats']['emails_with_phones']}")
    print(f"\nStatus Statistics:")
    print(f"  Unread: {stats['status_stats']['unread_count']}")
    print(f"  Flagged: {stats['status_stats']['flagged_count']}")
    print(f"  High importance: {stats['status_stats']['high_importance_count']}")
    
    if 'temporal_stats' in stats:
        print(f"\nTemporal Statistics:")
        print(f"  Most active hour: {stats['temporal_stats']['most_active_hour']}")
        print(f"  Weekend emails: {stats['temporal_stats']['weekend_emails']}")
    
    print(f"\n✓ Profiled data saved to: {profiled_file}")
    print(f"✓ Statistics saved to: {stats_file}")


def main() -> None:
    """Main entry point."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Profile and analyze email data')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file pattern (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Get input pattern from config or args
    input_pattern = args.input if args.input is not None else get_config_value(config, 'profiling', 'input_pattern', default='data/raw/emails_*.json')
    use_most_recent = get_config_value(config, 'profiling', 'use_most_recent', default=True)
    
    # Find matching files
    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found matching: {input_pattern}")
        return
    
    # Use the most recent file or all files
    if use_most_recent:
        files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        latest_file = files[0]
        profile_emails(latest_file, args.output, config)
    else:
        # Process all matching files
        for file_path in files:
            print(f"\nProcessing: {file_path}")
            profile_emails(file_path, args.output, config)


if __name__ == "__main__":
    main()

