"""
Text chunker for splitting email text into manageable chunks for embedding.
"""

from __future__ import annotations

from typing import Any


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict[str, Any]]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters (default: 600)
        chunk_overlap: Overlap between chunks in characters (default: 100)
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or len(text) <= chunk_size:
        return [{
            'text': text,
            'start': 0,
            'end': len(text),
            'chunk_index': 0
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Determine end position
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings near the target end
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
            
            # If no sentence boundary found, try word boundaries
            if end == start + chunk_size:
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in ' \t\n':
                        end = i + 1
                        break
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end,
                'chunk_index': chunk_index
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def chunk_email(email_data: dict[str, Any], chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict[str, Any]]:
    """
    Chunk an email's text content and preserve metadata.
    
    Args:
        email_data: Email data dictionary with 'body_text' and metadata
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunk dictionaries with text and email metadata
    """
    text = email_data.get('body_text', '')
    if not text:
        return []
    
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Add email metadata to each chunk
    for chunk in chunks:
        chunk['email_subject'] = email_data.get('subject', '')
        chunk['email_sender'] = email_data.get('sender', '')
        chunk['email_date'] = email_data.get('date', '')
        chunk['email_file_path'] = email_data.get('file_path', '')
        chunk['email_file_name'] = email_data.get('file_name', '')
    
    return chunks

