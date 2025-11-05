"""
Main entry point for email Q&A system.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from console_chat import ConsoleChat
from email_parser import detect_email_file_type, parse_eml_file, parse_msg_file, parse_text_input
from embedding_generator import EmbeddingGenerator
from faiss_indexer import FAISSIndexer
from ocr_processor import extract_text_from_image, is_image_file
from text_chunker import chunk_email


def index_emails(input_path: str, index_path: str = 'data/faiss_index', chunk_size: int = 600) -> None:
    """
    Index emails from input path.
    
    Args:
        input_path: Path to email files, text files, or images
        index_path: Path to save FAISS index
        chunk_size: Size of text chunks
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Eroare: Calea '{input_path}' nu există.")
        return
    
    print(f"Indexare emailuri din: {input_path}")
    print(f"Dimensiune chunk: {chunk_size} caractere\n")
    
    # Initialize components
    embedding_gen = EmbeddingGenerator()
    indexer = FAISSIndexer(index_path, dimension=embedding_gen.get_embedding_dimension())
    
    # Collect all files
    all_chunks = []
    processed_count = 0
    
    # Process files
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        # Find all supported files
        files_to_process = []
        for ext in ['.eml', '.msg', '.png', '.jpg', '.jpeg', '.txt']:
            files_to_process.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Găsite {len(files_to_process)} fișiere pentru procesare...\n")
    
    for file_path in files_to_process:
        try:
            print(f"Procesare: {file_path.name}...")
            
            # Determine file type and parse
            if is_image_file(file_path):
                # OCR processing
                ocr_result = extract_text_from_image(file_path)
                email_data = {
                    'subject': '',
                    'sender': '',
                    'date': '',
                    'body_text': ocr_result['text'],
                    'file_path': str(file_path),
                    'file_name': file_path.name
                }
            elif file_path.suffix.lower() == '.eml':
                email_data = parse_eml_file(file_path)
            elif file_path.suffix.lower() == '.msg':
                email_data = parse_msg_file(file_path)
            elif file_path.suffix.lower() == '.txt':
                # Plain text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                email_data = parse_text_input(text, {
                    'file_path': str(file_path),
                    'file_name': file_path.name
                })
            else:
                continue
            
            # Chunk email
            chunks = chunk_email(email_data, chunk_size=chunk_size)
            
            if chunks:
                all_chunks.extend(chunks)
                processed_count += 1
                print(f"  ✓ Extras {len(chunks)} chunks")
            else:
                print(f"  ⚠ Nu s-au extras chunks (text gol)")
        
        except Exception as e:
            print(f"  ✗ Eroare la procesare: {e}")
            continue
    
    if not all_chunks:
        print("\nNu s-au găsit chunks pentru indexare.")
        return
    
    print(f"\nGenerare embeddings pentru {len(all_chunks)} chunks...")
    
    # Generate embeddings
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedding_gen.generate_embeddings_batch(texts)
    
    # Add to index
    indexer.add_embeddings(embeddings, all_chunks)
    
    # Save index
    indexer.save()
    
    print(f"\n✓ Indexare completă!")
    print(f"  - Fișiere procesate: {processed_count}")
    print(f"  - Chunks totali: {len(all_chunks)}")
    print(f"  - Index salvat la: {index_path}")


def chat_mode(index_path: str = 'data/faiss_index') -> None:
    """
    Start chat mode.
    
    Args:
        index_path: Path to FAISS index
    """
    chat = ConsoleChat(index_path)
    chat.chat_loop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sistem de interogare AI pentru emailuri')
    subparsers = parser.add_subparsers(dest='command', help='Comandă')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Indexare emailuri')
    index_parser.add_argument('--input', '-i', type=str, required=True,
                             help='Cale către fișiere email (.eml, .msg), imagini, sau text')
    index_parser.add_argument('--index-path', '-p', type=str, default='data/faiss_index',
                             help='Cale pentru salvare index (default: data/faiss_index)')
    index_parser.add_argument('--chunk-size', '-c', type=int, default=600,
                             help='Dimensiune chunk în caractere (default: 600)')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Mod chat pentru interogare')
    chat_parser.add_argument('--index-path', '-p', type=str, default='data/faiss_index',
                            help='Cale către index (default: data/faiss_index)')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        index_emails(args.input, args.index_path, args.chunk_size)
    elif args.command == 'chat':
        chat_mode(args.index_path)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

