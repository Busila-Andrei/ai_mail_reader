"""
Console-based chat interface for querying emails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embedding_generator import EmbeddingGenerator
from faiss_indexer import FAISSIndexer
from llm_query import LLMQuery


class ConsoleChat:
    """Console-based chat interface for email Q&A."""
    
    def __init__(self, index_path: str | Path = 'data/faiss_index'):
        """
        Initialize console chat.
        
        Args:
            index_path: Path to FAISS index
        """
        self.index_path = Path(index_path)
        self.indexer = None
        self.embedding_gen = None
        self.llm = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize components."""
        print("Initializare sistem de chat...")
        
        # Load FAISS index
        if not (self.index_path / 'index.faiss').exists():
            print(f"Eroare: Index FAISS nu există la {self.index_path}")
            print("Rulare mai întâi 'python main.py index' pentru a crea indexul.")
            return
        
        self.indexer = FAISSIndexer(self.index_path)
        
        # Initialize embedding generator
        self.embedding_gen = EmbeddingGenerator()
        
        # Initialize LLM (optional, can work without it)
        try:
            self.llm = LLMQuery()
        except Exception as e:
            print(f"Warning: LLM nu a putut fi încărcat: {e}")
            print("Va folosi răspunsuri simple bazate pe context.")
            self.llm = None
        
        print("✓ Sistem inițializat")
    
    def chat_loop(self) -> None:
        """Main chat loop."""
        if not self.indexer or not self.embedding_gen:
            print("Eroare: Sistemul nu este inițializat corect.")
            return
        
        print("\n" + "="*60)
        print("Sistem de interogare emailuri")
        print("="*60)
        print("Introdu o întrebare despre emailuri sau 'quit' pentru a ieși.\n")
        
        while True:
            try:
                question = input("Întrebare: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("La revedere!")
                    break
                
                # Process question
                answer, sources = self._process_question(question)
                
                # Display answer
                print("\n" + "-"*60)
                print("Răspuns:")
                print(answer)
                print("\nSurse:")
                for i, source in enumerate(sources[:3], 1):
                    print(f"  {i}. {source}")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nLa revedere!")
                break
            except Exception as e:
                print(f"Eroare: {e}\n")
    
    def _process_question(self, question: str) -> tuple[str, list[str]]:
        """
        Process a question and return answer with sources.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, list of source descriptions)
        """
        # Generate embedding for question
        query_embedding = self.embedding_gen.generate_embedding(question)
        
        # Search in FAISS index
        results = self.indexer.search(query_embedding, k=5)
        
        if not results:
            return "Nu am găsit informații relevante în emailuri.", []
        
        # Generate answer using LLM or fallback
        if self.llm:
            answer = self.llm.generate_answer(question, results)
        else:
            # Simple fallback
            top_chunk = results[0].get('chunk', {})
            answer = f"Pe baza emailurilor, {top_chunk.get('text', '')[:300]}..."
        
        # Build source list
        sources = []
        for result in results[:3]:
            chunk = result.get('chunk', {})
            subject = chunk.get('email_subject', 'N/A')
            sender = chunk.get('email_sender', 'N/A')
            date = chunk.get('email_date', 'N/A')
            source = f"Email: '{subject}' de la {sender} ({date})"
            sources.append(source)
        
        return answer, sources

