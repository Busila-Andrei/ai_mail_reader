"""
FAISS indexer for storing and retrieving email embeddings.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSIndexer:
    """FAISS-based vector index for email chunks."""
    
    def __init__(self, index_path: str | Path | None = None, dimension: int = 384):
        """
        Initialize FAISS indexer.
        
        Args:
            index_path: Path to save/load index
            dimension: Dimension of embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        self.index_path = Path(index_path) if index_path else Path('data/faiss_index')
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize or load FAISS index."""
        index_file = self.index_path / 'index.faiss'
        metadata_file = self.index_path / 'metadata.pkl'
        
        if index_file.exists() and metadata_file.exists():
            print(f"Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(str(index_file))
            
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"✓ Loaded index with {self.index.ntotal} vectors")
        else:
            # Create new index
            print(f"Creating new FAISS index (dimension: {self.dimension})...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("✓ Index created")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: Array of embedding vectors (n_chunks, dimension)
            chunks: List of chunk dictionaries with metadata
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) must match number of chunks ({len(chunks)})")
        
        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(chunks)
        
        print(f"✓ Added {len(chunks)} chunks to index (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of result dictionaries with chunks and metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'chunk': self.metadata[idx].copy(),
                    'distance': float(distance),
                    'similarity': float(1 / (1 + distance)),  # Convert distance to similarity
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / 'index.faiss'
        metadata_file = self.index_path / 'metadata.pkl'
        
        print(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, str(index_file))
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Also save a JSON summary
        summary_file = self.index_path / 'summary.json'
        summary = {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexFlatL2'
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Index saved ({self.index.ntotal} vectors)")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'index_path': str(self.index_path)
        }

