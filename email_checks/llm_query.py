"""
LLM query module for generating answers using local LLM.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLMQuery:
    """Query local LLM for generating answers."""
    
    def __init__(self, model_name: str = 'microsoft/DialoGPT-medium', device: str = 'auto'):
        """
        Initialize LLM query handler.
        
        Args:
            model_name: Name of the LLM model (Llama 3, Mistral, etc.)
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the LLM model."""
        print(f"Loading LLM model: {self.model_name}...")
        print("Note: This may take a while on first run...")
        
        try:
            # Try to use pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device_map=self.device,
                torch_dtype="auto"
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Falling back to simple text generation...")
            self.pipeline = None
    
    def generate_answer(self, question: str, context_chunks: list[dict[str, Any]], max_length: int = 512) -> str:
        """
        Generate an answer to a question based on context chunks.
        
        Args:
            question: User's question
            context_chunks: List of relevant email chunks with metadata
            max_length: Maximum length of generated response
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            return "Nu am găsit informații relevante în emailuri pentru această întrebare."
        
        # Build context from chunks
        context_text = self._build_context(context_chunks)
        
        # Build prompt
        prompt = self._build_prompt(question, context_text)
        
        # Generate answer
        if self.pipeline:
            try:
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                generated_text = result[0]['generated_text']
                # Extract answer (remove prompt)
                answer = generated_text[len(prompt):].strip()
                
                if not answer:
                    answer = self._fallback_answer(question, context_chunks)
                
                return answer
            except Exception as e:
                print(f"Error generating answer: {e}")
                return self._fallback_answer(question, context_chunks)
        else:
            # Fallback to simple template-based answer
            return self._fallback_answer(question, context_chunks)
    
    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Build context text from chunks."""
        context_parts = []
        
        for i, result in enumerate(chunks[:5], 1):  # Use top 5 chunks
            chunk = result.get('chunk', {})
            text = chunk.get('text', '')
            subject = chunk.get('email_subject', '')
            sender = chunk.get('email_sender', '')
            date = chunk.get('email_date', '')
            
            context_part = f"[Fragment {i}]"
            if subject:
                context_part += f"\nSubiect: {subject}"
            if sender:
                context_part += f"\nExpeditor: {sender}"
            if date:
                context_part += f"\nData: {date}"
            context_part += f"\nConținut: {text}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM."""
        prompt = f"""Pe baza următoarelor fragmente din emailuri, răspunde la întrebare în română.

Fragmente din emailuri:
{context}

Întrebare: {question}

Răspuns:"""
        return prompt
    
    def _fallback_answer(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """Generate a fallback answer when LLM is not available."""
        if not context_chunks:
            return "Nu am găsit informații relevante pentru această întrebare."
        
        # Simple template-based answer
        top_chunk = context_chunks[0].get('chunk', {})
        text = top_chunk.get('text', '')
        subject = top_chunk.get('email_subject', '')
        
        answer = f"Pe baza emailului '{subject}', "
        
        # Try to extract relevant sentence
        sentences = text.split('.')
        if sentences:
            answer += sentences[0] + "."
        else:
            answer += text[:200] + "..."
        
        return answer

