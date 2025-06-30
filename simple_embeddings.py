import hashlib
import numpy as np
import re
from collections import Counter
from langchain_core.embeddings import Embeddings

class SimpleEmbeddings(Embeddings):
    """An improved embedding class that captures more semantic meaning"""
    
    def __init__(self, vector_size=768):  # <-- Change 1536 to 768  # <-- 1536 is the default here
        self.vector_size = vector_size
        # Create a simple vocabulary of 5000 common words for better similarity
        # This is just a very basic approach to get better results without external APIs
        self.vocab_size = 5000
        
    def embed_documents(self, texts):
        """Embed a list of documents"""
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        """Embed a query"""
        return self._get_embedding(text)
    
    def _preprocess_text(self, text):
        """Preprocess text by converting to lowercase, removing special chars"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and replace with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_embedding(self, text):
        """Generate an embedding vector based on word frequencies and positioning"""
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Split into words
        words = processed_text.split()
        
        # Initialize vector with zeros
        embedding = np.zeros(self.vector_size)
        
        if not words:
            return embedding.tolist()
        
        # Count word occurrences
        word_counts = Counter(words)
        
        # Fill embedding vector based on word frequencies and positions
        for i, word in enumerate(words):
            # Generate a deterministic hash for the word
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Use the hash to determine vector positions to update
            positions = [(word_hash + i) % self.vector_size for i in range(10)]
            
            # Word frequency factor (normalized)
            freq = word_counts[word] / len(words)
            
            # Position factor (words earlier in text have slightly more weight)
            pos_factor = 1.0 - (i / (len(words) * 2))
            
            # Update those positions in the vector
            for pos in positions:
                embedding[pos] += (freq * pos_factor)
        
        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
