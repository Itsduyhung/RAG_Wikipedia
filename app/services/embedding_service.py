from langchain_ollama import OllamaEmbeddings
from app.config import settings
import numpy as np
from typing import List
from app.services.rate_limiter import get_gemini_rate_limiter


class EmbeddingService:
    """Service tạo embeddings bằng Ollama local (nomic-embed-text, 768d)"""
    
    def __init__(self):
        base = getattr(settings, "OLLAMA_BASE_URL", "http://host.docker.internal:11434") or "http://host.docker.internal:11434"
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            base_url=base,
        )
        self.output_dimensionality = settings.DIMENSION_OF_MODEL
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector để đảm bảo chất lượng với smaller dimensions.
        Cần normalize khi dùng dimensions < 3072 (như 768, 1536).
        """
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            normalized = embedding_np / norm
            return normalized.tolist()
        return embedding
    
    def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một đoạn text với truncation và normalization
        """
        # Rate limiting: đảm bảo không vượt quá 15 RPM
        get_gemini_rate_limiter().wait_if_needed()
        
        # Gọi Ollama embeddings (nomic-embed-text)
        embedding = self.embeddings.embed_query(text)
        
        # Truncate về output_dimensionality nếu cần (ví dụ từ 1536 -> 768)
        if len(embedding) > self.output_dimensionality:
            embedding = embedding[: self.output_dimensionality]
        
        # Normalize để đảm bảo chất lượng
        return self._normalize_embedding(embedding)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều documents với truncation và normalization
        """
        # Batch embed với output_dimensionality
        embeddings: List[List[float]] = []
        rate_limiter = get_gemini_rate_limiter()
        
        for text in texts:
            # Rate limiting: đảm bảo không vượt quá 15 RPM
            rate_limiter.wait_if_needed()
            
            embedding = self.embeddings.embed_query(text)
            # Truncate về output_dimensionality nếu cần
            if len(embedding) > self.output_dimensionality:
                embedding = embedding[: self.output_dimensionality]
            
            # Normalize từng embedding
            normalized = self._normalize_embedding(embedding)
            embeddings.append(normalized)
        
        return embeddings


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton instance của EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
