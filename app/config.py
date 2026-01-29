from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os


def _strip_str(v: str) -> str:
    """Strip whitespace/newlines from env strings (avoid Ollama 'unqualified name' etc)."""
    return (v or "").strip()


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://rag_user:rag_password@127.0.0.1:5433/rag_db"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Google AI (optional, khi không dùng Ollama)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # OpenAI (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Provider selection
    # - ollama: local/self-hosted (Docker compose)
    # - gemini/openai: for Render deployment (no Ollama available)
    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "ollama"
    # Ollama local (trong Docker dùng http://ollama:11434, ngoài Docker dùng http://localhost:11434)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    # Models: LLM và embedding (Ollama model names)
    LLM_MODEL_NAME: str = "qwen2.5:1.5b"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    DIMENSION_OF_MODEL: int = 768
    # Context size cho Ollama chat (qwen2.5:1.5b = 2048; mistral = 8192)
    OLLAMA_NUM_CTX: int = 2048

    @field_validator("OLLAMA_BASE_URL", "LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME", mode="before")
    @classmethod
    def strip_ollama_strings(cls, v):
        if isinstance(v, str):
            return _strip_str(v)
        return v

    # Pinecone (Legacy - optional, for backward compatibility)
    PINECONE_API_KEY: Optional[str] = None
    
    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    
    # File Storage
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed_data"
    RAW_DIR: str = "./data/raw_data"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Docker environment and adjust paths
        if os.path.exists('/app'):
            self.DATA_DIR = "/app/data"
            self.UPLOAD_DIR = "/app/data/uploads"
            self.PROCESSED_DIR = "/app/data/processed_data"
            self.RAW_DIR = "/app/data/raw_data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
