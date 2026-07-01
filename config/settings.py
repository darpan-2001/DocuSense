from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "DOCU SENSE"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Groq API
    GROQ_API_KEY: str
    
    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    
    # LLM Configuration
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    
    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval Configuration
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 5
    DENSE_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Memory Configuration
    MAX_CONVERSATION_TURNS: int = 10
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "docu_sense.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
