from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class SourceCitation(BaseModel):
    """Schema for source citation."""
    
    document_name: str
    page_number: Optional[str] = None
    chunk_id: str
    relevance_score: float


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str
    detail: Optional[str] = None


class SessionDeleteResponse(BaseModel):
    """Response schema for session deletion."""
    
    message: str
    session_id: str
