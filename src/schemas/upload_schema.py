from pydantic import BaseModel
from typing import List, Optional


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload endpoint."""
    
    session_id: str
    message: str
    documents_processed: int
    total_chunks: int


class DocumentMetadata(BaseModel):
    """Schema for document metadata."""
    
    document_name: str
    file_type: str
    page_number: Optional[str] = None
    chunk_id: str
    session_id: str
