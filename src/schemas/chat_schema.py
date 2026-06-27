from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    
    session_id: str
    query: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    
    answer: str
    sources: list
