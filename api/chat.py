from fastapi import APIRouter, HTTPException
from src.pipeline.rag_pipeline import RAGPipeline
from src.memory.memory_manager import MemoryManager
from src.schemas.chat_schema import ChatRequest, ChatResponse
from config.logging import log

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize RAG pipeline and memory manager
memory_manager = MemoryManager()
rag_pipeline = RAGPipeline(memory_manager=memory_manager)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about uploaded documents.
    
    Uses hybrid retrieval (dense + BM25), cross-encoder reranking,
    and LLM to generate grounded answers with source citations.
    
    Args:
        request: Chat request with session_id and query
        
    Returns:
        Answer with source citations
    """
    log.info(f"Chat request for session {request.session_id}: {request.query[:50]}...")
    
    try:
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(
            query=request.query,
            session_id=request.session_id
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        log.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
