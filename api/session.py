from fastapi import APIRouter, HTTPException
from src.pipeline.rag_pipeline import RAGPipeline
from src.memory.memory_manager import MemoryManager
from src.vectordb.chroma_store import ChromaStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.schemas.response_schema import SessionDeleteResponse
from config.logging import log

router = APIRouter(prefix="/session", tags=["session"])

# Initialize components
memory_manager = MemoryManager()
chroma_store = ChromaStore()
bm25_retriever = BM25Retriever()


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """
    Delete a session and all associated data.
    
    Deletes:
    - Conversation memory
    - ChromaDB collection
    - BM25 index
    
    Args:
        session_id: Session identifier to delete
        
    Returns:
        Confirmation message
    """
    log.info(f"Deleting session: {session_id}")
    
    try:
        # Clear conversation memory
        memory_manager.clear_memory(session_id)
        
        # Delete ChromaDB collection
        chroma_store.delete_collection(session_id)
        
        # Delete BM25 index
        bm25_retriever.delete_session(session_id)
        
        return SessionDeleteResponse(
            message="Session deleted successfully",
            session_id=session_id
        )
        
    except Exception as e:
        log.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
