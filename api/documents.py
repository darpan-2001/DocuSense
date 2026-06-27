from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.pipeline.indexing_pipeline import IndexingPipeline
from src.schemas.upload_schema import DocumentUploadResponse
from config.logging import log

router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize indexing pipeline
indexing_pipeline = IndexingPipeline()


@router.post("", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index documents.
    
    Accepts multiple document files (PDF, TXT, DOCX, MD).
    Parses, chunks, embeds, and indexes them for retrieval.
    
    Returns:
        session_id for subsequent chat operations
    """
    log.info(f"Received {len(files)} files for upload")
    
    # Validate file types
    allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
    for file in files:
        from pathlib import Path
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed types: {list(allowed_extensions)}"
            )
    
    try:
        # Process documents through indexing pipeline
        result = await indexing_pipeline.process_documents(files)
        
        return DocumentUploadResponse(
            session_id=result["session_id"],
            message=f"Successfully processed {result['documents_processed']} documents",
            documents_processed=result["documents_processed"],
            total_chunks=result["total_chunks"]
        )
        
    except Exception as e:
        log.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
