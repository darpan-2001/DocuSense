# DOCU SENSE

A production-style Retrieval Augmented Generation (RAG) backend application for document understanding and question answering with source citations.

## Features

- **Multi-format Document Support**: Upload PDF, TXT, DOCX, and Markdown files
- **Semantic Chunking**: Intelligent document segmentation using embeddings
- **Hybrid Retrieval**: Combines dense vector search (Qdrant) with sparse BM25 retrieval
- **Query Enhancement**: HyDE (Hypothetical Document Embeddings) for improved retrieval
- **Cross-Encoder Reranking**: Re-ranks retrieved results for better relevance
- **Conversation Memory**: Session-based conversational context with configurable history
- **Source Citations**: Grounded responses with document and page references
- **Modular Architecture**: SOLID principles with dependency injection
- **Production Ready**: Comprehensive logging, error handling, and type hints

## Architecture

```
Upload Documents → Document Loader → Parser → Metadata Extraction → Text Cleaning
→ Semantic Chunking → HuggingFace Embeddings → Qdrant Vector DB

User Question → Conversation Memory → HyDE Query Generation → Hybrid Retrieval
(Dense + BM25) → Cross Encoder Reranker → Context Builder → Prompt Construction
→ Groq Llama-3.3-70B → Grounded Response → Update Memory
```

## Tech Stack

- **Backend**: FastAPI
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace (BAAI/bge-base-en-v1.5)
- **Vector Database**: Qdrant
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

## Prerequisites

- Python 3.9+
- Qdrant server running locally or remotely
- Groq API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd docu_sense
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

5. Start Qdrant server:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### Upload Documents

**POST** `/documents`

Upload and index documents for retrieval.

**Request**: multipart/form-data with files
- `files`: List of document files (PDF, TXT, DOCX, MD)

**Response**:
```json
{
  "session_id": "uuid-string",
  "message": "Successfully processed X documents",
  "documents_processed": 2,
  "total_chunks": 45
}
```

### Chat

**POST** `/chat`

Ask questions about uploaded documents.

**Request**:
```json
{
  "session_id": "uuid-string",
  "query": "Explain the retrieval stage."
}
```

**Response**:
```json
{
  "answer": "The retrieval stage involves...",
  "sources": [
    {
      "document_name": "document.pdf",
      "page_number": "3",
      "chunk_id": "uuid",
      "relevance_score": 0.95
    }
  ]
}
```

### Delete Session

**DELETE** `/session/{session_id}`

Delete a session and all associated data.

**Response**:
```json
{
  "message": "Session deleted successfully",
  "session_id": "uuid-string"
}
```

## Project Structure

```
docu_sense/
├── api/
│   ├── documents.py      # Document upload endpoint
│   ├── chat.py           # Chat endpoint
│   ├── session.py        # Session management endpoint
│   └── main.py           # FastAPI application
├── config/
│   ├── settings.py       # Application settings
│   └── logging.py        # Logging configuration
├── src/
│   ├── ingestion/        # Document loaders
│   ├── preprocessing/    # Text cleaning and metadata
│   ├── chunking/         # Semantic chunking
│   ├── embeddings/       # Embedding model
│   ├── vectordb/         # Qdrant store
│   ├── retrieval/        # Retrieval components
│   ├── query_processing/ # HyDE generator
│   ├── reranking/        # Cross-encoder reranker
│   ├── memory/           # Conversation memory
│   ├── context/          # Context builder
│   ├── prompts/          # RAG prompts
│   ├── llm/              # Groq client
│   ├── pipeline/         # Indexing and RAG pipelines
│   └── schemas/          # Pydantic schemas
├── tests/
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

## Advanced RAG Concepts Demonstrated

1. **Semantic Chunking**: Uses embeddings to create meaningful chunks instead of fixed-size splits
2. **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve retrieval
3. **Hybrid Retrieval**: Combines dense (semantic) and sparse (BM25) retrieval with weighted fusion
4. **Cross-Encoder Reranking**: Re-ranks retrieved results for better precision
5. **Session-based Memory**: Maintains conversation history separate from vector database
6. **Metadata Preservation**: Tracks document name, page number, and chunk ID throughout pipeline
7. **Source Citations**: Returns grounded responses with document references

## Configuration

Key configuration options in `.env`:

- `GROQ_API_KEY`: Your Groq API key
- `QDRANT_HOST`: Qdrant server host (default: localhost)
- `QDRANT_PORT`: Qdrant server port (default: 6333)
- `EMBEDDING_MODEL`: HuggingFace embedding model
- `LLM_MODEL`: Groq LLM model
- `DENSE_WEIGHT`: Weight for dense retrieval (default: 0.7)
- `BM25_WEIGHT`: Weight for BM25 retrieval (default: 0.3)
- `MAX_CONVERSATION_TURNS`: Maximum conversation turns to keep (default: 10)

## Testing with Swagger UI

1. Start the application
2. Navigate to `http://localhost:8000/docs`
3. Use the interactive UI to test endpoints:
   - Upload documents via `/documents`
   - Ask questions via `/chat`
   - Delete sessions via `/session/{session_id}`

## Testing with Postman

1. Import the API endpoints
2. Set up a POST request to `http://localhost:8000/documents`
3. Add files in form-data
4. Use the returned `session_id` for chat requests
5. Send POST to `http://localhost:8000/chat` with JSON body

## License

MIT License
