"""
=============================================================================
Oracle RAG Service - Main API
=============================================================================

🎓 CONCEPT: Retrieval-Augmented Generation (RAG)

Reference: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
           (Lewis et al., 2020 - Facebook AI Research)

Traditional LLMs have a knowledge cutoff and can hallucinate facts.
RAG solves this by:

1. RETRIEVE: Find relevant documents from a vector database
2. AUGMENT: Add these documents to the LLM prompt as context
3. GENERATE: LLM answers using the provided context

This is particularly powerful for our use case because:
- Movie dialogue changes (new releases, updates)
- We need precise timestamp-based answers
- Reduces hallucination by grounding responses in actual subtitles

=============================================================================
"""

import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain imports for RAG pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================


class QueryRequest(BaseModel):
    """
    Request model for the /ask endpoint.
    
    🎓 WHY PYDANTIC?
    Pydantic provides runtime type validation, which means:
    - Invalid requests fail fast with clear error messages
    - Automatic OpenAPI documentation generation
    - IDE autocomplete and type hints
    """
    query: str = Field(
        ..., 
        min_length=3,
        max_length=1000,
        description="The question to ask about the movie",
        examples=["What did the protagonist say about love?"]
    )
    movie_id: Optional[str] = Field(
        None,
        description="Filter by specific movie ID"
    )
    timestamp_start: Optional[float] = Field(
        None,
        ge=0,
        description="Start timestamp in seconds (for time-travel queries)"
    )
    timestamp_end: Optional[float] = Field(
        None,
        ge=0,
        description="End timestamp in seconds"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )


class SourceChunk(BaseModel):
    """Represents a retrieved subtitle chunk."""
    content: str
    movie_id: str
    timestamp_start: float
    timestamp_end: float
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: List[SourceChunk]
    model_used: str
    query_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    chroma_collections: int
    embedding_model: str


# =============================================================================
# Global State (initialized on startup)
# =============================================================================

# These will be initialized in the lifespan context manager
vectorstore: Optional[Chroma] = None
llm: Optional[Ollama] = None
embeddings: Optional[HuggingFaceEmbeddings] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🎓 CONCEPT: Application Lifespan Events
    
    FastAPI's lifespan context manager handles startup and shutdown.
    This is where we initialize expensive resources like:
    - Database connections
    - ML model loading
    - Vector store connections
    
    This pattern ensures resources are properly cleaned up on shutdown.
    """
    global vectorstore, llm, embeddings
    
    logger.info("🚀 Starting Oracle RAG Service...")
    
    # Initialize embeddings model (runs locally)
    # -------------------------------------------------------------------------
    # 🎓 CONCEPT: Sentence Embeddings
    # 
    # all-MiniLM-L6-v2 is a lightweight model (80MB) that converts text to
    # 384-dimensional vectors. Similar texts have similar vectors.
    # 
    # Why this model?
    # - Fast inference (~14k sentences/second on CPU)
    # - Good quality (trained on 1B+ sentence pairs)
    # - Small size (fits in Docker container easily)
    # -------------------------------------------------------------------------
    logger.info(f"📊 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize ChromaDB vector store
    # -------------------------------------------------------------------------
    # 🎓 CONCEPT: Vector Databases
    # 
    # Unlike traditional DBs that search by exact match or keywords,
    # vector DBs find "semantically similar" content using cosine similarity.
    # 
    # ChromaDB stores:
    # - The text content
    # - Its vector embedding
    # - Metadata (movie_id, timestamps) for filtering
    # -------------------------------------------------------------------------
    logger.info(f"🗄️ Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    
    try:
        vectorstore = Chroma(
            collection_name="movie_subtitles",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(f"✅ ChromaDB connected. Documents: {vectorstore._collection.count()}")
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB initialization warning: {e}")
        # Create new collection if it doesn't exist
        vectorstore = Chroma(
            collection_name="movie_subtitles",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    
    # Initialize Ollama LLM
    # -------------------------------------------------------------------------
    # 🎓 CONCEPT: Local LLMs with Ollama
    # 
    # Ollama runs open-source models locally. Benefits:
    # - No API costs
    # - Data privacy (nothing leaves your machine)
    # - No rate limits
    # 
    # llama3:8b is a good balance of quality and speed.
    # -------------------------------------------------------------------------
    logger.info(f"🤖 Connecting to Ollama at: {OLLAMA_BASE_URL}")
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,  # Lower = more deterministic responses
    )
    
    try:
        # Test LLM connection
        llm.invoke("Hello")
        logger.info(f"✅ Ollama connected with model: {LLM_MODEL}")
    except Exception as e:
        logger.warning(f"⚠️ Ollama not available: {e}. Running in mock mode.")
        llm = None
    
    logger.info("✅ Oracle RAG Service ready!")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Oracle RAG Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Oracle RAG Service",
    description="""
    🔮 **The Oracle** - Movie Intelligence through RAG
    
    This service answers questions about movies using their subtitle transcripts.
    It implements Retrieval-Augmented Generation (RAG) to provide accurate,
    grounded responses with source citations.
    
    ## Key Features
    
    - **Semantic Search**: Find relevant dialogue even with paraphrased queries
    - **Time-Travel Queries**: Ask about specific timestamps in the movie
    - **Source Citations**: Every answer includes the original subtitle chunks
    
    ## How It Works
    
    1. Your query is converted to a vector embedding
    2. Similar subtitle chunks are retrieved from ChromaDB
    3. Retrieved chunks + your query are sent to the LLM
    4. LLM generates an answer grounded in the actual movie dialogue
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for container orchestration.
    
    Returns the status of all dependencies (Ollama, ChromaDB).
    """
    ollama_ok = llm is not None
    collection_count = 0
    
    if vectorstore:
        try:
            collection_count = vectorstore._collection.count()
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        chroma_collections=collection_count,
        embedding_model=EMBEDDING_MODEL,
    )


@app.post("/ask", response_model=QueryResponse, tags=["RAG"])
async def ask_oracle(request: QueryRequest):
    """
    🔮 Ask the Oracle a question about movie dialogues.
    
    This endpoint implements the full RAG pipeline:
    1. **Retrieve** relevant subtitle chunks from the vector database
    2. **Augment** the prompt with retrieved context
    3. **Generate** an answer using the LLM
    
    ## Example Usage
    
    ```python
    import requests
    
    response = requests.post("http://localhost:8001/ask", json={
        "query": "What did they say about the treasure?",
        "movie_id": "inception",
        "timestamp_start": 3600,  # After 1 hour
        "timestamp_end": 5400,    # Until 1.5 hours
        "top_k": 5
    })
    ```
    """
    import time
    start_time = time.time()
    
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please run ingest.py first."
        )
    
    # Build metadata filter for ChromaDB
    # -------------------------------------------------------------------------
    # 🎓 CONCEPT: Metadata Filtering
    # 
    # ChromaDB allows filtering by metadata BEFORE semantic search.
    # This is crucial for our "time-travel" feature - we can restrict
    # search to a specific time window in the movie.
    # -------------------------------------------------------------------------
    where_filter = {}
    where_document_filter = None
    
    if request.movie_id:
        where_filter["movie_id"] = request.movie_id
    
    if request.timestamp_start is not None:
        where_filter["timestamp_start"] = {"$gte": request.timestamp_start}
    
    if request.timestamp_end is not None:
        where_filter["timestamp_end"] = {"$lte": request.timestamp_end}
    
    # Retrieve relevant chunks
    # -------------------------------------------------------------------------
    logger.info(f"🔍 Searching for: '{request.query[:50]}...' (top_k={request.top_k})")
    
    try:
        results = vectorstore.similarity_search_with_score(
            query=request.query,
            k=request.top_k,
            filter=where_filter if where_filter else None,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    if not results:
        return QueryResponse(
            answer="I couldn't find any relevant dialogue for your query. "
                   "Try broadening your search or check if the movie has been ingested.",
            sources=[],
            model_used=LLM_MODEL if llm else "none",
            query_time_ms=(time.time() - start_time) * 1000,
        )
    
    # Format sources
    sources = []
    context_chunks = []
    
    for doc, score in results:
        metadata = doc.metadata
        sources.append(SourceChunk(
            content=doc.page_content,
            movie_id=metadata.get("movie_id", "unknown"),
            timestamp_start=metadata.get("timestamp_start", 0),
            timestamp_end=metadata.get("timestamp_end", 0),
            relevance_score=float(1 - score),  # Convert distance to similarity
        ))
        context_chunks.append(doc.page_content)
    
    # Generate answer using LLM
    # -------------------------------------------------------------------------
    # 🎓 CONCEPT: Prompt Engineering for RAG
    # 
    # The key to good RAG is the prompt template. We:
    # 1. Provide clear instructions about using ONLY the context
    # 2. Include the retrieved chunks as numbered sources
    # 3. Ask for citations in the answer
    # -------------------------------------------------------------------------
    if llm is None:
        # Mock response when Ollama is not available
        answer = f"[Mock Mode] Based on the retrieved context, I found {len(sources)} relevant dialogue chunks. " \
                 f"The first chunk mentions: '{sources[0].content[:100]}...'"
    else:
        # Build RAG prompt
        context = "\n\n".join([
            f"[Chunk {i+1} - {sources[i].timestamp_start:.0f}s to {sources[i].timestamp_end:.0f}s]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are the Oracle, an AI assistant that answers questions about movies based on their dialogue.

IMPORTANT: Only use information from the provided context. If the answer isn't in the context, say so.

CONTEXT (from movie subtitles):
{context}

USER QUESTION: {request.query}

Provide a helpful answer based on the context above. Reference the chunk numbers when citing specific dialogue."""

        try:
            answer = llm.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"I found relevant dialogue but couldn't generate a summary. Error: {str(e)}"
    
    query_time = (time.time() - start_time) * 1000
    logger.info(f"✅ Query completed in {query_time:.1f}ms")
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=LLM_MODEL if llm else "mock",
        query_time_ms=query_time,
    )


@app.get("/collections", tags=["Debug"])
async def list_collections():
    """List all movies that have been ingested."""
    if vectorstore is None:
        return {"collections": [], "total_documents": 0}
    
    try:
        # Get unique movie IDs from metadata
        collection = vectorstore._collection
        total = collection.count()
        
        # Sample to get movie IDs (ChromaDB doesn't have distinct query)
        if total > 0:
            sample = collection.peek(min(total, 100))
            movie_ids = set()
            for metadata in sample.get("metadatas", []):
                if metadata and "movie_id" in metadata:
                    movie_ids.add(metadata["movie_id"])
            return {"collections": list(movie_ids), "total_documents": total}
        
        return {"collections": [], "total_documents": 0}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
