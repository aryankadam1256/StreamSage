"""
============================================================================
Movie Discovery Assistant - RAG-Powered Recommendation Service
============================================================================

🎓 LEARNING OBJECTIVES:
This service demonstrates a complete RAG (Retrieval-Augmented Generation)
implementation for conversational movie recommendations.

By reading this code, you'll understand:
1. How RAG combines vector search + LLM generation
2. Why we separate retrieval from generation
3. How to structure prompts for better responses
4. When to use metadata filtering vs semantic search

============================================================================
🏗️ ARCHITECTURE OVERVIEW:

User Query: "Recommend a mind-bending sci-fi movie"
    ↓
[Step 1: QUERY UNDERSTANDING]
    - Parse user intent
    - Extract filters (genre, year, rating)
    ↓
[Step 2: RETRIEVAL]
    - Convert query to embedding (384-dim vector)
    - Search ChromaDB for similar movies
    - Apply metadata filters (optional)
    - Get top-K candidates (default: 5)
    ↓
[Step 3: CONTEXT AUGMENTATION]
    - Format retrieved movies as context
    - Build prompt: System + Context + User Query
    ↓
[Step 4: GENERATION]
    - Send prompt to Mistral LLM
    - LLM generates personalized response
    - Include movie titles, plots, why it matches
    ↓
Response: "I recommend Inception (2010). It's a mind-bending sci-fi..."

============================================================================
"""

import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# HuggingFace Inference API
from huggingface_hub import InferenceClient

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables (with sensible defaults)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Required: Get from huggingface.co/settings/tokens
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "../../data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# HuggingFace model choices (free on Inference API)
# Options:
#   - "mistralai/Mistral-7B-Instruct-v0.2" (recommended)
#   - "meta-llama/Meta-Llama-3-8B-Instruct" (requires approval)
#   - "microsoft/Phi-3-mini-4k-instruct" (fast, smaller)
#   - "HuggingFaceH4/zephyr-7b-beta" (good alternative)
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class DiscoverRequest(BaseModel):
    """
    Request model for movie discovery.
    
    🎓 CONCEPT: Structured Input Validation
    
    Pydantic provides:
    - Automatic type checking (query must be a string)
    - Field validation (genre must be from a list)
    - Default values (top_k defaults to 5)
    - Auto-generated OpenAPI docs
    
    This prevents bugs and makes the API self-documenting!
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language query",
        examples=["Recommend a mind-bending sci-fi movie"]
    )
    
    # Optional filters
    genre: Optional[str] = Field(
        None,
        description="Filter by genre (e.g., 'Action', 'Drama')"
    )
    
    min_year: Optional[int] = Field(
        None,
        ge=1900,
        le=2025,
        description="Minimum release year"
    )
    
    max_year: Optional[int] = Field(
        None,
        ge=1900,
        le=2025,
        description="Maximum release year"
    )
    
    min_rating: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="Minimum IMDb rating"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of movies to retrieve for context"
    )


class MovieResult(BaseModel):
    """A single movie result."""
    title: str
    year: Optional[int]
    rating: Optional[float]
    genres: Optional[str]
    director: Optional[str]
    relevance_score: float


class DiscoverResponse(BaseModel):
    """Response from the discovery endpoint."""
    query: str
    answer: str
    recommended_movies: List[MovieResult]
    model_used: str
    retrieval_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    chroma_loaded: bool
    total_movies: int
    embedding_model: str
    llm_model: str


# ============================================================================
# Global State (Initialized on Startup)
# ============================================================================

vectorstore: Optional[Chroma] = None
hf_client: Optional[InferenceClient] = None
embeddings: Optional[HuggingFaceEmbeddings] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🎓 CONCEPT: Application Lifespan Management
    
    This is where we initialize "expensive" resources that:
    - Take time to load (models, databases)
    - Should be loaded ONCE, not per-request
    - Need cleanup on shutdown
    
    Without this, we'd reload ChromaDB on EVERY request (slow!)
    """
    global vectorstore, llm, embeddings
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Movie Discovery Assistant...")
    logger.info("=" * 60)
    
    # ========================================================================
    # Step 1: Load Embedding Model
    # ========================================================================
    # 🎓 CONCEPT: Embedding Models
    #
    # The same model must be used for:
    # 1. Creating movie embeddings (done in Colab)
    # 2. Creating query embeddings (done here)
    #
    # If you use different models, vectors won't be comparable!
    # ========================================================================
    
    logger.info(f"📊 Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("   (This takes ~5 seconds on first run...)")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embedding model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise
    
    # ========================================================================
    # Step 2: Load ChromaDB Vector Store
    # ========================================================================
    # 🎓 CONCEPT: Vector Database Persistence
    #
    # ChromaDB saves everything to disk in the persist_directory.
    # On startup, it:
    # 1. Loads the embeddings (4800+ vectors)
    # 2. Loads the metadata (genres, years, etc.)
    # 3. Rebuilds the search index
    #
    # This takes ~2-3 seconds but makes searches instant!
    # ========================================================================
    
    chroma_path = Path(CHROMA_DB_PATH).resolve()
    logger.info(f"🗄️ Loading ChromaDB from: {chroma_path}")
    
    if not chroma_path.exists():
        logger.error(f"❌ ChromaDB not found at {chroma_path}")
        logger.error("   Please run the Colab notebook and extract chroma_db.zip here!")
        raise FileNotFoundError(f"ChromaDB not found: {chroma_path}")
    
    try:
        vectorstore = Chroma(
            collection_name="movies",
            embedding_function=embeddings,
            persist_directory=str(chroma_path)
        )
        
        movie_count = vectorstore._collection.count()
        logger.info(f"✅ ChromaDB loaded with {movie_count} movies!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load ChromaDB: {e}")
        raise
    
    # ========================================================================
    # Step 3: Initialize HuggingFace Inference API Client
    # ========================================================================
    # 🎓 CONCEPT: HuggingFace Inference API
    #
    # HuggingFace provides FREE API access to many models:
    # - No local downloads (saves 4GB+)
    # - No GPU needed
    # - Access to 100+ models
    # - Free tier: 1000 requests/hour
    #
    # How to get API token:
    # 1. Go to huggingface.co
    # 2. Sign up (free)
    # 3. Go to Settings → Access Tokens
    # 4. Create new token (read access)
    # 5. Set as environment variable: HF_API_TOKEN
    #
    # Available models (free):
    # - mistralai/Mistral-7B-Instruct-v0.2 (recommended)
    # - HuggingFaceH4/zephyr-7b-beta (good quality)
    # - microsoft/Phi-3-mini-4k-instruct (fast)
    # ========================================================================
    
    global hf_client
    
    logger.info(f"🤖 Initializing HuggingFace Inference API")
    logger.info(f"   Model: {LLM_MODEL}")
    
    if not HF_API_TOKEN:
        logger.warning("⚠️ HF_API_TOKEN not set!")
        logger.warning("   Get your token from: https://huggingface.co/settings/tokens")
        logger.warning("   Set it as environment variable: HF_API_TOKEN=your_token_here")
        logger.warning("   Service will run in 'retrieval-only' mode (no LLM generation)")
        hf_client = None
    else:
        try:
            # Initialize InferenceClient
            hf_client = InferenceClient(
                model=LLM_MODEL,
                token=HF_API_TOKEN
            )
            
            # Test connection with a simple query
            logger.info("   Testing API connection...")
            test_response = hf_client.text_generation(
                "Say 'ready' if you can hear me.",
                max_new_tokens=10
            )
            logger.info(f"✅ HuggingFace API connected! Test response: {test_response[:50]}")
            
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace API not available: {e}")
            logger.warning("   Service will run in 'retrieval-only' mode (no LLM generation)")
            hf_client = None
    
    logger.info("=" * 60)
    logger.info("✅ Movie Discovery Assistant is READY!")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Movie Discovery Assistant...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Movie Discovery Assistant",
    description="""
    🎬 **Intelligent Movie Recommendations via RAG**
    
    This service uses Retrieval-Augmented Generation to provide:
    - Conversational movie recommendations
    - Semantic search (understands synonyms and context)
    - Genre/year/rating filtering
    - Natural language answers
    
    ## How It Works
    
    1. Your query is converted to a vector embedding
    2. Similar movies are retrieved from ChromaDB
    3. Retrieved movies + your query are sent to Mistral LLM
    4. Mistral generates a personalized recommendation
    
    ## Example Queries
    
    - "Recommend a mind-bending sci-fi movie"
    - "I want a comedy from the 90s"
    - "Movies like The Matrix"
    - "Christopher Nolan films with high ratings"
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def build_metadata_filter(request: DiscoverRequest) -> Dict[str, Any]:
    """
    Build ChromaDB metadata filter from request parameters.
    
    🎓 CONCEPT: Metadata Filtering
    
    ChromaDB supports filtering BEFORE semantic search:
    
    Without filtering:
    - Search all 4800 movies
    - Return top-5 most similar
    
    With filtering:
    - Filter to movies matching criteria (e.g., year >= 2010)
    - Search only those movies
    - Return top-5 from filtered set
    
    This makes queries like "sci-fi from 2010-2020" much more accurate!
    """
    where_filter = {}
    
    # Genre filter (partial match)
    if request.genre:
        # ChromaDB uses $contains for substring matching
        where_filter["genres"] = {"$contains": request.genre}
    
    # Year range filter
    if request.min_year is not None or request.max_year is not None:
        year_filter = {}
        if request.min_year:
            year_filter["$gte"] = request.min_year
        if request.max_year:
            year_filter["$lte"] = request.max_year
        where_filter["year"] = year_filter
    
    # Rating filter
    if request.min_rating is not None:
        where_filter["rating"] = {"$gte": request.min_rating}
    
    return where_filter if where_filter else None


def format_movies_for_context(results: List, distances: List) -> str:
    """
    Format retrieved movies as context for the LLM.
    
    🎓 CONCEPT: Context Engineering
    
    The LLM needs structured, clear context. We format like:
    
    Movie 1: Inception (2010)
    Genres: Action, Sci-Fi
    Rating: 8.8/10
    Plot: A thief who steals corporate secrets...
    
    This format:
    - Is easy for LLM to parse
    - Provides all relevant info
    - Maintains readability
    """
    context_parts = []
    
    for i, (doc, distance) in enumerate(zip(results['documents'][0], distances[0])):
        metadata = results['metadatas'][0][i]
        
        # Calculate similarity score (1 - distance)
        similarity = 1 - distance
        
        # Format movie info
        movie_text = f"""
Movie {i+1}: {metadata.get('title', 'Unknown')}
Year: {metadata.get('year', 'N/A')}
Genres: {metadata.get('genres', 'N/A')}
Rating: {metadata.get('rating', 'N/A')}/10
Relevance: {similarity:.2%}

{doc}
"""
        context_parts.append(movie_text.strip())
    
    return "\n\n---\n\n".join(context_parts)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Kubernetes/Docker can use this to verify the service is running.
    """
    hf_ok = hf_client is not None
    chroma_ok = vectorstore is not None
    movie_count = 0
    
    if vectorstore:
        try:
            movie_count = vectorstore._collection.count()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if (hf_ok and chroma_ok) else "degraded",
        ollama_connected=hf_ok,  # Reusing field name for compatibility
        chroma_loaded=chroma_ok,
        total_movies=movie_count,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL if hf_client else "not_connected"
    )


@app.post("/discover", response_model=DiscoverResponse, tags=["Discovery"])
async def discover_movies(request: DiscoverRequest):
    """
    🎬 Discover movies based on natural language query.
    
    This is the main RAG endpoint. It:
    1. Retrieves similar movies from ChromaDB
    2. Formats them as context
    3. Sends to Mistral LLM for a natural response
    
    ## Example Request
    
    ```json
    {
        "query": "Recommend a mind-bending sci-fi thriller",
        "min_year": 2000,
        "min_rating": 7.0,
        "top_k": 5
    }
    ```
    
    ## Example Response
    
    ```json
    {
        "answer": "I highly recommend Inception (2010)...",
        "recommended_movies": [
            {"title": "Inception", "year": 2010, "rating": 8.8, ...}
        ],
        ...
    }
    ```
    """
    
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Please check server logs."
        )
    
    logger.info(f"🔍 Query: '{request.query}'")
    
    # ========================================================================
    # Step 1: RETRIEVAL - Search ChromaDB
    # ========================================================================
    # 🎓 CONCEPT: Similarity Search
    #
    # query_embeddings:
    #   - Convert query to embedding
    #   - Compare to all movie embeddings
    #   - Use cosine similarity (measures angle between vectors)
    #
    # n_results: 
    #   - How many to retrieve
    #   - More = better context but slower
    #   - 5 is a good default
    #
    # where:
    #   - Metadata filter (optional)
    #   - Applied BEFORE semantic search
    # ========================================================================
    
    try:
        # Build metadata filter
        metadata_filter = build_metadata_filter(request)
        
        if metadata_filter:
            logger.info(f"   Filters: {metadata_filter}")
        
        # Search (ChromaDB handles embedding internally)
        results = vectorstore._collection.query(
            query_texts=[request.query],
            n_results=request.top_k,
            where=metadata_filter
        )
        
        logger.info(f"   Retrieved {len(results['documents'][0])} movies")
        
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    if not results['documents'][0]:
        return DiscoverResponse(
            query=request.query,
            answer="I couldn't find any movies matching your criteria. Try broadening your search!",
            recommended_movies=[],
            model_used=LLM_MODEL if llm else "retrieval_only",
            retrieval_count=0
        )
    
    # ========================================================================
    # Step 2: FORMAT CONTEXT
    # ========================================================================
    
    context = format_movies_for_context(results, results['distances'])
    
    # ========================================================================
    # Step 3: GENERATION - Call LLM
    # ========================================================================
    # 🎓 CONCEPT: HuggingFace Text Generation API
    #
    # The HF Inference API parameters:
    #
    # max_new_tokens: How many tokens to generate (default: 512)
    #   - 1 token ≈ 0.75 words
    #   - 512 tokens ≈ 380 words (good for recommendations)
    #
    # temperature: Controls randomness
    #   - 0.0 = Deterministic (same answer every time)
    #   - 0.7 = Balanced (our choice)
    #   - 1.0 = Very creative (sometimes random)
    #
    # top_p: Nucleus sampling (diversity control)
    #   - 0.9 = Consider top 90% probable tokens
    #   - Higher = more diverse, lower = more focused
    #
    # repetition_penalty: Prevents repetitive text
    #   - 1.0 = No penalty
    #   - 1.2 = Slight penalty (recommended)
    # ========================================================================
    
    if hf_client is None:
        # Fallback: Return retrieval results without LLM
        answer = "LLM not available. Here are the top matching movies:\n\n"
        for i, meta in enumerate(results['metadatas'][0][:3]):
            answer += f"{i+1}. {meta['title']} ({meta.get('year', 'N/A')})\n"
    else:
        # Build RAG prompt
        prompt = f"""You are a helpful movie recommendation assistant. Your job is to recommend movies based on the user's preferences.

You have access to information about the following movies (sorted by relevance):

{context}

User Query: {request.query}

Please provide a helpful recommendation. Include:
1. Which movie(s) you recommend and why
2. Brief plot summary
3. Why it matches their request
4. Any relevant details (director, cast, ratings)

Be conversational and enthusiastic! Cite the movie titles from the context above.
"""
        
        try:
            logger.info("🤖 Generating response with HuggingFace API...")
            
            # Call HuggingFace Inference API
            answer = hf_client.text_generation(
                prompt,
                max_new_tokens=512,  # ~380 words
                temperature=0.7,     # Balanced creativity
                top_p=0.9,          # Nucleus sampling
                repetition_penalty=1.2,  # Reduce repetition
                return_full_text=False  # Only return generated text, not prompt
            )
            
            logger.info("✅ Response generated!")
            
        except Exception as e:
            logger.error(f"❌ HuggingFace API error: {e}")
            answer = f"Error generating response: {str(e)}\n\nHowever, here are the top matching movies:\n\n"
            for i, meta in enumerate(results['metadatas'][0][:3]):
                answer += f"{i+1}. {meta['title']} ({meta.get('year', 'N/A')})\n"
    
    # ========================================================================
    # Step 4: FORMAT RESPONSE
    # ========================================================================
    
    # Extract movie results
    movies = []
    for i, meta in enumerate(results['metadatas'][0]):
        distance = results['distances'][0][i]
        movies.append(MovieResult(
            title=meta.get('title', 'Unknown'),
            year=meta.get('year'),
            rating=meta.get('rating'),
            genres=meta.get('genres'),
            director=meta.get('director'),
            relevance_score=round(1 - distance, 3)
        ))
    
    return DiscoverResponse(
        query=request.query,
        answer=answer,
        recommended_movies=movies,
        model_used=LLM_MODEL if hf_client else "retrieval_only",
        retrieval_count=len(movies)
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🎬 Starting Movie Discovery Assistant")
    print("=" * 60)
    print(f"📍 API will be available at: http://localhost:8002")
    print(f"📚 Docs will be at: http://localhost:8002/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
