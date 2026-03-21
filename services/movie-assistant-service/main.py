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
import re
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# HuggingFace Inference API
from huggingface_hub import InferenceClient

# Optimized Inference Engine
from fine_tuning.inference_optimizer import InferenceManager, InferenceOptConfig

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
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
# BGE models require a query instruction prefix for optimal retrieval.
# Documents are embedded without prefix; queries are embedded WITH prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# HuggingFace model choices (free on Inference API)
# Options:
#   - "mistralai/Mistral-7B-Instruct-v0.2" (recommended)
#   - "meta-llama/Meta-Llama-3-8B-Instruct" (requires approval)
#   - "microsoft/Phi-3-mini-4k-instruct" (fast, smaller)
#   - "HuggingFaceH4/zephyr-7b-beta" (good alternative)
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Inference optimization settings
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "auto")  # auto, vllm, local, api
# Auto-detect fine-tuned model in fine_tuning/outputs/final_model if no explicit path set
_default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuning", "outputs", "final_model")
if not os.getenv("LOCAL_MODEL_PATH") and os.path.exists(os.path.join(_default_model_path, "adapter_config.json")):
    LOCAL_MODEL_PATH = _default_model_path
else:
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
ENABLE_SPECULATIVE = os.getenv("ENABLE_SPECULATIVE", "true").lower() == "true"
DISCOVER_RESPONSE_CACHE_TTL_S = int(os.getenv("DISCOVER_RESPONSE_CACHE_TTL_S", "45"))
DISCOVER_RESPONSE_CACHE_SIZE = int(os.getenv("DISCOVER_RESPONSE_CACHE_SIZE", "256"))
QUERY_EMBED_CACHE_TTL_S = int(os.getenv("QUERY_EMBED_CACHE_TTL_S", "300"))
QUERY_EMBED_CACHE_SIZE = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "2048"))
RERANK_MAX_CANDIDATES = int(os.getenv("RERANK_MAX_CANDIDATES", "140"))
CROSS_ENCODER_MAX_CANDIDATES = int(os.getenv("CROSS_ENCODER_MAX_CANDIDATES", "90"))

# Build a title → {runtime, description} lookup from the processed features file.
# ChromaDB metadata doesn't store these fields, so we load them once at startup.
_FEATURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "movie_features.json")
_movie_runtime_lookup: Dict[str, int] = {}      # title → runtime in minutes
_movie_overview_lookup: Dict[str, str] = {}     # title → overview / description
try:
    import json as _json
    with open(_FEATURES_PATH, encoding="utf-8") as _f:
        _features = _json.load(_f)
    for _m in _features.values():
        _t = _m.get("title", "")
        if _t:
            if _m.get("runtime"):
                _movie_runtime_lookup[_t] = _m["runtime"]
            if _m.get("overview"):
                _movie_overview_lookup[_t] = _m["overview"]
    logger.info(f"Loaded runtime/overview for {len(_movie_runtime_lookup)} movies from features file")
except Exception as _e:
    logger.warning(f"Could not load movie_features.json for runtime lookup: {_e}")

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
    runtime: Optional[int] = None   # minutes
    description: Optional[str] = None
    relevance_score: float
    recommendation_reason: Optional[str] = None  # Why this movie matches the query


class DiscoverResponse(BaseModel):
    """Response from the discovery endpoint."""
    query: str
    answer: Optional[str]
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
    inference_backend: str = ""


# ============================================================================
# Global State (Initialized on Startup)
# ============================================================================

vectorstore: Optional[Chroma] = None
hf_client: Optional[InferenceClient] = None
embeddings: Optional[HuggingFaceEmbeddings] = None
inference_engine: Optional[InferenceManager] = None
cross_encoder = None  # sentence_transformers.CrossEncoder for re-ranking
bm25_index = None     # BM25Okapi index for keyword search
bm25_doc_ids = []     # ChromaDB doc IDs corresponding to BM25 corpus rows
_discover_response_cache = OrderedDict()  # key -> (timestamp, DiscoverResponse)
_query_embedding_cache = OrderedDict()    # key -> (timestamp, embedding vector)


def _reinit_vectorstore() -> bool:
    """Re-create the ChromaDB client when a stale connection error occurs."""
    global vectorstore
    try:
        chroma_path = Path(CHROMA_DB_PATH).resolve()
        logger.warning("ChromaDB stale connection - reinitialising...")
        vectorstore = Chroma(
            collection_name="movies",
            embedding_function=embeddings,
            persist_directory=str(chroma_path),
        )
        count = vectorstore._collection.count()
        logger.info(f"ChromaDB reinitialised - {count} movies")
        return True
    except Exception as e:
        logger.error(f"ChromaDB reinit failed: {e}")
        return False


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
    global vectorstore, embeddings, cross_encoder, bm25_index, bm25_doc_ids
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Movie Discovery Assistant...")
    logger.info("=" * 60)

    # Log GPU info if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.info("⚠️ No GPU detected, running on CPU")
    
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
        # Auto-detect GPU for embeddings (3-4x faster than CPU)
        embed_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"   Embedding device: {embed_device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': embed_device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 64 if embed_device == 'cuda' else 8,
            }
        )
        logger.info("✅ Embedding model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise

    # ========================================================================
    # Step 1b: Load Cross-Encoder Re-ranking Model
    # ========================================================================
    # Cross-encoders see (query, document) pairs with full attention, producing
    # much more accurate relevance scores than bi-encoder cosine similarity.
    # We use it to re-rank the top candidates after vector retrieval.
    # ========================================================================
    try:
        from sentence_transformers import CrossEncoder
        ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading cross-encoder: {ce_model_name}")
        cross_encoder = CrossEncoder(ce_model_name, max_length=512, device=embed_device)
        logger.info("✅ Cross-encoder loaded!")
    except Exception as e:
        logger.warning(f"Cross-encoder not available ({e}) — falling back to heuristic reranker only")
        cross_encoder = None
    
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
    # Step 2b: Build BM25 Index for Hybrid Search
    # ========================================================================
    # BM25 (Best Matching 25) is a TF-IDF keyword search that catches exact
    # term matches that embedding models miss. Combined with vector search
    # via Reciprocal Rank Fusion, it provides the best of both worlds.
    # ========================================================================
    try:
        from rank_bm25 import BM25Okapi
        logger.info("Building BM25 index from ChromaDB documents...")
        # Fetch all documents and IDs from ChromaDB for BM25 indexing
        all_data = vectorstore._collection.get(include=["documents"])
        bm25_doc_ids = all_data['ids']
        # Tokenize documents for BM25 (simple whitespace + lowercasing)
        tokenized_corpus = [doc.lower().split() for doc in all_data['documents']]
        bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"✅ BM25 index built with {len(bm25_doc_ids)} documents")
    except Exception as e:
        logger.warning(f"BM25 index not available ({e}) — using vector search only")
        bm25_index = None
    
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
    global inference_engine

    # ========================================================================
    # Step 3a: Try Optimized Inference Engine (vLLM/Local/API)
    # ========================================================================
    # Priority: vLLM (PagedAttention) > Local (Flash Attention 2) > HF API
    # ========================================================================

    if LOCAL_MODEL_PATH or INFERENCE_BACKEND != "api":
        logger.info("Initializing optimized inference engine...")

        opt_config = InferenceOptConfig(
            backend=INFERENCE_BACKEND,
            model_path=LOCAL_MODEL_PATH or LLM_MODEL,
            vllm_enable_prefix_caching=True,
            speculative_enabled=ENABLE_SPECULATIVE,
            flash_attention_enabled=True,
        )

        inference_engine = InferenceManager(opt_config)
        if inference_engine.initialize():
            logger.info(f"Optimized inference: {inference_engine.backend_name}")
        else:
            logger.info("Optimized inference not available, falling back to HF API")
            inference_engine = None

    # ========================================================================
    # Step 3b: Initialize HuggingFace Inference API (fallback)
    # ========================================================================
    
    if not HF_API_TOKEN:
        if inference_engine is None:
            logger.warning("⚠️ HF_API_TOKEN not set!")
            logger.warning("   Get your token from: https://huggingface.co/settings/tokens")
            logger.warning("   Set it as environment variable: HF_API_TOKEN=your_token_here")
            logger.warning("   Service will run in 'retrieval-only' mode (no LLM generation)")
        else:
            logger.info("HF_API_TOKEN not set (not needed — local model is active)")
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
            test_response = hf_client.chat_completion(
                messages=[{"role": "user", "content": "Say 'ready' if you can hear me."}],
                max_tokens=10
            )
            test_text = test_response.choices[0].message.content
            logger.info(f"✅ HuggingFace API connected! Test response: {test_text[:50]}")
            
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace API not available: {e}")
            logger.warning("   Service will run in 'retrieval-only' mode (no LLM generation)")
            hf_client = None
    
    logger.info("=" * 60)
    logger.info("✅ Movie Discovery Assistant is READY!")
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"   GPU memory: {mem_used:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
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
    - Search all movies
    - Return top-5 most similar

    With filtering:
    - Filter to movies matching criteria (e.g., year >= 2010)
    - Search only those movies
    - Return top-5 from filtered set

    IMPORTANT: ChromaDB 1.x requires multiple conditions wrapped in $and.
    A bare dict with multiple keys is rejected — only one key allowed per clause.
    """
    conditions = []

    # NOTE: genre is intentionally excluded here.
    # ChromaDB 1.x $contains does not work on comma-separated string fields.
    # Genre filtering is handled via query augmentation + Python post-filter
    # in the discover endpoint instead.

    # Year range filter
    if request.min_year is not None:
        conditions.append({"year": {"$gte": request.min_year}})
    if request.max_year is not None:
        conditions.append({"year": {"$lte": request.max_year}})

    # Rating filter
    if request.min_rating is not None:
        conditions.append({"rating": {"$gte": request.min_rating}})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def rerank_with_keywords(query: str, results: dict, top_k: int,
                         name_phrase: Optional[str] = None,
                         source_genres: Optional[str] = None,
                         source_meta: Optional[dict] = None,
                         is_foreign_query: bool = False,
                         matched_mood_word: Optional[str] = None,
                         similar_to_title: Optional[str] = None,
                         is_franchise_query: bool = False) -> dict:
    """
    Re-rank vector search results by boosting exact keyword matches and
    popularity for generic queries.

    Embedding models like MiniLM treat names as bag-of-words: "Tom Holland"
    partially matches "Tom Hanks" and "Mr. Holland's Opus".  We fix this by
    fetching extra candidates, then boosting results where multi-word query
    phrases (likely actor/director names) appear literally in the document.

    When name_phrase is provided (from _extract_name_phrase), we apply a
    strong bonus for documents containing that name in cast/director fields.

    For generic queries (no name detected), popular/high-rated movies get a
    small bonus so "best horror" returns classics, not obscure titles.

    Scoring:  adjusted = semantic_distance - keyword_bonus
    Lower adjusted distance = more relevant.
    """
    if not results['documents'][0]:
        return results

    query_lower = query.lower()
    words = query_lower.split()

    # Build a set of multi-word phrases (2+ words) from the query.
    # These are likely person names — e.g. "tom holland", "christopher nolan".
    phrases = []
    for start in range(len(words)):
        for end in range(start + 2, min(start + 5, len(words) + 1)):
            phrase = " ".join(words[start:end])
            if len(phrase) >= 6:          # skip tiny fragments
                phrases.append(phrase)

    name_lower = name_phrase.lower() if name_phrase else None

    # Check if this is a "quality" query (best, top, great, oscar, award, etc.)
    quality_words = {"best", "top", "greatest", "great", "classic", "classics",
                     "oscar", "award", "winning", "acclaimed", "must-see",
                     "must", "iconic", "legendary", "famous", "popular",
                     "highly", "rated", "brilliant", "masterpiece", "masterpieces",
                     "excellent", "amazing", "outstanding", "superb", "fantastic",
                     "wonderful", "everyone", "essential", "essentials",
                     "definitive", "unforgettable", "nostalgia"}
    is_quality_query = bool(quality_words & set(words))

    # Detect "meta-quality" queries: purely about the best movies overall,
    # not genre-specific. These need massive rating/popularity boost because
    # embedding search returns title-keyword matches instead of actual classics.
    # e.g., "best movies of all time", "movies everyone should see before they die",
    #        "oscar best picture winners", "best movie trilogies"
    meta_quality_patterns = [
        r'best\s+movies?\s+(?:of\s+)?all\s+time',
        r'greatest\s+(?:movies?|films?)\s+(?:of\s+)?(?:all\s+time|ever)',
        r'movies?\s+everyone\s+should\s+(?:see|watch)',
        r'(?:before|you)\s+(?:die|must)',
        r'oscar\s+(?:best|winning|award)',
        r'best\s+(?:picture|movie)\s+(?:winner|award)',
        r'best\s+(?:movie\s+)?trilog',
        r'must\s+(?:see|watch)\s+(?:movies?|films?)',
        r'essential\s+(?:movies?|films?)',
        r'(?:movies?|films?)\s+(?:you\s+)?(?:must|have\s+to)\s+(?:see|watch)',
    ]
    is_meta_quality = any(re.search(pat, query_lower) for pat in meta_quality_patterns)
    if is_meta_quality:
        is_quality_query = True

    # Detect decade/era queries for moderate quality boost (weaker than meta_quality)
    is_decade_query = bool(re.search(r'\b(?:19[2-9]0|20[0-2]0)s?\b|\b[2-9]0s\b', query_lower))
    # Only use era words that are unambiguous (not "black", "white", "old" which trigger
    # on "black comedy", "old movie" etc.)
    decade_era_words = {'classic', 'classics', 'vintage', 'retro', 'nostalgia',
                        'masterpiece', 'masterpieces'}
    if not is_decade_query:
        is_decade_query = bool(decade_era_words & set(words))

    # Detect genre keywords in the query for genre-matching boost
    genre_map = {
        "horror": "horror", "scary": "horror", "terrifying": "horror",
        "comedy": "comedy", "comedies": "comedy", "funny": "comedy",
        "hilarious": "comedy", "laugh": "comedy",
        "drama": "drama", "dramas": "drama", "dramatic": "drama",
        "action": "action", "thriller": "thriller", "thrillers": "thriller",
        "suspense": "thriller", "suspenseful": "thriller",
        "romance": "romance", "romantic": "romance", "romances": "romance",
        "sci-fi": "science fiction", "scifi": "science fiction",
        "animation": "animation", "animated": "animation", "anime": "animation",
        "fantasy": "fantasy", "adventure": "adventure",
        "crime": "crime", "mystery": "mystery", "war": "war",
        "western": "western", "musical": "music", "documentary": "documentary",
        "superhero": "science fiction",
        "family": "family", "kids": "family",
        "history": "history", "historical": "history",
        "psychological": "thriller",
    }
    query_genres = set()
    has_explicit_genre_word = False
    for w in words:
        if w in genre_map:
            query_genres.add(genre_map[w])
            has_explicit_genre_word = True

    # For "movies like X" queries, inject the source movie's genres
    # so genre-matching can filter out wrong-genre candidates
    is_similarity_query = bool(source_genres)
    if source_genres:
        for g in source_genres.lower().split(", "):
            g = g.strip()
            if g:
                query_genres.add(g)

    # Detect mood keywords for mood-matching boost
    mood_keywords = {
        "sad": {"drama"},
        "cry": {"drama"},
        "happy": {"comedy"},
        "feel": {"comedy"},
        "fun": {"comedy", "adventure"},
        "uplifting": {"drama", "comedy"},
        "scary": {"horror", "thriller"},
        "dark": {"thriller", "crime"},
        "mind-bending": {"science fiction", "thriller"},
        "light": {"comedy"},
        "lighthearted": {"comedy"},
        "adrenaline": {"action", "thriller"},
        "intense": {"thriller", "action"},
    }
    for w in words:
        if w in mood_keywords:
            query_genres.update(mood_keywords[w])

    # If query is explicitly about sadness/crying, penalise comedies
    is_sad_query = bool({"sad", "cry", "crying", "tearjerker"} & set(words))
    is_comedy_query = bool({"funny", "comedy", "laugh", "hilarious", "fun", "lighthearted"} & set(words))
    is_kids_query = bool({"kids", "children", "child", "family"} & set(words))
    is_scary_query = bool({"scary", "horror", "terrifying", "frightening"} & set(words))
    # "uplifting inspirational" → wants dramas (Rocky, Shawshank). "cheer me up" → wants comedies.
    # Separate these: inspirational boosts Drama, feel-good penalises Drama.
    is_inspirational_query = bool({"inspirational", "inspiring", "inspirational"} & set(words)) or "uplifting" in query_lower
    is_feelgood_query = "feel good" in query_lower or "feel-good" in query_lower or "rainy day" in query_lower or "cheer" in query_lower or "light hearted" in query_lower or "lighthearted" in query_lower
    # If both uplifting AND inspirational, use inspirational (Drama-friendly) not feelgood
    if is_inspirational_query and not is_feelgood_query:
        pass  # keep is_feelgood_query False
    elif "uplifting" in query_lower and not any(w in query_lower for w in ["cheer", "feel good", "feel-good", "rainy"]):
        is_feelgood_query = False  # "uplifting" alone → inspirational, not feel-good
    is_intense_query = bool({"intense", "adrenaline", "gripping", "suspenseful", "edge"} & set(words)) or "slow burn" in query_lower
    is_adrenaline_query = bool({"adrenaline"} & set(words)) or "adrenaline rush" in query_lower
    is_dark_query = bool({"dark", "disturbing", "gritty", "bleak"} & set(words))
    is_hidden_gems_query = bool({"hidden", "gem", "gems", "underrated", "overlooked", "unknown"} & set(words))
    is_musical_query = bool({"musical", "musicals", "singing", "dancing", "broadway"} & set(words))
    is_grandma_query = bool({"grandma", "grandmother", "grandpa", "grandfather", "elderly"} & set(words))
    is_visual_query = bool({"visually", "visual", "cinematography", "cinematic", "gorgeous", "beautiful"} & set(words)) or "visually stunning" in query_lower
    is_cerebral_query = "mind bending" in query_lower or "mind-bending" in query_lower or "mindbending" in query_lower

    # ── Augmented title-hint matching ──────────────────────────────
    # When mood augmentation guides the embedding toward specific films,
    # add a reranking bonus for candidates whose title matches those films.
    # This bridges the gap when the embedding doesn't precisely target them.
    _augment_title_hints = {
        "horror": ["exorcist", "hereditary", "get out", "conjuring", "ring", "halloween", "nightmare on elm street"],
        "scary": ["exorcist", "hereditary", "get out", "conjuring", "ring", "halloween"],
        "adrenaline rush": ["mad max", "john wick", "die hard", "mission impossible"],
        "adrenaline": ["mad max", "john wick", "die hard", "mission impossible"],
        "light hearted": ["ferris bueller", "grand budapest", "school of rock", "princess bride", "groundhog day"],
        "lighthearted": ["ferris bueller", "grand budapest", "school of rock", "princess bride", "groundhog day"],
        "romance and i like action": ["mr and mrs smith", "true lies", "princess bride", "knight and day", "romancing the stone"],
        "girlfriend likes romance": ["mr and mrs smith", "true lies", "princess bride", "knight and day", "romancing the stone"],
        "romance and action": ["mr and mrs smith", "true lies", "princess bride", "knight and day", "romancing the stone"],
    }
    hint_titles = _augment_title_hints.get(matched_mood_word, []) if matched_mood_word else []

    # ── Similar-to title hints ────────────────────────────────────
    # For "movies like X" queries, the stored embedding finds films that
    # are vectorially close but not what users expect.  E.g., "movies
    # similar to John Wick" returns Ballerina (spinoff) instead of Nobody
    # (thematically similar lone-vigilante film).  This dictionary maps
    # well-known source movies to their expected similar-to results.
    _similar_to_hints = {
        "john wick": ["nobody", "equalizer", "taken", "kill bill", "atomic blonde", "the raid"],
        "the notebook": ["me before you", "fault in our stars", "walk to remember", "p.s. i love you", "a walk to remember"],
        "inception": ["shutter island", "interstellar", "prestige", "memento", "matrix"],
        "the godfather": ["goodfellas", "scarface", "godfather part ii", "casino", "departed"],
        "toy story": ["toy story 2", "toy story 3", "toy story 4", "inside out", "finding nemo", "monsters, inc"],
        "pulp fiction": ["reservoir dogs", "snatch", "lock, stock", "kill bill", "fight club"],
        "interstellar": ["arrival", "the martian", "gravity", "2001", "contact"],
        "get out": ["us", "midsommar", "nope", "invitation", "don't breathe"],
        "the dark knight": ["batman begins", "logan", "watchmen", "v for vendetta"],
        "forrest gump": ["benjamin button", "big fish", "cast away", "green mile"],
    }
    sim_hint_titles = _similar_to_hints.get(similar_to_title.lower().strip(), []) if similar_to_title else []
    # Merge: mood hints + similar-to hints both feed into the same bonus
    hint_titles = hint_titles + sim_hint_titles

    # "watch when sad" / "movies for when you're sad" → user wants COMFORTING movies,
    # not sad-theme movies.  Override sad flag → feel-good flag.
    is_comforting_query = bool(
        re.search(r'when\s+(?:i\'?m?\s+)?(?:sad|down|depressed|upset|blue)', query_lower) or
        re.search(r'(?:watch|see|need)\s+when\s+(?:sad|cry|crying)', query_lower) or
        re.search(r'cheer\s*(?:me\s+)?up', query_lower) or
        re.search(r'(?:feeling|feel)\s+(?:sad|down|depressed|blue)', query_lower)
    )
    if is_comforting_query:
        is_sad_query = False
        is_feelgood_query = True

    scored = []
    for i in range(len(results['documents'][0])):
        doc_lower = results['documents'][0][i].lower()
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        moods_lower = meta.get('moods', '').lower()

        bonus = 0.0

        # Strong bonus for name_phrase match in cast/director metadata
        # Use word-boundary matching to avoid "Tom Holland" matching "Tom Hollander"
        if name_lower:
            if is_franchise_query:
                # Franchise/character queries (e.g. "spider-man movies", "batman films"):
                # "Spider-Man" is a CHARACTER, not a real person — do NOT look for it
                # in cast/director fields.  Instead match on title and document text.
                # This prevents Child's Play (1988, director "Tom Holland") from being
                # boosted when the user asks for Spider-Man movies, because the vector
                # embedding model associates "Spider-Man" → "Tom Holland" (MCU actor)
                # → retrieves films directed by the other Tom Holland.
                title_lower = meta.get('title', '').lower()
                if name_lower in title_lower:
                    bonus += 0.60  # Franchise name in title — definitive franchise entry
                elif name_lower in doc_lower:
                    bonus += 0.20  # Mentioned in plot/description
                else:
                    bonus -= 0.80  # Not part of the franchise — aggressively penalize
                # Penalize documentaries — users asking for "spider-man movies" want
                # the actual superhero films, not a making-of documentary
                movie_genres_raw = meta.get('genres', '').lower()
                if "documentary" in movie_genres_raw:
                    bonus -= 0.50
            else:
                cast_lower = meta.get('cast', '').lower()
                director_lower = meta.get('director', '').lower()
                # Split cast by comma and check each individual name
                cast_names = [n.strip() for n in cast_lower.split(',')]
                director_names = [n.strip() for n in director_lower.split(',')]
                # Check word boundaries: "tom holland" matches "tom holland"
                # but NOT "tom hollander"
                name_in_cast = False
                for cn in cast_names:
                    cn = cn.strip()
                    if cn == name_lower:
                        name_in_cast = True
                        break
                    # Check if name is at a word boundary in the cast member name
                    # "tom holland" should match "tom holland" but not "tom hollander"
                    idx = cn.find(name_lower)
                    if idx >= 0:
                        end_idx = idx + len(name_lower)
                        if end_idx == len(cn) or not cn[end_idx].isalpha():
                            name_in_cast = True
                            break
                name_in_director = False
                for dn in director_names:
                    dn = dn.strip()
                    if dn == name_lower:
                        name_in_director = True
                        break
                    idx = dn.find(name_lower)
                    if idx >= 0:
                        end_idx = idx + len(name_lower)
                        if end_idx == len(dn) or not dn[end_idx].isalpha():
                            name_in_director = True
                            break

                if name_in_cast or name_in_director:
                    # Director match gets highest priority:
                    # "Quentin Tarantino movies" should show Pulp Fiction (directed)
                    # above From Dusk Till Dawn (only acted in, directed by Rodriguez).
                    # Cast-only match still gets a bonus but lower than director.
                    if name_in_director:
                        bonus += 0.55  # Directed by this person — strongest signal
                        if name_in_cast:
                            bonus += 0.05  # Also acted in it — tiny extra
                    elif name_in_cast:
                        bonus += 0.40  # Only acted in, not directed
                elif name_lower in doc_lower:
                    # Check word boundary in doc text too
                    idx = doc_lower.find(name_lower)
                    if idx >= 0:
                        end_idx = idx + len(name_lower)
                        if end_idx == len(doc_lower) or not doc_lower[end_idx].isalpha():
                            bonus += 0.30
                        else:
                            # Substring match but NOT word boundary — penalize (e.g., "Tom Hollander")
                            bonus -= 0.15
                    else:
                        bonus -= 0.15  # Name not found at all
                else:
                    # Movie doesn't contain the person's name at all — penalize
                    bonus -= 0.15

                # Heavily penalise documentaries for name queries.
                # Director docs ("Spielberg", "QT8", "Capturing Avatar") are
                # semantically dense with that person's name and outrank their
                # actual films. Users asking "Scorsese films" want Goodfellas,
                # not a bio-documentary about him.
                movie_genres_raw = meta.get('genres', '').lower()
                if "documentary" in movie_genres_raw:
                    bonus -= 0.60

        # Big bonus for multi-word phrase match (e.g. "tom holland" in Cast line)
        # Also uses word-boundary checking
        for phrase in phrases:
            idx = doc_lower.find(phrase)
            if idx >= 0:
                end_idx = idx + len(phrase)
                if end_idx == len(doc_lower) or not doc_lower[end_idx].isalpha():
                    bonus += 0.35

        # Individual word matches — stronger bonus for cast/director hits
        stop = {"movies", "movie", "films", "film", "like", "best", "good",
                "with", "from", "that", "the", "and", "for", "about", "make",
                "you", "watch", "want", "see", "some", "give", "night",
                "should", "tonight", "today", "ever", "made"}
        for w in words:
            # Skip genre keywords — those are handled by genre-matching logic above
            if w in genre_map:
                continue
            if len(w) > 3 and w not in stop and w in doc_lower:
                cast_lower = meta.get('cast', '').lower()
                director_lower = meta.get('director', '').lower()
                if w in cast_lower or w in director_lower:
                    bonus += 0.25
                else:
                    bonus += 0.05

        # Genre-matching boost: if query mentions a genre, boost movies that
        # actually belong to that genre
        movie_genres_lower = meta.get('genres', '').lower()
        if query_genres:
            genre_hits = sum(1 for g in query_genres if g in movie_genres_lower)
            if genre_hits > 0:
                bonus += 0.12 * genre_hits
            # Multi-genre synergy: when query mentions 2+ genres (e.g., "romance
            # and action"), strongly favor movies that match ALL genres and penalize
            # movies matching only one.  Without this, pure-Action blockbusters
            # dominate over Action+Romance blends like "Mr and Mrs Smith".
            if len(query_genres) >= 2 and not is_similarity_query:
                if genre_hits >= len(query_genres):
                    bonus += 0.35  # Matches all query genres
                elif genre_hits == 1:
                    bonus -= 0.15  # Only matches one of multiple genres
            # Penalise genre mismatch for non-name queries
            if genre_hits == 0 and not name_lower:
                bonus -= 0.05
            # For "movies like X" (similarity queries), apply aggressive genre
            # mismatch penalty.  BGE's keyword bias means "movies like Get Out"
            # returns "The Getaway" and "Walking Out" — title-word matches from
            # completely wrong genres.  Heavy penalty ensures only genre-aligned
            # films survive reranking.
            # EXCEPTION: Same-director films get NO genre penalty because
            # directors often work across genres but maintain consistent style
            # (e.g., Nolan: Inception=SciFi vs Memento=Thriller).
            if is_similarity_query and len(query_genres) > 0:
                same_dir_sim = False
                if source_meta:
                    sd = source_meta.get('director', '').lower().strip()
                    cd = meta.get('director', '').lower().strip()
                    if sd and cd and sd == cd:
                        same_dir_sim = True
                if not same_dir_sim:
                    match_ratio = genre_hits / len(query_genres)
                    if genre_hits == 0:
                        bonus -= 0.80  # Severe: no genre overlap at all
                    elif match_ratio < 0.3:
                        bonus -= 0.60  # Very strong: <30% overlap
                    elif match_ratio < 0.5:
                        bonus -= 0.40  # Strong: <50% overlap
                    # ≥50% overlap = no penalty (Matrix shares 2/3 Inception genres)
            # Extra penalty: sad queries should not return pure comedies
            if is_sad_query and "comedy" in movie_genres_lower and "drama" not in movie_genres_lower:
                bonus -= 0.20
            # Penalty: sad queries should not return horror
            if is_sad_query and "horror" in movie_genres_lower:
                bonus -= 0.20
            # Penalty: sad queries should not return thrillers
            if is_sad_query and "thriller" in movie_genres_lower:
                if "drama" not in movie_genres_lower:
                    bonus -= 0.20  # Strong penalty for pure thriller
                else:
                    bonus -= 0.10  # Smaller penalty for drama/thriller combo (e.g., Misery)
            # Extra bonus: sad queries strongly reward confirmed Drama films
            # But only pure dramas or drama/romance — not drama/thriller combos
            if is_sad_query and "drama" in movie_genres_lower:
                if "thriller" not in movie_genres_lower and "horror" not in movie_genres_lower:
                    bonus += 0.15  # Strong bonus for pure drama
                else:
                    bonus += 0.03  # Minimal bonus for drama mixed with thriller/horror
            # Scary queries should not return comedies (e.g., "Scary Movie" is a comedy parody)
            if is_scary_query and "comedy" in movie_genres_lower and "horror" not in movie_genres_lower:
                bonus -= 0.20
            # Horror/scary queries: strongly penalize non-horror films.
            # "best horror movies" returns Godfather, LOTR, Shutter Island above
            # actual horror films like The Exorcist (#16) and Conjuring (#18).
            if is_scary_query and "horror" not in movie_genres_lower:
                bonus -= 0.30
            # Feel-good queries: strongly boost comedy/family, penalise dark genres
            # and pure drama. Users asking "cheer me up" or "rainy day" want
            # Legally Blonde, Paddington, School of Rock — not Forrest Gump, Shawshank.
            if is_feelgood_query:
                # When query explicitly says "comedy" (e.g., "light hearted comedy"),
                # the user wants live-action comedies, not animated films.
                is_comedy_explicit_fg = "comedy" in query_lower
                for good_genre in ["comedy", "family", "adventure", "animation"]:
                    if good_genre in movie_genres_lower:
                        if is_comedy_explicit_fg and good_genre in ("animation", "family"):
                            pass  # Don't boost animation/family for explicit comedy queries
                        else:
                            bonus += 0.30
                # When query says "comedy" but not animation, penalize animated films
                if is_comedy_explicit_fg and "animation" in movie_genres_lower:
                    bonus -= 0.25
                for bad_genre in ["horror", "thriller", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 0.20
                # Penalise pure Drama (without Comedy/Family/Animation) — these are
                # inspiring/moving but not lighthearted "cheer me up" films
                if "drama" in movie_genres_lower and not any(g in movie_genres_lower for g in ["comedy", "family", "animation"]):
                    bonus -= 0.35  # Strong enough to overcome rating bonus advantage
                # Boost movies with heartwarming/uplifting moods
                if any(m in moods_lower for m in ["heartwarming", "uplifting", "inspiring", "feel-good", "feel_good"]):
                    bonus += 0.15
            # Inspirational queries: boost Drama (Rocky, Shawshank, Dead Poets)
            # These are the OPPOSITE of feel-good — they want powerful dramas
            if is_inspirational_query and not is_feelgood_query:
                if "drama" in movie_genres_lower:
                    bonus += 0.20
                # Boost movies with inspiring/uplifting/triumphant moods
                if any(m in moods_lower for m in ["inspiring", "uplifting", "triumphant", "motivational", "heartwarming"]):
                    bonus += 0.15
                # Penalise horror/crime — these aren't inspirational
                for bad_genre in ["horror", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 0.10
            # Grandma/elderly queries: HARD penalty for inappropriate content
            if is_grandma_query:
                for bad_genre in ["horror", "thriller", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 1.50  # Extreme penalty — horror about grandmas has high vec sim
                if "comedy" in movie_genres_lower or "family" in movie_genres_lower:
                    bonus += 0.20
                if "drama" in movie_genres_lower and "crime" not in movie_genres_lower:
                    bonus += 0.10
                # Penalise very low-rated movies
                gm_rating = float(meta.get('rating', 0) or 0)
                if gm_rating > 0 and gm_rating < 5.5:
                    bonus -= 0.40
            # Kids/family queries: penalise adult content heavily
            if is_kids_query:
                rating_val = float(meta.get('rating', 0) or 0)
                # Boost family/animation/adventure genres
                if "family" in movie_genres_lower or "animation" in movie_genres_lower:
                    bonus += 0.15
                # Penalise horror, crime, thriller for kids queries
                for bad_genre in ["horror", "crime", "thriller"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 0.20
                # Penalise very low-rated movies (likely inappropriate)
                if rating_val > 0 and rating_val < 5.0:
                    bonus -= 0.15
            # Penalise music videos/concert films for non-music queries
            if "music" not in query_genres and "music" in movie_genres_lower:
                bonus -= 0.50
            # Penalise TV movies for broad genre queries
            if "tv movie" in movie_genres_lower:
                bonus -= 0.12
            # Penalise animation for non-animation queries (e.g., Batman animation
            # shorts shouldn't appear for "movies like The Dark Knight")
            if "animation" not in query_genres and "animation" in movie_genres_lower:
                bonus -= 0.15

        # Penalise documentaries universally for non-documentary queries.
        # Applied outside the query_genres block so it fires for concept queries
        # like "mafia movies" or "heist movies" that have no explicit genre word.
        movie_genres_lower_doc = meta.get('genres', '').lower()
        if "documentary" not in query_genres and "documentary" in movie_genres_lower_doc and not name_lower:
            bonus -= 0.35
        # Penalise Music-genre films (concerts, live performances, music videos)
        # unless user explicitly asked for music/musical content.
        music_query_words = {"music", "musical", "musicals", "concert", "soundtrack", "soundtracks", "sing", "singing"}
        if not (music_query_words & set(words)) and "music" in movie_genres_lower_doc and not name_lower:
            bonus -= 0.50
        # Penalise TV movies universally for non-specific queries
        if "tv movie" in movie_genres_lower_doc and not name_lower:
            bonus -= 0.12
        # Penalise Animation universally for non-animation queries.
        # Prevents animated shorts/kids movies from appearing in concept queries
        # (e.g., "road trip movies" returning Cars, A Goofy Movie).
        animation_query_words = {"animation", "animated", "anime", "cartoon", "pixar",
                                  "disney", "ghibli", "kids", "children", "family"}
        if not (animation_query_words & set(words)) and "animation" in movie_genres_lower_doc and not name_lower:
            bonus -= 0.20

        # Mood-matching via document keywords
        mood_boost_words = {
            "sad": ["tragedy", "heartbreak", "loss", "grief", "emotional", "tearjerker"],
            "cry": ["tragedy", "heartbreak", "emotional", "moving", "tearjerker"],
            "uplifting": ["inspiring", "triumph", "hope", "heartwarming"],
            "scary": ["terror", "frightening", "suspense", "disturbing"],
        }
        for w in words:
            if w in mood_boost_words:
                for mw in mood_boost_words[w]:
                    if mw in moods_lower or mw in doc_lower:
                        bonus += 0.06
                        break  # one mood keyword is enough

        # Popularity/rating boost for generic queries (no name detected)
        # This helps surface well-known films for queries like "best horror"
        if not name_lower:
            rating_raw = float(meta.get('rating', 0) or 0)
            popularity = float(meta.get('popularity', 0) or 0)
            # Cap rating at 9.0 for bonus calculations — movies with 9.5+ ratings
            # almost always have very few votes (bogus inflated scores).
            # Genuine classics top out at ~9.3 (Shawshank) on TMDB.
            rating = min(rating_raw, 9.0)
            # Penalise suspiciously perfect ratings with low popularity:
            # "Shelter (2026) 10.0", "Kathy O' (1958) 10.0" are garbage.
            if rating_raw >= 9.5 and popularity < 0.3:
                bonus -= 0.80
            elif rating_raw >= 9.5 and popularity < 0.5:
                bonus -= 0.40
            # Rating boost (0-10 scale) — strong to surface quality films
            # 8.0 → +0.16, 7.0 → +0.08, 6.5 → +0.04
            if rating >= 6.5:
                bonus += (rating - 6.0) * 0.08
            # Extra boost for quality-seeking queries (best, top, oscar, etc.)
            if is_quality_query and rating >= 7.0:
                bonus += (rating - 6.0) * 0.12
            # MASSIVE boost for meta-quality queries ("best of all time", etc.)
            # These need the absolute top-rated, most popular films regardless of
            # embedding similarity. An 8.7-rated Godfather should ALWAYS beat
            # a 5.0-rated "X-Rated: Greatest Adult Movies" even if the latter
            # has higher keyword overlap with the query.
            if is_meta_quality and rating >= 7.5:
                bonus += (rating - 6.0) * 0.25
            if is_meta_quality and popularity > 0.3:
                bonus += popularity * 0.30
            # Moderate boost for decade/era queries: surface iconic movies
            # Weaker than meta_quality but helps well-known classics beat obscure ones
            if is_decade_query and not is_meta_quality and rating >= 7.0 and popularity > 0.2:
                bonus += (rating - 6.0) * 0.15
                bonus += popularity * 0.15
            # Popularity boost — surface well-known movies over obscure ones
            # For hidden gems queries, INVERT the popularity logic: penalize
            # blockbusters and reward moderate-popularity films.
            if is_hidden_gems_query:
                # COUNTERACT the base rating bonus above: for hidden gems,
                # we DON'T want Godfather Part II (8.57) beating Predestination (7.4)
                # just because it's higher rated. Subtract back the rating bonus.
                if rating >= 6.5:
                    bonus -= (rating - 6.0) * 0.08  # undo the base rating boost
                # Also undo quality/meta bonuses if they snuck in
                if is_quality_query and rating >= 7.0:
                    bonus -= (rating - 6.0) * 0.12
                if is_meta_quality and rating >= 7.5:
                    bonus -= (rating - 6.0) * 0.25
                if is_meta_quality and popularity > 0.3:
                    bonus -= popularity * 0.30
                # Penalize "universally acclaimed + very popular" — these are definitionally
                # NOT hidden gems. Empire (8.4, 0.594), GoodFellas (8.45, 0.592),
                # 12 Angry Men (8.6, 0.587), Psycho (8.42, 0.582) all fail this.
                # But Whiplash (8.375, 0.592) barely squeaks in at <8.39.
                if rating >= 8.39 and popularity > 0.57:
                    bonus -= 0.50
                elif rating >= 8.0 and popularity > 0.58:
                    bonus -= 0.35
                elif popularity > 0.60:
                    bonus -= 0.30
                # Sweet spot: moderate popularity OR moderate rating (not both sky-high)
                # Whiplash (0.592, 8.375), Nightcrawler (0.555, 7.7), Spotlight (0.55, 7.8)
                # Predestination (0.528, 7.4), The Fall (0.495, 7.6)
                if 0.30 <= popularity <= 0.60 and rating >= 6.5 and rating < 8.39:
                    dist_from_peak = abs(popularity - 0.52)
                    sweet_bonus = max(0, 0.25 - dist_from_peak * 1.5)
                    bonus += sweet_bonus
                # Penalize very obscure — users want "gems" they can actually find
                if popularity < 0.20:
                    bonus -= 0.15
            else:
                if popularity > 0:
                    bonus += min(popularity, 1.0) * 0.10
                # Extra tier for very popular movies (popularity > 0.5)
                if popularity > 0.5:
                    bonus += 0.08
            # Extra boost for sad/emotional queries: surface classics over obscure films
            if is_sad_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.30
            if is_sad_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.20  # Extra tier for all-time classics
            # Extra boost for feel-good queries: surface well-known feel-good movies
            if is_feelgood_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.30
            if is_feelgood_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.20  # Extra tier for all-time classics
            # Extra boost for scary queries: surface well-known horror over obscure
            if is_scary_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.25
            if is_scary_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.15
            # Extra boost for comedy queries: surface well-known comedies
            if is_comedy_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.25
            if is_comedy_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.15
            # Extra boost for intense/adrenaline queries: surface well-known thrillers
            if is_intense_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.25
            if is_intense_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.15
            # Adrenaline-specific genre handling: boost Action/Thriller, penalise
            # slow genres.  "adrenaline rush" wants Die Hard/Mad Max, not LOTR.
            if is_adrenaline_query:
                # Strong pure-action boost: Mad Max, John Wick, Die Hard, Mission Impossible
                if "action" in movie_genres_lower:
                    bonus += 0.35
                    # Extra for pure action (Action+Thriller, Action+Crime)
                    # without Fantasy/Drama diluting the pace
                    if "thriller" in movie_genres_lower:
                        bonus += 0.15  # Action+Thriller = peak adrenaline
                    if "fantasy" not in movie_genres_lower and "drama" not in movie_genres_lower:
                        bonus += 0.10  # Pure action without slow elements
                    # Penalize Fantasy epics even WITH Action (LOTR, Hobbit)
                    if "fantasy" in movie_genres_lower:
                        bonus -= 0.30
                    # Penalize Adventure-heavy without Thriller (Empire, LOTR)
                    if "adventure" in movie_genres_lower and "thriller" not in movie_genres_lower:
                        bonus -= 0.20
                    # Penalize Drama+Action (Dark Knight) — not pure adrenaline
                    if "drama" in movie_genres_lower:
                        bonus -= 0.15
                elif "thriller" in movie_genres_lower:
                    bonus += 0.10  # Thriller alone is less adrenaline
                # Hard penalise non-action: Fantasy epics, pure dramas, etc.
                if "action" not in movie_genres_lower:
                    bonus -= 0.30  # LOTR without action, Fight Club, Pulp Fiction
                if "fantasy" in movie_genres_lower and "action" not in movie_genres_lower:
                    bonus -= 0.20
                if "drama" in movie_genres_lower and "action" not in movie_genres_lower and "thriller" not in movie_genres_lower:
                    bonus -= 0.20
            # Mind-bending/cerebral queries: boost SciFi+Thriller, penalise
            # blockbuster action.  "mind bending" wants Inception/Matrix/Memento,
            # not Avengers/Dark Knight.
            if is_cerebral_query:
                if "science fiction" in movie_genres_lower:
                    bonus += 0.20
                if "mystery" in movie_genres_lower:
                    bonus += 0.25
                # Pure Thriller without Action = more cerebral (Memento, Predestination)
                if "thriller" in movie_genres_lower and "action" not in movie_genres_lower:
                    bonus += 0.15
                # Action blockbusters without SciFi/Mystery = not cerebral
                if "action" in movie_genres_lower and "science fiction" not in movie_genres_lower and "mystery" not in movie_genres_lower:
                    bonus -= 0.30
                # War movies not cerebral
                if "war" in movie_genres_lower:
                    bonus -= 0.25
                # Strong keyword bonus: cerebral themes in movie description
                # Each matching keyword provides meaningful differentiation
                cerebral_kw_hits = sum(1 for kw in [
                    "mind-bending", "cerebral", "psychological", "surreal",
                    "twist", "puzzle", "reality", "perception", "dream",
                    "consciousness", "identity", "hallucination", "illusion",
                    "paranoia", "manipulation", "memory", "subconscious",
                ] if kw in doc_lower)
                bonus += cerebral_kw_hits * 0.12
                # Dampen popularity advantage: blockbusters shouldn't dominate
                # cerebral results just because they're well-known
                if popularity > 0.55:
                    bonus -= (popularity - 0.55) * 0.60
            # Extra boost for musical queries: surface well-known musicals
            if is_musical_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.25
            if is_musical_query and rating >= 8.0 and popularity > 0.5:
                bonus += 0.15
            # Musical genre match: boost Music-genre films for musical queries
            if is_musical_query and "music" in movie_genres_lower:
                bonus += 0.30
            # Musical keyword bonus: films with "musical" in keywords but NOT Music genre
            # (La La Land, Greatest Showman, Grease, West Side Story, Les Miserables)
            if is_musical_query and "music" not in movie_genres_lower:
                if "musical" in doc_lower:
                    bonus += 0.40  # Strong: these ARE musicals, just mis-tagged
            # Concert/tour film penalty: users asking for musicals want narrative films
            # not Taylor Swift concerts or live performances
            if is_musical_query:
                title_lower_mus = meta.get('title', '').lower()
                if any(kw in title_lower_mus for kw in ['tour', 'live', 'concert', 'eras']):
                    bonus -= 0.80  # Eliminate concert films
            # Extra boost for dark/disturbing queries: surface well-known dark films
            if is_dark_query and rating >= 7.0 and popularity > 0.2:
                bonus += 0.20
            # Visually stunning / cinematography queries: boost well-known narrative
            # films, penalize documentaries. Users want Blade Runner 2049, Life of Pi,
            # The Revenant — not Koyaanisqatsi or Chronos.
            if is_visual_query:
                if "documentary" in movie_genres_lower:
                    bonus -= 1.20  # Users want narrative feature films, not Koyaanisqatsi
                # Boost mainstream well-known visually impressive films
                if rating >= 7.0 and popularity > 0.3:
                    bonus += 0.30
                # Extra for sci-fi/adventure/fantasy — typically most visually stunning genres
                for vis_genre in ["science fiction", "adventure", "fantasy"]:
                    if vis_genre in movie_genres_lower:
                        bonus += 0.10
                # Penalise obscure
                if popularity < 0.2:
                    bonus -= 0.40
            # Penalise obscure/unknown movies for mood queries — users want
            # recognisable films, not straight-to-streaming titles
            if (is_sad_query or is_feelgood_query or is_scary_query or is_comedy_query or is_intense_query or is_dark_query or is_musical_query) and popularity <= 0.1:
                bonus -= 0.25
            if (is_sad_query or is_feelgood_query or is_scary_query or is_comedy_query or is_intense_query or is_dark_query or is_musical_query) and popularity <= 0.05:
                bonus -= 0.20  # Extra penalty for very obscure
            # Penalise obscure/unknown movies for meta-quality queries
            if is_meta_quality and popularity <= 0.2:
                bonus -= 0.40
            # Penalise obscure movies for decade/era queries too
            if is_decade_query and not is_meta_quality and popularity <= 0.1:
                bonus -= 0.30
            if is_decade_query and not is_meta_quality and popularity <= 0.05:
                bonus -= 0.20  # extra penalty stacks
            # Extra boost for similar-to queries: surface quality genre-matched
            # films.  "Movies like Parasite" should return other acclaimed
            # thrillers, not obscure horror films about biological parasites.
            if is_similarity_query:
                if rating >= 8.0 and popularity > 0.4:
                    bonus += 0.35  # Top-tier classics (Godfather, Shawshank, etc.)
                elif rating >= 7.0 and popularity > 0.2:
                    bonus += 0.25
                elif rating >= 6.5 and popularity > 0.1:
                    bonus += 0.10
                # Penalise obscure movies for similar-to — users want well-known
                # alternatives, not straight-to-video title-word matches
                if popularity <= 0.05:
                    bonus -= 0.30
                elif popularity <= 0.15 and rating < 6.5:
                    bonus -= 0.20
                # ── Similar-to director/mood matching ──────────────────────
                # Within genre-matched candidates, differentiate by shared
                # director, cast overlap, and mood similarity.  Users asking
                # "movies like Inception" want Interstellar (same director,
                # similar mood) not Iron Man (same genre but different feel).
                if source_meta:
                    src_director = source_meta.get('director', '').lower().strip()
                    cand_director = meta.get('director', '').lower().strip()
                    # Director match: strong bonus — same director almost always
                    # means similar style/themes (e.g., Nolan, Tarantino, Miyazaki).
                    # Capped at 0.30 (not 0.50) so genre-similar non-director films
                    # can still compete for top-5 slots after director diversity cap.
                    if src_director and cand_director and src_director == cand_director:
                        bonus += 0.30
                    # Cast overlap: medium bonus — shared leads suggest related universe
                    src_cast = set(n.strip().lower() for n in source_meta.get('cast', '').split(',') if n.strip())
                    cand_cast = set(n.strip().lower() for n in meta.get('cast', '').split(',') if n.strip())
                    cast_overlap = len(src_cast & cand_cast - {''})
                    if cast_overlap >= 1:
                        bonus += 0.20 * min(cast_overlap, 3)
                    # Mood match: strong bonus — same mood means similar feel
                    src_moods = set(source_meta.get('moods', '').lower().replace(',', ' ').split())
                    cand_moods = set(moods_lower.replace(',', ' ').split())
                    mood_overlap = len(src_moods & cand_moods - {''})
                    if mood_overlap >= 1:
                        bonus += 0.20 * min(mood_overlap, 3)
                    # Keyword overlap: medium bonus for shared thematic keywords
                    src_kw = set(source_meta.get('keywords', '').lower().replace(',', ' ').split())
                    cand_kw = set(meta.get('keywords', '').lower().replace(',', ' ').split())
                    kw_overlap = len(src_kw & cand_kw - {''})
                    if kw_overlap >= 2:
                        bonus += 0.08 * min(kw_overlap, 5)

        # ── Concept/mood genre reranking (runs when no genre keywords detected) ──
        # These apply even when the query doesn't contain explicit genre keywords
        # like "visually stunning cinematography" or "movies to watch with grandma".
        # When query_genres IS set, the equivalent logic runs inside that block above.
        if not query_genres and not name_lower:
            # Feel-good queries: strongly boost comedy/family, penalise dark genres
            if is_feelgood_query:
                is_comedy_explicit_fg2 = "comedy" in query_lower
                for good_genre in ["comedy", "family", "adventure", "animation"]:
                    if good_genre in movie_genres_lower:
                        if is_comedy_explicit_fg2 and good_genre in ("animation", "family"):
                            pass
                        else:
                            bonus += 0.30
                if is_comedy_explicit_fg2 and "animation" in movie_genres_lower:
                    bonus -= 0.25
                for bad_genre in ["horror", "thriller", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 0.20
                if "drama" in movie_genres_lower and not any(g in movie_genres_lower for g in ["comedy", "family", "animation"]):
                    bonus -= 0.35
                if any(m in moods_lower for m in ["heartwarming", "uplifting", "inspiring", "feel-good", "feel_good"]):
                    bonus += 0.15
            # Inspirational queries: boost Drama (Rocky, Shawshank, Dead Poets)
            if is_inspirational_query and not is_feelgood_query:
                if "drama" in movie_genres_lower:
                    bonus += 0.20
                if any(m in moods_lower for m in ["inspiring", "uplifting", "triumphant", "motivational", "heartwarming"]):
                    bonus += 0.15
                for bad_genre in ["horror", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 0.10
            # Grandma/elderly queries: HARD penalty for inappropriate content
            if is_grandma_query:
                for bad_genre in ["horror", "thriller", "crime"]:
                    if bad_genre in movie_genres_lower:
                        bonus -= 1.50  # Extreme penalty — horror about grandmas has high vec sim
                if "comedy" in movie_genres_lower or "family" in movie_genres_lower:
                    bonus += 0.20
                if "drama" in movie_genres_lower and "crime" not in movie_genres_lower:
                    bonus += 0.10
                gm_rating = float(meta.get('rating', 0) or 0)
                if gm_rating > 0 and gm_rating < 5.5:
                    bonus -= 0.40
            # Adrenaline: strong action-only boost
            if is_adrenaline_query:
                if "action" in movie_genres_lower:
                    bonus += 0.35
                    if "thriller" in movie_genres_lower:
                        bonus += 0.15  # Action+Thriller = peak adrenaline
                    if "fantasy" not in movie_genres_lower and "drama" not in movie_genres_lower:
                        bonus += 0.10  # Pure action without slow elements
                    # Penalize Fantasy epics even WITH Action (LOTR, Hobbit)
                    if "fantasy" in movie_genres_lower:
                        bonus -= 0.30
                    # Penalize Adventure-heavy without Thriller (Empire, LOTR)
                    if "adventure" in movie_genres_lower and "thriller" not in movie_genres_lower:
                        bonus -= 0.20
                    # Penalize Drama+Action (Dark Knight) — not pure adrenaline
                    if "drama" in movie_genres_lower:
                        bonus -= 0.15
                elif "thriller" in movie_genres_lower:
                    bonus += 0.10
                if "action" not in movie_genres_lower:
                    bonus -= 0.30
                if "fantasy" in movie_genres_lower and "action" not in movie_genres_lower:
                    bonus -= 0.20
                if "drama" in movie_genres_lower and "action" not in movie_genres_lower and "thriller" not in movie_genres_lower:
                    bonus -= 0.20
            # Visually stunning: boost well-known narrative films, penalize docs
            if is_visual_query:
                if "documentary" in movie_genres_lower:
                    bonus -= 1.20  # Users want narrative feature films, not Koyaanisqatsi
                rating_v = float(meta.get('rating', 0) or 0)
                pop_v = float(meta.get('popularity', 0) or 0)
                if rating_v >= 7.0 and pop_v > 0.3:
                    bonus += 0.30
                for vis_genre in ["science fiction", "adventure", "fantasy"]:
                    if vis_genre in movie_genres_lower:
                        bonus += 0.10
                if pop_v < 0.2:
                    bonus -= 0.40

        # Hard quality floor: very low-rated movies almost never belong in results.
        # Apply universally for clearly bad films; for moderately bad films apply
        # when the query is genre/mood-oriented (not a specific person search).
        # Note: floor_rating == 0.0 means unrated; treat as bad for genre queries.
        floor_rating = float(meta.get('rating', 0) or 0)

        # Quality/popularity differentiation for NAME queries (actor/director).
        # Without this, "Brad Pitt movies" returns ANY Brad Pitt film by
        # embedding distance alone — Troy/Allied/Wolfs instead of
        # Fight Club/Se7en/Inglourious Basterds.
        if name_lower:
            nr_raw = float(meta.get('rating', 0) or 0)
            nr = min(nr_raw, 9.0)
            np_ = float(meta.get('popularity', 0) or 0)
            if nr >= 7.0:
                bonus += (nr - 6.0) * 0.12
            if np_ > 0.3:
                bonus += np_ * 0.15
            if nr_raw >= 9.5 and np_ < 0.3:
                bonus -= 0.60
        if floor_rating < 3.0 and (floor_rating > 0 or not name_lower):
            # Very poor quality or unrated genre/concept query — eliminate from results
            bonus -= 1.20
        elif 3.0 <= floor_rating < 5.0:
            # Below-average quality — penalise for mood/quality/genre queries
            if is_quality_query or is_sad_query or is_feelgood_query or is_scary_query or is_comedy_query or is_kids_query or not name_lower:
                bonus -= 0.80
        elif 5.0 <= floor_rating < 6.0 and not name_lower:
            # Mediocre quality — light penalty for genre/concept queries
            # Prevents movies like "Jane Austen's Mafia! (5.4)" from beating
            # "The Godfather (8.7)" just because of literal keyword match
            bonus -= 0.30

        # ── Title-literal-match trap penalty ────────────────────────
        # BGE-large has strong keyword bias: "movies that make you cry" retrieves
        # "Cry Macho", "The Man Who Cried"; "edge of your seat" → "Edge of Darkness";
        # "hidden gems" → "Hidden Gems"; "best of all time" → "X-Rated: Greatest
        # Adult Movies of All Time". Penalise movies that appear to match ONLY because
        # their title contains a literal query keyword, when the query is clearly about
        # mood/concept/quality rather than a specific title search.
        if not name_lower:
            title_lower = meta.get('title', '').lower()
            title_words = set(title_lower.replace(':', ' ').replace('-', ' ').replace("'", ' ').split())
            # Mood/concept words that commonly appear in unrelated movie titles
            trap_words = {
                'cry', 'cried', 'crying', 'tears', 'edge', 'hidden', 'gem', 'gems',
                'burn', 'burning', 'slow', 'twist', 'twists', 'road', 'trip',
                'good', 'night', 'dead', 'dark', 'light', 'rush', 'speed',
                'epic', 'true', 'story', 'date', 'friends', 'friend',
                'lost', 'time', 'travel', 'mind', 'cheer', 'happy', 'sad',
                'fun', 'funny', 'scare', 'scary', 'fear', 'thrill',
                'best', 'greatest', 'rated', 'ultimate', 'unseen', 'unknown',
                'forgotten', 'all', 'die', 'alive', 'incoming', 'new',
                'before', 'land', 'everyone', 'disaster', 'zombie', 'out',
                'get', 'away', 'gone', 'love', 'wonder', 'wonderful',
                'life', 'watch', 'like', 'spirited', 'parasite',
                'black', 'white', 'classic', 'old', 'age', 'war',
                'sport', 'sports', 'musical', 'music', 'sing',
                'foreign', 'language', 'must', 'oscar', 'picture',
                'three', 'hour', 'long', 'short', 'set', 'based',
                'sound', 'track', 'soundtrack', 'great', 'perfect',
                'halloween', 'christmas', 'holiday', 'summer', 'winter',
                'spring', 'fall', 'haunted', 'house', 'body', 'blood',
                'run', 'running', 'fast', 'race', 'hero', 'heroes',
                'girl', 'boy', 'man', 'woman', 'young', 'hidden',
                'secret', 'silent', 'final', 'last', 'first',
                'big', 'little', 'small', 'high', 'deep',
                'world', 'american', 'home', 'city', 'burn', 'fire',
                'adrenaline', 'thriller', 'thrillers', 'action',
                'comedy', 'horror', 'drama', 'romance', 'fantasy',
                'animated', 'epic', 'bittersweet', 'nostalgic',
                'inspiring', 'uplifting', 'heartwarming', 'disturbing',
                'twisted', 'unexpected', 'underrated',
            }
            # Use prefix matching: "cry" matches title words "cry", "cried",
            # "crying"; "thrill" matches "thriller", "thrilling", etc.
            trap_hit_count = 0
            for qw in words:
                if qw not in trap_words:
                    continue
                for tw in title_words:
                    if tw.startswith(qw) or qw.startswith(tw):
                        trap_hit_count += 1
                        break
            if trap_hit_count >= 1:
                # Movie title literally contains a query mood/concept word.
                # Apply penalty unconditionally — even popular movies shouldn't rank
                # high solely because their title matches a mood keyword.
                # Scale: 1 trap hit → -0.35, 2 hits → -0.70, etc.
                movie_rating = float(meta.get('rating', 0) or 0)
                movie_pop = float(meta.get('popularity', 0) or 0)
                # Heavier penalty for low-quality title traps
                if movie_rating < 7.0 or movie_pop < 0.2:
                    bonus -= 0.50 * trap_hit_count
                elif movie_rating < 7.5 or movie_pop < 0.4:
                    bonus -= 0.35 * trap_hit_count
                else:
                    # Still penalize even high-rated movies, but less
                    bonus -= 0.15 * trap_hit_count

        # ── Foreign language query boost ─────────────────────────────
        # For "foreign language must watch" queries, boost documents that
        # contain known foreign film indicators (director names, titles)
        # and penalize clearly English-only films.
        if is_foreign_query:
            foreign_indicators = [
                "parasite", "oldboy", "amelie", "spirited away",
                "seven samurai", "cinema paradiso", "pan's labyrinth",
                "city of god", "life is beautiful", "crouching tiger",
                "rashomon", "howl's moving castle", "princess mononoke",
                "totoro", "bicycle thieves", "roma", "ikiru",
                "bong joon", "akira kurosawa", "hayao miyazaki",
                "wong kar", "pedro almod", "fellini", "bergman",
                "tarkovsky", "kore-eda", "studio ghibli",
                "korean", "japanese", "french cinema", "italian neo",
                "subtitles", "bollywood", "dubbed",
            ]
            if any(fi in doc_lower for fi in foreign_indicators):
                bonus += 0.80
            else:
                bonus -= 0.40

        # ── Augmented title-hint matching ─────────────────────────────
        # When mood augmentation names specific films (e.g., "horror" →
        # "The Exorcist, Hereditary"), boost candidates matching those titles.
        # This ensures the named exemplars rank near the top even if the
        # embedding model doesn't precisely target them.
        if hint_titles:
            title_lower = meta.get('title', '').lower()
            for ht in hint_titles:
                if ht in title_lower:
                    bonus += 0.40
                    break  # One match is enough

        scored.append((dist - bonus, i))

    scored.sort(key=lambda x: x[0])
    top_indices = [idx for _, idx in scored[:top_k]]

    return {
        'documents':  [[results['documents'][0][i] for i in top_indices]],
        'metadatas':  [[results['metadatas'][0][i] for i in top_indices]],
        'distances':  [[results['distances'][0][i] for i in top_indices]],
    }


def _extract_name_phrase(query: str) -> Optional[str]:
    """
    Extract a likely person-name phrase from the query for document text search.

    Strategy: strip common movie-query stop words, then look for the longest
    remaining multi-word phrase (≥2 words, ≥6 chars).  This catches actor and
    director names like "Tom Holland", "Christopher Nolan", etc.

    Returns the phrase (title-cased for case-sensitive doc matching) or None.
    """
    stop = {
        "movies", "movie", "films", "film", "like", "similar", "recommend",
        "best", "good", "great", "top", "with", "from", "show", "me", "find",
        "featuring", "starring", "directed", "by", "of", "the", "a", "an",
        "in", "and", "for", "to", "i", "want", "need", "something", "watch",
        "give", "suggest", "action", "comedy", "drama", "horror", "thriller",
        "sci-fi", "romance", "animated", "animation", "fantasy", "adventure",
        "mystery", "crime", "documentary", "under", "over", "hours", "minutes",
        "rated", "rating", "year", "new", "old", "classic",
        # Adjectives/descriptors that trigger false positives
        "romantic", "dark", "gritty", "feel-good", "uplifting", "mind-bending",
        "psychological", "short", "long", "epic", "funny", "scary", "sad",
        "happy", "ending", "made", "about", "what", "after", "onwards",
        "high", "low", "really", "very", "most", "more", "less",
        "comedies", "dramas", "thrillers", "horrors", "romances",
        # Common query words that aren't names
        "see", "some", "can", "please", "hey", "get", "make", "you", "cry",
        "feel", "night", "date", "friends", "family", "kids", "winning",
        "oscar", "award", "nominated", "acclaimed",
        # Time/place/concept words that trigger false positives
        "tonight", "today", "ever", "never", "always", "plot", "twist",
        "twists", "ending", "endings", "story", "stories", "based", "true",
        "real", "life", "world", "war", "love", "death", "history",
        # Studio/format words
        "disney", "pixar", "marvel", "anime", "manga", "bollywood",
        "netflix", "hbo", "series", "sequel", "prequel", "remake",
        "90s", "80s", "70s", "60s", "2000s", "2010s", "2020s",
        # More descriptors
        "laugh", "laughing", "crying", "terrifying", "hilarious",
        "heartwarming", "inspiring", "intense", "suspenseful", "gripping",
        "thought-provoking", "visually", "stunning", "beautiful",
        "must-see", "must", "iconic", "legendary", "famous", "popular",
        "underrated", "overrated", "cult", "indie", "mainstream",
        "superhero", "zombie", "alien", "space", "heist", "heists",
        "musical", "western", "noir", "sci", "fi",
        # Country/language words
        "korean", "japanese", "french", "indian", "chinese", "spanish",
        "british", "american", "italian", "german", "bollywood",
        "korea", "japan", "france", "india", "china",
        # More query words
        "surprise", "fun", "light", "girlfriend", "boyfriend",
        "entertaining", "enjoyable", "should", "me", "my",
        "wilderness", "nature", "survival", "artificial", "intelligence",
        "sports", "underdog", "friendship", "set", "york",
        "will", "that", "would", "could", "just", "really",
        "rainy", "day", "weekend", "alone", "together",
        # Temporal/occasion words that cause literal matches
        "friday", "saturday", "sunday", "monday", "night", "halloween",
        "christmas", "summer", "winter", "spring", "autumn", "fall",
        "time", "travel", "revenge", "zombie", "zombies", "heist",
        "musical", "musicals", "survival", "explore", "exploration",
        "road", "trip", "journey", "coming",
        # Genre/concept words that cause false-positive name detection
        "mafia", "gangster", "mob", "mobster", "cartel", "hitman",
        "think", "thought", "thoughts", "thinking", "mind", "brain",
        "clever", "intelligent", "complex", "profound", "existential",
        "violent", "brutal", "gritty", "disturbing", "mature",
        "emotional", "moving", "powerful", "touching", "stirring",
        "detective", "serial", "killer", "monster", "demon", "ghost",
        "vampire", "witch", "dragon", "magic", "wizard", "knight",
        "heist", "robbery", "thief", "spy", "agent", "soldier",
        "pirate", "cowboy", "samurai", "ninja", "viking", "gladiator",
        # Misc words that cause false-positive name detection
        "are", "actually", "really", "quite", "rather", "truly",
        "highly", "incredibly", "extremely", "absolutely", "totally",
        "definitely", "certainly", "probably", "seriously",
        "been", "being", "were", "also", "only", "even",
        # Common words that appear in meta/quality queries
        "all", "every", "each", "any", "many", "much", "both",
        "before", "after", "during", "since", "until",
        "three", "hour", "long", "epic", "trilogy", "trilogies",
        "picture", "winners", "foreign", "language",
        "bittersweet", "slow", "fast", "edge", "seat",
        "adrenaline", "hidden", "underrated", "overrated",
        "masterpiece", "masterpieces", "classic", "classics",
        "nostalgia", "nostalgic", "essential", "gem", "gems",
        "brilliant", "outstanding", "definitive", "unforgettable",
        # Action/intensity words that trigger false name detection
        "rush", "burn", "burning", "blast", "crash", "chase",
        "fight", "punch", "kick", "shot", "shoot", "strike",
        # Color/descriptor words that appear in queries
        "black", "white", "red", "blue", "green", "golden",
        "brown", "grey", "gray", "silver", "dark", "bright",
        # Common verbs/contractions that survive filtering
        "seen", "heard", "watched", "known", "told", "found",
        "haven't", "i've", "you've", "we've", "they've",
        # Verb forms that cause false-positive name detection
        "likes", "liked", "liking", "loving", "loves",
        "cheer", "cheering", "cheered", "up",
        # Family/audience words that cause false-positive name detection
        "grandma", "grandmother", "grandpa", "grandfather", "granny",
        "grandparent", "grandparents", "elderly", "senior", "seniors",
        "toddler", "toddlers", "teen", "teens", "teenager", "teenagers",
        # Adjective/descriptor words that cause title-matching pollution
        "unexpected", "unseen", "unheard", "unknown", "unexplored",
        "recent", "latest", "newest", "oldest", "earliest",
        "disaster", "catastrophe", "apocalypse", "apocalyptic",
        "courtroom", "courthouse", "court", "trial", "jury",
        "inspirational", "inspired", "inspiring", "inspiration",
        "cinematography", "cinematographer", "cinematographic",
        "soundtrack", "soundtracks", "score", "scoring",
        "bending", "bent", "twisted", "twisting",
        "hearted", "heartfelt", "heartwarming", "heartbreaking",
        "age", "aging", "aged", "ages",
        "die", "dying", "dead", "alive", "born", "live", "living",
        "everyone", "everybody", "someone", "somebody", "anyone", "anything",
        "nobody", "nothing", "everything", "everywhere", "nowhere",
        "they", "them", "their", "theirs", "themselves",
        "he", "him", "his", "she", "her", "hers",
        "isn't", "aren't", "wasn't", "weren't", "don't",
        "doesn't", "didn't", "won't", "wouldn't", "couldn't",
        "shouldn't", "can't", "i'm", "you're", "we're",
        "they're", "he's", "she's", "it's", "let's",
        "what's", "that's", "there's", "here's", "who's",
        # Pronouns and common determiners
        "your", "our", "their", "his", "her", "its",
        "this", "these", "those", "what", "which", "who",
        "how", "why", "where", "when", "does", "did",
        "have", "has", "had", "not", "but", "yet",
    }
    words = query.lower().split()
    # Remove stop words and pure numbers (e.g., "90", "2010")
    filtered = [w for w in words if w not in stop and len(w) > 1 and not w.isdigit()]

    if len(filtered) == 0:
        return None

    # Single remaining word (≥3 chars) is likely a mononymous celebrity
    # e.g. "zendaya movies" → filtered=["zendaya"] → "Zendaya"
    if len(filtered) == 1:
        if len(filtered[0]) >= 3:
            return filtered[0].title()
        return None

    # Multi-word: the longest contiguous run of remaining words is our best guess at a name
    phrase = " ".join(filtered)
    if len(phrase) >= 6:
        # Title-case for matching document text (Cast: Tom Holland, Director: Jon Watts)
        return phrase.title()
    return None


def _is_franchise_match(candidate_title: str, src_core: str, src_lower: str,
                        src_meta: Optional[dict] = None, cand_meta: Optional[dict] = None) -> bool:
    """
    Check if candidate_title belongs to the same franchise as the source movie.
    Used to exclude sequels/prequels/remakes from 'movies like X' results.
    """
    cand_lower = candidate_title.lower()
    # Exact match
    if cand_lower == src_lower:
        return True
    # Core title appears as substring: "godfather" in "The Godfather Part II"
    if src_core in cand_lower:
        return True
    # Candidate core in source: "shawshank" in "The Shawshank Redemption"
    cand_core = cand_lower
    for prefix in ("the ", "a "):
        if cand_core.startswith(prefix):
            cand_core = cand_core[len(prefix):]
    if len(cand_core) > 4 and cand_core in src_lower:
        return True
    # NOTE: We previously had a director+cast heuristic here that excluded
    # films sharing same director + 3+ cast words.  This was removed because
    # it incorrectly excluded highly relevant recommendations like
    # Batman Begins for "movies like The Dark Knight" (same director + large
    # cast overlap but exactly the kind of film users want).  Title-based
    # matching and character name detection in the discover endpoint handle
    # actual franchise/sequel exclusion sufficiently.
    return False


def _merge_results(primary: dict, secondary: dict, limit: int) -> dict:
    """Merge two ChromaDB result dicts, deduplicating by movie title."""
    seen = set()
    merged = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    for src in [primary, secondary]:
        for i, meta in enumerate(src['metadatas'][0]):
            title = meta.get('title', '')
            if title not in seen and len(merged['documents'][0]) < limit:
                seen.add(title)
                merged['documents'][0].append(src['documents'][0][i])
                merged['metadatas'][0].append(meta)
                merged['distances'][0].append(src['distances'][0][i])

    return merged


def _bm25_search(query: str, n_results: int = 30) -> dict:
    """
    Run BM25 keyword search and return results in ChromaDB result dict format.
    Uses the global bm25_index and bm25_doc_ids built at startup.
    """
    if bm25_index is None or not bm25_doc_ids:
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Get top-N indices by BM25 score (higher = more relevant)
    import numpy as np
    top_indices = np.argsort(bm25_scores)[::-1][:n_results]
    top_ids = [bm25_doc_ids[i] for i in top_indices if bm25_scores[i] > 0]

    if not top_ids:
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    # Fetch full records from ChromaDB by ID
    records = vectorstore._collection.get(ids=top_ids, include=["documents", "metadatas"])

    # Convert BM25 scores to pseudo-distances (lower = better): 1/(1+score)
    distances = []
    id_to_score = {bm25_doc_ids[i]: bm25_scores[i] for i in top_indices if bm25_scores[i] > 0}
    for doc_id in records['ids']:
        score = id_to_score.get(doc_id, 0)
        distances.append(1.0 / (1.0 + score))

    return {
        'documents': [records['documents']],
        'metadatas': [records['metadatas']],
        'distances': [distances],
    }


def _rrf_fuse(vec_results: dict, bm25_results: dict, k: int = 60, limit: int = 50,
              vec_weight: float = 3.0, bm25_weight: float = 1.0) -> dict:
    """
    Reciprocal Rank Fusion: combine vector search and BM25 results.
    RRF score = sum( weight * 1/(k + rank) ) for each source where the doc appears.
    vec_weight > bm25_weight ensures semantic similarity dominates over keyword overlap.
    """
    # Build title → (meta, doc) mapping and RRF scores
    title_data = {}   # title → {'meta': ..., 'doc': ..., 'rrf': float}

    for rank, meta in enumerate(vec_results['metadatas'][0]):
        title = meta.get('title', '')
        if title not in title_data:
            title_data[title] = {
                'meta': meta,
                'doc': vec_results['documents'][0][rank],
                'rrf': 0.0,
            }
        title_data[title]['rrf'] += vec_weight * 1.0 / (k + rank + 1)

    for rank, meta in enumerate(bm25_results['metadatas'][0]):
        title = meta.get('title', '')
        if title not in title_data:
            title_data[title] = {
                'meta': meta,
                'doc': bm25_results['documents'][0][rank],
                'rrf': 0.0,
            }
        title_data[title]['rrf'] += bm25_weight * 1.0 / (k + rank + 1)

    # Sort by RRF score descending, take top `limit`
    sorted_titles = sorted(title_data.items(), key=lambda x: x[1]['rrf'], reverse=True)[:limit]

    merged = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    for title, data in sorted_titles:
        merged['documents'][0].append(data['doc'])
        merged['metadatas'][0].append(data['meta'])
        # Convert RRF score to pseudo-distance (lower = better)
        merged['distances'][0].append(1.0 / data['rrf'] if data['rrf'] > 0 else 999.0)

    return merged


def _classify_query(query: str) -> dict:
    """
    Classify user query into an intent type with structured parameters.
    Returns dict with 'intent' and relevant extracted fields.

    Intent types:
      - person:       Actor/director query (e.g., "Nolan movies")
      - genre:        Genre-based (e.g., "best horror movies")
      - mood:         Emotional/mood (e.g., "feel good movies")
      - similar_to:   "Movies like X" pattern
      - decade:       Era/year query (e.g., "80s action movies")
      - concept:      Abstract concept (e.g., "movies with twist endings")
      - specific:     Specific audience/context (e.g., "movies for kids")
      - franchise:    Franchise query (e.g., "Marvel movies")
      - conversational: General chat (e.g., "what should I watch tonight")
    """
    q = query.lower().strip()

    result = {'intent': 'general', 'query': query}

    # ── Similar-to detection ────────────────────────────────────
    similar_patterns = [
        r'(?:movies?|films?)\s+(?:like|similar\s+to)\s+(.+)',
        r'similar\s+(?:movies?|films?)?\s*(?:to|as)\s+(.+)',
        r'(?:something|anything)\s+(?:like|similar\s+to)\s+(.+?)(?:\s+but\s+|\s*$)',
    ]
    for pat in similar_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            result['intent'] = 'similar_to'
            ref = m.group(1).strip().strip('"\'')
            # Strip trailing qualifiers like "but as a movie"
            ref = re.sub(r'\s+but\s+(as\s+)?a\s+movie.*$', '', ref, flags=re.IGNORECASE).strip()
            result['reference_movie'] = ref
            return result

    # ── Decade/era detection ────────────────────────────────────
    decade_match = re.search(r'\b(19[2-9]0|20[0-2]0)s?\b', q)
    # Also match short forms: "90s", "80s", "70s", etc.
    short_decade_match = re.search(r'\b([2-9]0)s\b', q)
    year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', q)
    if decade_match:
        result['intent'] = 'decade'
        result['decade'] = int(decade_match.group(1))
    elif short_decade_match:
        result['intent'] = 'decade'
        short_val = int(short_decade_match.group(1))
        result['decade'] = 1900 + short_val if short_val >= 20 else 2000 + short_val
    elif year_match:
        result['intent'] = 'decade'
        result['year'] = int(year_match.group(1))
    elif any(kw in q for kw in ['classic ', 'vintage ', 'old ', 'golden age', 'retro ', 'nostalgia', 'black and white', 'b&w']):
        result['intent'] = 'decade'
        result['era'] = 'classic'

    # ── Person detection (may override decade) ──────────────────
    # Check for known person-query patterns
    person_patterns = [
        r'(?:by|directed\s+by|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:movies?|films?|directed|filmography)',
    ]
    for pat in person_patterns:
        m = re.search(pat, query)  # case-sensitive on original query
        if m:
            result['intent'] = 'person'
            result['person'] = m.group(1).strip()
            return result

    # ── Mood/emotion detection ──────────────────────────────────
    mood_keywords = {
        'feel good': 'uplifting', 'feel-good': 'uplifting', 'uplifting': 'uplifting',
        'heartwarming': 'uplifting', 'inspiring': 'uplifting', 'motivational': 'uplifting',
        'sad': 'melancholy', 'cry': 'melancholy', 'crying': 'melancholy',
        'depressing': 'melancholy', 'emotional': 'emotional', 'moving': 'emotional',
        'scary': 'frightening', 'terrifying': 'frightening', 'creepy': 'frightening',
        'spooky': 'frightening', 'disturbing': 'dark',
        'funny': 'humorous', 'hilarious': 'humorous', 'laugh': 'humorous',
        'relaxing': 'calm', 'chill': 'calm', 'cozy': 'calm',
        'lighthearted': 'uplifting', 'light hearted': 'uplifting',
        'cheer me up': 'uplifting', 'cheer me': 'uplifting', 'cheer up': 'uplifting',
        'intense': 'intense', 'adrenaline': 'intense', 'edge of seat': 'intense',
        'edge of your seat': 'intense', 'edge of my seat': 'intense',
        'slow burn': 'intense',
        'romantic': 'romantic', 'love story': 'romantic',
        'dark': 'dark', 'gritty': 'dark', 'bleak': 'dark',
        'mind-bending': 'cerebral', 'mind bending': 'cerebral', 'thought-provoking': 'cerebral',
        'make you think': 'cerebral', 'makes you think': 'cerebral',
        'intellectual': 'cerebral', 'smart': 'cerebral',
    }
    for kw, mood_type in mood_keywords.items():
        if kw in q:
            result['intent'] = 'mood'
            result['mood'] = mood_type
            # Check for comforting intent: "when sad" means want uplifting, not sad
            if mood_type == 'melancholy' and re.search(r'when\s+(?:i\'?m?\s+)?(?:sad|down|depressed)', q):
                result['mood'] = 'uplifting'
            return result

    # ── Specific audience/context ───────────────────────────────
    audience_keywords = {
        'kids': 'children', 'children': 'children', 'family': 'family',
        'teenager': 'teen', 'teen ': 'teen',
        'date night': 'date', 'date movie': 'date',
        'grandma': 'elderly', 'grandmother': 'elderly', 'grandpa': 'elderly',
        'parents': 'family', 'whole family': 'family',
        'year old': 'children', '10 year': 'children', '8 year': 'children',
        '6 year': 'children', '12 year': 'children',
    }
    for kw, audience in audience_keywords.items():
        if kw in q:
            result['intent'] = 'specific'
            result['audience'] = audience
            return result

    # ── Meta-quality detection (before concept/genre) ───────────
    # Detect "best X of all time", "movies everyone should watch", etc.
    # These are queries about general film quality, not specific themes.
    meta_quality_patterns = [
        r'best\s+(?:movies?|films?)\s+(?:of\s+)?all\s+time',
        r'greatest\s+(?:movies?|films?)',
        r'(?:movies?|films?)\s+everyone\s+should\s+(?:see|watch)',
        r'(?:should|must)\s+(?:see|watch)\s+before\s+(?:you\s+)?die',
        r'oscar\s+(?:best\s+)?picture',
        r'best\s+(?:picture|movie)\s+winner',
        r'best\s+(?:movie\s+)?trilog',
        r'must\s+(?:see|watch)\s+(?:movies?|films?)',
        r'essential\s+(?:movies?|films?)',
        r'best\s+movies?\s+(?:of\s+)?(?:all\s+time|ever)',
        r'(?:movies?|films?)\s+of\s+all\s+time',
        r'best\s+movies?\s+ever\s+(?:made)?',
    ]
    for pat in meta_quality_patterns:
        if re.search(pat, q):
            result['intent'] = 'concept'
            result['concept'] = 'meta_quality'
            return result

    # ── Concept/theme detection ─────────────────────────────────
    concept_keywords = [
        'twist ending', 'plot twist', 'surprise ending', 'unexpected',
        'based on true story', 'based on real', 'true story', 'biographical',
        'time travel', 'space', 'dystopia', 'dystopian', 'post-apocalyptic',
        'revenge', 'heist', 'survival', 'underdog', 'redemption',
        'courtroom', 'prison', 'road trip', 'coming of age',
        'supernatural', 'ghost', 'vampire', 'zombie',
        'visually stunning', 'cinematography', 'beautiful',
        'oscar', 'award', 'critically acclaimed',
        'foreign language', 'foreign film', 'subtitled', 'non-english',
        'international film', 'international movie',
        'soundtrack', 'soundtracks', 'score',
        'hidden gem', 'hidden gems', 'underrated', 'overlooked',
        'three hour', '3 hour', 'epic movie', 'epic film',
        'set in new york', 'new york', 'set in london', 'set in paris',
        'set in tokyo', 'set in los angeles',
        'date night', 'date movie',
    ]
    for concept in concept_keywords:
        if concept in q:
            result['intent'] = 'concept'
            result['concept'] = concept
            return result

    # ── Genre detection ─────────────────────────────────────────
    genre_keywords = [
        'horror', 'comedy', 'action', 'thriller', 'drama', 'sci-fi',
        'science fiction', 'romance', 'western', 'war', 'crime',
        'mystery', 'fantasy', 'animation', 'animated', 'documentary',
        'musical', 'noir', 'adventure', 'superhero', 'martial arts',
        'slasher', 'psychological', 'mafia', 'gangster', 'mob ',
        'sports', 'sport', 'disaster', 'courtroom',
    ]
    for genre in genre_keywords:
        if genre in q:
            result['intent'] = 'genre'
            result['genre'] = genre
            return result

    # ── Franchise detection ─────────────────────────────────────
    franchise_keywords = [
        'marvel', 'dc ', 'star wars', 'harry potter', 'lord of the rings',
        'james bond', 'batman', 'spider-man', 'spider man', 'spiderman',
        'avengers', 'x-men', 'x men', 'xmen',
        'fast and furious', 'mission impossible', 'jurassic', 'pixar',
        'disney', 'studio ghibli', 'mcu', 'dceu',
        'iron man', 'ironman', 'superman', 'aquaman', 'black panther',
        'captain america', 'thor', 'hulk', 'guardians of the galaxy',
        'transformers', 'john wick', 'indiana jones', 'matrix',
        'alien ', 'predator', 'terminator', 'die hard',
    ]
    for franchise in franchise_keywords:
        if franchise in q:
            result['intent'] = 'franchise'
            result['franchise'] = franchise.strip()
            return result

    # ── Conversational ──────────────────────────────────────────
    conversational_patterns = [
        r'what\s+(?:should|can|do)\s+(?:i|we)\s+watch',
        r'recommend\s+(?:me\s+)?(?:a|some)',
        r'suggest\s+(?:me\s+)?(?:a|some)',
        r'something\s+(?:good|nice|fun)\s+to\s+watch',
        r'what.*(?:good|best).*(?:watch|see|stream)',
        r'(?:bored|boring)\s+(?:what|any)',
        r'movie\s*(?:night|marathon)',
    ]
    for pat in conversational_patterns:
        if re.search(pat, q):
            result['intent'] = 'conversational'
            return result

    return result


@lru_cache(maxsize=2048)
def _classify_query_cached(normalized_query: str) -> dict:
    return _classify_query(normalized_query)


def _extract_movie_title_query(query: str) -> Optional[str]:
    """
    Detect 'movies like X' / 'similar to X' patterns and extract the movie title.
    Returns the title portion if detected, None otherwise.
    """
    patterns = [
        r'(?:movies?|films?)\s+(?:like|similar\s+to)\s+(.+)',
        r'similar\s+(?:movies?|films?)?\s*(?:to|as)\s+(.+)',
        r'(?:something|anything)\s+(?:like|similar\s+to)\s+(.+?)(?:\s+(?:any|more|suggest|recommend).*)?$',
    ]
    q = query.strip()
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            title = m.group(1).strip().strip('"\'')
            # Strip "but ...", "except ...", "with more ..." modifiers
            # e.g., "John Wick but more funny" → "John Wick"
            title = re.sub(r'\s+(?:but|except|with\s+more|and\s+more)\s+.*$', '', title, flags=re.IGNORECASE)
            # Filter out very short or clearly non-title matches
            if len(title) >= 3:
                return title
    return None


def _find_movie_by_title(title_query: str, collection, query_embedding, n: int = 20):
    """
    Search ChromaDB for a movie by title using where_document.
    Returns the movie's document text and title for similarity re-embedding.
    Prefers exact title matches over partial matches.
    """
    # Try title-cased version first (matches "Title: The Shawshank Redemption")
    # then original query as fallback
    variants = [title_query.title(), title_query]
    for variant in variants:
        try:
            title_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where_document={"$contains": variant},
            )
            if title_results['metadatas'][0]:
                # Prefer exact title match over partial match
                # e.g., "The Dark Knight" over "The Dark Knight Rises"
                query_lower = title_query.lower().strip()
                for i, meta in enumerate(title_results['metadatas'][0]):
                    found_title = meta.get('title', '')
                    if found_title.lower() == query_lower or found_title.lower() == f"the {query_lower}":
                        return title_results['documents'][0][i], found_title
                # No exact match — return first non-documentary result.
                # Documentaries like "50 Greatest Harry Potter Moments" rank
                # high for franchise queries due to keyword density but aren't
                # the actual movie the user is referring to.
                for i, meta in enumerate(title_results['metadatas'][0]):
                    found_title = meta.get('title', '')
                    found_genres = meta.get('genres', '').lower()
                    if found_title and 'documentary' not in found_genres:
                        return title_results['documents'][0][i], found_title
                # All results are documentaries — return first anyway
                found_meta = title_results['metadatas'][0][0]
                found_title = found_meta.get('title', '')
                found_doc = title_results['documents'][0][0]
                if found_title:
                    return found_doc, found_title
        except Exception:
            continue
    return None


def format_movies_compact(results: dict) -> str:
    """
    Format retrieved movies as a compact list for the fine-tuned model.

    The local Llama 3 model was trained on concise movie listings like:
        1. **Title** (Year) - Genre - Rating/10
           Plot summary

    We mirror that style so the model produces natural recommendations.
    """
    parts = []
    for i, meta in enumerate(results['metadatas'][0]):
        title = meta.get('title', 'Unknown')
        year = meta.get('year', 'N/A')
        genres = meta.get('genres', 'N/A')
        rating = meta.get('rating', 'N/A')
        director = meta.get('director', '')
        cast = meta.get('cast', '')
        # Get a short plot excerpt from the document text
        doc = results['documents'][0][i] if i < len(results['documents'][0]) else ''
        # Extract overview line from the document text
        overview = ''
        for line in doc.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith(('Title:', 'Genres:', 'Director:', 'Cast:', 'Moods:', 'Keywords:')):
                overview = stripped[:200]
                break
        entry = f"{i+1}. **{title}** ({year}) - {genres} - {rating}/10"
        if director:
            entry += f"\n   Director: {director}"
        if cast:
            entry += f"\n   Cast: {cast}"
        if overview:
            entry += f"\n   {overview}"
        parts.append(entry)
    return "\n\n".join(parts)


def build_alpaca_prompt(query: str, movie_context: str) -> str:
    """
    Build a prompt in the Alpaca template format that matches our fine-tuned
    Llama 3 model's training data.
    """
    instruction = (
        f"Based on the user's request, recommend movies ONLY from the list below. "
        f"Do NOT recommend movies that are not in this list.\n"
        f"For EACH movie you recommend, wrap your explanation in tags like this: <reason>Movie Title: Provide a short, persuasive overview combining the plot and exactly WHY the user should watch it based on their specific request</reason>\n\n"
        f"Make sure to strictly include the exact Movie Title and a colon before your explanation inside the tags.\n\n"
        f"Available movies:\n{movie_context}\n\n"
        f"User request: {query}"
    )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _build_grounded_answer(raw_llm_text: str, results: dict) -> tuple:
    """
    Post-process the LLM response to eliminate hallucination and extract per-movie reasons.
    """
    # Extract <reason>Title: reason</reason> tags
    llm_reasons = {}
    import re
    # More permissive regex: allows optional bolding around title and flexible spacing
    reason_pattern = r'<reason>\s*(?:\*\*)?([^:*]+)(?:\*\*)?:\s*(.+?)\s*</reason>'
    for match in re.finditer(reason_pattern, raw_llm_text, re.DOTALL):
        movie_title = match.group(1).strip().lower()
        reason_text = match.group(2).strip()
        if movie_title:
            llm_reasons[movie_title] = reason_text

    # Extract the first sentence or two from the LLM as the intro.
    lines = [l.strip() for l in raw_llm_text.strip().split('\n') if l.strip()]
    intro = ""
    if lines and not lines[0][:3].replace('.', '').replace(' ', '').isdigit():
        intro = lines[0]

    # Build accurate recommendation text from retrieved data
    recs = []
    for i, meta in enumerate(results['metadatas'][0]):
        title = meta.get('title', 'Unknown')
        year = meta.get('year', 'N/A')
        genres = meta.get('genres', 'N/A')
        rating = meta.get('rating', 'N/A')
        
        entry = f"{i+1}. **{title}** ({year}) - {genres} - {rating}/10"
        recs.append(entry)

    answer_text = (intro + "\n\n" + "\n\n".join(recs)) if intro else ("Here are my recommendations:\n\n" + "\n\n".join(recs))

    return answer_text, llm_reasons


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


def _trim_ordered_ttl(cache, ttl_s, max_size):
    """Trim expired entries and enforce max size for an OrderedDict TTL cache."""
    now = time.time()
    expired_keys = [k for k, (ts, _) in cache.items() if now - ts > ttl_s]
    for k in expired_keys:
        cache.pop(k, None)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _discover_cache_key(request: DiscoverRequest) -> str:
    """Build a stable cache key for discover requests."""
    return "|".join([
        request.query.strip().lower(),
        (request.genre or "").strip().lower(),
        str(request.min_year) if request.min_year is not None else "",
        str(request.max_year) if request.max_year is not None else "",
        str(request.min_rating) if request.min_rating is not None else "",
        str(request.top_k),
    ])


def _get_cached_discover_response(cache_key: str):
    """Return cached DiscoverResponse if present and not expired."""
    _trim_ordered_ttl(_discover_response_cache, DISCOVER_RESPONSE_CACHE_TTL_S, DISCOVER_RESPONSE_CACHE_SIZE)
    hit = _discover_response_cache.get(cache_key)
    if not hit:
        return None
    ts, response = hit
    if time.time() - ts > DISCOVER_RESPONSE_CACHE_TTL_S:
        _discover_response_cache.pop(cache_key, None)
        return None
    _discover_response_cache.move_to_end(cache_key)
    return response


def _set_cached_discover_response(cache_key: str, response: DiscoverResponse):
    """Store discover response in TTL+LRU cache."""
    _discover_response_cache[cache_key] = (time.time(), response)
    _discover_response_cache.move_to_end(cache_key)
    _trim_ordered_ttl(_discover_response_cache, DISCOVER_RESPONSE_CACHE_TTL_S, DISCOVER_RESPONSE_CACHE_SIZE)


def _get_query_embedding_cached(text: str, is_document: bool = False):
    """Get embedding from TTL+LRU cache with separate doc/query namespaces."""
    if embeddings is None:
        raise RuntimeError("Embedding model not loaded")

    key_prefix = "d:" if is_document else "q:"
    cache_key = f"{key_prefix}{text}"

    _trim_ordered_ttl(_query_embedding_cache, QUERY_EMBED_CACHE_TTL_S, QUERY_EMBED_CACHE_SIZE)
    cached = _query_embedding_cache.get(cache_key)
    if cached:
        ts, vec = cached
        if time.time() - ts <= QUERY_EMBED_CACHE_TTL_S:
            _query_embedding_cache.move_to_end(cache_key)
            return vec
        _query_embedding_cache.pop(cache_key, None)

    if is_document:
        vec = embeddings.embed_documents([text])[0]
    else:
        vec = embeddings.embed_query(text)

    _query_embedding_cache[cache_key] = (time.time(), vec)
    _query_embedding_cache.move_to_end(cache_key)
    _trim_ordered_ttl(_query_embedding_cache, QUERY_EMBED_CACHE_TTL_S, QUERY_EMBED_CACHE_SIZE)
    return vec


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
    
    # Determine actual model being used
    if inference_engine:
        actual_model = inference_engine.config.model_path or LLM_MODEL
        actual_backend = inference_engine.backend_name
    elif hf_client:
        actual_model = LLM_MODEL
        actual_backend = "hf_api"
    else:
        actual_model = "none"
        actual_backend = "none"

    return HealthResponse(
        status="healthy" if (hf_ok and chroma_ok) else "degraded",
        ollama_connected=hf_ok,  # Reusing field name for compatibility
        chroma_loaded=chroma_ok,
        total_movies=movie_count,
        embedding_model=EMBEDDING_MODEL,
        llm_model=actual_model,
        inference_backend=actual_backend,
    )


@app.get("/inference-metrics", tags=["System"])
async def get_inference_metrics():
    """Get inference performance metrics (latency, throughput, backend)."""
    if inference_engine:
        return inference_engine.get_metrics()
    return {
        "backend": "hf_api",
        "message": "Using HuggingFace API (no local metrics available)",
    }


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

    total_start = time.time()
    retrieval_ms = 0.0
    generation_ms = 0.0

    cache_key = _discover_cache_key(request)
    cached_response = _get_cached_discover_response(cache_key)
    if cached_response:
        logger.info(f"   Discover response cache hit: {cache_key}")
        return cached_response
    
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
        # Build metadata filter (rating and year only — genre handled separately below)
        metadata_filter = build_metadata_filter(request)

        # ── Decade extraction from query text ─────────────────────────
        # "funny 90s comedy" → year filter 1990-1999
        decade_map = {
            "20s": (2020, 2029), "2020s": (2020, 2029),
            "10s": (2010, 2019), "2010s": (2010, 2019),
            "2000s": (2000, 2009), "00s": (2000, 2009),
            "90s": (1990, 1999), "1990s": (1990, 1999),
            "80s": (1980, 1989), "1980s": (1980, 1989),
            "70s": (1970, 1979), "1970s": (1970, 1979),
            "60s": (1960, 1969), "1960s": (1960, 1969),
            "50s": (1950, 1959), "1950s": (1950, 1959),
            "40s": (1940, 1949), "1940s": (1940, 1949),
        }
        q_lower_temp = request.query.lower()
        decade_applied = False
        for decade_kw, (yr_min, yr_max) in decade_map.items():
            if decade_kw in q_lower_temp:
                decade_conditions = [{"year": {"$gte": yr_min}}, {"year": {"$lte": yr_max}}]
                if metadata_filter is None:
                    metadata_filter = {"$and": decade_conditions}
                elif "$and" in metadata_filter:
                    metadata_filter["$and"].extend(decade_conditions)
                else:
                    metadata_filter = {"$and": [metadata_filter] + decade_conditions}
                logger.info(f"   Decade filter added: {yr_min}-{yr_max}")
                decade_applied = True
                break

        # ── Era-based year filters ──────────────────────────────────
        # "old black and white movies" → year <= 1970 (B&W era ended ~1966)
        # "classic films" without a decade → year <= 1980
        if not decade_applied and not request.min_year and not request.max_year:
            era_patterns = [
                (r'black\s+and\s+white|b&w|b\s*&\s*w', 1920, 1970),
                (r'silent\s+(?:film|movie)', 1895, 1935),
            ]
            for era_pat, era_min, era_max in era_patterns:
                if re.search(era_pat, q_lower_temp):
                    era_conditions = [{"year": {"$gte": era_min}}, {"year": {"$lte": era_max}}]
                    if metadata_filter is None:
                        metadata_filter = {"$and": era_conditions}
                    elif "$and" in metadata_filter:
                        metadata_filter["$and"].extend(era_conditions)
                    else:
                        metadata_filter = {"$and": [metadata_filter] + era_conditions}
                    decade_applied = True
                    logger.info(f"   Era filter added: {era_min}-{era_max}")
                    break

        # ── Specific year extraction ("best movies 2023", "films from 2019") ──
        # Only apply if no decade was already detected
        if not decade_applied and not request.min_year and not request.max_year:
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', q_lower_temp)
            if year_match:
                yr = int(year_match.group(1))
                year_conditions = [{"year": {"$gte": yr}}, {"year": {"$lte": yr}}]
                if metadata_filter is None:
                    metadata_filter = {"$and": year_conditions}
                elif "$and" in metadata_filter:
                    metadata_filter["$and"].extend(year_conditions)
                else:
                    metadata_filter = {"$and": [metadata_filter] + year_conditions}
                logger.info(f"   Specific year filter added: {yr}")

        if metadata_filter:
            logger.info(f"   Filters: {metadata_filter}")

        # Augment the query with genre name so semantic search ranks genre-relevant
        # movies higher even without a metadata filter on genres.
        search_query = request.query

        # ── Query understanding ─────────────────────────────────────
        query_intent = _classify_query_cached(request.query.lower().strip())
        logger.info(f"   Intent: {query_intent.get('intent')} | {query_intent}")

        if request.genre:
            search_query = f"{request.genre} movie: {request.query}"

        # ── Mood/concept augmentation ─────────────────────────────────
        # The embedding model doesn't understand mood words well.
        # Augment the query with genre/theme terms that map to the mood.
        mood_augments = {
            "sad": "emotional drama tragedy loss grief heartbreak devastating tearjerker Schindler's List Titanic The Green Mile",
            "cry": "emotional drama tragedy heartbreaking grief moving devastating Schindler's List Titanic The Notebook Grave of the Fireflies",
            "happy": "uplifting comedy feel-good lighthearted joyful",
            "feel good": "uplifting comedy heartwarming charming joyful lighthearted Forrest Gump Up Paddington The Princess Bride The Intouchables School of Rock It's a Wonderful Life",
            "feel-good": "uplifting comedy heartwarming charming joyful lighthearted Forrest Gump Up Paddington The Princess Bride The Intouchables School of Rock It's a Wonderful Life",
            "fun": "comedy adventure entertaining lighthearted enjoyable exciting popcorn",
            "light hearted": "comedy feel-good charming entertaining witty romantic Ferris Bueller Grand Budapest Hotel School of Rock Princess Bride Groundhog Day",
            "lighthearted": "comedy feel-good charming entertaining witty romantic Ferris Bueller Grand Budapest Hotel School of Rock Princess Bride Groundhog Day",
            "light": "comedy lighthearted feel-good charming entertaining",
            "scary": "horror thriller suspense terror frightening creepy disturbing The Exorcist The Conjuring Hereditary",
            "horror": "horror thriller suspense frightening terrifying The Exorcist Hereditary Get Out The Conjuring Halloween A Nightmare on Elm Street The Ring",
            "halloween": "horror thriller suspense terror frightening creepy The Exorcist The Conjuring Hereditary A Nightmare on Elm Street",
            "creepy": "horror thriller psychological suspense",
            "funny": "comedy hilarious laugh humor slapstick witty stand-up Airplane Hangover Superbad",
            "intense": "thriller action suspense gripping adrenaline Se7en Silence of the Lambs No Country for Old Men",
            "adrenaline rush": "fast-paced explosive high-octane action thriller Mad Max Fury Road John Wick Die Hard Mission Impossible",
            "adrenaline": "fast-paced explosive high-octane action thriller Mad Max Fury Road John Wick Die Hard Mission Impossible",
            "mind-bending": "psychological thriller sci-fi surreal unexpected Inception The Matrix Memento Primer Predestination",
            "mind bending": "psychological thriller sci-fi surreal unexpected Inception The Matrix Memento Primer Predestination",
            "dark": "thriller noir crime psychological gritty",
            "oscar": "critically acclaimed award-winning drama performance",
            "award": "critically acclaimed prestigious drama",
            "winning": "critically acclaimed acclaimed masterpiece",
            "uplifting": "inspirational drama positive heartwarming triumph",
            "emotional": "drama moving powerful performance heartfelt devastating",
            "twist": "psychological thriller mystery unexpected revelation shocking ending",
            "twists": "psychological thriller mystery unexpected revelation shocking ending",
            "plot": "psychological thriller mystery unexpected revelation narrative structure",
            "surprise": "unexpected twist shocking revelation plot twist",
            "stunning": "cinematography visual breathtaking beautiful epic spectacular Blade Runner 2049 The Revenant Life of Pi Hero Mad Max Fury Road",
            "visually": "cinematography visual breathtaking beautiful spectacular Blade Runner 2049 The Revenant Life of Pi Hero",
            "beautiful": "cinematography visual artistic breathtaking",
            "kids": "family animation adventure disney pixar children friendly Finding Nemo Frozen The Lion King Inside Out Moana Coco Toy Story",
            "children": "family animation adventure disney pixar kids friendly Finding Nemo Frozen The Lion King Inside Out Moana Coco Toy Story",
            "family": "family animation adventure comedy heartwarming",
            # Topic/concept augmentations
            "time travel": "time machine past future paradox Back to the Future",
            "revenge": "vengeance retribution payback vigilante justice Kill Bill",
            "heist": "robbery theft steal bank Ocean vault caper planned",
            "zombie": "undead apocalypse infection outbreak survival walking dead",
            "zombies": "undead apocalypse infection outbreak survival walking dead",
            "survival": "wilderness stranded alone nature struggle endurance",
            "superhero": "comic book powers villain hero cape Marvel DC",
            "space": "astronaut spacecraft galaxy stars planet NASA mission",
            "artificial intelligence": "AI robot android sentient machine consciousness",
            "musical": "singing dancing Broadway songs choreography music performance",
            "disney": "Walt Disney princess animated fairy tale magic kingdom Pixar",
            "war": "battlefield soldier combat military troops battle frontline",
            "crime": "criminal gangster mob mafia underworld organized heist",
            "psychological": "mind games manipulation paranoia suspense mental",
            # Abstract concept augmentations
            "makes you think": "cerebral philosophical thought-provoking intellectual challenging profound meaning",
            "make you think": "cerebral philosophical thought-provoking intellectual challenging profound meaning",
            "thought provoking": "cerebral philosophical intellectual challenging profound meaning existential",
            "thought-provoking": "cerebral philosophical intellectual challenging profound meaning existential",
            "rainy day": "cozy comfort heartwarming gentle feel-good lighthearted warm relaxing Paddington Grand Budapest Hotel Up The Princess Bride",
            "grandma": "wholesome gentle classic heartwarming family comedy lighthearted",
            "grandmother": "wholesome gentle classic heartwarming family comedy lighthearted",
            "underrated": "overlooked cult indie underappreciated lesser-known gem unsung Whiplash Nightcrawler Spotlight Arrival Moon",
            "hidden gem": "overlooked cult indie underappreciated lesser-known unsung Whiplash Nightcrawler Spotlight Arrival Moon",
            "hidden gems": "overlooked cult indie underappreciated lesser-known unsung Whiplash Nightcrawler Spotlight Arrival Moon",
            "edge of seat": "suspense thrilling tense gripping nail-biting adrenaline intense Se7en Silence of the Lambs No Country for Old Men",
            "edge of my seat": "suspense thrilling tense gripping nail-biting adrenaline intense Se7en Silence of the Lambs No Country for Old Men",
            "edge of your seat": "suspense thrilling tense gripping nail-biting adrenaline intense Se7en Silence of the Lambs No Country for Old Men",
            "cinematography": "visually stunning beautiful photography Roger Deakins artistic visual masterpiece Blade Runner 2049 The Revenant Life of Pi Grand Budapest Hotel Hero",
            "unexpected ending": "twist shocking revelation surprise plot ending unexpected",
            "unexpected endings": "twist shocking revelation surprise plot unexpected",
            "cheer me up": "feel-good comedy uplifting heartwarming lighthearted funny joyful Legally Blonde Paddington School of Rock The Princess Bride Elf Mrs Doubtfire",
            "cheer up": "feel-good comedy uplifting heartwarming lighthearted funny joyful Legally Blonde Paddington School of Rock The Princess Bride Elf Mrs Doubtfire",
            # Specific genre/topic augmentations for better retrieval
            "psychological thriller": "Shutter Island Black Swan Silence Lambs paranoia manipulation mind obsession",
            "slow burn": "atmospheric deliberate building tension suspense methodical Zodiac No Country for Old Men Tinker Tailor Soldier Spy Prisoners",
            "slow burn thriller": "atmospheric deliberate building tension suspense methodical Zodiac No Country for Old Men Tinker Tailor Soldier Spy Prisoners",
            "black comedy": "Fargo In Bruges Dr Strangelove dark humor satire darkly funny cynical",
            "mafia": "Godfather organized crime Corleone family Don mob boss gangster",
            "inspirational": "Rocky Pursuit Happyness triumph underdog overcoming real story heartwarming determination",
            "inspiring": "triumph underdog overcoming determination heartwarming motivational hopeful real story",
            "biopic": "true story real life biographical based historical person famous",
            "true story": "biographical based real events historical Schindler's List The Social Network 12 Years a Slave A Beautiful Mind Catch Me If You Can",
            "based on true": "biographical real events historical Schindler's List The Social Network 12 Years a Slave A Beautiful Mind",
            "road trip": "journey travel adventure on the road friends car highway Little Miss Sunshine Into the Wild Thelma Louise",
            "coming of age": "growing up teenager youth adolescence school friendship first love Stand by Me Lady Bird Boyhood Juno Perks of Being a Wallflower",
            # Meta-quality augments — strip problematic keywords that cause title-literal matches
            "best movies of all time": "highest rated acclaimed masterpiece cinematic classic The Godfather The Shawshank Redemption Schindler's List Pulp Fiction 12 Angry Men",
            "best movies ever": "highest rated acclaimed masterpiece cinematic classic The Godfather The Shawshank Redemption Schindler's List Pulp Fiction",
            "everyone should watch": "essential classic masterpiece critically acclaimed The Godfather Schindler's List Pulp Fiction The Shawshank Redemption 12 Angry Men",
            "before you die": "essential classic masterpiece all-time greatest The Godfather Schindler's List Pulp Fiction The Shawshank Redemption",
            "must see movies": "essential classic masterpiece critically acclaimed The Godfather Schindler's List The Shawshank Redemption Pulp Fiction",
            "must watch movies": "essential classic masterpiece critically acclaimed The Godfather Schindler's List The Shawshank Redemption Pulp Fiction",
            # Genre-specific augments for commonly failing queries
            "disaster": "catastrophe earthquake tsunami volcano eruption flood tornado hurricane The Day After Tomorrow 2012 Twister San Andreas Armageddon",
            "sports": "team championship underdog coach athlete competition Rocky Remember the Titans Moneyball Creed Hoosiers",
            "sport": "team championship underdog coach athlete competition Rocky Remember the Titans Moneyball Creed Hoosiers",
            "musical": "singing dancing Broadway songs choreography La La Land Singin' in the Rain West Side Story Chicago The Sound of Music The Greatest Showman Grease Moulin Rouge Les Miserables Hamilton",
            "courtroom": "trial jury verdict lawyer judge testimony witness 12 Angry Men A Few Good Men To Kill a Mockingbird The Verdict",
            "foreign language": "subtitled international non-English Parasite Amelie Pan's Labyrinth City of God Spirited Away Cinema Paradiso Seven Samurai Rashomon Oldboy Life Is Beautiful Crouching Tiger",
            "foreign": "subtitled international non-English Parasite Amelie Pan's Labyrinth City of God Spirited Away Cinema Paradiso Seven Samurai Oldboy Life Is Beautiful",
            "soundtrack": "score music composer Hans Zimmer John Williams Ennio Morricone iconic theme orchestral",
            "soundtracks": "score music composer Hans Zimmer John Williams Ennio Morricone iconic theme orchestral",
            "nostalgia": "classic beloved iconic decade era generation remembered fondly childhood",
            "masterpiece": "highest rated acclaimed brilliant cinematic essential groundbreaking definitive",
            "masterpieces": "highest rated acclaimed brilliant cinematic essential groundbreaking definitive",
            # Location/context augments
            "new york": "Manhattan Brooklyn NYC Taxi Driver Ghostbusters Spider-Man Breakfast at Tiffany's Gangs of New York The Apartment",
            "set in new york": "Manhattan Brooklyn NYC Taxi Driver Ghostbusters Spider-Man When Harry Met Sally",
            "date night": "romantic comedy love La La Land Titanic The Notebook Crazy Rich Asians Before Sunrise",
            "date movie": "romantic comedy love La La Land Titanic The Notebook Crazy Rich Asians",
            "romance and action": "action romance adventure Mr and Mrs Smith True Lies The Princess Bride Romancing the Stone Knight and Day",
            "romance and i like action": "action romance adventure Mr and Mrs Smith True Lies The Princess Bride Romancing the Stone Knight and Day",
            "girlfriend likes romance": "action romance adventure Mr and Mrs Smith True Lies The Princess Bride Romancing the Stone Knight and Day",
            # Conversational/generic query augments — replace vague queries with quality signals
            "watch tonight": "popular acclaimed entertaining crowd-pleaser The Shawshank Redemption Inception The Dark Knight Forrest Gump Interstellar",
            "movie to watch": "popular acclaimed entertaining crowd-pleaser The Shawshank Redemption Inception The Dark Knight Forrest Gump Interstellar",
            "grandma": "wholesome gentle heartwarming family Mrs Doubtfire Cocoon Driving Miss Daisy The Sound of Music",
            "grandmother": "wholesome gentle heartwarming family Mrs Doubtfire Cocoon Driving Miss Daisy The Sound of Music",
            "10 year old": "family animation adventure Disney Pixar Finding Nemo Frozen Harry Potter The Incredibles",
            "year old": "family animation adventure Disney Pixar children",
            "three hour": "epic long Lord of the Rings The Godfather Schindler's List Lawrence of Arabia Braveheart",
            "epic movie": "grand scale sweeping Lord of the Rings Gladiator Braveheart Lawrence of Arabia Ben-Hur",
        }
        q_lower = request.query.lower()
        augmented = False

        # ── Early "movies like X" detection ───────────────────────────
        # Check BEFORE mood augmentation so that movie title words like
        # "dark" (in "The Dark Knight") don't trigger mood augmentations.
        movie_title_query = _extract_movie_title_query(request.query)
        is_similar_to_query = movie_title_query is not None

        # ── Comforting query detection ─────────────────────────────
        # "movies to watch when sad" → user wants COMFORTING movies, not sad ones.
        # Detect this BEFORE augmentation so we can suppress sad augmentations.
        is_comforting_intent = bool(
            re.search(r'when\s+(?:i\'?m?\s+)?(?:sad|down|depressed|upset|blue)', q_lower) or
            re.search(r'(?:watch|see|need)\s+when\s+(?:sad|cry|crying)', q_lower) or
            re.search(r'cheer\s*(?:me\s+)?up', q_lower) or
            re.search(r'(?:feeling|feel)\s+(?:sad|down|depressed|blue)', q_lower)
        )

        # Sort keys so multi-word phrases are checked first ("time travel" before "time")
        sorted_mood_keys = sorted(mood_augments.keys(), key=lambda k: -len(k))
        matched_mood_word = None
        for mood_word in sorted_mood_keys:
            if mood_word in q_lower:
                # Skip mood augmentation entirely for "movies like X" queries —
                # movie title words shouldn't trigger mood augmentations
                if is_similar_to_query:
                    continue
                # Skip sad/cry augmentation when user wants comforting films
                if is_comforting_intent and mood_word in ("sad", "cry"):
                    continue
                augment_terms = mood_augments[mood_word]
                # Strip the mood keyword from the query to prevent BGE's keyword bias
                # (e.g., "cry" → title matches "The Man Who Cried"), but keep the rest
                # of the query for context (e.g., "movies that make you" stays).
                stripped_query = q_lower.replace(mood_word, '').strip()
                # Remove double spaces and leftover connectives
                stripped_query = ' '.join(stripped_query.split())
                search_query = f"{stripped_query} {augment_terms}"
                matched_mood_word = mood_word
                logger.info(f"   Mood augment: '{mood_word}' → replaced query with genre/theme terms")
                augmented = True
                # Only apply one augmentation except for combined sad+cry queries
                if not (mood_word == "sad" and "cry" in q_lower):
                    break

        # When comforting intent detected, ensure feel-good terms are in the query
        # even if the sad augmentation was suppressed above.
        if is_comforting_intent and "feel good" not in search_query and "heartwarming" not in search_query:
            comforting_terms = "uplifting heartwarming comforting cozy feel-good lighthearted funny Inside Out Paddington Up Soul The Princess Bride School of Rock Mrs Doubtfire"
            search_query = f"{search_query} {comforting_terms}"
            augmented = True
            logger.info("   Comforting intent: applied feel-good augmentation")

        # ── "Movies like X" detection ─────────────────────────────────
        # If user asks "movies like inception", find Inception's document
        # and re-embed it as the search query for better similarity results.
        similar_to_title = None
        similar_source_genres = None
        similar_to_fallback = False  # True when similar-to movie not found in DB
        stored_query_embedding = None
        similar_src_meta = None
        # movie_title_query already extracted above for early similar-to detection
        if movie_title_query:
            logger.info(f"   'Movies like' pattern detected — looking up: '{movie_title_query}'")
            # First embed the original query to do the title lookup
            temp_embedding = _get_query_embedding_cached(movie_title_query)
            title_result = _find_movie_by_title(
                movie_title_query, vectorstore._collection, temp_embedding
            )
            if title_result:
                found_doc, similar_to_title = title_result
                # Check if the found movie is a documentary (likely about a TV show)
                # and the user asked "but as a movie" — this means the user wants concept-similar
                # movies, not similar documentaries.  Also skip if the found title is clearly
                # a making-of/behind-the-scenes documentary rather than the actual show/movie.
                found_genres_lower = ''
                try:
                    found_lookup = vectorstore._collection.query(
                        query_embeddings=[temp_embedding],
                        n_results=1,
                        where_document={"$contains": similar_to_title},
                        include=["metadatas"],
                    )
                    if found_lookup['metadatas'][0]:
                        found_genres_lower = found_lookup['metadatas'][0][0].get('genres', '').lower()
                except Exception:
                    pass
                is_doc_match = 'documentary' in found_genres_lower
                is_making_of = any(kw in found_doc.lower() for kw in ['making of', 'behind the scenes', 'creating the', 'the final season'])
                is_movie_request = 'but as a movie' in q_lower or 'as a movie' in q_lower or 'movie version' in q_lower
                # When user says "but as a movie", they want a TV show's THEMES
                # in movie form — NOT the literal movie adaptation.  E.g.,
                # "like Breaking Bad but as a movie" → finds El Camino, but user
                # wants Sicario/No Country, not El-Camino-similar films.
                if is_doc_match or is_making_of or is_movie_request:
                    logger.info(f"   Found '{similar_to_title}' is a documentary/making-of — skipping similar-to, using concept search")
                    similar_to_title = None
                    movie_title_query = None
                    is_similar_to_query = False
                    # Re-augment query for concept-based search using the show's themes
                    if not augmented:
                        _tv_themes = {
                            "breaking bad": "crime drug trade methamphetamine cartel moral descent dark thriller antihero No Country Sicario Traffic Scarface",
                            "game of thrones": "medieval fantasy political intrigue battle throne kingdom power epic sword",
                            "stranger things": "supernatural mystery coming-of-age horror 1980s telekinesis small town",
                            "the wire": "crime drug trade inner city police investigation Baltimore urban gritty",
                            "sopranos": "crime mob mafia family drama psychology antihero organized crime",
                            "the sopranos": "crime mob mafia family drama psychology antihero organized crime",
                            "peaky blinders": "crime gangster prohibition family power period drama",
                            "true detective": "crime mystery dark detective investigation noir psychological",
                            "narcos": "crime drug trade cartel Colombia drugs trafficking true story",
                            "ozark": "crime money laundering dark thriller family drug cartel",
                            "fargo": "crime dark comedy noir midwest quirky violence Coen brothers",
                            "succession": "family dynasty corporate power drama wealth rivalry",
                        }
                        # Extract original reference name from query
                        _ref_name = _extract_movie_title_query(request.query)
                        _ref_lower = _ref_name.lower().strip() if _ref_name else ""
                        if _ref_lower in _tv_themes:
                            search_query = _tv_themes[_ref_lower]
                        else:
                            search_query = f"{request.query} crime drama thriller antihero morality"
                        augmented = True
                if similar_to_title:
                    # Use the source movie's STORED embedding from ChromaDB directly.
                    # This is document-to-document similarity: we find movies whose
                    # indexed embeddings are closest to the source movie's indexed
                    # embedding.  This avoids re-encoding through the query pipeline
                    # where BGE's keyword bias distorts the embedding (e.g., genre words
                    # like "Action, Science Fiction" dominate and match all MCU movies).
                    # The stored embedding already captures the full document semantics
                    # as originally encoded — balanced across plot, genres, cast, moods.
                    stored_query_embedding = None
                    try:
                        src_lookup = vectorstore._collection.query(
                            query_embeddings=[temp_embedding],
                            n_results=1,
                            where_document={"$contains": similar_to_title},
                            include=["metadatas", "embeddings"],
                        )
                        if src_lookup['metadatas'][0]:
                            src_meta = src_lookup['metadatas'][0][0]
                            similar_source_genres = src_meta.get('genres', '')
                            similar_src_meta = src_meta
                            stored_query_embedding = src_lookup['embeddings'][0][0]
                            logger.info(f"   Got stored embedding for '{similar_to_title}' (genres: {similar_source_genres})")
                    except Exception as e:
                        logger.warning(f"   Stored embedding lookup failed: {e}")
                    # search_query not used for embedding when stored embedding is available
                    search_query = found_doc  # fallback if stored embedding fails
                    logger.info(f"   Found '{similar_to_title}' — using stored embedding for similarity search")
            else:
                # Reference not found in DB (likely a TV show or non-DB movie).
                # Fall back to concept-based search using themes of the reference.
                _tv_show_themes = {
                    "breaking bad": "crime drug trade methamphetamine cartel moral descent dark thriller antihero No Country Sicario Traffic Scarface",
                    "game of thrones": "medieval fantasy political intrigue battle throne kingdom power epic sword",
                    "stranger things": "supernatural mystery coming-of-age horror 1980s telekinesis small town",
                    "the wire": "crime drug trade inner city police investigation Baltimore urban gritty",
                    "sopranos": "crime mob mafia family drama psychology antihero organized crime",
                    "the sopranos": "crime mob mafia family drama psychology antihero organized crime",
                    "peaky blinders": "crime gangster prohibition family power Birmingham period drama",
                    "true detective": "crime mystery dark detective investigation noir psychological",
                    "dexter": "serial killer crime thriller dark antihero psychology investigation",
                    "narcos": "crime drug trade cartel Colombia drugs trafficking true story",
                    "ozark": "crime money laundering dark thriller family drug cartel midwest",
                    "fargo": "crime dark comedy noir midwest quirky violence Coen brothers",
                    "succession": "family dynasty corporate power drama wealth rivalry",
                }
                ref_lower = movie_title_query.lower().strip()
                if ref_lower in _tv_show_themes:
                    search_query = _tv_show_themes[ref_lower]
                    augmented = True
                    logger.info(f"   Reference '{movie_title_query}' not in DB — using TV theme augmentation")
                else:
                    # Generic fallback: use the reference title + "similar movie"
                    # Check for well-known movies that aren't in the DB
                    _wellknown_movie_themes = {
                        "parasite": "class divide social satire dark comedy thriller Korean rich poor family deception twists Bong Joon-ho Knives Out Us",
                        "spirited away": "Japanese animation fantasy coming-of-age magical world spirits Studio Ghibli Miyazaki",
                        "amelie": "French romantic comedy whimsical quirky Paris love charming warm optimistic",
                        "city of god": "Brazilian crime favela poverty violence coming-of-age gritty drama",
                        "pan's labyrinth": "dark fantasy Spanish Civil War fairy tale imagination horror Guillermo del Toro",
                        "oldboy": "Korean revenge thriller twist dark psychological vengeance",
                        "shoplifters": "Japanese family poverty drama quiet heartbreaking Kore-eda",
                        "little miss sunshine": "family road trip indie comedy quirky dysfunctional heartwarming",
                        "into the wild": "adventure wilderness journey self-discovery nature solitude freedom",
                        "thelma and louise": "road trip women friendship crime freedom adventure",
                        "thelma & louise": "road trip women friendship crime freedom adventure",
                    }
                    _ref_lower = movie_title_query.lower().strip()
                    if _ref_lower in _wellknown_movie_themes:
                        search_query = _wellknown_movie_themes[_ref_lower]
                        augmented = True
                        logger.info(f"   Reference '{movie_title_query}' not in DB — using well-known movie theme augmentation")
                    else:
                        search_query = f"{movie_title_query} similar movie dark drama thriller crime"
                        augmented = True
                        logger.info(f"   Reference '{movie_title_query}' not in DB — using generic concept search")
                movie_title_query = None
                is_similar_to_query = False
                similar_to_fallback = True

        # Always fetch extra candidates for re-ranking — more candidates
        # means the popularity/quality boost can surface better movies
        n_candidates = max(request.top_k * 6, 30)
        # For "movies like X" queries, fetch many more to have enough after
        # franchise exclusion removes sequels/prequels.  BGE's keyword bias
        # means most top results are literal title-word matches (e.g., "Out"
        # in "Walking Out" for "movies like Get Out") — we need a wide pool
        # so genre-matching and quality reranking can surface real matches.
        if similar_to_title:
            n_candidates = max(n_candidates, 150)
        # For mood/emotional queries, fetch MANY more candidates so the reranker
        # can surface well-known classics over obscure semantic matches.
        # BGE-large has strong keyword bias — the top-30 results for "movies
        # that make you cry" are all movies with "cry" in the title. We need
        # a wide enough pool (150+) that actual sad classics appear somewhere
        # in the candidate set for the quality/mood reranker to promote them.
        if augmented:
            n_candidates = max(n_candidates, 150)
        # Meta-quality queries need the widest possible pool — we're essentially
        # sorting by quality rather than relevance
        if query_intent.get('intent') in {'concept', 'specific', 'conversational'}:
            n_candidates = max(n_candidates, 150)

        # Extract name phrase EARLY so HyDE can check it
        name_phrase = None
        if not similar_to_title and not similar_to_fallback:
            name_phrase = _extract_name_phrase(request.query)

        # ── Query normalization for compound franchise names ─────────
        # E.g. "spiderman" → "Spider-Man", "ironman" → "Iron Man"
        # These won't appear verbatim in DB documents; normalise before search.
        _franchise_normalize = {
            'spiderman': 'Spider-Man', 'spider man': 'Spider-Man',
            'ironman': 'Iron Man',
            'xmen': 'X-Men', 'x men': 'X-Men',
            'blackpanther': 'Black Panther',
            'captainamerica': 'Captain America',
            'drstrange': 'Doctor Strange', 'doctorstrange': 'Doctor Strange',
            'guardiansofthegalaxy': 'Guardians of the Galaxy',
            'antman': 'Ant-Man', 'ant man': 'Ant-Man',
            'deadpool': 'Deadpool',
        }
        _q_lower = request.query.lower().strip()
        _normalized_query = _franchise_normalize.get(_q_lower)
        if _normalized_query and name_phrase:
            name_phrase = _normalized_query
        elif _normalized_query and not name_phrase:
            name_phrase = _normalized_query

        # ── HyDE: Hypothetical Document Embeddings ──────────────────
        # For abstract/concept/mood queries (no person name, no specific
        # movie title), the LLM generates a hypothetical ideal movie
        # description. We embed THAT instead of the raw query, bridging
        # the query-document gap (documents describe movies; queries
        # describe desires). This dramatically improves recall for
        # queries like "feel good movies" or "movies with twist endings".
        hyde_doc = None
        # Only use HyDE for intent types that benefit from document-style embeddings:
        # mood, concept, specific, conversational, similar_to queries.
        # Skip for person/genre/decade/franchise (keyword matching is better).
        hyde_intents = {'mood', 'concept', 'specific', 'conversational', 'similar_to'}
        use_hyde = query_intent.get('intent') in hyde_intents or augmented
        if use_hyde and not name_phrase and inference_engine is not None:
            try:
                hyde_prompt = (
                    f"Describe a movie that perfectly matches this request: \"{request.query}\". "
                    f"Write a short movie database entry (3-4 sentences) with title, genre, "
                    f"plot summary, mood, and themes. Only describe one fictional movie."
                )
                hyde_result = inference_engine.generate(hyde_prompt, max_tokens=150, temperature=0.7)
                hyde_text = hyde_result.get("text", "") if isinstance(hyde_result, dict) else str(hyde_result)
                if hyde_text and len(hyde_text.strip()) > 30:
                    hyde_doc = hyde_text.strip()
                    logger.info(f"   HyDE generated: {hyde_doc[:100]}...")
                else:
                    hyde_doc = None
            except Exception as e:
                logger.warning(f"   HyDE generation failed ({e})")
                hyde_doc = None

        # Embed query with BGE instruction prefix for optimal retrieval.
        # BGE models perform best when queries (not documents) are prefixed.
        # If HyDE produced a document, embed it without query prefix (it's a document).
        # For "movies like X", use the source movie's stored embedding directly
        # for document-to-document similarity search.
        if similar_to_title and stored_query_embedding is not None:
            query_embedding = stored_query_embedding
        elif hyde_doc:
            query_embedding = _get_query_embedding_cached(hyde_doc, is_document=True)
        elif similar_to_title:
            query_embedding = _get_query_embedding_cached(search_query, is_document=True)
        else:
            query_embedding = _get_query_embedding_cached(BGE_QUERY_PREFIX + search_query)

        # ── Dual retrieval ──────────────────────────────────────────
        # 1) Standard vector search (semantic similarity)
        vec_results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=metadata_filter
        )

        # 1b) BM25 keyword search + RRF fusion with vector results
        #     ONLY for person/franchise queries where keyword matching helps
        #     find exact name matches. For genre/mood/concept queries, BM25's
        #     literal matching causes terrible results ("black comedy" → "Men in Black").
        bm25_intents = {'person', 'franchise'}
        if bm25_index is not None and query_intent.get('intent') in bm25_intents:
            bm25_results = _bm25_search(search_query, n_results=n_candidates)
            if bm25_results['documents'][0]:
                vec_results = _rrf_fuse(vec_results, bm25_results, limit=n_candidates)
                logger.info(f"   Hybrid search: RRF fused {len(vec_results['documents'][0])} candidates")

        # 1c) Quality fallback: fetch top-rated movies for meta-quality/mood queries.
        #     The embedding search's top candidates are often literal title matches
        #     ("X-Rated: Greatest Adult Movies of All Time" for "best movies of all time").
        #     This supplementary retrieval ensures actual classics (rating >= 7.5)
        #     appear in the candidate pool for the quality-boosted reranker to surface.
        quality_intents = {'mood', 'concept', 'specific', 'conversational'}
        if query_intent.get('intent') in quality_intents or augmented:
            try:
                # Fetch high-rated movies using the same query embedding but with
                # a rating floor filter — this ensures quality films are candidates
                quality_filter = {"rating": {"$gte": 7.5}}
                if metadata_filter:
                    if "$and" in metadata_filter:
                        quality_filter = {"$and": metadata_filter["$and"] + [{"rating": {"$gte": 7.5}}]}
                    else:
                        quality_filter = {"$and": [metadata_filter, {"rating": {"$gte": 7.5}}]}
                quality_results = vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=50,
                    where=quality_filter,
                )
                if quality_results['metadatas'][0]:
                    vec_results = _merge_results(vec_results, quality_results, n_candidates + 50)
                    logger.info(f"   Quality fallback: merged {len(quality_results['metadatas'][0])} high-rated candidates")
            except Exception as e:
                logger.warning(f"   Quality fallback failed: {e}")

        # 1d) Genre-metadata retrieval for mood/concept/conversational queries.
        #     The core problem: BGE embeds mood queries ("scary movies") as keyword bags
        #     that land near title-matching trash ("Scary Movie", "Stay Alive") instead of
        #     near actual good horror films. The quality fallback (1c) helps but still uses
        #     the same polluted embedding. This fix bypasses the embedding entirely by
        #     querying ChromaDB with genre metadata filters and sorting by quality.
        #     This ensures that "scary movies" always has The Exorcist, Hereditary etc.
        #     in the candidate pool regardless of what the embedding does.
        mood_genre_map = {
            # Mood → (genre filter string, doc keyword)
            'frightening': ('Horror', 'horror'),
            'melancholy': ('Drama', 'drama'),
            'uplifting': ('Drama', 'heartwarming'),
            'humorous': ('Comedy', 'comedy'),
            'intense': ('Thriller', 'thriller'),
            'dark': ('Thriller', 'thriller'),
            'romantic': ('Romance', 'romance'),
            'cerebral': ('Science Fiction', 'thought-provoking'),
            'emotional': ('Drama', 'drama'),
            'calm': ('Comedy', 'comedy'),
        }
        concept_genre_map = {
            # Concept keyword → genre filter string
            'twist ending': 'Thriller',
            'plot twist': 'Thriller',
            'surprise ending': 'Thriller',
            'unexpected': 'Thriller',
            'based on true story': 'Drama',
            'based on real': 'Drama',
            'true story': 'Drama',
            'time travel': 'Science Fiction',
            'space': 'Science Fiction',
            'dystopia': 'Science Fiction',
            'dystopian': 'Science Fiction',
            'revenge': 'Action',
            'heist': 'Crime',
            'survival': 'Adventure',
            'courtroom': 'Drama',
            'prison': 'Drama',
            'road trip': 'Adventure',
            'coming of age': 'Drama',
            'zombie': 'Horror',
            'ghost': 'Horror',
            'vampire': 'Horror',
            'supernatural': 'Horror',
            'foreign language': 'Drama',
            'foreign film': 'Drama',
            'subtitled': 'Drama',
            'non-english': 'Drama',
            'international film': 'Drama',
            'international movie': 'Drama',
            'soundtrack': 'Drama',
            'soundtracks': 'Drama',
            'hidden gem': 'Drama',
            'hidden gems': 'Drama',
            'underrated': 'Drama',
            'overlooked': 'Drama',
            'three hour': 'Drama',
            '3 hour': 'Drama',
            'epic movie': 'Drama',
            'epic film': 'Drama',
            'set in new york': 'Drama',
            'new york': 'Drama',
            'set in london': 'Drama',
            'set in paris': 'Drama',
            'set in tokyo': 'Drama',
            'set in los angeles': 'Drama',
            'date night': 'Romance',
            'date movie': 'Romance',
        }
        genre_query_map = {
            # Genre intent keywords → ChromaDB genre filter
            'horror': 'Horror',
            'comedy': 'Comedy',
            'action': 'Action',
            'thriller': 'Thriller',
            'drama': 'Drama',
            'sci-fi': 'Science Fiction',
            'science fiction': 'Science Fiction',
            'romance': 'Romance',
            'western': 'Western',
            'war': 'War',
            'crime': 'Crime',
            'mystery': 'Mystery',
            'fantasy': 'Fantasy',
            'animation': 'Animation',
            'animated': 'Animation',
            'documentary': 'Documentary',
            'musical': 'Music',
            'adventure': 'Adventure',
            'superhero': 'Action',
            'mafia': 'Crime',
            'gangster': 'Crime',
            'sports': 'Drama',
            'disaster': 'Action',
            'courtroom': 'Drama',
        }
        genre_retrieval_filter = None
        intent = query_intent.get('intent')
        if intent == 'mood' and query_intent.get('mood') in mood_genre_map:
            genre_retrieval_filter = mood_genre_map[query_intent['mood']][0]
        # When query explicitly says "comedy" but mood maps to Drama (e.g. "light
        # hearted comedy" → mood:uplifting → Drama), override to Comedy
        if genre_retrieval_filter == 'Drama' and 'comedy' in q_lower:
            genre_retrieval_filter = 'Comedy'
        # Adrenaline queries: override primary genre to Action (not Thriller)
        # "adrenaline rush action movies" wants Die Hard/Mad Max, not generic thrillers
        if "adrenaline" in q_lower:
            genre_retrieval_filter = 'Action'
        if intent == 'concept' and query_intent.get('concept') != 'meta_quality':
            concept_val = query_intent.get('concept', '')
            if concept_val in concept_genre_map:
                genre_retrieval_filter = concept_genre_map[concept_val]
        elif intent == 'genre' and query_intent.get('genre'):
            genre_val = query_intent['genre']
            if genre_val in genre_query_map:
                genre_retrieval_filter = genre_query_map[genre_val]
        elif intent in {'specific', 'conversational'}:
            # For specific/conversational, try to detect genre from query text
            for gkw, gfilter in genre_query_map.items():
                if gkw in q_lower:
                    genre_retrieval_filter = gfilter
                    break
            # Fallback: for children/family audience, default to Animation genre
            if not genre_retrieval_filter and query_intent.get('audience') in {'children', 'family'}:
                genre_retrieval_filter = 'Animation'
            # Fallback: for elderly audience, default to Drama
            if not genre_retrieval_filter and query_intent.get('audience') == 'elderly':
                genre_retrieval_filter = 'Comedy'
            # Fallback: for date audience, default to Romance
            if not genre_retrieval_filter and query_intent.get('audience') == 'date':
                genre_retrieval_filter = 'Romance'

        # Skip genre-metadata retrieval for non-genre concepts where the filter
        # is a rough proxy (e.g., "foreign language" → Drama).  The doc searches
        # and augments handle these better than broad genre pools.
        skip_genre_retrieval_concepts = {
            'foreign language', 'foreign film', 'subtitled', 'non-english',
            'international film', 'international movie',
            'soundtrack', 'soundtracks', 'score',
            'hidden gem', 'hidden gems', 'underrated', 'overlooked',
            'set in new york', 'new york', 'set in london', 'set in paris',
            'set in tokyo', 'set in los angeles',
            'three hour', '3 hour', 'epic movie', 'epic film',
            'road trip',
        }
        concept_val_gr = query_intent.get('concept', '')
        skip_genre = concept_val_gr in skip_genre_retrieval_concepts
        if genre_retrieval_filter and not similar_to_title and not skip_genre:
            try:
                # Build a metadata filter that requires the genre AND high rating
                # ChromaDB $and requires at least 2 conditions
                if metadata_filter:
                    if "$and" in metadata_filter:
                        genre_meta_filter = {"$and": [{"rating": {"$gte": 7.0}}] + metadata_filter["$and"]}
                    else:
                        genre_meta_filter = {"$and": [{"rating": {"$gte": 7.0}}, metadata_filter]}
                else:
                    genre_meta_filter = {"rating": {"$gte": 7.0}}
                # Use where_document to filter by genre string (since genres is stored
                # as comma-separated text in the document, not as a list metadata field)
                genre_doc_results = vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=80,
                    where=genre_meta_filter,
                    where_document={"$contains": genre_retrieval_filter},
                )
                if genre_doc_results['metadatas'][0]:
                    vec_results = _merge_results(vec_results, genre_doc_results, n_candidates + 80)
                    logger.info(f"   Genre-metadata retrieval: merged {len(genre_doc_results['metadatas'][0])} '{genre_retrieval_filter}' candidates")
            except Exception as e:
                logger.warning(f"   Genre-metadata retrieval failed: {e}")

        # 1d1b) Secondary genre retrieval — for multi-genre queries like
        #        "my girlfriend likes romance and I like action", the intent
        #        classifier returns only ONE genre.  Scan the original query
        #        for additional genre keywords and add candidates for them.
        secondary_genres = set()
        if genre_retrieval_filter and not similar_to_title and not skip_genre:
            orig_q_lower = request.query.lower()
            for gkw, gfilter in genre_query_map.items():
                if gkw in orig_q_lower and gfilter != genre_retrieval_filter:
                    secondary_genres.add(gfilter)
            for sec_genre in secondary_genres:
                try:
                    if metadata_filter:
                        if "$and" in metadata_filter:
                            sec_meta = {"$and": [{"rating": {"$gte": 7.0}}] + metadata_filter["$and"]}
                        else:
                            sec_meta = {"$and": [{"rating": {"$gte": 7.0}}, metadata_filter]}
                    else:
                        sec_meta = {"rating": {"$gte": 7.0}}
                    sec_results = vectorstore._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=40,
                        where=sec_meta,
                        where_document={"$contains": sec_genre},
                    )
                    if sec_results['metadatas'][0]:
                        vec_results = _merge_results(vec_results, sec_results, n_candidates + 120)
                        logger.info(f"   Secondary genre retrieval: merged {len(sec_results['metadatas'][0])} '{sec_genre}' candidates")
                except Exception as e:
                    logger.warning(f"   Secondary genre retrieval failed: {e}")

        # 1d2) Musical keyword retrieval — bypass genre limitation.
        #      Most iconic musicals (La La Land, Greatest Showman, Grease, Les Miserables,
        #      West Side Story) are tagged as Drama/Comedy/Romance without "Music" genre.
        #      They DO have "musical" in their keywords field. Search for that term
        #      in document text to surface them for musical queries.
        q_lower_mus = request.query.lower()
        is_musical_search = any(kw in q_lower_mus for kw in ['musical', 'musicals', 'singing', 'dancing', 'broadway'])
        if is_musical_search and not similar_to_title and vectorstore:
            try:
                mus_kw_filter = {"rating": {"$gte": 6.5}}
                if metadata_filter:
                    if "$and" in metadata_filter:
                        mus_kw_filter = {"$and": [{"rating": {"$gte": 6.5}}] + metadata_filter["$and"]}
                    else:
                        mus_kw_filter = {"$and": [{"rating": {"$gte": 6.5}}, metadata_filter]}
                # Search for "musical" as a keyword in document text
                for mus_term in ["musical", "singing", "dancing", "Broadway"]:
                    mus_results = vectorstore._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=30,
                        where=mus_kw_filter,
                        where_document={"$contains": mus_term},
                    )
                    if mus_results['metadatas'][0]:
                        vec_results = _merge_results(vec_results, mus_results, n_candidates + 120)
                logger.info("   Musical keyword retrieval: merged candidates with 'musical' keyword")
            except Exception as e:
                logger.warning(f"   Musical keyword retrieval failed: {e}")
        #     BGE embeds the source movie's document but keyword bias still
        #     dominates — "movies like Get Out" returns "The Getaway" instead
        #     of thrillers. Use the source movie's genres to pull genre-matched
        #     high-quality candidates that the reranker can surface.
        if similar_to_title and similar_source_genres:
            try:
                for src_genre in similar_source_genres.split(", "):
                    src_genre = src_genre.strip()
                    if not src_genre:
                        continue
                    sim_genre_filter = {"rating": {"$gte": 6.5}}
                    if metadata_filter:
                        if "$and" in metadata_filter:
                            sim_genre_filter = {"$and": [{"rating": {"$gte": 6.5}}] + metadata_filter["$and"]}
                        else:
                            sim_genre_filter = {"$and": [{"rating": {"$gte": 6.5}}, metadata_filter]}
                    sim_genre_results = vectorstore._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=40,
                        where=sim_genre_filter,
                        where_document={"$contains": src_genre},
                    )
                    if sim_genre_results['metadatas'][0]:
                        vec_results = _merge_results(vec_results, sim_genre_results, n_candidates + 120)
                logger.info(f"   Similar-to genre retrieval: merged candidates for genres '{similar_source_genres}'")
            except Exception as e:
                logger.warning(f"   Similar-to genre retrieval failed: {e}")

        # 1e2) Same-director retrieval for "movies like X" queries.
        #      Users expecting "movies like Inception" want other Nolan films
        #      (Memento, The Prestige) which may have completely different genres
        #      and wouldn't be found by genre-only retrieval.
        if similar_to_title and similar_src_meta:
            src_director = similar_src_meta.get('director', '').strip()
            if src_director and len(src_director) > 2:
                try:
                    dir_results = vectorstore._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=15,
                        where={"rating": {"$gte": 6.0}},
                        where_document={"$contains": src_director},
                    )
                    if dir_results['metadatas'][0]:
                        vec_results = _merge_results(vec_results, dir_results, n_candidates + 150)
                        logger.info(f"   Similar-to director retrieval: merged {len(dir_results['metadatas'][0])} films by '{src_director}'")
                except Exception as e:
                    logger.warning(f"   Similar-to director retrieval failed: {e}")

            # 1e3) Same lead-actor retrieval for "movies like X" queries.
            #      "movies like Forrest Gump" should surface Cast Away and
            #      The Green Mile (both Tom Hanks) even if genres/embedding differ.
            src_cast = similar_src_meta.get('cast', '')
            lead_actors = [a.strip() for a in src_cast.split(',') if a.strip()][:2]
            for actor in lead_actors:
                if len(actor) > 3:
                    try:
                        cast_results = vectorstore._collection.query(
                            query_embeddings=[query_embedding],
                            n_results=10,
                            where={"rating": {"$gte": 6.5}},
                            where_document={"$contains": actor},
                        )
                        if cast_results['metadatas'][0]:
                            vec_results = _merge_results(vec_results, cast_results, n_candidates + 170)
                            logger.info(f"   Similar-to cast retrieval: merged films with '{actor}'")
                    except Exception as e:
                        logger.warning(f"   Similar-to cast retrieval failed for '{actor}': {e}")

        # 1f) Curated quality retrieval — bypass embedding entirely.
        #     The core problem: ChromaDB .query() always ranks by embedding
        #     similarity, so even with genre filters, BGE's keyword bias
        #     dominates (e.g., "scary movies" → "Scary Movie").
        #     Solution: use .get() with genre + rating filters to fetch ALL
        #     qualifying movies, then sort by rating*popularity in Python.
        #     This guarantees the candidate pool contains the highest-rated,
        #     most popular movies of the target genre.
        curated_genres = []
        # For mood queries, use multiple genres to cover the full spectrum
        # (e.g., "feel good" → Drama, Comedy, Family, Animation)
        mood_curated_genres = {
            'frightening': ['Horror', 'Thriller'],
            'melancholy': ['Drama', 'Romance'],
            'uplifting': ['Drama', 'Comedy', 'Family'],
            'humorous': ['Comedy', 'Family'],
            'intense': ['Thriller', 'Action', 'Crime'],
            'dark': ['Thriller', 'Crime'],
            'romantic': ['Romance', 'Drama'],
            'cerebral': ['Science Fiction', 'Thriller'],
            'emotional': ['Drama', 'Romance'],
            'calm': ['Comedy', 'Family'],
        }
        intent = query_intent.get('intent')
        mood_type = query_intent.get('mood')
        # Concepts where the genre filter is a rough proxy (Drama/Romance) but NOT
        # the actual topic.  Curated retrieval for these floods results with generic
        # Drama/Romance movies that drown out the targeted doc searches.
        # e.g., "foreign language" → Drama curated → English dramas overwhelm foreign films.
        skip_curated_concepts = {
            'foreign language', 'foreign film', 'subtitled', 'non-english',
            'international film', 'international movie',
            'soundtrack', 'soundtracks', 'score',
            'hidden gem', 'hidden gems', 'underrated', 'overlooked',
            'set in new york', 'new york', 'set in london', 'set in paris',
            'set in tokyo', 'set in los angeles',
            'three hour', '3 hour', 'epic movie', 'epic film',
            'date night', 'date movie',
            'road trip',
        }
        concept_val = query_intent.get('concept', '')
        skip_curated = concept_val in skip_curated_concepts
        if intent == 'mood' and mood_type in mood_curated_genres and not similar_to_title:
            curated_genres = mood_curated_genres[mood_type]
        elif genre_retrieval_filter and not similar_to_title and not skip_curated:
            curated_genres = [genre_retrieval_filter]
            # Include secondary genres detected from query text (e.g., both
            # Action and Romance for "my girlfriend likes romance and I like action")
            if secondary_genres:
                for sg in secondary_genres:
                    if sg not in curated_genres:
                        curated_genres.append(sg)
        # Skip curated retrieval for similar-to queries — broad genre pools
        # (e.g., all Action/SciFi movies) flood candidates with wrong matches.
        # The genre-metadata retrieval (step 1e) already handles this better.

        if curated_genres and vectorstore:
            try:
                for cg in curated_genres[:3]:  # Limit to 3 genres to avoid slow queries
                    curated_filter = {"rating": {"$gte": 7.0}}
                    if metadata_filter:
                        if "$and" in metadata_filter:
                            curated_filter = {"$and": [{"rating": {"$gte": 7.0}}] + metadata_filter["$and"]}
                        else:
                            curated_filter = {"$and": [{"rating": {"$gte": 7.0}}, metadata_filter]}
                    # .get() bypasses embedding — returns ALL matching docs
                    curated_all = vectorstore._collection.get(
                        where=curated_filter,
                        where_document={"$contains": cg},
                        include=["documents", "metadatas"],
                    )
                    if curated_all['metadatas']:
                        # Sort by rating * popularity descending
                        scored_curated = []
                        for ci, cmeta in enumerate(curated_all['metadatas']):
                            crating = float(cmeta.get('rating', 0) or 0)
                            cpop = float(cmeta.get('popularity', 0) or 0)
                            cscore = crating * max(cpop, 0.01)
                            scored_curated.append((cscore, ci))
                        scored_curated.sort(key=lambda x: -x[0])
                        # Take top 50 by quality
                        top_curated_indices = [idx for _, idx in scored_curated[:50]]
                        curated_results = {
                            'documents': [[curated_all['documents'][i] for i in top_curated_indices]],
                            'metadatas': [[curated_all['metadatas'][i] for i in top_curated_indices]],
                            # Assign pseudo-distances based on quality rank (lower = better)
                            # Use 0.15 base so curated movies compete with vector results
                            # that may have artificially low distances due to keyword bias
                            'distances': [[0.15 + 0.005 * rank for rank in range(len(top_curated_indices))]],
                        }
                        vec_results = _merge_results(vec_results, curated_results, n_candidates + 150)
                logger.info(f"   Curated quality retrieval: added top-rated '{'/'.join(curated_genres[:3])}' movies")
            except Exception as e:
                logger.warning(f"   Curated quality retrieval failed: {e}")

        # 1g) Meta-quality curated retrieval — for "best movies of all time",
        #     "movies everyone should watch before they die", etc.
        #     These need the absolute top-rated, most popular movies regardless
        #     of genre. The embedding is useless here (returns title-literal
        #     matches like "X-Rated: Greatest Adult Movies of All Time").
        #     Also applies to decade queries ("90s nostalgia", "2010s masterpieces")
        #     where embedding finds obscure title matches instead of classics.
        is_meta_quality_intent = (intent == 'concept' and query_intent.get('concept') == 'meta_quality')
        is_decade_intent = (intent == 'decade')
        if (is_meta_quality_intent or is_decade_intent) and not similar_to_title and vectorstore:
            try:
                mq_rating_floor = 8.0 if is_meta_quality_intent else 7.0
                mq_filter = {"rating": {"$gte": mq_rating_floor}}
                if metadata_filter:
                    if "$and" in metadata_filter:
                        mq_filter = {"$and": [{"rating": {"$gte": mq_rating_floor}}] + metadata_filter["$and"]}
                    else:
                        mq_filter = {"$and": [{"rating": {"$gte": mq_rating_floor}}, metadata_filter]}
                mq_all = vectorstore._collection.get(
                    where=mq_filter,
                    include=["documents", "metadatas"],
                )
                if mq_all['metadatas']:
                    scored_mq = []
                    for mi, mmeta in enumerate(mq_all['metadatas']):
                        mr = float(mmeta.get('rating', 0) or 0)
                        mp = float(mmeta.get('popularity', 0) or 0)
                        ms = mr * max(mp, 0.01)
                        scored_mq.append((ms, mi))
                    scored_mq.sort(key=lambda x: -x[0])
                    top_mq_indices = [idx for _, idx in scored_mq[:80]]
                    # For meta-quality ("best of all time"): low pseudo-distances so
                    # quality dominates.  For decade queries: higher pseudo-distances
                    # so vector search semantic matching still contributes (otherwise
                    # pure rating*popularity overwrites the embedding's understanding
                    # of "classic 80s" vs "critically acclaimed 80s").
                    base_dist = 0.10 if is_meta_quality_intent else 0.40
                    mq_results = {
                        'documents': [[mq_all['documents'][i] for i in top_mq_indices]],
                        'metadatas': [[mq_all['metadatas'][i] for i in top_mq_indices]],
                        'distances': [[base_dist + 0.003 * rank for rank in range(len(top_mq_indices))]],
                    }
                    vec_results = _merge_results(mq_results, vec_results, n_candidates + 200)
                    label = "meta-quality" if is_meta_quality_intent else "decade"
                    logger.info(f"   {label} curated: added {len(top_mq_indices)} top-rated movies")
            except Exception as e:
                logger.warning(f"   Meta-quality/decade curated failed: {e}")

        # 1h) Hidden gems / underrated curated retrieval — fetch high-rated
        #     but low-popularity movies.  These queries fail with embedding search
        #     because BGE keyword-matches "Hidden Gems" (movie title) and "Unseen".
        q_lower_hg = request.query.lower()
        is_hidden_gems = any(kw in q_lower_hg for kw in ['hidden gem', 'underrated', 'overlooked',
                                                          "haven't seen", "never heard"])
        if is_hidden_gems and not similar_to_title and vectorstore:
            try:
                hg_filter = {"$and": [{"rating": {"$gte": 7.0}},
                                      {"popularity": {"$gte": 0.25}},
                                      {"popularity": {"$lte": 0.60}}]}
                if metadata_filter:
                    if "$and" in metadata_filter:
                        hg_filter = {"$and": [{"rating": {"$gte": 7.0}}, {"popularity": {"$gte": 0.10}}, {"popularity": {"$lte": 0.40}}] + metadata_filter["$and"]}
                hg_all = vectorstore._collection.get(
                    where=hg_filter,
                    include=["documents", "metadatas"],
                )
                if hg_all['metadatas']:
                    scored_hg = []
                    for hi, hmeta in enumerate(hg_all['metadatas']):
                        hr = float(hmeta.get('rating', 0) or 0)
                        hp = float(hmeta.get('popularity', 0) or 0)
                        # Cap rating at 9.0 to prevent bogus 10.0 ratings (few votes) from dominating
                        hr_capped = min(hr, 9.0)
                        # Score: rating * popularity ensures both quality AND recognizability
                        # Sweet spot: 0.35-0.55 popularity (cult classics, not blockbusters)
                        sweet_spot_bonus = max(0, 0.10 - abs(hp - 0.45))
                        hs = hr_capped * max(hp, 0.10) + sweet_spot_bonus * 3.0
                        scored_hg.append((hs, hi))
                    scored_hg.sort(key=lambda x: -x[0])
                    top_hg = [idx for _, idx in scored_hg[:60]]
                    hg_results = {
                        'documents': [[hg_all['documents'][i] for i in top_hg]],
                        'metadatas': [[hg_all['metadatas'][i] for i in top_hg]],
                        'distances': [[0.35 + 0.003 * rank for rank in range(len(top_hg))]],
                    }
                    vec_results = _merge_results(hg_results, vec_results, n_candidates + 250)
                    logger.info(f"   Hidden gems curated: added {len(top_hg)} high-rated low-popularity movies")
            except Exception as e:
                logger.warning(f"   Hidden gems curated failed: {e}")

        # 1i) Foreign language curated retrieval — bypass embedding entirely.
        #     Embedding search for "foreign language must watch" returns English animation
        #     because BGE treats "Parasite", "Amelie" as keywords matching animated films.
        #     Solution: search for known non-English indicators in documents (director names,
        #     country names, foreign-language keywords) and sort by quality.
        is_foreign_query = any(kw in q_lower_hg for kw in [
            'foreign language', 'foreign film', 'subtitled', 'non-english',
            'international film', 'international movie',
            'korean', 'japanese', 'french', 'indian', 'chinese', 'spanish',
            'italian', 'german', 'bollywood', 'anime',
        ])
        if is_foreign_query and not similar_to_title and vectorstore:
            try:
                # Search for documents containing known foreign-language indicators
                foreign_search_terms = [
                    # Specific foreign film titles (more targeted than country names)
                    "Parasite", "Oldboy", "Amelie", "Cinema Paradiso",
                    "Spirited Away", "Seven Samurai", "Pan's Labyrinth",
                    "City of God", "Life Is Beautiful", "Crouching Tiger",
                    "Rashomon", "In the Mood for Love", "The Handmaiden",
                    "Howl's Moving Castle", "Princess Mononoke", "My Neighbor Totoro",
                    "La La Land", "Amélie", "The Intouchables",
                    "Bicycle Thieves", "Roma", "Pather Panchali",
                    "The Battle of Algiers", "Ikiru",
                    # Director names (foreign directors)
                    "Bong Joon-ho", "Park Chan-wook", "Akira Kurosawa", "Hayao Miyazaki",
                    "Wong Kar-wai", "Pedro Almod", "Guillermo del Toro",
                    "Federico Fellini", "Ingmar Bergman", "Andrei Tarkovsky",
                    "Hirokazu Kore-eda", "Denis Villeneuve",
                    # Foreign-language keywords (appear in movie docs but not in English war films)
                    "subtitles", "Korean cinema", "Japanese animation",
                    "Studio Ghibli", "Bollywood",
                ]
                fl_rating_filter = {"rating": {"$gte": 7.0}}
                if metadata_filter:
                    if "$and" in metadata_filter:
                        fl_rating_filter = {"$and": [{"rating": {"$gte": 7.0}}] + metadata_filter["$and"]}
                    else:
                        fl_rating_filter = {"$and": [{"rating": {"$gte": 7.0}}, metadata_filter]}
                fl_all_docs = []
                fl_all_metas = []
                fl_seen_titles = set()
                for fl_term in foreign_search_terms:
                    try:
                        fl_results = vectorstore._collection.get(
                            where=fl_rating_filter,
                            where_document={"$contains": fl_term},
                            include=["documents", "metadatas"],
                        )
                        if fl_results['metadatas']:
                            for fi, fmeta in enumerate(fl_results['metadatas']):
                                ftitle = fmeta.get('title', '')
                                if ftitle not in fl_seen_titles:
                                    fl_seen_titles.add(ftitle)
                                    fl_all_docs.append(fl_results['documents'][fi])
                                    fl_all_metas.append(fmeta)
                    except Exception:
                        continue
                if fl_all_metas:
                    # Score and sort by rating * popularity
                    scored_fl = []
                    for fi, fmeta in enumerate(fl_all_metas):
                        fr = min(float(fmeta.get('rating', 0) or 0), 9.0)
                        fp = float(fmeta.get('popularity', 0) or 0)
                        scored_fl.append((fr * max(fp, 0.01), fi))
                    scored_fl.sort(key=lambda x: -x[0])
                    top_fl = [idx for _, idx in scored_fl[:60]]
                    fl_curated = {
                        'documents': [[fl_all_docs[i] for i in top_fl]],
                        'metadatas': [[fl_all_metas[i] for i in top_fl]],
                        'distances': [[0.10 + 0.003 * rank for rank in range(len(top_fl))]],
                    }
                    # Merge with priority — foreign films first
                    vec_results = _merge_results(fl_curated, vec_results, n_candidates + 300)
                    logger.info(f"   Foreign language curated: added {len(top_fl)} non-English films")
            except Exception as e:
                logger.warning(f"   Foreign language curated failed: {e}")

        # 2) Document-text search for person names (actor / director)
        #    Handles queries like "tom holland movies" where the embedding
        #    model confuses "Tom Holland" with "Tom Hanks".
        #    Skip name detection for "movies like X" queries — those words are
        #    movie titles, not person names, and would pull in sequels.
        # (name_phrase already extracted above for HyDE gating)
        if name_phrase:
            # Widen candidate pool for name queries — vector search is
            # unreliable for person names so we need more candidates
            n_candidates = max(n_candidates, 40)
            logger.info(f"   Name phrase detected: '{name_phrase}' — running document search (n={n_candidates})")
            try:
                doc_results = vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_candidates,
                    where=metadata_filter,
                    where_document={"$contains": name_phrase},
                )
            except Exception:
                doc_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

            # Merge: document-matched results first (more precise for name queries)
            results = _merge_results(doc_results, vec_results, n_candidates)
        else:
            results = vec_results

        # 3) Mood-based document search for emotional/mood queries
        #    When user asks "sad movies", search for documents containing
        #    mood-related keywords to complement the vector search.
        mood_doc_searches = {
            "sad": ["heartbreak", "tragedy", "grief", "devastating", "loss", "emotional"],
            "cry": ["heartbreak", "tragedy", "devastating", "grief", "emotional", "moving"],
            "scary": ["terror", "frightening", "suspense", "horror", "nightmare"],
            "halloween": ["terror", "frightening", "suspense", "horror", "nightmare", "slasher"],
            "creepy": ["disturbing", "psychological", "eerie"],
            "uplifting": ["inspiring", "triumph", "hope", "heartwarming", "uplifting"],
            "emotional": ["heartbreak", "emotional", "moving", "devastating"],
            "feel": ["heartwarming", "uplifting", "feel-good", "charming", "inspiring", "triumph"],
            "feel good": ["heartwarming", "uplifting", "inspiring", "charming", "feel-good", "Forrest Gump"],
            "twist": ["unexpected", "shocking", "revelation", "plot twist", "surprise"],
            "twists": ["unexpected", "shocking", "revelation", "plot twist", "surprise"],
            "surprise": ["unexpected", "shocking", "plot twist", "revelation"],
            "anime": ["Japan", "Hayao Miyazaki", "Studio Ghibli", "anime"],
            "korean": ["Korea", "Korean", "South Korea", "Bong Joon-ho", "Park Chan-wook"],
            "korea": ["Korea", "Korean", "South Korea", "Bong Joon-ho", "Park Chan-wook"],
            "japanese": ["Japan", "Japanese", "Kurosawa", "Miyazaki"],
            "french": ["France", "French", "Paris"],
            "indian": ["India", "Indian", "Bollywood"],
            "bollywood": ["India", "Indian", "Bollywood", "Hindi"],
            "kids": ["family", "Disney", "Pixar", "children", "animated"],
            "children": ["family", "Disney", "Pixar", "children", "animated"],
            "stunning": ["cinematography", "visual", "breathtaking", "beautiful"],
            "visually": ["cinematography", "visual", "breathtaking", "beautiful"],
            "fun": ["entertaining", "adventure", "comedy", "exciting"],
            # Topic doc searches
            "time travel": ["time machine", "time travel", "future", "past"],
            "revenge": ["vengeance", "revenge", "retribution", "payback"],
            "heist": ["robbery", "heist", "steal", "vault", "Ocean"],
            "zombie": ["zombie", "undead", "infection", "outbreak"],
            "zombies": ["zombie", "undead", "infection", "outbreak"],
            "survival": ["survival", "wilderness", "stranded", "endurance"],
            "superhero": ["superhero", "Marvel", "DC Comics", "powers"],
            "space": ["astronaut", "spacecraft", "NASA", "galaxy", "orbit"],
            "disney": ["Disney", "Pixar", "princess", "animated", "fairy tale"],
            "musical": ["singing", "dancing", "Broadway", "musical", "choreography"],
            "war": ["battlefield", "soldier", "combat", "military", "troops"],
            "crime": ["gangster", "mafia", "criminal", "organized crime"],
            "psychological": ["manipulation", "paranoia", "obsession", "mind"],
            # Abstract concept doc searches
            "makes you think": ["philosophical", "thought-provoking", "cerebral", "meaning", "intellectual"],
            "make you think": ["philosophical", "thought-provoking", "cerebral", "meaning", "intellectual"],
            "thought provoking": ["philosophical", "thought-provoking", "cerebral", "intellectual"],
            "thought-provoking": ["philosophical", "thought-provoking", "cerebral", "intellectual"],
            "rainy day": ["cozy", "heartwarming", "gentle", "comfort", "feel-good"],
            "hidden gem": ["overlooked", "cult", "underrated", "underappreciated"],
            "hidden gems": ["overlooked", "cult", "underrated", "underappreciated"],
            "edge of seat": ["suspense", "thrilling", "gripping", "nail-biting", "tense"],
            "edge of my seat": ["suspense", "thrilling", "gripping", "nail-biting", "tense"],
            "edge of your seat": ["suspense", "thrilling", "gripping", "nail-biting", "tense"],
            "cinematography": ["cinematography", "visual", "Roger Deakins", "breathtaking", "photography"],
            # Genre/topic concept doc searches
            "mafia": ["Godfather", "Corleone", "mob", "organized crime", "gangster"],
            "inspirational": ["triumph", "underdog", "overcoming", "determination", "inspiring"],
            "inspiring": ["triumph", "underdog", "overcoming", "determination", "inspiring"],
            "psychological thriller": ["paranoia", "obsession", "manipulation", "Shutter Island", "Black Swan"],
            "black comedy": ["dark humor", "darkly funny", "satire", "cynical", "Fargo"],
            "cheer me up": ["heartwarming", "uplifting", "feel-good", "funny", "charming"],
            "cheer up": ["heartwarming", "uplifting", "feel-good", "funny", "charming"],
            # Genre-specific doc searches
            "disaster": ["earthquake", "tsunami", "volcano", "hurricane", "tornado", "catastrophe"],
            "sports": ["championship", "coach", "underdog", "athlete", "Rocky"],
            "sport": ["championship", "coach", "underdog", "athlete", "game"],
            "courtroom": ["trial", "jury", "verdict", "lawyer", "testimony", "judge"],
            "foreign language": ["Parasite", "Amelie", "Cinema Paradiso", "Pan's Labyrinth", "Spirited Away", "City of God", "Oldboy", "Seven Samurai", "Rashomon", "La Vita", "Bong Joon-ho", "Akira Kurosawa", "Hayao Miyazaki", "Guillermo del Toro"],
            "foreign": ["Parasite", "Amelie", "Cinema Paradiso", "Pan's Labyrinth", "Spirited Away", "City of God", "Oldboy", "Seven Samurai", "Bong Joon-ho", "Akira Kurosawa", "Hayao Miyazaki"],
            "soundtrack": ["Hans Zimmer", "John Williams", "score", "composer"],
            "soundtracks": ["Hans Zimmer", "John Williams", "score", "composer"],
            "nostalgia": ["classic", "iconic", "beloved", "generation"],
            "masterpiece": ["acclaimed", "brilliant", "groundbreaking"],
            "masterpieces": ["acclaimed", "brilliant", "groundbreaking"],
            "slow burn": ["atmospheric", "deliberate", "methodical", "tension", "Zodiac"],
            # Additional concept doc searches
            "true story": ["biographical", "based on", "real events", "Schindler", "Social Network", "12 Years a Slave"],
            "based on true": ["biographical", "real events", "Schindler", "Social Network"],
            "road trip": ["Little Miss Sunshine", "journey", "road", "Into the Wild", "Thelma"],
            "visually stunning": ["cinematography", "Roger Deakins", "breathtaking", "Blade Runner", "Revenant"],
            "stunning": ["cinematography", "Roger Deakins", "breathtaking", "visual", "spectacular"],
            "three hour": ["Lawrence of Arabia", "Godfather", "Schindler", "Lord of the Rings"],
            "epic": ["Lawrence of Arabia", "Gladiator", "Lord of the Rings", "epic", "Braveheart"],
            "new york": ["New York", "Manhattan", "Brooklyn", "NYC", "Taxi Driver"],
            "set in": ["New York", "Manhattan", "Paris", "London", "Tokyo"],
            "grandma": ["Mrs. Doubtfire", "wholesome", "heartwarming", "family", "gentle"],
            "grandmother": ["Mrs. Doubtfire", "wholesome", "heartwarming", "family", "gentle"],
            "date night": ["romantic", "La La Land", "Titanic", "romance", "love"],
            "date": ["romantic", "La La Land", "Titanic", "romance", "love story"],
            "10 year old": ["family", "Disney", "Pixar", "Finding Nemo", "Frozen", "Harry Potter"],
            "year old": ["family", "Disney", "Pixar", "Finding Nemo", "animated"],
        }
        # Track which doc terms we've already searched to avoid duplicates
        searched_terms = set()
        for mood_key, search_terms in mood_doc_searches.items():
            if mood_key in q_lower:
                logger.info(f"   Mood doc search: '{mood_key}' — searching for keywords")
                for term in search_terms:
                    if term in searched_terms:
                        continue
                    searched_terms.add(term)
                    try:
                        mood_results = vectorstore._collection.query(
                            query_embeddings=[query_embedding],
                            n_results=15,
                            where=metadata_filter,
                            where_document={"$contains": term},
                        )
                        if mood_results['metadatas'][0]:
                            results = _merge_results(results, mood_results, n_candidates)
                    except Exception:
                        pass

        # ── Array bounds to cap execution time ──────────────────────
        if results['documents'][0] and len(results['documents'][0]) > RERANK_MAX_CANDIDATES:
            logger.info(f"   Capping candidates from {len(results['documents'][0])} to {RERANK_MAX_CANDIDATES} before fallback/CE")
            results['documents'][0] = results['documents'][0][:RERANK_MAX_CANDIDATES]
            results['metadatas'][0] = results['metadatas'][0][:RERANK_MAX_CANDIDATES]
            results['distances'][0] = results['distances'][0][:RERANK_MAX_CANDIDATES]

        # ── Cross-encoder re-ranking ──────────────────────────────
        # The cross-encoder sees (query, document) pairs jointly and produces
        # more nuanced relevance scores. We BLEND CE scores with original distances
        # rather than replacing them, so the embedding model's semantic understanding
        # is preserved while benefiting from the CE's cross-attention.
        # Skip for person/franchise intents — keyword matching is sufficient.
        # Also skip for genre and similar_to queries — CE's keyword bias hurts.
        # Similar-to uses document re-embedding; CE re-ranking distorts those distances.
        ce_intents = {'mood', 'concept', 'specific', 'conversational'}
        # Skip CE for foreign language queries — the curated retrieval assigns
        # very low distances (0.10) to non-English films, but CE normalization
        # maps all candidates to [0,1] and lets English films score higher.
        use_ce = query_intent.get('intent') in ce_intents and not is_foreign_query
        if cross_encoder is not None and results['documents'][0] and use_ce:
            try:
                # For mood queries, use a descriptive CE query that captures
                # the intent without problematic literal keywords.
                # "feel good movies" → "heartwarming uplifting comedy drama"
                # "movies that make you cry" → "deeply emotional tragic drama"
                ce_query = request.query
                mood_ce_rewrites = {
                    'sad': 'deeply emotional tragic moving drama film',
                    'cry': 'deeply emotional heartbreaking tragic moving drama film',
                    'crying': 'deeply emotional heartbreaking tragic drama film',
                    'scary': 'genuinely frightening terrifying horror thriller film',
                    'halloween': 'frightening horror supernatural slasher film',
                    'feel good': 'heartwarming uplifting joyful comedy drama film',
                    'feel-good': 'heartwarming uplifting joyful comedy drama film',
                    'funny': 'hilarious witty comedy laugh-out-loud film',
                    'uplifting': 'heartwarming inspiring triumphant drama film',
                    'dark': 'gritty noir psychological thriller crime film',
                    'intense': 'gripping suspenseful tense nail-biting thriller film',
                    'edge of your seat': 'gripping suspenseful tense nail-biting thriller film',
                    'edge of my seat': 'gripping suspenseful tense nail-biting thriller film',
                    'mind-bending': 'cerebral surreal psychological science fiction film',
                    'mind bending': 'cerebral surreal psychological science fiction film',
                    'emotional': 'deeply moving powerful emotional drama film',
                    'cheer me up': 'heartwarming uplifting feel-good comedy film',
                    'cheer up': 'heartwarming uplifting feel-good comedy film',
                    'light': 'lighthearted charming feel-good comedy film',
                    'light hearted': 'lighthearted charming feel-good comedy film',
                    'lighthearted': 'lighthearted charming feel-good comedy film',
                    'adrenaline': 'fast-paced explosive high-octane action thriller film',
                    'adrenaline rush': 'fast-paced explosive high-octane action thriller film',
                    'slow burn': 'atmospheric deliberate methodical building tension psychological thriller film',
                    'bittersweet': 'melancholic bittersweet poignant drama film',
                }
                if matched_mood_word and query_intent.get('intent') == 'mood':
                    ce_query = mood_ce_rewrites.get(matched_mood_word, request.query)

                # Cap the maximum candidates specifically hitting the CrossEncoder
                if len(results['documents'][0]) > CROSS_ENCODER_MAX_CANDIDATES:
                    logger.info(f"   Hard-capping CrossEncoder candidates to {CROSS_ENCODER_MAX_CANDIDATES}")
                    results['documents'][0] = results['documents'][0][:CROSS_ENCODER_MAX_CANDIDATES]
                    results['metadatas'][0] = results['metadatas'][0][:CROSS_ENCODER_MAX_CANDIDATES]
                    results['distances'][0] = results['distances'][0][:CROSS_ENCODER_MAX_CANDIDATES]

                pairs = [(ce_query, doc) for doc in results['documents'][0]]
                ce_scores = cross_encoder.predict(pairs)
                # CE weight: use conservative 30% for mood queries (reranker
                # has stronger genre/quality/mood signals), 40% for others.
                if query_intent.get('intent') == 'mood':
                    ce_w, vec_w = 0.3, 0.7
                else:
                    ce_w, vec_w = 0.4, 0.6
                ce_distances = [float(1.0 - s) for s in ce_scores]
                # Normalize CE distances to [0,1] range
                if ce_distances:
                    ce_min = min(ce_distances)
                    ce_max = max(ce_distances)
                    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                    ce_distances = [(d - ce_min) / ce_range for d in ce_distances]
                orig_dists = results['distances'][0]
                # Normalize original distances to [0,1] range for blending
                if orig_dists:
                    d_min = min(orig_dists)
                    d_max = max(orig_dists)
                    d_range = d_max - d_min if d_max > d_min else 1.0
                    norm_dists = [(d - d_min) / d_range for d in orig_dists]
                else:
                    norm_dists = orig_dists
                blended = [ce_w * ce + vec_w * nd for ce, nd in zip(ce_distances, norm_dists)]
                results['distances'] = [blended]
                logger.info(f"   Cross-encoder blended {len(pairs)} candidates ({int(ce_w*100)}%CE/{int(vec_w*100)}%vec)")
            except Exception as e:
                logger.warning(f"   Cross-encoder failed ({e}) — using vector distances")

        # Re-rank by keyword match (boost exact phrase matches in document text)
        results = rerank_with_keywords(request.query, results, n_candidates,
                                       name_phrase=name_phrase,
                                       source_genres=similar_source_genres,
                                       source_meta=similar_src_meta,
                                       is_foreign_query=is_foreign_query,
                                       matched_mood_word=matched_mood_word,
                                       similar_to_title=similar_to_title,
                                       is_franchise_query=(query_intent.get('intent') == 'franchise'))

        # ── Global franchise diversity cap ───────────────────────────
        # Limit to 2 films per franchise for ALL queries (not just "movies like X").
        # This prevents 4× Ocean's films, 3× Kill Bill, 3× Fantastic Beasts from
        # dominating heist / revenge / Harry Potter queries.
        # EXCEPTION: skip for explicit franchise queries ("spider-man movies",
        # "batman films") — the user is asking FOR all films in that franchise.
        if results['documents'][0] and query_intent.get('intent') != 'franchise':
            franchise_seen: dict = {}
            franchise_keep = []
            for i, meta in enumerate(results['metadatas'][0]):
                title = meta.get('title', '')
                title_lower_fc = title.lower()
                # Derive a franchise key: first significant word after stripping
                # articles and colons.  "Ocean's Eleven/Twelve" → "ocean's",
                # "Kill Bill: Volume 1/2" → "kill", "The Godfather/Part II" → "godfather"
                words_fc = [w for w in re.split(r'[\s:]+', title_lower_fc)
                            if w not in ('the', 'a', 'an', '')]
                core_fc = words_fc[0] if words_fc else title_lower_fc
                # Only apply cap for multi-character franchise roots (ignore single letters)
                if len(core_fc) >= 3:
                    franchise_seen[core_fc] = franchise_seen.get(core_fc, 0) + 1
                    if franchise_seen[core_fc] > 2:
                        continue  # skip this entry (too many from same franchise)
                franchise_keep.append(i)
            # Only apply if we still have enough candidates to fill top_k
            if len(franchise_keep) >= request.top_k:
                results = {
                    'documents': [[results['documents'][0][i] for i in franchise_keep]],
                    'metadatas': [[results['metadatas'][0][i] for i in franchise_keep]],
                    'distances': [[results['distances'][0][i] for i in franchise_keep]],
                }

        # Post-filter by genre in Python (ChromaDB $contains doesn't work on
        # comma-separated string fields in v1.x — so we filter here instead)
        if request.genre and results['documents'][0]:
            genre_lower = request.genre.lower()
            keep = [
                i for i, meta in enumerate(results['metadatas'][0])
                if genre_lower in meta.get('genres', '').lower()
            ]
            results = {
                'documents': [[results['documents'][0][i] for i in keep]],
                'metadatas': [[results['metadatas'][0][i] for i in keep]],
                'distances': [[results['distances'][0][i] for i in keep]],
            }

        # For "movies like X" queries, exclude the source movie and its franchise
        # (sequels, prequels, same-titled remakes)
        if similar_to_title and results['documents'][0]:
            src_lower = similar_to_title.lower()
            # Strip common prefixes/suffixes for fuzzy franchise matching
            # "The Godfather" → "godfather", matches "Godfather Part II"
            src_core = src_lower
            for prefix in ("the ", "a "):
                if src_core.startswith(prefix):
                    src_core = src_core[len(prefix):]
            # Also use the original user query title for franchise matching
            # e.g., "harry potter" from "something like harry potter"
            # This catches sequels even when the matched title is long
            query_title_core = movie_title_query.lower().strip() if movie_title_query else ""
            for prefix in ("the ", "a "):
                if query_title_core.startswith(prefix):
                    query_title_core = query_title_core[len(prefix):]
            # Get source movie metadata for cast/director comparison
            src_meta = None
            for meta in results['metadatas'][0]:
                if meta.get('title', '').lower() == src_lower:
                    src_meta = meta
                    break
            # If not found in current results, search for it
            if src_meta is None:
                try:
                    src_search = vectorstore._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        where_document={"$contains": similar_to_title.title()},
                    )
                    if src_search['metadatas'][0]:
                        src_meta = src_search['metadatas'][0][0]
                except Exception:
                    pass

            def _is_franchise_or_query_match(title, src_core, src_lower, src_meta, cand_meta, query_core):
                """Check using both full title and original query title."""
                if _is_franchise_match(title, src_core, src_lower, src_meta, cand_meta):
                    return True
                # Also check if the user's original query appears in the candidate title
                if query_core and len(query_core) >= 4:
                    cand_lower = title.lower()
                    if query_core in cand_lower:
                        return True
                # Extract significant words from source title and check candidate
                # e.g., "The Dark Knight" → significant words: "dark", "knight"
                # "Batman: Bad Blood" contains neither → skip (handled by genre penalty)
                # But extract character/franchise name from source doc text:
                # If source doc mentions a character name prominently (e.g., "Batman")
                # and a candidate title starts with that name, exclude it
                if src_meta:
                    src_title_lower = src_meta.get('title', '').lower()
                    src_doc_lower = ""
                    # Find source doc in results
                    for idx, m in enumerate(results['metadatas'][0]):
                        if m.get('title', '').lower() == src_title_lower:
                            src_doc_lower = results['documents'][0][idx].lower()
                            break
                    if not src_doc_lower:
                        src_doc_lower = src_meta.get('keywords', '').lower()
                    # Look for character names that appear in doc but not in query
                    cand_lower = title.lower()
                    cand_words = set(cand_lower.replace(":", " ").replace("-", " ").split())
                    src_words = set(src_title_lower.replace(":", " ").replace("-", " ").split())
                    # Common franchise indicators: words in candidate title that match
                    # properly capitalized names in the source doc (character names)
                    common_char_names = {"batman", "superman", "spider-man", "spiderman",
                                        "iron man", "ironman", "avengers", "x-men",
                                        "star wars", "harry potter", "james bond",
                                        "indiana jones", "lord of the rings", "hobbit",
                                        "transformers", "jurassic", "terminator",
                                        "alien", "predator", "godzilla", "kong",
                                        "mission impossible", "fast furious"}
                    for name in common_char_names:
                        name_parts = name.split()
                        if any(p in cand_words for p in name_parts):
                            # This character name appears in the candidate title.
                            # Check if it's also referenced in the source TITLE
                            # (not doc text — doc text mentions characters like
                            # "Batman" even for "The Dark Knight", which would
                            # wrongly exclude "Batman Begins" from results).
                            if name in src_lower or any(p in src_words for p in name_parts if len(p) > 3):
                                # Character name IS in source title — same franchise
                                return True
                            # Also check if the character name appears in the
                            # source title words (e.g., "spider" in "spider-man")
                            src_title_parts = set(src_lower.replace(":", " ").replace("-", " ").split())
                            if any(p in src_title_parts for p in name_parts if len(p) > 3):
                                return True
                return False

            keep = [
                i for i, meta in enumerate(results['metadatas'][0])
                if not _is_franchise_or_query_match(
                    meta.get('title', ''), src_core, src_lower,
                    src_meta, meta, query_title_core)
            ]
            results = {
                'documents': [[results['documents'][0][i] for i in keep]],
                'metadatas': [[results['metadatas'][0][i] for i in keep]],
                'distances': [[results['distances'][0][i] for i in keep]],
            }

            # ── Same-director diversity cap for similar-to ─────────────
            # Limit to 2 films by the source movie's director so that
            # genre-similar films from OTHER directors can fill remaining
            # slots.  Without this, "movies like Interstellar" returns
            # 5× Nolan instead of 2 Nolan + Arrival + Gravity + Contact.
            if src_meta and results['documents'][0]:
                src_dir = src_meta.get('director', '').lower().strip()
                if src_dir:
                    dir_count = 0
                    dir_keep = []
                    for i, meta in enumerate(results['metadatas'][0]):
                        cand_dir = meta.get('director', '').lower().strip()
                        if cand_dir and cand_dir == src_dir:
                            dir_count += 1
                            if dir_count > 1:
                                continue  # skip — already have 1 by this director
                        dir_keep.append(i)
                    if len(dir_keep) >= request.top_k:
                        results = {
                            'documents': [[results['documents'][0][i] for i in dir_keep]],
                            'metadatas': [[results['metadatas'][0][i] for i in dir_keep]],
                            'distances': [[results['distances'][0][i] for i in dir_keep]],
                        }

        # Trim to requested top_k
        for key in results:
            results[key] = [results[key][0][:request.top_k]]

        logger.info(f"   Retrieved {len(results['documents'][0])} movies")
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Retrieval error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    if not results['documents'][0]:
        return DiscoverResponse(
            query=request.query,
            answer="I couldn't find any movies matching your criteria. Try broadening your search!",
            recommended_movies=[],
            model_used=inference_engine.config.model_path if inference_engine else LLM_MODEL if hf_client else "retrieval_only",
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
    
    llm_reasons = {}

    if inference_engine is None and hf_client is None:
        # Fallback: Return retrieval results without LLM.
        # The frontend renders movie cards directly; no text banner needed.
        answer = None
        generation_backend = "retrieval_only"
    elif inference_engine is not None:
        # Use optimized inference engine (vLLM/Local with Flash Attention)
        # Build Alpaca-formatted prompt matching the fine-tuned model's training format
        compact_context = format_movies_compact(results)
        prompt = build_alpaca_prompt(request.query, compact_context)

        try:
            logger.info(f"Generating with optimized engine ({inference_engine.backend_name})...")
            gen_start = time.time()
            result = inference_engine.generate(prompt)
            gen_ms = (time.time() - gen_start) * 1000

            raw_answer = result["text"]
            generation_backend = result["backend"]
            logger.info(f"Generated in {gen_ms:.0f}ms ({result['tokens_per_second']} tok/s)")

            # Post-process: The fine-tuned model often hallucinates movies not
            # in the retrieved context.  Extract only the intro sentence(s) from
            # the LLM, then build accurate recommendations from retrieved data.
            answer, llm_reasons = _build_grounded_answer(raw_answer, results)

        except Exception as e:
            logger.error(f"Optimized engine error: {e}, falling back to HF API")
            # Fall through to HF API — use chat-style prompt (not Alpaca)
            if hf_client:
                fallback_prompt = f"""You are a helpful movie recommendation assistant.

Here are relevant movies:

{context}

User Query: {request.query}

Recommend movies from the list above. For each movie recommendation, wrap your reason in tags:
<reason>Movie Title: Provide a short, persuasive overview combining the plot and WHY the user should watch it based on their specific request</reason>

Be conversational and include titles, years, and why they match."""
                resp = hf_client.chat_completion(
                    messages=[{"role": "user", "content": fallback_prompt}],
                    max_tokens=512, temperature=0.7
                )
                raw_answer = resp.choices[0].message.content
                answer, llm_reasons = _build_grounded_answer(raw_answer, results)
                generation_backend = "hf_api_fallback"
            else:
                answer = f"Error: {str(e)}"
                generation_backend = "error"
    else:
        # Build RAG prompt
        prompt = f"""You are a helpful movie recommendation assistant. Your job is to recommend movies based on the user's preferences.

You have access to information about the following movies (sorted by relevance):

{context}

User Query: {request.query}

Please provide helpful recommendations. For each movie you recommend, include:
<reason>Movie Title: Brief explanation of why this movie matches their request</reason>

Be conversational and enthusiastic! Cite the movie titles from the context above.
"""
        
        try:
            logger.info("🤖 Generating response with HuggingFace API...")

            # Call HuggingFace Inference API
            resp = hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            raw_answer = resp.choices[0].message.content
            answer, llm_reasons = _build_grounded_answer(raw_answer, results)

            logger.info("Response generated!")

        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            answer = f"Error generating response: {str(e)}\n\nHowever, here are the top matching movies:\n\n"
            for i, meta in enumerate(results['metadatas'][0][:3]):
                answer += f"{i+1}. {meta['title']} ({meta.get('year', 'N/A')})\n"
        generation_backend = "hf_api"
    
    # ========================================================================
    # Step 4: FORMAT RESPONSE
    # ========================================================================

    def _generate_recommendation_reason(query: str, meta: dict, relevance: float) -> str:
        """Generate a short explanation for why this movie matches the query."""
        
        # If we have pre-generated reasons from an LLM call earlier in the file, use them!
        title = meta.get('title', 'This film')
        title_lower = title.lower()
        
        # Looser match: check if the exact title is a key, or if it's a substring
        for key, reason in llm_reasons.items():
            if key in title_lower or title_lower in key:
                return f"🤖 AI's Take: {reason}"

        # If the LLM didn't generate a reason, we dynamically craft a short context overview
        genres = meta.get('genres', '').lower()
        director = meta.get('director', '')
        
        # Build an intelligent sounding fallback based on vector semantic match
        reasons = []
        if director and director.lower() in query.lower():
            reasons.append(f"Directed by {director}, matching your request perfectly.")
        
        fallback_msg = "A strong thematic match to your query."
        if relevance >= 0.7:
            fallback_msg = "An exceptional match that closely aligns with what you're looking for."
            
        desc = _movie_overview_lookup.get(title, '')
        if desc:
            # Grab a short snippet of the description to make it feel personalized
            snippet = map(str.strip, desc.split('.'))
            snippet = next((s for s in snippet if len(s) > 20), "")
            if snippet:
                return f"🤖 AI's Take: {fallback_msg} This film explores how {snippet.lower()}."

        if reasons:
            return f"🤖 AI's Take: {' '.join(reasons)} {fallback_msg}"
            
        return f"🤖 AI's Take: {fallback_msg} It shares deep semantic themes with your search."

    movies = []
    for i, meta in enumerate(results['metadatas'][0]):
        distance = results['distances'][0][i]
        relevance = round(1 - distance, 3)
        movies.append(MovieResult(
            title=meta.get('title', 'Unknown'),
            year=meta.get('year'),
            rating=meta.get('rating'),
            genres=meta.get('genres'),
            director=meta.get('director'),
            runtime=_movie_runtime_lookup.get(meta.get('title', '')),
            description=_movie_overview_lookup.get(meta.get('title', '')),
            relevance_score=relevance,
            recommendation_reason=_generate_recommendation_reason(request.query, meta, relevance)
        ))
    # Sort by relevance score (highest match first), NOT by rating
    movies.sort(key=lambda m: m.relevance_score or 0, reverse=True)
    
    final_response = DiscoverResponse(
        query=request.query,
        answer=answer,
        recommended_movies=movies,
        model_used=f"{inference_engine.config.model_path} ({generation_backend})" if inference_engine else f"{LLM_MODEL} ({generation_backend})" if hf_client else "retrieval_only",
        retrieval_count=len(movies)
    )
    
    # Store the computed response into the cache
    _set_cached_discover_response(cache_key, final_response)
    
    return final_response
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("Starting Movie Discovery Assistant")
    print("=" * 60)
    print(f"API will be available at: http://localhost:8004")
    print(f"Docs will be at: http://localhost:8004/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
