"""
Vector Database Creator
Creates a ChromaDB vector store from processed movie features.
"""

import json
import logging
import os
import requests
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "processed" / "movie_features.json"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

def load_features() -> Dict[str, Dict]:
    """Load processed movie features."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_document_text(movie: Dict) -> str:
    """
    Create a rich text representation of the movie for embedding.
    Includes title, overview, genres, mood, and keywords.
    """
    # Basic info
    title = movie.get("title", "")
    year = movie.get("release_year", "")
    genres = ", ".join(movie.get("genres", []))
    
    # Rich descriptors
    moods = ", ".join(movie.get("mood_tags", []))
    keywords = ", ".join(movie.get("keywords", [])[:10])  # Top 10 keywords
    
    # Cast & Crew
    director = movie.get("director", {}).get("name", "") if movie.get("director") else ""
    cast = ", ".join(movie.get("cast_names", [])[:3])  # Top 3 cast
    
    # Overview
    overview = movie.get("overview", "")
    
    # Construct the text blob
    text = f"Title: {title} ({year})\n"
    text += f"Genres: {genres}\n"
    if director:
        text += f"Director: {director}\n"
    if cast:
        text += f"Cast: {cast}\n"
    text += f"Mood: {moods}\n"
    text += f"Keywords: {keywords}\n"
    text += f"Plot: {overview}"
    
    return text

def main():
    logger.info("=== Vector Database Creation Started ===")
    
    # 1. Load Data
    logger.info(f"Loading features from {INPUT_FILE}...")
    movies_data = load_features()
    logger.info(f"Loaded {len(movies_data)} movies.")
    
    # 2. Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {CHROMA_DB_DIR}...")
    # Disable telemetry to prevent connection errors
    settings = Settings(anonymized_telemetry=False)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR), settings=settings)
    
    # Custom Embedding Function using HF InferenceClient
    from huggingface_hub import InferenceClient
    
    class HFInferenceEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __init__(self, api_key, model_name):
            self.client = InferenceClient(token=api_key)
            self.model_name = model_name
            
        def __call__(self, input: List[str]) -> List[List[float]]:
            embeddings = []
            import time
            
            for text in input:
                retries = 3
                for attempt in range(retries):
                    try:
                        # Add delay to avoid rate limits
                        time.sleep(0.5) 
                        embedding = self.client.feature_extraction(text, model=self.model_name)
                        embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
                        break # Success, exit retry loop
                    except Exception as e:
                        if attempt < retries - 1:
                            wait = (attempt + 1) * 2
                            logger.warning(f"API Error: {e}. Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            logger.error(f"Failed to embed text after {retries} attempts: {e}")
                            embeddings.append([0.0] * 384) # Zero vector fallback
            return embeddings

    # Use Hugging Face API for embeddings
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print("⚠️  HF_TOKEN environment variable not found.")
        api_key = input("Please enter your Hugging Face Write Token: ").strip()
    
    emb_fn = HFInferenceEmbeddingFunction(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name="movies",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. Prepare Data for Insertion
    logger.info("Preparing documents...")
    ids = []
    documents = []
    metadatas = []
    
    batch_size = 5
    # DEBUG: Slice for testing
    movies_items = list(movies_data.items())[:5]
    total_movies = len(movies_items)
    logger.info(f"DEBUG: Processing only {total_movies} movies for testing.")
    
    for i, (movie_id, movie) in enumerate(movies_items):
        # Create embedding text
        doc_text = create_document_text(movie)
        
        # Prepare metadata (for filtering)
        # ChromaDB metadata values must be str, int, float, or bool
        metadata = {
            "title": movie.get("title", ""),
            "year": movie.get("release_year") or 0,
            "rating": movie.get("vote_average", 0.0),
            "popularity_score": movie.get("popularity_score", 0.0),
            "genres": ", ".join(movie.get("genres", [])),
            "moods": ", ".join(movie.get("mood_tags", [])),
            "director": movie.get("director", {}).get("name", "") if movie.get("director") else "",
        }
        
        ids.append(str(movie_id))
        documents.append(doc_text)
        metadatas.append(metadata)
        
        # Batch insert
        if len(ids) >= batch_size:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(f"Processed {i+1}/{total_movies} movies...")
            ids = []
            documents = []
            metadatas = []
            
    # Insert remaining
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Processed {total_movies}/{total_movies} movies.")
        
    logger.info(f"✅ Successfully created Vector DB with {collection.count()} items.")
    logger.info("=== Vector Database Creation Completed ===")

if __name__ == "__main__":
    main()
