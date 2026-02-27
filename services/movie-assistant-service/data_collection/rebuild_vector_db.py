"""
Rebuild ChromaDB from movie_features.json
==========================================
Reads all 6150 movies from the existing processed dataset
(which already has full cast info) and rebuilds ChromaDB
with proper document text including actor names.

This fixes actor-based queries like "Tom Holland Spider-Man movies".
"""

import json
import logging
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths (relative to this file's location inside data_collection/)
SERVICE_ROOT = Path(__file__).parent.parent
FEATURES_FILE = SERVICE_ROOT / "data" / "processed" / "movie_features.json"
CHROMA_DB_PATH = SERVICE_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32  # Smaller batches for larger model
COLLECTION_NAME = "movies"


def create_document_text(movie: dict) -> str:
    """
    Build the text blob that gets embedded.
    Cast names are INCLUDED so actor queries like
    'Tom Holland movies' match correctly via vector search.
    """
    title   = movie.get("title", "")
    year    = movie.get("release_year", "")
    genres  = ", ".join(movie.get("genres", []))
    moods   = ", ".join(movie.get("mood_tags", []))
    keywords = ", ".join(movie.get("keywords", [])[:10])
    overview = movie.get("overview", "")
    wiki    = movie.get("wiki_summary", "")

    director_obj = movie.get("director")
    director = director_obj.get("name", "") if isinstance(director_obj, dict) else ""

    # Include up to 8 cast members so actor queries work well
    cast_names = movie.get("cast_names", [])[:8]
    cast = ", ".join(cast_names)

    text = f"Title: {title} ({year})\n"
    text += f"Genres: {genres}\n"
    if director:
        text += f"Director: {director}\n"
    if cast:
        text += f"Cast: {cast}\n"
    if moods:
        text += f"Mood: {moods}\n"
    if keywords:
        text += f"Keywords: {keywords}\n"
    text += f"Plot: {overview}"
    if wiki:
        text += f"\n{wiki[:200]}"   # small Wikipedia snippet for extra context

    return text


def build_metadata(movie: dict) -> dict:
    """
    Build ChromaDB metadata (must be str/int/float/bool only).
    Matches the fields that main.py reads for MovieResult responses
    and for metadata filters (genre $contains, rating $gte, year $gte/$lte).
    """
    director_obj = movie.get("director")
    director = director_obj.get("name", "") if isinstance(director_obj, dict) else ""

    genres_list = movie.get("genres", [])
    genres_str  = ", ".join(genres_list)

    cast_names = movie.get("cast_names", [])[:8]
    cast_str   = ", ".join(cast_names)

    return {
        "title":     movie.get("title", ""),
        "year":      int(movie.get("release_year") or 0),
        "rating":    float(movie.get("vote_average") or 0.0),
        "genres":    genres_str,
        "director":  director,
        "cast":      cast_str,
        "moods":     ", ".join(movie.get("mood_tags", [])),
        "popularity": float(movie.get("popularity_score") or 0.0),
    }


def main():
    logger.info("=" * 60)
    logger.info("ChromaDB Rebuild from movie_features.json")
    logger.info("=" * 60)

    # ── 1. Load processed features ─────────────────────────────────
    if not FEATURES_FILE.exists():
        logger.error(f"Features file not found: {FEATURES_FILE}")
        sys.exit(1)

    logger.info(f"Loading {FEATURES_FILE} ...")
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        movies_data: dict = json.load(f)

    total = len(movies_data)
    logger.info(f"Loaded {total} movies.")

    # ── 2. Load embedding model (local GPU) ────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device} ...")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": BATCH_SIZE,
        },
    )
    logger.info("Embedding model ready.")

    # ── 3. Connect to ChromaDB ─────────────────────────────────────
    logger.info(f"Connecting to ChromaDB at {CHROMA_DB_PATH} ...")
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH), settings=settings)

    # Delete existing collection to start fresh with proper cast data
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception:
        logger.info(f"No existing '{COLLECTION_NAME}' collection found — creating new.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created.")

    # ── 4. Build and upsert in batches ─────────────────────────────
    items = list(movies_data.items())
    processed = 0
    skipped   = 0

    def flush_batch(batch_ids, batch_docs, batch_metas):
        """Embed and upsert one batch."""
        vectors = embedder.embed_documents(batch_docs)
        collection.upsert(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    batch_ids   = []
    batch_docs  = []
    batch_metas = []

    for movie_id, movie in items:
        # Skip movies without a meaningful overview
        if not movie.get("overview", "").strip():
            skipped += 1
            continue

        doc_text = create_document_text(movie)
        metadata = build_metadata(movie)

        batch_ids.append(str(movie_id))
        batch_docs.append(doc_text)
        batch_metas.append(metadata)

        if len(batch_ids) >= BATCH_SIZE:
            flush_batch(batch_ids, batch_docs, batch_metas)
            processed += len(batch_ids)
            logger.info(f"  Progress: {processed}/{total} upserted ...")
            batch_ids, batch_docs, batch_metas = [], [], []

    # Flush remainder
    if batch_ids:
        flush_batch(batch_ids, batch_docs, batch_metas)
        processed += len(batch_ids)

    final_count = collection.count()
    logger.info("=" * 60)
    logger.info(f"Done! {final_count} movies in ChromaDB.")
    logger.info(f"Skipped {skipped} movies (no overview).")
    logger.info("Restart the movie-assistant service to pick up the new DB.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
