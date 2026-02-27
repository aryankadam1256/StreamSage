"""
Oracle RAG Service - ChromaDB Ingestion Orchestrator
=====================================================
Orchestrates the full subtitle ingestion pipeline:
    .srt file → Parse → Chunk → Embed → Store in ChromaDB

Pipeline Stage: 4 of 4 (Parse → Chunk → Embed → Store)

This is the main entry point for ingesting movie subtitles into
the Oracle's knowledge base. It can be run as a standalone script
or imported by the FastAPI service for on-demand ingestion.

Usage (CLI):
    # Ingest a single movie
    python -m data_pipeline.ingest --file data/subtitles/inception.srt --movie-id inception

    # Ingest all .srt files in a directory
    python -m data_pipeline.ingest --dir data/subtitles/

    # Re-ingest (delete existing + re-add)
    python -m data_pipeline.ingest --file inception.srt --movie-id inception --force

ChromaDB Storage Strategy:
    - Single collection "oracle_subtitles" holds all movies
    - Each chunk stored with metadata: {movie_id, chunk_index, timestamp_start, timestamp_end}
    - movie_id filter narrows search to specific movie at query time
    - This scales better than per-movie collections for maintenance and cross-movie queries
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import chromadb

from .config import CHROMADB, DATA_PATHS, LOG_LEVEL
from .srt_parser import parse_srt
from .chunker import chunk_subtitles, get_chunk_stats
from .embedder import SubtitleEmbedder

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ChromaDB Client Management
# =============================================================================

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.

    Uses PersistentClient to store data on disk. The persist directory
    is configured in config.py (DATA_PATHS["chromadb_dir"]).

    Returns:
        chromadb.PersistentClient instance.
    """
    persist_dir = CHROMADB["persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    logger.info(f"ChromaDB client initialized (persist_dir={persist_dir})")
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get or create the Oracle subtitles collection in ChromaDB.

    ChromaDB HNSW index parameters are set here for optimal
    retrieval quality/speed tradeoff:
        - hnsw:space = "cosine" (matches normalized embeddings)
        - hnsw:construction_ef = 200 (index build quality)
        - hnsw:search_ef = 100 (search quality at query time)
        - hnsw:M = 16 (graph connectivity)

    Returns:
        chromadb.Collection for subtitle chunks.
    """
    collection = client.get_or_create_collection(
        name=CHROMADB["collection_name"],
        metadata={
            "hnsw:space": CHROMADB["hnsw_space"],
            "hnsw:construction_ef": CHROMADB["hnsw_construction_ef"],
            "hnsw:search_ef": CHROMADB["hnsw_search_ef"],
            "hnsw:M": CHROMADB["hnsw_M"],
        },
    )

    logger.info(
        f"Collection '{CHROMADB['collection_name']}': "
        f"{collection.count()} existing documents"
    )
    return collection


# =============================================================================
# Ingestion Pipeline
# =============================================================================

def ingest_subtitle(
    srt_path: str | Path,
    movie_id: str,
    embedder: SubtitleEmbedder,
    collection: chromadb.Collection,
    force: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline for a single movie subtitle file.

    Pipeline steps:
        1. Check if movie already ingested (skip unless force=True)
        2. Parse .srt file → list of SubtitleEntry
        3. Chunk entries → list of SubtitleChunk (sliding window)
        4. Generate embeddings → numpy array (384-dim vectors)
        5. Store in ChromaDB → documents + embeddings + metadata

    Args:
        srt_path: Path to the .srt subtitle file.
        movie_id: Unique identifier for the movie (e.g., "inception").
        embedder: Initialized SubtitleEmbedder instance.
        collection: ChromaDB collection to store chunks in.
        force: If True, delete existing chunks for this movie and re-ingest.

    Returns:
        Dict with ingestion statistics:
        {
            "movie_id": "inception",
            "status": "ingested",
            "entries_parsed": 1247,
            "chunks_created": 178,
            "embeddings_generated": 178,
            "total_duration_seconds": 8821.5,
            "elapsed_seconds": 12.3,
        }
    """
    srt_path = Path(srt_path)
    start_time = time.time()

    logger.info(f"{'='*60}")
    logger.info(f"Ingesting: {movie_id} ({srt_path.name})")
    logger.info(f"{'='*60}")

    # Step 0: Check if already ingested
    existing = collection.get(where={"movie_id": movie_id})
    if existing["ids"] and not force:
        logger.info(
            f"Movie '{movie_id}' already ingested "
            f"({len(existing['ids'])} chunks). Use --force to re-ingest."
        )
        return {
            "movie_id": movie_id,
            "status": "skipped",
            "existing_chunks": len(existing["ids"]),
        }

    # Step 0b: If force, delete existing chunks first
    if existing["ids"] and force:
        logger.info(f"Force mode: deleting {len(existing['ids'])} existing chunks")
        collection.delete(where={"movie_id": movie_id})

    # Step 1: Parse .srt file
    logger.info("Stage 1/4: Parsing SRT file...")
    entries = parse_srt(srt_path)

    if not entries:
        logger.warning(f"No valid entries parsed from {srt_path}")
        return {"movie_id": movie_id, "status": "empty", "entries_parsed": 0}

    # Step 2: Chunk entries
    logger.info("Stage 2/4: Chunking subtitle entries...")
    chunks = chunk_subtitles(entries, movie_id)
    stats = get_chunk_stats(chunks)
    logger.info(f"  Chunk stats: {json.dumps(stats, indent=2)}")

    # Step 3: Generate embeddings
    logger.info("Stage 3/4: Generating embeddings...")
    embeddings = embedder.embed_chunks(chunks)

    # Step 4: Store in ChromaDB
    logger.info("Stage 4/4: Storing in ChromaDB...")
    _store_chunks(collection, chunks, embeddings)

    elapsed = time.time() - start_time

    result = {
        "movie_id": movie_id,
        "status": "ingested",
        "srt_file": srt_path.name,
        "entries_parsed": len(entries),
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
        "total_duration_seconds": round(stats.get("total_duration_seconds", 0), 1),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"Done: {movie_id} | {len(entries)} entries → {len(chunks)} chunks "
        f"in {elapsed:.2f}s"
    )

    return result


def _store_chunks(
    collection: chromadb.Collection,
    chunks: list,
    embeddings,
) -> None:
    """
    Store subtitle chunks and their embeddings in ChromaDB.

    ChromaDB has a batch limit, so we insert in batches of 500 documents.

    Each document is stored with:
        - id: chunk_id (e.g., "inception_chunk_0")
        - document: chunk text (used for keyword search fallback)
        - embedding: 384-dim dense vector
        - metadata: movie_id, chunk_index, timestamps, num_entries

    Args:
        collection: Target ChromaDB collection.
        chunks: List of SubtitleChunk objects.
        embeddings: numpy array of embeddings, shape (len(chunks), 384).
    """
    batch_size = 500  # ChromaDB recommended batch size

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        ids = [c.chunk_id for c in batch_chunks]
        documents = [c.text for c in batch_chunks]
        metadatas = [
            {
                "movie_id": c.movie_id,
                "chunk_index": c.chunk_index,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
                "num_entries": c.num_entries,
                "start_entry_index": c.start_entry_index,
                "end_entry_index": c.end_entry_index,
            }
            for c in batch_chunks
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info(
            f"  Stored batch {i // batch_size + 1}: "
            f"{len(batch_chunks)} chunks (total: {min(i + batch_size, len(chunks))})"
        )


# =============================================================================
# Batch Ingestion (Multiple Movies)
# =============================================================================

def ingest_directory(
    subtitles_dir: str | Path,
    embedder: SubtitleEmbedder,
    collection: chromadb.Collection,
    force: bool = False,
) -> list[dict]:
    """
    Ingest all .srt files from a directory.

    Movie ID is derived from the filename (without extension).
    For example: "inception.srt" → movie_id = "inception"

    Args:
        subtitles_dir: Path to directory containing .srt files.
        embedder: Initialized SubtitleEmbedder instance.
        collection: ChromaDB collection.
        force: If True, re-ingest even if already exists.

    Returns:
        List of result dicts (one per movie).
    """
    subtitles_dir = Path(subtitles_dir)
    srt_files = sorted(subtitles_dir.glob("*.srt"))

    if not srt_files:
        logger.warning(f"No .srt files found in {subtitles_dir}")
        return []

    logger.info(f"Found {len(srt_files)} .srt files in {subtitles_dir}")

    results = []
    for srt_file in srt_files:
        movie_id = srt_file.stem.lower().replace(" ", "_")
        result = ingest_subtitle(srt_file, movie_id, embedder, collection, force)
        results.append(result)

    # Save ingestion log
    _save_ingestion_log(results)

    # Summary
    ingested = [r for r in results if r["status"] == "ingested"]
    skipped = [r for r in results if r["status"] == "skipped"]
    logger.info(f"\nIngestion complete: {len(ingested)} ingested, {len(skipped)} skipped")

    return results


def _save_ingestion_log(results: list[dict]) -> None:
    """Save ingestion results to a JSON log file."""
    log_path = Path(DATA_PATHS["ingestion_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log if present
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)

    existing.extend(results)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Ingestion log saved to {log_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for subtitle ingestion."""
    parser = argparse.ArgumentParser(
        description="Oracle RAG Service - Subtitle Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single subtitle file
  python -m data_pipeline.ingest --file data/subtitles/inception.srt --movie-id inception

  # Ingest all .srt files in a directory
  python -m data_pipeline.ingest --dir data/subtitles/

  # Force re-ingestion (delete existing + re-add)
  python -m data_pipeline.ingest --dir data/subtitles/ --force

  # List collection stats
  python -m data_pipeline.ingest --stats
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .srt file")
    group.add_argument("--dir", type=str, help="Path to directory of .srt files")
    group.add_argument("--stats", action="store_true", help="Show collection statistics")

    parser.add_argument("--movie-id", type=str, help="Movie ID (required with --file)")
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-ingestion (delete existing chunks first)"
    )

    args = parser.parse_args()

    # Initialize ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if args.stats:
        _print_stats(collection)
        return

    # Initialize embedder
    embedder = SubtitleEmbedder()

    if args.file:
        if not args.movie_id:
            # Derive from filename
            args.movie_id = Path(args.file).stem.lower().replace(" ", "_")
            logger.info(f"Movie ID derived from filename: {args.movie_id}")

        result = ingest_subtitle(
            args.file, args.movie_id, embedder, collection, args.force
        )
        print(json.dumps(result, indent=2))

    elif args.dir:
        results = ingest_directory(args.dir, embedder, collection, args.force)
        print(json.dumps(results, indent=2))


def _print_stats(collection: chromadb.Collection) -> None:
    """Print collection statistics."""
    count = collection.count()
    print(f"\nCollection: {CHROMADB['collection_name']}")
    print(f"Total documents: {count}")

    if count > 0:
        # Get unique movie_ids
        # ChromaDB doesn't have a distinct query, so we sample
        sample = collection.get(limit=min(count, 10000), include=["metadatas"])
        movie_ids = set(m["movie_id"] for m in sample["metadatas"])
        print(f"Movies ingested: {len(movie_ids)}")
        for mid in sorted(movie_ids):
            movie_chunks = [
                m for m in sample["metadatas"] if m["movie_id"] == mid
            ]
            print(f"  - {mid}: {len(movie_chunks)} chunks")


if __name__ == "__main__":
    main()
