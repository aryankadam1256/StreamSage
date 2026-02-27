"""
Batch Ingestion Script for Oracle RAG Service
===============================================
Optimized bulk ingestion that processes all SRT files in one pass
instead of one-by-one. Dramatically faster for large movie libraries.

Usage:
    cd services/oracle-rag-service
    python batch_ingest.py             # Ingest all new SRT files
    python batch_ingest.py --force     # Re-ingest everything
"""

import json
import time
import logging
import argparse
from pathlib import Path

import chromadb
import numpy as np

from data_pipeline.config import CHROMADB, EMBEDDING
from data_pipeline.srt_parser import parse_srt
from data_pipeline.chunker import chunk_subtitles
from data_pipeline.embedder import SubtitleEmbedder

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_ingest")

SUBTITLES_DIR = Path(__file__).parent / "data" / "subtitles"


def main():
    parser = argparse.ArgumentParser(description="Batch ingest all SRT files into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Re-ingest all movies (delete existing first)")
    args = parser.parse_args()

    start_time = time.time()

    # --- Step 1: Connect to ChromaDB ---
    logger.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB["persist_dir"])
    collection = client.get_or_create_collection(
        name=CHROMADB["collection_name"],
        metadata={"hnsw:space": CHROMADB["hnsw_space"]},
    )
    existing_count = collection.count()
    logger.info(f"ChromaDB: {existing_count} existing documents")

    # --- Step 2: Find already-ingested movie_ids ---
    ingested_ids = set()
    if not args.force and existing_count > 0:
        logger.info("Checking existing movie_ids...")
        sample = collection.get(limit=min(existing_count, 50000), include=["metadatas"])
        ingested_ids = {m["movie_id"] for m in sample["metadatas"]}
        logger.info(f"Already ingested: {len(ingested_ids)} movies")
    elif args.force and existing_count > 0:
        logger.info("Force mode: deleting all existing documents...")
        # Delete in batches (ChromaDB has limits)
        all_ids = collection.get(limit=min(existing_count, 50000))["ids"]
        # Delete in chunks of 5000
        for i in range(0, len(all_ids), 5000):
            batch = all_ids[i:i+5000]
            collection.delete(ids=batch)
        logger.info(f"Deleted {len(all_ids)} documents")

    # --- Step 3: Discover SRT files to process ---
    srt_files = sorted(SUBTITLES_DIR.glob("*.srt"))
    logger.info(f"Found {len(srt_files)} SRT files in {SUBTITLES_DIR}")

    # Filter out already-ingested
    to_process = []
    for srt_path in srt_files:
        movie_id = srt_path.stem.lower().replace(" ", "_")
        if movie_id not in ingested_ids:
            to_process.append((srt_path, movie_id))

    logger.info(f"New movies to ingest: {len(to_process)} (skipping {len(srt_files) - len(to_process)} already ingested)")

    if not to_process:
        logger.info("Nothing to ingest. Done.")
        return

    # --- Step 4: Parse and chunk all files ---
    logger.info("Parsing and chunking all SRT files...")
    all_chunks = []
    movies_processed = 0
    parse_start = time.time()

    for i, (srt_path, movie_id) in enumerate(to_process):
        try:
            entries = parse_srt(srt_path)
            if not entries:
                continue
            chunks = chunk_subtitles(entries, movie_id)
            all_chunks.extend(chunks)
            movies_processed += 1
        except Exception as e:
            logger.warning(f"Failed to parse {srt_path.name}: {e}")

        if (i + 1) % 1000 == 0:
            logger.info(f"  Parsed {i + 1}/{len(to_process)} files, {len(all_chunks)} chunks so far")

    parse_time = time.time() - parse_start
    logger.info(f"Parsed {movies_processed} movies → {len(all_chunks)} chunks in {parse_time:.1f}s")

    if not all_chunks:
        logger.info("No chunks to ingest. Done.")
        return

    # --- Step 5: Embed all chunks at once (batch) ---
    logger.info(f"Embedding {len(all_chunks)} chunks (batch mode on GPU)...")
    embedder = SubtitleEmbedder()
    embed_start = time.time()

    # Extract texts for embedding
    texts = [c.text for c in all_chunks]

    # Embed in large batches
    batch_size = 256
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_emb = embedder.embed_texts(batch_texts)
        all_embeddings.append(batch_emb)
        if (i + batch_size) % 5000 < batch_size:
            logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    embed_time = time.time() - embed_start
    logger.info(f"Embedded {len(all_chunks)} chunks in {embed_time:.1f}s "
                f"({len(all_chunks)/embed_time:.0f} chunks/sec)")

    # --- Step 6: Store in ChromaDB (batch) ---
    logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB...")
    store_start = time.time()

    chroma_batch_size = 5000  # ChromaDB batch limit
    for i in range(0, len(all_chunks), chroma_batch_size):
        end = min(i + chroma_batch_size, len(all_chunks))
        batch_chunks = all_chunks[i:end]
        batch_emb = embeddings[i:end]

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
            embeddings=batch_emb.tolist(),
            metadatas=metadatas,
        )
        logger.info(f"  Stored batch: {end}/{len(all_chunks)}")

    store_time = time.time() - store_start
    total_time = time.time() - start_time

    # --- Summary ---
    final_count = collection.count()
    logger.info("=" * 60)
    logger.info("BATCH INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Movies processed:  {movies_processed}")
    logger.info(f"  Chunks created:    {len(all_chunks)}")
    logger.info(f"  Total in ChromaDB: {final_count}")
    logger.info(f"  Parse time:        {parse_time:.1f}s")
    logger.info(f"  Embed time:        {embed_time:.1f}s")
    logger.info(f"  Store time:        {store_time:.1f}s")
    logger.info(f"  Total time:        {total_time:.1f}s")


if __name__ == "__main__":
    main()
