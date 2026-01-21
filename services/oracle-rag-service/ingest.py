"""
=============================================================================
Oracle RAG Service - Subtitle Ingestion Pipeline
=============================================================================

🎓 CONCEPT: Data Ingestion for RAG

Before we can answer questions about movies, we need to:
1. PARSE: Read .srt subtitle files and extract text + timestamps
2. CLEAN: Remove HTML tags, speaker labels, and artifacts
3. CHUNK: Split into meaningful segments (5-minute windows)
4. EMBED: Convert each chunk to a vector embedding
5. STORE: Save embeddings + metadata in ChromaDB

This script is run once per movie to populate the vector database.

Usage:
    python ingest.py --file /path/to/movie.srt --movie-id "inception"
    python ingest.py --directory /path/to/subtitles/

=============================================================================
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import pysrt
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_WINDOW_SECONDS = 300  # 5 minutes
CHUNK_OVERLAP_SECONDS = 30  # 30 seconds overlap
MIN_CHUNK_CHARS = 100       # Minimum characters per chunk

# Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SubtitleEntry:
    """Represents a single subtitle line with timing."""
    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class SubtitleChunk:
    """
    A chunk of subtitles grouped by time window.
    
    🎓 WHY CHUNKING?
    
    LLMs have context limits, and embeddings work best on coherent segments.
    Chunking subtitles by time ensures:
    - Each chunk has semantic coherence (same scene)
    - Chunks are small enough for efficient retrieval
    - We preserve timestamp metadata for time-travel queries
    """
    movie_id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str


# =============================================================================
# Subtitle Cleaning Functions
# =============================================================================

def srt_time_to_seconds(time_obj) -> float:
    """
    Convert pysrt time object to seconds.
    
    🎓 SRT TIME FORMAT
    SRT uses format: HH:MM:SS,mmm (e.g., 01:23:45,678)
    We convert to total seconds for easier filtering.
    """
    return (
        time_obj.hours * 3600 +
        time_obj.minutes * 60 +
        time_obj.seconds +
        time_obj.milliseconds / 1000
    )


def clean_subtitle_text(text: str) -> str:
    """
    Clean subtitle text by removing artifacts.
    
    🎓 COMMON SRT ARTIFACTS
    
    Subtitles often contain:
    - HTML tags: <i>italics</i>, <b>bold</b>, <font color="red">colored</font>
    - Speaker labels: [NARRATOR], (John):, JOHN:
    - Sound effects: ♪ Music playing ♪, [gunshot], (phone ringing)
    - Timing artifacts: extra newlines, weird spacing
    
    We clean these to improve embedding quality.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove {\an8} and similar ASS/SSA style tags
    text = re.sub(r'\{[^}]+\}', '', text)
    
    # Remove speaker labels like [NARRATOR] or (John):
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'\([^)]+\):', '', text)
    text = re.sub(r'^[A-Z][A-Z\s]+:', '', text, flags=re.MULTILINE)
    
    # Remove music symbols and sound effects
    text = re.sub(r'[♪♫#]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)  # Remove (sound effects)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def parse_srt_file(file_path: str) -> List[SubtitleEntry]:
    """
    Parse an SRT file into structured subtitle entries.
    
    🎓 SRT FILE FORMAT
    
    SRT files have this structure:
    
    1
    00:00:01,000 --> 00:00:04,000
    First subtitle line
    Second line of first subtitle
    
    2
    00:00:05,500 --> 00:00:09,200
    Next subtitle
    
    pysrt handles the parsing, we just clean and structure.
    """
    logger.info(f"📖 Parsing SRT file: {file_path}")
    
    try:
        subs = pysrt.open(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try latin-1 encoding as fallback
        subs = pysrt.open(file_path, encoding='latin-1')
    
    entries = []
    for sub in subs:
        cleaned_text = clean_subtitle_text(sub.text)
        if cleaned_text:  # Skip empty entries
            entries.append(SubtitleEntry(
                index=sub.index,
                start_seconds=srt_time_to_seconds(sub.start),
                end_seconds=srt_time_to_seconds(sub.end),
                text=cleaned_text
            ))
    
    logger.info(f"   Found {len(entries)} subtitle entries")
    return entries


# =============================================================================
# Chunking Logic
# =============================================================================

def chunk_subtitles(
    entries: List[SubtitleEntry],
    movie_id: str,
    window_seconds: int = CHUNK_WINDOW_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS
) -> List[SubtitleChunk]:
    """
    Group subtitle entries into time-based chunks.
    
    🎓 CHUNKING STRATEGY
    
    We use a SLIDING WINDOW approach:
    - Window size: 5 minutes (captures scene context)
    - Overlap: 30 seconds (prevents losing context at boundaries)
    
    Why 5 minutes?
    - Long enough to capture a complete scene
    - Short enough for precise retrieval
    - Balances context vs. specificity
    
    Why overlap?
    - A question might relate to dialogue at a window boundary
    - Overlap ensures relevant content is in at least one chunk
    """
    if not entries:
        return []
    
    # Get movie duration
    max_time = max(e.end_seconds for e in entries)
    
    chunks = []
    chunk_index = 0
    window_start = 0
    
    while window_start < max_time:
        window_end = window_start + window_seconds
        
        # Collect all entries in this window
        window_entries = [
            e for e in entries
            if e.start_seconds < window_end and e.end_seconds > window_start
        ]
        
        if window_entries:
            # Combine text from all entries in window
            combined_text = " ".join(e.text for e in window_entries)
            
            # Only create chunk if it meets minimum length
            if len(combined_text) >= MIN_CHUNK_CHARS:
                chunks.append(SubtitleChunk(
                    movie_id=movie_id,
                    chunk_index=chunk_index,
                    start_seconds=window_start,
                    end_seconds=min(window_end, max_time),
                    text=combined_text
                ))
                chunk_index += 1
        
        # Slide window (with overlap)
        window_start += (window_seconds - overlap_seconds)
    
    logger.info(f"   Created {len(chunks)} chunks from {len(entries)} entries")
    return chunks


# =============================================================================
# Vector Store Operations
# =============================================================================

def initialize_vectorstore() -> Chroma:
    """
    Initialize the ChromaDB vector store with HuggingFace embeddings.
    
    🎓 VECTOR EMBEDDINGS
    
    Embeddings convert text to numerical vectors. Similar texts
    have similar vectors (close in vector space).
    
    all-MiniLM-L6-v2:
    - Output dimension: 384
    - Speed: ~14,000 sentences/second on CPU
    - Quality: Trained on 1B+ sentence pairs
    - Size: ~80MB (fits easily in container)
    """
    logger.info(f"📊 Loading embedding model: {EMBEDDING_MODEL}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info(f"🗄️ Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    
    # Ensure directory exists
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name="movie_subtitles",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    return vectorstore


def ingest_chunks(vectorstore: Chroma, chunks: List[SubtitleChunk]) -> int:
    """
    Add subtitle chunks to the vector store.
    
    🎓 METADATA IS KEY
    
    We store metadata alongside each chunk:
    - movie_id: Filter by specific movie
    - timestamp_start/end: Enable time-travel queries
    - chunk_index: Track chunk ordering
    
    This metadata allows ChromaDB to filter BEFORE semantic search,
    making queries like "What happened at minute 45?" possible.
    """
    if not chunks:
        logger.warning("No chunks to ingest")
        return 0
    
    # Convert to LangChain Documents
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "movie_id": chunk.movie_id,
                "chunk_index": chunk.chunk_index,
                "timestamp_start": chunk.start_seconds,
                "timestamp_end": chunk.end_seconds,
            }
        )
        for chunk in chunks
    ]
    
    logger.info(f"💾 Ingesting {len(documents)} chunks into ChromaDB...")
    
    # Add documents to vector store
    vectorstore.add_documents(documents)
    
    # Persist to disk
    # Note: ChromaDB auto-persists but explicit call ensures durability
    
    logger.info(f"✅ Successfully ingested {len(documents)} chunks")
    return len(documents)


# =============================================================================
# Main Ingestion Workflow
# =============================================================================

def ingest_srt_file(file_path: str, movie_id: Optional[str] = None) -> int:
    """
    Complete ingestion pipeline for a single SRT file.
    
    Args:
        file_path: Path to the .srt file
        movie_id: Optional movie identifier (defaults to filename)
    
    Returns:
        Number of chunks ingested
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.suffix.lower() == '.srt':
        raise ValueError(f"Not an SRT file: {file_path}")
    
    # Use filename as movie_id if not provided
    if movie_id is None:
        movie_id = file_path.stem.lower().replace(' ', '_')
    
    logger.info(f"🎬 Ingesting movie: {movie_id}")
    logger.info(f"   Source: {file_path}")
    
    # Step 1: Parse SRT file
    entries = parse_srt_file(str(file_path))
    
    # Step 2: Chunk by time windows
    chunks = chunk_subtitles(entries, movie_id)
    
    # Step 3: Initialize vector store
    vectorstore = initialize_vectorstore()
    
    # Step 4: Ingest into ChromaDB
    count = ingest_chunks(vectorstore, chunks)
    
    logger.info(f"🎉 Completed ingestion for {movie_id}")
    return count


def ingest_directory(directory_path: str) -> Dict[str, int]:
    """
    Ingest all SRT files in a directory.
    
    Returns:
        Dictionary mapping movie_id to chunk count
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    srt_files = list(directory.glob("*.srt"))
    
    if not srt_files:
        logger.warning(f"No SRT files found in {directory}")
        return {}
    
    logger.info(f"📁 Found {len(srt_files)} SRT files in {directory}")
    
    results = {}
    for srt_file in srt_files:
        movie_id = srt_file.stem.lower().replace(' ', '_')
        try:
            count = ingest_srt_file(str(srt_file), movie_id)
            results[movie_id] = count
        except Exception as e:
            logger.error(f"❌ Failed to ingest {srt_file}: {e}")
            results[movie_id] = 0
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """
    Command-line interface for the ingestion script.
    
    Examples:
        # Ingest a single file
        python ingest.py --file movie.srt --movie-id inception
        
        # Ingest all files in a directory
        python ingest.py --directory /data/subtitles
        
        # Ingest with custom chunking parameters
        python ingest.py --file movie.srt --window 600 --overlap 60
    """
    parser = argparse.ArgumentParser(
        description="Ingest movie subtitles into ChromaDB for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --file movie.srt --movie-id inception
  python ingest.py --directory /data/subtitles
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        help="Path to a single .srt file"
    )
    group.add_argument(
        "--directory", "-d",
        help="Path to directory containing .srt files"
    )
    
    parser.add_argument(
        "--movie-id", "-m",
        help="Movie identifier (defaults to filename)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=CHUNK_WINDOW_SECONDS,
        help=f"Chunk window size in seconds (default: {CHUNK_WINDOW_SECONDS})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP_SECONDS,
        help=f"Chunk overlap in seconds (default: {CHUNK_OVERLAP_SECONDS})"
    )
    
    args = parser.parse_args()
    
    # Update chunking parameters if provided
    global CHUNK_WINDOW_SECONDS, CHUNK_OVERLAP_SECONDS
    CHUNK_WINDOW_SECONDS = args.window
    CHUNK_OVERLAP_SECONDS = args.overlap
    
    try:
        if args.file:
            count = ingest_srt_file(args.file, args.movie_id)
            print(f"\n✅ Ingested {count} chunks")
        else:
            results = ingest_directory(args.directory)
            print(f"\n✅ Ingestion complete:")
            for movie_id, count in results.items():
                print(f"   {movie_id}: {count} chunks")
    
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
