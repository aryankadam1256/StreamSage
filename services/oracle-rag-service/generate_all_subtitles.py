"""
Batch SRT Generator for Oracle RAG Service
============================================
Generates synthetic subtitle files for all movies in the StreamSage database
using rich metadata (overview, wiki_summary, cast, genres, keywords).

This enables the Oracle to answer questions about ANY movie, not just
the ones with manually created SRT files.

Usage:
    python generate_all_subtitles.py

Output:
    Creates .srt files in data/subtitles/ for each movie.
    Existing files are skipped (use --force to overwrite).
"""

import json
import re
import os
import sys
import time
import argparse
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

MOVIE_FEATURES_PATH = Path(__file__).parent.parent / "movie-assistant-service" / "data" / "processed" / "movie_features.json"
SUBTITLES_DIR = Path(__file__).parent / "data" / "subtitles"


# =============================================================================
# Helpers
# =============================================================================

def title_to_movie_id(title: str) -> str:
    """
    Convert movie title to movie_id matching the frontend's titleToMovieId().

    JS: title.toLowerCase().replace(/[^a-z0-9\\s]/g, '').replace(/\\s+/g, '_').trim()
    """
    movie_id = title.lower()
    movie_id = re.sub(r'[^a-z0-9\s]', '', movie_id)
    movie_id = re.sub(r'\s+', '_', movie_id).strip('_')
    return movie_id


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    # Split on sentence-ending punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty strings and very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# =============================================================================
# SRT Content Builder
# =============================================================================

def build_srt_content(movie: dict) -> str:
    """
    Build SRT-formatted subtitle content from movie metadata.

    Structure:
    1. Genre/setting lines
    2. Character introductions from cast
    3. Plot content from overview + wiki_summary
    4. Thematic elements from keywords/mood

    Each entry is a sentence or short passage formatted as an SRT subtitle
    entry with timestamps spread across the movie's runtime.
    """
    title = movie.get("title", "Unknown")
    overview = movie.get("overview", "")
    wiki = movie.get("wiki_summary", "")
    runtime = movie.get("runtime", 120)  # default 2 hours
    genres = movie.get("genres", [])
    keywords = movie.get("keywords", [])
    mood_tags = movie.get("mood_tags", [])
    top_cast = movie.get("top_cast", [])
    tagline = movie.get("tagline", "")
    director = movie.get("director", {})

    lines = []

    # --- Section 1: Opening / Genre Context ---
    if tagline:
        lines.append(tagline)

    genre_str = ", ".join(genres[:3]) if genres else "Drama"
    director_name = director.get("name", "Unknown") if isinstance(director, dict) else str(director)
    lines.append(f"{title} is a {genre_str.lower()} film directed by {director_name}.")

    # --- Section 2: Overview sentences ---
    if overview:
        overview_sentences = split_into_sentences(overview)
        lines.extend(overview_sentences)

    # --- Section 3: Character introductions ---
    for i, cast_member in enumerate(top_cast[:6]):
        if isinstance(cast_member, dict):
            actor = cast_member.get("name", "")
            character = cast_member.get("character", "")
            if actor and character:
                lines.append(f"{character}, played by {actor}.")

    # --- Section 4: Wiki summary (main plot content) ---
    if wiki:
        wiki_sentences = split_into_sentences(wiki)
        lines.extend(wiki_sentences)

    # --- Section 5: Thematic/mood elements ---
    if keywords and len(keywords) >= 3:
        kw_str = ", ".join(keywords[:5])
        lines.append(f"The film explores themes of {kw_str}.")

    if mood_tags:
        mood_str = " and ".join(mood_tags[:3])
        lines.append(f"The tone is {mood_str}.")

    # Sanity check: ensure minimum content
    if len(lines) < 5:
        lines.append(f"This is {title}, a {genre_str.lower()} film.")
        lines.append("The story unfolds with compelling characters and dramatic tension.")

    # --- Format as SRT ---
    runtime_seconds = max(runtime * 60, 300)  # At least 5 minutes
    num_entries = len(lines)
    interval = runtime_seconds / (num_entries + 1)  # Spread evenly

    srt_entries = []
    for i, line in enumerate(lines):
        entry_num = i + 1
        start_sec = interval * entry_num
        end_sec = start_sec + min(interval * 0.85, 6.0)  # Each entry ~85% of interval or max 6s

        start_ts = format_srt_timestamp(start_sec)
        end_ts = format_srt_timestamp(end_sec)

        # Split long lines into two display lines (SRT convention: max ~42 chars per line)
        display_text = line
        if len(line) > 80:
            mid = len(line) // 2
            # Find nearest space to midpoint
            space_pos = line.rfind(' ', 0, mid + 10)
            if space_pos > mid - 15:
                display_text = line[:space_pos] + "\n" + line[space_pos + 1:]

        srt_entries.append(f"{entry_num}\n{start_ts} --> {end_ts}\n{display_text}\n")

    return "\n".join(srt_entries)


# =============================================================================
# Main Generator
# =============================================================================

def generate_all(force: bool = False) -> dict:
    """
    Generate SRT files for all movies in movie_features.json.

    Returns:
        Stats dict with counts of generated, skipped, and failed.
    """
    # Load movie database
    print(f"Loading movie database from {MOVIE_FEATURES_PATH}...")
    with open(MOVIE_FEATURES_PATH, encoding="utf-8") as f:
        movie_db = json.load(f)

    movies = list(movie_db.values())
    print(f"Found {len(movies)} movies.")

    # Ensure output directory exists
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"generated": 0, "skipped": 0, "failed": 0, "total": len(movies)}
    start_time = time.time()

    for i, movie in enumerate(movies):
        title = movie.get("title", "")
        if not title:
            stats["failed"] += 1
            continue

        movie_id = title_to_movie_id(title)
        if not movie_id:
            stats["failed"] += 1
            continue

        srt_path = SUBTITLES_DIR / f"{movie_id}.srt"

        # Skip existing files unless force
        if srt_path.exists() and not force:
            stats["skipped"] += 1
            continue

        try:
            srt_content = build_srt_content(movie)
            srt_path.write_text(srt_content, encoding="utf-8")
            stats["generated"] += 1
        except Exception as e:
            print(f"  FAILED: {title} ({movie_id}): {e}")
            stats["failed"] += 1

        # Progress every 500 movies
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i + 1}/{len(movies)} | "
                  f"Generated: {stats['generated']} | "
                  f"Skipped: {stats['skipped']} | "
                  f"Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Generated: {stats['generated']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Failed:    {stats['failed']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT files for all movies")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SRT files")
    args = parser.parse_args()

    generate_all(force=args.force)
