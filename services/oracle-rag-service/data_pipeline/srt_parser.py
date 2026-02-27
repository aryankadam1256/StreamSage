"""
Oracle RAG Service - SRT Subtitle Parser
==========================================
Parses .srt (SubRip) subtitle files into structured data.

SRT Format Reference:
    Each subtitle entry consists of:
    1. A sequential index number
    2. A timestamp line: HH:MM:SS,mmm --> HH:MM:SS,mmm
    3. One or more lines of text
    4. A blank line separator

    Example:
        1
        00:01:15,200 --> 00:01:18,500
        What do you mean by that?

        2
        00:01:19,000 --> 00:01:22,300
        I mean exactly what I said.

Pipeline Stage: 1 of 4 (Parse → Chunk → Embed → Store)

This module handles:
    - Reading .srt files with encoding detection
    - Parsing timestamps into seconds (float)
    - Cleaning text (stripping HTML, sound effects, music cues)
    - Filtering empty/invalid entries
    - Outputting structured subtitle entries as list of dicts
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from .config import SRT_PARSER

logger = logging.getLogger(__name__)


# =============================================================================
# Data Model
# =============================================================================

@dataclass
class SubtitleEntry:
    """A single parsed subtitle entry with timing and cleaned text."""
    index: int
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    text: str               # Cleaned subtitle text
    raw_text: str           # Original text before cleaning

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# =============================================================================
# Timestamp Parsing
# =============================================================================

# Pattern: HH:MM:SS,mmm --> HH:MM:SS,mmm
TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_timestamp(hours: str, minutes: str, seconds: str, millis: str) -> float:
    """
    Convert timestamp components to total seconds.

    Args:
        hours: Hours component (0-99)
        minutes: Minutes component (0-59)
        seconds: Seconds component (0-59)
        millis: Milliseconds component (0-999)

    Returns:
        Total time in seconds as float (e.g., 3661.500 for 1:01:01,500)
    """
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def parse_timestamp_line(line: str) -> Optional[tuple[float, float]]:
    """
    Parse a timestamp line into (start_seconds, end_seconds).

    Args:
        line: A line like "00:01:15,200 --> 00:01:18,500"

    Returns:
        Tuple of (start_time, end_time) in seconds, or None if invalid.
    """
    match = TIMESTAMP_PATTERN.search(line)
    if not match:
        return None

    groups = match.groups()
    start = parse_timestamp(groups[0], groups[1], groups[2], groups[3])
    end = parse_timestamp(groups[4], groups[5], groups[6], groups[7])
    return (start, end)


# =============================================================================
# Text Cleaning
# =============================================================================

# Pre-compile removal patterns from config for performance
_REMOVE_PATTERNS = [re.compile(p) for p in SRT_PARSER["remove_patterns"]]


def clean_subtitle_text(text: str) -> str:
    """
    Clean subtitle text by removing HTML tags, sound effects, music cues,
    and normalizing whitespace.

    Cleaning steps (in order):
        1. Strip HTML tags (<i>, <b>, </i>, <font>, etc.)
        2. Remove bracketed annotations [music playing], (sighs)
        3. Remove music note markers
        4. Remove leading dashes (speaker change indicators)
        5. Normalize whitespace (collapse multiple spaces, strip)

    Args:
        text: Raw subtitle text, may contain HTML/annotations.

    Returns:
        Cleaned text string. May be empty if all content was annotations.
    """
    cleaned = text

    # Apply each removal pattern
    for pattern in _REMOVE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Normalize whitespace: collapse multiple spaces/newlines into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# =============================================================================
# SRT File Parser
# =============================================================================

def read_srt_file(filepath: Path) -> str:
    """
    Read an .srt file, trying multiple encodings.

    .srt files from different sources use different encodings.
    We try UTF-8 first (most common), then fall back to Latin-1
    and Windows codepages.

    Args:
        filepath: Path to the .srt file.

    Returns:
        File contents as string.

    Raises:
        ValueError: If file cannot be read with any supported encoding.
    """
    for encoding in SRT_PARSER["encodings"]:
        try:
            return filepath.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(
        f"Cannot read {filepath} with any supported encoding: "
        f"{SRT_PARSER['encodings']}"
    )


def parse_srt(filepath: str | Path) -> list[SubtitleEntry]:
    """
    Parse an .srt file into a list of SubtitleEntry objects.

    The parser is a state machine with three states:
        INDEX    → expecting a numeric index line
        TIMING   → expecting a timestamp line (HH:MM:SS,mmm --> HH:MM:SS,mmm)
        TEXT     → collecting text lines until blank line

    Args:
        filepath: Path to the .srt subtitle file.

    Returns:
        List of SubtitleEntry objects, sorted by start_time.
        Entries with empty cleaned text are filtered out.

    Example:
        >>> entries = parse_srt("data/subtitles/inception.srt")
        >>> entries[0].text
        "What is the most resilient parasite?"
        >>> entries[0].start_time
        75.2
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    if not filepath.suffix.lower() == ".srt":
        raise ValueError(f"Expected .srt file, got: {filepath.suffix}")

    raw_content = read_srt_file(filepath)
    lines = raw_content.replace("\r\n", "\n").split("\n")

    entries = []
    state = "INDEX"     # State machine: INDEX → TIMING → TEXT → INDEX
    current_index = 0
    current_start = 0.0
    current_end = 0.0
    current_text_lines = []

    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()

        if state == "INDEX":
            # Expecting a numeric index (e.g., "1", "2", "123")
            if line_stripped.isdigit():
                current_index = int(line_stripped)
                state = "TIMING"
            # Skip blank lines between entries

        elif state == "TIMING":
            # Expecting timestamp line
            timing = parse_timestamp_line(line_stripped)
            if timing:
                current_start, current_end = timing
                current_text_lines = []
                state = "TEXT"
            else:
                # Malformed entry - reset to INDEX
                logger.debug(
                    f"Malformed timestamp at line {line_num}: '{line_stripped}'"
                )
                state = "INDEX"

        elif state == "TEXT":
            if line_stripped == "":
                # Blank line = end of this subtitle entry
                raw_text = " ".join(current_text_lines)
                cleaned = clean_subtitle_text(raw_text)

                if len(cleaned) >= SRT_PARSER["min_text_length"]:
                    entries.append(SubtitleEntry(
                        index=current_index,
                        start_time=current_start,
                        end_time=current_end,
                        text=cleaned,
                        raw_text=raw_text,
                    ))

                state = "INDEX"
            else:
                # Accumulate text lines
                current_text_lines.append(line_stripped)

    # Handle last entry (file may not end with blank line)
    if state == "TEXT" and current_text_lines:
        raw_text = " ".join(current_text_lines)
        cleaned = clean_subtitle_text(raw_text)
        if len(cleaned) >= SRT_PARSER["min_text_length"]:
            entries.append(SubtitleEntry(
                index=current_index,
                start_time=current_start,
                end_time=current_end,
                text=cleaned,
                raw_text=raw_text,
            ))

    # Sort by start time (some .srt files have out-of-order entries)
    entries.sort(key=lambda e: e.start_time)

    logger.info(
        f"Parsed {filepath.name}: {len(entries)} entries "
        f"({len(lines)} lines, "
        f"duration: {entries[-1].end_time:.1f}s)" if entries else
        f"Parsed {filepath.name}: 0 entries ({len(lines)} lines)"
    )

    return entries


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into human-readable HH:MM:SS timestamp.

    Args:
        seconds: Time in seconds (e.g., 3661.5)

    Returns:
        Formatted string (e.g., "1:01:01")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
