"""
=============================================================================
Oracle RAG Service - Main API Server
=============================================================================

Phase 4: Streaming, Multi-Turn Conversation & Suggested Questions

Builds on Phase 2 (RAG core) and Phase 3 (integration) to add:
- Server-Sent Events (SSE) streaming for token-by-token response display
- Multi-turn conversation history in LLM context (last 3 turns)
- Dynamic suggested questions endpoint based on movie content

Architecture:
    User Query + Conversation History
        ↓
    [Query Understanding] → Classify intent, extract timestamp hints
        ↓
    [Retrieval] → ChromaDB vector search with movie_id filter
        ↓                + optional timestamp proximity weighting
    [Context Construction] → Format chunks + prior conversation turns
        ↓
    [Generation] → Ollama streaming OR sync generate → answer
        ↓
    Response: SSE stream of tokens  OR  {answer, sources[], model, query_time_ms}

Endpoints:
    POST /ask             - Ask a question (sync, full response JSON)
    POST /ask/stream      - Ask a question (async SSE token stream)
    GET  /suggestions/{id} - Dynamic suggested questions for a movie
    GET  /collections     - List ingested movies and chunk counts
    GET  /health          - Service health check

Reference: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
           (Lewis et al., 2020 - Facebook AI Research)
=============================================================================
"""

import os
import re
import json
import time
import logging
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import chromadb

from data_pipeline.config import CHROMADB, EMBEDDING
from data_pipeline.embedder import SubtitleEmbedder
from data_pipeline.srt_parser import format_timestamp

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class ConversationTurn(BaseModel):
    """
    A single turn in the conversation history.

    Roles:
    - "user": the human's question
    - "oracle": the assistant's answer

    The frontend accumulates these from the chatHistory state and sends
    the last N turns so the LLM has context for follow-up questions.
    """
    role: str = Field(..., pattern="^(user|oracle)$")
    content: str = Field(..., min_length=1, max_length=2000)


class QueryRequest(BaseModel):
    """
    Request schema for the /ask and /ask/stream endpoints.

    The frontend sends:
    - query: The user's natural language question
    - movie_id: Which movie to search (scoped to one movie)
    - timestamp: Optional approximate timestamp hint (seconds)
    - top_k: Number of chunks to retrieve (default 5)
    - conversation_history: Prior turns for multi-turn context (optional)
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about the movie",
        examples=["What did the characters discuss about the signal?"],
    )
    movie_id: str = Field(
        ...,
        min_length=1,
        description="Movie identifier (matches the ingested movie_id)",
        examples=["inception", "the_signal"],
    )
    timestamp: Optional[float] = Field(
        None,
        ge=0,
        description="Optional timestamp hint in seconds (searches nearby chunks)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of relevant chunks to retrieve",
    )
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Previous conversation turns for multi-turn context (last 3 used)",
    )
    already_watched: bool = Field(
        default=False,
        description="True = user has already watched the film; disables spoiler protection and timestamp filtering",
    )


class SourceChunk(BaseModel):
    """A retrieved subtitle chunk with timestamp and relevance metadata."""
    content: str
    movie_id: str
    timestamp_start: float
    timestamp_end: float
    relevance_score: float


class QueryResponse(BaseModel):
    """Response schema for the /ask endpoint."""
    model_config = {"protected_namespaces": ()}

    answer: str
    sources: list[SourceChunk]
    model_used: str
    query_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    chroma_documents: int
    embedding_model: str
    llm_model: str


# =============================================================================
# Query Understanding
# =============================================================================

# Intent patterns for classifying user queries
# Covers diverse genres and query complexity (simple → complex)
INTENT_PATTERNS = {
    "quote_search": [
        r"\bquote\b", r"\bsay\b", r"\bsaid\b", r"\btell\b", r"\btold\b",
        r"\bwords?\b", r"\bline\b", r"\bdialogue\b", r"\bexact\b",
    ],
    "scene_finding": [
        r"\bscene\b", r"\bhappen(s|ed)?\b", r"\boccur\b",
        r"\bmoment\b", r"\bpart where\b", r"\bwhen did\b",
        r"\bwhen does\b",
    ],
    "theme_analysis": [
        r"\btheme\b", r"\bmeaning\b", r"\bmeans?\b", r"\bsymbol\b",
        r"\bmessage\b", r"\bexplore\b", r"\bmetaphor\b", r"\bdeeper\b",
    ],
    "character_query": [
        r"\bcharacter\b", r"\bwho is\b", r"\bwho was\b", r"\bwho are\b",
        r"\brelationship\b", r"\bpersonality\b", r"\bmotivat\b",
        r"\bantagonist\b", r"\bprotagonist\b", r"\bvillain\b", r"\bhero\b",
        r"\bwho\b",
    ],
    "plot_question": [
        r"\bwhy did\b", r"\bwhy does\b", r"\bwhy is\b",
        r"\bexplain\b", r"\bhow did\b", r"\bhow does\b",
        r"\bwhat is going on\b", r"\bwhat's going on\b",
        r"\bconfused\b", r"\bunderstand\b", r"\bmissed\b",
    ],
    "prediction_request": [
        r"\bwhat happens next\b", r"\bwhat.s next\b",
        r"\bspoiler\b", r"\bending\b", r"\btwist\b",
        r"\bhow does it end\b", r"\bwho dies\b", r"\bwho survives\b",
        r"\bgoing to die\b", r"\bgoing to happen\b",
    ],
    "mood_reaction": [
        r"\bscary\b", r"\bcreepy\b", r"\bfunny\b", r"\bsad\b",
        r"\bintense\b", r"\bboring\b", r"\bconfusing\b",
        r"\bwild\b", r"\bcrazy\b", r"\btense\b", r"\bemotional\b",
        r"\bheartbreaking\b", r"\bhilarious\b", r"\bfreaked\b",
    ],
    "timestamp_query": [
        r"\bminute\b", r"\bhour\b", r"\bbeginning\b",
        r"\bstart\b", r"\bfinish\b", r"\bmiddle\b", r"\bmark\b",
        r"\bat \d+:\d+\b",
    ],
}


def classify_intent(query: str) -> str:
    """
    Classify query intent to optimize retrieval and prompt strategy.

    Returns one of: quote_search, scene_finding, theme_analysis,
    character_query, plot_question, prediction_request, mood_reaction,
    timestamp_query, or general.
    """
    query_lower = query.lower()
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, query_lower))
        if score > 0:
            scores[intent] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


def extract_timestamp_hint(query: str) -> Optional[float]:
    """
    Extract a timestamp from natural language queries.

    Handles patterns like:
    - "at the 45 minute mark" → 2700.0
    - "around 1 hour 20 minutes" → 4800.0
    - "in the beginning" → 300.0
    - "near the end" → None (can't determine without movie length)
    """
    query_lower = query.lower()

    # "X hour(s) Y minute(s)" pattern
    match = re.search(r"(\d+)\s*hours?\s*(\d+)?\s*minutes?", query_lower)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2) or 0)
        return hours * 3600 + minutes * 60

    # "X minute(s)" pattern
    match = re.search(r"(\d+)\s*minutes?", query_lower)
    if match:
        return int(match.group(1)) * 60

    # "beginning" / "opening"
    if re.search(r"\b(beginning|opening|start|first)\b", query_lower):
        return 300.0  # ~5 minutes in

    # "middle"
    if re.search(r"\bmiddle\b", query_lower):
        return None  # Can't determine without movie length

    return None


# =============================================================================
# Retrieval Engine
# =============================================================================

def retrieve_chunks(
    collection: chromadb.Collection,
    embedder: SubtitleEmbedder,
    query: str,
    movie_id: str,
    top_k: int = 5,
    timestamp_hint: Optional[float] = None,
) -> list[dict]:
    """
    Retrieve relevant subtitle chunks from ChromaDB.

    Process:
    1. Embed the query using the same model used for ingestion
    2. Search ChromaDB with movie_id metadata filter
    3. Optionally re-rank by timestamp proximity if hint provided
    4. Return top-K results as structured dicts

    Args:
        collection: ChromaDB collection with subtitle chunks.
        embedder: Initialized SubtitleEmbedder for query embedding.
        query: User's natural language question.
        movie_id: Movie to search within.
        top_k: Number of results to return.
        timestamp_hint: Optional approximate timestamp (seconds) to boost
                        nearby chunks.

    Returns:
        List of dicts with keys: content, movie_id, timestamp_start,
        timestamp_end, relevance_score.
    """
    # Step 1: Embed the query
    query_embedding = embedder.embed_texts([query])[0].tolist()

    # Step 2: Search ChromaDB with movie_id filter
    # Retrieve more than top_k if we'll re-rank by timestamp
    fetch_k = top_k * 3 if timestamp_hint else top_k

    # Build where filter — when timestamp is provided, only retrieve
    # chunks that START before the viewer's current position (spoiler prevention)
    if timestamp_hint is not None:
        where_filter = {
            "$and": [
                {"movie_id": movie_id},
                {"timestamp_start": {"$lte": timestamp_hint}},
            ]
        }
    else:
        where_filter = {"movie_id": movie_id}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return []

    if not results["ids"][0]:
        return []

    # Step 3: Build result list with relevance scores
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity score (0 to 1)
        relevance = max(0.0, 1.0 - dist)

        chunks.append({
            "content": doc,
            "movie_id": meta["movie_id"],
            "timestamp_start": meta["timestamp_start"],
            "timestamp_end": meta["timestamp_end"],
            "relevance_score": relevance,
        })

    # Step 4: Re-rank by timestamp proximity if hint provided
    if timestamp_hint is not None and chunks:
        chunks = _rerank_by_timestamp(chunks, timestamp_hint)

    # Return top_k after potential re-ranking
    return chunks[:top_k]


def _rerank_by_timestamp(
    chunks: list[dict],
    timestamp_hint: float,
    time_weight: float = 0.3,
) -> list[dict]:
    """
    Re-rank retrieved chunks by combining semantic relevance with
    timestamp proximity to the user's hint.

    Combined score = (1 - time_weight) * relevance + time_weight * proximity

    proximity = 1.0 / (1.0 + |chunk_midpoint - timestamp_hint| / 60)
    This gives a score of 1.0 for exact match, decaying with distance.

    Args:
        chunks: Retrieved chunks with relevance_score.
        timestamp_hint: Target timestamp in seconds.
        time_weight: How much to weight proximity vs relevance (0.0-1.0).

    Returns:
        Re-ranked list of chunks (highest combined score first).
    """
    for chunk in chunks:
        midpoint = (chunk["timestamp_start"] + chunk["timestamp_end"]) / 2
        distance_minutes = abs(midpoint - timestamp_hint) / 60.0
        proximity = 1.0 / (1.0 + distance_minutes)

        chunk["_combined_score"] = (
            (1 - time_weight) * chunk["relevance_score"]
            + time_weight * proximity
        )

    chunks.sort(key=lambda c: c["_combined_score"], reverse=True)

    # Clean up internal field
    for chunk in chunks:
        chunk.pop("_combined_score", None)

    return chunks


# =============================================================================
# Prompt Engineering
# =============================================================================

def build_system_prompt(user_timestamp: Optional[float] = None, already_watched: bool = False) -> str:
    """
    Build the Oracle system prompt, optionally with timestamp awareness.

    When a timestamp is provided, the prompt enforces a hard spoiler boundary:
    the LLM is told exactly where the viewer is in the movie and instructed
    to only discuss events up to that point.

    When already_watched=True, all spoiler protection is removed and the Oracle
    can discuss the film freely as a post-watch companion.

    Args:
        user_timestamp: Viewer's current position in seconds, or None.
        already_watched: If True, no spoiler restrictions apply.

    Returns:
        Complete system prompt string.
    """
    if already_watched:
        timestamp_context = (
            "The viewer has ALREADY WATCHED the entire film and wants to discuss it freely. "
            "You may discuss any part of the movie — plot twists, the ending, character arcs, anything. "
            "No spoiler restrictions apply."
        )
        spoiler_block = (
            "FULL KNOWLEDGE MODE:\n"
            "- Discuss any aspect of the film — including the ending, twists, and character fates\n"
            "- Draw connections across the whole movie\n"
            "- Reference specific scenes and their significance to the overall story\n"
            "- Compare early and late scenes to highlight themes and foreshadowing"
        )
    elif user_timestamp is not None:
        ts_formatted = format_timestamp(user_timestamp)
        timestamp_context = (
            f"The viewer is currently at {ts_formatted} in the movie. "
            "ONLY discuss events and dialogue from this point and earlier. "
            "Anything after this timestamp has NOT been seen yet."
        )
        spoiler_block = (
            "SPOILER PROTECTION:\n"
            "- You can ONLY discuss what appears in the provided excerpts\n"
            "- If the answer to their question hasn't been revealed yet in the excerpts, say something like: "
            '"That hasn\'t been revealed yet! Keep watching — you\'re in for a ride."\n'
            "- NEVER hint at what's coming, even vaguely\n"
            "- If a character's true role hasn't been shown yet, only describe what's been seen so far"
        )
    else:
        timestamp_context = (
            "Answer based only on the provided subtitle excerpts. "
            "Do not reveal future events or twists."
        )
        spoiler_block = (
            "SPOILER PROTECTION:\n"
            "- You can ONLY discuss what appears in the provided excerpts\n"
            "- If the answer to their question hasn't been revealed yet in the excerpts, say something like: "
            '"That hasn\'t been revealed yet! Keep watching — you\'re in for a ride."\n'
            "- NEVER hint at what's coming, even vaguely\n"
            "- If a character's true role hasn't been shown yet, only describe what's been seen so far"
        )

    if already_watched:
        core_rules = (
            "1. Use the subtitle excerpts for specific dialogue, AND your general knowledge about the film for broader context\n"
            "2. Warm, conversational tone — like a friend who has seen the film many times\n"
            "3. NEVER include timestamps, excerpt numbers, or technical details in your answer\n"
            "4. Keep dialogue quotes SHORT (under 15 words). Pick the most impactful line\n"
            "5. ONE cohesive answer — never go excerpt-by-excerpt"
        )
        response_style = (
            "- Character questions: Discuss the character's FULL arc, motivations, and fate\n"
            "- Plot questions: Explain the whole story, including twists and the ending\n"
            "- Theme questions: Analyze themes across the entire film, including resolution\n"
            "- Mood/vibe questions: Engage with the viewer's reflection on the full experience\n"
            "- Spoiler questions: Answer fully — the viewer has seen it all\n"
            "- If not in excerpts: Draw on your knowledge of the film to answer"
        )
    else:
        core_rules = (
            "1. ONLY use information from the subtitle excerpts provided below\n"
            "2. Warm, conversational tone — like a knowledgeable friend watching alongside them\n"
            "3. NEVER include timestamps, excerpt numbers, or technical details in your answer\n"
            "4. Keep dialogue quotes SHORT (under 15 words). Pick the most impactful line\n"
            "5. ONE cohesive answer — never go excerpt-by-excerpt"
        )
        response_style = (
            "- Character questions: Describe based on what they've said and done SO FAR\n"
            f"- Predictions or spoiler requests: Deflect warmly — \"No spoilers! Trust me, keep watching.\"\n"
            "- Plot questions: Explain only what has happened in the excerpts\n"
            "- Mood/vibe questions: Engage with the viewer's experience, validate their feelings\n"
            "- Theme questions: Discuss only themes visible in the excerpts so far\n"
            "- If not covered in excerpts: \"The movie hasn't gotten into that yet from what you've watched.\"\n"
            "- End with ONE short teaser that builds excitement (without spoiling)"
        )

    return f"""You are The Oracle — a movie watching companion guiding viewers in real-time.

CONTEXT: {timestamp_context}

CORE RULES:
{core_rules}

{spoiler_block}

RESPONSE STYLE:
{response_style}

STRICT LIMIT: 80 words maximum. Be concise."""


def build_rag_prompt(
    query: str,
    chunks: list[dict],
    intent: str,
    conversation_history: Optional[List] = None,
    user_timestamp: Optional[float] = None,
    already_watched: bool = False,
) -> str:
    """
    Construct the prompt for the LLM with retrieved context.

    The prompt structure:
    1. System instruction (The Oracle persona + timestamp-aware rules)
    2. Retrieved subtitle excerpts with timestamps
    3. Optional: last 3 turns of conversation history (for follow-up questions)
    4. User's current question
    5. Intent-specific instruction suffix

    Multi-turn benefit: if the user asks "What about earlier?" the model
    sees the prior question/answer and understands what "earlier" refers to.

    Args:
        query: User's original question.
        chunks: Retrieved subtitle chunks with timestamps.
        intent: Classified query intent (quote_search, scene_finding, etc.)
        conversation_history: Prior ConversationTurn objects (last 3 used).
        user_timestamp: Viewer's current position in seconds (for spoiler boundary).
        already_watched: If True, no spoiler restrictions apply.

    Returns:
        Complete prompt string ready for LLM.
    """
    # Build timestamp-aware system prompt
    system_prompt = build_system_prompt(user_timestamp, already_watched=already_watched)

    # Format subtitle excerpts
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        ts_start = format_timestamp(chunk["timestamp_start"])
        ts_end = format_timestamp(chunk["timestamp_end"])
        relevance_pct = f"{chunk['relevance_score'] * 100:.0f}%"

        context_lines.append(
            f"[Excerpt {i}] ({ts_start} - {ts_end}) [relevance: {relevance_pct}]\n"
            f"{chunk['content']}"
        )

    context = "\n\n".join(context_lines)

    # Intent-specific suffix — guides the LLM's response style per query type
    if already_watched:
        suffix_map = {
            "quote_search": "Share the most memorable dialogue from this film with context about its significance.",
            "scene_finding": "Describe this scene and its importance to the overall story.",
            "theme_analysis": "Analyze this theme across the entire film, including how it resolves.",
            "character_query": "Discuss this character's full arc, motivations, and ultimate fate.",
            "plot_question": "Explain the full story context, including twists and the ending if relevant.",
            "prediction_request": "Answer fully — the viewer has seen the whole film. Discuss what actually happened.",
            "mood_reaction": "Engage with the viewer's reflection on the completed film experience.",
            "timestamp_query": "Describe this scene and its significance in the context of the full film.",
            "general": "Answer conversationally, drawing on both the excerpts and your knowledge of the full film.",
        }
    else:
        suffix_map = {
            "quote_search": "Share the most relevant dialogue with brief context. No timestamps.",
            "scene_finding": "Describe what's happening in this moment based on the excerpts. Don't reveal what comes after.",
            "theme_analysis": "Discuss the themes visible so far. Quote key dialogue. Tease deeper layers without specifics.",
            "character_query": "Describe this character based on what the viewer has seen — their words, actions, and relationships so far. Don't reveal their arc beyond the excerpts.",
            "plot_question": "Explain clearly what has happened based on the excerpts. If the viewer seems confused, help clarify. Don't reveal anything beyond what's shown.",
            "prediction_request": "Do NOT reveal what happens next. Deflect warmly and build excitement: 'No spoilers! Keep watching...'",
            "mood_reaction": "Engage with the viewer's emotional reaction. Validate their feelings and connect it to what's happening in the story so far.",
            "timestamp_query": "Describe what's happening at this point. Only reference events up to this moment.",
            "general": "Answer conversationally based only on the excerpts. Quote dialogue when helpful. End with an excitement-building teaser.",
        }
    suffix = suffix_map.get(intent, suffix_map["general"])

    # Build conversation history block (last 3 turns)
    history_block = ""
    if conversation_history:
        recent = conversation_history[-3:]  # Use at most last 3 turns
        lines = []
        for turn in recent:
            role_label = "User" if turn.role == "user" else "Oracle"
            # Truncate each prior turn to keep the prompt manageable
            content = turn.content[:400] + "..." if len(turn.content) > 400 else turn.content
            lines.append(f"{role_label}: {content}")
        history_block = "\n\nCONVERSATION HISTORY (for context):\n" + "\n".join(lines)

    prompt = f"""{system_prompt}

SUBTITLE EXCERPTS:
{context}{history_block}

CURRENT QUESTION: {query}

{suffix}"""

    return prompt


# =============================================================================
# Ollama LLM Client
# =============================================================================

class OllamaClient:
    """
    Client for generating text with Ollama's local LLM server.

    Wraps the ollama Python package for synchronous generation.
    Falls back to a retrieval-only mode if Ollama is unavailable.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.connected = False
        self._client = None
        self._async_client = None

    def connect(self) -> bool:
        """Test connection to Ollama server."""
        try:
            import ollama
            self._client = ollama.Client(host=self.base_url)
            self._async_client = ollama.AsyncClient(host=self.base_url)
            # Test with a simple request
            self._client.list()
            self.connected = True
            logger.info(f"Ollama connected at {self.base_url} (model: {self.model})")
            return True
        except Exception as e:
            logger.warning(
                f"Ollama unavailable at {self.base_url}: {e}. "
                f"Running in retrieval-only mode (no LLM generation)."
            )
            self.connected = False
            return False

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Full prompt string (system + context + question).
            temperature: Sampling temperature. Low (0.1-0.3) for factual
                        grounding, high (0.7-1.0) for creative responses.

        Returns:
            Generated text string.
        """
        if not self.connected or not self._client:
            return self._fallback_response(prompt)

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 220,
                    "repeat_penalty": 1.1,
                    "num_gpu": 99,
                },
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._fallback_response(prompt)

    def generate_stream(self, prompt: str, temperature: float = 0.2):
        """
        Stream tokens from the LLM using Ollama's streaming API.

        Yields raw response chunks from ollama. Each chunk is a dict with:
            {"response": "token", "done": False/True}

        Streaming is preferable for long answers because the user sees
        text appearing progressively instead of waiting for full generation.

        Usage:
            for chunk in client.generate_stream(prompt):
                token = chunk["response"]
                if chunk["done"]:
                    break

        Args:
            prompt: Full prompt string (system + context + question).
            temperature: Sampling temperature.

        Yields:
            Ollama response chunk dicts.
        """
        if not self.connected or not self._client:
            # Yield the fallback response as a single chunk
            yield {"response": self._fallback_response(prompt), "done": True}
            return

        try:
            stream = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 220,
                    "repeat_penalty": 1.1,
                    "num_gpu": 99,
                },
                stream=True,
            )
            for chunk in stream:
                yield chunk
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield {"response": self._fallback_response(prompt), "done": True}

    def _fallback_response(self, prompt: str) -> str:
        """Generate a basic response when Ollama is unavailable."""
        return (
            "[Retrieval-Only Mode - Ollama not connected]\n\n"
            "The relevant subtitle excerpts have been retrieved and are shown "
            "in the source chunks below. To get AI-generated summaries, "
            "ensure Ollama is running with the llama3:8b model."
        )

    async def async_generate_stream(self, prompt: str, temperature: float = 0.2):
        """
        Async streaming version using ollama.AsyncClient.

        Unlike generate_stream() which blocks the event loop, this method
        uses the async Ollama client so the event loop stays unblocked.
        This allows SSE events (like sources) to be flushed to the client
        immediately rather than being held up by synchronous I/O.

        Yields:
            Ollama response chunk dicts.
        """
        if not self.connected or not self._async_client:
            yield {"response": self._fallback_response(prompt), "done": True}
            return

        try:
            stream = await self._async_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 220,
                    "repeat_penalty": 1.1,
                    "num_gpu": 99,
                },
                stream=True,
            )
            async for chunk in stream:
                yield chunk
        except Exception as e:
            logger.error(f"Ollama async streaming failed: {e}")
            yield {"response": self._fallback_response(prompt), "done": True}


# =============================================================================
# Global State (initialized on startup)
# =============================================================================

embedder: Optional[SubtitleEmbedder] = None
collection: Optional[chromadb.Collection] = None
ollama_client: Optional[OllamaClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager. Initializes expensive resources on
    startup and cleans up on shutdown.

    Resources initialized:
    1. SubtitleEmbedder (loads sentence-transformers model into memory)
    2. ChromaDB collection (connects to persistent vector store)
    3. OllamaClient (tests connection to local LLM server)
    """
    global embedder, collection, ollama_client

    logger.info("Starting Oracle RAG Service...")

    # 1. Initialize embedder (same model used for ingestion)
    logger.info(f"Loading embedding model: {EMBEDDING['model_name']}")
    embedder = SubtitleEmbedder()

    # 2. Connect to ChromaDB
    logger.info(f"Connecting to ChromaDB: {CHROMADB['persist_dir']}")
    try:
        client = chromadb.PersistentClient(path=CHROMADB["persist_dir"])
        collection = client.get_or_create_collection(
            name=CHROMADB["collection_name"],
            metadata={"hnsw:space": CHROMADB["hnsw_space"]},
        )
        doc_count = collection.count()
        logger.info(f"ChromaDB connected: {doc_count} documents in '{CHROMADB['collection_name']}'")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        collection = None

    # 3. Connect to Ollama
    logger.info(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
    ollama_client = OllamaClient(OLLAMA_BASE_URL, LLM_MODEL)
    ollama_client.connect()

    logger.info("Oracle RAG Service ready!")

    yield  # Application runs

    logger.info("Shutting down Oracle RAG Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Oracle RAG Service",
    description=(
        "The Oracle - Movie Intelligence through RAG. "
        "Answers questions about movies using their subtitle transcripts. "
        "Retrieves relevant dialogue chunks and generates grounded answers "
        "with timestamp citations."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check for container orchestration and gateway."""
    doc_count = 0
    if collection:
        try:
            doc_count = collection.count()
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if ollama_client and ollama_client.connected else "degraded",
        ollama_connected=bool(ollama_client and ollama_client.connected),
        chroma_documents=doc_count,
        embedding_model=EMBEDDING["model_name"],
        llm_model=LLM_MODEL,
    )


@app.post("/ask", response_model=QueryResponse, tags=["RAG"])
async def ask_oracle(request: QueryRequest):
    """
    Ask the Oracle a question about a movie's dialogue.

    Full RAG pipeline:
    1. Classify query intent (quote, scene, theme, character, timestamp)
    2. Extract timestamp hints from natural language
    3. Retrieve relevant subtitle chunks from ChromaDB
    4. Re-rank by timestamp proximity if applicable
    5. Construct grounded prompt with retrieved context
    6. Generate answer via Ollama LLM
    7. Return answer with source citations

    Example:
        POST /ask
        {
            "query": "What did they discuss about the Fibonacci pattern?",
            "movie_id": "the_signal",
            "top_k": 5
        }
    """
    start_time = time.time()

    if collection is None or embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. ChromaDB or embedder not initialized.",
        )

    # Step 1: Query understanding
    intent = classify_intent(request.query)
    # When already_watched, ignore timestamp — retrieve all chunks freely
    timestamp_hint = None if request.already_watched else (request.timestamp or extract_timestamp_hint(request.query))

    logger.info(
        f"Query: '{request.query[:60]}...' | movie={request.movie_id} | "
        f"intent={intent} | ts_hint={timestamp_hint}"
    )

    # Step 2: Retrieve relevant chunks
    chunks = retrieve_chunks(
        collection=collection,
        embedder=embedder,
        query=request.query,
        movie_id=request.movie_id,
        top_k=request.top_k,
        timestamp_hint=timestamp_hint,
    )

    if not chunks:
        return QueryResponse(
            answer=(
                f"I couldn't find any relevant dialogue for movie '{request.movie_id}'. "
                "This movie may not have been ingested yet, or try rephrasing your question."
            ),
            sources=[],
            model_used=LLM_MODEL if ollama_client and ollama_client.connected else "none",
            query_time_ms=(time.time() - start_time) * 1000,
        )

    # Step 3: Build prompt with retrieved context + conversation history
    prompt = build_rag_prompt(
        request.query, chunks, intent,
        conversation_history=request.conversation_history or None,
        user_timestamp=timestamp_hint,
        already_watched=request.already_watched,
    )

    # Step 4: Generate answer
    answer = ollama_client.generate(prompt) if ollama_client else (
        "Ollama not initialized. See source chunks below for raw dialogue."
    )

    # Step 5: Format response
    sources = [
        SourceChunk(
            content=c["content"],
            movie_id=c["movie_id"],
            timestamp_start=c["timestamp_start"],
            timestamp_end=c["timestamp_end"],
            relevance_score=round(c["relevance_score"], 3),
        )
        for c in chunks
    ]

    query_time = (time.time() - start_time) * 1000

    logger.info(
        f"Response: {len(sources)} sources | {len(answer)} chars | "
        f"{query_time:.1f}ms | intent={intent}"
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=LLM_MODEL if ollama_client and ollama_client.connected else "retrieval-only",
        query_time_ms=round(query_time, 1),
    )


@app.post("/ask/stream", tags=["RAG"])
async def ask_oracle_stream(request: QueryRequest):
    """
    Streaming version of /ask using Server-Sent Events (SSE).

    Unlike /ask which waits for the full response, this endpoint streams
    tokens as they are generated, so the frontend can display text
    progressively (like ChatGPT-style typing effect).

    SSE event format (newline-delimited JSON over text/event-stream):
        data: {"type": "sources", "sources": [...], "intent": "..."}\\n\\n
        data: {"type": "token", "content": "The "}\\n\\n
        data: {"type": "token", "content": "Oracle "}\\n\\n
        data: {"type": "done", "model_used": "llama3:8b", "query_time_ms": 1234}\\n\\n

    The frontend uses fetch() + ReadableStream (not EventSource, since
    EventSource doesn't support POST requests).

    Design choice: sources are sent FIRST so the UI can render timestamp
    citations immediately while the LLM answer streams in.
    """
    start_time = time.time()

    if collection is None or embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. ChromaDB or embedder not initialized.",
        )

    # Steps 1-3: same as /ask (query understanding + retrieval + prompt)
    intent = classify_intent(request.query)
    # When already_watched, ignore timestamp — retrieve all chunks freely
    timestamp_hint = None if request.already_watched else (request.timestamp or extract_timestamp_hint(request.query))

    logger.info(
        f"[stream] Query: '{request.query[:60]}' | movie={request.movie_id} | "
        f"intent={intent} | history_turns={len(request.conversation_history)}"
    )

    chunks = retrieve_chunks(
        collection=collection,
        embedder=embedder,
        query=request.query,
        movie_id=request.movie_id,
        top_k=request.top_k,
        timestamp_hint=timestamp_hint,
    )

    # Format sources for the client
    sources_payload = [
        {
            "content": c["content"],
            "movie_id": c["movie_id"],
            "timestamp_start": c["timestamp_start"],
            "timestamp_end": c["timestamp_end"],
            "relevance_score": round(c["relevance_score"], 3),
        }
        for c in chunks
    ]

    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Generator that yields SSE-formatted events.

        Protocol:
        1. Immediately emit sources (so the UI renders citations at once)
        2. Stream LLM tokens one-by-one as they arrive from Ollama
        3. Emit a final "done" event with metadata
        """
        # Event 1: Sources (sent immediately, before any LLM call)
        sources_event = json.dumps({
            "type": "sources",
            "sources": sources_payload,
            "intent": intent,
        })
        yield f"data: {sources_event}\n\n"

        if not chunks:
            # No results — emit a single token event explaining why
            no_result = json.dumps({
                "type": "token",
                "content": (
                    f"I couldn't find any relevant dialogue for '{request.movie_id}'. "
                    "This movie may not be ingested yet, or try rephrasing your question."
                ),
            })
            yield f"data: {no_result}\n\n"
        else:
            # Build the prompt with history
            prompt = build_rag_prompt(
                request.query, chunks, intent,
                conversation_history=request.conversation_history or None,
                user_timestamp=timestamp_hint,
                already_watched=request.already_watched,
            )

            # Stream tokens from Ollama (async to avoid blocking event loop)
            if ollama_client:
                last_token = ""
                async for chunk in ollama_client.async_generate_stream(prompt):
                    token = chunk.get("response", "")
                    if token:
                        last_token = token
                        token_event = json.dumps({"type": "token", "content": token})
                        yield f"data: {token_event}\n\n"
                    if chunk.get("done"):
                        break
                # If output was truncated mid-sentence, add graceful ending
                if last_token and not last_token.rstrip().endswith((".", "!", "?", '"', "...")):
                    ending = json.dumps({"type": "token", "content": "..."})
                    yield f"data: {ending}\n\n"
            else:
                fallback = json.dumps({
                    "type": "token",
                    "content": "Ollama not initialized. See source chunks below for raw dialogue.",
                })
                yield f"data: {fallback}\n\n"

        # Final event with metadata
        query_time = (time.time() - start_time) * 1000
        done_event = json.dumps({
            "type": "done",
            "model_used": LLM_MODEL if ollama_client and ollama_client.connected else "retrieval-only",
            "query_time_ms": round(query_time, 1),
        })
        yield f"data: {done_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time SSE
        },
    )


@app.get("/suggestions/{movie_id}", tags=["RAG"])
async def get_suggestions(movie_id: str):
    """
    Return dynamic suggested questions for a specific movie.

    Generates questions by sampling the movie's actual ingested content —
    the suggestions are grounded in what the Oracle actually knows.

    Strategy:
    1. Fetch up to 20 chunks from ChromaDB for this movie
    2. Pick 3 random well-distributed chunks (beginning, middle, end)
    3. Extract key phrases/words to form targeted questions
    4. Supplement with generic fallbacks if content is sparse

    Returns:
        {
            "movie_id": "inception",
            "suggestions": [
                "What did the characters discuss about reality?",
                "What happens around 0:45 in the movie?",
                ...
            ]
        }
    """
    if collection is None:
        return {"movie_id": movie_id, "suggestions": _generic_suggestions()}

    try:
        # Fetch a sample of the movie's chunks to ground suggestions in real content
        results = collection.get(
            where={"movie_id": movie_id},
            limit=20,
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return {
                "movie_id": movie_id,
                "suggestions": _generic_suggestions(),
            }

        docs = results["documents"]
        metas = results["metadatas"]
        n = len(docs)

        # Sample well-distributed chunks (beginning, ~1/3, ~2/3, near end)
        indices = []
        for fraction in [0.0, 0.33, 0.66, 0.9]:
            idx = min(int(fraction * n), n - 1)
            if idx not in indices:
                indices.append(idx)

        suggestions = []

        for idx in indices[:4]:
            doc = docs[idx]
            meta = metas[idx]
            ts = meta.get("timestamp_start", 0)
            ts_str = format_timestamp(ts)

            # Extract the first 120 chars of the chunk for a grounded question
            snippet = doc[:120].strip().rstrip(".,!?")

            # Build a variety of question types rotating through intents
            qtype = len(suggestions) % 3
            if qtype == 0:
                suggestions.append(f"What's happening around {ts_str} in the movie?")
            elif qtype == 1:
                # Pick the most interesting 3 words from the snippet
                words = [w for w in snippet.split() if len(w) > 4][:3]
                if words:
                    phrase = " ".join(words)
                    suggestions.append(f"What do the characters say about {phrase.lower()}?")
                else:
                    suggestions.append(f"What themes emerge from the dialogue near {ts_str}?")
            else:
                suggestions.append(f"Who is speaking around {ts_str} and what are they discussing?")

        # Pad with generic fallbacks if we have fewer than 4
        generic = _generic_suggestions()
        while len(suggestions) < 4:
            candidate = generic[len(suggestions) % len(generic)]
            if candidate not in suggestions:
                suggestions.append(candidate)

        return {"movie_id": movie_id, "suggestions": suggestions[:4]}

    except Exception as e:
        logger.error(f"Error generating suggestions for {movie_id}: {e}")
        return {"movie_id": movie_id, "suggestions": _generic_suggestions()}


def _generic_suggestions() -> list[str]:
    """Fallback suggestions when no movie content is available."""
    return [
        "What did the characters discuss about reality?",
        "What happens in the opening scene?",
        "Who is the main character talking to?",
        "What themes does the dialogue explore?",
    ]


@app.get("/collections", tags=["Debug"])
async def list_collections():
    """
    List all ingested movies with their chunk counts.

    Returns:
        {
            "movies": [{"movie_id": "inception", "chunks": 178}],
            "total_documents": 356
        }
    """
    if collection is None:
        return {"movies": [], "total_documents": 0}

    try:
        total = collection.count()
        if total == 0:
            return {"movies": [], "total_documents": 0}

        # Get all metadata to find unique movie_ids and counts
        sample = collection.get(
            limit=min(total, 50000),
            include=["metadatas"],
        )

        movie_counts = {}
        for meta in sample["metadatas"]:
            mid = meta.get("movie_id", "unknown")
            movie_counts[mid] = movie_counts.get(mid, 0) + 1

        movies = [
            {"movie_id": mid, "chunks": count}
            for mid, count in sorted(movie_counts.items())
        ]

        return {"movies": movies, "total_documents": total}

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return {"error": str(e)}


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
