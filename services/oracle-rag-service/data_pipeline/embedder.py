"""
Oracle RAG Service - Embedding Generator
==========================================
Generates dense vector embeddings for subtitle chunks using
sentence-transformers.

Pipeline Stage: 3 of 4 (Parse → Chunk → Embed → Store)

Why sentence-transformers/all-MiniLM-L6-v2?
--------------------------------------------
- 384-dimensional embeddings (compact, fast similarity search)
- Trained on 1B+ sentence pairs for semantic similarity
- 5x faster than larger models (mpnet, instructor) with ~95% quality
- Same model used in movie-assistant-service (consistency across platform)
- 80MB model size - fits easily in memory alongside the RAG service

Embedding Process:
    1. Load model once at initialization (cached in memory)
    2. Batch subtitle chunk texts (batch_size=64 for throughput)
    3. Encode to dense vectors, normalize for cosine similarity
    4. Return as numpy arrays for ChromaDB ingestion

Note on Tokenization:
    all-MiniLM-L6-v2 has a max sequence length of 256 word-piece tokens
    (~200 words, ~1200 characters). Our chunks are sized to stay within
    this limit (max_chunk_chars=1500 in config). Texts exceeding the
    limit are automatically truncated by the model.
"""

import logging
import time

import numpy as np

from .chunker import SubtitleChunk
from .config import EMBEDDING

logger = logging.getLogger(__name__)


class SubtitleEmbedder:
    """
    Generates embeddings for subtitle chunks using sentence-transformers.

    Usage:
        embedder = SubtitleEmbedder()
        embeddings = embedder.embed_chunks(chunks)
        # embeddings.shape = (num_chunks, 384)
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """
        Initialize the embedder by loading the sentence-transformer model.

        Args:
            model_name: HuggingFace model name. Defaults to config value.
            device: Compute device ("cuda", "cpu", "auto"). Defaults to config.
        """
        self.model_name = model_name or EMBEDDING["model_name"]
        self.device = device or EMBEDDING["device"]
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model into memory."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.time()

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(self.model_name, device=device)
        elapsed = time.time() - start

        logger.info(
            f"Loaded {self.model_name} on {device} in {elapsed:.2f}s "
            f"(dim={self.model.get_sentence_embedding_dimension()})"
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts...")
        start = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING["batch_size"],
            normalize_embeddings=EMBEDDING["normalize"],
            show_progress_bar=len(texts) > 100,
        )

        elapsed = time.time() - start
        rate = f"{len(texts) / elapsed:.1f} texts/sec" if elapsed > 0 else "instant"
        logger.info(f"Embedded {len(texts)} texts in {elapsed:.2f}s ({rate})")

        return embeddings

    def embed_chunks(self, chunks: list[SubtitleChunk]) -> np.ndarray:
        """
        Generate embeddings for a list of subtitle chunks.

        Extracts the text field from each chunk and embeds them
        in batches for efficiency.

        Args:
            chunks: List of SubtitleChunk objects.

        Returns:
            numpy array of shape (len(chunks), embedding_dim).
            Index i corresponds to chunks[i].
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()
