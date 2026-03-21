"""
Day 2: Embeddings - Understanding the Foundation of Modern AI

This script demonstrates how embeddings work by:
1. Loading a sentence transformer model
2. Converting text to vectors (embeddings)
3. Comparing similarity between different texts
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model (same one used in Oracle service)
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Create embeddings for movie descriptions
texts = [
    "A thrilling action movie with car chases and explosions",
    "An exciting film with vehicle pursuits and action",
    "A romantic comedy about finding love in Paris",
    "A horror movie about a haunted house",
    "A documentary about climate change",
]

print("\nGenerating embeddings...")
embeddings = model.encode(texts)

print(f"Shape: {embeddings.shape}")  # (5, 384) = 5 texts, 384 dimensions each
print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

# Cosine similarity function
def cosine_similarity(a, b):
    """
    Cosine similarity measures the angle between two vectors.
    Returns 1.0 for identical vectors, 0.0 for orthogonal (unrelated).
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
print("\n" + "="*60)
print("SIMILARITY SCORES (1.0 = identical, 0.0 = unrelated)")
print("="*60)

similarities = [
    ("'Action movie' vs 'Exciting film with pursuits'", cosine_similarity(embeddings[0], embeddings[1])),
    ("'Action movie' vs 'Romantic comedy'", cosine_similarity(embeddings[0], embeddings[2])),
    ("'Action movie' vs 'Horror movie'", cosine_similarity(embeddings[0], embeddings[3])),
    ("'Action movie' vs 'Documentary'", cosine_similarity(embeddings[0], embeddings[4])),
]

for desc, score in similarities:
    if score > 0.7:
        emoji = "[HIGH]"
    elif score > 0.4:
        emoji = "[MEDIUM]"
    else:
        emoji = "[LOW]"
    print(f"\n{desc}: {score:.3f} {emoji}")

print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print("The ACTION movie is most similar to EXCITING FILM (same meaning, different words)")
print("This is SEMANTIC similarity - it understands meaning, not just keywords!")
print("\nIf this were keyword search:")
print("  [X] 'action movie' wouldn't match 'exciting film' (no shared words)")
print("  [OK] But embeddings understand they mean the same thing!")

print("\n" + "="*60)
print("HOW THIS WORKS IN STREAMSAGE:")
print("="*60)
print("1. User types: 'mind-bending sci-fi'")
print("2. System embeds query --> [0.23, -0.45, 0.78, ...]")
print("3. ChromaDB searches 6,147 movie embeddings")
print("4. Finds 'Inception' even if description doesn't contain 'mind-bending'")
print("5. Because the MEANING matches!")
