"""
Day 3: ChromaDB - Understanding Vector Databases

This script demonstrates:
1. Creating a ChromaDB collection
2. Storing embeddings with metadata
3. Searching by semantic similarity
4. Using metadata filters
"""

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

# Create ChromaDB client (in-memory, not persistent)
print("Creating ChromaDB client...")
client = chromadb.Client(Settings(anonymized_telemetry=False))

# Create a collection (like a table in SQL)
print("Creating 'mini_movies' collection...")
collection = client.create_collection(
    name="mini_movies",
    metadata={"description": "A mini movie database for learning"}
)
print(f"Collection created!\n")

# Sample movie data
movies = [
    {
        "title": "Inception",
        "description": "A mind-bending thriller about dreams within dreams",
        "genre": "Sci-Fi, Thriller",
        "director": "Christopher Nolan",
        "year": 2010,
        "rating": 8.8
    },
    {
        "title": "The Dark Knight",
        "description": "Batman faces the Joker in a dark crime thriller",
        "genre": "Action, Crime",
        "director": "Christopher Nolan",
        "year": 2008,
        "rating": 9.0
    },
    {
        "title": "The Notebook",
        "description": "A romantic love story spanning decades",
        "genre": "Romance, Drama",
        "director": "Nick Cassavetes",
        "year": 2004,
        "rating": 7.8
    },
    {
        "title": "The Matrix",
        "description": "A computer hacker discovers reality is a simulation",
        "genre": "Sci-Fi, Action",
        "director": "Lana Wachowski",
        "year": 1999,
        "rating": 8.7
    },
    {
        "title": "The Conjuring",
        "description": "Paranormal investigators help a family terrorized by a dark presence",
        "genre": "Horror",
        "director": "James Wan",
        "year": 2013,
        "rating": 7.5
    }
]

# Add movies to ChromaDB
print("Adding movies to ChromaDB...")
print("(This is what happens in services/movie-assistant-service/data_collection/create_vector_db.py)")
print()

for movie in movies:
    # Create text representation for embedding
    text = f"{movie['title']}. {movie['description']}. Genres: {movie['genre']}. Director: {movie['director']}"

    # Generate embedding
    embedding = model.encode(text).tolist()

    # Add to ChromaDB with metadata
    collection.add(
        ids=[movie['title']],  # Unique ID
        embeddings=[embedding],  # The 384-dim vector
        metadatas=[{  # Searchable metadata
            "title": movie['title'],
            "genre": movie['genre'],
            "director": movie['director'],
            "year": movie['year'],
            "rating": movie['rating']
        }],
        documents=[text]  # Original text (optional, for display)
    )
    print(f"  Added: {movie['title']}")

print(f"\nTotal movies in DB: {collection.count()}")

# ============================================================================
# Now let's SEARCH!
# ============================================================================

print("\n" + "="*60)
print("SEARCH EXPERIMENT 1: Semantic Search")
print("="*60)

query = "mind-bending science fiction"
print(f"Query: '{query}'")
print("(Note: 'mind-bending' doesn't appear in any movie description!)")
print()

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("Top 3 results:")
for i, (title, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
    similarity = 1 - distance  # Convert distance to similarity
    print(f"  {i+1}. {title} - {similarity*100:.1f}% match")

print("\nWhy Inception ranked #1?")
print("  The embedding captured 'mind-bending' relates to 'dreams within dreams'")
print("  Even though the exact words don't match!")

# ============================================================================
# SEARCH EXPERIMENT 2: Metadata Filtering
# ============================================================================

print("\n" + "="*60)
print("SEARCH EXPERIMENT 2: Metadata Filtering")
print("="*60)

query = "best movies"
print(f"Query: '{query}' with filter: director = 'Christopher Nolan'")
print()

results = collection.query(
    query_texts=[query],
    n_results=3,
    where={"director": "Christopher Nolan"}  # Only Nolan films
)

print("Results (only Nolan films):")
for i, title in enumerate(results['ids'][0]):
    meta = results['metadatas'][0][i]
    print(f"  {i+1}. {title} ({meta['year']}) - Rating: {meta['rating']}/10")

# ============================================================================
# SEARCH EXPERIMENT 3: Combined Semantic + Metadata
# ============================================================================

print("\n" + "="*60)
print("SEARCH EXPERIMENT 3: Semantic + Metadata Combined")
print("="*60)

query = "scary movies"
print(f"Query: '{query}' with filter: rating >= 7.5")
print()

results = collection.query(
    query_texts=[query],
    n_results=3,
    where={"rating": {"$gte": 7.5}}  # Simplified - single condition
)

print("Results:")
for i, title in enumerate(results['ids'][0]):
    meta = results['metadatas'][0][i]
    print(f"  {i+1}. {title} - {meta['genre']} - {meta['rating']}/10")

# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "="*60)
print("KEY CONCEPTS:")
print("="*60)
print("""
1. COLLECTION = Table in SQL
   - Stores embeddings + metadata + documents

2. EMBEDDINGS = The 384-dim vectors
   - Used for semantic similarity search

3. METADATA = Structured data (genre, year, rating)
   - Used for filtering (like SQL WHERE clause)

4. QUERY = Search operation
   - Finds top-K most similar items
   - Can combine semantic search + metadata filters

5. DISTANCE = How different two embeddings are
   - Lower distance = more similar
   - Similarity = 1 - distance
""")

print("="*60)
print("IN STREAMSAGE:")
print("="*60)
print("Movie Assistant has:")
print("  - Collection: 'movies'")
print("  - 6,147 documents")
print("  - 1024-dim embeddings (BGE-large)")
print("  - Metadata: title, year, genres, director, rating, cast")
print()
print("Oracle RAG has:")
print("  - Collection: '{movie_id}_subtitles'")
print("  - Variable documents (depends on subtitle length)")
print("  - 384-dim embeddings (all-MiniLM-L6-v2)")
print("  - Metadata: timestamp_start, timestamp_end, scene_index")
print("\n" + "="*60)
