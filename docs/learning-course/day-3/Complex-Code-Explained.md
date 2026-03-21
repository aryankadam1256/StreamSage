# Complex Code Explained - create_vector_db.py

Breaking down the complex parts of `services/movie-assistant-service/data_collection/create_vector_db.py` with detailed examples.

---

## File Overview

**Purpose:** Convert 6,147 movies from JSON → ChromaDB vector database

**Location:** `services/movie-assistant-service/data_collection/create_vector_db.py`

**What it does:**
1. Loads movie data from JSON
2. Creates rich text representations
3. Generates embeddings via HuggingFace API
4. Stores in ChromaDB with metadata

---

## Part 1: Creating Rich Text Representations

### Function: `create_document_text()`

**Location:** Lines 34-66

```python
def create_document_text(movie: Dict) -> str:
    """
    Create a rich text representation of the movie for embedding.
    Includes title, overview, genres, mood, and keywords.
    """
    # Basic info
    title = movie.get("title", "")
    year = movie.get("release_year", "")
    genres = ", ".join(movie.get("genres", []))

    # Rich descriptors
    moods = ", ".join(movie.get("mood_tags", []))
    keywords = ", ".join(movie.get("keywords", [])[:10])

    # Cast & Crew
    director = movie.get("director", {}).get("name", "") if movie.get("director") else ""
    cast = ", ".join(movie.get("cast_names", [])[:3])

    # Overview
    overview = movie.get("overview", "")

    # Construct the text blob
    text = f"Title: {title} ({year})\n"
    text += f"Genres: {genres}\n"
    if director:
        text += f"Director: {director}\n"
    if cast:
        text += f"Cast: {cast}\n"
    text += f"Mood: {moods}\n"
    text += f"Keywords: {keywords}\n"
    text += f"Plot: {overview}"

    return text
```

### Example Input/Output:

**Input (movie dictionary):**
```python
{
    "title": "Inception",
    "release_year": 2010,
    "genres": ["Sci-Fi", "Thriller", "Action"],
    "mood_tags": ["thought_provoking", "mind_bending", "intense"],
    "keywords": ["dream", "subconscious", "heist", "reality", "layers", "psychology"],
    "director": {"name": "Christopher Nolan"},
    "cast_names": ["Leonardo DiCaprio", "Ellen Page", "Tom Hardy"],
    "overview": "A thief who steals corporate secrets through dream-sharing..."
}
```

**Output (text for embedding):**
```
Title: Inception (2010)
Genres: Sci-Fi, Thriller, Action
Director: Christopher Nolan
Cast: Leonardo DiCaprio, Ellen Page, Tom Hardy
Mood: thought_provoking, mind_bending, intense
Keywords: dream, subconscious, heist, reality, layers, psychology
Plot: A thief who steals corporate secrets through dream-sharing
technology is given the inverse task of planting an idea...
```

### Why This Format?

1. **Structured sections** - Embedding model learns different concepts:
   - "Title:" → Movie identity
   - "Genres:" → Category
   - "Director:" → Creator style
   - "Mood:" → Emotional tone
   - "Keywords:" → Core themes
   - "Plot:" → Narrative

2. **Rich context** - More information = better embeddings:
   ```
   Bad:  "Inception"
   Good: "Title: Inception (2010)\nGenres: Sci-Fi, Thriller\n..."

   Query: "mind-bending sci-fi"
   Bad embedding can't match well
   Good embedding sees "mind_bending" mood + "Sci-Fi" genre
   ```

3. **Consistent format** - All movies follow same structure:
   - Makes embeddings comparable
   - Model learns the pattern

---

## Part 2: Custom Embedding Function with Retry Logic

### Class: `HFInferenceEmbeddingFunction`

**Location:** Lines 85-111

```python
class HFInferenceEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        import time

        for text in input:
            retries = 3
            for attempt in range(retries):
                try:
                    # Add delay to avoid rate limits
                    time.sleep(0.5)
                    embedding = self.client.feature_extraction(text, model=self.model_name)
                    embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
                    break # Success, exit retry loop
                except Exception as e:
                    if attempt < retries - 1:
                        wait = (attempt + 1) * 2
                        logger.warning(f"API Error: {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Failed to embed text after {retries} attempts: {e}")
                        embeddings.append([0.0] * 384) # Zero vector fallback
        return embeddings
```

### Understanding This Code:

**1. Why a custom class?**
- ChromaDB expects a callable that returns embeddings
- We need to call HuggingFace API with retry logic
- Inherit from `EmbeddingFunction` to integrate with ChromaDB

**2. The retry mechanism:**
```python
retries = 3
for attempt in range(retries):
    # Attempt 0, 1, 2
```

**Example execution flow:**
```
Attempt 0: sleep(0.5s) → API call → ❌ Rate limit error
           → wait = (0+1)*2 = 2s → sleep(2s)

Attempt 1: sleep(0.5s) → API call → ❌ Network timeout
           → wait = (1+1)*2 = 4s → sleep(4s)

Attempt 2: sleep(0.5s) → API call → ✅ Success!
           → break (exit loop)
```

**3. The fallback:**
```python
embeddings.append([0.0] * 384)  # Zero vector
```
If all 3 attempts fail, use a zero vector:
- Prevents crashing the entire pipeline
- Zero vector has low similarity to everything
- Can be fixed later by re-embedding

**4. Rate limiting:**
```python
time.sleep(0.5)  # Wait 0.5s between requests
```
HuggingFace free tier: ~1,000 requests/hour
- 6,147 movies = 6,147 requests
- At 0.5s per request = ~51 minutes total
- Prevents "429 Too Many Requests" errors

---

## Part 3: Batch Processing

### The Main Loop

**Location:** Lines 143-174

```python
batch_size = 25
movies_items = list(movies_data.items())
total_movies = len(movies_items)

for i, (movie_id, movie) in enumerate(movies_items):
    # Create embedding text
    doc_text = create_document_text(movie)

    # Prepare metadata
    metadata = {
        "title": movie.get("title", ""),
        "year": movie.get("release_year") or 0,
        "rating": movie.get("vote_average", 0.0),
        "popularity_score": movie.get("popularity_score", 0.0),
        "genres": ", ".join(movie.get("genres", [])),
        "moods": ", ".join(movie.get("mood_tags", [])),
        "director": movie.get("director", {}).get("name", "") if movie.get("director") else "",
    }

    ids.append(str(movie_id))
    documents.append(doc_text)
    metadatas.append(metadata)

    # Batch insert every 25 movies
    if len(ids) >= batch_size:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Processed {i+1}/{total_movies} movies...")
        ids = []
        documents = []
        metadatas = []
```

### Why Batching?

**Without batching (single API call per movie):**
```
Movie 1 → API call → 1 second
Movie 2 → API call → 1 second
Movie 3 → API call → 1 second
...
6,147 movies → 6,147 seconds = 102 minutes
```

**With batching (25 movies per call):**
```
Movies 1-25   → 1 API call → 1 second
Movies 26-50  → 1 API call → 1 second
Movies 51-75  → 1 API call → 1 second
...
6,147 movies → 246 API calls = 4-5 minutes
```

**Savings: 20x faster!**

### Visual Representation:

```
Accumulation Phase:
ids = []
documents = []
metadatas = []

Add Movie 1 → ids = ["550"]
              documents = ["Title: Fight Club..."]
              metadatas = [{"title": "Fight Club", ...}]

Add Movie 2 → ids = ["550", "13"]
   ...

Add Movie 25 → ids = ["550", "13", ..., "680"]  # 25 items
               BATCH FULL!

Insert Phase:
collection.upsert(
    ids=["550", "13", ..., "680"],
    documents=[...],  # 25 documents
    metadatas=[...]   # 25 metadata dicts
)

Reset:
ids = []
documents = []
metadatas = []

(Repeat for next batch)
```

---

## Part 4: Metadata Transformation

### Why This Code?

```python
metadata = {
    "title": movie.get("title", ""),
    "year": movie.get("release_year") or 0,  # ← Note the 'or 0'
    "rating": movie.get("vote_average", 0.0),
    "genres": ", ".join(movie.get("genres", [])),  # ← List to string
}
```

### Problem & Solution:

**Input data structure:**
```python
{
    "title": "Inception",
    "release_year": 2010,
    "genres": ["Sci-Fi", "Thriller"],  # ← List
    "director": {                       # ← Nested dict
        "name": "Christopher Nolan",
        "id": 525
    }
}
```

**ChromaDB metadata restrictions:**
```python
# ✅ Allowed types:
"title": "Inception"        # str
"year": 2010                # int
"rating": 8.8               # float
"is_favorite": True         # bool

# ❌ NOT allowed:
"genres": ["Sci-Fi", "Thriller"]     # List - ERROR!
"director": {"name": "Nolan", ...}   # Dict - ERROR!
```

**Transformations:**
```python
# 1. List → comma-separated string
["Sci-Fi", "Thriller"] → "Sci-Fi, Thriller"

# 2. Nested dict → extract value
{"name": "Christopher Nolan", "id": 525} → "Christopher Nolan"

# 3. None values → default
release_year = None → 0

# 4. Missing keys → default
.get("title", "") → "" if key doesn't exist
```

---

## Part 5: The Upsert Operation

```python
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)
```

### What's "Upsert"?

**Upsert = Update + Insert**

```python
# Scenario 1: ID doesn't exist → INSERT
collection.upsert(
    ids=["NEW_ID"],
    documents=["New movie"],
    metadatas=[{"title": "New Movie"}]
)
# Result: New document added

# Scenario 2: ID already exists → UPDATE
collection.upsert(
    ids=["550"],  # Already exists!
    documents=["Updated text for Fight Club"],
    metadatas=[{"title": "Fight Club", "rating": 9.0}]  # Updated rating
)
# Result: Existing document updated (not duplicated)
```

### Why Not Separate Insert/Update?

**With upsert:**
```python
# Simple - works for both cases
collection.upsert(ids=[...], documents=[...], metadatas=[...])
```

**Without upsert:**
```python
# Complex - need to check first
if collection.get(ids=["550"]):
    collection.update(ids=["550"], ...)
else:
    collection.add(ids=["550"], ...)
```

---

## Part 6: Error Handling

### The Try-Except Pattern:

```python
for attempt in range(retries):
    try:
        embedding = self.client.feature_extraction(text, model=self.model_name)
        embeddings.append(embedding.tolist())
        break  # Success - exit retry loop
    except Exception as e:
        if attempt < retries - 1:
            # Not last attempt - retry
            wait = (attempt + 1) * 2
            logger.warning(f"API Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
        else:
            # Last attempt - give up
            logger.error(f"Failed after {retries} attempts: {e}")
            embeddings.append([0.0] * 384)
```

### Example Error Scenarios:

**Scenario 1: Temporary network glitch**
```
Attempt 0: ❌ Connection timeout
Attempt 1: ✅ Success!
(Only 1 retry needed)
```

**Scenario 2: Persistent API error**
```
Attempt 0: ❌ API rate limit
Attempt 1: ❌ API rate limit
Attempt 2: ❌ API rate limit
→ Use zero vector fallback
(All retries exhausted)
```

**Scenario 3: Invalid text**
```
Attempt 0: ❌ Text too long (>512 tokens)
Attempt 1: ❌ Text too long
Attempt 2: ❌ Text too long
→ Use zero vector
(Retrying won't help - text is the problem)
```

---

## Part 7: Collection Configuration

```python
collection = chroma_client.get_or_create_collection(
    name="movies",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)
```

### Understanding Each Parameter:

**1. `name="movies"`**
- Collection identifier
- Like a SQL table name
- Must be unique per ChromaDB instance

**2. `embedding_function=emb_fn`**
- Auto-embeds documents on insert
- When you add text, it automatically calls the embedding function
- No need to manually compute embeddings

**3. `metadata={"hnsw:space": "cosine"}`**
- Distance metric for similarity
- Options:
  - `"cosine"` - Measures angle (good for text, scale-invariant)
  - `"l2"` - Euclidean distance (sensitive to magnitude)
  - `"ip"` - Inner product (for normalized vectors)

**Why cosine for text?**
```python
# Example:
embedding_1 = [1, 0, 0]
embedding_2 = [2, 0, 0]  # Same direction, different magnitude

# Cosine similarity = 1.0 (identical direction)
# L2 distance = 1.0 (different magnitude)

# For text, direction matters more than magnitude
# "very happy" and "happy" point the same semantic direction
```

---

## Summary: Complete Pipeline

```
1. LOAD DATA
   movie_features.json → 6,147 movies in memory

2. CREATE COLLECTION
   Initialize ChromaDB with cosine similarity

3. FOR EACH MOVIE:
   a. create_document_text() → Rich text representation
   b. Extract metadata → Transform lists/dicts to primitives
   c. Accumulate in batch (25 movies)

4. WHEN BATCH FULL:
   a. HFInferenceEmbeddingFunction() → Generate 25 embeddings
   b. collection.upsert() → Insert all 25 at once
   c. Reset batch arrays

5. FINAL BATCH:
   Insert remaining movies (< 25)

6. DONE:
   6,147 movies stored in ChromaDB
```

**Total time: ~51 minutes** (for free HuggingFace API with rate limiting)

**Output: `data/chroma_db/` folder** containing SQLite + binary files

---

[← Back to Day 3](README.md) | [Next: Flow Diagram →](Flow-Diagram.md)
