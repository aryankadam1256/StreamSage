# Day 3 Interview Questions & Answers - ChromaDB

Common interview questions about vector databases with strong, detailed answers.

---

## Basic Concepts

### Q1: What is ChromaDB and why would you use it?

**Answer:**

ChromaDB is a **vector database** designed for storing and searching high-dimensional embeddings efficiently.

**Key features:**
- Stores text embeddings (vectors) alongside metadata
- Performs fast similarity search using HNSW indexing
- Supports metadata filtering (like SQL WHERE clauses)
- Embeds directly in Python applications

**Why use it:**
- Traditional databases (SQL) are optimized for exact matches (`WHERE title = 'Inception'`)
- ChromaDB is optimized for similarity (`find movies SIMILAR TO Inception`)
- Use case: Semantic search, recommendation systems, RAG applications

**In StreamSage:**
- Movie Assistant: Finds semantically similar movies
- Oracle RAG: Retrieves relevant subtitle chunks for Q&A

---

### Q2: How does ChromaDB store data internally?

**Answer:**

ChromaDB uses a **hybrid storage approach**:

1. **SQLite database** (chroma.sqlite3):
   - Collections metadata
   - Document text
   - Metadata (title, year, rating, etc.)
   - Used for fast filtering

2. **Binary files** (collection_id/):
   - Embeddings (vectors) as float32 arrays
   - HNSW graph structure
   - Used for vector operations

**Why this hybrid approach?**
- SQL is great for structured queries: `WHERE year > 2010`
- Binary files are efficient for vector math: cosine similarity calculations
- Combining both gives us **filtered similarity search**

**Example:**
```python
# Find sci-fi movies from 2010+ similar to "Inception"
collection.query(
    query_texts=["Inception-like movie"],
    where={"year": {"$gte": 2010}, "genre": "Sci-Fi"}
)
# SQL filters candidates, vector search finds most similar
```

---

### Q3: Explain HNSW. Why is it faster than brute-force search?

**Answer:**

**HNSW** = Hierarchical Navigable Small World graph

**Brute-force approach:**
```python
# Compare query to ALL documents
for doc in all_6147_movies:
    similarity = cosine_similarity(query, doc.embedding)
# Time: O(n) = 6,147 comparisons
```

**HNSW approach:**
```
Layer 3: [A] ←→ [B] ←→ [C]        (coarse, few nodes)
           ↓      ↓      ↓
Layer 2: [A-D-E] [F-G] [H-C]      (medium density)
           ↓ ↓ ↓   ↓ ↓   ↓ ↓
Layer 1: [All 6,147 nodes]         (fine, all movies)

Search process:
1. Start at top layer (Layer 3)
2. Navigate to nearest neighbor
3. Drop down a layer
4. Repeat until bottom
5. Return K nearest neighbors

Time: O(log n) ≈ 12 comparisons instead of 6,147
```

**Why it works:**
- Top layers create "shortcuts" across the space
- Each layer gets progressively more detailed
- Never need to check every single document

**Performance in StreamSage:**
- Brute force: ~500ms for 6,147 movies
- HNSW: ~30ms for 6,147 movies
- **16x speedup!**

---

### Q4: What's the difference between ChromaDB and a traditional vector database like Pinecone?

**Answer:**

| Feature | ChromaDB | Pinecone |
|---------|----------|----------|
| **Deployment** | Embedded in Python | Cloud-hosted |
| **Setup** | `pip install chromadb` | API key + subscription |
| **Best for** | Development, small-medium scale | Production, large scale |
| **Cost** | Free | Paid (after free tier) |
| **Scaling** | Single machine | Distributed across servers |
| **Latency** | ~30ms (local) | ~50-100ms (network) |

**When to use ChromaDB:**
- Prototyping and development
- < 10M documents
- Don't want to manage infrastructure
- Need offline capabilities

**When to use Pinecone:**
- Production at scale (millions of users)
- > 10M documents
- Need multi-region redundancy
- Want managed service

**In StreamSage:**
- ChromaDB is perfect for 6,147 movies
- Switching to Pinecone would be overkill (and cost money)

---

## Data Structures

### Q5: What data types can you store as metadata in ChromaDB?

**Answer:**

**Allowed types:**
```python
metadata = {
    "title": "Inception",       # str ✓
    "year": 2010,                # int ✓
    "rating": 8.8,               # float ✓
    "is_favorite": True          # bool ✓
}
```

**NOT allowed:**
```python
metadata = {
    "genres": ["Sci-Fi", "Thriller"],    # list ✗
    "director": {"name": "Nolan"},       # dict ✗
    "tags": None                          # None ✗
}
```

**Why the restriction?**
- Metadata is stored in SQLite columns
- SQL columns must have fixed types
- Simplifies indexing for fast filtering

**Workaround for lists/dicts:**
```python
# Convert list to comma-separated string
"genres": "Sci-Fi, Thriller, Action"

# Extract value from dict
"director": "Christopher Nolan"  # instead of {"name": "Nolan"}
```

**Example from StreamSage:**
```python
metadata = {
    "title": movie.get("title", ""),
    "year": movie.get("release_year") or 0,
    "rating": movie.get("vote_average", 0.0),
    "genres": ", ".join(movie.get("genres", [])),  # List → String
    "director": movie.get("director", {}).get("name", "")  # Dict → String
}
```

---

### Q6: How does the upsert operation work?

**Answer:**

**Upsert = Update + Insert** (does both intelligently)

**Behavior:**
```python
collection.upsert(
    ids=["550"],
    embeddings=[[0.23, -0.45, ...]],
    metadatas=[{"title": "Fight Club", "rating": 8.8}]
)
```

**If ID "550" doesn't exist:**
- Insert new document

**If ID "550" already exists:**
- Update existing document (replace embedding, metadata, document)

**Why upsert is better than separate insert/update:**

Without upsert:
```python
# Have to check first
if collection.get(ids=["550"]):
    collection.update(...)
else:
    collection.add(...)
# More code, multiple DB operations
```

With upsert:
```python
# Just do it - ChromaDB handles the logic
collection.upsert(...)
# One operation, cleaner code
```

**Use case in StreamSage:**
- Re-running `create_vector_db.py` doesn't create duplicates
- Updating movie metadata (new rating) replaces old data
- Idempotent operation (safe to run multiple times)

---

## Operations

### Q7: Explain how metadata filtering works in ChromaDB queries.

**Answer:**

Metadata filtering uses **SQL-like WHERE clauses** to narrow search:

**Basic filter:**
```python
# Find movies from 2010 or later
collection.query(
    query_texts=["action movies"],
    where={"year": {"$gte": 2010}}
)
```

**Compound filters:**
```python
# Year >= 2000 AND rating > 8.0
collection.query(
    query_texts=["thriller"],
    where={
        "$and": [
            {"year": {"$gte": 2000}},
            {"rating": {"$gt": 8.0}}
        ]
    }
)
```

**Operators available:**
- `$eq`: Equal to
- `$ne`: Not equal
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$lt`: Less than
- `$lte`: Less than or equal
- `$in`: In list
- `$nin`: Not in list

**Internal process:**
```
1. SQL Filter Phase:
   SELECT id FROM embedding_metadata
   WHERE key = 'year' AND int_value >= 2010

   Result: 3,200 movie IDs (out of 6,147)

2. Vector Search Phase:
   Only search embeddings for those 3,200 IDs
   Find top K most similar

3. Combine:
   Return movies that:
   - Match metadata filter (year >= 2010)
   - Are semantically similar to query
```

**Performance benefit:**
- Without filter: Search 6,147 movies (~50ms)
- With filter: Search 3,200 movies (~25ms)
- Filter reduces search space = faster results

---

### Q8: Walk me through a complete query execution from user input to response.

**Answer:**

**User query:** "mind-bending sci-fi from 2010+"

**Step-by-step execution:**

**1. Embed the query (10ms)**
```python
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
query_embedding = model.encode("mind-bending sci-fi from 2010+")
# Result: [0.23, -0.45, 0.78, ..., 0.12] (1024 floats)
```

**2. Detect filters from query (5ms)**
```python
# NLP parsing extracts: year >= 2010
filters = {"year": {"$gte": 2010}}
```

**3. SQL metadata filter (5ms)**
```sql
SELECT id FROM embedding_metadata
WHERE key = 'year' AND int_value >= 2010
```
```
Result: [550, 27, 155, ..., 680] (3,200 IDs)
```

**4. HNSW vector search on filtered set (25ms)**
```python
# Only search embeddings for IDs from step 3
# Navigate HNSW graph
# Compute cosine similarity
# Return top 100 candidates by similarity
```
```
Result IDs: [550, 27, 155, 680, 13]
Distances: [0.12, 0.18, 0.23, 0.25, 0.28]
```

**5. Retrieve full data (5ms)**
```sql
SELECT e.id, em.*, ed.document
FROM embeddings e
JOIN embedding_metadata em ON e.id = em.id
JOIN embedding_documents ed ON e.id = ed.id
WHERE e.id IN (550, 27, 155, 680, 13)
```

**6. Format response**
```python
{
    'ids': [['550', '27', '155', '680', '13']],
    'distances': [[0.12, 0.18, 0.23, 0.25, 0.28]],
    'metadatas': [[
        {"title": "Inception", "year": 2010, "rating": 8.8},
        {"title": "Interstellar", "year": 2014, "rating": 8.6},
        ...
    ]],
    'documents': [["Title: Inception...", ...]]
}
```

**Total time: ~50ms**

**7. Application post-processing**
- Convert distance → similarity (1 - distance)
- Sort by relevance
- Add recommendation reasons
- Return to frontend

---

## Advanced Topics

### Q9: How would you handle updates to movie data in ChromaDB?

**Answer:**

**Scenario:** A movie's rating changes from 8.0 to 8.8

**Option 1: Upsert (Recommended)**
```python
# Update just the metadata
collection.upsert(
    ids=["550"],
    metadatas=[{"title": "Fight Club", "rating": 8.8}]  # New rating
)
# Pros: Simple, preserves embedding
# Cons: Need to know the ID
```

**Option 2: Delete + Re-add**
```python
collection.delete(ids=["550"])
collection.add(
    ids=["550"],
    embeddings=[new_embedding],
    metadatas=[new_metadata]
)
# Pros: Complete refresh
# Cons: Requires re-embedding (expensive)
```

**Option 3: Partial metadata update (if ChromaDB supports it)**
```python
collection.update_metadata(
    ids=["550"],
    metadata={"rating": 8.8}
)
# Pros: Only touches what changed
# Cons: May not be available in all ChromaDB versions
```

**Best practice in StreamSage:**
- Metadata-only changes (rating, popularity): Use upsert
- Content changes (new overview, genres): Re-embed and upsert
- Bulk updates: Run `create_vector_db.py` script (idempotent)

---

### Q10: What challenges would you face scaling ChromaDB to 1 million movies?

**Answer:**

**Challenges:**

1. **Memory**
   - 1M movies × 1024 dims × 4 bytes = ~4 GB for embeddings alone
   - HNSW graph: Additional ~3-5 GB
   - Total: ~10 GB RAM minimum

2. **Search latency**
   - HNSW at 1M docs: ~150-200ms (vs 30ms for 6K)
   - Still acceptable, but noticeable

3. **Disk space**
   - Embeddings: 4 GB
   - Metadata (SQLite): 1-2 GB
   - Documents: 2-3 GB
   - Total: ~10 GB disk

4. **Index build time**
   - Building HNSW for 1M docs: ~30-60 minutes
   - Startup time increases

**Solutions:**

1. **Distributed sharding**
   ```python
   shard_1 = movies 1-250K     (Action, Adventure)
   shard_2 = movies 250K-500K  (Comedy, Drama)
   shard_3 = movies 500K-750K  (Horror, Sci-Fi)
   shard_4 = movies 750K-1M    (Romance, Thriller)

   # Query all shards in parallel, merge results
   ```

2. **Switch to managed service**
   - Pinecone, Weaviate, Milvus
   - Handle scaling automatically
   - Cost: ~$100-500/month

3. **Optimize query**
   - Metadata pre-filtering reduces search space
   - `WHERE genre = 'Sci-Fi'` → search only 100K instead of 1M

4. **Lower-dimensional embeddings**
   - Switch from 1024-dim to 384-dim
   - Saves 60% memory
   - Small quality loss

**When to make the switch:**
- < 100K docs: ChromaDB embedded
- 100K - 1M docs: ChromaDB server mode
- > 1M docs: Managed service (Pinecone, etc.)

---

## Summary: Key Points for Interviews

1. **ChromaDB stores embeddings + metadata + documents** in hybrid format
2. **HNSW graph** enables fast similarity search (O(log n) vs O(n))
3. **Metadata filtering** combines SQL power with vector search
4. **Upsert** prevents duplicates and simplifies updates
5. **Good for < 1M documents**, then consider managed services

---

[← Back to Day 3](README.md) | [View Exercises →](exercises/)
