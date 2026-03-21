# Day 2: Embeddings - The Foundation of Modern AI

**Time:** 3-4 hours
**Prerequisites:** Day 1 completed

---

## Learning Objectives

By the end of Day 2, you will understand:
- What embeddings are and why they matter
- How text is converted to numbers
- Semantic similarity vs keyword matching
- Sentence transformers
- How StreamSage uses embeddings

---

## What Are Embeddings?

Embeddings convert **text → numbers** that capture **meaning**:

```
"happy movie"     → [0.82, 0.15, -0.33, 0.67, ...]  (384 or 1024 numbers)
"joyful film"     → [0.80, 0.17, -0.31, 0.65, ...]  (similar!)
"sad documentary" → [-0.45, 0.62, 0.28, -0.51, ...] (very different)
```

**Key insight:** Similar meanings = similar vectors = we can find related items!

---

## Step 1: Run the Exercise (30 min)

```bash
cd c:/Users/mailp/StreamSage
python docs/learning-course/day-2/exercises/learn_embeddings.py
```

**What you'll see:**
```
Loading model...
Model loaded!

Generating embeddings...
Shape: (5, 384)  # 5 texts, each becomes 384 numbers

SIMILARITY SCORES:
'Action movie' vs 'Exciting film with pursuits': 0.747 [HIGH]
'Action movie' vs 'Romantic comedy': 0.377 [LOW]
```

**Key observation:**
- "Action movie" is 74.7% similar to "exciting film"
- Even though they share NO words!
- This is **semantic similarity**

---

## Step 2: Modify the Code (15 min)

**Exercise:** Add your own texts:

```python
texts = [
    "A thrilling action movie with car chases and explosions",
    "An exciting film with vehicle pursuits and action",
    "A romantic comedy about finding love in Paris",
    "A horror movie about a haunted house",
    "A documentary about climate change",
    # ADD YOUR OWN:
    "A mind-bending psychological thriller",
    "A heartwarming family drama",
]
```

**Then compare:**
```python
# Compare "Action movie" to new texts:
print("Action vs Psychological:", cosine_similarity(embeddings[0], embeddings[5]))
print("Action vs Family drama:", cosine_similarity(embeddings[0], embeddings[6]))
```

**Predict:** Which will be more similar? Why?

---

## Step 3: Understanding the Code (15 min)

**Key sections:**

### 1. Loading the Model
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```
- Downloads a pre-trained model (first time only ~90MB)
- Model was trained on billions of sentences
- Learned which words/phrases mean similar things

### 2. Generating Embeddings
```python
embeddings = model.encode(texts)
# texts: list of 5 strings
# embeddings: numpy array of shape (5, 384)
```
- Each text becomes 384 numbers
- Numbers capture semantic meaning
- Process is deterministic (same text → same embedding)

### 3. Cosine Similarity
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
- Measures angle between two vectors
- Returns 1.0 for identical (0° angle)
- Returns 0.0 for orthogonal (90° angle)
- Returns -1.0 for opposite (180° angle)

**Geometric visualization:**
```
Vector space:
           ╱ "action movie" [0.7, 0.5]
          ╱   small angle ≈ 25°
         ╱    similarity = cos(25°) = 0.90
        ○────→ "exciting film" [0.8, 0.4]
       origin

      ↑ "romantic comedy" [0.1, 0.9]
      │  large angle ≈ 75°
      │  similarity = cos(75°) = 0.26
      ○────→ "action movie"
```

---

## Step 4: Why This Matters for StreamSage (10 min)

### Movie Assistant
```python
# User query
query = "mind-bending sci-fi"

# Convert to embedding
query_embedding = model.encode(query)
# [0.23, -0.45, 0.78, ..., 0.12] (1024 dims using BGE-large)

# ChromaDB searches 6,147 movie embeddings
# Finds movies with similar embeddings
# Even if "mind-bending" doesn't appear in description!
```

### Oracle RAG
```python
# User question
question = "What did Neo say about the Matrix?"

# Convert to embedding
question_embedding = model.encode(question)
# [0.12, -0.34, 0.56, ..., 0.89] (384 dims using all-MiniLM)

# ChromaDB searches subtitle chunks
# Finds dialogue that answers the question
```

---

## Checkpoint Questions

### 1. What is the embedding dimension for all-MiniLM-L6-v2?

**Answer:** 384 dimensions
- Each text becomes a vector of 384 numbers
- Oracle service uses this model (small & fast)
- Movie Assistant uses BGE-large: 1024 dimensions (larger & more accurate)

---

### 2. Why is "action movie" similar to "exciting film" even though they share no words?

**Answer:** The embedding model was trained on billions of sentences and learned that:
- "action" and "exciting" often appear in similar contexts
- "movie" and "film" mean the same thing
- The semantic meaning is captured in the vector space
- Words with similar meanings have similar vector representations

---

### 3. If you search for "Christopher Nolan films" in ChromaDB, what does the system actually compare?

**Answer:**
1. Your query "Christopher Nolan films" → embedded → `[0.12, -0.45, 0.89, ...]` (1024 numbers)
2. ChromaDB has 6,147 movie embeddings stored
3. It calculates cosine similarity between your query vector and all 6,147 movie vectors
4. Returns the movies with the highest similarity scores
5. Result: Inception, Interstellar, The Dark Knight (even if "Nolan" isn't in their descriptions!)

---

### 4. What's the difference between keyword matching and semantic search?

**Keyword matching (old way):**
```
Query: "mind-bending"
Database: Search for exact word "mind-bending" in descriptions
Result: Only movies containing that exact word
Problem: Misses "psychological thriller", "thought-provoking", etc.
```

**Semantic search (embedding way):**
```
Query: "mind-bending" → embedding
Database: Find similar embeddings
Result: Movies about complex psychology, reality-questioning plots
Benefit: Finds semantically related content, not just keyword matches
```

---

## Interview Q&A

### Q: Explain embeddings to a non-technical person.

**Answer:**
> Imagine you're organizing books in a library, but instead of alphabetical order, you arrange them by how similar their stories are. Books about space exploration would be close together, even if they have different authors or titles. Embeddings do the same thing with text - they convert words into numbers that capture their meaning, so a computer can understand "happy" and "joyful" are similar, even though they're different words.

---

### Q: Why use a pre-trained model instead of training your own?

**Answer:**
> Pre-trained models like all-MiniLM-L6-v2 were trained on billions of sentences, which would take months and cost thousands of dollars. They've learned general language understanding that works for most tasks. We can use them immediately for tasks like movie search without any training. If we needed domain-specific understanding (medical terms, legal language), we could fine-tune them, but for general movie descriptions, pre-trained models work great.

---

### Q: What's the trade-off between 384-dim and 1024-dim embeddings?

**Answer:**

| Dimension | Model | Quality | Speed | Memory | Use Case |
|-----------|-------|---------|-------|--------|----------|
| 384 | all-MiniLM-L6-v2 | Good | Fast | 90 MB | Oracle (subtitles) |
| 1024 | BGE-large | Excellent | Slower | 1.3 GB | Movie Assistant |

For Oracle, subtitles are short chunks, so 384 dims is sufficient. For Movie Assistant, we're comparing complex movie descriptions with genres, cast, plots, so 1024 dims captures more nuance.

---

## Next Steps

Once you complete Day 2:
- ✅ You understand what embeddings are
- ✅ You know how to generate them
- ✅ You understand semantic similarity
- 🎯 Ready for Day 3: **ChromaDB** (where these embeddings are stored!)

---

[← Back to Course](../README.md) | [Continue to Day 3 →](../day-3/README.md)
