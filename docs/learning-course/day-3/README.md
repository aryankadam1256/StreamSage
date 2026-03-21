# Day 3: ChromaDB - Vector Databases

**Time:** 4-5 hours
**Prerequisites:** Day 1 & Day 2 completed

---

## Learning Objectives

By the end of Day 3, you will understand:
- How ChromaDB stores data (format & structure)
- HNSW indexing for fast similarity search
- Metadata filtering combined with vector search
- How StreamSage uses ChromaDB for two different use cases
- Complex code patterns in `create_vector_db.py`

---

## Files in This Module

| File | Description | Time |
|------|-------------|------|
| [Data-Storage-Format.md](Data-Storage-Format.md) | How ChromaDB stores data internally | 1 hour |
| [Complex-Code-Explained.md](Complex-Code-Explained.md) | Breaking down create_vector_db.py | 1.5 hours |
| [Flow-Diagram.md](Flow-Diagram.md) | Complete flow diagrams | 1 hour |
| [Interview-QA.md](Interview-QA.md) | Interview questions & answers | 30 min |
| [exercises/learn_chromadb.py](exercises/learn_chromadb.py) | Hands-on coding | 1 hour |

---

## Learning Path

### Step 1: Understand the Basics (30 min)

Read [Data-Storage-Format.md](Data-Storage-Format.md)

**Focus on:**
- Section 1: Storage architecture (SQLite + binary)
- Section 2: Collection structure
- Section 6: Real data example from StreamSage

**Checkpoint:** Can you explain where metadata vs embeddings are stored?

---

### Step 2: Run the Exercise (30 min)

Run the hands-on script:
```bash
cd c:/Users/mailp/StreamSage
python learn_chromadb.py
```

**What you'll learn:**
- Creating a ChromaDB collection
- Adding documents with embeddings
- Semantic search queries
- Metadata filtering

**Experiment:**
- Change the query to "romantic movies"
- Add a filter for movies after 2005
- See how results change

---

### Step 3: Deep Dive into Complex Code (1.5 hours)

Read [Complex-Code-Explained.md](Complex-Code-Explained.md)

Then open the actual file in VS Code:
```
services/movie-assistant-service/data_collection/create_vector_db.py
```

**Focus on:**
- Lines 34-66: `create_document_text()` function
- Lines 85-111: Custom embedding function with retry logic
- Lines 143-174: Batch processing loop

**As you read, ask:**
- Why is this text format used?
- What happens if the API fails?
- Why batch 25 movies at a time?

---

### Step 4: Understand the Full Pipeline (1 hour)

Read [Flow-Diagram.md](Flow-Diagram.md)

**Study both diagrams:**
1. Movie Discovery flow (Movie Assistant)
2. Oracle RAG flow (Subtitle Q&A)

**Trace a request manually:**
- Pick a query: "Christopher Nolan films"
- Trace through each step in the diagram
- Understand what happens at each stage

---

### Step 5: Interview Preparation (30 min)

Review [Interview-QA.md](Interview-QA.md)

**Read all 10 questions and practice answering them OUT LOUD:**
- Q1-3: Basic concepts
- Q4-6: Data structures
- Q7-10: Operations and scaling

**Practice:**
- Record yourself answering Q3 (HNSW explanation)
- If you can explain it clearly, you understand it!

---

## Checkpoint Questions

Answer these to test your understanding:

### 1. ChromaDB Storage
**Q:** Where are movie embeddings stored physically?
- [ ] In SQLite as BLOB
- [ ] In binary files
- [ ] In JSON files

**Q:** Where are metadata fields (title, year, rating) stored?
- [ ] In binary files
- [ ] In SQLite tables
- [ ] In memory only

---

### 2. Query Execution
**Q:** In what order do these operations happen?
- [ ] A. Apply metadata filter → B. Vector search → C. Embed query
- [ ] A. Embed query → B. Apply metadata filter → C. Vector search
- [ ] A. Vector search → B. Embed query → C. Apply metadata filter

---

### 3. Performance
**Q:** Why is ChromaDB faster than brute-force search?
- [ ] It uses caching
- [ ] It uses HNSW graph indexing
- [ ] It uses GPU acceleration
- [ ] It uses multi-threading

---

### 4. Data Types
**Q:** Which metadata is INVALID?
```python
metadata = {
    "title": "Inception",
    "genres": ["Sci-Fi", "Thriller"],  # <-- This one?
    "year": 2010,
    "rating": 8.8
}
```

---

### 5. Real-World Usage
**Q:** In StreamSage, which service uses ChromaDB?
- [ ] Movie Assistant only
- [ ] Oracle RAG only
- [ ] Movie Assistant and Oracle RAG
- [ ] All services

---

## Practical Exercise

**Challenge:** Modify `learn_chromadb.py` to add these movies:

```python
{
    "title": "Parasite",
    "description": "A poor family schemes to become employed by a wealthy family",
    "genre": "Thriller, Drama",
    "director": "Bong Joon-ho",
    "year": 2019,
    "rating": 8.5
},
{
    "title": "Everything Everywhere All at Once",
    "description": "A multiverse adventure about family and identity",
    "genre": "Sci-Fi, Comedy",
    "director": "Dan Kwan",
    "year": 2022,
    "rating": 8.0
}
```

Then query:
- "family drama movies"
- "multiverse sci-fi"

See which movie ranks higher for each query!

---

## Next Steps

Once you complete Day 3:
- ✅ You understand embeddings (Day 2)
- ✅ You understand vector databases (Day 3)
- 🎯 Ready for Day 4: **RAG (Retrieval-Augmented Generation)**

RAG combines embeddings + ChromaDB + LLMs to answer questions!

---

[← Back to Course](../README.md) | [Continue to Day 4 →](../day-4/README.md)
