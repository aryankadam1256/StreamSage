# Vector Database & Retrieval: Theory vs. Implementation

In this document, we bridge the gap between **RAG Theory** (what you study for an interview) and our **Project Implementation** (what we actually built in the Movie Assistant).

---

## 1. The Vector Database (ChromaDB)

### 📚 The Theory
Unlike a traditional SQL database that searches for exact keyword matches (e.g., `WHERE genre = 'Action'`), a **Vector Database** searches by meaning (Semantic Search). It stores text as high-dimensional arrays of numbers (embeddings). When a user searches, the DB calculates the geometric distance between the user's query vector and the document vectors.

### 💻 Our Project Implementation (`create_vector_db.py`)
In our project, we used **ChromaDB**. But we didn't just dump raw JSON into it. 

**The Text Assembly Trick:**
Before converting a movie to an embedding, our code combines all the engineered features from Unit 2 into a single, dense "Rich Text Blob".
```python
# From create_vector_db.py
text = f"Title: {title} ({year})\n"
text += f"Genres: {genres}\n"
text += f"Mood: {moods}\n"
text += f"Keywords: {keywords}\n"
text += f"Plot: {overview}"
```
**Why?** The embedding model (`all-MiniLM-L6-v2`) only understands text. By flattening the data this way, the vector captures the *relationship* between the mood, the plot, and the keywords simultaneously.

---

## 2. Hybrid Search (BM25 + Vectors)

### 📚 The Theory
Semantic vector search is amazing at understanding concepts (e.g., matching "aliens" to "extraterrestrial"), but it actually struggles with exact keywords (e.g., "Christopher Nolan" or specific character names). 
To fix this, modern RAG systems use **Hybrid Search**: they run a Vector Search (for meaning) AND a Keyword Search (BM25/TF-IDF) simultaneously, and combine the results.

### 💻 Our Project Implementation (`main.py`)
If you look closely at our `main.py`, we implemented a highly advanced Hybrid Search mechanism:
1. **ChromaDB:** Handles the semantic vector search.
2. **BM25Okapi:** During startup, our app loads all documents into a BM25 index.
3. **The Fallback / Combination:** If the vector search misses an exact keyword match, the BM25 index catches it by calculating term frequency. 

---

## 3. Cross-Encoder Re-Ranking

### 📚 The Theory
Standard embedding models (Bi-Encoders) are fast but slightly inaccurate because they compare two vectors independently using a simple dot-product. 
A **Cross-Encoder** is a more powerful neural network that looks at the User Query and the Document *at the same time* and scores their relevance from 0 to 1. It is too slow to run on 6,000 movies, but perfect for re-scoring the top 50 results.

### 💻 Our Project Implementation (`main.py`)
We implemented a two-stage retrieval pipeline:
1. **Stage 1 (Fast):** ChromaDB grabs the top ~100 candidate movies in milliseconds.
2. **Stage 2 (Precise):** We pass those 100 movies through our Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). 
```python
# Concept from main.py
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# The cross-encoder looks at the query and the movie plot together and assigns a highly accurate score.
```
This ensures the final 5 movies we send to the LLM are the absolute best matches possible, dramatically reducing hallucinations.

---

## Interview Study Guide

*   **Q: Why didn't you just use SQL for searching movies?**
    *   **A:** Because users search with natural language like "I want a dark, mind-bending movie." SQL can't understand concepts or synonyms. ChromaDB converts that intent into math to find semantic matches.
*   **Q: What happens if a user searches for a specific actor name? Doesn't vector search fail at that?**
    *   **A:** Yes, vector search struggles with exact nouns. That's why I implemented Hybrid Search in `main.py` using BM25. The BM25 index catches the exact keyword matches, while ChromaDB handles the semantic meaning.
*   **Q: How did you ensure high retrieval accuracy?**
    *   **A:** Two-stage retrieval. I used ChromaDB for fast, broad retrieval (top 100), and then implemented a Cross-Encoder (`ms-marco`) to accurately re-rank those candidates before feeding them to the LLM.
