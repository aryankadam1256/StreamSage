# Movie Discovery Assistant - Quick Reference

## 🎯 What Did We Build?

A **RAG-powered movie recommendation service** that replaces the binge predictor with:
- Conversational recommendations
- Semantic understanding (not just keyword matching)
- Natural language answers powered by Mistral LLM

---

## 📁 Files Created

```
services/movie-assistant-service/
├── main.py              # Main FastAPI service (500+ lines, heavily commented)
├── requirements.txt     # Python dependencies
├── README.md           # Setup and troubleshooting guide
└── test_service.py     # Test script with sample queries
```

---

## 🎓 Key Concepts Explained

### 1. RAG (Retrieval-Augmented Generation)
**What**: Combine vector search + LLM generation  
**Why**: LLMs hallucinate facts; RAG grounds them in real data  
**How**: Retrieve relevant movies → Include in prompt → LLM answers based on retrieved context

### 2. Vector Embeddings
**What**: Convert text to numbers (384-dim vectors)  
**Why**: Similar movies have similar vectors  
**Example**: 
- "Inception" → [0.23, -0.45, 0.78, ...]
- "Interstellar" → [0.25, -0.43, 0.81, ...] (close!)
- "Toy Story" → [-0.12, 0.67, -0.34, ...] (far!)

### 3. ChromaDB
**What**: Vector database (stores embeddings + metadata)  
**Why**: Fast semantic search (finds similar vectors in milliseconds)  
**How**: 
1. Convert query to vector
2. Compare to all movie vectors (cosine similarity)
3. Return top-K most similar

### 4. Metadata Filtering
**What**: Filter by year/genre/rating BEFORE semantic search  
**Why**: Makes queries like "sci-fi from 2010-2020" accurate  
**Example**: 
- Without: Search all 4800 movies
- With: Filter to 2010-2020 → Search only those

### 5. Prompt Engineering
**What**: How you structure the LLM prompt  
**Why**: Determines response quality  
**Our Prompt**:
```
System: You are a helpful movie assistant
Context: [Retrieved movies]
User Query: Recommend a sci-fi movie
```

---

## 🚀 How to Use

### Setup (One-time)
```bash
# 1. Install Ollama
# Download from: https://ollama.com

# 2. Pull Mistral model
ollama pull mistral:7b

# 3. Extract ChromaDB
# Extract chroma_db.zip to: d:/PROJECTS/StreamSage/data/chroma_db

# 4. Install Python dependencies
cd services/movie-assistant-service
pip install -r requirements.txt
```

### Run the Service
```bash
python main.py
```

### Test
```bash
# Option 1: Interactive test script
python test_service.py

# Option 2: Swagger UI
# Visit: http://localhost:8002/docs

# Option 3: cURL
curl -X POST "http://localhost:8002/discover" \
  -H "Content-Type: application/json" \
  -d '{"query": "Recommend a mind-bending sci-fi movie"}'
```

---

## 🎮 Example Queries

### Basic Recommendation
```json
{
  "query": "Recommend a mind-bending sci-fi movie"
}
```
→ Returns: Inception, Interstellar, The Matrix, etc.

### With Genre Filter
```json
{
  "query": "I want an action thriller",
  "genre": "Action",
  "min_year": 2000,
  "max_year": 2015
}
```
→ Only searches Action movies from 2000-2015

### With Rating Filter
```json
{
  "query": "Christopher Nolan films",
  "min_rating": 8.0
}
```
→ Only returns highly-rated Nolan movies

### Similarity Search
```json
{
  "query": "Movies like The Matrix"
}
```
→ Finds movies with similar plot/themes/style

---

## 🔄 What Happens on Each Request?

```
User Query: "mind-bending sci-fi"
    ↓
[1] Convert to embedding (0.5s)
    → [0.23, -0.45, 0.78, ...]
    ↓
[2] Search ChromaDB (0.3s)
    → Top-5: Inception, Interstellar, etc.
    ↓
[3] Build prompt (0.1s)
    → System + Movies + Query
    ↓
[4] Call Mistral LLM (2-3s)
    → Generate personalized answer
    ↓
[5] Return response (0.1s)
    → JSON with answer + movie list
    
Total: ~3-5 seconds
```

---

## 🎯 Next Steps

1. ✅ Run Colab notebook → Get `chroma_db.zip`
2. ✅ Install Ollama + Mistral
3. ✅ Run `python main.py`
4. ✅ Test with `test_service.py`
5. 🔄 Switch to Llama3 and compare
6. 🔄 Integrate with frontend

---

## 🐛 Common Issues

**"ChromaDB not found"**
→ Extract `chroma_db.zip` to `data/chroma_db`

**"Ollama not available"**
→ Run `ollama list` to check if Ollama is running

**"Service is slow"**
→ First query is slow (model loading). Subsequent queries are fast.

**"Import error"**
→ Install dependencies: `pip install -r requirements.txt`

---

## 📊 Performance Tips

- **Faster queries**: Reduce `top_k` from 5 to 3
- **Better answers**: Increase `top_k` to 8-10
- **Use filters**: Genre/year filters make search more accurate
- **GPU**: If you have a GPU, embeddings will be faster

---

## 🎓 Learning Path

If you want to understand the code deeply:

1. **Start with** `main.py` → Read the docstrings
2. **Understand** RAG architecture diagram (top of file)
3. **Trace a request** from `/discover` endpoint through the code
4. **Experiment** with different prompts in the code
5. **Compare** responses with/without LLM

---

## 🔀 Mistral vs Llama3 (Coming Next)

We'll test both models and compare:
- **Response quality**: Which gives better recommendations?
- **Speed**: Which is faster?
- **Creativity**: Which is more conversational?
- **Accuracy**: Which follows context better?

To switch models, just change `LLM_MODEL` in the code!
