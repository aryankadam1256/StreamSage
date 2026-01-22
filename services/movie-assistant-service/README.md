# Movie Discovery Assistant - Setup Guide

## 🚀 Prerequisites

Before running the service, you need:

### 1. HuggingFace Account & API Token (Free!)
1. Go to https://huggingface.co and sign up (free)
2. Go to Settings → Access Tokens: https://huggingface.co/settings/tokens
3. Click "New token"
4. Name it (e.g., "StreamSage")
5. Select "Read" access (sufficient for inference)

### 3. ChromaDB Data
Run the Colab notebook (`03_movie_assistant_data.ipynb`) and:
1. Download `chroma_db.zip`
2. Extract it to: `d:/PROJECTS/StreamSage/data/chroma_db`

---

## 📦 Installation Steps

### 1. Navigate to Service Directory
```bash
cd d:/PROJECTS/StreamSage/services/movie-assistant-service
```

### 2. Create .env File
```bash
cd d:/PROJECTS/StreamSage/services/movie-assistant-service

# Create .env file
echo HF_API_TOKEN=your_token_here > .env

# Or copy the example and edit
cp .env.example .env
# Then edit .env and add your token
```

Your `.env` should look like:
```
HF_API_TOKEN=hf_xxxxxxxxxxxxx
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Takes ~2-3 minutes.

---

## 🎬 Running the Service

### 1. Verify Ollama is Running
```bash
ollama list
```
You should see `mistral:7b` in the list.

### 2. Start the Service
```bash
python main.py
```

You should see:
```
🚀 Starting Movie Discovery Assistant...
📊 Loading embedding model: all-MiniLM-L6-v2
✅ Embedding model loaded!
🗄️ Loading ChromaDB from: ...
✅ ChromaDB loaded with 4803 movies!
🤖 Connecting to Ollama at: http://localhost:11434
✅ Ollama connected!
✅ Movie Discovery Assistant is READY!
```

### 3. Access the API
- **API Endpoint**: http://localhost:8002
- **Interactive Docs**: http://localhost:8002/docs (Swagger UI)
- **Health Check**: http://localhost:8002/health

---

## 🧪 Testing

### Option 1: Use the Swagger UI
1. Go to http://localhost:8002/docs
2. Click on `POST /discover`
3. Click "Try it out"
4. Enter a query like:
```json
{
  "query": "Recommend a mind-bending sci-fi movie",
  "min_year": 2000,
  "min_rating": 7.0,
  "top_k": 5
}
```
5. Click "Execute"

### Option 2: Use cURL
```bash
curl -X POST "http://localhost:8002/discover" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Recommend a mind-bending sci-fi movie\"}"
```

### Option 3: Use Python Test Script
See `test_service.py` in this directory.

---

## 🐛 Troubleshooting

### Error: "ChromaDB not found"
**Solution**: Extract `chroma_db.zip` to `d:/PROJECTS/StreamSage/data/chroma_db`

### Error: "Ollama not available"
**Solution**: 
1. Check if Ollama is running: `ollama list`
2. Start Ollama service
3. Verify model is pulled: `ollama run mistral:7b`

### Error: "Failed to load embedding model"
**Solution**: 
1. Check internet connection (first download only)
2. Install sentence-transformers: `pip install sentence-transformers`

### Service is slow
**Possible causes**:
- First query is slow (model loading)
- Large top_k value (reduce to 5)
- CPU-only mode (embeddings are slower without GPU)

**Normal speeds**:
- First query: ~5-10 seconds
- Subsequent queries: ~2-4 seconds

---

## 🎓 What's Happening Under the Hood?

When you send a query, the service:

1. **Loads your query** → "mind-bending sci-fi movie"
2. **Converts to embedding** → [0.23, -0.45, 0.78, ...] (384 numbers)
3. **Searches ChromaDB** → Finds movies with similar embeddings
4. **Retrieves top-5 movies** → Inception, Interstellar, The Matrix, etc.
5. **Builds prompt** → System message + Movies + Your query
6. **Sends to Mistral** → LLM generates personalized response
7. **Returns answer** → "I recommend Inception (2010) because..."

Each step takes ~0.5-1 second.

---

## 🔄 Next Steps

1. ✅ Run the service
2. ✅ Test with various queries
3. ✅ Compare Mistral vs Llama3 (we'll switch models later)
4. ✅ Integrate with frontend
