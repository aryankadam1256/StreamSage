# StreamSage Learning Roadmap

A structured guide to understand every concept in this project from scratch to mastery.

---

## How to Use This Guide

1. **Follow the order** - Each module builds on the previous
2. **Read the code** - Don't just skim; actually trace through the files
3. **Run experiments** - Modify code, break things, understand why they break
4. **Take notes** - Write down concepts in your own words

**Estimated Time:** 2-3 weeks (2-3 hours/day)

---

## Prerequisites Checklist

Before starting, ensure you understand these basics:

### Python (Required)
- [ ] Functions, classes, decorators
- [ ] List comprehensions, generators
- [ ] `async/await` basics
- [ ] Working with JSON and APIs

### Machine Learning (Helpful but not required)
- [ ] What is a neural network (high-level)
- [ ] What is training vs inference
- [ ] What are embeddings (we'll teach this in depth)

### Tools
- [ ] Git basics (clone, commit, push)
- [ ] Command line navigation
- [ ] VS Code or similar editor

---

## Learning Path Overview

```
Week 1: Foundation
├── Module 1: Project Setup & Architecture
├── Module 2: API Gateway & Microservices
└── Module 3: Vector Databases & Embeddings

Week 2: Core ML Concepts
├── Module 4: RAG Pipeline (Oracle Service)
├── Module 5: Sentiment Analysis (BERT/Transformers)
└── Module 6: Sequence Models (LSTM/Binge Predictor)

Week 3: Advanced Topics
├── Module 7: Hybrid Retrieval & Reranking (Movie Assistant)
├── Module 8: LLM Fine-tuning (QLoRA, DPO)
└── Module 9: Full Stack Integration
```

---

# WEEK 1: FOUNDATION

## Module 1: Project Setup & Architecture (Day 1)

### Goal
Understand the overall structure and run the project locally.

### Steps

**1.1 Read the Project Overview**
```
Files to read:
├── README.md                    # 15 min - Project overview
├── QUICKSTART.md                # 10 min - How to run it
└── docs/ARCHITECTURE.md         # 20 min - System design
```

**1.2 Run the Project**
```bash
# Start everything
SETUP_AND_RUN.bat

# Or manually:
# Terminal 1: cd services/movie-assistant-service && python main.py
# Terminal 2: cd gateway && python main.py
# Terminal 3: cd frontend && npm run dev
```

**1.3 Test Each Endpoint**
- Open http://localhost:3000 - Click around the UI
- Open http://localhost:8000/docs - Try the API in Swagger
- Open http://localhost:8004/docs - Test movie discovery directly

**1.4 Understand the Architecture**
Draw this diagram yourself on paper:
```
Frontend (React) → Gateway (FastAPI) → Services (ML)
                                        ├── Movie Assistant
                                        ├── Oracle RAG
                                        ├── Binge LSTM
                                        └── Sentiment BERT
```

### Checkpoint Questions
- [ ] Why do we have a gateway instead of calling services directly?
- [ ] What port does each service run on?
- [ ] What happens if the Movie Assistant service is down?

---

## Module 2: API Gateway Pattern (Day 2)

### Goal
Understand how the gateway routes requests and why it exists.

### Steps

**2.1 Read the Gateway Code**
```
File: gateway/main.py

Focus on:
├── Line ~20-50: FastAPI app setup, CORS config
├── Line ~60-100: Service endpoint definitions
├── Line ~100-150: Proxy logic (how requests are forwarded)
└── Line ~150+: Health check aggregation
```

**2.2 Key Concepts to Understand**

| Concept | What It Means |
|---------|---------------|
| CORS | Cross-Origin Resource Sharing - why browsers block direct service calls |
| Proxy | Gateway receives request, forwards to backend, returns response |
| Health Aggregation | Gateway pings all services and reports combined status |
| Rate Limiting | (Not implemented here) - controlling request frequency |

**2.3 Experiment**
```bash
# Stop the movie-assistant service
# Then hit: curl http://localhost:8000/health
# What do you see? The gateway should report it as unhealthy
```

**2.4 Read About Microservices**
- Google: "API Gateway pattern microservices"
- Google: "Why use microservices vs monolith"

### Checkpoint Questions
- [ ] What is the single responsibility of the gateway?
- [ ] How does async/await help with proxying requests?
- [ ] What would you add to this gateway for production?

---

## Module 3: Vector Databases & Embeddings (Day 3-4)

### Goal
Understand how text becomes numbers (embeddings) and how we search them.

### This is the MOST IMPORTANT module - embeddings are the foundation of modern AI

### Steps

**3.1 What Are Embeddings?**

Embeddings convert text to numbers (vectors) that capture meaning:
```
"happy movie" → [0.8, 0.2, 0.1, ...]   (768 or 1024 numbers)
"joyful film" → [0.79, 0.21, 0.11, ...]  (similar numbers!)
"sad movie"   → [-0.3, 0.5, 0.8, ...]    (very different)
```

Similar meanings = similar numbers = we can find related items!

**3.2 Hands-On: Play with Embeddings**

Create a file `learn_embeddings.py`:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
texts = [
    "A thrilling action movie with car chases",
    "An exciting film with vehicle pursuits",
    "A romantic comedy about finding love",
]

embeddings = model.encode(texts)

# Calculate similarity (cosine similarity)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Action vs Exciting:", cosine_sim(embeddings[0], embeddings[1]))  # High!
print("Action vs Romance:", cosine_sim(embeddings[0], embeddings[2]))   # Low!
```

**3.3 Understand ChromaDB**

ChromaDB stores embeddings and lets us search them efficiently.

```
Read: services/movie-assistant-service/data_collection/create_vector_db.py

Key operations:
1. Load movie data (JSON)
2. Create text description for each movie
3. Generate embedding using BGE model
4. Store in ChromaDB with metadata
```

**3.4 Why BGE-large vs MiniLM?**

| Model | Dimensions | Quality | Speed |
|-------|------------|---------|-------|
| all-MiniLM-L6-v2 | 384 | Good | Fast |
| BAAI/bge-large-en-v1.5 | 1024 | Excellent | Slower |

Movie Assistant uses BGE-large for better semantic understanding.

### Checkpoint Questions
- [ ] What is the cosine similarity between two identical texts?
- [ ] Why do we store metadata (genres, year) alongside embeddings?
- [ ] What happens if we embed a query and search ChromaDB?

---

# WEEK 2: CORE ML CONCEPTS

## Module 4: RAG Pipeline - Oracle Service (Day 5-6)

### Goal
Understand Retrieval-Augmented Generation (RAG) - the most important pattern in modern AI.

### What is RAG?

```
User Question
     │
     ▼
┌─────────────┐    "What did Neo say about the Matrix?"
│   Embed     │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────┐    Search for similar text chunks
│  Retrieve   │    → Finds: "Neo: The Matrix is everywhere..."
│  Context    │
└──────┬──────┘
       │
       ▼
┌─────────────┐    Prompt: "Context: {retrieved text}\nQuestion: {user question}"
│   Generate  │    → LLM generates answer using the context
│   Answer    │
└─────────────┘
```

**Why RAG?** LLMs don't know your specific data. RAG lets them answer using YOUR documents.

### Steps

**4.1 Read the Oracle Service Code**
```
File: services/oracle-rag-service/main.py

Trace this flow:
1. /ask endpoint receives question + movie_id
2. Question is embedded using sentence-transformers
3. ChromaDB searched for relevant subtitle chunks
4. Chunks filtered by timestamp (spoiler protection!)
5. Prompt constructed with context + question
6. Ollama (Llama 3) generates answer
7. Response streamed back via SSE
```

**4.2 Understand Chunking**
```
File: services/oracle-rag-service/ingest.py

Subtitles are split into chunks:
- Each chunk = ~3-5 dialogue lines
- Overlap between chunks (don't lose context)
- Timestamp stored for spoiler filtering
```

**4.3 Key RAG Concepts**

| Concept | Explanation |
|---------|-------------|
| Chunking | Breaking documents into searchable pieces |
| Top-K Retrieval | Return K most similar chunks |
| Context Window | How much text the LLM can see at once |
| Prompt Engineering | Crafting the prompt for best results |

**4.4 Experiment**
```bash
# Start Oracle service
cd services/oracle-rag-service
python main.py

# Test with curl
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"movie_id": "inception", "question": "What is the deepest dream level?"}'
```

### Checkpoint Questions
- [ ] Why do we chunk documents instead of embedding entire files?
- [ ] What is the trade-off between small and large chunks?
- [ ] How does spoiler protection work technically?

---

## Module 5: Sentiment Analysis with Transformers (Day 7-8)

### Goal
Understand how BERT/DistilBERT classifies text sentiment.

### Steps

**5.1 What is BERT?**

BERT (Bidirectional Encoder Representations from Transformers):
- Reads text in BOTH directions (not just left-to-right)
- Understands context: "not bad" = positive, "bad" = negative
- Pre-trained on massive text corpus, fine-tuned for specific tasks

**5.2 Read the Sentiment Service**
```
File: services/sentiment-service/main.py

Key components:
1. Load DistilBERT model (smaller, faster BERT)
2. Tokenize input text (convert words to numbers)
3. Run through transformer layers
4. Classification head outputs: positive/negative + confidence
```

**5.3 Understand Tokenization**
```python
# What happens inside:
text = "This movie is great!"
tokens = ["[CLS]", "this", "movie", "is", "great", "!", "[SEP]"]
token_ids = [101, 2023, 3185, 2003, 2307, 999, 102]
# These numbers go into the model
```

**5.4 Why DistilBERT?**
- BERT: 110M parameters, slower
- DistilBERT: 66M parameters (40% smaller), 60% faster, 97% accuracy

**5.5 Read the Training Notebook**
```
File: notebooks/train_bert.ipynb

Understand:
1. How training data is loaded (reviews + labels)
2. How the model is fine-tuned on movie reviews
3. How to evaluate accuracy
```

### Checkpoint Questions
- [ ] What does "bidirectional" mean in BERT?
- [ ] Why is [CLS] token special?
- [ ] How does fine-tuning differ from training from scratch?

---

## Module 6: Sequence Models - LSTM (Day 9-10)

### Goal
Understand how LSTMs predict sequences (binge watching behavior).

### Steps

**6.1 What is an LSTM?**

LSTM (Long Short-Term Memory):
- Processes sequences one step at a time
- Remembers important information from earlier steps
- Forgets irrelevant information
- Perfect for: time series, viewing history, text sequences

```
Watch History: [Action, Action, Comedy, Action, ???]
                  ↓       ↓       ↓       ↓
               [LSTM] → [LSTM] → [LSTM] → [LSTM] → Prediction
                          ↑ memory flows through ↑
```

**6.2 Read the Binge Service**
```
File: services/binge-service/main.py

Key components:
1. Feature engineering: convert watch history to numbers
2. Load Keras LSTM model
3. Predict continuation probability
4. Return risk level + recommendations
```

**6.3 Understand Features**

The model uses these features:
```
- Genre sequence (encoded as numbers)
- Rating trend (increasing? decreasing?)
- Watch velocity (episodes per day)
- Time of day patterns
- Session duration
```

**6.4 Read the Training Notebook**
```
File: notebooks/train_rnn.ipynb

Understand:
1. How synthetic data is generated
2. LSTM architecture (layers, units)
3. Training loop and loss function
4. Saving model as .h5 file
```

**6.5 LSTM vs Simple Neural Network**

| Feature | Simple NN | LSTM |
|---------|-----------|------|
| Input | Fixed size | Variable sequence |
| Memory | None | Remembers past |
| Order | Doesn't matter | Order matters |
| Use case | Classification | Time series |

### Checkpoint Questions
- [ ] Why can't a simple neural network handle watch history?
- [ ] What is the "forget gate" in LSTM?
- [ ] How does return_sequences=True change the output?

---

# WEEK 3: ADVANCED TOPICS

## Module 7: Hybrid Retrieval & Reranking (Day 11-13)

### Goal
Master the Movie Assistant's advanced retrieval pipeline.

### This is the MOST COMPLEX service - take your time!

### Steps

**7.1 Overview of the Pipeline**
```
Query → Expand → [Vector Search] + [BM25] → Fuse → Rerank → Filter → Output
                      ↓              ↓
                  Semantic       Keyword
                  Matching       Matching
```

**7.2 Read main.py Section by Section**
```
File: services/movie-assistant-service/main.py (~4000 lines)

Read in this order:
1. Lines 1-200: Imports, model loading, initialization
2. Lines 200-400: Query expansion (mood detection, name extraction)
3. Lines 400-600: ChromaDB vector search
4. Lines 600-800: BM25 keyword search
5. Lines 800-1000: Reciprocal Rank Fusion (RRF)
6. Lines 1000-1200: Cross-encoder reranking
7. Lines 1200-1600: Custom reranking rules
8. Lines 1600+: API endpoints
```

**7.3 Key Concepts**

| Concept | What It Does | Why It Matters |
|---------|-------------|----------------|
| Query Expansion | "like Inception" → adds "mind-bending, Nolan" | Improves recall |
| Hybrid Search | Vector + BM25 together | Catches both semantic and keyword matches |
| RRF (Reciprocal Rank Fusion) | Merges two ranked lists | Balanced combination |
| Cross-Encoder | Scores (query, doc) pairs | More accurate than vector distance |
| Custom Rules | Genre bonus, franchise dedup | Domain-specific improvements |

**7.4 Why Hybrid > Vector Only?**

```
Query: "Quentin Tarantino movies"

Vector Search: Returns action/crime films (semantically similar)
BM25 Search: Returns films with "Tarantino" in metadata (exact match)
Hybrid: Returns actual Tarantino films (best of both)
```

**7.5 Understand Reranking**
```python
# Cross-encoder reranking pseudocode
for movie in candidates:
    score = cross_encoder.predict([query, movie.description])
    movie.relevance = 0.3 * score + 0.7 * movie.vector_score

# Custom reranking
if query mentions genre and movie matches:
    movie.relevance *= 1.2  # Boost
if same franchise appears twice:
    remove duplicates  # Dedup
```

**7.6 Run the Benchmark**
```bash
# The project has a 100-query benchmark
python test_100.py

# Study why some queries score "Excellent" and others "Weak"
```

### Checkpoint Questions
- [ ] Why is cross-encoder more accurate but slower?
- [ ] What is the problem with showing 5 MCU films for "superhero movies"?
- [ ] How would you improve queries that currently score "Weak"?

---

## Module 8: LLM Fine-tuning (Day 14-16)

### Goal
Understand how Llama 3 was fine-tuned for movie recommendations.

### Steps

**8.1 What is Fine-tuning?**
```
Base Model (Llama 3 8B)
    ↓
    │ Knows general language
    │ Doesn't know movie recommendations
    ↓
Fine-tuned Model
    │ Knows general language
    │ ALSO knows movie recommendations!
```

**8.2 Why QLoRA?**

Training 8B parameters requires ~32GB GPU memory. We don't have that!

**QLoRA solution:**
1. **Quantization (Q)**: Compress model to 4-bit (8GB instead of 32GB)
2. **LoRA**: Only train small adapter matrices (~1% of parameters)

```
Full Fine-tuning: Train all 8B parameters → Needs 32GB VRAM
QLoRA: Train 80M adapter parameters → Needs 8GB VRAM ✓
```

**8.3 Read the Fine-tuning Notebook**
```
File: services/movie-assistant-service/fine_tuning/llama_finetune_colab.ipynb

Key sections:
1. Load base Llama 3 with 4-bit quantization
2. Add LoRA adapters to attention layers
3. Load training data (2,946 Q&A pairs)
4. Train with Hugging Face Trainer
5. Save adapter weights
```

**8.4 Understanding the Training Data**
```
File: services/movie-assistant-service/dataset_generation/dataset_builder.py

Each training example:
{
  "instruction": "Recommend movies for someone who likes action and sci-fi",
  "response": "Here are some great picks: 1. The Matrix (1999)..."
}
```

**8.5 What is DPO/GRPO?**

After SFT (supervised fine-tuning), we further align the model:

| Method | What It Does |
|--------|-------------|
| DPO (Direct Preference Optimization) | Learn from "good vs bad" response pairs |
| GRPO (Group Relative Policy Optimization) | Learn from ranked response groups |

```
DPO Training Example:
Query: "Recommend a thriller"
Good Response: "I recommend Se7en (1995)..." ✓ (train toward this)
Bad Response: "Here's a comedy: Home Alone..." ✗ (train away from this)
```

**8.6 Read the DPO Notebook**
```
File: services/movie-assistant-service/fine_tuning/dpo_alignment_colab.ipynb

Understand:
1. How preference pairs are created
2. DPO loss function (no reward model needed)
3. How it improves response quality
```

### Checkpoint Questions
- [ ] Why is LoRA rank 16 instead of 64?
- [ ] What happens if training data has mistakes?
- [ ] How do you evaluate if fine-tuning helped?

---

## Module 9: Full Stack Integration (Day 17-18)

### Goal
Understand how frontend, gateway, and services work together.

### Steps

**9.1 Trace a Complete Request**

```
User types: "movies like Inception"
     │
     ▼
[Frontend: SearchHero.jsx]
     │ axios.post('/api/v1/discover', {query: "..."})
     ▼
[Gateway: main.py]
     │ Forwards to localhost:8004/discover
     ▼
[Movie Assistant: main.py]
     │ Query → Embed → Search → Rerank → (Optional LLM)
     ▼
[Response flows back]
     │
     ▼
[Frontend: MovieGrid.jsx renders results]
```

**9.2 Read the Frontend Code**
```
Files to read:
├── frontend/src/App.jsx           # Routing setup
├── frontend/src/api.js            # API client
├── frontend/src/pages/
│   ├── LandingPage.jsx            # Entry point
│   └── HomePage.jsx               # Main app
└── frontend/src/components/
    ├── SearchHero.jsx             # Search input
    ├── MovieGrid.jsx              # Results display
    └── MovieCard.jsx              # Individual movie
```

**9.3 Understand State Flow**
```
SearchHero (handles input)
    │ User types query
    │ Calls API on submit
    ▼
HomePage (manages state)
    │ Stores results in state
    │ Passes to children
    ▼
MovieGrid (displays results)
    │ Maps over movies
    │ Renders MovieCard for each
```

**9.4 SSE Streaming (Oracle)**
```javascript
// How streaming works in OracleChat.jsx
const eventSource = new EventSource('/api/v1/ask/stream');
eventSource.onmessage = (event) => {
    // Each event = one token from LLM
    setResponse(prev => prev + event.data);
};
```

**9.5 Docker Integration**
```
File: docker-compose.yml

Understand:
- How services are networked together
- Volume mounts for data persistence
- Health check configuration
- Port mappings
```

### Checkpoint Questions
- [ ] Why does the frontend call the gateway, not services directly?
- [ ] How does SSE differ from WebSockets?
- [ ] What happens when one service is down?

---

## Module 10: Review & Mastery (Day 19-21)

### Goal
Consolidate learning and fill any gaps.

### Final Exercises

**Exercise 1: Modify the Retrieval Pipeline**
```
Task: Add a "year filter" to movie discovery
- User can specify: { "query": "...", "min_year": 2010, "max_year": 2020 }
- Filter results in the reranking stage
```

**Exercise 2: Add a New Endpoint**
```
Task: Create /similar/{movie_id} endpoint
- Given a movie ID, return 5 similar movies
- Use the existing embedding to find nearest neighbors
```

**Exercise 3: Improve Sentiment**
```
Task: Add "neutral" as a third sentiment class
- Modify the model to output 3 classes
- Update the training data
- Retrain in Colab
```

**Exercise 4: Explain to Someone Else**
```
Task: Record yourself explaining:
1. How RAG works (5 min)
2. Why hybrid retrieval beats vector-only (5 min)
3. What QLoRA does (5 min)

If you can explain it clearly, you understand it!
```

---

## Quick Reference: What to Explain in Interviews

### "Tell me about this project"
> StreamSage is a microservices-based movie recommendation system that demonstrates RAG, fine-tuned LLMs, transformers, and LSTMs. The flagship feature uses hybrid retrieval (vector + BM25), cross-encoder reranking, and a QLoRA fine-tuned Llama 3 model.

### "What was the hardest part?"
> Getting retrieval quality right. My initial pipeline scored 40% accuracy. Through iterative improvements—hybrid search, cross-encoder reranking, custom rules—I reached 70% on a 100-query benchmark.

### "How does RAG work?"
> RAG retrieves relevant documents using vector similarity, then passes them as context to an LLM. The LLM generates answers grounded in the retrieved content, which prevents hallucination and enables answering questions about specific data.

### "Why QLoRA instead of full fine-tuning?"
> Full fine-tuning of an 8B model requires 32GB+ VRAM. QLoRA combines 4-bit quantization with Low-Rank Adapters, reducing memory to 8GB while achieving similar quality. I trained on a free Colab GPU.

---

## Resources for Deeper Learning

### Embeddings & Vector Search
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Getting Started](https://docs.trychroma.com/)

### RAG
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)

### Transformers & BERT
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Fine-tuning LLMs
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face PEFT Library](https://huggingface.co/docs/peft)

### System Design
- [Microservices Patterns (Sam Newman)](https://www.oreilly.com/library/view/building-microservices/9781491950340/)

---

Good luck! Take it one module at a time. Understanding comes from DOING, not just reading.
