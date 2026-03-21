# Day 1: Project Overview & Architecture

**Time:** 2-3 hours
**Prerequisites:** None (this is the starting point!)

---

## Learning Objectives

By the end of Day 1, you will understand:
- The overall StreamSage architecture
- Why microservices pattern was chosen
- What each service does
- How the API Gateway routes requests
- The flow of a complete request through the system

---

## Step 1: Read the Overview (15 min)

Open `../../README.md` (the root README) and read sections:
- Project Overview
- Features
- Project Structure

**As you read, note:**
- How many services exist? (5)
- What does each one do?
- What port does each run on?

---

## Step 2: Run the Project (10 min)

```bash
cd c:/Users/mailp/StreamSage
SETUP_AND_RUN.bat
```

**Test each component:**
1. Frontend: http://localhost:5177
2. Gateway: http://localhost:8000/docs
3. Movie Assistant: http://localhost:8004/docs

**Try searching:**
- "movies like Inception"
- "Christopher Nolan films"
- "relaxing comedy movies"

---

## Step 3: Understand the Architecture (20 min)

Read `../../ARCHITECTURE.md`

**Key sections:**
- System Overview (the big diagram)
- Service Architecture Details
- Data Flow Patterns

---

## Checkpoint Questions

### 1. What is StreamSage? (2-3 sentences)

**Model Answer:**
> StreamSage is a microservices-based movie recommendation platform demonstrating modern AI/ML techniques. It uses a fine-tuned Llama 3 8B model with RAG for movie discovery, an Oracle chatbot for movie dialogue Q&A, LSTM for binge prediction, and DistilBERT for sentiment analysis. All services communicate through an API Gateway presented via a React frontend.

---

### 2. List all 5 services with their ports

| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8000 | Routes requests, health aggregation |
| Movie Assistant | 8004 | Movie discovery with fine-tuned Llama 3 |
| Oracle RAG | 8001 | Subtitle Q&A using Ollama |
| Binge Predictor | 8002 | LSTM for watch behavior |
| Sentiment Engine | 8003 | DistilBERT sentiment classification |

---

### 3. What happens when you search for "movies like Inception"?

**Step-by-step:**
1. Frontend (React) → User types, clicks "Discover"
2. axios.post('/api/v1/discover') → Gateway :8000
3. Gateway → Proxies to Movie Assistant :8004
4. Movie Assistant:
   - Embeds query → [0.23, -0.45, ...] (1024 dims)
   - ChromaDB vector search (semantic)
   - BM25 keyword search
   - Reciprocal Rank Fusion merges results
   - Cross-encoder re-ranks
   - Generates per-movie recommendations
   - Sorts by relevance (not rating)
5. Response JSON → Gateway → Frontend
6. MovieDiscover.jsx renders movie cards with "AI's Take"

---

### 4. Why use a Gateway instead of calling services directly?

**Model Answer:**
> The Gateway provides a single entry point for the frontend, simplifying CORS configuration. It routes requests to appropriate backend services, aggregates health checks across all services, and could easily add authentication, rate limiting, or request logging in the future. This implements the API Gateway pattern from microservices architecture.

---

### 5. Why microservices instead of a monolith?

**Model Answer:**
> For this project, microservices provide: (1) Educational isolation - each ML technique is in its own service, (2) Independent scaling - the Movie Assistant can scale separately from Sentiment Engine, (3) Technology flexibility - LSTM runs on TensorFlow, sentiment uses PyTorch, (4) Fault isolation - if Oracle crashes, movie discovery still works.

---

[← Back to Course](../README.md) | [Continue to Day 2 →](../day-2/README.md)
