# StreamSage Interview Guide

A comprehensive guide for discussing this project in technical interviews. Organized by topic with potential questions and strong answers.

---

## Table of Contents
1. [Project Overview Questions](#project-overview-questions)
2. [Machine Learning Questions](#machine-learning-questions)
3. [System Design Questions](#system-design-questions)
4. [RAG & LLM Questions](#rag--llm-questions)
5. [Data Engineering Questions](#data-engineering-questions)
6. [DevOps Questions](#devops-questions)
7. [Frontend Questions](#frontend-questions)
8. [Challenges & Trade-offs](#challenges--trade-offs)

---

## Project Overview Questions

### "Tell me about this project."

**Strong Answer:**
> "StreamSage is a microservices-based movie recommendation platform I built to demonstrate several AI/ML techniques. The flagship feature is a movie discovery assistant that uses a fine-tuned Llama 3 model combined with a retrieval-augmented generation pipeline.
>
> The system has five services: a movie assistant with hybrid vector/BM25 search, an Oracle service that answers questions about movie dialogues using subtitles, a binge predictor using LSTM for viewing behavior, a sentiment analyzer using DistilBERT, and an API gateway that orchestrates everything.
>
> I chose a microservices architecture because each AI technique can be studied independently, and it mirrors how production ML systems are built. The project went through five development phases over about two weeks of active development."

### "Why did you choose this specific project?"

**Strong Answer:**
> "I wanted a project that would let me implement multiple ML paradigms in a realistic context. Movies are a great domain because:
> 1. There's rich publicly available data through TMDB
> 2. The queries are varied—some semantic, some keyword-based—which tests retrieval systems well
> 3. It's easy to evaluate results qualitatively
>
> More importantly, I wanted to show I could take a model from training to deployment, not just run notebooks. This project includes the full pipeline: data collection, feature engineering, model training, API design, and frontend integration."

---

## Machine Learning Questions

### "Explain your fine-tuning approach for Llama 3."

**Strong Answer:**
> "I used QLoRA, which combines quantization with Low-Rank Adaptation. Here's why each part matters:
>
> **4-bit Quantization:** Reduces the model's memory footprint from ~32GB to ~8GB, making it trainable on a single GPU like an A100 in Colab. The quality loss is minimal because we're only using the base model for feature extraction, and the LoRA adapters do the heavy lifting.
>
> **Low-Rank Adapters:** Instead of updating all 8 billion parameters, LoRA adds small trainable matrices (rank 8-16) to the attention layers. This means we only train ~1% of the parameters while achieving similar performance to full fine-tuning.
>
> I trained on 2,946 Q&A pairs that I generated using a combination of template-based generation and manual curation. The training took about 3 hours on Colab."

### "What's the difference between DPO and RLHF?"

**Strong Answer:**
> "Both align model outputs with human preferences, but DPO is simpler:
>
> **RLHF (Reinforcement Learning from Human Feedback):**
> - Requires training a separate reward model
> - Uses PPO to optimize the language model against that reward
> - Complex pipeline with multiple models and training phases
>
> **DPO (Direct Preference Optimization):**
> - Directly optimizes the language model using preference pairs
> - No reward model needed—the preference signal comes from chosen/rejected response pairs
> - Mathematically equivalent to RLHF but in a closed-form solution
> - Much simpler to implement and train
>
> I used DPO because it's more stable to train and requires less infrastructure. I created preference pairs by generating multiple responses and ranking them by quality."

### "Why did you choose DistilBERT over BERT for sentiment analysis?"

**Strong Answer:**
> "DistilBERT is a distilled version of BERT that's 40% smaller and 60% faster while retaining 97% of BERT's performance on most tasks. For sentiment classification, which is a relatively simple task compared to question answering, this trade-off is excellent.
>
> In production, inference latency and memory matter. DistilBERT runs comfortably on CPU with sub-100ms latency, while BERT would need GPU acceleration for similar performance. Since movie reviews don't require the full depth of BERT's reasoning, DistilBERT is the practical choice."

### "Explain your LSTM architecture for binge prediction."

**Strong Answer:**
> "The task is sequence classification: given a user's watch history, predict if they'll continue watching. LSTMs are well-suited because:
>
> 1. **Temporal patterns matter:** A user who watched 5 episodes rapidly is different from one who watched 5 episodes over a month
> 2. **Variable length sequences:** Users have different history lengths; LSTMs handle this naturally
> 3. **Long-term dependencies:** Genre preferences established weeks ago still influence today's predictions
>
> The architecture is: two stacked LSTM layers (64 and 32 units) with dropout for regularization, followed by dense layers for classification. I use return_sequences=True in the first LSTM so the second layer sees the full hidden state sequence."

---

## System Design Questions

### "Why microservices instead of a monolith?"

**Strong Answer:**
> "For this project, microservices provide several benefits:
>
> 1. **Educational isolation:** Each ML technique is in its own service, making it easier to understand and modify
> 2. **Independent scaling:** The movie assistant handles 90% of requests; I can scale it without touching sentiment
> 3. **Technology flexibility:** The LSTM runs on TensorFlow, sentiment uses PyTorch—no conflicts
> 4. **Fault isolation:** If the Oracle service crashes, movie discovery still works
>
> The trade-off is operational complexity, but Docker Compose handles that for development. In production, I'd move to Kubernetes."

### "Walk me through a request from the frontend to Movie Assistant."

**Strong Answer:**
> "Happy to. Let's trace a query like 'movies similar to Inception':
>
> 1. **Frontend:** React component sends POST to `/api/v1/discover` via axios
> 2. **Gateway:** FastAPI receives it, validates the schema, proxies to `:8004/discover`
> 3. **Query Processing:** The service expands 'similar to Inception' to include concepts like 'mind-bending', 'Christopher Nolan', 'thriller'
> 4. **Parallel Retrieval:**
>    - ChromaDB vector search using BGE embeddings (semantic)
>    - BM25 index search (keyword matching)
> 5. **Fusion:** Reciprocal Rank Fusion merges the two result lists
> 6. **Cross-Encoder:** Re-ranks top 50 results using ms-marco-MiniLM for pairwise relevance
> 7. **Custom Reranking:** Applies genre matching, deduplication, mood alignment
> 8. **Optional LLM:** If enabled, generates a narrative explanation
> 9. **Response:** JSON with recommended movies, sorted by relevance and rating
> 10. **Frontend:** MovieGrid component renders the results"

### "Why did you use an API Gateway?"

**Strong Answer:**
> "The gateway serves several purposes:
>
> 1. **Single entry point:** Frontend only needs to know one URL, not four services
> 2. **CORS handling:** Configured once in the gateway
> 3. **Health aggregation:** One endpoint shows all service statuses
> 4. **SSE proxying:** The gateway passes through streaming responses from Oracle
> 5. **Future extensibility:** I could add authentication, rate limiting, or request logging here
>
> It implements the API Gateway pattern from microservices architecture. In production, you'd use something like Kong or AWS API Gateway, but for learning, a FastAPI implementation shows the concepts clearly."

---

## RAG & LLM Questions

### "Why do you use hybrid retrieval instead of just vector search?"

**Strong Answer:**
> "Vector search excels at semantic similarity but misses exact keyword matches. Consider:
>
> - **Query:** 'Quentin Tarantino movies'
> - **Vector search:** Might return violent crime dramas that are semantically similar
> - **BM25:** Returns exactly films directed by Tarantino because it matches the name
>
> Conversely:
> - **Query:** 'movies like John Wick but with a female lead'
> - **BM25:** Struggles because there's no keyword overlap with, say, Atomic Blonde
> - **Vector search:** Understands the semantic intent and retrieves action films with female protagonists
>
> By combining them with Reciprocal Rank Fusion, I get the best of both worlds. My benchmark showed a 15% improvement in retrieval quality with hybrid versus vector-only."

### "Explain your two-stage ranking pipeline."

**Strong Answer:**
> "Stage one is fast and broad: retrieve 100 candidates using vector search and BM25. This takes ~50ms.
>
> Stage two is slow but precise: cross-encoder re-ranks the top candidates by computing a relevance score for each (query, document) pair. This is computationally expensive—each pair requires a forward pass through a transformer. For 100 candidates, that's 100 forward passes.
>
> The two-stage approach is a common pattern in production search systems. Fast retrieval gets candidates, slow re-ranking ensures precision. I blend the cross-encoder score (30%) with the original vector score (70%) to balance semantic understanding with retrieval confidence."

### "How does your spoiler protection work in Oracle?"

**Strong Answer:**
> "When ingesting subtitles, I store the timestamp for each dialogue chunk as metadata. At query time, if the user specifies they've only watched up to, say, the 60-minute mark, I filter the ChromaDB query to exclude any chunks with start_time > 3600 seconds.
>
> This is implemented as a metadata filter in ChromaDB's query API. It's an example of how RAG systems need domain-specific features beyond basic retrieval."

---

## Data Engineering Questions

### "How did you build your movie dataset?"

**Strong Answer:**
> "I used the TMDB API to collect data on 6,147 movies. The pipeline:
>
> 1. **API Collection:** Batch fetched movies by popularity, extracting title, overview, genres, cast, crew, ratings
> 2. **Rate Limiting:** TMDB allows 40 requests/second, so I implemented exponential backoff
> 3. **Feature Engineering:** Concatenated title + overview + genres + cast names into a single text field for embedding
> 4. **Embedding:** Used BAAI/bge-large-en-v1.5 to generate 1024-dimensional vectors
> 5. **Storage:** Loaded into ChromaDB with all metadata for filtering
>
> I also built an expansion script that can add more movies incrementally without rebuilding the entire database."

### "Why ChromaDB over other vector databases?"

**Strong Answer:**
> "ChromaDB was the right choice for this project because:
>
> 1. **Simplicity:** Embeds directly in Python, no separate server needed for development
> 2. **Metadata filtering:** Native support for filtering by genre, year, rating during search
> 3. **Persistence:** Saves to disk automatically, survives restarts
> 4. **Open source:** No licensing costs for educational use
>
> For production with millions of documents, I'd consider Pinecone for managed scaling or Milvus for self-hosted. But at 6,147 documents, ChromaDB is fast and simple."

---

## DevOps Questions

### "Describe your Docker setup."

**Strong Answer:**
> "The docker-compose.yml defines six services:
>
> 1. **Frontend:** Node alpine image, builds React app
> 2. **Gateway:** Python slim image, runs FastAPI
> 3. **Movie Assistant:** Python with CUDA support, loads embedding models
> 4. **Oracle:** Python slim, connects to Ollama for LLM
> 5. **Binge/Sentiment:** Python slim, lightweight model loading
> 6. **Ollama:** Separate container for LLM inference
>
> Volumes persist ChromaDB data between restarts. I use healthchecks to ensure services start in the right order—the gateway waits for backends to be healthy before accepting traffic."

### "How do you handle secrets?"

**Strong Answer:**
> "All secrets are loaded from environment variables:
>
> - `TMDB_API_KEY` for data collection
> - `HF_API_TOKEN` for HuggingFace Inference API
> - No hardcoded keys anywhere in the codebase
>
> I have a `.env.example` file showing required variables, and `.env` is in `.gitignore`. This follows the 12-factor app methodology for configuration."

---

## Frontend Questions

### "Why React + Vite + Tailwind?"

**Strong Answer:**
> "Each choice serves a purpose:
>
> - **React:** Component-based architecture makes it easy to build reusable UI elements like MovieCard
> - **Vite:** Much faster than Create React App for development—hot module replacement in milliseconds
> - **Tailwind CSS:** Utility-first approach means I can prototype quickly without writing custom CSS files
>
> The frontend is intentionally simple because the focus of this project is the ML backend. But it demonstrates I can build a complete full-stack application."

---

## Challenges & Trade-offs

### "What was the hardest part of this project?"

**Strong Answer:**
> "Getting the retrieval quality right was the most challenging. My first version achieved only 40% 'Good or Excellent' results on my 100-query benchmark. The issues were:
>
> 1. **Keyword bias:** Director name queries were matching random action movies
> 2. **Franchise pollution:** Queries for 'superhero movies' returned 10 MCU films
> 3. **Mood misalignment:** 'Feel-good comedies' sometimes returned dark comedies
>
> I solved these iteratively:
> - Added name-phrase detection to boost exact matches
> - Implemented franchise deduplication
> - Built a mood lexicon for genre penalty/bonus
>
> By R21, I reached 70% 'Good or Excellent'—a significant improvement that required understanding both the data and the retrieval algorithms deeply."

### "What would you do differently?"

**Strong Answer:**
> "Three things:
>
> 1. **Evaluation first:** I should have built the 100-query benchmark before tuning the pipeline. Early iterations didn't have a systematic way to measure improvement.
>
> 2. **Better training data:** The 2,946 Q&A pairs could be higher quality. I'd use GPT-4 to generate initial drafts and have human annotators refine them.
>
> 3. **Production monitoring:** In a real system, I'd add logging to track which queries perform poorly and use that to improve the pipeline continuously."

### "If you had more time, what would you add?"

**Strong Answer:**
> "In priority order:
>
> 1. **User feedback loop:** Capture thumbs up/down on recommendations to improve ranking over time
> 2. **Personalization:** Track user preferences and adjust retrieval weights
> 3. **Multi-modal search:** Allow image-based queries ('find me movies with this aesthetic')
> 4. **Collaborative filtering:** Hybrid recommendations combining content-based and user-based signals
> 5. **A/B testing infrastructure:** Measure the impact of pipeline changes systematically"

---

## Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| Movies in database | 6,147 |
| Training Q&A pairs | 2,946 |
| Embedding dimensions | 1,024 (BGE-large) |
| Model parameters (Llama 3) | 8 billion |
| LoRA rank | 16 |
| Benchmark accuracy (R21) | 70% Good or Excellent |
| Services | 5 |
| Development phases | 5 |
| Active development time | ~13 days |

---

## GitHub Discussion Points

**Commit Message Style:**
> "I use conventional commits—feat, fix, chore, docs prefixes. Each commit is atomic and describes a complete change. You can see the project evolved through clear phases: setup, ML development, integration, security, and UI polish."

**Why No CI/CD?**
> "For an educational project running locally, CI/CD adds complexity without benefit. In production, I'd add GitHub Actions for linting, testing, and Docker image builds."

**Code Review Readiness:**
> "The codebase is organized for readability: each service is self-contained, there's comprehensive documentation, and I've included educational comments explaining ML concepts inline."
