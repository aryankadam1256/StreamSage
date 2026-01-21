# 🎬 StreamSage

> **A Microservices-based Movie Intelligence Platform**

StreamSage is an educational project that teaches AI/ML concepts through building a real-world application. It combines three cutting-edge AI paradigms:

| Service | AI Concept | Use Case |
|---------|------------|----------|
| 🔮 **Oracle** | RAG (Retrieval-Augmented Generation) | Answer questions about movie dialogues |
| 📊 **Binge Predictor** | LSTM (Sequence Modeling) | Predict viewing drop-off probability |
| 💬 **Sentiment Engine** | BERT (Transformers) | Analyze movie review sentiment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│              VibeBar | OracleChat | BingeGauge                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API GATEWAY (FastAPI)                        │
│                   Routes, Auth, Rate Limiting                    │
└────────┬────────────────┬────────────────────┬──────────────────┘
         │                │                    │
         ▼                ▼                    ▼
┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐
│   Oracle    │  │ Binge Predictor │  │ Sentiment Engine  │
│  RAG + LLM  │  │      LSTM       │  │       BERT        │
└──────┬──────┘  └────────┬────────┘  └─────────┬─────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐
│  ChromaDB   │  │  binge_model.h5 │  │ sentiment_model/  │
│ (Vectors)   │  │    (Keras)      │  │   (PyTorch)       │
└─────────────┘  └─────────────────┘  └───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM recommended (for Ollama)
- Git

### 1. Clone & Start Services

```bash
# Clone the repository
git clone https://github.com/yourusername/StreamSage.git
cd StreamSage

# Start all services
docker-compose up -d

# First-time only: Pull the LLM model
docker exec -it streamsage-ollama ollama pull llama3:8b
```

### 2. Verify Services

| Service | URL | Health Check |
|---------|-----|--------------|
| Frontend | http://localhost:3000 | Visual |
| Gateway | http://localhost:8000/docs | Swagger UI |
| Oracle | http://localhost:8001/docs | Swagger UI |
| Binge | http://localhost:8002/docs | Swagger UI |
| Sentiment | http://localhost:8003/health | JSON |
| Ollama | http://localhost:11434 | API |

---

## 📁 Project Structure

```
StreamSage/
├── docker-compose.yml          # Service orchestration
├── README.md                   # You are here!
│
├── frontend/                   # React + Vite + Tailwind
│   ├── Dockerfile
│   ├── src/
│   │   ├── components/
│   │   │   ├── VibeBar.jsx     # Sentiment visualization
│   │   │   ├── OracleChat.jsx  # Time-travel Q&A
│   │   │   └── BingeGauge.jsx  # Watch predictor gauge
│   │   └── App.jsx
│   └── package.json
│
├── gateway/                    # FastAPI API Gateway
│   ├── Dockerfile
│   └── main.py
│
├── services/
│   ├── oracle-rag-service/     # RAG Pipeline
│   │   ├── Dockerfile
│   │   ├── main.py             # FastAPI endpoints
│   │   ├── ingest.py           # SRT processing
│   │   └── requirements.txt
│   │
│   ├── binge-service/          # LSTM Predictor
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   │
│   └── sentiment-service/      # BERT Classifier
│       ├── Dockerfile
│       ├── main.py
│       └── requirements.txt
│
├── notebooks/                  # Google Colab Training
│   ├── train_rnn.ipynb         # Binge model training
│   └── train_bert.ipynb        # Sentiment model training
│
└── data/                       # Local data storage
    ├── subtitles/              # .srt files
    ├── chromadb/               # Vector database
    └── models/                 # Trained model files
```

---

## 🎓 Learning Modules

This project is designed as a hands-on learning experience:

### Module 1: Docker & Microservices
- Container fundamentals
- Service orchestration
- Inter-service communication

### Module 2: RAG Pipeline (Oracle)
- Vector embeddings
- Semantic search
- LLM integration

### Module 3: Sequence Modeling (Binge)
- RNN/LSTM architecture
- Time-series prediction
- Model serialization

### Module 4: Transformers (Sentiment)
- BERT architecture
- Fine-tuning
- PyTorch serving

### Module 5: Frontend Integration
- React components
- API consumption
- Real-time updates

---

## 🔧 Development

### Running Individual Services

```bash
# Oracle service only
docker-compose up -d ollama oracle-rag-service

# Just the frontend
cd frontend && npm run dev
```

### Ingesting Subtitle Data

```bash
# Place .srt files in data/subtitles/
docker exec -it streamsage-oracle python ingest.py
```

### Training Models (Google Colab)

1. Open `notebooks/train_rnn.ipynb` in Google Colab
2. Run all cells
3. Download `binge_model.h5`
4. Place in `data/models/`

---

## 📚 References

- *Building Microservices* - Sam Newman
- *Deep Learning* - Goodfellow, Bengio & Courville
- *Attention Is All You Need* - Vaswani et al.
- *BERT: Pre-training of Deep Bidirectional Transformers* - Devlin et al.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with 💜 for learning AI/ML through practical application**
