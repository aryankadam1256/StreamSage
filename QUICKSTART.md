# StreamSage - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Start the Backend Services

```bash
docker-compose up -d
```

This starts:
- ✅ Ollama (LLM server)
- ✅ Oracle RAG service
- ✅ Binge Predictor
- ✅ Sentiment Analyzer
- ✅ API Gateway

**First time?** Wait 2-3 minutes for services to initialize.

### Step 2: Pull the LLM Model (First Time Only)

```bash
docker exec -it streamsage-ollama ollama pull llama3:8b
```

This downloads Llama3 (4.7GB). It's a one-time setup.

**Tip**: Use `phi3:mini` (2GB) if you have limited bandwidth:
```bash
docker exec -it streamsage-ollama ollama pull phi3:mini
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 4: Start the Frontend

```bash
npm run dev
```

Frontend will run on **http://localhost:3000** with hot reload!

### Step 5: Explore the Dashboard

Open http://localhost:3000 and try:

1. **💬 Sentiment Vibe** - Analysis works immediately with pre-trained BERT
2. **📊 Binge Gauge** - Uses mock model (replace after Colab training)
3. **🔮 Oracle Chat** - Needs subtitle data (see below)

---

## 📁 Optional: Add Subtitle Data for Oracle

### Get Subtitle Files

Download `.srt` files from:
- [OpenSubtitles](https://www.opensubtitles.org/)
- [Subscene](https://subscene.com/)

Place in `data/subtitles/`:
```
data/subtitles/inception.srt
```

### Ingest Subtitles

```bash
docker exec -it streamsage-oracle python ingest.py \
  --file /app/data/subtitles/inception.srt \
  --movie-id inception
```

This takes ~30 seconds per movie.

### Try the Oracle

1. Go to **🔮 Oracle Chat** tab
2. Set Movie ID: `inception`
3. Ask: "What did they say about reality vs dreams?"
4. Get timestamp-accurate answers!

---

## 🔍 Verify Everything Works

### Check Service Health

```bash
# All services
docker-compose ps

# Individual health checks
curl http://localhost:8000/health  # Gateway
curl http://localhost:8001/health  # Oracle
curl http://localhost:8002/health  # Binge
curl http://localhost:8003/health  # Sentiment
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f oracle-rag-service
```

---

## 🛑 Stop Services

```bash
# Stop all
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v
```

---

## 📚 API Documentation

Each service has Swagger UI:

- Gateway: http://localhost:8000/docs
- Oracle: http://localhost:8001/docs
- Binge: http://localhost:8002/docs
- Sentiment: http://localhost:8003/health (Flask, no auto-docs)

---

## 🐛 Quick Troubleshooting

### Port Already in Use

Edit ports in `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # Change 3000 to 3001
```

### Ollama Uses Too Much RAM

Use a smaller model:
```bash
docker exec -it streamsage-ollama ollama pull phi3:mini
```

Then update Oracle service:
```bash
docker-compose restart oracle-rag-service
```

### Frontend Can't Reach Backend

Create `frontend/.env`:
```bash
VITE_API_URL=http://localhost:8000/api/v1
```

Restart frontend:
```bash
npm run dev
```

---

## ✅ What's Next?

1. ✅ **Explore the Code** - Read the educational comments in each service
2. ✅ **Customize the UI** - Edit React components with hot reload
3. ✅ **Train Models on Colab** - Coming in Phase 2!
4. ✅ **Add Your Own Movies** - Ingest more subtitle files

---

**Need help?** Check [walkthrough.md](file:///C:/Users/Laxmi%20Computers/.gemini/antigravity/brain/e4030b2b-3f13-4115-bce6-1673eb62a788/walkthrough.md) for detailed explanations!
