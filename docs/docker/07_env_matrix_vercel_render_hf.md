# Environment Variable Matrix (Vercel / Render / HF)

## Frontend (Vercel)

Required:
- `VITE_API_URL=https://<gateway-host>/api/v1`

## Gateway (Render)

Required:
- `ORACLE_SERVICE_URL=https://<oracle-host>`
- `BINGE_SERVICE_URL=https://<binge-host>`
- `SENTIMENT_SERVICE_URL=https://<sentiment-host>`
- `MOVIE_ASSISTANT_SERVICE_URL=https://<movie-assistant-host-or-hf-endpoint>`
- `LOG_LEVEL=INFO`

## Oracle RAG Service (Render)

Required:
- `OLLAMA_BASE_URL` (only if Ollama is reachable in same environment)
- `LLM_MODEL=llama3:8b` (or managed model identifier)
- `LOG_LEVEL=INFO`

If Ollama is not deployed in cloud:
- switch to managed inference path and update oracle code/env accordingly.

## Binge Service (Render)

Required:
- `MODEL_PATH=/app/models/binge_model.h5`
- `LOG_LEVEL=INFO`

## Sentiment Service (Render)

Required:
- `MODEL_PATH=/app/models/sentiment_model`
- `LOG_LEVEL=INFO`

## Movie Assistant (HF or container host)

Required:
- `HF_API_TOKEN=<token>`
- `INFERENCE_BACKEND=api`
- `CHROMA_DB_PATH=/app/data/chroma_db`
- `HF_HUB_DISABLE_XET=1`
- `HF_HUB_DOWNLOAD_TIMEOUT=600`
- `HF_HUB_ETAG_TIMEOUT=600`

## Registry Pull Settings (local/VM)

In `.env.deploy`:
- `REGISTRY_USER=<dockerhub_user_or_org>`
- `IMAGE_TAG=<release_tag>`
- `HF_API_TOKEN=<token>`
