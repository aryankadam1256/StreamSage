"""
=============================================================================
API Gateway - Central Router
=============================================================================

🎓 CONCEPT: API Gateway Pattern

Reference: "Building Microservices" by Sam Newman (Chapter 11)
           "Microservices Patterns" by Chris Richardson

Why an API Gateway?

In a microservices architecture, clients shouldn't:
1. Know about every service's location/port
2. Handle service discovery themselves
3. Make multiple requests for composite data
4. Manage authentication separately with each service

The API Gateway provides:
- SINGLE ENTRY POINT: One URL for all API calls
- ROUTING: Forward requests to the right service
- COMPOSITION: Aggregate data from multiple services
- CROSS-CUTTING CONCERNS: Auth, rate limiting, logging

Trade-offs:
✅ Simpler client code
✅ Centralized security
✅ Better for mobile/frontend (fewer requests)
❌ Single point of failure (mitigate with multiple instances)
❌ Additional network hop (minimal latency cost)

=============================================================================
"""

import os
import logging
import asyncio
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Service URLs from environment
ORACLE_SERVICE_URL = os.getenv("ORACLE_SERVICE_URL", "http://localhost:8001")
BINGE_SERVICE_URL = os.getenv("BINGE_SERVICE_URL", "http://localhost:8002")
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://localhost:8003")
MOVIE_ASSISTANT_SERVICE_URL = os.getenv("MOVIE_ASSISTANT_SERVICE_URL", "http://localhost:8004")

# Timeout for service calls
SERVICE_TIMEOUT = 300.0
HEALTH_TIMEOUT = 5.0
ORACLE_STREAM_TIMEOUT = 300.0
MOVIE_DISCOVER_TIMEOUT = 300.0
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "626d6c744ce54f356ec6ce2d0ff3b6e6")
TMDB_IMAGE_TIMEOUT = float(os.getenv("TMDB_IMAGE_TIMEOUT", "3.5"))
TMDB_IMAGE_CACHE_TTL = int(os.getenv("TMDB_IMAGE_CACHE_TTL", "86400"))  # 24h
TMDB_IMAGE_NEGATIVE_CACHE_TTL = int(os.getenv("TMDB_IMAGE_NEGATIVE_CACHE_TTL", "60"))  # 1m

# Retry policy for transient network issues.
GATEWAY_GET_RETRIES = int(os.getenv("GATEWAY_GET_RETRIES", "2"))
GATEWAY_POST_RETRIES = int(os.getenv("GATEWAY_POST_RETRIES", "1"))
GATEWAY_RETRY_BACKOFF_MS = int(os.getenv("GATEWAY_RETRY_BACKOFF_MS", "120"))


_gateway_client: httpx.AsyncClient | None = None
_tmdb_image_cache: Dict[int, Dict[str, Any]] = {}


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result


def _get_gateway_client() -> httpx.AsyncClient:
    if _gateway_client is None:
        raise HTTPException(status_code=503, detail="Gateway HTTP client not initialized")
    return _gateway_client


async def _request_with_retry(
    method: str,
    url: str,
    *,
    retries: int,
    timeout: float,
    **kwargs,
) -> httpx.Response:
    client = _get_gateway_client()
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            return await client.request(method, url, timeout=timeout, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            backoff_s = (GATEWAY_RETRY_BACKOFF_MS / 1000.0) * (attempt + 1)
            await asyncio.sleep(backoff_s)

    assert last_exc is not None
    raise last_exc


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="StreamSage API Gateway",
    description="""
    🌐 **Central API Gateway** for StreamSage Platform
    
    This gateway routes requests to three AI services:
    - 🔮 **Oracle RAG Service**: Movie dialogue Q&A
    - 📊 **Binge Predictor**: Watch pattern analysis
    - 💬 **Sentiment Analyzer**: Review sentiment detection
    - 🎬 **Movie Assistant**: Fine-tuned Llama 3 movie discovery
    
    ## Architecture Benefits
    
    - **Single Entry Point**: Clients call one URL
    - **Service Abstraction**: Internal services can change without client updates
    - **Request Aggregation**: Combine multiple service calls
    - **Centralized Logging**: All requests flow through here
    
    ## API Routes
    
    - `/api/v1/oracle/*` → Oracle RAG Service
    - `/api/v1/binge/*` → Binge Predictor
    - `/api/v1/sentiment/*` → Sentiment Analyzer
    """,
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize a single shared HTTP client with connection pooling."""
    global _gateway_client

    limits = httpx.Limits(
        max_connections=int(os.getenv("GATEWAY_MAX_CONNECTIONS", "120")),
        max_keepalive_connections=int(os.getenv("GATEWAY_MAX_KEEPALIVE", "40")),
        keepalive_expiry=float(os.getenv("GATEWAY_KEEPALIVE_EXPIRY", "30.0")),
    )
    _gateway_client = httpx.AsyncClient(limits=limits)
    logger.info("Gateway HTTP client initialized with pooled connections")


@app.on_event("shutdown")
async def shutdown_event():
    """Close shared HTTP client on shutdown."""
    global _gateway_client
    if _gateway_client is not None:
        await _gateway_client.aclose()
        _gateway_client = None
        logger.info("Gateway HTTP client closed")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Gateway health check with downstream service status.
    
    🎓 HEALTH CHECK PATTERN
    
    A good health check includes:
    1. Gateway's own status
    2. Connectivity to downstream services
    3. Response time indicators
    
    This enables:
    - Load balancer health monitoring
    - Service discovery integration
    - Quick debugging (which service is down?)
    """
    health_status = {
        "gateway": "healthy",
        "services": {}
    }
    
    # Check each service
    services = [
        ("oracle", f"{ORACLE_SERVICE_URL}/health"),
        ("binge", f"{BINGE_SERVICE_URL}/health"),
        ("sentiment", f"{SENTIMENT_SERVICE_URL}/health"),
        ("movie_assistant", f"{MOVIE_ASSISTANT_SERVICE_URL}/health"),
    ]

    for service_name, url in services:
        try:
            resp = await _request_with_retry(
                "GET",
                url,
                retries=GATEWAY_GET_RETRIES,
                timeout=HEALTH_TIMEOUT,
            )
            health_status["services"][service_name] = {
                "status": "healthy" if resp.status_code == 200 else "degraded",
            }
        except Exception as e:
            health_status["services"][service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Overall status
    all_healthy = all(
        s.get("status") == "healthy" 
        for s in health_status["services"].values()
    )
    health_status["overall"] = "healthy" if all_healthy else "degraded"
    
    return health_status


# =============================================================================
# Oracle RAG Service Routes
# =============================================================================

@app.post("/api/v1/oracle/ask", tags=["Oracle"])
async def oracle_ask(request: Request):
    """
    🔮 Ask the Oracle about movie dialogues.
    
    Forwards request to Oracle RAG Service.
    """
    body = await request.json()
    
    try:
        response = await _request_with_retry(
            "POST",
            f"{ORACLE_SERVICE_URL}/ask",
            json=body,
            retries=GATEWAY_POST_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Oracle service error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Oracle service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to reach Oracle service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Oracle service unavailable: {str(e)}"
        )


@app.get("/api/v1/oracle/collections", tags=["Oracle"])
async def oracle_collections():
    """List available movie collections in Oracle."""
    try:
        response = await _request_with_retry(
            "GET",
            f"{ORACLE_SERVICE_URL}/collections",
            retries=GATEWAY_GET_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/oracle/ask/stream", tags=["Oracle"])
async def oracle_ask_stream(request: Request):
    """
    🔮 Streaming version of Oracle /ask via Server-Sent Events.

    Proxies the SSE stream from the Oracle service directly to the client.
    Uses httpx streaming mode so tokens flow through without buffering.
    The oracle service sends events in this format:
        data: {"type": "sources", ...}
        data: {"type": "token", "content": "..."}
        data: {"type": "done", ...}
    """
    body = await request.body()

    async def stream_proxy():
        # Use a long timeout for streaming (LLM on CPU can take 60-180s for first token)
        client = _get_gateway_client()
        try:
            async with client.stream(
                "POST",
                f"{ORACLE_SERVICE_URL}/ask/stream",
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=ORACLE_STREAM_TIMEOUT,
            ) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except Exception as e:
            logger.error(f"Oracle stream proxy error: {e}")
            error_event = f'data: {{"type":"error","message":"{str(e)}"}}\n\n'
            yield error_event.encode()

    return StreamingResponse(
        stream_proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/oracle/suggestions/{movie_id}", tags=["Oracle"])
async def oracle_suggestions(movie_id: str):
    """Get dynamic suggested questions for a specific movie."""
    try:
        response = await _request_with_retry(
            "GET",
            f"{ORACLE_SERVICE_URL}/suggestions/{movie_id}",
            retries=GATEWAY_GET_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# =============================================================================
# Binge Predictor Routes
# =============================================================================

@app.post("/api/v1/binge/predict", tags=["Binge Predictor"])
async def binge_predict(request: Request):
    """
    📊 Predict binge-watching probability.
    
    Forwards request to Binge Predictor Service.
    """
    body = await request.json()
    
    try:
        response = await _request_with_retry(
            "POST",
            f"{BINGE_SERVICE_URL}/predict",
            json=body,
            retries=GATEWAY_POST_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Binge service error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Binge service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to reach Binge service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Binge service unavailable: {str(e)}"
        )


# =============================================================================
# Sentiment Service Routes
# =============================================================================

@app.post("/api/v1/sentiment/analyze", tags=["Sentiment"])
async def sentiment_analyze(request: Request):
    """
    💬 Analyze text sentiment.
    
    Forwards request to Sentiment Service.
    """
    body = await request.json()
    
    try:
        response = await _request_with_retry(
            "POST",
            f"{SENTIMENT_SERVICE_URL}/analyze",
            json=body,
            retries=GATEWAY_POST_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Sentiment service error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Sentiment service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to reach Sentiment service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Sentiment service unavailable: {str(e)}"
        )


@app.post("/api/v1/sentiment/batch", tags=["Sentiment"])
async def sentiment_batch(request: Request):
    """Batch sentiment analysis."""
    body = await request.json()

    try:
        response = await _request_with_retry(
            "POST",
            f"{SENTIMENT_SERVICE_URL}/batch",
            json=body,
            retries=GATEWAY_POST_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Sentiment batch error: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# =============================================================================
# Composite Endpoints (Gateway-Specific)
# =============================================================================

@app.post("/api/v1/composite/movie-analysis", tags=["Composite"])
async def composite_movie_analysis(request: Request):
    """
    🎯 Composite endpoint: Analyze movie with multiple services.
    
    🎓 CONCEPT: API Composition
    
    Instead of making 3 separate requests from the frontend:
    1. GET oracle data
    2. GET sentiment
    3. GET binge prediction
    
    The gateway can combine them into ONE request.
    
    Benefits:
    - Fewer round trips (better for mobile)
    - Parallel execution (faster total time)
    - Consistent error handling
    
    Example Request:
    {
        "movie_id": "inception",
        "user_query": "What's the main theme?",
        "review_text": "Amazing cinematography!",
        "watch_history": [...]
    }
    """
    body = await request.json()
    
    results = {
        "movie_id": body.get("movie_id"),
        "oracle": None,
        "sentiment": None,
        "binge": None,
        "errors": []
    }
    
    # Make parallel requests
    async def call_oracle():
        if body.get("user_query"):
            try:
                resp = await _request_with_retry(
                    "POST",
                    f"{ORACLE_SERVICE_URL}/ask",
                    json={"query": body["user_query"], "movie_id": body.get("movie_id")},
                    retries=GATEWAY_POST_RETRIES,
                    timeout=SERVICE_TIMEOUT,
                )
                results["oracle"] = resp.json()
            except Exception as e:
                results["errors"].append(f"Oracle: {str(e)}")

    async def call_sentiment():
        if body.get("review_text"):
            try:
                resp = await _request_with_retry(
                    "POST",
                    f"{SENTIMENT_SERVICE_URL}/analyze",
                    json={"text": body["review_text"]},
                    retries=GATEWAY_POST_RETRIES,
                    timeout=SERVICE_TIMEOUT,
                )
                results["sentiment"] = resp.json()
            except Exception as e:
                results["errors"].append(f"Sentiment: {str(e)}")

    async def call_binge():
        if body.get("watch_history"):
            try:
                resp = await _request_with_retry(
                    "POST",
                    f"{BINGE_SERVICE_URL}/predict",
                    json={
                        "user_id": body.get("user_id", "unknown"),
                        "watch_history": body["watch_history"]
                    },
                    retries=GATEWAY_POST_RETRIES,
                    timeout=SERVICE_TIMEOUT,
                )
                results["binge"] = resp.json()
            except Exception as e:
                results["errors"].append(f"Binge: {str(e)}")
        
    # Execute in parallel
    await asyncio.gather(
        call_oracle(),
        call_sentiment(),
        call_binge(),
        return_exceptions=True
    )
    
    return results


# =============================================================================
# Movie Assistant / TMDB Routes
# ============================================================================= 

@app.get("/api/v1/movies/{tmdb_id}/images", tags=["Movies"])
async def get_movie_images(tmdb_id: int):
    """
    Fetch high-res poster and backdrop URLs for a specific movie directly from TMDB.
    """
    if not TMDB_API_KEY:
        return {
            "poster_path": None,
            "backdrop_path": None,
            "posters": [],
            "backdrops": [],
            "tmdb_available": False,
            "detail": "TMDB API key not configured",
        }

    now = time.time()
    cached = _tmdb_image_cache.get(tmdb_id)
    if cached:
        payload = dict(cached.get("payload", {}))
        ttl = TMDB_IMAGE_CACHE_TTL if payload.get("tmdb_available") else TMDB_IMAGE_NEGATIVE_CACHE_TTL
        if now - cached.get("ts", 0) < ttl:
            payload["cache_hit"] = True
            return payload

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=images"
    try:
        response = await _request_with_retry("GET", url, retries=1, timeout=TMDB_IMAGE_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        primary_poster = f"https://image.tmdb.org/t/p/w780{data.get('poster_path')}" if data.get("poster_path") else None
        primary_backdrop = f"https://image.tmdb.org/t/p/original{data.get('backdrop_path')}" if data.get("backdrop_path") else None

        images = data.get("images") or {}
        poster_urls = _dedupe_urls([
            *([primary_poster] if primary_poster else []),
            *[
                f"https://image.tmdb.org/t/p/w780{item.get('file_path')}"
                for item in (images.get("posters") or [])
                if item.get("file_path")
            ],
        ])
        backdrop_urls = _dedupe_urls([
            *([primary_backdrop] if primary_backdrop else []),
            *[
                f"https://image.tmdb.org/t/p/original{item.get('file_path')}"
                for item in (images.get("backdrops") or [])
                if item.get("file_path")
            ],
        ])

        payload = {
            "poster_path": poster_urls[0] if poster_urls else None,
            "backdrop_path": backdrop_urls[0] if backdrop_urls else None,
            "posters": poster_urls,
            "backdrops": backdrop_urls,
            "tmdb_available": True,
            "cache_hit": False,
        }
        _tmdb_image_cache[tmdb_id] = {"ts": now, "payload": payload}
        return payload
    except httpx.HTTPStatusError as e:
        logger.error(f"TMDB API error: {e}")
        if cached:
            payload = dict(cached.get("payload", {}))
            payload.update({"cache_hit": True, "stale_cache": True})
            return payload
        payload = {
            "poster_path": None,
            "backdrop_path": None,
            "posters": [],
            "backdrops": [],
            "tmdb_available": False,
            "cache_hit": False,
            "detail": f"TMDB fetch failed ({e.response.status_code})",
        }
        _tmdb_image_cache[tmdb_id] = {"ts": now, "payload": payload}
        return payload
    except Exception as e:
        logger.error(f"Failed to fetch from TMDB: {e}")
        if cached:
            payload = dict(cached.get("payload", {}))
            payload.update({"cache_hit": True, "stale_cache": True})
            return payload
        payload = {
            "poster_path": None,
            "backdrop_path": None,
            "posters": [],
            "backdrops": [],
            "tmdb_available": False,
            "cache_hit": False,
            "detail": "TMDB proxy unavailable",
        }
        _tmdb_image_cache[tmdb_id] = {"ts": now, "payload": payload}
        return payload

@app.post("/api/v1/discover", tags=["Movie Assistant"])
async def movie_discover(request: Request):
    """
    🎬 Discover movies with the fine-tuned Llama 3 assistant.

    Forwards request to Movie Assistant Service (RAG + fine-tuned LLM).
    """
    body = await request.json()

    try:
        response = await _request_with_retry(
            "POST",
            f"{MOVIE_ASSISTANT_SERVICE_URL}/discover",
            json=body,
            retries=GATEWAY_POST_RETRIES,
            timeout=MOVIE_DISCOVER_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Movie assistant error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Movie assistant error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to reach Movie assistant: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Movie assistant unavailable: {str(e)}"
        )


@app.get("/api/v1/discover/metrics", tags=["Movie Assistant"])
async def movie_assistant_metrics():
    """Get inference performance metrics from Movie Assistant."""
    try:
        response = await _request_with_retry(
            "GET",
            f"{MOVIE_ASSISTANT_SERVICE_URL}/inference-metrics",
            retries=GATEWAY_GET_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/v1/model-info", tags=["Movie Assistant"])
async def model_info():
    """Get current model/backend info from Movie Assistant."""
    try:
        response = await _request_with_retry(
            "GET",
            f"{MOVIE_ASSISTANT_SERVICE_URL}/health",
            retries=GATEWAY_GET_RETRIES,
            timeout=SERVICE_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "llm_model": data.get("llm_model", "unknown"),
            "inference_backend": data.get("inference_backend", "unknown"),
            "total_movies": data.get("total_movies", 0),
            "status": data.get("status", "unknown"),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": str(request.url),
            "method": request.method,
        },
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
