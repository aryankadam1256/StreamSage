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
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Timeout for service calls
SERVICE_TIMEOUT = 30.0


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
    async with httpx.AsyncClient(timeout=5.0) as client:
        services = [
            ("oracle", f"{ORACLE_SERVICE_URL}/health"),
            ("binge", f"{BINGE_SERVICE_URL}/health"),
            ("sentiment", f"{SENTIMENT_SERVICE_URL}/health"),
        ]
        
        for service_name, url in services:
            try:
                resp = await client.get(url)
                health_status["services"][service_name] = {
                    "status": "healthy" if resp.status_code == 200 else "degraded",
                    "response_time_ms": resp.elapsed.total_seconds() * 1000
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
    
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{ORACLE_SERVICE_URL}/ask",
                json=body
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
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        try:
            response = await client.get(f"{ORACLE_SERVICE_URL}/collections")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
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
    
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{BINGE_SERVICE_URL}/predict",
                json=body
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
    
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{SENTIMENT_SERVICE_URL}/analyze",
                json=body
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
    
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{SENTIMENT_SERVICE_URL}/batch",
                json=body
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
    
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
        # Make parallel requests
        import asyncio
        
        async def call_oracle():
            if body.get("user_query"):
                try:
                    resp = await client.post(
                        f"{ORACLE_SERVICE_URL}/ask",
                        json={"query": body["user_query"], "movie_id": body.get("movie_id")}
                    )
                    results["oracle"] = resp.json()
                except Exception as e:
                    results["errors"].append(f"Oracle: {str(e)}")
        
        async def call_sentiment():
            if body.get("review_text"):
                try:
                    resp = await client.post(
                        f"{SENTIMENT_SERVICE_URL}/analyze",
                        json={"text": body["review_text"]}
                    )
                    results["sentiment"] = resp.json()
                except Exception as e:
                    results["errors"].append(f"Sentiment: {str(e)}")
        
        async def call_binge():
            if body.get("watch_history"):
                try:
                    resp = await client.post(
                        f"{BINGE_SERVICE_URL}/predict",
                        json={
                            "user_id": body.get("user_id", "unknown"),
                            "watch_history": body["watch_history"]
                        }
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
