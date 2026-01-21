"""
=============================================================================
Binge Predictor Service - Main API
=============================================================================

🎓 CONCEPT: Sequence Modeling with LSTM

Reference: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
           "Deep Learning" by Goodfellow, Bengio & Courville (Chapter 10)

Why LSTM for Binge Prediction?

User watch behavior is inherently SEQUENTIAL:
- User watches Movie A → Movie B → Movie C → ???
- Will they continue watching (binge) or drop off?

Traditional neural networks can't remember previous inputs.
RNNs (Recurrent Neural Networks) have "memory" but suffer from:
- Vanishing gradients (forget long-term patterns)
- Exploding gradients (unstable training)

LSTMs solve this with three "gates":
1. FORGET GATE: What old info to discard
2. INPUT GATE: What new info to store
3. OUTPUT GATE: What to output based on memory

This makes LSTMs perfect for capturing:
- "User likes action movies followed by comedies"
- "User watches more on weekends"
- "User tends to stop after 3 movies"

=============================================================================
"""

import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./models/binge_model.h5")

# Global model reference
model = None


# =============================================================================
# Pydantic Models
# =============================================================================

class WatchHistoryItem(BaseModel):
    """Represents a single movie watch event."""
    movie_id: str = Field(..., description="Movie identifier")
    genre_ids: List[int] = Field(default=[], description="Genre IDs (e.g., [28, 12] for Action, Adventure)")
    rating: Optional[float] = Field(None, ge=0, le=10, description="User's rating if available")
    watch_duration_pct: float = Field(1.0, ge=0, le=1, description="Percentage of movie watched")
    timestamp: int = Field(..., description="Unix timestamp of watch")


class PredictionRequest(BaseModel):
    """
    Request for binge prediction.
    
    🎓 FEATURE ENGINEERING
    
    We don't just pass movie IDs to the model. We extract features:
    - Genre sequences (what genres did they watch?)
    - Rating patterns (are they consistently high/low?)
    - Watch velocity (how fast are they watching?)
    - Completion rates (do they finish movies?)
    - Time of day (morning vs night behavior)
    """
    user_id: str = Field(..., description="User identifier")
    watch_history: List[WatchHistoryItem] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Recent watch history (last N movies)"
    )
    current_hour: int = Field(
        default=20,
        ge=0,
        le=23,
        description="Current hour (0-23) for time-based features"
    )


class PredictionResponse(BaseModel):
    """Response with binge probability."""
    user_id: str
    continue_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability user will watch another movie"
    )
    risk_level: str = Field(
        ...,
        description="low/medium/high based on drop-off probability"
    )
    recommendation: str = Field(
        ...,
        description="Action recommendation for retention"
    )
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str


# =============================================================================
# Feature Engineering
# =============================================================================

def extract_features(request: PredictionRequest) -> np.ndarray:
    """
    Convert watch history to model input features.
    
    🎓 WHY FEATURE ENGINEERING?
    
    Raw data (movie IDs, timestamps) isn't useful to the model.
    We need to extract meaningful patterns:
    
    1. Sequence Features (capture order)
       - Genre sequence: [Action, Comedy, Action] → patterns
       - Rating trend: [8, 7, 5] → declining interest
    
    2. Aggregate Features (capture overall behavior)
       - Average rating: Overall satisfaction
       - Completion rate: Engagement level
       - Watch velocity: Binge tendency
    
    3. Temporal Features (capture time patterns)
       - Time of day: Night watchers behave differently
       - Session duration: Long sessions → more likely to continue
    """
    history = request.watch_history
    
    # Aggregate features
    avg_rating = np.mean([h.rating or 5.0 for h in history])
    avg_completion = np.mean([h.watch_duration_pct for h in history])
    
    # Recency features
    timestamps = [h.timestamp for h in history]
    if len(timestamps) > 1:
        time_diffs = np.diff(sorted(timestamps))
        avg_time_between = np.mean(time_diffs) / 3600  # Hours
    else:
        avg_time_between = 2.0  # Default
    
    # Genre diversity (more diverse = more engaged)
    all_genres = [g for h in history for g in h.genre_ids]
    genre_diversity = len(set(all_genres)) / max(len(all_genres), 1)
    
    # Session features
    session_length = len(history)
    is_night = 1.0 if request.current_hour >= 20 or request.current_hour <= 4 else 0.0
    
    # Recent trend (last 3 vs first 3)
    if len(history) >= 6:
        recent_rating = np.mean([h.rating or 5.0 for h in history[-3:]])
        early_rating = np.mean([h.rating or 5.0 for h in history[:3]])
        rating_trend = recent_rating - early_rating
    else:
        rating_trend = 0.0
    
    # Combine into feature vector
    features = np.array([
        avg_rating / 10,              # Normalize to [0, 1]
        avg_completion,
        min(avg_time_between, 10) / 10,  # Cap and normalize
        genre_diversity,
        min(session_length, 10) / 10,
        is_night,
        (rating_trend + 5) / 10,      # Shift to positive
    ]).reshape(1, -1)
    
    return features


# =============================================================================
# Mock Model (Used when no trained model available)
# =============================================================================

class MockBingeModel:
    """
    Mock model for development/testing.
    
    This simulates model behavior using simple heuristics.
    Replace with actual Keras model after training on Colab.
    """
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Heuristic prediction based on features.
        
        Higher probability if:
        - High ratings (satisfied user)
        - High completion rate (engaged)
        - Short time between movies (in binge mode)
        - Night time (established viewing session)
        """
        avg_rating = features[0, 0]
        completion = features[0, 1]
        time_gap = features[0, 2]
        session_length = features[0, 4]
        is_night = features[0, 5]
        
        # Base probability
        prob = 0.5
        
        # Adjust based on features
        prob += (avg_rating - 0.5) * 0.3    # Rating influence
        prob += (completion - 0.5) * 0.2     # Completion influence
        prob -= time_gap * 0.15              # Longer gaps → lower prob
        prob += session_length * 0.1         # Longer sessions → higher prob
        prob += is_night * 0.1               # Night boost
        
        # Clamp to [0, 1]
        prob = max(0.1, min(0.95, prob))
        
        return np.array([[prob]])


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize model on startup.
    
    🎓 MODEL LOADING PATTERN
    
    We load the model once at startup, not per-request.
    This is crucial for performance:
    - Model loading takes seconds
    - Inference takes milliseconds
    
    For production, consider:
    - Model versioning
    - A/B testing different models
    - Gradual rollouts
    """
    global model
    
    logger.info("🚀 Starting Binge Predictor Service...")
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load actual Keras model
            from tensorflow import keras
            model = keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Loaded model from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model: {e}. Using mock model.")
            model = MockBingeModel()
    else:
        logger.warning(f"⚠️ Model not found at {MODEL_PATH}. Using mock model.")
        logger.info("   Run the Colab notebook to train and download the model.")
        model = MockBingeModel()
    
    logger.info("✅ Binge Predictor Service ready!")
    
    yield
    
    logger.info("🛑 Shutting down Binge Predictor Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Binge Predictor Service",
    description="""
    📊 **Binge Predictor** - Watch Pattern Intelligence
    
    This service predicts whether a user will continue watching (binge)
    or drop off after their current movie, using LSTM sequence modeling.
    
    ## How It Works
    
    1. Send the user's recent watch history
    2. Features are extracted from the sequence
    3. LSTM model predicts continuation probability
    4. Risk level and recommendations are generated
    
    ## Use Cases
    
    - **Retention alerts**: Identify users likely to churn
    - **Content recommendations**: Suggest movies to keep users engaged
    - **UI optimization**: Show "Continue Watching" at the right moment
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_binge(request: PredictionRequest):
    """
    📊 Predict binge continuation probability.
    
    ## Example Request
    
    ```json
    {
        "user_id": "user_123",
        "watch_history": [
            {"movie_id": "m1", "genre_ids": [28], "rating": 8.5, "watch_duration_pct": 1.0, "timestamp": 1705684800},
            {"movie_id": "m2", "genre_ids": [35], "rating": 7.0, "watch_duration_pct": 0.9, "timestamp": 1705692000}
        ],
        "current_hour": 22
    }
    ```
    
    ## Response Interpretation
    
    - **continue_probability > 0.7**: User likely to binge
    - **continue_probability 0.4-0.7**: Uncertain, consider engagement tactics
    - **continue_probability < 0.4**: High drop-off risk
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and upload the model."
        )
    
    # Extract features
    features = extract_features(request)
    
    # Make prediction
    prediction = model.predict(features)
    prob = float(prediction[0][0])
    
    # Determine risk level and recommendation
    if prob >= 0.7:
        risk_level = "low"
        recommendation = "User is engaged! Consider showing 'Next Episode' autoplay."
    elif prob >= 0.4:
        risk_level = "medium"
        recommendation = "Show personalized recommendations to maintain engagement."
    else:
        risk_level = "high"
        recommendation = "High drop-off risk! Consider sending a 'Continue where you left off' notification later."
    
    return PredictionResponse(
        user_id=request.user_id,
        continue_probability=round(prob, 3),
        risk_level=risk_level,
        recommendation=recommendation,
        model_version="mock" if isinstance(model, MockBingeModel) else "v1.0",
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
