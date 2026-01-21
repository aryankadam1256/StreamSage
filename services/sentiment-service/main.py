"""
=============================================================================
Sentiment Service - Main API (Flask)
=============================================================================

🎓 CONCEPT: Transformers and BERT

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)

Why BERT for Sentiment Analysis?

Traditional NLP (bag-of-words, TF-IDF) loses word order and context.
RNNs/LSTMs process sequentially, missing long-range dependencies.

TRANSFORMERS introduced ATTENTION:
- Every word can "attend to" every other word
- No sequential bottleneck
- Captures "not good" ≠ "good" through position-aware attention

BERT adds BIDIRECTIONAL understanding:
- Previous models: "The bank [?]" → looked left only
- BERT: "The [?] by the river" → looks both directions
- Result: "bank" is correctly interpreted as riverbank, not financial institution

For sentiment, this means BERT understands:
- Negation: "not bad" = positive
- Sarcasm: "Oh great, another meeting" = negative
- Context: "This movie is sick!" = positive (slang)

WHY FLASK INSTEAD OF FASTAPI?
- Demonstrates alternative framework (educational)
- Flask is more widely used in ML serving
- Simpler for synchronous workloads
- Both are valid choices in production

=============================================================================
"""

import os
import logging
from typing import Optional

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./models/sentiment_model")
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Global model and tokenizer
model = None
tokenizer = None
device = None


# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """
    Load the sentiment analysis model.
    
    🎓 MODEL LOADING STRATEGY
    
    1. First, try to load fine-tuned model from MODEL_PATH
       (This is the model you'll train on Colab and download)
    
    2. If not found, fall back to a pre-trained HuggingFace model
       (distilbert-base-uncased-finetuned-sst-2-english)
    
    DistilBERT is chosen because:
    - 40% smaller than BERT
    - 60% faster inference
    - 97% of BERT's accuracy
    
    Perfect for production sentiment analysis.
    """
    global model, tokenizer, device
    
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")
    
    # Try loading custom model first
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"📦 Loading custom model from {MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            model.to(device)
            model.eval()
            logger.info("✅ Custom model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"⚠️ Failed to load custom model: {e}")
    
    # Fall back to pre-trained model
    logger.info(f"📦 Loading pre-trained model: {DEFAULT_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
        model.to(device)
        model.eval()
        logger.info("✅ Pre-trained model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for given text.
    
    🎓 INFERENCE PIPELINE
    
    1. TOKENIZATION: Convert text to token IDs
       "I love this movie!" → [101, 1045, 2293, 2023, 3185, 999, 102]
       
       - [CLS] token (101): Special start token
       - Word pieces: "movie" might become ["movie"] or ["mov", "##ie"]
       - [SEP] token (102): Special end token
    
    2. ENCODING: Pass through BERT layers
       - 12 transformer layers (for base BERT)
       - Each layer refines token representations
       - Final [CLS] token representation = sentence embedding
    
    3. CLASSIFICATION: Linear layer on [CLS] embedding
       - Maps 768-dim vector to 2 classes (positive/negative)
       - Softmax for probabilities
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    # Tokenize input
    # -------------------------------------------------------------------------
    # 🎓 TOKENIZER PARAMETERS
    # 
    # max_length=512: BERT's maximum sequence length
    # truncation=True: Cut off longer texts
    # padding=True: Pad shorter texts with [PAD] tokens
    # return_tensors="pt": Return PyTorch tensors
    # -------------------------------------------------------------------------
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get prediction
    probs = probabilities[0].cpu().numpy()
    predicted_class = int(probs.argmax())
    confidence = float(probs.max())
    
    # Map to labels (SST-2 format: 0=negative, 1=positive)
    label_map = {0: "negative", 1: "positive"}
    
    return {
        "label": label_map.get(predicted_class, "unknown"),
        "confidence": round(confidence, 4),
        "scores": {
            "negative": round(float(probs[0]), 4),
            "positive": round(float(probs[1]), 4),
        }
    }


# =============================================================================
# Flask Application
# =============================================================================

app = Flask(__name__)


@app.before_request
def ensure_model_loaded():
    """Ensure model is loaded before handling requests."""
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model on request: {e}")


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns model status and configuration.
    """
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else DEFAULT_MODEL,
        "device": str(device) if device else "unknown",
    })


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    """
    💬 Analyze sentiment of text.
    
    ## Request Body
    
    ```json
    {
        "text": "This movie was absolutely amazing!",
        "include_explanation": true
    }
    ```
    
    ## Response
    
    ```json
    {
        "text": "This movie was absolutely amazing!",
        "sentiment": {
            "label": "positive",
            "confidence": 0.9876,
            "scores": {
                "negative": 0.0124,
                "positive": 0.9876
            }
        },
        "explanation": "High confidence positive sentiment detected."
    }
    ```
    """
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    text = data.get("text", "").strip()
    include_explanation = data.get("include_explanation", False)
    
    if not text:
        return jsonify({"error": "Text field is required"}), 400
    
    if len(text) > 10000:
        return jsonify({"error": "Text too long (max 10000 characters)"}), 400
    
    # Predict sentiment
    try:
        result = predict_sentiment(text)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Build response
    response = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "sentiment": result,
    }
    
    # Add explanation if requested
    if include_explanation:
        conf = result["confidence"]
        label = result["label"]
        
        if conf >= 0.9:
            strength = "very high confidence"
        elif conf >= 0.7:
            strength = "high confidence"
        elif conf >= 0.5:
            strength = "moderate confidence"
        else:
            strength = "low confidence"
        
        response["explanation"] = f"{strength.capitalize()} {label} sentiment detected."
    
    return jsonify(response)


@app.route("/batch", methods=["POST"])
def analyze_batch():
    """
    Analyze sentiment for multiple texts.
    
    ## Request Body
    
    ```json
    {
        "texts": [
            "Great movie!",
            "Terrible acting.",
            "It was okay."
        ]
    }
    ```
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify({"error": "texts field is required"}), 400
    
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 texts per batch"}), 400
    
    results = []
    for text in texts:
        try:
            result = predict_sentiment(str(text))
            results.append({
                "text": text[:50] + "..." if len(str(text)) > 50 else text,
                "sentiment": result,
            })
        except Exception as e:
            results.append({
                "text": text[:50] + "..." if len(str(text)) > 50 else text,
                "error": str(e),
            })
    
    return jsonify({
        "count": len(results),
        "results": results,
    })


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
