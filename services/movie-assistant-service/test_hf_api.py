import os
from huggingface_hub import InferenceClient

token = os.getenv('HF_TOKEN')
print(f"Token present: {bool(token)}")

try:
    client = InferenceClient(token=token)
    print("Testing HF InferenceClient...")
    
    # Test with feature extraction
    result = client.feature_extraction(
        "This is a test sentence",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"✅ Success! Embedding shape: {len(result)} dimensions")
    print(f"First 5 values: {result[:5]}")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
