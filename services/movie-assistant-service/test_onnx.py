from chromadb.utils import embedding_functions
import time

try:
    print("Initializing ONNX embedding function...")
    emb_fn = embedding_functions.ONNXMiniLM_L6_V2()
    
    print("Generating embedding...")
    start = time.time()
    embeddings = emb_fn(["This is a test sentence."])
    end = time.time()
    
    print(f"✅ Success! Embedding shape: {len(embeddings[0])}")
    print(f"Time taken: {end - start:.4f}s")
except Exception as e:
    print(f"❌ Error: {e}")
