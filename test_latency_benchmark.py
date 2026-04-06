import requests
import time
import json
import statistics

BASE_URL = "http://127.0.0.1:8004/discover"

queries = [
    # General queries (broad categories)
    "great sci-fi movies",
    "romantic comedies from the 90s",
    "action movies with car chases",
    
    # Very specific / narrow intent
    "leonardo dicaprio movies directed by martin scorsese",
    "scary movies about haunted houses",
    
    # Emotional / subjective
    "movies that will make me cry",
    "feel good family movies for the weekend",
]

def run_query(query, name_prefix="Test"):
    payload = {"query": query, "top_k": 5}
    start = time.time()
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - start
        
        # Verify result format
        if "recommended_movies" not in data or not data["recommended_movies"]:
            print(f"[{name_prefix}] WARNING: No movies returned for query: {query}")
            
        return latency
    except Exception as e:
        print(f"[{name_prefix}] ERROR running query '{query}': {e}")
        return None

def test_latency():
    print("="*60)
    print(" STREAM SAGE LATENCY BENCHMARK ")
    print("="*60)
    
    # 1. Warm-up
    print("\nPhase 1: Warmup (Loading models/RAM)")
    run_query("warmup query just to load everything", "Warmup")
    
    print("\nPhase 2 & 3: Cold vs Cache Sequential Benchmark")
    cold_latencies = []
    cached_latencies = []
    
    for i, q in enumerate(queries):
        # Cold Request
        cold_lat = run_query(q, f"Cold {i+1}")
        if cold_lat:
            cold_latencies.append(cold_lat)
            print(f"  [Q{i+1}] Cold: '{q[:30]}...' -> {cold_lat:.2f}s")
            
        # Cache Hit Request (Immediate, well within 45s TTL)
        cache_lat = run_query(q, f"Cache {i+1}")
        if cache_lat:
            cached_latencies.append(cache_lat)
            print(f"  [Q{i+1}] Hit:  '{q[:30]}...' -> {cache_lat:.4f}s")
            
    print("\n="*60)
    print(" SUMMARY:")
    if cold_latencies and cached_latencies:
        avg_cold = statistics.mean(cold_latencies)
        avg_cached = statistics.mean(cached_latencies)
        print(f" Average First-Time Latency: {avg_cold:.2f}s")
        print(f" Average Cached Latency:     {avg_cached:.4f}s")
        if avg_cold > 0:
            improvement = ((avg_cold - avg_cached) / avg_cold) * 100
            print(f" Cache Speedup Improvement:  {improvement:.1f}%")
        print("="*60)

if __name__ == "__main__":
    test_latency()
