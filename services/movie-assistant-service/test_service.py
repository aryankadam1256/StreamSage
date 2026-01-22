"""
Movie Discovery Assistant - Test Script
========================================

This script tests the RAG service with various queries to demonstrate:
1. Semantic understanding (synonyms, context)
2. Metadata filtering (genre, year, rating)
3. Response quality
"""

import requests
import json
from typing import Dict, Any

# Service URL
BASE_URL = "http://localhost:8002"


def test_health():
    """Test if the service is running."""
    print("\n" + "=" * 60)
    print("🏥 Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Ollama Connected: {'✅' if data['ollama_connected'] else '❌'}")
        print(f"ChromaDB Loaded: {'✅' if data['chroma_loaded'] else '❌'}")
        print(f"Total Movies: {data['total_movies']}")
        print(f"Embedding Model: {data['embedding_model']}")
        print(f"LLM Model: {data['llm_model']}")
        
        return data['status'] == 'healthy'
    except Exception as e:
        print(f"❌ Service not reachable: {e}")
        return False


def test_query(
    query: str,
    genre: str = None,
    min_year: int = None,
    max_year: int = None,
    min_rating: float = None,
    top_k: int = 5
):
    """
    Test a discovery query.
    
    Args:
        query: Natural language query
        genre: Filter by genre (optional)
        min_year: Minimum release year (optional)
        max_year: Maximum release year (optional)
        min_rating: Minimum rating (optional)
        top_k: Number of movies to retrieve
    """
    print("\n" + "=" * 60)
    print(f"🔍 Query: '{query}'")
    if genre:
        print(f"   Genre: {genre}")
    if min_year or max_year:
        year_range = f"{min_year or '?'}-{max_year or '?'}"
        print(f"   Year: {year_range}")
    if min_rating:
        print(f"   Min Rating: {min_rating}")
    print("=" * 60)
    
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    if genre:
        payload["genre"] = genre
    if min_year:
        payload["min_year"] = min_year
    if max_year:
        payload["max_year"] = max_year
    if min_rating:
        payload["min_rating"] = min_rating
    
    try:
        response = requests.post(
            f"{BASE_URL}/discover",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Print answer
        print("\n🤖 Assistant's Response:")
        print("-" * 60)
        print(data['answer'])
        print("-" * 60)
        
        # Print recommended movies
        print(f"\n📋 Retrieved Movies ({data['retrieval_count']}):")
        for i, movie in enumerate(data['recommended_movies'], 1):
            print(f"\n{i}. {movie['title']} ({movie.get('year', 'N/A')})")
            print(f"   Rating: {movie.get('rating', 'N/A')}/10")
            print(f"   Genres: {movie.get('genres', 'N/A')}")
            if movie.get('director'):
                print(f"   Director: {movie['director']}")
            print(f"   Relevance: {movie['relevance_score']:.1%}")
        
        print(f"\n⚙️ Model Used: {data['model_used']}")
        return data
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (LLM might be slow)")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_test_suite():
    """Run a suite of test queries."""
    print("\n" + "=" * 70)
    print("🎬 Movie Discovery Assistant - Test Suite")
    print("=" * 70)
    
    # Check health first
    if not test_health():
        print("\n❌ Service is not healthy. Please check setup.")
        return
    
    print("\n✅ Service is healthy! Running test queries...")
    
    # Test 1: Simple semantic query
    test_query(
        query="Recommend a mind-bending sci-fi movie"
    )
    
    input("\nPress Enter to continue to next test...")
    
    # Test 2: Genre + Year filter
    test_query(
        query="Give me an action thriller",
        genre="Action",
        min_year=2000,
        max_year=2015
    )
    
    input("\nPress Enter to continue to next test...")
    
    # Test 3: Rating filter
    test_query(
        query="Recommend a highly-rated Christopher Nolan film",
        min_rating=8.0
    )
    
    input("\nPress Enter to continue to next test...")
    
    # Test 4: Similarity query
    test_query(
        query="Movies like The Matrix"
    )
    
    input("\nPress Enter to continue to next test...")
    
    # Test 5: Comedy from specific era
    test_query(
        query="I want a funny movie from the 90s",
        genre="Comedy",
        min_year=1990,
        max_year=1999
    )
    
    print("\n" + "=" * 70)
    print("✅ Test suite completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_test_suite()
