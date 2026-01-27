"""
Test TMDB API using httpx library (alternative to requests)
"""
import httpx
import json

API_KEY = "626d6c744ce54f356ec6ce2d0ff3b6e6"
BASE_URL = "https://api.themoviedb.org/3"

print("Testing TMDB API with httpx library...")
print("="*70)

try:
    # Test with httpx
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(
            f"{BASE_URL}/movie/550",
            params={"api_key": API_KEY}
        )
        response.raise_for_status()
        movie = response.json()
        
        print("✓ SUCCESS!")
        print(f"\nMovie Data:")
        print(f"  Title: {movie.get('title')}")
        print(f"  Year: {movie.get('release_date', '')[:4]}")
        print(f"  Runtime: {movie.get('runtime')} minutes")
        print(f"  Rating: {movie.get('vote_average')}/10")
        print(f"  Genres: {', '.join([g['name'] for g in movie.get('genres', [])])}")
        
        # Test fetching multiple movies
        print("\n" + "="*70)
        print("Fetching 5 popular movies...")
        response = client.get(
            f"{BASE_URL}/movie/popular",
            params={"api_key": API_KEY, "page": 1}
        )
        popular = response.json()
        
        for i, movie in enumerate(popular.get('results', [])[:5], 1):
            print(f"{i}. {movie.get('title')} ({movie.get('release_date', 'N/A')[:4]})")
        
        print("\n✓ httpx library works! We can use this for data collection.")
        
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    print("\nThe network issue persists even with httpx.")
