"""
Quick test to verify TMDB API connectivity and show sample movie data.
"""
import requests
import json
import time

API_KEY = "626d6c744ce54f356ec6ce2d0ff3b6e6"
BASE_URL = "https://api.themoviedb.org/3"

def test_api_connection():
    """Test if TMDB API is accessible."""
    print("Testing TMDB API connection...")
    try:
        response = requests.get(
            f"{BASE_URL}/movie/popular",
            params={"api_key": API_KEY, "page": 1},
            timeout=10
        )
        response.raise_for_status()
        print("✓ API connection successful!")
        return response.json()
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return None

def fetch_sample_movies(count=10):
    """Fetch sample movie data."""
    print(f"\nFetching {count} sample movies...")
    movies = []
    
    for movie_id in [550, 155, 13, 680, 27205, 278, 238, 424, 389, 129]:  # Popular movie IDs
        try:
            print(f"Fetching movie {movie_id}...", end=" ")
            response = requests.get(
                f"{BASE_URL}/movie/{movie_id}",
                params={"api_key": API_KEY},
                timeout=10
            )
            response.raise_for_status()
            movie = response.json()
            movies.append({
                "id": movie.get("id"),
                "title": movie.get("title"),
                "year": movie.get("release_date", "")[:4],
                "runtime": movie.get("runtime"),
                "genres": [g["name"] for g in movie.get("genres", [])],
                "vote_average": movie.get("vote_average"),
            })
            print("✓")
            time.sleep(0.25)  # Rate limiting
        except Exception as e:
            print(f"✗ {e}")
    
    return movies

if __name__ == "__main__":
    # Test connection
    popular_data = test_api_connection()
    
    if popular_data:
        print(f"\nFound {len(popular_data.get('results', []))} popular movies")
        print("\nSample popular movies:")
        for i, movie in enumerate(popular_data.get("results", [])[:5], 1):
            print(f"{i}. {movie.get('title')} ({movie.get('release_date', 'N/A')[:4]})")
    
    # Fetch detailed sample data
    print("\n" + "="*70)
    sample_movies = fetch_sample_movies(10)
    
    if sample_movies:
        print("\n" + "="*70)
        print("SAMPLE MOVIE DATA (10 movies):")
        print("="*70)
        print(json.dumps(sample_movies, indent=2, ensure_ascii=False))
        print("\n✓ Successfully fetched data for", len(sample_movies), "movies")
    else:
        print("\n✗ No movies fetched - API connection issues")
