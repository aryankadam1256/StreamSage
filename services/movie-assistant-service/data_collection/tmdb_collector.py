"""
TMDB Data Collector
Collects movie data from The Movie Database (TMDB) API.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm
import logging

from .config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    COLLECTION_PARAMS,
    GENRES,
    DATA_PATHS,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBCollector:
    """Collects movie data from TMDB API."""
    
    def __init__(self, api_key: str = TMDB_API_KEY):
        self.api_key = api_key
        self.base_url = TMDB_BASE_URL
        self.session = requests.Session()
        self.rate_limit_delay = COLLECTION_PARAMS["rate_limit_delay"]
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to TMDB API with rate limiting and error handling."""
        if params is None:
            params = {}
        
        params["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
    
    def get_popular_movies(self, num_pages: int = 250) -> List[Dict]:
        """Fetch popular movies from TMDB."""
        logger.info(f"Fetching popular movies ({num_pages} pages)...")
        movies = []
        
        for page in tqdm(range(1, num_pages + 1), desc="Popular movies"):
            data = self._make_request("movie/popular", {"page": page, "language": "en-US"})
            if data and "results" in data:
                # Filter for English original language
                results = [m for m in data["results"] if m.get("original_language") == "en"]
                movies.extend(results)
            else:
                logger.warning(f"Failed to fetch page {page}")
        
        logger.info(f"Collected {len(movies)} popular English movies")
        return movies
    
    def discover_movies_by_genre(self, genre_id: int, num_pages: int = 25) -> List[Dict]:
        """Discover movies by genre."""
        logger.info(f"Discovering movies for genre {GENRES.get(genre_id, genre_id)}...")
        movies = []
        
        params = {
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "vote_count.gte": COLLECTION_PARAMS["min_vote_count"],
            "primary_release_date.gte": f"{COLLECTION_PARAMS['min_year']}-01-01",
            "primary_release_date.lte": f"{COLLECTION_PARAMS['max_year']}-12-31",
            "with_original_language": "en",  # English only
        }
        
        for page in tqdm(range(1, num_pages + 1), desc=f"Genre {GENRES.get(genre_id, genre_id)}"):
            params["page"] = page
            data = self._make_request("discover/movie", params)
            if data and "results" in data:
                movies.extend(data["results"])
            else:
                logger.warning(f"Failed to fetch page {page} for genre {genre_id}")
        
        logger.info(f"Collected {len(movies)} English movies for genre {GENRES.get(genre_id, genre_id)}")
        return movies
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed information for a specific movie."""
        return self._make_request(f"movie/{movie_id}")
    
    def get_movie_credits(self, movie_id: int) -> Optional[Dict]:
        """Get cast and crew information for a movie."""
        return self._make_request(f"movie/{movie_id}/credits")
    
    def get_movie_keywords(self, movie_id: int) -> Optional[Dict]:
        """Get keywords for a movie."""
        return self._make_request(f"movie/{movie_id}/keywords")
    
    def get_movie_recommendations(self, movie_id: int) -> Optional[Dict]:
        """Get recommended similar movies."""
        return self._make_request(f"movie/{movie_id}/recommendations")
    
    def collect_complete_movie_data(self, movie_id: int) -> Optional[Dict]:
        """Collect all available data for a single movie."""
        movie_data = {
            "id": movie_id,
            "details": None,
            "credits": None,
            "keywords": None,
            "recommendations": None,
        }
        
        # Get details
        details = self.get_movie_details(movie_id)
        if not details:
            logger.warning(f"Failed to get details for movie {movie_id}")
            return None
        movie_data["details"] = details
        
        # Get credits
        credits = self.get_movie_credits(movie_id)
        if credits:
            movie_data["credits"] = credits
        
        # Get keywords
        keywords = self.get_movie_keywords(movie_id)
        if keywords:
            movie_data["keywords"] = keywords
        
        # Get recommendations
        recommendations = self.get_movie_recommendations(movie_id)
        if recommendations:
            movie_data["recommendations"] = recommendations
        
        return movie_data
    
    def collect_all_movies(self) -> Dict[int, Dict]:
        """Collect comprehensive movie dataset."""
        checkpoint_path = Path("data/raw/checkpoint.json")
        complete_movies = {}
        
        if checkpoint_path.exists():
            logger.info("Loading checkpoint...")
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    complete_movies = json.load(f)
                # Convert keys back to int
                complete_movies = {int(k): v for k, v in complete_movies.items()}
                logger.info(f"Loaded {len(complete_movies)} movies from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                complete_movies = {}

        logger.info("Starting comprehensive movie data collection...")
        
        # Step 1: Collect movie IDs from popular movies
        num_popular_pages = COLLECTION_PARAMS["num_popular_movies"] // 20
        popular_movies = self.get_popular_movies(num_pages=num_popular_pages)
        movie_ids: Set[int] = {movie["id"] for movie in popular_movies if "id" in movie}
        
        # Step 2: Collect movie IDs from genre-based discovery
        num_genre_pages = COLLECTION_PARAMS["num_per_genre"] // 20
        for genre_id in GENRES.keys():
            genre_movies = self.discover_movies_by_genre(genre_id, num_pages=num_genre_pages)
            movie_ids.update({movie["id"] for movie in genre_movies if "id" in movie})
        
        logger.info(f"Total unique movie IDs collected: {len(movie_ids)}")
        
        # Step 3: Collect complete data for each movie
        failed_ids = []
        
        # Filter out already collected movies
        remaining_ids = [mid for mid in movie_ids if mid not in complete_movies]
        logger.info(f"Remaining movies to collect: {len(remaining_ids)}")
        
        count = 0
        for movie_id in tqdm(remaining_ids, desc="Collecting complete movie data"):
            movie_data = self.collect_complete_movie_data(movie_id)
            if movie_data:
                complete_movies[movie_id] = movie_data
            else:
                failed_ids.append(movie_id)
            
            # Save checkpoint every 100 movies
            count += 1
            if count % 100 == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(complete_movies, f, ensure_ascii=False)
        
        # Final save and remove checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
        logger.info(f"Successfully collected data for {len(complete_movies)} movies")
        if failed_ids:
            logger.warning(f"Failed to collect data for {len(failed_ids)} movies")
        
        return complete_movies
    
    def save_data(self, data: Dict, filepath: str):
        """Save collected data to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {filepath}")


def main():
    """Main function to run data collection."""
    logger.info("=== TMDB Data Collection Started ===")
    
    # Initialize collector
    collector = TMDBCollector()
    
    # Collect all movie data
    movies_data = collector.collect_all_movies()
    
    # Save to file
    collector.save_data(movies_data, DATA_PATHS["raw_movies"])
    
    logger.info("=== TMDB Data Collection Completed ===")
    logger.info(f"Total movies collected: {len(movies_data)}")


if __name__ == "__main__":
    main()
