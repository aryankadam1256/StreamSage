"""
Feature Engineering Module
Processes raw TMDB data and creates derived features for the movie dataset.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import re

from .config import FEATURE_PARAMS, DATA_PATHS, GENRES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Processes raw movie data and creates engineered features."""
    
    def __init__(self):
        self.runtime_categories = FEATURE_PARAMS["runtime_categories"]
        self.mood_keywords = FEATURE_PARAMS["mood_keywords"]
        self.style_keywords = FEATURE_PARAMS["style_keywords"]
    
    def extract_runtime_category(self, runtime: Optional[int]) -> str:
        """Categorize movie runtime."""
        if not runtime or runtime == 0:
            return "unknown"
        
        for category, (min_runtime, max_runtime) in self.runtime_categories.items():
            if min_runtime <= runtime < max_runtime:
                return category
        
        return "epic"  # Default for very long movies
    
    def extract_mood_tags(self, keywords: List[str], genres: List[str], overview: str) -> List[str]:
        """Extract mood tags based on keywords, genres, and overview."""
        mood_tags = set()
        
        # Combine all text sources
        text = " ".join(keywords + genres + [overview]).lower()
        
        # Check for mood keywords
        for mood, mood_words in self.mood_keywords.items():
            if any(word in text for word in mood_words):
                mood_tags.add(mood)
        
        # Genre-based mood inference
        genre_mood_map = {
            "Horror": "dark",
            "Thriller": "intense",
            "Comedy": "lighthearted",
            "Romance": "emotional",
            "Drama": "emotional",
            "Science Fiction": "thought_provoking",
            "Mystery": "mind_bending",
        }
        
        for genre in genres:
            if genre in genre_mood_map:
                mood_tags.add(genre_mood_map[genre])
        
        return list(mood_tags)
    
    def extract_style_tags(self, keywords: List[str], overview: str) -> List[str]:
        """Extract style tags based on keywords and overview."""
        style_tags = set()
        
        text = " ".join(keywords + [overview]).lower()
        
        for style, style_words in self.style_keywords.items():
            if any(word in text for word in style_words):
                style_tags.add(style)
        
        return list(style_tags)
    
    def extract_cast_crew_info(self, credits: Optional[Dict]) -> Dict:
        """Extract structured cast and crew information."""
        if not credits:
            return {
                "director": None,
                "top_cast": [],
                "cast_names": [],
                "crew_names": [],
            }
        
        # Extract director
        director = None
        crew = credits.get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                director = {
                    "id": person.get("id"),
                    "name": person.get("name"),
                }
                break
        
        # Extract top cast (top 10)
        cast = credits.get("cast", [])[:10]
        top_cast = [
            {
                "id": person.get("id"),
                "name": person.get("name"),
                "character": person.get("character"),
                "order": person.get("order"),
            }
            for person in cast
        ]
        
        cast_names = [person.get("name") for person in cast if person.get("name")]
        crew_names = [person.get("name") for person in crew if person.get("name")]
        
        return {
            "director": director,
            "top_cast": top_cast,
            "cast_names": cast_names,
            "crew_names": crew_names,
        }
    
    def calculate_popularity_score(self, vote_average: float, vote_count: int, popularity: float) -> float:
        """Calculate a normalized popularity score."""
        # Weighted combination of vote average, vote count, and TMDB popularity
        if vote_count == 0:
            return 0.0
        
        # Normalize vote average (0-10 scale)
        normalized_rating = vote_average / 10.0
        
        # Log scale for vote count (diminishing returns)
        import math
        normalized_votes = min(1.0, math.log10(vote_count + 1) / 5.0)
        
        # Normalize popularity (TMDB popularity can vary widely)
        normalized_popularity = min(1.0, popularity / 1000.0)
        
        # Weighted average
        score = (0.4 * normalized_rating + 0.3 * normalized_votes + 0.3 * normalized_popularity)
        
        return round(score, 3)
    
    def extract_keywords_list(self, keywords_data: Optional[Dict]) -> List[str]:
        """Extract keyword strings from TMDB keywords data."""
        if not keywords_data or "keywords" not in keywords_data:
            return []
        
        return [kw.get("name", "") for kw in keywords_data["keywords"] if kw.get("name")]
    
    def extract_similar_movies(self, recommendations: Optional[Dict]) -> List[int]:
        """Extract similar movie IDs from recommendations."""
        if not recommendations or "results" not in recommendations:
            return []
        
        return [movie["id"] for movie in recommendations["results"] if "id" in movie]
    
    def determine_era_tag(self, release_year: Optional[int]) -> str:
        """Determine era tag based on release year."""
        if not release_year:
            return "unknown"
        
        if release_year < 1980:
            return "classic"
        elif release_year < 2010:
            return "modern"
        else:
            return "contemporary"
    
    def process_movie(self, movie_data: Dict) -> Optional[Dict]:
        """Process a single movie and extract all features."""
        details = movie_data.get("details")
        if not details:
            return None
        
        # Basic metadata
        movie_id = details.get("id")
        title = details.get("title", "")
        original_title = details.get("original_title", "")
        overview = details.get("overview", "")
        tagline = details.get("tagline", "")
        
        # Temporal features
        release_date = details.get("release_date", "")
        release_year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None
        
        # Runtime
        runtime = details.get("runtime", 0)
        runtime_category = self.extract_runtime_category(runtime)
        
        # Genres
        genres_data = details.get("genres", [])
        genres = [g.get("name") for g in genres_data if g.get("name")]
        genre_ids = [g.get("id") for g in genres_data if g.get("id")]
        genre_primary = genres[0] if genres else None
        genre_secondary = genres[1] if len(genres) > 1 else None
        
        # Keywords
        keywords = self.extract_keywords_list(movie_data.get("keywords"))
        
        # Mood and style tags
        mood_tags = self.extract_mood_tags(keywords, genres, overview)
        style_tags = self.extract_style_tags(keywords, overview)
        
        # Cast and crew
        cast_crew = self.extract_cast_crew_info(movie_data.get("credits"))
        
        # Ratings and popularity
        vote_average = details.get("vote_average", 0.0)
        vote_count = details.get("vote_count", 0)
        popularity = details.get("popularity", 0.0)
        popularity_score = self.calculate_popularity_score(vote_average, vote_count, popularity)
        
        # Similar movies
        similar_movies = self.extract_similar_movies(movie_data.get("recommendations"))
        
        # Era tag
        era_tag = self.determine_era_tag(release_year)
        
        # Language
        original_language = details.get("original_language", "")
        
        # Budget and revenue
        budget = details.get("budget", 0)
        revenue = details.get("revenue", 0)
        
        # Construct feature dictionary
        features = {
            # Basic metadata
            "id": movie_id,
            "title": title,
            "original_title": original_title,
            "overview": overview,
            "tagline": tagline,
            
            # Temporal features
            "release_date": release_date,
            "release_year": release_year,
            "decade": (release_year // 10 * 10) if release_year else None,
            "era_tag": era_tag,
            
            # Runtime features
            "runtime": runtime,
            "runtime_category": runtime_category,
            
            # Genre features
            "genres": genres,
            "genre_ids": genre_ids,
            "genre_primary": genre_primary,
            "genre_secondary": genre_secondary,
            
            # Content features
            "keywords": keywords,
            "mood_tags": mood_tags,
            "style_tags": style_tags,
            
            # Cast and crew
            "director": cast_crew["director"],
            "top_cast": cast_crew["top_cast"],
            "cast_names": cast_crew["cast_names"],
            "crew_names": cast_crew["crew_names"],
            
            # Ratings and popularity
            "vote_average": vote_average,
            "vote_count": vote_count,
            "popularity": popularity,
            "popularity_score": popularity_score,
            
            # Similarity
            "similar_movies": similar_movies,
            
            # Other metadata
            "original_language": original_language,
            "budget": budget,
            "revenue": revenue,
        }
        
        # Add Wikipedia enrichment data if available
        wiki_enrichment = movie_data.get("wiki_enrichment", {})
        if wiki_enrichment:
            features["wiki_summary"] = wiki_enrichment.get("wiki_summary")
            features["awards"] = wiki_enrichment.get("wiki_awards", [])
            features["wiki_notable"] = wiki_enrichment.get("wiki_notable", [])
            
            # Merge wiki themes into keywords
            wiki_themes = wiki_enrichment.get("wiki_themes", [])
            if wiki_themes:
                features["keywords"] = list(set(keywords + wiki_themes))
            
            # Add award info to style tags if Oscar-winning
            if "Oscar" in features.get("awards", []):
                features["style_tags"] = list(set(style_tags + ["award_winning"]))
        
        return features
    
    def process_all_movies(self, raw_movies_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """Process all movies and create feature dataset."""
        logger.info(f"Processing {len(raw_movies_data)} movies...")
        
        processed_movies = {}
        failed_count = 0
        
        for movie_id, movie_data in raw_movies_data.items():
            features = self.process_movie(movie_data)
            if features:
                processed_movies[movie_id] = features
            else:
                failed_count += 1
        
        logger.info(f"Successfully processed {len(processed_movies)} movies")
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} movies")
        
        return processed_movies
    
    def save_features(self, features: Dict, filepath: str):
        """Save processed features to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Features saved to {filepath}")


def main():
    """Main function to run feature engineering."""
    logger.info("=== Feature Engineering Started ===")
    
    # Load raw movie data
    logger.info(f"Loading raw movie data from {DATA_PATHS['raw_movies']}...")
    with open(DATA_PATHS['raw_movies'], 'r', encoding='utf-8') as f:
        raw_movies = json.load(f)
    
    # Convert string keys to int if needed
    if raw_movies and isinstance(list(raw_movies.keys())[0], str):
        raw_movies = {int(k): v for k, v in raw_movies.items()}
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Process all movies
    processed_features = engineer.process_all_movies(raw_movies)
    
    # Save processed features
    engineer.save_features(processed_features, DATA_PATHS["processed_features"])
    
    logger.info("=== Feature Engineering Completed ===")
    logger.info(f"Total movies with features: {len(processed_features)}")


if __name__ == "__main__":
    main()
