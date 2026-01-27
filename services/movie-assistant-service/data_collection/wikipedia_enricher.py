"""
Wikipedia Data Enricher
Fetches additional movie information from Wikipedia to enhance the dataset.
"""

import requests
import time
import logging
import re
from typing import Dict, Optional, List
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaEnricher:
    """Enriches movie data with Wikipedia information."""
    
    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, rate_limit_delay: float = 0.2):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self.cache = {}  # Cache to avoid duplicate requests
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to Wikipedia API with rate limiting."""
        params["format"] = "json"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(self.WIKI_API_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Wikipedia request failed: {e}")
            return None
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[str]:
        """
        Search for a movie on Wikipedia and return the page title.
        
        Args:
            title: Movie title
            year: Release year (helps disambiguate)
        
        Returns:
            Wikipedia page title if found
        """
        # Try with year first for disambiguation
        search_queries = []
        if year:
            search_queries.append(f"{title} ({year} film)")
            search_queries.append(f"{title} {year} film")
        search_queries.append(f"{title} (film)")
        search_queries.append(title)
        
        for query in search_queries:
            cache_key = f"search:{query}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
            }
            
            data = self._make_request(params)
            if data and "query" in data and data["query"]["search"]:
                results = data["query"]["search"]
                
                # Find best match
                for result in results:
                    result_title = result["title"].lower()
                    if "film" in result_title or title.lower() in result_title:
                        page_title = result["title"]
                        self.cache[cache_key] = page_title
                        return page_title
        
        return None
    
    def get_page_content(self, page_title: str) -> Optional[Dict]:
        """
        Get Wikipedia page content including extract, categories, and infobox.
        
        Args:
            page_title: Wikipedia page title
        
        Returns:
            Dictionary with page content
        """
        cache_key = f"content:{page_title}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get page extract (summary)
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts|categories|pageprops",
            "exintro": True,  # Only intro section
            "explaintext": True,  # Plain text, no HTML
            "exsectionformat": "plain",
            "cllimit": 50,  # Categories limit
        }
        
        data = self._make_request(params)
        if not data or "query" not in data:
            return None
        
        pages = data["query"]["pages"]
        if not pages:
            return None
        
        page_id = list(pages.keys())[0]
        if page_id == "-1":  # Page not found
            return None
        
        page_data = pages[page_id]
        
        result = {
            "title": page_data.get("title"),
            "extract": page_data.get("extract", ""),
            "categories": [
                cat["title"].replace("Category:", "")
                for cat in page_data.get("categories", [])
            ],
        }
        
        self.cache[cache_key] = result
        return result
    
    def extract_movie_info(self, wiki_content: Dict) -> Dict:
        """
        Extract useful movie information from Wikipedia content.
        
        Args:
            wiki_content: Raw Wikipedia content
        
        Returns:
            Structured movie information
        """
        extract = wiki_content.get("extract", "")
        categories = wiki_content.get("categories", [])
        
        # Extract plot summary (first 2-3 sentences from extract)
        sentences = extract.split(". ")
        plot_summary = ". ".join(sentences[:3]) + "." if sentences else ""
        
        # Extract themes and keywords from categories
        themes = []
        awards = []
        cultural_info = []
        
        for category in categories:
            cat_lower = category.lower()
            
            # Award mentions
            if "oscar" in cat_lower or "academy award" in cat_lower:
                awards.append("Oscar")
            if "golden globe" in cat_lower:
                awards.append("Golden Globe")
            if "bafta" in cat_lower:
                awards.append("BAFTA")
            
            # Themes and genres
            theme_keywords = [
                "science fiction", "dystopian", "psychological", "superhero",
                "time travel", "artificial intelligence", "space", "war",
                "romance", "historical", "biographical", "animated",
                "comedy", "horror", "thriller", "drama", "action",
                "mystery", "fantasy", "adventure", "crime", "musical"
            ]
            for keyword in theme_keywords:
                if keyword in cat_lower and keyword not in themes:
                    themes.append(keyword)
            
            # Cultural/period info
            period_patterns = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
            for period in period_patterns:
                if period in cat_lower:
                    cultural_info.append(f"{period} film")
        
        # Extract notable info from text
        notable_aspects = []
        
        # Check for director mentions
        director_patterns = [
            r"directed by ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+)'s \d{4} film",
        ]
        for pattern in director_patterns:
            match = re.search(pattern, extract)
            if match:
                notable_aspects.append(f"Directed by {match.group(1)}")
                break
        
        # Check for critical acclaim
        if "critically acclaimed" in extract.lower():
            notable_aspects.append("Critically acclaimed")
        if "box office" in extract.lower():
            notable_aspects.append("Box office success")
        if "cult" in extract.lower():
            notable_aspects.append("Cult classic")
        
        return {
            "wiki_summary": plot_summary,
            "wiki_themes": themes,
            "wiki_awards": list(set(awards)),
            "wiki_cultural_info": cultural_info,
            "wiki_notable": notable_aspects,
        }
    
    def enrich_movie(self, movie: Dict) -> Dict:
        """
        Enrich a movie with Wikipedia data.
        
        Args:
            movie: Movie dictionary with title and release_year
        
        Returns:
            Movie dictionary with added Wikipedia data
        """
        title = movie.get("title", "")
        year = movie.get("release_year")
        
        if not title:
            return movie
        
        # Search for the movie
        page_title = self.search_movie(title, year)
        if not page_title:
            logger.debug(f"Wikipedia page not found for: {title}")
            return movie
        
        # Get page content
        wiki_content = self.get_page_content(page_title)
        if not wiki_content:
            return movie
        
        # Extract structured info
        wiki_info = self.extract_movie_info(wiki_content)
        
        # Merge with movie data
        enriched_movie = movie.copy()
        enriched_movie.update(wiki_info)
        
        # Enhance existing fields
        if wiki_info.get("wiki_themes"):
            existing_keywords = enriched_movie.get("keywords", [])
            enriched_movie["keywords"] = list(set(existing_keywords + wiki_info["wiki_themes"]))
        
        if wiki_info.get("wiki_awards"):
            enriched_movie["awards"] = wiki_info["wiki_awards"]
        
        logger.debug(f"Enriched: {title} with Wikipedia data")
        return enriched_movie
    
    def enrich_movies_batch(self, movies: Dict[int, Dict], max_movies: int = 1000) -> Dict[int, Dict]:
        """
        Enrich multiple movies with Wikipedia data.
        
        Args:
            movies: Dictionary of movie_id -> movie data
            max_movies: Maximum number of movies to enrich (API rate limits)
        
        Returns:
            Enriched movies dictionary
        """
        logger.info(f"Enriching up to {max_movies} movies with Wikipedia data...")
        
        # Sort by popularity to enrich most popular first
        sorted_movies = sorted(
            movies.items(),
            key=lambda x: x[1].get("popularity_score", 0),
            reverse=True
        )[:max_movies]
        
        enriched = {}
        enriched_count = 0
        
        for movie_id, movie in sorted_movies:
            enriched_movie = self.enrich_movie(movie)
            enriched[movie_id] = enriched_movie
            
            if "wiki_summary" in enriched_movie:
                enriched_count += 1
        
        # Add remaining movies without enrichment
        for movie_id, movie in movies.items():
            if movie_id not in enriched:
                enriched[movie_id] = movie
        
        logger.info(f"Successfully enriched {enriched_count} movies with Wikipedia data")
        return enriched


def main():
    """Test Wikipedia enricher."""
    enricher = WikipediaEnricher()
    
    # Test with a known movie
    test_movie = {
        "title": "Inception",
        "release_year": 2010,
        "genres": ["Action", "Science Fiction", "Thriller"],
        "keywords": [],
    }
    
    enriched = enricher.enrich_movie(test_movie)
    
    print("=== Wikipedia Enrichment Test ===")
    print(f"Title: {enriched['title']}")
    print(f"Wiki Summary: {enriched.get('wiki_summary', 'N/A')[:200]}...")
    print(f"Wiki Themes: {enriched.get('wiki_themes', [])}")
    print(f"Wiki Awards: {enriched.get('wiki_awards', [])}")
    print(f"Wiki Notable: {enriched.get('wiki_notable', [])}")


if __name__ == "__main__":
    main()
