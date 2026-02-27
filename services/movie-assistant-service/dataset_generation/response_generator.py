"""
Response Generator
Generates natural language responses with movie recommendations.

KEY CONCEPT - Why Good Training Data Descriptions Matter:
When we fine-tune an LLM, the model learns to MIMIC our response patterns.
If our training data has truncated garbage like "Dr" or "E", the model will
learn to generate similarly broken responses. This is called "garbage in,
garbage out" - the #1 rule of ML/Data Science.

Changes made:
- Smart sentence splitting that handles abbreviations (Dr., Mr., U.S., etc.)
- Minimum description length enforcement (20 chars)
- Fallback to genre+year description when overview is too short
"""

import re
import random
from typing import List, Dict, Optional

# Common abbreviations that should NOT be treated as sentence endings
# This prevents "Dr." from being split as end-of-sentence
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "jr", "sr", "st", "vs", "etc", "inc",
    "ltd", "corp", "prof", "gen", "sgt", "pvt", "cpl", "lt", "col",
    "u.s", "u.k", "l.a", "n.y", "p.m", "a.m",
}


class ResponseGenerator:
    """Generates conversational responses with movie recommendations."""
    
    # Response opening templates
    OPENING_TEMPLATES = {
        "similarity": [
            "Based on {movie}'s {aspect}, I recommend:",
            "If you enjoyed {movie}, you'll love these:",
            "Since you liked {movie}, here are some similar films:",
            "Great choice! Here are movies like {movie}:",
            "{movie} is excellent! You might also enjoy:",
            "Fans of {movie} often love these films:",
        ],
        "actor": [
            "Here are some great {actor} films:",
            "{actor} has been in many excellent movies. Here are my top picks:",
            "You'll love these {actor} performances:",
            "Some of {actor}'s best work includes:",
            "{actor} is fantastic! Check out these films:",
        ],
        "director": [
            "Here are some excellent {director} films:",
            "{director} has directed many acclaimed movies:",
            "You'll appreciate {director}'s work in these films:",
            "{director}'s filmography includes these gems:",
            "Some of {director}'s best work:",
        ],
        "genre": [
            "Here are some excellent {genre} films:",
            "You'll love these {genre} movies:",
            "Great {genre} films to check out:",
            "Some top-rated {genre} movies:",
            "If you're in the mood for {genre}, try these:",
        ],
        "runtime": [
            "Perfect! Here are great movies {constraint}:",
            "Here are some excellent films {constraint}:",
            "You'll enjoy these movies {constraint}:",
            "Great options {constraint}:",
        ],
        "mood": [
            "Here are some {mood} films:",
            "If you're looking for something {mood}, try these:",
            "These {mood} movies should hit the spot:",
            "Perfect {mood} films for you:",
            "You'll love these {mood} movies:",
        ],
        "complex": [
            "Based on your criteria, here are some perfect matches:",
            "I found some great films that match what you're looking for:",
            "These movies check all your boxes:",
            "Here are some excellent recommendations:",
            "You'll love these films:",
        ],
    }
    
    @staticmethod
    def format_runtime(minutes: int) -> str:
        """Format runtime in a readable way."""
        hours = minutes // 60
        mins = minutes % 60
        
        if hours == 0:
            return f"{mins}m"
        elif mins == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {mins}m"
    
    @staticmethod
    def _extract_first_sentence(text: str, max_length: int = 150) -> str:
        """
        Extract the first meaningful sentence from a movie overview.

        WHY this is needed:
        The naive approach `text.split('.')[0]` breaks on abbreviations:
        - "Dr. Strange discovers..." → "Dr" (BROKEN)
        - "E.T. the Extra-Terrestrial..." → "E" (BROKEN)
        - "In 1945. Berlin falls..." → "In 1945" (OK but too short)

        HOW this works:
        1. Use regex to find sentence boundaries: ". " followed by uppercase letter
        2. Skip matches where the word before the period is a known abbreviation
        3. Enforce a minimum length of 20 characters
        4. If still too short, fall back to max_length truncation at word boundary

        INTERVIEW TIP: This is a classic NLP preprocessing problem.
        Production systems use spaCy's sentence segmenter or NLTK's punkt tokenizer.
        We use regex here to avoid adding heavy dependencies.
        """
        if not text or len(text) < 20:
            return text or ""

        # Pattern: period/exclamation/question mark, followed by space and uppercase
        # This catches real sentence boundaries while skipping abbreviations
        sentence_end_pattern = re.compile(r'([.!?])\s+([A-Z])')

        best_split = ""
        for match in sentence_end_pattern.finditer(text):
            pos = match.start() + 1  # position right after the punctuation

            # Check if the word before the period is an abbreviation
            before_period = text[:match.start()].rstrip().rsplit(None, 1)
            if before_period:
                last_word = before_period[-1].lower().rstrip('.')
                if last_word in ABBREVIATIONS:
                    continue  # Skip this - it's an abbreviation, not sentence end

            candidate = text[:pos].strip()

            # Must be at least 20 chars to be a meaningful sentence
            if len(candidate) >= 20:
                best_split = candidate
                break  # Take the first valid sentence

        # If we found a good sentence, use it
        if best_split and len(best_split) <= max_length:
            return best_split

        # Fallback: truncate at word boundary near max_length
        if len(text) <= max_length:
            return text

        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."

    @staticmethod
    def get_movie_description(movie: Dict, query_context: Optional[str] = None) -> str:
        """
        Generate a brief description for a movie recommendation.

        This is what appears in our training data, so quality here directly
        impacts model output quality.
        """
        title = movie.get("title", "Unknown")
        year = movie.get("release_year", "")
        runtime = movie.get("runtime", 0)
        genres = movie.get("genres", [])
        overview = movie.get("overview", "")
        mood_tags = movie.get("mood_tags", [])

        # Format basic info
        runtime_str = ResponseGenerator.format_runtime(runtime) if runtime else ""
        year_str = f"({year})" if year else ""

        # Create description parts
        parts = []

        # Add genre if available
        if genres:
            genre_str = genres[0] if len(genres) == 1 else f"{genres[0]}/{genres[1]}"
            parts.append(genre_str)

        # Add runtime
        if runtime_str:
            parts.append(runtime_str)

        # Create brief description from overview using smart extraction
        brief = ""
        if overview:
            brief = ResponseGenerator._extract_first_sentence(overview)

            # Quality gate: if the extracted sentence is too short, it's garbage
            if len(brief) < 20:
                # Fallback: use a longer truncation
                if len(overview) > 20:
                    brief = overview[:120].rsplit(' ', 1)[0] + "..."
                else:
                    brief = ""

        # Add mood/style context if relevant
        if mood_tags and query_context in ["mood", "complex"]:
            if not brief:
                brief = f"A {mood_tags[0]} film"

        # Format the full description
        info_str = " - ".join(parts) if parts else ""

        # Combine everything
        description = f"**{title}** {year_str}"
        if info_str:
            description += f" - {info_str}"
        if brief:
            description += f"\n   {brief}"

        return description
    
    @staticmethod
    def generate_opening(query_type: str, **kwargs) -> str:
        """Generate an opening line for the response."""
        templates = ResponseGenerator.OPENING_TEMPLATES.get(query_type, ResponseGenerator.OPENING_TEMPLATES["complex"])
        template = random.choice(templates)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            # Fallback to generic opening
            return "Here are some great recommendations:"
    
    @staticmethod
    def generate_response(
        query_type: str,
        movies: List[Dict],
        query_context: Optional[Dict] = None
    ) -> str:
        """
        Generate a complete response with movie recommendations.
        
        Args:
            query_type: Type of query (similarity, actor, director, genre, runtime, mood, complex)
            movies: List of movie dictionaries to recommend
            query_context: Additional context for the query (e.g., movie name, actor name, etc.)
        
        Returns:
            Formatted response string
        """
        if not movies:
            return "I couldn't find any movies matching your criteria. Could you try a different search?"
        
        # Generate opening
        context = query_context or {}
        opening = ResponseGenerator.generate_opening(query_type, **context)
        
        # Generate movie recommendations
        recommendations = []
        for i, movie in enumerate(movies, 1):
            desc = ResponseGenerator.get_movie_description(movie, query_type)
            recommendations.append(f"{i}. {desc}")
        
        # Combine opening and recommendations
        response = opening + "\n\n" + "\n\n".join(recommendations)
        
        return response
    
    @staticmethod
    def generate_similarity_response(source_movie: Dict, similar_movies: List[Dict]) -> str:
        """Generate response for similarity-based queries."""
        # Determine what aspect to highlight
        aspects = []
        if source_movie.get("genres"):
            aspects.append(f"{source_movie['genres'][0].lower()} genre")
        if source_movie.get("mood_tags"):
            aspects.append(f"{source_movie['mood_tags'][0]} tone")
        if source_movie.get("director"):
            aspects.append("directorial style")
        
        aspect = random.choice(aspects) if aspects else "style"
        
        context = {
            "movie": source_movie.get("title", "that film"),
            "aspect": aspect,
        }
        
        return ResponseGenerator.generate_response("similarity", similar_movies, context)
    
    @staticmethod
    def generate_actor_response(actor_name: str, movies: List[Dict]) -> str:
        """Generate response for actor-based queries."""
        context = {"actor": actor_name}
        return ResponseGenerator.generate_response("actor", movies, context)
    
    @staticmethod
    def generate_director_response(director_name: str, movies: List[Dict]) -> str:
        """Generate response for director-based queries."""
        context = {"director": director_name}
        return ResponseGenerator.generate_response("director", movies, context)
    
    @staticmethod
    def generate_genre_response(genre: str, movies: List[Dict]) -> str:
        """Generate response for genre-based queries."""
        context = {"genre": genre}
        return ResponseGenerator.generate_response("genre", movies, context)
    
    @staticmethod
    def generate_runtime_response(constraint: str, movies: List[Dict]) -> str:
        """Generate response for runtime-based queries."""
        context = {"constraint": constraint}
        return ResponseGenerator.generate_response("runtime", movies, context)
    
    @staticmethod
    def generate_mood_response(mood: str, movies: List[Dict]) -> str:
        """Generate response for mood-based queries."""
        context = {"mood": mood}
        return ResponseGenerator.generate_response("mood", movies, context)
    
    @staticmethod
    def generate_complex_response(movies: List[Dict]) -> str:
        """Generate response for complex multi-criteria queries."""
        return ResponseGenerator.generate_response("complex", movies)
