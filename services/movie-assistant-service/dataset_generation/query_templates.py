"""
Query Templates for Dataset Generation
Defines templates for different query types to create diverse training examples.
"""

from typing import List, Dict
import random


class QueryTemplates:
    """Templates for generating diverse user queries."""
    
    # Similarity-based query templates
    SIMILARITY_TEMPLATES = [
        "I loved {movie}. What else should I watch?",
        "Can you recommend movies similar to {movie}?",
        "I just finished {movie} and want something like it",
        "Movies like {movie} please",
        "What's similar to {movie}?",
        "I really enjoyed {movie}. Any recommendations?",
        "Suggest movies similar to {movie}",
        "I'm looking for something like {movie}",
        "Give me recommendations based on {movie}",
        "What should I watch if I liked {movie}?",
        "I want more movies like {movie}",
        "Recommend films similar to {movie}",
        "{movie} was amazing! What's next?",
        "I need something similar to {movie}",
        "Show me movies like {movie}",
    ]
    
    # Cast/Crew query templates
    ACTOR_TEMPLATES = [
        "Movies with {actor}",
        "Show me films starring {actor}",
        "What movies has {actor} been in?",
        "I want to watch {actor} movies",
        "Recommend {actor} films",
        "Best movies featuring {actor}",
        "Films with {actor} in them",
        "What should I watch with {actor}?",
        "Give me {actor}'s best movies",
        "I love {actor}. What should I watch?",
    ]
    
    DIRECTOR_TEMPLATES = [
        "Movies directed by {director}",
        "Show me {director} films",
        "What has {director} directed?",
        "I want to watch {director} movies",
        "Recommend films by {director}",
        "Best {director} movies",
        "{director}'s filmography",
        "What should I watch from {director}?",
        "Give me {director}'s best work",
        "I love {director}'s style. What should I watch?",
    ]
    
    MULTI_ACTOR_TEMPLATES = [
        "Movies with {actor1} and {actor2}",
        "Films starring both {actor1} and {actor2}",
        "Show me movies featuring {actor1} and {actor2}",
        "What movies have {actor1} and {actor2} together?",
        "Recommend films with {actor1} and {actor2}",
    ]
    
    # Genre query templates
    SINGLE_GENRE_TEMPLATES = [
        "Good {genre} movies",
        "Recommend {genre} films",
        "Best {genre} movies",
        "I want to watch a {genre} movie",
        "Show me {genre} films",
        "What are some great {genre} movies?",
        "Suggest {genre} movies",
        "I'm in the mood for {genre}",
        "Top {genre} films",
        "Give me {genre} recommendations",
    ]
    
    MULTI_GENRE_TEMPLATES = [
        "{genre1} {genre2} movies",
        "I want a {genre1} movie that's also {genre2}",
        "Recommend {genre1}-{genre2} films",
        "Show me {genre1} {genre2} movies",
        "Best {genre1} {genre2} films",
        "I'm looking for {genre1} with {genre2} elements",
        "Suggest {genre1} {genre2} movies",
        "Give me {genre1} and {genre2} combined",
    ]
    
    # Runtime query templates
    RUNTIME_SHORT_TEMPLATES = [
        "Movies under {duration}",
        "Quick watch under {duration}",
        "Short films under {duration}",
        "I have {duration}. What should I watch?",
        "Recommend movies shorter than {duration}",
        "Films under {duration}",
        "Something quick, under {duration}",
        "Movies less than {duration}",
    ]
    
    RUNTIME_LONG_TEMPLATES = [
        "Epic films over {duration}",
        "Long movies over {duration}",
        "I want something epic, over {duration}",
        "Movies longer than {duration}",
        "Recommend long films over {duration}",
        "Epic movies over {duration}",
    ]
    
    # Mood/Vibe query templates
    MOOD_TEMPLATES = [
        "I need something {mood}",
        "{mood} movies",
        "Recommend {mood} films",
        "I'm in the mood for something {mood}",
        "Show me {mood} movies",
        "What are some {mood} films?",
        "Suggest {mood} movies",
        "I want a {mood} movie",
        "Give me {mood} recommendations",
        "Best {mood} films",
        "I'm feeling {mood}, what should I watch?",
        "Something {mood} please",
    ]
    
    # Complex multi-criteria templates
    COMPLEX_TEMPLATES = [
        "A {genre} movie under {duration} with {mood} vibes",
        "{genre} film with {actor} that's {mood}",
        "Something {mood} in the {genre} genre under {duration}",
        "{genre} movie from the {decade}s that's {mood}",
        "I want a {mood} {genre} film with great {style}",
        "{genre} movie with {style} under {duration}",
        "Recommend a {mood} {genre} film featuring {actor}",
        "{genre} movie that's {mood} and has {style}",
        "Something from the {decade}s, {genre}, and {mood}",
        "A {mood} film in the {genre} genre with {actor}",
    ]
    
    @staticmethod
    def get_similarity_query(movie_title: str) -> str:
        """Generate a similarity-based query."""
        template = random.choice(QueryTemplates.SIMILARITY_TEMPLATES)
        return template.format(movie=movie_title)
    
    @staticmethod
    def get_actor_query(actor_name: str) -> str:
        """Generate an actor-based query."""
        template = random.choice(QueryTemplates.ACTOR_TEMPLATES)
        return template.format(actor=actor_name)
    
    @staticmethod
    def get_director_query(director_name: str) -> str:
        """Generate a director-based query."""
        template = random.choice(QueryTemplates.DIRECTOR_TEMPLATES)
        return template.format(director=director_name)
    
    @staticmethod
    def get_multi_actor_query(actor1: str, actor2: str) -> str:
        """Generate a multi-actor query."""
        template = random.choice(QueryTemplates.MULTI_ACTOR_TEMPLATES)
        return template.format(actor1=actor1, actor2=actor2)
    
    @staticmethod
    def get_single_genre_query(genre: str) -> str:
        """Generate a single genre query."""
        template = random.choice(QueryTemplates.SINGLE_GENRE_TEMPLATES)
        return template.format(genre=genre.lower())
    
    @staticmethod
    def get_multi_genre_query(genre1: str, genre2: str) -> str:
        """Generate a multi-genre query."""
        template = random.choice(QueryTemplates.MULTI_GENRE_TEMPLATES)
        return template.format(genre1=genre1.lower(), genre2=genre2.lower())
    
    @staticmethod
    def get_runtime_short_query(duration: str = "2 hours") -> str:
        """Generate a short runtime query."""
        template = random.choice(QueryTemplates.RUNTIME_SHORT_TEMPLATES)
        return template.format(duration=duration)
    
    @staticmethod
    def get_runtime_long_query(duration: str = "3 hours") -> str:
        """Generate a long runtime query."""
        template = random.choice(QueryTemplates.RUNTIME_LONG_TEMPLATES)
        return template.format(duration=duration)
    
    @staticmethod
    def get_mood_query(mood: str) -> str:
        """Generate a mood-based query."""
        template = random.choice(QueryTemplates.MOOD_TEMPLATES)
        return template.format(mood=mood)
    
    @staticmethod
    def get_complex_query(params: Dict[str, str]) -> str:
        """Generate a complex multi-criteria query."""
        # Filter templates to find ones that can be formatted with the given params
        valid_templates = []
        for template in QueryTemplates.COMPLEX_TEMPLATES:
            # Check if all keys in template are present in params
            # Simple check: count braces or try formatting
            try:
                template.format(**params)
                valid_templates.append(template)
            except KeyError:
                continue
        
        if not valid_templates:
            # Fallback: return a generic query using available params
            parts = [f"{k}: {v}" for k, v in params.items()]
            return f"Find a movie with " + ", ".join(parts)
            
        template = random.choice(valid_templates)
        return template.format(**params)


# Mood descriptors for mood queries
MOOD_DESCRIPTORS = [
    "uplifting",
    "dark and gritty",
    "feel-good",
    "mind-bending",
    "emotional",
    "intense",
    "lighthearted",
    "thought-provoking",
    "inspiring",
    "suspenseful",
    "heartwarming",
    "disturbing",
    "whimsical",
    "tense",
    "cheerful",
]

# Style descriptors for complex queries
STYLE_DESCRIPTORS = [
    "cinematography",
    "visuals",
    "dialogue",
    "action sequences",
    "performances",
    "storytelling",
    "soundtrack",
    "atmosphere",
]

# Duration options
DURATION_OPTIONS_SHORT = [
    "90 minutes",
    "1.5 hours",
    "2 hours",
    "100 minutes",
    "an hour and a half",
]

DURATION_OPTIONS_LONG = [
    "2.5 hours",
    "3 hours",
    "150 minutes",
    "180 minutes",
]
