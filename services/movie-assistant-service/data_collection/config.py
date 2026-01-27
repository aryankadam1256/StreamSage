import os
from dotenv import load_dotenv

load_dotenv()

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"

# Hugging Face Configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Data Collection Parameters
COLLECTION_PARAMS = {
    "num_popular_movies": 5000,  # Number of popular movies to collect
    "num_per_genre": 500,  # Number of movies per genre
    "min_vote_count": 50,   # Lowered slightly to get more movies
    "min_year": 1970,  # Earliest release year
    "max_year": 2024,  # Latest release year
    "languages": ["en"],  # Primary languages to focus on
    "rate_limit_delay": 0.25,  # Delay between API calls (seconds)
}

# Genres to collect (TMDB genre IDs)
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    "runtime_categories": {
        "short": (0, 90),
        "medium": (90, 150),
        "long": (150, 180),
        "epic": (180, 999),
    },
    "mood_keywords": {
        "dark": ["dark", "gritty", "noir", "bleak", "disturbing", "violent", "brutal"],
        "uplifting": ["uplifting", "inspiring", "heartwarming", "feel-good", "hopeful", "triumphant"],
        "intense": ["intense", "gripping", "suspenseful", "tense", "thrilling", "edge-of-your-seat"],
        "lighthearted": ["lighthearted", "fun", "whimsical", "playful", "cheerful", "amusing"],
        "thought_provoking": ["thought-provoking", "philosophical", "deep", "complex", "intellectual", "cerebral"],
        "emotional": ["emotional", "touching", "moving", "poignant", "tearjerker", "heartfelt"],
        "mind_bending": ["mind-bending", "twist", "surreal", "psychological", "reality-bending", "confusing"],
    },
    "style_keywords": {
        "visually_stunning": ["beautiful", "cinematography", "visual", "gorgeous", "stunning", "aesthetic"],
        "dialogue_driven": ["dialogue", "conversation", "talking", "verbal", "witty", "script"],
        "action_packed": ["action", "fight", "chase", "explosion", "combat", "battle"],
        "character_study": ["character", "performance", "acting", "portrait", "study", "introspective"],
    },
}

# Dataset Generation Parameters
DATASET_PARAMS = {
    "total_examples": 4000,  # Target number of training examples
    "query_type_distribution": {
        "similarity": 0.25,  # 25% similarity queries
        "cast_crew": 0.20,  # 20% cast/crew queries
        "genre": 0.20,  # 20% genre queries
        "runtime": 0.10,  # 10% runtime queries
        "mood": 0.15,  # 15% mood queries
        "complex": 0.10,  # 10% complex multi-criteria queries
    },
    "train_split": 0.8,  # 80% training
    "val_split": 0.1,  # 10% validation
    "test_split": 0.1,  # 10% test
    "recommendations_per_response": (3, 5),  # Min and max recommendations per response
}

# File Paths
DATA_PATHS = {
    "raw_movies": "data/raw/movies.json",
    "raw_credits": "data/raw/credits.json",
    "raw_keywords": "data/raw/keywords.json",
    "processed_features": "data/processed/movie_features.json",
    "train_dataset": "data/datasets/train.jsonl",
    "val_dataset": "data/datasets/val.jsonl",
    "test_dataset": "data/datasets/test.jsonl",
}
