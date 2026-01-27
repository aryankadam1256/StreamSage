"""
Dataset Builder
Orchestrates the creation of instruction-response pairs for fine-tuning.
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from .query_templates import (
    QueryTemplates,
    MOOD_DESCRIPTORS,
    STYLE_DESCRIPTORS,
    DURATION_OPTIONS_SHORT,
    DURATION_OPTIONS_LONG,
)
from .response_generator import ResponseGenerator
from data_collection.config import DATASET_PARAMS, DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds instruction-response dataset for fine-tuning."""
    
    def __init__(self, movies_data: Dict[int, Dict]):
        self.movies_data = movies_data
        self.movies_list = list(movies_data.values())
        self.response_generator = ResponseGenerator()
        
        # Index movies by various attributes for efficient querying
        self._build_indices()
        
    def _build_indices(self):
        """Build indices for efficient movie lookup."""
        logger.info("Building movie indices...")
        
        # Index by genre
        self.genre_index = defaultdict(list)
        for movie in self.movies_list:
            for genre in movie.get("genres", []):
                self.genre_index[genre].append(movie)
        
        # Index by actor
        self.actor_index = defaultdict(list)
        for movie in self.movies_list:
            for actor_name in movie.get("cast_names", [])[:10]:  # Top 10 cast
                self.actor_index[actor_name].append(movie)
        
        # Index by director
        self.director_index = defaultdict(list)
        for movie in self.movies_list:
            director = movie.get("director")
            if director and director.get("name"):
                self.director_index[director["name"]].append(movie)
        
        # Index by mood
        self.mood_index = defaultdict(list)
        for movie in self.movies_list:
            for mood in movie.get("mood_tags", []):
                self.mood_index[mood].append(movie)
        
        # Index by runtime category
        self.runtime_index = defaultdict(list)
        for movie in self.movies_list:
            category = movie.get("runtime_category")
            if category:
                self.runtime_index[category].append(movie)
        
        logger.info(f"Indexed {len(self.movies_list)} movies")
        logger.info(f"Genres: {len(self.genre_index)}, Actors: {len(self.actor_index)}, Directors: {len(self.director_index)}")
    
    def _get_similar_movies(self, source_movie: Dict, count: int = 5) -> List[Dict]:
        """Get similar movies based on genres, mood, and TMDB recommendations."""
        similar_ids = set(source_movie.get("similar_movies", []))
        source_genres = set(source_movie.get("genres", []))
        source_moods = set(source_movie.get("mood_tags", []))
        
        # Score all movies by similarity
        candidates = []
        for movie in self.movies_list:
            if movie["id"] == source_movie["id"]:
                continue
            
            score = 0
            
            # TMDB recommendation bonus
            if movie["id"] in similar_ids:
                score += 10
            
            # Genre overlap
            movie_genres = set(movie.get("genres", []))
            genre_overlap = len(source_genres & movie_genres)
            score += genre_overlap * 3
            
            # Mood overlap
            movie_moods = set(movie.get("mood_tags", []))
            mood_overlap = len(source_moods & movie_moods)
            score += mood_overlap * 2
            
            # Same director bonus
            source_director = source_movie.get("director")
            movie_director = movie.get("director")
            
            if (source_director and movie_director and 
                source_director.get("name") == movie_director.get("name")):
                score += 5
            
            # Similar era
            if source_movie.get("era_tag") == movie.get("era_tag"):
                score += 1
            
            if score > 0:
                candidates.append((score, movie))
        
        # Sort by score and return top matches
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [movie for _, movie in candidates[:count]]
    
    def _select_random_movies(self, candidates: List[Dict], count: int) -> List[Dict]:
        """Randomly select movies from candidates, weighted by popularity."""
        if len(candidates) <= count:
            return candidates
        
        # Weight by popularity score
        weights = [movie.get("popularity_score", 0.5) for movie in candidates]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(candidates)] * len(candidates)
        
        selected = random.choices(candidates, weights=weights, k=count)
        return selected
    
    def generate_similarity_examples(self, num_examples: int) -> List[Dict]:
        """Generate similarity-based query examples."""
        logger.info(f"Generating {num_examples} similarity examples...")
        examples = []
        
        # Select source movies (popular ones work best)
        source_movies = sorted(self.movies_list, key=lambda x: x.get("popularity_score", 0), reverse=True)
        source_movies = source_movies[:min(num_examples, len(source_movies))]
        
        for source_movie in source_movies[:num_examples]:
            # Generate query
            query = QueryTemplates.get_similarity_query(source_movie["title"])
            
            # Get similar movies
            similar_movies = self._get_similar_movies(source_movie, count=random.randint(3, 5))
            
            if not similar_movies:
                continue
            
            # Generate response
            response = self.response_generator.generate_similarity_response(source_movie, similar_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        logger.info(f"Generated {len(examples)} similarity examples")
        return examples
    
    def generate_actor_examples(self, num_examples: int) -> List[Dict]:
        """Generate actor-based query examples."""
        logger.info(f"Generating {num_examples} actor examples...")
        examples = []
        
        # Get actors with enough movies
        eligible_actors = [(actor, movies) for actor, movies in self.actor_index.items() if len(movies) >= 3]
        
        if len(eligible_actors) < num_examples:
            logger.warning(f"Only {len(eligible_actors)} actors with 3+ movies")
            num_examples = len(eligible_actors)
        
        selected_actors = random.sample(eligible_actors, min(num_examples, len(eligible_actors)))
        
        for actor_name, actor_movies in selected_actors:
            # Generate query
            query = QueryTemplates.get_actor_query(actor_name)
            
            # Select best movies by this actor
            selected_movies = self._select_random_movies(actor_movies, count=random.randint(3, 5))
            
            # Generate response
            response = self.response_generator.generate_actor_response(actor_name, selected_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        logger.info(f"Generated {len(examples)} actor examples")
        return examples
    
    def generate_director_examples(self, num_examples: int) -> List[Dict]:
        """Generate director-based query examples."""
        logger.info(f"Generating {num_examples} director examples...")
        examples = []
        
        # Get directors with enough movies
        eligible_directors = [(director, movies) for director, movies in self.director_index.items() if len(movies) >= 3]
        
        if len(eligible_directors) < num_examples:
            logger.warning(f"Only {len(eligible_directors)} directors with 3+ movies")
            num_examples = len(eligible_directors)
        
        selected_directors = random.sample(eligible_directors, min(num_examples, len(eligible_directors)))
        
        for director_name, director_movies in selected_directors:
            # Generate query
            query = QueryTemplates.get_director_query(director_name)
            
            # Select best movies by this director
            selected_movies = self._select_random_movies(director_movies, count=random.randint(3, 5))
            
            # Generate response
            response = self.response_generator.generate_director_response(director_name, selected_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        logger.info(f"Generated {len(examples)} director examples")
        return examples
    
    def generate_genre_examples(self, num_examples: int) -> List[Dict]:
        """Generate genre-based query examples."""
        logger.info(f"Generating {num_examples} genre examples...")
        examples = []
        
        # Single genre queries
        single_genre_count = num_examples // 2
        multi_genre_count = num_examples - single_genre_count
        
        # Single genre
        available_genres = list(self.genre_index.keys())
        for _ in range(single_genre_count):
            genre = random.choice(available_genres)
            query = QueryTemplates.get_single_genre_query(genre)
            
            genre_movies = self.genre_index[genre]
            selected_movies = self._select_random_movies(genre_movies, count=random.randint(3, 5))
            
            response = self.response_generator.generate_genre_response(genre.lower(), selected_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        # Multi-genre
        for _ in range(multi_genre_count):
            if len(available_genres) < 2:
                break
            
            genre1, genre2 = random.sample(available_genres, 2)
            query = QueryTemplates.get_multi_genre_query(genre1, genre2)
            
            # Find movies with both genres
            genre1_movies = set(m["id"] for m in self.genre_index[genre1])
            genre2_movies = set(m["id"] for m in self.genre_index[genre2])
            common_ids = genre1_movies & genre2_movies
            
            if common_ids:
                common_movies = [m for m in self.movies_list if m["id"] in common_ids]
                selected_movies = self._select_random_movies(common_movies, count=random.randint(3, 5))
                
                response = self.response_generator.generate_genre_response(
                    f"{genre1.lower()}-{genre2.lower()}", selected_movies
                )
                
                examples.append({
                    "instruction": query,
                    "input": "",
                    "output": response,
                })
        
        logger.info(f"Generated {len(examples)} genre examples")
        return examples
    
    def generate_runtime_examples(self, num_examples: int) -> List[Dict]:
        """Generate runtime-based query examples."""
        logger.info(f"Generating {num_examples} runtime examples...")
        examples = []
        
        short_count = num_examples // 2
        long_count = num_examples - short_count
        
        # Short runtime queries
        for _ in range(short_count):
            duration = random.choice(DURATION_OPTIONS_SHORT)
            query = QueryTemplates.get_runtime_short_query(duration)
            
            # Parse duration to minutes
            if "90" in duration or "1.5" in duration or "hour and a half" in duration:
                max_runtime = 90
            elif "100" in duration:
                max_runtime = 100
            else:  # 2 hours
                max_runtime = 120
            
            # Find movies under this runtime
            candidates = [m for m in self.movies_list if m.get("runtime", 0) > 0 and m.get("runtime", 0) <= max_runtime]
            selected_movies = self._select_random_movies(candidates, count=random.randint(3, 5))
            
            response = self.response_generator.generate_runtime_response(f"under {duration}", selected_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        # Long runtime queries
        for _ in range(long_count):
            duration = random.choice(DURATION_OPTIONS_LONG)
            query = QueryTemplates.get_runtime_long_query(duration)
            
            # Parse duration to minutes
            if "2.5" in duration or "150" in duration:
                min_runtime = 150
            else:  # 3 hours / 180 minutes
                min_runtime = 180
            
            # Find movies over this runtime
            candidates = [m for m in self.movies_list if m.get("runtime", 0) >= min_runtime]
            selected_movies = self._select_random_movies(candidates, count=random.randint(3, 5))
            
            response = self.response_generator.generate_runtime_response(f"over {duration}", selected_movies)
            
            examples.append({
                "instruction": query,
                "input": "",
                "output": response,
            })
        
        logger.info(f"Generated {len(examples)} runtime examples")
        return examples
    
    def generate_mood_examples(self, num_examples: int) -> List[Dict]:
        """Generate mood-based query examples."""
        logger.info(f"Generating {num_examples} mood examples...")
        examples = []
        
        for _ in range(num_examples):
            mood = random.choice(MOOD_DESCRIPTORS)
            query = QueryTemplates.get_mood_query(mood)
            
            # Map mood descriptor to mood tags
            mood_tag_map = {
                "uplifting": "uplifting",
                "dark and gritty": "dark",
                "feel-good": "uplifting",
                "mind-bending": "mind_bending",
                "emotional": "emotional",
                "intense": "intense",
                "lighthearted": "lighthearted",
                "thought-provoking": "thought_provoking",
                "inspiring": "uplifting",
                "suspenseful": "intense",
                "heartwarming": "emotional",
                "disturbing": "dark",
                "whimsical": "lighthearted",
                "tense": "intense",
                "cheerful": "lighthearted",
            }
            
            mood_tag = mood_tag_map.get(mood, mood.replace("-", "_").replace(" ", "_"))
            
            # Find movies with this mood
            candidates = self.mood_index.get(mood_tag, [])
            
            if not candidates:
                # Fallback: search in all movies
                candidates = [m for m in self.movies_list if mood_tag in m.get("mood_tags", [])]
            
            if candidates:
                selected_movies = self._select_random_movies(candidates, count=random.randint(3, 5))
                response = self.response_generator.generate_mood_response(mood, selected_movies)
                
                examples.append({
                    "instruction": query,
                    "input": "",
                    "output": response,
                })
        
        logger.info(f"Generated {len(examples)} mood examples")
        return examples
    
    def generate_complex_examples(self, num_examples: int) -> List[Dict]:
        """Generate complex multi-criteria query examples."""
        logger.info(f"Generating {num_examples} complex examples...")
        examples = []
        
        for _ in range(num_examples):
            # Randomly select criteria to combine
            criteria = {}
            
            # Genre (always include)
            criteria["genre"] = random.choice(list(self.genre_index.keys())).lower()
            
            # Randomly add other criteria
            if random.random() > 0.5:
                criteria["mood"] = random.choice(MOOD_DESCRIPTORS)
            
            if random.random() > 0.5:
                criteria["duration"] = random.choice(DURATION_OPTIONS_SHORT + DURATION_OPTIONS_LONG)
            
            if random.random() > 0.5:
                criteria["decade"] = random.choice(["70", "80", "90", "2000", "2010"])
            
            if random.random() > 0.5:
                criteria["style"] = random.choice(STYLE_DESCRIPTORS)
            
            if random.random() > 0.3 and self.actor_index:
                criteria["actor"] = random.choice(list(self.actor_index.keys()))
            
            # Generate query
            query = QueryTemplates.get_complex_query(criteria)
            
            # Find matching movies (start with genre, then filter)
            candidates = self.genre_index.get(criteria["genre"].title(), self.movies_list)
            
            # Apply filters
            if "mood" in criteria:
                mood_tag = criteria["mood"].replace("-", "_").replace(" ", "_")
                candidates = [m for m in candidates if mood_tag in m.get("mood_tags", [])]
            
            if "decade" in criteria:
                decade = int(criteria["decade"])
                candidates = [m for m in candidates if m.get("decade") == decade]
            
            if "actor" in criteria:
                actor_name = criteria["actor"]
                candidates = [m for m in candidates if actor_name in m.get("cast_names", [])]
            
            if candidates:
                selected_movies = self._select_random_movies(candidates, count=random.randint(3, 5))
                response = self.response_generator.generate_complex_response(selected_movies)
                
                examples.append({
                    "instruction": query,
                    "input": "",
                    "output": response,
                })
        
        logger.info(f"Generated {len(examples)} complex examples")
        return examples
    
    def build_dataset(self) -> List[Dict]:
        """Build complete dataset with all query types."""
        logger.info("=== Building Complete Dataset ===")
        
        total_examples = DATASET_PARAMS["total_examples"]
        distribution = DATASET_PARAMS["query_type_distribution"]
        
        # Calculate number of examples per type
        num_similarity = int(total_examples * distribution["similarity"])
        num_cast_crew = int(total_examples * distribution["cast_crew"])
        num_genre = int(total_examples * distribution["genre"])
        num_runtime = int(total_examples * distribution["runtime"])
        num_mood = int(total_examples * distribution["mood"])
        num_complex = int(total_examples * distribution["complex"])
        
        # Split cast_crew between actors and directors
        num_actor = num_cast_crew // 2
        num_director = num_cast_crew - num_actor
        
        # Generate examples
        all_examples = []
        
        all_examples.extend(self.generate_similarity_examples(num_similarity))
        all_examples.extend(self.generate_actor_examples(num_actor))
        all_examples.extend(self.generate_director_examples(num_director))
        all_examples.extend(self.generate_genre_examples(num_genre))
        all_examples.extend(self.generate_runtime_examples(num_runtime))
        all_examples.extend(self.generate_mood_examples(num_mood))
        all_examples.extend(self.generate_complex_examples(num_complex))
        
        # Shuffle
        random.shuffle(all_examples)
        
        logger.info(f"=== Dataset Built: {len(all_examples)} total examples ===")
        return all_examples
    
    def split_dataset(self, examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets."""
        train_split = DATASET_PARAMS["train_split"]
        val_split = DATASET_PARAMS["val_split"]
        
        total = len(examples)
        train_size = int(total * train_split)
        val_size = int(total * val_split)
        
        train_data = examples[:train_size]
        val_data = examples[train_size:train_size + val_size]
        test_data = examples[train_size + val_size:]
        
        logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_dataset(self, examples: List[Dict], filepath: str):
        """Save dataset to JSONL file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {filepath}")


def main():
    """Main function to build and save dataset."""
    logger.info("=== Dataset Generation Started ===")
    
    # Load processed movie features
    logger.info(f"Loading movie features from {DATA_PATHS['processed_features']}...")
    with open(DATA_PATHS['processed_features'], 'r', encoding='utf-8') as f:
        movies_data = json.load(f)
    
    # Convert string keys to int if needed
    if movies_data and isinstance(list(movies_data.keys())[0], str):
        movies_data = {int(k): v for k, v in movies_data.items()}
    
    # Initialize dataset builder
    builder = DatasetBuilder(movies_data)
    
    # Build dataset
    all_examples = builder.build_dataset()
    
    # Split dataset
    train_data, val_data, test_data = builder.split_dataset(all_examples)
    
    # Save datasets
    builder.save_dataset(train_data, DATA_PATHS["train_dataset"])
    builder.save_dataset(val_data, DATA_PATHS["val_dataset"])
    builder.save_dataset(test_data, DATA_PATHS["test_dataset"])
    
    logger.info("=== Dataset Generation Completed ===")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()
