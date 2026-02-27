"""
============================================================================
Preference Data Generator for DPO/GRPO Training
============================================================================

Generates preference pairs (chosen vs rejected) from the SFT dataset.
Used for alignment training with DPO or GRPO.

Strategy:
  - "Chosen" = High-quality recommendations (correct genre, diverse, detailed)
  - "Rejected" = Degraded versions (wrong genre, no details, repetitive, etc.)

This creates realistic failure modes that the model learns to avoid.
============================================================================
"""

import json
import random
import copy
from pathlib import Path
from typing import List, Dict, Tuple


class PreferenceDataGenerator:
    """Generate preference pairs for DPO/GRPO alignment training."""

    def __init__(self, movie_features_path: str, sft_dataset_path: str):
        self.movies = self._load_movies(movie_features_path)
        self.sft_data = self._load_sft_data(sft_dataset_path)

        # Build indexes for degradation strategies
        self.genre_index = {}
        self.movies_by_id = {}
        for movie in self.movies:
            mid = movie.get("id")
            if mid:
                self.movies_by_id[mid] = movie
            for genre in movie.get("genres", []):
                self.genre_index.setdefault(genre, []).append(movie)

        self.all_genres = list(self.genre_index.keys())

    def _load_movies(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_sft_data(self, path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    # ========================================================================
    # Rejection Strategies
    # ========================================================================

    def _reject_wrong_genre(self, example: Dict) -> str:
        """Recommend movies from wrong genre."""
        instruction = example["instruction"].lower()

        # Find what genre the user wants
        target_genre = None
        for genre in self.all_genres:
            if genre.lower() in instruction:
                target_genre = genre
                break

        # Pick movies from a DIFFERENT genre
        if target_genre:
            wrong_genres = [g for g in self.all_genres if g != target_genre]
        else:
            wrong_genres = self.all_genres

        wrong_genre = random.choice(wrong_genres)
        wrong_movies = random.sample(
            self.genre_index.get(wrong_genre, self.movies[:10]),
            min(3, len(self.genre_index.get(wrong_genre, self.movies[:10])))
        )

        response = f"Here are some great {wrong_genre} movies:\n\n"
        for i, movie in enumerate(wrong_movies, 1):
            title = movie.get("title", "Unknown")
            year = movie.get("release_year", "N/A")
            response += f"{i}. **{title}** ({year})\n"

        return response

    def _reject_too_few_recs(self, example: Dict) -> str:
        """Only give 1 recommendation instead of multiple."""
        output = example["output"]
        # Extract just the first recommendation
        lines = output.split("\n")
        short_response = []
        found_first = False
        for line in lines:
            short_response.append(line)
            if line.strip().startswith("1.") or line.strip().startswith("1 "):
                found_first = True
            elif found_first and (line.strip().startswith("2.") or line.strip().startswith("2 ")):
                break

        if len(short_response) > 2:
            return "\n".join(short_response[:4])
        return output[:150]  # Truncate heavily

    def _reject_no_details(self, example: Dict) -> str:
        """Remove all details, just list titles."""
        output = example["output"]
        # Extract just movie titles
        titles = []
        for line in output.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("**")):
                # Extract title portion only
                if "**" in line:
                    start = line.find("**") + 2
                    end = line.find("**", start)
                    if end > start:
                        titles.append(line[start:end])
                elif "." in line[:5]:
                    title_part = line.split(".", 1)[-1].strip()
                    # Remove everything after parenthesis
                    if "(" in title_part:
                        title_part = title_part[:title_part.index("(")].strip()
                    if title_part:
                        titles.append(title_part)

        if titles:
            response = "Here are some movies:\n"
            for i, title in enumerate(titles[:5], 1):
                response += f"{i}. {title}\n"
            return response

        return "I recommend some movies. Check them out."

    def _reject_repetitive(self, example: Dict) -> str:
        """Repeat the same information multiple times."""
        output = example["output"]
        # Take first 100 chars and repeat with slight variation
        snippet = output[:120]
        return f"{snippet}\n\nAlso, {snippet.lower()}\n\nTo summarize, {snippet.lower()}"

    def _reject_off_topic(self, example: Dict) -> str:
        """Ignore the query and give generic response."""
        generic_responses = [
            "Movies are a great form of entertainment. There are many genres to choose from including action, comedy, drama, and more. I suggest browsing popular streaming platforms to find something you like.",
            "I think you should watch whatever is popular right now. Check the trending section on your streaming service for the latest releases.",
            "There are thousands of movies released every year. It really depends on your personal taste. Try exploring different genres.",
            "Cinema has evolved significantly over the decades. From silent films to modern blockbusters, there is something for everyone.",
        ]
        return random.choice(generic_responses)

    def _reject_factually_wrong(self, example: Dict) -> str:
        """Introduce factual errors (wrong years, directors)."""
        output = example["output"]
        modified = output

        # Randomly modify years by shifting them
        import re
        years = re.findall(r'\((\d{4})\)', modified)
        for year in years:
            wrong_year = str(int(year) + random.choice([-15, -10, 10, 15, 20]))
            modified = modified.replace(f"({year})", f"({wrong_year})", 1)

        # Swap director names if present
        director_names = ["Christopher Nolan", "Steven Spielberg", "Martin Scorsese",
                          "Quentin Tarantino", "James Cameron", "Ridley Scott"]
        for name in director_names:
            if name in modified:
                wrong_name = random.choice([n for n in director_names if n != name])
                modified = modified.replace(name, wrong_name, 1)
                break

        return modified

    # ========================================================================
    # Main Generation
    # ========================================================================

    REJECTION_METHODS = [
        ("wrong_genre", "_reject_wrong_genre"),
        ("too_few_recs", "_reject_too_few_recs"),
        ("no_details", "_reject_no_details"),
        ("repetitive", "_reject_repetitive"),
        ("off_topic", "_reject_off_topic"),
        ("factually_wrong", "_reject_factually_wrong"),
    ]

    def generate_preference_pair(self, example: Dict) -> Dict:
        """Generate a single preference pair from an SFT example."""
        # Chosen = original high-quality SFT response
        chosen = example["output"]

        # Rejected = degraded version using random strategy
        strategy_name, method_name = random.choice(self.REJECTION_METHODS)
        method = getattr(self, method_name)
        rejected = method(example)

        return {
            "prompt": example["instruction"],
            "chosen": chosen,
            "rejected": rejected,
            "rejection_strategy": strategy_name,
        }

    def generate_dataset(
        self,
        num_pairs: int = 2000,
        train_split: float = 0.9,
        output_dir: str = "data/preference_data",
    ) -> Dict[str, int]:
        """Generate full preference dataset for DPO/GRPO training."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sample from SFT data (with replacement if needed)
        if num_pairs > len(self.sft_data):
            samples = random.choices(self.sft_data, k=num_pairs)
        else:
            samples = random.sample(self.sft_data, num_pairs)

        # Generate pairs
        pairs = []
        strategy_counts = {}

        for example in samples:
            pair = self.generate_preference_pair(example)

            # Quality check: chosen and rejected must be different
            if pair["chosen"].strip() != pair["rejected"].strip():
                pairs.append(pair)
                strategy = pair["rejection_strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Shuffle
        random.shuffle(pairs)

        # Split
        split_idx = int(len(pairs) * train_split)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        # Save
        train_path = output_path / "preference_train.jsonl"
        val_path = output_path / "preference_val.jsonl"

        for path, data in [(train_path, train_pairs), (val_path, val_pairs)]:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        stats = {
            "total_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "strategy_distribution": strategy_counts,
        }

        # Save stats
        with open(output_path / "preference_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Preference dataset generated:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Train: {stats['train_pairs']}")
        print(f"  Val: {stats['val_pairs']}")
        print(f"  Strategy distribution:")
        for strategy, count in sorted(strategy_counts.items()):
            print(f"    {strategy}: {count}")

        return stats


# ============================================================================
# GRPO Reward Functions
# ============================================================================

class RecommendationRewardFunctions:
    """Reward functions for GRPO training on movie recommendations."""

    @staticmethod
    def format_reward(response: str) -> float:
        """Reward for correct response format (numbered list with details)."""
        score = 0.0

        # Has numbered recommendations
        import re
        numbered = re.findall(r'^\d+\.', response, re.MULTILINE)
        if len(numbered) >= 2:
            score += 0.3
        if len(numbered) >= 3:
            score += 0.2

        # Has bold titles
        if "**" in response:
            score += 0.2

        # Has year in parentheses
        years = re.findall(r'\(\d{4}\)', response)
        if years:
            score += 0.15

        # Has opening line
        if response.strip() and not response.strip()[0].isdigit():
            score += 0.15

        return min(score, 1.0)

    @staticmethod
    def recommendation_count_reward(response: str) -> float:
        """Reward for having the right number of recommendations (3-5)."""
        import re
        numbered = re.findall(r'^\d+\.', response, re.MULTILINE)
        count = len(numbered)

        if count == 0:
            return 0.0
        elif count == 1:
            return 0.2
        elif count == 2:
            return 0.5
        elif 3 <= count <= 5:
            return 1.0
        elif count <= 7:
            return 0.7
        else:
            return 0.4  # Too many

    @staticmethod
    def diversity_reward(response: str) -> float:
        """Reward for diverse recommendations (different genres/eras)."""
        import re
        years = [int(y) for y in re.findall(r'\((\d{4})\)', response)]

        if len(years) < 2:
            return 0.3

        # Check year diversity (span of decades)
        year_range = max(years) - min(years)
        decade_diversity = min(year_range / 30.0, 1.0)  # Max reward at 30yr span

        # Check genre word diversity
        genre_words = {"action", "comedy", "drama", "thriller", "horror", "sci-fi",
                       "romance", "adventure", "fantasy", "mystery", "animation",
                       "crime", "documentary", "western", "musical"}
        found_genres = sum(1 for g in genre_words if g in response.lower())
        genre_diversity = min(found_genres / 3.0, 1.0)  # Max reward at 3 genres

        return 0.5 * decade_diversity + 0.5 * genre_diversity

    @staticmethod
    def relevance_reward(response: str, query: str) -> float:
        """Reward for recommendations matching the query intent."""
        query_lower = query.lower()
        response_lower = response.lower()
        score = 0.0

        # Check if genre mentioned in query appears in response
        genres = ["action", "comedy", "drama", "thriller", "horror", "sci-fi",
                  "romance", "adventure", "fantasy", "mystery", "animation"]
        for genre in genres:
            if genre in query_lower and genre in response_lower:
                score += 0.3
                break

        # Check if mood words match
        moods = ["dark", "uplifting", "intense", "lighthearted", "emotional",
                 "mind-bending", "thought-provoking", "funny", "scary"]
        for mood in moods:
            if mood in query_lower and mood in response_lower:
                score += 0.2
                break

        # Check if actor/director mentioned in query appears in response
        # Simple heuristic: capitalized multi-word names
        import re
        names = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', query)
        for name in names:
            if name.lower() in response_lower:
                score += 0.3
                break

        # Response length (not too short, not too long)
        word_count = len(response.split())
        if 50 <= word_count <= 300:
            score += 0.2
        elif 30 <= word_count <= 400:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def compute_combined_reward(
        response: str,
        query: str,
        weights: Dict[str, float] = None
    ) -> float:
        """Compute weighted combination of all reward signals."""
        if weights is None:
            weights = {
                "format": 0.2,
                "count": 0.2,
                "diversity": 0.3,
                "relevance": 0.3,
            }

        rf = RecommendationRewardFunctions
        rewards = {
            "format": rf.format_reward(response),
            "count": rf.recommendation_count_reward(response),
            "diversity": rf.diversity_reward(response),
            "relevance": rf.relevance_reward(response, query),
        }

        combined = sum(rewards[k] * weights[k] for k in weights)
        return combined, rewards


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate preference data for DPO/GRPO")
    parser.add_argument("--movies", default="data/processed/movie_features.json")
    parser.add_argument("--sft-data", default="data/datasets/train.jsonl")
    parser.add_argument("--output", default="data/preference_data")
    parser.add_argument("--num-pairs", type=int, default=2000)
    parser.add_argument("--train-split", type=float, default=0.9)
    args = parser.parse_args()

    generator = PreferenceDataGenerator(args.movies, args.sft_data)
    generator.generate_dataset(
        num_pairs=args.num_pairs,
        train_split=args.train_split,
        output_dir=args.output,
    )
