# Code Walkthrough: `feature_engineering.py`

This document provides a detailed explanation of the `FeatureEngineer` class and how it transforms raw data.

## Class Structure

The `FeatureEngineer` class is the core of this module. It is initialized with configuration parameters from `config.py`.

```python
class FeatureEngineer:
    def __init__(self):
        self.runtime_categories = FEATURE_PARAMS["runtime_categories"]
        self.mood_keywords = FEATURE_PARAMS["mood_keywords"]
        self.style_keywords = FEATURE_PARAMS["style_keywords"]
```

## Key Methods

### 1. `extract_runtime_category(self, runtime)`
**Goal:** Convert minutes into a category (short, medium, long, epic).
**Logic:**
- Iterates through the `runtime_categories` dictionary.
- Checks if the movie's runtime falls within a specific range.
- Returns "epic" if it exceeds all defined ranges.

### 2. `extract_mood_tags(self, keywords, genres, overview)`
**Goal:** Infer the mood of the movie.
**Logic:**
1.  **Combine Text:** Concatenates keywords, genres, and the overview into a single text blob.
2.  **Keyword Matching:** Checks if any defined `mood_keywords` (e.g., "sad", "tearjerker") appear in the text.
3.  **Genre Inference:** Uses a `genre_mood_map` to map genres to moods (e.g., Horror -> "dark").
4.  **Deduplication:** Uses a `set` to ensure unique tags.

### 3. `calculate_popularity_score(self, vote_average, vote_count, popularity)`
**Goal:** Create a balanced 0-1 score for ranking.
**Formula:**
```python
score = (0.4 * normalized_rating + 0.3 * normalized_votes + 0.3 * normalized_popularity)
```
- **Normalized Rating:** `vote_average / 10.0` (Simple 0-1 scale).
- **Normalized Votes:** `log10(vote_count + 1) / 5.0` (Logarithmic scale to handle massive vote differences).
- **Normalized Popularity:** `popularity / 1000.0` (Capped at 1.0).

### 4. `process_movie(self, movie_data)`
**Goal:** The master function that processes a single movie entry.
**Steps:**
1.  **Extract Metadata:** Pulls basic fields like title, overview, release date.
2.  **Run Extractors:** Calls all the helper methods (`extract_runtime_category`, `extract_mood_tags`, etc.).
3.  **Structure Data:** Creates a clean, flat dictionary `features` containing all processed data.
4.  **Enrichment:** If Wikipedia data is available (`wiki_enrichment`), it merges summaries and themes into the feature set.

## Execution Flow (`main` function)

1.  **Load Data:** Reads `data/raw/movies.json`.
2.  **Initialize:** Creates an instance of `FeatureEngineer`.
3.  **Process:** Calls `process_all_movies` to iterate through the dataset.
4.  **Save:** Writes the result to `data/processed/movie_features.json`.

## Example Output

Input (Raw):
```json
{
  "details": {
    "title": "Inception",
    "runtime": 148,
    "genres": [{"name": "Action"}, {"name": "Sci-Fi"}]
  }
}
```

Output (Processed):
```json
{
  "title": "Inception",
  "runtime": 148,
  "runtime_category": "long",
  "genres": ["Action", "Sci-Fi"],
  "mood_tags": ["mind_bending", "intense"],
  "popularity_score": 0.985
}
```
