# Execution Flow: Data Transformation

This document visualizes how data flows through our Feature Engineering pipeline.

## The Pipeline

```mermaid
graph TD
    A[Raw Data (movies.json)] --> B(FeatureEngineer Class)
    B --> C{Process Movie}
    
    C --> D[Extract Metadata]
    C --> E[Calculate Scores]
    C --> F[Infer Tags]
    C --> G[Flatten Structures]
    
    D --> H(Final Feature Set)
    E --> H
    F --> H
    G --> H
    
    H --> I[Processed Data (movie_features.json)]
```

## Step-by-Step Transformation

### 1. Input: Raw TMDB Data
The input is a complex, nested JSON object straight from the API.
```json
"550": {
    "details": {
        "title": "Fight Club",
        "vote_average": 8.4,
        "vote_count": 24000,
        "genres": [{"id": 18, "name": "Drama"}]
    },
    "keywords": {
        "keywords": [{"id": 825, "name": "support group"}]
    }
}
```

### 2. Processing: The `FeatureEngineer`
The engine takes this object and applies logic:

1.  **Normalization:**
    - `vote_average` (8.4) -> `0.84`
    - `vote_count` (24000) -> `0.87` (log scale)
    - **Result:** `popularity_score` = **0.855**

2.  **Tag Inference:**
    - Genre "Drama" -> Mood "emotional"
    - Keyword "support group" + Overview text -> Mood "dark"
    - **Result:** `mood_tags` = `["emotional", "dark"]`

3.  **Flattening:**
    - `genres` list of objects -> `["Drama"]`
    - `keywords` list of objects -> `["support group"]`

### 3. Output: Processed Features
The final output is a clean, flat object ready for the Dataset Builder.
```json
{
    "id": 550,
    "title": "Fight Club",
    "popularity_score": 0.855,
    "mood_tags": ["emotional", "dark"],
    "genres": ["Drama"],
    "keywords": ["support group"]
}
```

## Why This Matters
This transformation is crucial because:
1.  **Efficiency:** We discard unused data (like poster paths, production companies) to save memory.
2.  **Usability:** Flat lists (`["Drama"]`) are easier to filter than lists of objects.
3.  **Intelligence:** The `popularity_score` and `mood_tags` add value that wasn't in the raw data, allowing for smarter queries.
