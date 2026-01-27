# Feature Engineering Concepts

## What is Feature Engineering?
Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. These features make machine learning algorithms work better.

In the context of our **Movie Assistant**, raw data comes from the TMDB API in a nested JSON format. We need to transform this into a structured, flat format that is easier for our LLM to understand and for us to query.

## Key Techniques Used

### 1. Categorization (Binning)
**Concept:** Converting continuous numerical data into discrete categories.
**Why:** It helps in grouping similar items. A user is more likely to ask for a "short movie" than a "movie under 90 minutes".
**Implementation:**
- **Runtime:** We convert raw minutes (e.g., 88, 102, 145) into categories:
    - `short`: < 90 mins
    - `medium`: 90-120 mins
    - `long`: 120-150 mins
    - `epic`: > 150 mins
- **Era:** We convert release years into eras:
    - `classic`: Pre-1980
    - `modern`: 1980-2010
    - `contemporary`: Post-2010

### 2. Keyword Extraction & Tagging
**Concept:** Deriving new tags based on the presence of specific words in text fields.
**Why:** Movies don't always come with "mood" tags. We need to infer them to answer queries like "I want a dark and intense movie".
**Implementation:**
- **Mood Extraction:** We scan the movie's overview, keywords, and genres. If we find words like "shadow", "fear", or "grim", we tag the movie as `dark`.
- **Style Extraction:** Similarly, we look for words indicating style, such as "surreal", "fast-paced", or "minimalist".

### 3. Normalization & Scoring
**Concept:** Adjusting values measured on different scales to a notionally common scale.
**Why:** Raw popularity numbers from TMDB can vary wildly (from 0.5 to 5000+). We need a balanced score to rank movies effectively.
**Implementation:**
- **Popularity Score:** We calculate a custom score (0.0 to 1.0) using a weighted formula:
    - 40% **Vote Average** (Quality)
    - 30% **Vote Count** (Reliability - log scaled)
    - 30% **TMDB Popularity** (Trending status)

### 4. Data Flattening & Structuring
**Concept:** Converting complex nested objects into simple lists or strings.
**Why:** LLMs and simple search algorithms prefer flat structures.
**Implementation:**
- **Cast & Crew:** Instead of a deep list of objects, we extract just the `Director` name and the top 10 `Cast` names as simple strings.
- **Genres:** We convert a list of dictionaries `[{'id': 1, 'name': 'Action'}]` into a simple list `['Action']`.

## The "Why" Behind Our Features

| Feature | User Query It Solves |
| :--- | :--- |
| `mood_tags` | "I'm in the mood for something *uplifting*." |
| `runtime_category` | "I only have an hour, give me something *short*." |
| `era_tag` | "Show me some *classic* 70s movies." |
| `popularity_score` | "What are the *best* movies of 2023?" |
| `similar_movies` | "I liked *Inception*, what else should I watch?" |

## Next Steps
Now that we understand the *concepts*, let's look at the *code* that implements them in `02_code_walkthrough.md`.
