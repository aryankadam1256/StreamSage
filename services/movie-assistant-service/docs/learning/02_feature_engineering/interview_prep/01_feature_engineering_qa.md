# Interview Prep: Feature Engineering

## Common Questions & Answers

### Q1: What is Feature Engineering and why is it important?
**Answer:** Feature engineering is the process of transforming raw data into features that better represent the underlying problem to predictive models.
**In this project:** We transformed raw API data (nested JSON) into structured features like `mood_tags` and `popularity_score`. This was important because raw data often lacks the semantic meaning needed for user queries (e.g., users ask for "dark" movies, but the API only gives "Horror" genre).

### Q2: How did you handle categorical data?
**Answer:** I used a technique called **binning** or **categorization**.
**Example:** For movie runtime, instead of using the raw minute count (which is too granular), I grouped them into categories like `short` (<90m), `medium`, and `epic` (>150m). This maps better to how users naturally describe movie length.

### Q3: How did you calculate the "Popularity Score"? Why not just use the API's popularity?
**Answer:** The API's popularity metric can be volatile and hard to interpret (ranging from 0 to thousands).
**My Approach:** I created a composite score (0 to 1) using a weighted average:
- **Vote Average (40%):** Reflects quality.
- **Vote Count (30%):** Reflects reliability (using log scaling to dampen the effect of massive outliers).
- **TMDB Popularity (30%):** Reflects current trending status.
This ensures a movie with a 10.0 rating but only 1 vote doesn't rank higher than a classic with an 8.5 rating and 100,000 votes.

### Q4: How did you extract "Mood" from the data?
**Answer:** Since "mood" isn't a standard field, I implemented a rule-based inference system:
1.  **Keyword Matching:** Scanned the overview and keywords for specific terms (e.g., "shadow" -> "dark").
2.  **Genre Mapping:** Mapped genres to moods (e.g., "Thriller" -> "intense").
This allowed me to enrich the dataset with semantic tags that enable more natural language queries.

### Q5: What challenges did you face with text data?
**Answer:** The main challenge was unstructured text in the `overview`.
**Solution:** I combined all text sources (overview, keywords, genres) into a single lowercased string for efficient keyword searching. This simple "bag of words" approach was sufficient for tag extraction without needing complex NLP models at this stage.

### Q6: How would you improve this if you had more time?
**Answer:**
1.  **Embeddings:** Instead of keyword matching, I would use an embedding model (like BERT) to generate vector representations of the overview for semantic similarity search.
2.  **Sentiment Analysis:** Use NLP to determine if a movie is "uplifting" or "depressing" based on the review text, rather than just keywords.
3.  **User Personalization:** Incorporate user interaction data to weight features differently for different users.
