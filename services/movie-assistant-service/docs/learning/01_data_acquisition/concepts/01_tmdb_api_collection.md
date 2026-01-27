# Data Acquisition Concepts: TMDB API & Browser-Based Collection

## 1. The TMDB API (The Movie Database)

### What is it?
The TMDB API is a RESTful web service that provides access to a vast database of movies, TV shows, actors, and images. It is the primary data source for StreamSage.

### Key Endpoints Used
We utilized several specific endpoints to build our dataset:

1.  **`GET /movie/popular`**:
    -   **Purpose**: To get a list of currently popular movies.
    -   **Why**: Ensures our dataset includes movies that users are likely to ask about.
    -   **Parameters**: `page` (pagination), `language` (en-US).

2.  **`GET /discover/movie`**:
    -   **Purpose**: To find movies based on specific criteria (genres).
    -   **Why**: Ensures diversity in our dataset. We collected top movies for every major genre (Action, Comedy, Horror, etc.).
    -   **Parameters**: `with_genres`, `sort_by` (popularity.desc), `vote_count.gte` (quality filter).

3.  **`GET /movie/{movie_id}`**:
    -   **Purpose**: To get detailed metadata for a specific movie.
    -   **Data Points**: Title, overview (plot), release date, runtime, vote average, genres, spoken languages.

4.  **`GET /movie/{movie_id}/credits`**:
    -   **Purpose**: To get cast and crew information.
    -   **Data Points**: Director, top cast members.

5.  **`GET /movie/{movie_id}/keywords`**:
    -   **Purpose**: To get plot keywords.
    -   **Why**: Crucial for semantic search and understanding the "vibe" of a movie (e.g., "time travel", "dystopia").

6.  **`GET /movie/{movie_id}/recommendations`**:
    -   **Purpose**: To get a list of similar movies recommended by TMDB.
    -   **Why**: Useful for building a graph of related movies or validating our own recommendation model.

---

## 2. Rate Limiting & API Etiquette

### The Challenge
APIs are not infinite resources. Providers like TMDB impose **rate limits** to prevent abuse and ensure server stability.
-   **TMDB Limit**: Generally around 40-50 requests per 10 seconds (varies by key type), or roughly 4 requests per second.

### Our Strategy: Client-Side Throttling
To stay safe and avoid `429 Too Many Requests` errors, we implemented a **fixed delay** between requests.

-   **Implementation**: A `sleep` function (using `setTimeout` in JS) that pauses execution for **250ms** (0.25 seconds) before every API call.
-   **Math**: $1 / 0.25s = 4$ requests/second. This is within the safe zone.

```javascript
// The "Sleep" mechanism
await new Promise(resolve => setTimeout(resolve, 250));
```

---

## 3. The Browser-Based Approach (The "Why")

### The Problem: Network Layer Issues
Initially, we attempted to collect data using a Python script (`requests` library). However, we encountered persistent **Network Unreachable** errors.
-   **Diagnosis**: The Python environment on the local machine likely had firewall restrictions, proxy misconfigurations, or DNS issues preventing it from reaching `api.themoviedb.org`.
-   **Constraint**: Debugging low-level network configs on a Windows machine can be time-consuming and invasive.

### The Solution: Browser as the Client
Web browsers (Chrome, Edge) often have different network paths and proxy settings than command-line tools. Since the user could access TMDB in their browser, we moved the code *to* the browser.

### Advantages
1.  **Bypassed Network Blocks**: Leveraged the browser's working internet connection.
2.  **Visual Feedback**: Real-time progress bars and logs are easier to build in HTML/CSS than in a terminal.
3.  **Zero Setup**: No need to install Python dependencies (`pip install requests`) if we just run a `.html` file.

### Technical Challenges & Solutions
1.  **CORS (Cross-Origin Resource Sharing)**:
    -   *Issue*: Browsers block requests from one domain (your local file) to another (TMDB) unless the server explicitly allows it.
    -   *Luck*: TMDB's API is CORS-friendly and allows requests from any origin (`Access-Control-Allow-Origin: *`), making this approach viable.

2.  **Persistence (Saving Data)**:
    -   *Issue*: Browsers can't write directly to the user's hard drive (security risk).
    -   *Solution*: We used **`localStorage`** to save "checkpoints" (progress) and **`Blob` objects** to generate a downloadable JSON file at the end.

---

## 4. Checkpointing (Fault Tolerance)

### Why it matters
Downloading 6,000 movies involves ~24,000 API calls. At 4 calls/second, this takes ~1.5 hours.
-   **Risk**: If the browser crashes, internet disconnects, or power fails after 80 minutes, you lose everything.

### Implementation
We implemented a **Checkpoint System**:
1.  **State Tracking**: We keep a list of `movieIds` collected and a dictionary of `completeMovies`.
2.  **Periodic Save**: Every 100 movies, we serialize this state to JSON and save it to the browser's `localStorage`.
3.  **Resume Capability**: On page load, the script checks `localStorage`. If data exists, it loads it into memory, allowing the user to continue exactly where they left off.

```javascript
// Concept
if (moviesCollected % 100 === 0) {
    localStorage.setItem('checkpoint', JSON.stringify(currentState));
}
```
