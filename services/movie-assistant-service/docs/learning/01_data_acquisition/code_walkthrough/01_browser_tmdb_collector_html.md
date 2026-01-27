# Code Walkthrough: Browser-Based TMDB Collector

## Overview
The `browser_tmdb_collector.html` is a self-contained Single Page Application (SPA) that runs entirely in the browser. It orchestrates the fetching of thousands of movie records from the TMDB API.

**File Location:** `services/movie-assistant-service/browser_tmdb_collector.html`

---

## 1. HTML Structure (The Skeleton)

The HTML provides the user interface. It's divided into logical sections:

-   **Status Card (`.status-card`)**: Displays high-level info like "Status" (Running/Paused), "Current Stage" (Popular/Genre), and "Elapsed Time".
-   **Progress Bars (`.progress-section`)**:
    -   *Overall Progress*: Tracks the total journey (Popular + Genres + Details).
    -   *Stage Progress*: Tracks the current specific task (e.g., "Page 5 of 250").
-   **Stats Grid (`.stats-grid`)**: Real-time counters for "Movies Collected", "API Calls", and "Errors".
-   **Controls (`.button-group`)**: Buttons to Start, Pause, and Download.
-   **Log Console (`.log-container`)**: A scrolling div that shows detailed event logs (successes, errors).

## 2. JavaScript Logic (The Brain)

### Configuration
```javascript
const API_KEY = '...'; // Your TMDB Key
const BASE_URL = 'https://api.themoviedb.org/3';
const RATE_LIMIT_DELAY = 250; // 250ms = 0.25 seconds
```
We define constants at the top for easy tuning. `RATE_LIMIT_DELAY` is the most critical one for API safety.

### State Management
We use a global `state` object to track everything. This makes it easy to save/load progress.
```javascript
let state = {
    isRunning: false,
    movieIds: new Set(),      // Unique Set to avoid duplicates
    completeMovies: {},       // The final data store
    // ... counters and flags
};
```

### The `makeRequest` Function (The Engine)
This is the core wrapper for `fetch`. It handles two things: **Throttling** and **Error Handling**.

```javascript
async function makeRequest(endpoint, params = {}) {
    // 1. Construct URL with API Key
    const url = new URL(`${BASE_URL}/${endpoint}`);
    // ... append params ...

    try {
        // 2. RATE LIMITING: Wait before fetching
        await new Promise(resolve => setTimeout(resolve, RATE_LIMIT_DELAY));

        // 3. Execute Fetch
        const response = await fetch(url);

        // 4. Handle HTTP Errors (like 404 or 500)
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        return await response.json();
    } catch (error) {
        // 5. Graceful Failure: Log and return null, don't crash
        log(`Error: ${error.message}`, 'error');
        return null;
    }
}
```

### Collection Stages

The collection is broken down into 3 sequential stages:

#### Stage 1: `collectPopularMovies()`
-   Loops from page 1 to `numPopularPages` (250).
-   Calls `/movie/popular`.
-   Adds IDs to `state.movieIds`.

#### Stage 2: `collectGenreMovies()`
-   Iterates through our defined list of genres (Action, Comedy, etc.).
-   For each genre, loops through `numGenrePages` (25).
-   Calls `/discover/movie` with `with_genres` and `sort_by=popularity.desc`.
-   Ensures we get the "best" movies of every type, not just the globally popular ones.

#### Stage 3: `collectCompleteMovieData()`
-   Iterates through the unique `state.movieIds` collected in stages 1 & 2.
-   For each ID, makes **4 parallel-ish calls** (sequentially awaited in our code for safety):
    1.  Details (`/movie/{id}`)
    2.  Credits (`/movie/{id}/credits`)
    3.  Keywords (`/movie/{id}/keywords`)
    4.  Recommendations (`/movie/{id}/recommendations`)
-   Combines this data into a single object and saves it to `state.completeMovies`.

### Checkpointing (The Safety Net)
```javascript
if (state.stageProgress.current % 100 === 0) {
    saveCheckpoint();
}
```
Every 100 movies during Stage 3, we call `saveCheckpoint()`.
-   **`saveCheckpoint()`**: `JSON.stringify(state)` -> `localStorage`.
-   **`loadCheckpoint()`**: `localStorage` -> `JSON.parse()` -> `state`.
-   This ensures that if you refresh the page, the `window.onload` event can restore your progress.

### Data Export
Since we can't write to a file directly, we use a **Blob** download.
```javascript
function downloadData() {
    // Convert JS Object to JSON string
    const dataStr = JSON.stringify(state.completeMovies, null, 2);
    // Create a "virtual file" in memory
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    // Create a temporary link to it
    const url = URL.createObjectURL(dataBlob);
    // Programmatically click the link
    link.click();
}
```
