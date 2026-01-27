# Interview Prep: API Integration & Data Collection

## Core Concepts

### Q1: How do you handle API Rate Limits in a production application?
**Answer:**
"Rate limiting is crucial to respect the provider's constraints and avoid being banned. In this project, I implemented a **client-side throttling** mechanism.
-   I used a 'Token Bucket' style approach (simplified as a fixed delay) where I enforced a 250ms sleep before every request.
-   For more robust systems, I would implement **Exponential Backoff**: if I receive a `429 Too Many Requests` error, I wait X seconds, then 2X, then 4X, before retrying.
-   In a distributed system (backend), I would use a queue (like Redis or RabbitMQ) to decouple the request generation from the request execution, ensuring the consumer only processes jobs at the allowed rate."

### Q2: Why did you choose a Browser-based approach over Python for data collection?
**Answer:**
"It was a pragmatic decision driven by environment constraints.
-   **The Problem**: The Python environment on the development machine had persistent network configuration issues (likely firewall or proxy related) that blocked access to `api.themoviedb.org`.
-   **The Solution**: Since the browser had working internet access, I shifted the runtime environment to the browser.
-   **Trade-offs**: Browser JS has limitations (CORS, file system access), but it allowed us to bypass the network blocker immediately. I handled the file system limitation by using `localStorage` for checkpoints and `Blob` downloads for the final dataset."

### Q3: What is CORS and how did it affect your implementation?
**Answer:**
"CORS (Cross-Origin Resource Sharing) is a browser security feature that prevents a script on `site-a.com` from calling an API on `site-b.com` unless `site-b` explicitly allows it.
-   Since I was running a local HTML file (`file://`), my 'origin' was null or local.
-   Fortunately, the TMDB API is public-facing and sets the header `Access-Control-Allow-Origin: *`, which permits requests from any origin.
-   If they hadn't, I would have had to set up a local proxy server (e.g., using Node.js or Nginx) to forward requests, adding the necessary headers."

### Q4: How do you handle data persistence in a browser app?
**Answer:**
"For small amounts of data, `localStorage` or `sessionStorage` is fine (synchronous, string-only, ~5MB limit).
-   For this project, our checkpoint data was growing large. If it exceeded 5MB, `localStorage` would throw a QuotaExceeded error.
-   **Better Approach**: For larger datasets, I would use **IndexedDB**, which is asynchronous, supports structured data (not just strings), and has much higher storage limits (hundreds of MBs).
-   **Final Export**: I used the `Blob` API and `URL.createObjectURL` to generate a downloadable file client-side, effectively 'saving' the data to the user's disk."

### Q5: What is the difference between `fetch` and `axios`?
**Answer:**
"Both are used for HTTP requests, but:
-   **`fetch`**: Native to modern browsers. It's built-in, lightweight, but requires more boilerplate (e.g., you have to manually check `response.ok` and call `.json()`). It also doesn't reject promises on HTTP error status codes (like 404).
-   **`axios`**: A 3rd party library. It has better defaults (automatically transforms JSON, rejects on 4xx/5xx errors), supports request/response interceptors (great for adding auth tokens globally), and works in both Browser and Node.js environments.
-   I used `fetch` here to keep the tool dependency-free (zero setup)."

---

## Scenario Questions

### Q: You need to scrape 1 million pages. Your IP gets banned after 1000 requests. What do you do?
**Answer:**
"I would implement **IP Rotation**.
1.  Use a proxy service (like Bright Data or ScraperAPI) that routes requests through a pool of residential IPs.
2.  Rotate User-Agent headers to mimic different browsers.
3.  Slow down the request rate (be more 'polite')."

### Q: The API data is messy (missing fields, wrong types). How do you handle it?
**Answer:**
"I would implement a **Validation & Cleaning Layer** (schema validation).
-   In Python, I'd use **Pydantic** models to define the expected structure. If a field is missing, Pydantic can either set a default value or flag the record as invalid.
-   In this project, I performed basic checks (e.g., `if (!movie.details) continue`) to ensure data integrity before adding it to the dataset."
