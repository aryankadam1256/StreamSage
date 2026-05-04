# Prompt Engineering: Theory vs. Implementation

In this document, we explore how the final step of RAG (Generation) is handled. This bridges the gap between the retrieved facts and the LLM's final response.

---

## 1. Context Assembly

### 📚 The Theory
LLMs (like LLaMA 3 or Mistral) have a "Context Window" (e.g., 8,000 tokens). This is their short-term memory. In RAG, we must inject the facts retrieved from the database into this context window so the LLM has the necessary information to answer the question without hallucinating.

### 💻 Our Project Implementation (`main.py`)
After the Cross-Encoder ranks the top 5 movies, we extract their metadata and stitch them together into a structured string.

```python
# Concept from our code:
context_string = ""
for i, movie in enumerate(top_5_movies):
    context_string += f"[{i+1}] Title: {movie.title}\n"
    context_string += f"    Year: {movie.year}, Rating: {movie.rating}/10\n"
    context_string += f"    Genres: {movie.genres}\n"
    context_string += f"    Plot: {movie.description}\n\n"
```
This exact string is pasted into the prompt. The formatting (using numbers, clear labels like "Plot:") helps the LLM easily parse the information.

---

## 2. System Prompts & Hallucination Prevention

### 📚 The Theory
An LLM inherently wants to use its pre-trained memory to answer questions. If you ask it for a 2024 movie, it might invent one. **Prompt Engineering** is the practice of explicitly instructing the LLM to ignore its internal memory and strictly rely on the provided context.

### 💻 Our Project Implementation
In our codebase, we define a strict System Prompt. It generally follows this architecture:

1.  **Persona Assignment:** "You are an enthusiastic movie recommendation assistant." (This aligns with the style we trained it on).
2.  **Constraint Enforcement:** "Use ONLY the context provided below. If the context does not contain the answer, say 'I don't have enough information'." (This explicitly prevents hallucination).
3.  **Data Injection:** `[INSERT CONTEXT HERE]`
4.  **Task:** `[INSERT USER QUERY HERE]`

---

## 3. Metadata Filtering Before Prompting

### 📚 The Theory
Sometimes, it's inefficient to use Prompt Engineering to filter data. For example, if a user asks for "Movies released after 2010," you shouldn't rely on the LLM to read the dates and filter them. You should filter them at the database level.

### 💻 Our Project Implementation
In `main.py`, before we even build the prompt or search ChromaDB, we parse the user's intent:
```python
# From main.py
if request.min_year is not None:
    conditions.append({"year": {"$gte": request.min_year}})
```
We pass these conditions directly to ChromaDB's `$gte` (greater than or equal to) metadata filters. 
**Why this is brilliant:** This guarantees that *every single movie* injected into the LLM prompt is already mathematically guaranteed to be released after 2010. The LLM doesn't have to think; it just has to write a beautiful response based on guaranteed data.

---

## Interview Study Guide

*   **Q: How do you prevent your LLM from hallucinating fake movies?**
    *   **A:** I use strict System Prompting that forces the model to rely *only* on the context block. More importantly, I use hard metadata filtering at the database level (e.g., filtering by year or rating) *before* the data reaches the LLM, mathematically eliminating the possibility of it recommending a movie that violates the user's hard constraints.
*   **Q: What happens if the context window of the LLM gets full?**
    *   **A:** In my implementation, I strictly limit the retrieval to the top `K` results (default 5). By extracting only the necessary fields (Title, Plot, Genres) and dropping irrelevant data, I ensure the prompt stays well within LLaMA 3's 8K token limit, keeping latency low and avoiding context overflow.
