# Model Performance Analysis & Stress Test Plan

## 1. Analysis of Current Performance

### ✅ Strengths (What's working)
1.  **Style & Formatting:** The model consistently uses the **Bold Title** + (Year) + Genre + Duration format. This proves the fine-tuning worked for *style*.
2.  **Tone:** The responses are enthusiastic ("Avatar is a perfect choice!", "You'll love these...").
3.  **Intent Recognition:** It correctly identifies actors (Cumberbatch, DiCaprio), genres (Romance), and similar movies (Interstellar -> Gravity).

### ⚠️ Weaknesses (What needs fixing)
1.  **Hallucination / Repetition:**
    *   *Benedict Cumberbatch query:* It repeated "The Hobbit: The Desolation of Smaug" 4 times!
    *   *Avengers query:* Repeated "The Avengers" 4 times.
    *   *Avatar query:* Hallucinated "Avatar 5 (2029)" (which doesn't exist yet) and repeated Avatar 2/3.
2.  **Ordering Logic:**
    *   *Spider-Man (Tobey Maguire):* It listed them in reverse order (3, 2, 1) instead of 1, 2, 3.
3.  **Knowledge Cutoff/Accuracy:**
    *   It seems to be "making up" the list when it doesn't have the RAG context. Remember, **we are testing it WITHOUT the Vector DB right now**, so it's relying purely on its internal memory (which is why it repeats things).

> [!NOTE]
> **Crucial Context:** The repetition happens because we are running "Inference Only" (Brain only) without the "Retrieval" (Book). The model is trying to fill the list from memory and getting stuck in loops. This will likely vanish when we connect it to the Vector DB (RAG), but we should still improve the model's diversity.

---

## 2. Stress Test Results (Analysis of User Logs)

### 🚨 Critical Issues Found
1.  **Severe Repetition:**
    *   *Query:* "I want a movie where the bad guy wins." -> Repeated "The Devil's Advocate" 4 times.
    *   *Query:* "Mess with my head" -> Alternated between "The Prestige" and "The Cell" repeatedly.
    *   **Cause:** The model is stuck in a local minimum. Without RAG context, it finds one "good" answer and keeps outputting it because it has high probability.

2.  **Context Failure:**
    *   *Query:* "I liked the first one. What about the sequel?"
    *   *Response:* It hallucinated a sequel to "The Uninvited" instead of looking at the previous chat context.
    *   **Cause:** Our simple `while` loop in the notebook does not pass the *history* of the conversation back to the model. It treats every query as brand new.

3.  **Hallucination:**
    *   *Query:* "Nudity and violence" -> Suggested "The Devil's Advocate" (which fits) but repeated it.
    *   *Query:* "Romantic comedy Oscar winner" -> Suggested "The Proposal" (Did NOT win an Oscar). It prioritized the "RomCom" part over the "Oscar" part.

### ✅ What Worked
1.  **Abstract Queries:**
    *   *Query:* "Warm hug on a rainy day" -> Suggested "The Princess Bride", "The Last Unicorn". This is **excellent** semantic understanding!
    *   *Query:* "Mess with my head" -> Suggested "The Prestige", "The Cell". Perfect genre matching.

2.  **Complex Constraints:**
    *   *Query:* "Sci-fi 90s under 2 hours, No Star Wars" -> It correctly gave Star Trek movies (technically fits "No Star Wars" but is very close). It respected the time constraint (1h 45m).

---

## 3. Recommended Fixes

### A. Fix Repetition (Immediate Code Change)
We need to force the model to be more diverse.
*   **Action:** Update the inference parameters in the notebook.
    *   `repetition_penalty`: Increase from `1.1` to **1.3**.
    *   `no_repeat_ngram_size`: Set to **2** (prevents repeating 2-word phrases).
    *   `temperature`: Increase to **0.9** (more creativity).

### B. Fix Context (Future Feature)
*   **Action:** The current notebook loop is stateless. To fix "What about the sequel?", we need to append the previous answer to the next prompt.
    *   *Plan:* We will implement this in the actual `main.py` backend, but for the notebook, we can update the loop to store `chat_history`.

### C. Fix Accuracy (RAG)
*   **Action:** The "Oscar" hallucination will be solved by RAG. The Vector DB will only return movies that *actually* have the "Oscar" tag in their metadata, so the LLM won't have to guess.
