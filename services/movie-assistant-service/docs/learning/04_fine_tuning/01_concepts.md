# Fine-Tuning: The "What, Why, and How"

You asked for a breakdown of exactly what we are doing in this phase. Here is the deep dive.

## 1. WHAT are we doing?
We are taking a **General Purpose LLM** (Llama 3) and turning it into a **Specialized Movie Expert**.

Think of Llama 3 like a brilliant college graduate. It knows a lot about everything (history, coding, poetry), but it doesn't know *exactly* how you want it to behave as a Movie Assistant.

**Fine-Tuning** is the process of giving it a specific job training course.

## 2. WHY are we doing it?
Why not just use the base model? Three reasons:

### A. Style & Tone
*   **Base Model:** Might answer "Here is a list: Matrix, Inception." (Boring)
*   **Our Goal:** "You absolutely have to watch **The Matrix**! It defines the genre..." (Enthusiastic)
We want a specific personality that is consistent every time.

### B. Formatting
*   **Base Model:** Might give a numbered list one time, and a paragraph the next.
*   **Our Goal:** We want **Bold Titles**, specific bullet points for "Why it fits", and a clear structure. Fine-tuning forces this consistency.

### C. Reasoning
We want the model to learn the *logic* of recommendation.
*   *Input:* "I liked Inception."
*   *Output:* "Since you liked the mind-bending aspect of Inception, you will like..."
We are teaching it to connect the user's taste to the recommendation explicitly.

## 3. HOW are we doing it? (The Technical Part)

Training a huge model (8 Billion parameters) usually requires massive supercomputers. How are we doing it on a free Google Colab GPU?

### The Secret Weapon: LoRA (Low-Rank Adaptation)
Imagine the LLM's brain is a massive encyclopedia.
*   **Full Fine-Tuning:** Rewriting the entire encyclopedia. (Too slow, too expensive).
*   **LoRA:** Sticking a small "Post-it Note" on the relevant pages.

We freeze the main model and only train a tiny "adapter" layer (the Post-it note) that sits on top. This adapter learns our specific style.
*   **Result:** We get 99% of the performance of full training, but it uses 100x less memory.

### The Tool: Unsloth
We are using a library called **Unsloth**.
*   It optimizes the math behind the training (backpropagation).
*   It makes the process **2x faster** and uses **70% less memory** than standard methods.
*   This is the only reason we can do this on a free tier!

## Summary
| Aspect | Description |
| :--- | :--- |
| **Input** | Our `train.jsonl` (3,000 examples of "Good Behavior") |
| **Process** | The model tries to predict our answers. We correct it when it's wrong. |
| **Output** | A small file (~100MB) called "Adapters" that we load on top of Llama 3. |
