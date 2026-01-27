# Execution Flow: The Fine-Tuning Pipeline

This document visualizes how we transform a base model into our specialized Movie Assistant.

## The Training Pipeline

```mermaid
graph TD
    A[Base Model (Llama 3 8B)] -->|Quantize 4-bit| B(Frozen Base Model)
    C[Dataset (train.jsonl)] -->|Format| D(Alpaca Prompts)
    
    B --> E{Training Loop}
    D --> E
    
    E -->|Forward Pass| F[Calculate Loss]
    F -->|Backprop| G[Update LoRA Adapters]
    
    G -->|Repeat 60 steps| E
    
    E -->|Finish| H[Saved Adapters (lora_model)]
    
    H -->|Merge at Runtime| I[Final Movie Assistant]
    B -->|Merge at Runtime| I
```

## Step-by-Step Breakdown

### 1. Initialization
*   **Base Model:** We download Llama 3 (16GB).
*   **Quantization:** We compress it to 4-bit (5.5GB) so it fits on the GPU.
*   **Freezing:** We lock all 8 billion parameters. They will NOT change.

### 2. Adapter Injection
*   We add tiny new matrices (LoRA Adapters) next to the frozen layers.
*   These are the *only* things we will train.
*   Total trainable parameters: ~40 Million (0.5% of the total).

### 3. The Training Loop
For each batch of data (User Query + Ideal Answer):
1.  **Prediction:** The model tries to guess the next word.
2.  **Loss Calculation:** We compare its guess to our `train.jsonl` answer.
3.  **Update:** We tweak the **LoRA Adapters** to make the guess better next time.
4.  **Repeat:** We do this for 60 steps (covering our dataset).

### 4. The Result
*   We don't save the whole 8B model again.
*   We only save the **Adapters** (~100MB).
*   **At Runtime:** We load the Base Model + Adapters to get the full intelligence.

## Why this works
This "Split Brain" approach (Frozen Base + Trainable Adapters) allows us to keep the general intelligence of Llama 3 (English, Logic, World Knowledge) while injecting our specific style (Movie Formatting, Enthusiasm) without catastrophic forgetting.
