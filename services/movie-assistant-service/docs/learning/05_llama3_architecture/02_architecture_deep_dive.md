# Architecture Deep Dive: Every Component Explained

This document matches the architecture diagram component-by-component. For each block in the image, you'll understand **what it does**, **why it exists**, and **the math behind it**.

---

## Part 1: The Input Block

### 1.1 Tokenizer (Text → Tokens)

**What it does:** Converts raw text into integer IDs that the model understands.

**How it works:** LLaMA 3 uses **Byte-Pair Encoding (BPE)** with a vocabulary of **128,256 tokens**. BPE works by:
1. Starting with individual characters.
2. Repeatedly merging the most frequent pair of adjacent tokens into a new token.
3. After training, common words like "movie" are a single token, while rare words get split into subwords.

**Example:**
```
Input:  "Recommend sci-fi movies like Inception"
Tokens: ["Recommend", " sci", "-fi", " movies", " like", " In", "ception"]
IDs:    [67375, 14175, 12, 9523, 1093, 763, 13415]
```

**Why 128K vocabulary?** LLaMA 2 used only 32K tokens. The 4x larger vocabulary in LLaMA 3 means:
- Fewer tokens per sentence (more efficient)
- Better handling of code, non-English languages, and rare words
- Slightly larger embedding table, but the trade-off is worth it

**Special Tokens in LLaMA 3:**
| Token | ID | Purpose |
| :--- | :--- | :--- |
| `<\|begin_of_text\|>` | 128000 | Start of sequence |
| `<\|end_of_text\|>` | 128001 | End of sequence |
| `<\|start_header_id\|>` | 128006 | Start of role header |
| `<\|end_header_id\|>` | 128007 | End of role header |
| `<\|eot_id\|>` | 128009 | End of turn |

These special tokens are how the model knows where "system", "user", and "assistant" messages begin and end during our fine-tuning.

### 1.2 Token Embeddings (Tokens → Vectors)

**What it does:** Converts each integer token ID into a dense vector of 4,096 numbers.

**How it works:** It's a simple lookup table (matrix of shape `[128,256 × 4,096]`). Each row is a learned vector for one token.

```
Token ID 67375 ("Recommend") → [0.023, -0.145, 0.891, ..., 0.034]  (4,096 numbers)
```

**Intuition:** These 4,096 numbers encode the meaning of the token. Similar words (like "suggest" and "recommend") will have similar vectors. The model learned these during pre-training on 15T+ tokens.

**Memory:** 128,256 tokens × 4,096 dimensions × 2 bytes (FP16) = **~1 GB** just for the embedding table.

---

## Part 2: The Decoder Block (Repeated 32 Times)

This is the heart of the model. Each of the 32 blocks has the same structure but different learned weights. The diagram shows the inside of "Decoder Block 1" — blocks 2 through 32 are identical in structure.

### 2.1 RMS Norm (Pre-Normalization)

**Where in the diagram:** Appears before both the Attention block and the FFN block (two RMSNorm per decoder layer).

**What it does:** Normalizes the input to have consistent scale, preventing values from exploding or vanishing as they pass through 32 layers.

**Why RMSNorm instead of LayerNorm?**

Standard LayerNorm:
```
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β
```

RMSNorm (what LLaMA uses):
```
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = sqrt(mean(x²))
```

**Key differences:**
1. **No mean subtraction** — removes the re-centering step
2. **No bias (β)** — only has a scale parameter (γ)
3. **~10-15% faster** than LayerNorm with equivalent performance

**Why Pre-Norm?** The normalization happens *before* the attention/FFN, not after. This is called **Pre-Norm** architecture (vs. Post-Norm in the original Transformer). Pre-Norm makes training more stable, especially for deep models like 32 layers.

**Parameters per RMSNorm:** Just 4,096 (one γ per dimension). With 65 RMSNorm layers (2 per block × 32 + 1 final), that's only ~267K parameters total — negligible.

---

### 2.2 Grouped Query Attention (GQA) Block

This is the most important and complex component. Let's break it down exactly as shown in the diagram.

#### Step 1: Q, K, V Projections

**What the diagram shows:** Input flows into three separate linear layers producing Q (Query), K (Key), and V (Value).

**The math:**
```
Q = x · W_Q    (shape: [seq_len, 4096] × [4096, 4096] → [seq_len, 4096])
K = x · W_K    (shape: [seq_len, 4096] × [4096, 1024] → [seq_len, 1024])
V = x · W_V    (shape: [seq_len, 4096] × [4096, 1024] → [seq_len, 1024])
```

**Wait — why are K and V smaller?** This is the "Grouped" in GQA. Instead of 32 Key heads and 32 Value heads (standard Multi-Head Attention), LLaMA 3 uses only **8 KV heads**. Each KV head is shared by 4 Query heads.

**The analogy:** Imagine a classroom with 32 students (Query heads) and 8 textbooks (KV heads). Every 4 students share one textbook. They each ask different questions (different Q), but refer to the same information source (same K, V).

#### Why GQA?

| Attention Type | Q Heads | KV Heads | KV Cache Size | Quality |
| :--- | :--- | :--- | :--- | :--- |
| Multi-Head (MHA) | 32 | 32 | 100% (baseline) | Best |
| **Grouped Query (GQA)** | **32** | **8** | **25%** | **~Same as MHA** |
| Multi-Query (MQA) | 32 | 1 | 3% | Slightly worse |

GQA gives us **4x smaller KV cache** during inference with almost no quality loss. This is critical for:
- Serving longer sequences (8K context)
- Fitting in GPU memory during inference
- Faster generation speed

#### Step 2: Rotary Positional Encoding (RoPE)

**What the diagram shows:** The wavy (∿) symbol applied to Q and K (but NOT V).

**What it does:** Injects position information into the attention computation. Without RoPE, the model has no idea whether "movie" is the 1st word or the 500th word.

**How it works (intuition):** RoPE rotates the Q and K vectors by an angle that depends on their position in the sequence. When you compute the dot product Q·K (attention score), the rotation encodes the **relative distance** between two tokens.

**The math (simplified):**
```
For position m, dimension pair (2i, 2i+1):

  q'[2i]   = q[2i] · cos(mθᵢ) - q[2i+1] · sin(mθᵢ)
  q'[2i+1] = q[2i] · sin(mθᵢ) + q[2i+1] · cos(mθᵢ)

where θᵢ = 1 / 10000^(2i/d)
```

**Why rotation (not addition)?**
- Traditional positional encodings (sinusoidal, learned) are **added** to the input once.
- RoPE is **applied at every attention layer** to Q and K directly.
- The dot product `Q·K` naturally encodes **relative position** (distance between tokens), not absolute position.
- This makes RoPE more generalizable — it can extrapolate to sequence lengths longer than seen in training.

**Why only Q and K?** Because attention scores come from `Q·K`. The Value vectors just carry content — they don't need position information.

#### Step 3: KV Cache (Inference Only)

**What the diagram shows:** A "KV Cache" box labeled "(For Inferencing only)".

**What it does:** During autoregressive generation (one token at a time), we store all previously computed K and V vectors so we don't recompute them.

**Without KV Cache (naive):**
```
Token 1: Compute Q₁, K₁, V₁ → Attention
Token 2: Compute Q₁, K₁, V₁, Q₂, K₂, V₂ → Attention   (K₁, V₁ recomputed!)
Token 3: Compute Q₁, K₁, V₁, Q₂, K₂, V₂, Q₃, K₃, V₃   (everything recomputed!)
```
This is O(n²) per token — terrible.

**With KV Cache:**
```
Token 1: Compute Q₁, K₁, V₁ → Store K₁, V₁ in cache
Token 2: Compute Q₂, K₂, V₂ → Append K₂, V₂ to cache → Attend to full cache
Token 3: Compute Q₃, K₃, V₃ → Append K₃, V₃ to cache → Attend to full cache
```
Now each new token only computes **one** new Q, K, V — and reuses the cache. This is O(n) per token.

**KV Cache size for LLaMA 3 8B:**
```
Per token: 2 (K+V) × 8 (KV heads) × 128 (head dim) × 2 (bytes FP16) = 4 KB
Full 8K context: 4 KB × 8,192 × 32 layers = ~8 GB
```
Without GQA (32 KV heads instead of 8), this would be **~32 GB** — wouldn't even fit on most GPUs!

#### Step 4: Attention Computation

**What the diagram shows:** The "Attention Output" box above Q, K, V.

**The math:**
```
Attention(Q, K, V) = softmax(Q · K^T / √d_head) · V
```

Broken down:
1. **Score:** Multiply each Query with all Keys → `Q · K^T` gives a score matrix
2. **Scale:** Divide by `√128 = 11.31` to prevent softmax saturation
3. **Mask:** Apply causal mask (set future positions to -∞) so token 5 can't see token 6
4. **Softmax:** Convert scores to probabilities (0 to 1, sums to 1)
5. **Weighted sum:** Multiply probabilities by Values → the output

**Causal Masking (why it's "Decoder-Only"):**
```
Position:    1    2    3    4
Token 1:  [1.0, -∞,  -∞,  -∞]    ← can only see itself
Token 2:  [0.3, 0.7, -∞,  -∞]    ← can see tokens 1-2
Token 3:  [0.1, 0.2, 0.7, -∞]    ← can see tokens 1-3
Token 4:  [0.1, 0.1, 0.3, 0.5]   ← can see tokens 1-4
```

This is what makes it autoregressive — each position can only attend to past positions and itself.

#### Step 5: Output Projection

After attention, the multi-head outputs are concatenated and projected:
```
O = Concat(head₁, head₂, ..., head₃₂) · W_O
```
Shape: `[seq_len, 32 × 128]` × `[4096, 4096]` → `[seq_len, 4096]`

### 2.3 Residual Connection (⊕)

**What the diagram shows:** The "+" (⊕) symbol after Attention Output, adding back to the input.

```
output = x + Attention(RMSNorm(x))
```

**Why residual connections?** Without them, stacking 32 layers causes **vanishing gradients** — the learning signal dies before reaching early layers. The residual connection creates a "highway" for gradients to flow directly backward through all 32 layers.

**Intuition:** Each layer adds a *refinement* to the input rather than completely transforming it. Layer 1's output = input + small adjustment. This makes the network much easier to train.

---

### 2.4 Feed-Forward Network (SwiGLU)

**What the diagram shows:** The yellow "FeedForward Network (SwiGLU)" box.

**Standard FFN (original Transformer):**
```
FFN(x) = ReLU(x · W₁) · W₂
```

**SwiGLU (what LLaMA 3 uses):**
```
SwiGLU(x) = (Swish(x · W_gate) ⊙ (x · W_up)) · W_down
```

Where:
- `W_gate`: [4096, 14336] — the "gate" projection
- `W_up`: [4096, 14336] — the "up" projection
- `W_down`: [14336, 4096] — the "down" projection
- `Swish(x) = x · σ(x)` where σ is sigmoid
- `⊙` means element-wise multiplication

**Why SwiGLU instead of ReLU?**

| Activation | Formula | Pros | Cons |
| :--- | :--- | :--- | :--- |
| ReLU | max(0, x) | Simple, fast | Kills negative values ("dying ReLU") |
| GELU | x · Φ(x) | Smooth, used in BERT | Slightly more expensive |
| **SwiGLU** | **Swish(xW_gate) ⊙ xW_up** | **Best quality, gating mechanism** | **3 matrices instead of 2** |

SwiGLU has a **gating mechanism**: the gate projection decides *how much* information flows through. This selective filtering produces better representations.

**Why 14,336?** The intermediate size is ~3.5x the hidden size (4,096 × 3.5 ≈ 14,336). SwiGLU uses 3 matrices instead of 2, so the intermediate size is chosen to keep total parameters roughly equal to a standard FFN with 4x expansion.

**Parameters per FFN layer:**
```
3 × 4,096 × 14,336 = 176,160,768 (~176M per layer)
32 layers: ~5.6 Billion parameters (70% of the model!)
```

### 2.5 Second Residual Connection (⊕)

```
output = attention_output + FFN(RMSNorm(attention_output))
```

Same principle as the first residual — adds the FFN's refinement on top.

---

## Part 3: The Output Block

### 3.1 Final RMS Norm

After all 32 decoder blocks, one final RMSNorm normalizes the output before the prediction head.

### 3.2 Linear (Unembedding) Layer

**What it does:** Projects from hidden dimension (4,096) back to vocabulary size (128,256).

```
logits = hidden_state · W_vocab^T    (shape: [seq_len, 4096] × [4096, 128256] → [seq_len, 128256])
```

Each of the 128,256 numbers in the output represents the model's "confidence" that the corresponding token is the next one.

**Weight Tying:** In many models, the unembedding matrix `W_vocab` is the **same** matrix as the input embeddings (transposed). LLaMA 3 does NOT tie weights — it has separate parameters for embedding and unembedding.

### 3.3 Inference Path: Softmax → Output Token

**What the diagram shows:** The green path at the top.

```
probabilities = softmax(logits)    # Convert to 0-1 range
next_token = sample(probabilities)  # Pick a token
```

**Sampling strategies:**
- **Greedy:** Always pick the highest probability token (deterministic)
- **Temperature:** Divide logits by T before softmax. T > 1 = more random, T < 1 = more focused
- **Top-p (nucleus):** Only consider tokens whose cumulative probability ≥ p
- **Top-k:** Only consider the top k tokens

In our project's inference, we use `temperature=0.7`, `top_p=0.9`, and `repetition_penalty=1.3`.

### 3.4 Training Path: Logits → Loss → Backpropagation

**What the diagram shows:** The right side of the diagram (blue path).

```
loss = CrossEntropyLoss(logits, targets)
```

**How training works:**
1. **Forward pass:** Text goes through all 32 layers → produces logits
2. **Loss computation:** Compare logits with the actual next token (from `train.jsonl`)
3. **Backpropagation:** Compute gradients — how much each parameter contributed to the error
4. **Weight update:** Adjust parameters to reduce the loss

In our LoRA setup, only the adapter parameters get updated (step 4). The base model's 8B parameters are frozen.

**Cross-Entropy Loss (the math):**
```
For a single token prediction:
  Loss = -log(P(correct_token))

If model assigns 90% probability to the correct token: Loss = -log(0.9) = 0.105  (low - good!)
If model assigns 10% probability to the correct token: Loss = -log(0.1) = 2.302  (high - bad!)
```

The model learns by minimizing this loss across all tokens in the training data.

---

## Part 4: Putting It All Together

### Complete Data Flow (Single Forward Pass)

```
"Recommend movies like Inception"
        ↓
   [Tokenizer: BPE → 7 token IDs]
        ↓
   [Embedding: 7 × 4096 matrix]
        ↓
   ┌─── Decoder Block 1 ────────────────────────┐
   │  RMSNorm → GQA (with RoPE) → ⊕ Residual   │
   │  RMSNorm → SwiGLU FFN → ⊕ Residual         │
   └────────────────────────────────────────────┘
        ↓
   ┌─── Decoder Block 2 ────────────────────────┐
   │  (same structure, different weights)         │
   └────────────────────────────────────────────┘
        ↓
       ... (×32 total)
        ↓
   [Final RMSNorm]
        ↓
   [Linear: 4096 → 128,256 logits]
        ↓
   [Softmax → "You" (most probable next token)]
        ↓
   [Append "You" to input, repeat for next token...]
```

### Why 32 Layers?

Research has shown that different layers learn different things:
- **Layers 1-8:** Surface patterns (syntax, grammar, token relationships)
- **Layers 9-20:** Semantic understanding (meaning, concepts, entity recognition)
- **Layers 21-32:** Task-specific reasoning (following instructions, generating responses)

This is why fine-tuning (especially LoRA at higher layers) can change the model's behavior without destroying its language understanding.

---

## Summary Table

| Component | What It Does | Why It Matters |
| :--- | :--- | :--- |
| **Tokenizer (BPE)** | Text → integer IDs | 128K vocab = efficient encoding |
| **Embeddings** | IDs → 4096-dim vectors | Semantic representation of tokens |
| **RMSNorm** | Stabilizes layer inputs | Faster than LayerNorm, prevents explosion |
| **GQA** | Attention with shared KV | 4x smaller cache, nearly same quality |
| **RoPE** | Rotational position encoding | Enables relative position, length generalization |
| **KV Cache** | Stores past K,V for reuse | O(n) instead of O(n²) per token |
| **SwiGLU** | Gated feed-forward network | Better representations than ReLU/GELU |
| **Residual (⊕)** | Adds input back to output | Prevents vanishing gradients across 32 layers |
| **Linear Head** | 4096 → 128K logits | Maps back to token probabilities |
| **Softmax** | Logits → probabilities | Picks the next token |
| **Cross-Entropy** | Measures prediction error | Training signal for backpropagation |
