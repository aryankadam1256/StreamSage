# Interview Prep: LLaMA 3 Architecture

Deep questions you'll face in ML/AI interviews. Each answer is structured to demonstrate both theoretical understanding and practical experience from our StreamSage project.

---

## Foundational Architecture Questions

### Q1: Explain the LLaMA 3 architecture. Walk me through how it generates text.

**Answer:**
LLaMA 3 is a **decoder-only transformer** with 8 billion parameters spread across 32 identical decoder layers.

**The data flow:**
1. **Tokenization:** Input text is split into subword tokens using BPE (128K vocabulary).
2. **Embedding:** Each token ID is looked up in an embedding table to get a 4,096-dimensional vector.
3. **32 Decoder Blocks:** Each block applies:
   - **RMSNorm** (pre-normalization for stability)
   - **Grouped Query Attention** with 32 query heads and 8 KV heads
   - **Residual connection** (adds attention output back to input)
   - **RMSNorm** again
   - **SwiGLU feed-forward network** (gated activation, 14,336 intermediate dim)
   - **Residual connection** again
4. **Output:** Final RMSNorm → Linear projection to 128K vocabulary → Softmax → sample next token.

Generation is **autoregressive** — the model produces one token at a time, appending each to the input for the next step. A **KV cache** stores previously computed key-value pairs to avoid recomputation.

*In our project, we fine-tuned this model with LoRA adapters on the Q, K, V, O, Gate, Up, and Down projections to specialize it for movie recommendations.*

---

### Q2: What is Grouped Query Attention (GQA)? Why does LLaMA 3 use it instead of standard Multi-Head Attention?

**Answer:**
In standard **Multi-Head Attention (MHA)**, you have N query heads, N key heads, and N value heads. LLaMA 3 uses **GQA** where 32 query heads share only **8 key-value head groups** — each KV head serves 4 query heads.

**Why?** The bottleneck during inference is the **KV cache**. For each generated token, you must store all previous K and V vectors across all layers. With MHA (32 KV heads), an 8K context sequence needs:
```
32 KV heads × 128 dim × 2 bytes × 8192 tokens × 32 layers × 2 (K+V) ≈ 32 GB
```

With GQA (8 KV heads), it's only **~8 GB** — a 4x reduction. Research shows this has negligible quality impact because the Key and Value projections capture more redundant information than the Query projections.

**The spectrum:**
- **MHA:** N query heads, N KV heads (full, most memory)
- **GQA:** N query heads, G KV heads where G < N (balanced)
- **MQA:** N query heads, 1 KV head (minimal, some quality loss)

LLaMA 3 picks the middle ground with G=8.

---

### Q3: What is RoPE (Rotary Positional Encoding) and why is it better than absolute positional embeddings?

**Answer:**
RoPE encodes position by **rotating** the query and key vectors by an angle proportional to their position in the sequence.

**The math:** For position m, each pair of dimensions (2i, 2i+1) in Q and K is rotated by angle `m × θᵢ` where `θᵢ = 1/10000^(2i/d)`.

**Why this is better than absolute position embeddings:**

1. **Relative position:** When you compute `Q·K^T`, the rotation angles **subtract**, so the attention score naturally depends on the **distance** between tokens, not their absolute positions. Token 5 attending to token 3 has the same relative signal as token 100 attending to token 98.

2. **Length extrapolation:** The model can handle sequence lengths longer than it was trained on because RoPE doesn't rely on a fixed-size position table. The rotation formula works for any position m.

3. **Applied at every layer:** Unlike learned positional embeddings (added once at the input), RoPE is applied to Q and K at every attention layer, giving the model fresh position signal at each depth.

4. **Not applied to Values:** V vectors carry content, not position. This separation is cleaner than adding position to the entire representation.

*In our project, RoPE enables the model to handle both short queries ("movies like Inception") and long multi-turn conversations without retraining.*

---

### Q4: Explain SwiGLU. Why did LLaMA 3 choose it over ReLU or GELU?

**Answer:**
SwiGLU is a **gated activation** function used in the feed-forward network of each transformer layer.

**Formula:**
```
SwiGLU(x) = SiLU(x · W_gate) ⊙ (x · W_up)
```

Where SiLU(x) = x × sigmoid(x), also called "Swish".

**What the gating does:** The gate projection decides *how much* of each feature to let through. This is fundamentally different from ReLU (which just zeroes negatives) or GELU (which smoothly zeroes small values). The gate can learn complex filtering patterns.

**The trade-off:** SwiGLU requires **3 weight matrices** (gate, up, down) instead of 2 (up, down in standard FFN). To keep the parameter count similar, LLaMA 3 uses an intermediate dimension of 14,336 (≈ 3.5× hidden dim) instead of the typical 4× expansion.

**Empirical results:** In the PaLM paper, SwiGLU consistently outperformed ReLU, GELU, and other activations across all model sizes. The quality improvement is ~1-2% on benchmarks, which is significant at scale.

*In our LoRA config, we target all three SwiGLU matrices (gate_proj, up_proj, down_proj) because they contain ~70% of the model's parameters.*

---

### Q5: Why does LLaMA 3 use RMSNorm instead of LayerNorm? Why Pre-Norm instead of Post-Norm?

**Answer:**

**RMSNorm vs LayerNorm:**
- LayerNorm: Subtracts mean, divides by std, applies scale (γ) and bias (β).
- RMSNorm: Only divides by root-mean-square, applies scale (γ). No mean subtraction, no bias.

RMSNorm is ~10-15% faster because it skips the mean computation, and experiments show the mean subtraction doesn't help for transformers. The bias term β is also unnecessary because the subsequent linear layers can learn an offset.

**Pre-Norm vs Post-Norm:**
- **Post-Norm** (original Transformer): `output = LayerNorm(x + Sublayer(x))`
- **Pre-Norm** (LLaMA 3): `output = x + Sublayer(RMSNorm(x))`

Pre-Norm puts the normalization **before** each sublayer rather than after. This makes training much more stable for two reasons:
1. The residual connection carries unnormalized values, so gradients flow freely backward through the residual path.
2. Each sublayer receives normalized input, preventing activation explosions in deep networks.

The downside: Pre-Norm models can be slightly harder to converge to the same quality as Post-Norm at very small model sizes. But at 8B+ parameters, it's strictly better for training stability.

---

## Inference & Optimization Questions

### Q6: What is a KV Cache? Why is it critical for LLM inference performance?

**Answer:**
During autoregressive generation, the model produces one token at a time. Without a KV cache, generating token N requires recomputing the Key and Value vectors for all N-1 previous tokens across all 32 layers — this is O(N²) total computation.

**With KV cache:** We store K and V from all previous tokens. For each new token, we only compute Q, K, V for that single new token, append K and V to the cache, and compute attention against the full cache. This is O(N) per token.

**Memory math for LLaMA 3 8B:**
```
Per token: 2 (K+V) × 8 (GQA heads) × 128 (head dim) × 2 (FP16 bytes) = 4 KB
Per token across 32 layers: 4 KB × 32 = 128 KB
Full 8K context: 128 KB × 8,192 = ~1 GB
```

Without GQA (32 KV heads instead of 8): **~4 GB** — GQA reduces this 4x.

**PagedAttention (vLLM):** Further optimizes by managing the KV cache like virtual memory pages, eliminating fragmentation and pre-allocation waste.

*In our deployment, we use vLLM's PagedAttention with block size 16, which can serve multiple concurrent movie recommendation requests efficiently.*

---

### Q7: Explain Flash Attention. How does it achieve O(N) memory instead of O(N²)?

**Answer:**
Standard attention computes the full N×N attention matrix `S = Q·K^T`, which requires O(N²) memory. For N=8,192, that's a 67M element matrix **per head per layer**.

**Flash Attention avoids materializing this matrix entirely.** It uses **tiling**:
1. Divide Q, K, V into small blocks that fit in GPU SRAM (fast on-chip memory).
2. For each block of Q, iterate through all blocks of K and V.
3. Compute partial attention scores, maintain running softmax statistics (max and sum).
4. Accumulate the final output incrementally.

**The key insight:** Softmax is decomposable. You can compute it block-by-block if you track the running maximum and normalize at the end. This means you never need the full N×N matrix — you only need memory for one tile at a time.

**Result:** O(N) memory, and it's actually 2-4x **faster** than standard attention because it eliminates memory-bound operations (reading/writing the huge attention matrix to GPU HBM).

*In our training code, Flash Attention 2 is enabled with `attn_implementation="flash_attention_2"`, which is critical for fitting long sequences in our RTX 4080's 16GB VRAM.*

---

### Q8: What happens during 4-bit quantization (QLoRA)? How can the model still work with such low precision?

**Answer:**
4-bit quantization compresses each weight from 16-bit float (2 bytes) to 4-bit integer (0.5 bytes), a 4x memory reduction.

**NF4 (Normal Float 4)** specifically:
1. Assumes weights follow a normal distribution (which they empirically do).
2. Uses 16 quantization levels optimally spaced for a normal distribution (not uniformly spaced like INT4).
3. For each group of 64-128 weights, stores a scale factor and zero point.

**Why it still works:**
- Weights are stored in 4-bit, but **computation happens in BF16**. Before each matrix multiplication, the relevant weights are dequantized on-the-fly to BF16.
- The quantization error is small because NF4 levels match the actual weight distribution.
- **Double quantization** further compresses the scale factors themselves.

**QLoRA adds LoRA on top:** The frozen base is 4-bit, but the LoRA adapters (A, B matrices) are trained in full FP16 precision. The adapters learn to "correct" any errors introduced by quantization.

*In our config, we use `bnb_4bit_quant_type="nf4"` with `bnb_4bit_use_double_quant=True`, fitting the entire 8B model + training overhead into 16GB VRAM.*

---

## Architecture Design Questions

### Q9: Why is the FFN intermediate dimension 14,336 (not 16,384 = 4 × 4,096)?

**Answer:**
In a standard transformer FFN with ReLU, the intermediate dimension is typically 4× the hidden dimension (4 × 4,096 = 16,384), using 2 weight matrices.

SwiGLU uses **3 weight matrices** (gate, up, down) instead of 2. To keep the total parameter count roughly equivalent, the intermediate dimension is reduced:
```
Standard FFN: 2 × d × 4d = 8d² parameters per layer
SwiGLU FFN:   3 × d × d_ff parameters per layer

To match: 3 × d × d_ff = 8d²
d_ff = 8d/3 ≈ 2.67d

For d=4096: d_ff ≈ 10,923
```

LLaMA 3 actually uses 14,336 (≈ 3.5d), which is **larger** than the equal-parameter equivalent. Meta chose to allocate more parameters to the FFN because experiments showed the FFN benefits more from capacity than the attention layers.

14,336 is also chosen to be a multiple of 256 for GPU memory alignment and efficient matrix multiplication.

---

### Q10: Why does LLaMA 3 NOT tie the input and output embedding weights?

**Answer:**
**Weight tying** means using the same matrix for both the input embedding (token → vector) and the output projection (vector → token logits). Models like GPT-2 and T5 do this to save parameters.

**LLaMA 3 uses separate weights** because:
1. **Different functions:** The input embedding maps tokens to a general semantic space. The output projection maps from a task-specific representation back to token probabilities. These may benefit from different learned representations.
2. **Parameter budget:** At 8B parameters, the embedding table (128K × 4,096 ≈ 525M) is only ~6.5% of the model. Tying saves ~525M parameters but the quality trade-off isn't worth it at this scale.
3. **Quantization flexibility:** With separate weights, you can quantize the input embeddings and output head differently if needed.

---

### Q11: In our project, we target 7 modules with LoRA. Could you explain which are most important and why?

**Answer:**
The 7 target modules map directly to the architecture:

**Attention (4 modules):** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- These control **what the model attends to** and **how it combines information**
- `q_proj` has the most impact — it determines the "questions" each token asks
- `k_proj` and `v_proj` can use smaller adapters since they have fewer heads (GQA: 8 vs 32)

**FFN (3 modules):** `gate_proj`, `up_proj`, `down_proj`
- These control **knowledge retrieval and transformation**
- `gate_proj` is crucial — it's the gating mechanism that filters information
- Together they contain ~70% of all parameters

**For movie recommendations specifically:**
- Adapting attention modules changes **what context the model focuses on** (user preferences, genre keywords, actor names)
- Adapting FFN modules changes **what knowledge is activated** (movie facts, recommendation patterns)
- Targeting all 7 gives the best quality. Targeting only attention (4 modules) would be cheaper but miss the knowledge adaptation.

*Research shows that targeting all linear layers gives ~15% better quality than attention-only LoRA, which is why our config includes all 7.*

---

### Q12: Walk me through what happens to a single weight during LoRA fine-tuning.

**Answer:**
Take one specific weight in `q_proj` of Layer 16, say `W[100][200]`:

**Before LoRA:**
```
W[100][200] = 0.00342 (stored in NF4 4-bit format)
Output row 100 = input · W[100, :]
```

**After adding LoRA (rank 32):**
```
Original W is FROZEN (still 0.00342 in 4-bit).
We add two new matrices:
  A[200, :] → row vector of 32 numbers (trainable, FP16)
  B[:, 100] → column vector of 32 numbers (trainable, FP16)

Output row 100 = input · W[100, :] + (α/√r) × input · A · B[:, 100]
                 ↑ frozen base          ↑ trainable adapter
```

**During training:**
1. Forward: Compute both terms, add them.
2. Loss: Compare output with target.
3. Backward: Compute gradients for A and B (NOT for W).
4. Update: AdamW adjusts A and B by a tiny amount.

**After 3 epochs:** A and B have been tuned so that the adapter adds exactly the right correction to make the model recommend movies in our specific style.

**At merge time:**
```
W_new[100][200] = W[100][200] + (α/√r) × sum(A[200, k] × B[k, 100] for k in 0..31)
                = 0.00342 + small_correction
                ≈ 0.00358
```

The adapter's correction is permanently baked into the weight.

---

## Rapid-Fire Questions

### Q13: What is the context length of LLaMA 3 8B?
**Answer:** 8,192 tokens (~6,000 words). This is 2x the 4,096 of LLaMA 2, achieved by training with longer sequences and RoPE's natural extrapolation ability.

### Q14: How many attention heads does LLaMA 3 8B have?
**Answer:** 32 query heads and 8 key-value heads (GQA with ratio 4:1). Each head has dimension 128 (4096/32).

### Q15: What optimizer does LLaMA 3 use during pre-training?
**Answer:** AdamW with β₁=0.9, β₂=0.95, weight decay=0.1. We use AdamW 8-bit for fine-tuning to save memory.

### Q16: Why is the tokenizer vocabulary 128,256 and not a round number like 128,000?
**Answer:** The base vocabulary has 128,000 tokens from BPE training, plus 256 reserved special tokens (begin_of_text, end_of_text, role headers, padding, etc.). 128,000 + 256 = 128,256.

### Q17: What would happen if we removed all residual connections?
**Answer:** Training would fail catastrophically. Without residual connections, gradients must pass through 32 layers of matrix multiplications, and they vanish exponentially. The model would be no better than a random predictor. Residual connections create a "gradient highway" that allows learning signal to reach early layers.

### Q18: The diagram shows 32 decoder blocks. Why 32? Why not 64 or 16?
**Answer:** It's a design trade-off. For a fixed parameter budget of ~8B:
- **More layers, narrower (e.g., 64 × 2,048 dim):** Better at compositional reasoning but slower (more sequential steps).
- **Fewer layers, wider (e.g., 16 × 5,793 dim):** Better parallelism but less depth for complex reasoning.
- **32 × 4,096:** The sweet spot found through extensive experimentation at Meta. It balances depth (reasoning) with width (capacity per layer).

### Q19: How does our fine-tuned model differ from base LLaMA 3 during inference?
**Answer:** Structurally, they are identical — same architecture, same forward pass, same KV cache. The only difference is the numerical values of the weights. Our LoRA adapters have been merged into the base weights, shifting the probability distributions so the model prefers movie recommendation formats, enthusiastic tone, and structured outputs. The same Softmax → sampling process produces different tokens because the logits are different.

### Q20: If you could only target ONE module with LoRA, which would you choose and why?
**Answer:** `q_proj`. The query projection determines what each token "asks" during attention — it's the most impactful single point of control. Modifying Q changes the attention patterns across the entire model, which cascades into different outputs. Research confirms that Q-only LoRA retains ~85% of the quality of full 7-module LoRA, while K-only or V-only retain ~70%.
