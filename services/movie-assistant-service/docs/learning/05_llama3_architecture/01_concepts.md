# LLaMA 3 8B Architecture: The Complete Picture

You're fine-tuning a model, but do you actually know what's happening inside it? This module tears apart the **LLaMA 3 8B** architecture piece by piece so you understand every layer, every operation, and every design decision.

## 1. The Big Picture

LLaMA 3 8B is a **Decoder-Only Transformer**. That means it only predicts the **next token** given all previous tokens. It never looks forward — only backward.

Think of it like writing a sentence left-to-right: you pick each word based on everything you've written so far, never peeking at what comes next.

### Architecture at a Glance

| Specification | Value |
| :--- | :--- |
| **Parameters** | 8.03 Billion |
| **Architecture** | Decoder-Only Transformer |
| **Decoder Blocks (Layers)** | 32 |
| **Hidden Dimension (d_model)** | 4,096 |
| **Attention Heads (Queries)** | 32 |
| **KV Heads (Keys/Values)** | 8 (Grouped Query Attention) |
| **Head Dimension** | 128 (4096 / 32) |
| **FFN Intermediate Size** | 14,336 |
| **Vocabulary Size** | 128,256 tokens |
| **Max Context Length** | 8,192 tokens |
| **Normalization** | RMSNorm (Pre-Norm) |
| **Activation** | SwiGLU |
| **Positional Encoding** | RoPE (Rotary Positional Embeddings) |

## 2. The Three Blocks

The architecture diagram shows three major sections. Everything flows top-to-bottom during inference and loops back during training.

### Block 1: Input Block
```
Texts/Prompts → Tokenizer → Tokens → Embeddings
```
This is the **entrance**. Raw text gets converted into numbers the model can compute on.

### Block 2: Decoder Blocks (×32)
```
For each of the 32 layers:
  RMS Norm → GQA Attention → Residual → RMS Norm → SwiGLU FFN → Residual
```
This is the **brain**. Each layer refines the representation. Layer 1 might understand syntax, Layer 16 might understand concepts, Layer 32 might understand the task.

### Block 3: Output Block
```
RMS Norm → Linear → Softmax → Output Token    (Inference)
RMS Norm → Linear → Logits → Loss → Backprop  (Training)
```
This is the **mouth**. It converts the final representation back into a probability distribution over all 128K tokens.

## 3. Why "Decoder-Only"?

The original Transformer (2017) had both an **Encoder** (reads input) and a **Decoder** (generates output). GPT and LLaMA dropped the encoder entirely.

**Why?** Because for text generation, the decoder alone is sufficient. Every token attends to all previous tokens through **causal (masked) self-attention**. The model learns to simultaneously understand AND generate.

| Architecture | Example Models | Use Case |
| :--- | :--- | :--- |
| Encoder-Only | BERT, RoBERTa | Classification, NER |
| Encoder-Decoder | T5, BART | Translation, Summarization |
| **Decoder-Only** | **LLaMA 3**, GPT-4 | **Text Generation, Chat, Reasoning** |

## 4. Parameter Count Breakdown

Where do those 8 Billion parameters actually live?

| Component | Parameters | % of Total |
| :--- | :--- | :--- |
| **Token Embeddings** | 128,256 × 4,096 = ~525M | ~6.5% |
| **Attention (Q,K,V,O) × 32 layers** | 32 × (3 × 4,096 × 4,096 + 4,096 × 4,096) / GQA savings ≈ 1.8B | ~22% |
| **FFN (Gate, Up, Down) × 32 layers** | 32 × (3 × 4,096 × 14,336) ≈ 5.6B | ~70% |
| **RMSNorm × 65** | 65 × 4,096 ≈ 0.27M | ~0.003% |
| **Output Head (tied/untied)** | ~525M | ~1.5% |

**Key Insight:** ~70% of the model's parameters are in the **Feed-Forward Networks**, not attention. This is why your LoRA config targets `gate_proj`, `up_proj`, `down_proj` alongside the attention projections — skipping them would miss most of the model's capacity.

## 5. What Makes LLaMA 3 Different from LLaMA 2?

| Feature | LLaMA 2 | LLaMA 3 |
| :--- | :--- | :--- |
| **Vocabulary** | 32K tokens | 128K tokens (4x larger) |
| **Tokenizer** | SentencePiece (BPE) | Tiktoken-based (BPE) |
| **Attention** | GQA (same) | GQA (same) |
| **Training Data** | 2T tokens | 15T+ tokens (7.5x more) |
| **Context Length** | 4,096 | 8,192 (2x longer) |

The biggest improvement in LLaMA 3 isn't the architecture — it's the **data**. 7.5x more training tokens means the model has seen vastly more of the internet, books, and code. The 4x larger vocabulary also means fewer tokens per word, making it more efficient at encoding text.

## 6. How This Connects to Our Project

In our StreamSage project, we fine-tune this 8 billion parameter model to become a specialized movie recommendation assistant:

1. **The Input Block** processes user queries like "Movies like Inception" through the same tokenizer.
2. **The 32 Decoder Blocks** are mostly **frozen** during our LoRA training — we only add tiny adapter matrices beside the Q, K, V, O, Gate, Up, Down projections.
3. **The Output Block** generates movie recommendations token by token, using the same softmax distribution but now biased toward our trained style.

When we quantize to 4-bit (QLoRA), we're compressing those 8B parameters from 16-bit floats (16GB) down to 4-bit integers (~5.5GB), making it trainable on a single GPU.

## Summary

| Concept | One-Liner |
| :--- | :--- |
| **Decoder-Only** | Predicts next token from all previous tokens |
| **32 Layers** | Each layer refines understanding progressively |
| **GQA** | Efficient attention with shared Key/Value heads |
| **SwiGLU** | Better activation function for the FFN |
| **RoPE** | Position information encoded as rotations |
| **RMSNorm** | Simpler, faster normalization |
| **128K Vocab** | Larger vocabulary = more efficient tokenization |
| **8,192 Context** | Can read ~6,000 words at once |
