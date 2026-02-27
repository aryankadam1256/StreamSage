# Code Walkthrough: Architecture ↔ Project Code Mapping

This document connects every component from the LLaMA 3 architecture diagram to the **actual code** in our StreamSage project. For each architectural component, you'll see exactly where it appears in our codebase and how we configure/optimize it.

---

## 1. The Input Block → Tokenizer & Embeddings

### Architecture Diagram: `Texts/Prompts → Tokenizer → Tokens → Embeddings`

### In Our Code:

**File:** `fine_tuning/train_local.py:142`
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

**What's happening:**
- `AutoTokenizer` loads the LLaMA 3 **BPE tokenizer** with 128,256 tokens
- `pad_token = eos_token` — LLaMA 3 doesn't have a dedicated pad token, so we reuse the end-of-sequence token
- `padding_side = "right"` — pads sequences on the right for decoder-only models (left padding would break causal attention)

**The Embedding Layer** is inside the model itself. When you call `model.forward()`, the first thing that happens internally is:
```python
# Inside LlamaModel (huggingface transformers):
inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len] → [batch, seq_len, 4096]
```
This is the lookup table that converts token IDs to 4,096-dimensional vectors.

### Special Tokens for Chat Format

**File:** `fine_tuning/llama_finetune_colab.ipynb` (Cell: Data Formatting)
```python
alpaca_prompt = """Below is an instruction that describes a task. Write a response.

### Instruction:
{}

### Response:
{}"""
```

On the local training script (`train_local.py`), the LLaMA 3 chat template is used:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Recommend movies like Inception<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Here are my recommendations...<|eot_id|>
```

These special tokens are part of the **128K vocabulary** — the tokenizer knows their IDs and the model has learned their meaning during Meta's pre-training.

---

## 2. Quantization → How We Compress 8B Parameters

### Architecture Diagram: This isn't shown in the diagram, but it's how we **fit** the model onto a GPU.

### In Our Code:

**File:** `fine_tuning/train_local.py:112-117`
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,        # Nested quantization
)
```

**What each parameter does:**

| Parameter | Value | What It Does |
| :--- | :--- | :--- |
| `load_in_4bit` | `True` | Loads all 8B weights as 4-bit integers (16GB → ~5.5GB) |
| `bnb_4bit_quant_type` | `"nf4"` | Uses Normal Float 4 — values are optimally distributed for neural network weights (better than uniform INT4) |
| `bnb_4bit_compute_dtype` | `bfloat16` | During computation, weights are dequantized to BF16 on-the-fly. We don't do math in 4-bit — that would be too imprecise |
| `bnb_4bit_use_double_quant` | `True` | Quantizes the quantization constants too, saving another ~0.4GB |

**How NF4 relates to the architecture:**
Every matrix in the architecture diagram (Q, K, V, O, Gate, Up, Down projections) has its weights stored in NF4 format. When data flows through them, the weights are temporarily dequantized to BF16 for the matrix multiplication, then the result continues in BF16.

---

## 3. The Decoder Block → Attention + FFN

### Architecture Diagram: `RMS Norm → GQA → ⊕ → RMS Norm → SwiGLU FFN → ⊕`

### 3.1 Loading with Flash Attention 2

**File:** `fine_tuning/train_local.py:121-128`
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # <-- THIS
    torch_dtype=torch.bfloat16,
)
```

**`attn_implementation="flash_attention_2"`** changes how the **GQA block** in the diagram computes `softmax(QK^T/√d)V`. Instead of:
1. Computing the full `QK^T` matrix (O(n²) memory)
2. Applying softmax
3. Multiplying by V

Flash Attention does it all in **one fused kernel** with O(n) memory using tiling. It never materializes the full n×n attention matrix.

**Fallback:** If the GPU doesn't support Flash Attention (requires Ampere+, compute ≥ 8.0), it falls back to SDPA (Scaled Dot Product Attention), PyTorch's native implementation.

### 3.2 LoRA Adapters → What We Actually Train

**File:** `fine_tuning/optimization_config.py:44-52`
```python
"target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",     # ← GQA block in diagram
    "gate_proj", "up_proj", "down_proj",          # ← SwiGLU FFN in diagram
],
```

**This is where the architecture diagram meets our training code.** Each name maps directly to a component in the diagram:

| Config Name | Diagram Component | Matrix Shape | What It Does |
| :--- | :--- | :--- | :--- |
| `q_proj` | Q in GQA block | [4096, 4096] | Generates Query vectors |
| `k_proj` | K in GQA block | [4096, 1024] | Generates Key vectors (8 heads) |
| `v_proj` | V in GQA block | [4096, 1024] | Generates Value vectors (8 heads) |
| `o_proj` | Attention Output | [4096, 4096] | Projects attention output back |
| `gate_proj` | SwiGLU gate (W_gate) | [4096, 14336] | Gate for SwiGLU activation |
| `up_proj` | SwiGLU up (W_up) | [4096, 14336] | Up projection in SwiGLU |
| `down_proj` | SwiGLU down (W_down) | [14336, 4096] | Down projection in SwiGLU |

**LoRA inserts small adapter matrices next to each of these:**

```python
"lora_r": 32,        # Adapter rank
"lora_alpha": 64,    # Scaling factor
"use_rslora": True,  # Rank-stabilized LoRA
```

For each target module, instead of the original:
```
output = x · W                    (W is frozen, 4-bit)
```

LoRA adds:
```
output = x · W + x · A · B       (A and B are trainable, FP16)
```

Where `A` is [4096, 32] and `B` is [32, 4096]. This means we only train `32 × 4096 × 2 = 262,144` parameters per adapter instead of `4096 × 4096 = 16,777,216`.

**rsLoRA** (rank-stabilized): Scales the adapter output by `α/√r` instead of `α/r`. This stabilizes training at higher ranks (like our rank=32).

### 3.3 Gradient Checkpointing → Trading Compute for Memory

**File:** `fine_tuning/optimization_config.py:53`
```python
"use_gradient_checkpointing": "unsloth",  # 30% less VRAM
```

During the training path in the diagram (backpropagation arrow), PyTorch normally stores the activations from ALL 32 decoder blocks for the backward pass. With gradient checkpointing, it **discards** intermediate activations and **recomputes** them during backprop. This saves ~30% VRAM at the cost of ~20% more compute.

The `"unsloth"` variant is a custom implementation that is smarter about what to checkpoint, reducing the recomputation overhead.

---

## 4. The Output Block → Loss & Backpropagation

### Architecture Diagram: `RMS Norm → Linear → Logits → Loss → Backprop`

### 4.1 Training Configuration

**File:** `fine_tuning/optimization_config.py:56-63`
```python
"num_train_epochs": 3,
"per_device_train_batch_size": 6,
"gradient_accumulation_steps": 4,     # Effective batch: 6 × 4 = 24
"learning_rate": 2e-4,
"warmup_ratio": 0.05,
"lr_scheduler_type": "cosine",
"optim": "adamw_8bit",
```

**How this maps to the diagram:**

1. **Forward Pass** (top to bottom in diagram): Input → 32 decoder blocks → Linear → Logits
2. **Loss** (right side): `CrossEntropyLoss(logits, targets)` — compares model predictions with our `train.jsonl` ground truth
3. **Backpropagation** (right side arrow going back): Gradients flow backward through the Linear head → 32 decoder blocks → but ONLY update the LoRA adapter weights
4. **Update Weights** (bottom right): AdamW 8-bit optimizer adjusts the LoRA parameters

**Effective batch size = 24** means the model sees 24 examples before each weight update. This is achieved by processing 6 sequences at a time and accumulating gradients for 4 steps.

### 4.2 Cosine Learning Rate Schedule

```python
"lr_scheduler_type": "cosine",
"warmup_ratio": 0.05,
```

The learning rate starts at 0, warms up to `2e-4` over the first 5% of steps, then follows a cosine curve down to ~0. This gives the model big updates in the middle of training and tiny refinements at the end.

### 4.3 Sequence Packing

```python
"packing": True,
```

The diagram shows one sequence flowing through the model. With packing, we cram **multiple short sequences into one** (up to `max_seq_length=2048`):

```
Without packing:  ["Recommend movies" + PAD PAD PAD PAD PAD]  ← wasted compute on PADs
With packing:     ["Recommend movies" + "Best sci-fi" + "Action comedies"]  ← no waste
```

This gives ~5x throughput improvement because movie recommendation prompts are short (100-300 tokens each).

### 4.4 NEFTune

```python
"neftune_noise_alpha": 5,
```

This adds small random noise to the **Embeddings** layer (the first component in the diagram). During training:
```
embeddings = embed_tokens(input_ids) + α × random_noise
```

This forces the model to be robust to slight input variations, improving generalization by +2-5% on benchmarks.

---

## 5. KV Cache → Inference Optimization

### Architecture Diagram: `KV Cache (For Inferencing only)`

### In Our Code:

**File:** `fine_tuning/inference_optimizer.py` and `fine_tuning/optimization_config.py:237-239`
```python
"kv_cache": {
    "block_size": 16,                    # PagedAttention block size
    "enable_chunked_prefill": True,      # Chunk long prefills
},
```

**During inference** (the green path in the diagram going to "Output Token"), the KV Cache stores all previously computed K and V tensors. Our vLLM configuration uses **PagedAttention** which manages the KV cache like virtual memory:

- Cache is divided into blocks of 16 tokens
- Blocks are allocated on-demand (no pre-allocation waste)
- Shared prefixes (like our system prompt) are cached once and reused

**File:** `fine_tuning/optimization_config.py:250-258`
```python
"generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,       # Applied to logits before softmax
    "top_p": 0.9,             # Nucleus sampling
    "top_k": 50,              # Top-k sampling
    "repetition_penalty": 1.2, # Penalize already-generated tokens
},
```

These parameters control the **Softmax → Output Token** step in the diagram:
- **temperature=0.7** — divides logits by 0.7 before softmax, making the distribution more peaked
- **top_p=0.9** — only considers tokens whose cumulative probability reaches 90%
- **repetition_penalty=1.2** — reduces the logit of any token already in the output

---

## 6. Complete Architecture ↔ Code Map

| Diagram Component | Code Location | What We Configure |
| :--- | :--- | :--- |
| **Texts/Prompts** | `train.jsonl` dataset | Movie recommendation examples |
| **Tokenizer** | `AutoTokenizer.from_pretrained()` | BPE, 128K vocab, special chat tokens |
| **Embeddings** | Inside model, `embed_tokens` | NEFTune noise during training |
| **RMS Norm** | Inside model, `input_layernorm` | Not directly configured |
| **Q, K, V projections** | LoRA `target_modules: q/k/v_proj` | Rank-32 adapters attached |
| **RoPE** | Inside model, `rotary_emb` | Not directly configured |
| **KV Cache** | `inference_config["kv_cache"]` | PagedAttention blocks |
| **GQA** | `attn_implementation="flash_attention_2"` | Flash Attention kernel |
| **Attention Output** | LoRA `target_modules: o_proj` | Rank-32 adapter attached |
| **⊕ Residual** | Inside model, hardcoded | Not configurable |
| **SwiGLU FFN** | LoRA `target_modules: gate/up/down_proj` | Rank-32 adapters attached |
| **Linear (output head)** | Inside model, `lm_head` | NOT targeted by LoRA |
| **Softmax** | `generation_config` | temperature, top_p, top_k |
| **Loss** | `SFTTrainer` / `DPOTrainer` | Cross-entropy / DPO loss |
| **Backpropagation** | `gradient_checkpointing="unsloth"` | Recompute-based memory saving |
| **Weight Update** | `optim="adamw_8bit"` | 8-bit optimizer states |

---

## 7. Why We Target These Specific Modules

Looking at the diagram, there are only a few types of learnable weight matrices in each decoder block:

```
Attention:  q_proj, k_proj, v_proj, o_proj    (4 matrices)
FFN:        gate_proj, up_proj, down_proj      (3 matrices)
Norm:       weight (γ in RMSNorm)              (not targeted — too small)
```

We target **all 7 projection matrices** because:
1. Attention projections control **what the model pays attention to** (critical for understanding movie queries)
2. FFN projections control **what knowledge is retrieved/generated** (critical for generating recommendations)
3. RMSNorm weights are tiny (4,096 params each) and don't benefit much from adaptation

This gives us adapters on **70% of the model's parameters** by count (the FFN) plus the most impactful attention layers.
