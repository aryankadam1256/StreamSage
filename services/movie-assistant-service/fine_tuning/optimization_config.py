"""
============================================================================
Training & Inference Optimization Configuration
============================================================================

Centralized config for all optimization techniques:

TRAINING OPTIMIZATIONS:
  - Flash Attention 2: Fused attention kernels (2-4x faster, O(N) memory)
  - Sequence Packing: Pack multiple short sequences into one (5x throughput)
  - NEFTune: Noise embeddings for better generalization
  - Gradient Checkpointing: Trade compute for memory (Unsloth variant)
  - Mixed Precision: BF16/FP16 automatic selection
  - 8-bit AdamW: Memory-efficient optimizer
  - LoRA+/rsLoRA: Rank-stabilized LoRA for better convergence
  - Cosine Annealing: Better LR schedule than linear decay

ALIGNMENT OPTIMIZATIONS:
  - DPO: Direct Preference Optimization (no reward model needed)
  - GRPO: Group Relative Policy Optimization (best for reasoning)
  - PPO: Proximal Policy Optimization (classic RLHF)

INFERENCE OPTIMIZATIONS:
  - vLLM: PagedAttention + continuous batching
  - Speculative Decoding: Draft model acceleration
  - Flash Attention 2: Fused inference kernels
  - Quantization: GPTQ/AWQ/BnB for memory reduction
  - KV Cache Optimization: Sliding window + compression

============================================================================
"""


# ============================================================================
# SFT Training Configuration
# ============================================================================

SFT_CONFIG = {
    # --- Model ---
    "model_name": "unsloth/llama-3-8b-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,

    # --- LoRA ---
    "lora_r": 32,                    # Rank (higher = more capacity, more memory)
    "lora_alpha": 64,                # Alpha (rule of thumb: 2x rank)
    "lora_dropout": 0,               # 0 is optimized by Unsloth
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "use_rslora": True,              # Rank-stabilized LoRA (better convergence)
    "use_gradient_checkpointing": "unsloth",  # 30% less VRAM

    # --- Training Hyperparameters ---
    "num_train_epochs": 3,
    "per_device_train_batch_size": 6,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,            # 5% warmup (better than fixed steps)
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",   # Cosine annealing > linear decay
    "optim": "adamw_8bit",           # 8-bit Adam (2x less optimizer memory)

    # --- Sequence Packing ---
    "packing": True,                 # Pack short sequences together (5x faster)

    # --- NEFTune ---
    "neftune_noise_alpha": 5,        # Noise injection (improves generalization)

    # --- Flash Attention ---
    "use_flash_attention": True,     # Auto-enabled on Ampere+ GPUs

    # --- Mixed Precision ---
    "fp16": "auto",                  # Auto-detect: FP16 for T4, BF16 for A100+
    "bf16": "auto",

    # --- Logging ---
    "logging_steps": 5,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,

    # --- Seed ---
    "seed": 3407,
}


# ============================================================================
# DPO (Direct Preference Optimization) Configuration
# ============================================================================
# Best for: Movie recommendation quality alignment
# Why: Creates preference pairs (good vs bad recommendations)
#       No reward model needed, stable training, works great with LoRA

DPO_CONFIG = {
    # --- DPO Specific ---
    "beta": 0.1,                     # KL penalty (lower = more deviation from ref)
    "loss_type": "sigmoid",          # sigmoid (standard), hinge, ipo
    "label_smoothing": 0.0,          # 0 = no smoothing
    "max_prompt_length": 512,
    "max_completion_length": 1024,

    # --- Training ---
    "num_train_epochs": 1,           # DPO typically needs fewer epochs
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,           # Lower LR for alignment
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "weight_decay": 0.01,

    # --- LoRA (if continuing from SFT) ---
    "lora_r": 16,                    # Can be smaller for DPO
    "lora_alpha": 32,

    # --- Logging ---
    "logging_steps": 5,
    "eval_steps": 25,
    "save_steps": 50,
}


# ============================================================================
# GRPO (Group Relative Policy Optimization) Configuration
# ============================================================================
# Best for: Reasoning-heavy tasks, multi-criteria recommendations
# Why: No reward model, groups responses and ranks them relatively
#       Works well for tasks where "better" is relative

GRPO_CONFIG = {
    # --- GRPO Specific ---
    "num_generations": 4,            # Number of responses per prompt to compare
    "max_prompt_length": 512,
    "max_completion_length": 1024,
    "temperature": 0.9,              # Sampling temp for generations
    "top_p": 0.95,

    # --- Reward Functions ---
    "reward_functions": [
        "format_reward",             # Correct response format
        "recommendation_count",      # Has enough recommendations
        "diversity_reward",          # Genre/era diversity in recs
        "relevance_reward",          # Recommendations match query
    ],
    "reward_weights": [0.2, 0.2, 0.3, 0.3],

    # --- Training ---
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,   # Small batch due to multiple generations
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-5,              # Very low LR for GRPO
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",

    # --- KL Penalty ---
    "kl_coef": 0.05,                # KL divergence coefficient

    # --- Logging ---
    "logging_steps": 1,
    "eval_steps": 10,
    "save_steps": 25,
}


# ============================================================================
# PPO (Proximal Policy Optimization) Configuration
# ============================================================================
# Classic RLHF - more complex but battle-tested
# Requires a reward model

PPO_CONFIG = {
    # --- PPO Specific ---
    "kl_penalty": "kl",             # kl, abs, mse, full
    "init_kl_coef": 0.2,
    "target_kl": 6.0,
    "clip_range": 0.2,
    "vf_coef": 0.1,
    "max_grad_norm": 1.0,
    "gamma": 1.0,
    "lam": 0.95,

    # --- Generation ---
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,

    # --- Training ---
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1.41e-5,
    "optim": "adamw_8bit",

    # --- Logging ---
    "logging_steps": 1,
}


# ============================================================================
# Inference Optimization Configuration
# ============================================================================

INFERENCE_CONFIG = {
    # --- vLLM (PagedAttention) ---
    "vllm": {
        "enabled": True,
        "gpu_memory_utilization": 0.90,    # Use 90% of GPU memory
        "max_model_len": 2048,
        "dtype": "auto",                    # auto, float16, bfloat16
        "quantization": "awq",              # None, awq, gptq, squeezellm
        "enforce_eager": False,             # False = use CUDA graphs
        "enable_prefix_caching": True,      # Cache common prefixes
        "max_num_batched_tokens": 4096,     # Continuous batching budget
        "max_num_seqs": 64,                 # Max concurrent sequences
        "swap_space": 4,                    # CPU swap space (GB)
    },

    # --- Speculative Decoding ---
    "speculative_decoding": {
        "enabled": True,
        "draft_model": "unsloth/llama-3-8b-bnb-4bit",  # Small draft model
        "num_speculative_tokens": 5,         # Tokens to speculate ahead
        "speculative_max_model_len": 2048,
    },

    # --- Flash Attention ---
    "flash_attention": {
        "enabled": True,                     # Auto-detect GPU capability
        "sliding_window": None,              # None = full attention
    },

    # --- KV Cache ---
    "kv_cache": {
        "block_size": 16,                    # PagedAttention block size
        "enable_chunked_prefill": True,      # Chunk long prefills
    },

    # --- Quantization Options ---
    "quantization": {
        "method": "awq",                     # awq, gptq, bnb, gguf
        "bits": 4,                           # 4-bit quantization
        "group_size": 128,                   # Quantization group size
        "zero_point": True,                  # Asymmetric quantization
    },

    # --- Generation Defaults ---
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    },
}


# ============================================================================
# Preference Data Generation Configuration
# ============================================================================

PREFERENCE_CONFIG = {
    # How many preference pairs to generate
    "num_pairs": 2000,

    # Quality criteria for "chosen" (good) responses
    "chosen_criteria": {
        "min_recommendations": 3,
        "max_recommendations": 5,
        "require_year": True,
        "require_genre_match": True,
        "require_plot_description": True,
        "min_response_length": 200,
    },

    # Degradation strategies for "rejected" (bad) responses
    "rejection_strategies": [
        "wrong_genre",           # Recommend wrong genre movies
        "too_few_recs",          # Only 1 recommendation
        "no_details",            # No plot/year/rating info
        "repetitive",            # Repeat same movie info
        "off_topic",             # Ignore user query
        "factually_wrong",       # Wrong year/director/cast
    ],

    # Train/val split for preference data
    "train_split": 0.9,
    "val_split": 0.1,
}
