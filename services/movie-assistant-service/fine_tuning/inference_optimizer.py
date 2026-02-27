"""
============================================================================
Inference Optimization Engine
============================================================================

Production-grade inference optimizations for the Movie Discovery Assistant.

Supports multiple backends with automatic fallback:
  1. vLLM (BEST)       - PagedAttention + continuous batching + CUDA graphs
  2. Unsloth/HF        - Flash Attention 2 + KV cache + torch.compile
  3. HuggingFace API   - Free cloud inference (fallback)

Optimization Techniques:
  - PagedAttention:       Virtual memory for KV cache (no waste)
  - Continuous Batching:  Process requests as they arrive (no padding)
  - Flash Attention 2:    Fused CUDA kernels (2-4x faster, O(N) memory)
  - Speculative Decoding: Small draft model predicts, large model verifies
  - KV Cache:             Stores past key/values (O(1) per new token)
  - Prefix Caching:       Cache system prompt across requests
  - CUDA Graphs:          Capture & replay GPU ops (reduce launch overhead)
  - Quantization:         AWQ/GPTQ 4-bit (2x less memory, minimal quality loss)
  - torch.compile:        Compile model for optimized execution

============================================================================
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InferenceOptConfig:
    """Configuration for inference optimizations."""

    # --- Backend Selection ---
    # Priority: vllm > local_transformers > huggingface_api
    backend: str = "auto"  # "auto", "vllm", "local", "api"

    # --- Model ---
    model_path: str = ""  # Path to model or HF model ID
    dtype: str = "auto"   # "auto", "float16", "bfloat16"

    # --- vLLM Options ---
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 2048
    vllm_quantization: Optional[str] = "awq"  # None, "awq", "gptq"
    vllm_enforce_eager: bool = False           # False = CUDA graphs
    vllm_enable_prefix_caching: bool = True
    vllm_max_num_seqs: int = 64
    vllm_swap_space: int = 4

    # --- Speculative Decoding ---
    speculative_enabled: bool = True
    speculative_draft_model: str = ""
    speculative_num_tokens: int = 5

    # --- Flash Attention ---
    flash_attention_enabled: bool = True

    # --- Generation Defaults ---
    max_new_tokens: int = 512
    temperature: float = 0.4
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 4


# ============================================================================
# vLLM Backend (PagedAttention + Continuous Batching)
# ============================================================================

class VLLMInferenceEngine:
    """
    vLLM-based inference with PagedAttention.

    PagedAttention:
      - Traditional: Pre-allocates contiguous KV cache per sequence
        -> Wastes memory due to fragmentation & over-allocation
      - PagedAttention: Uses virtual memory paging for KV cache
        -> Allocates fixed-size blocks on demand
        -> No fragmentation, 2-4x more sequences in same memory

    Continuous Batching:
      - Traditional: Wait for batch to fill, process together
        -> First-come requests wait for last-to-arrive
      - Continuous: Process new requests as they arrive
        -> Slots freed as sequences finish
        -> ~2x higher throughput

    CUDA Graphs:
      - Captures a sequence of GPU operations once
      - Replays without CPU-GPU synchronization overhead
      - Eliminates kernel launch latency for repeated patterns
    """

    def __init__(self, config: InferenceOptConfig):
        self.config = config
        self.engine = None
        self.tokenizer = None

    def initialize(self):
        """Initialize vLLM engine with all optimizations."""
        try:
            from vllm import LLM, SamplingParams

            engine_args = {
                "model": self.config.model_path,
                "dtype": self.config.dtype,
                "max_model_len": self.config.vllm_max_model_len,
                "gpu_memory_utilization": self.config.vllm_gpu_memory_utilization,
                "enforce_eager": self.config.vllm_enforce_eager,
                "enable_prefix_caching": self.config.vllm_enable_prefix_caching,
                "max_num_seqs": self.config.vllm_max_num_seqs,
                "swap_space": self.config.vllm_swap_space,
            }

            # Quantization
            if self.config.vllm_quantization:
                engine_args["quantization"] = self.config.vllm_quantization

            # Speculative decoding
            if self.config.speculative_enabled and self.config.speculative_draft_model:
                engine_args["speculative_model"] = self.config.speculative_draft_model
                engine_args["num_speculative_tokens"] = self.config.speculative_num_tokens

            self.engine = LLM(**engine_args)
            logger.info("vLLM engine initialized with PagedAttention")
            logger.info(f"  Quantization: {self.config.vllm_quantization or 'none'}")
            logger.info(f"  Prefix caching: {self.config.vllm_enable_prefix_caching}")
            logger.info(f"  CUDA graphs: {not self.config.vllm_enforce_eager}")
            if self.config.speculative_enabled:
                logger.info(f"  Speculative decoding: {self.config.speculative_draft_model}")

            return True
        except ImportError:
            logger.warning("vLLM not installed. Install with: pip install vllm")
            return False
        except Exception as e:
            logger.error(f"vLLM initialization failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with vLLM (PagedAttention + continuous batching)."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
        )

        outputs = self.engine.generate([prompt], params)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation with continuous batching."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
        )

        outputs = self.engine.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


# ============================================================================
# Local Transformers Backend (Flash Attention 2 + torch.compile)
# ============================================================================

class LocalTransformersEngine:
    """
    HuggingFace Transformers with Flash Attention 2 and torch.compile.

    Flash Attention 2:
      - Standard attention: O(N^2) memory, O(N^2) compute
      - Flash Attention 2: O(N) memory, same compute but FUSED
      - Fused kernels: Combines Q*K, softmax, *V into single CUDA kernel
      - Avoids writing N^2 attention matrix to HBM (memory bandwidth bound)
      - Result: 2-4x faster, can handle 4x longer sequences

    torch.compile:
      - Traces the model computation graph
      - Fuses operations (e.g., LayerNorm + Dropout + Add -> single kernel)
      - Eliminates unnecessary memory reads/writes
      - First run is slow (compilation), subsequent runs are ~20% faster
    """

    def __init__(self, config: InferenceOptConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda"

    def initialize(self):
        """Initialize model with Flash Attention 2 and optimizations.
        Supports both full models and LoRA adapter (PEFT) checkpoints.
        """
        try:
            import json
            import os
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Check GPU
            if not torch.cuda.is_available():
                logger.warning("No GPU available, falling back to CPU")
                self.device = "cpu"

            # Detect if this is a LoRA adapter checkpoint
            adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")
            is_lora = os.path.exists(adapter_config_path)

            # Quantization config (4-bit for memory efficiency)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            load_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            # Determine attention implementation
            if self.config.flash_attention_enabled:
                try:
                    import flash_attn
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2: ENABLED")
                except ImportError:
                    logger.info("flash_attn not installed, using default attention (sdpa)")

            if is_lora:
                # LoRA adapter: load base model first, then apply adapters
                from peft import PeftModel
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                base_model_id = adapter_cfg.get("base_model_name_or_path")
                logger.info(f"LoRA adapter detected. Base model: {base_model_id}")
                logger.info("Loading base model (4-bit)...")
                base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
                logger.info("Applying LoRA adapters...")
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                logger.info("Fine-tuned Llama 3 (LoRA) loaded successfully!")
            else:
                # Full model: load directly
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path, **load_kwargs
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model loaded: {self.config.model_path}")
            return True

        except Exception as e:
            logger.error(f"Local model initialization failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with Flash Attention 2 + KV cache."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", self.config.no_repeat_ngram_size),
                use_cache=True,     # KV cache
                do_sample=True,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# HuggingFace API Backend (Fallback)
# ============================================================================

class HuggingFaceAPIEngine:
    """HuggingFace Inference API as fallback (no GPU needed)."""

    def __init__(self, config: InferenceOptConfig):
        self.config = config
        self.client = None

    def initialize(self):
        """Initialize HF Inference API client."""
        try:
            from huggingface_hub import InferenceClient

            token = os.getenv("HF_API_TOKEN")
            if not token:
                logger.warning("HF_API_TOKEN not set")
                return False

            self.client = InferenceClient(
                model=self.config.model_path or "mistralai/Mistral-7B-Instruct-v0.2",
                token=token,
            )
            logger.info("HuggingFace API client initialized")
            return True
        except Exception as e:
            logger.error(f"HF API initialization failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate via HF Inference API."""
        return self.client.text_generation(
            prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            return_full_text=False,
        )


# ============================================================================
# Unified Inference Manager (Auto-selects best backend)
# ============================================================================

class InferenceManager:
    """
    Manages inference with automatic backend selection and fallback.

    Priority:
      1. vLLM (if GPU available + vllm installed)
      2. Local Transformers (if GPU available)
      3. HuggingFace API (always available with token)
    """

    def __init__(self, config: InferenceOptConfig = None):
        self.config = config or InferenceOptConfig()
        self.engine = None
        self.backend_name = None
        self._metrics = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time_ms": 0,
        }

    def initialize(self) -> bool:
        """Initialize the best available backend."""
        backends = self._get_backend_priority()

        for name, engine_class in backends:
            logger.info(f"Trying backend: {name}")
            engine = engine_class(self.config)
            if engine.initialize():
                self.engine = engine
                self.backend_name = name
                logger.info(f"Using backend: {name}")
                return True
            logger.info(f"Backend {name} not available, trying next...")

        logger.error("No inference backend available!")
        return False

    def _get_backend_priority(self):
        """Get ordered list of backends to try."""
        if self.config.backend == "vllm":
            return [("vllm", VLLMInferenceEngine)]
        elif self.config.backend == "local":
            return [("local", LocalTransformersEngine)]
        elif self.config.backend == "api":
            return [("api", HuggingFaceAPIEngine)]
        else:  # auto
            return [
                ("vllm", VLLMInferenceEngine),
                ("local", LocalTransformersEngine),
                ("api", HuggingFaceAPIEngine),
            ]

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with metrics tracking."""
        if self.engine is None:
            raise RuntimeError("No inference engine initialized. Call initialize() first.")

        start_time = time.time()
        response = self.engine.generate(prompt, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000

        # Track metrics
        self._metrics["total_requests"] += 1
        self._metrics["total_time_ms"] += elapsed_ms
        tokens = len(response.split())  # Approximate
        self._metrics["total_tokens_generated"] += tokens

        return {
            "text": response,
            "backend": self.backend_name,
            "latency_ms": round(elapsed_ms, 1),
            "tokens_generated": tokens,
            "tokens_per_second": round(tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
        }

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Batch generation (vLLM only, falls back to sequential)."""
        if hasattr(self.engine, "generate_batch"):
            start_time = time.time()
            responses = self.engine.generate_batch(prompts, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            return [{
                "text": r,
                "backend": self.backend_name,
                "latency_ms": round(elapsed_ms / len(prompts), 1),
            } for r in responses]
        else:
            # Sequential fallback
            return [self.generate(p, **kwargs) for p in prompts]

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference performance metrics."""
        avg_latency = 0
        avg_throughput = 0
        if self._metrics["total_requests"] > 0:
            avg_latency = self._metrics["total_time_ms"] / self._metrics["total_requests"]
            total_seconds = self._metrics["total_time_ms"] / 1000
            avg_throughput = self._metrics["total_tokens_generated"] / total_seconds if total_seconds > 0 else 0

        return {
            "backend": self.backend_name,
            "total_requests": self._metrics["total_requests"],
            "avg_latency_ms": round(avg_latency, 1),
            "avg_tokens_per_second": round(avg_throughput, 1),
            "total_tokens_generated": self._metrics["total_tokens_generated"],
        }


# ============================================================================
# Speculative Decoding Helper
# ============================================================================

class SpeculativeDecoder:
    """
    Speculative Decoding: Use a small FAST model to draft tokens,
    then verify with the large ACCURATE model in a single pass.

    How it works:
      1. Draft model generates K tokens autoregressively (fast)
      2. Target model verifies ALL K tokens in one forward pass (parallel)
      3. Accept tokens that match, reject from first mismatch
      4. Net result: Generate multiple tokens per target model forward pass

    Speedup depends on acceptance rate:
      - High acceptance (~80%): ~3-4x speedup
      - Medium acceptance (~50%): ~2x speedup
      - Low acceptance (<30%): No benefit (draft model too different)

    Best draft models for Llama 3 8B:
      - TinyLlama 1.1B (same architecture, similar tokenizer)
      - Llama 3.2 1B (same family, best acceptance rate)
    """

    def __init__(self, target_model, draft_model, tokenizer, num_speculative_tokens: int = 5):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self._acceptance_count = 0
        self._total_speculated = 0

    def generate(self, input_ids, max_new_tokens: int = 256, **kwargs):
        """
        Speculative decoding generation loop.

        For use in custom training/evaluation scripts where you have
        direct access to both models. For production, use vLLM's built-in
        speculative decoding (configured via InferenceOptConfig).
        """
        import torch

        generated = input_ids.clone()
        temperature = kwargs.get("temperature", 0.7)

        for _ in range(0, max_new_tokens, self.num_speculative_tokens):
            # Step 1: Draft model generates K tokens
            draft_tokens = []
            draft_input = generated.clone()

            with torch.inference_mode():
                for _ in range(self.num_speculative_tokens):
                    draft_output = self.draft_model(draft_input)
                    draft_logits = draft_output.logits[:, -1, :] / temperature
                    draft_probs = torch.softmax(draft_logits, dim=-1)
                    draft_token = torch.multinomial(draft_probs, 1)
                    draft_tokens.append(draft_token)
                    draft_input = torch.cat([draft_input, draft_token], dim=-1)

            # Step 2: Target model verifies in one pass
            candidate = torch.cat([generated] + draft_tokens, dim=-1)

            with torch.inference_mode():
                target_output = self.target_model(candidate)
                target_logits = target_output.logits / temperature

            # Step 3: Accept/reject
            accepted = 0
            for i, draft_token in enumerate(draft_tokens):
                pos = generated.shape[1] + i - 1
                target_probs = torch.softmax(target_logits[:, pos, :], dim=-1)
                draft_prob = target_probs[0, draft_token[0, 0]].item()

                # Accept if probability is reasonable
                if draft_prob > 0.1 or draft_token[0, 0] == torch.argmax(target_logits[:, pos, :]):
                    accepted += 1
                    self._acceptance_count += 1
                else:
                    break
                self._total_speculated += 1

            # Add accepted tokens
            if accepted > 0:
                generated = torch.cat(
                    [generated] + draft_tokens[:accepted], dim=-1
                )

            # Add one target model token after accepted sequence
            next_logits = target_logits[:, generated.shape[1] - 1, :]
            next_probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(next_probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if next_token[0, 0] == self.tokenizer.eos_token_id:
                break

        return generated

    @property
    def acceptance_rate(self) -> float:
        if self._total_speculated == 0:
            return 0.0
        return self._acceptance_count / self._total_speculated
