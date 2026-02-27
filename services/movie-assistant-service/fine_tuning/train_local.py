"""
============================================================================
Movie Discovery Assistant - Optimized Local Training Script
============================================================================
GPU: NVIDIA RTX 4080 SUPER (16GB VRAM, Compute 8.9 Ada Lovelace)

Training Optimizations:
  1. Flash Attention 2      - Native on Ada Lovelace (2-4x faster)
  2. 4-bit QLoRA             - NF4 quantization + double quantization
  3. Sequence Packing        - Pack short sequences together (~5x throughput)
  4. NEFTune                 - Noisy embeddings (+2-5% quality)
  5. rsLoRA                  - Rank-stabilized LoRA (higher rank, stable)
  6. 8-bit AdamW             - Memory-efficient optimizer
  7. Cosine Annealing LR     - Better convergence than linear
  8. BF16 Mixed Precision    - Native BF16 on Ada Lovelace
  9. Gradient Checkpointing  - Trade compute for VRAM
  10. tf32 for matrix ops    - Faster matmul on Ada Lovelace
============================================================================
"""

import os
import sys
import time
import json
import torch
import logging
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================================
# Setup
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force unbuffered stdout/stderr for live monitoring
import functools
print = functools.partial(print, flush=True)

# Redirect all output to a log file for live tailing
LOG_FILE = Path(__file__).parent / "outputs" / "training_live.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

class TeeOutput:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()

sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
sys.stderr = TeeOutput(LOG_FILE.with_suffix(".err.log"), sys.stderr)

# Enable TF32 for faster matmul on Ada Lovelace
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_DIR = BASE_DIR / "fine_tuning" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Pre-quantized for fast download
# Fallback if unsloth model not available
FALLBACK_MODEL = "meta-llama/Meta-Llama-3-8B"

# ============================================================================
# GPU Info
# ============================================================================
def print_gpu_info():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    major, minor = torch.cuda.get_device_capability()
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print(f"  GPU:              {gpu_name}")
    print(f"  VRAM:             {gpu_mem:.1f} GB")
    print(f"  Compute:          {major}.{minor}")
    print(f"  Flash Attention:  {'YES' if major >= 8 else 'NO'}")
    print(f"  BF16:             {'YES' if major >= 8 else 'NO'}")
    print(f"  TF32:             {'YES' if major >= 8 else 'NO'}")
    print(f"  PyTorch:          {torch.__version__}")
    print(f"  CUDA:             {torch.version.cuda}")
    print("=" * 60)
    return gpu_mem, major

# ============================================================================
# Load Model with Quantization + Flash Attention 2
# ============================================================================
def load_model(model_name=MODEL_NAME):
    logger.info(f"Loading model: {model_name}")

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 (better than INT4)
        bnb_4bit_compute_dtype=torch.bfloat16, # BF16 compute on Ada
        bnb_4bit_use_double_quant=True,        # Nested quantization (saves ~0.4GB)
    )

    # Load model with Flash Attention 2
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",  # Flash Attention 2
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        logger.info("Flash Attention 2: ENABLED")
    except Exception as e:
        logger.warning(f"Flash Attention 2 not available ({e}), using SDPA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",  # Scaled Dot Product Attention (PyTorch native)
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        logger.info("SDPA Attention: ENABLED (Flash Attention fallback)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for k-bit training (freeze base, enable adapters)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    logger.info(f"Model loaded: {model_name} (4-bit quantized)")
    return model, tokenizer

# ============================================================================
# Apply LoRA with rsLoRA
# ============================================================================
def apply_lora(model, rank=32):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,              # Alpha = 2 * rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=0.0,                  # No dropout (optimized)
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,                   # rsLoRA: rank-stabilized
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA applied: rank={rank}, alpha={rank*2}, rsLoRA=True")
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model

# ============================================================================
# Prepare Dataset with Packing
# ============================================================================
def prepare_dataset(tokenizer, dataset_dir=DATASET_DIR):
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"

    logger.info(f"Loading datasets from {dataset_dir}")
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    val_dataset = load_dataset("json", data_files=str(val_path), split="train")

    logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    # Alpaca prompt format
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    eos_token = tokenizer.eos_token

    def formatting_func(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = alpaca_prompt.format(instruction, inp, output) + eos_token
            texts.append(text)
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_func, batched=True)
    val_dataset = val_dataset.map(formatting_func, batched=True)

    # Analyze sequence lengths
    sample_texts = train_dataset["text"][:200]
    lengths = [len(tokenizer.encode(t)) for t in sample_texts]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    logger.info(f"Sequence lengths - avg: {avg_len:.0f}, max: {max_len}")
    logger.info(f"Packing efficiency: ~{2048 / avg_len:.1f}x sequences per batch slot")

    return train_dataset, val_dataset

# ============================================================================
# Training
# ============================================================================
def train(model, tokenizer, train_dataset, val_dataset, output_dir=OUTPUT_DIR):
    max_seq_length = 512               # Max observed is 312; 512 gives headroom without waste
    batch_size = 4                     # 4 fits in 16GB with 512 seq len (no packing)
    eval_batch_size = 1                # Eval uses more VRAM (no grad checkpointing)
    grad_accum = 4                     # Effective batch = 4 * 4 = 16

    # SFTConfig combines TrainingArguments + SFT-specific params (trl >= 0.17)
    sft_config = SFTConfig(
        output_dir=str(output_dir),

        # --- Batch ---
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,

        # --- Schedule ---
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",        # Cosine annealing

        # --- Optimizer ---
        optim="adamw_8bit",                # 8-bit AdamW
        weight_decay=0.01,
        max_grad_norm=1.0,

        # --- Precision ---
        bf16=True,                          # BF16 on Ada Lovelace
        fp16=False,
        tf32=True,                          # TF32 matmul

        # --- Gradient Checkpointing ---
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # --- Logging & Eval ---
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,                    # Eval every 100 steps (fewer evals = less OOM risk)
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # --- Performance ---
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        group_by_length=True,              # Group similar lengths for efficiency

        # --- SFT Specific ---
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,                     # Disabled: causes OOM with 2048 seq on 16GB
        neftune_noise_alpha=5,             # OPTIMIZATION: NEFTune (+2-5% quality)

        # --- Misc ---
        seed=3407,
        report_to="none",
        logging_dir=str(output_dir / "logs"),
    )

    # Live log callback - writes each log entry to a file for tailing
    from transformers import TrainerCallback

    log_file = output_dir / "train_log.jsonl"

    class LiveLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {"step": state.global_step, "epoch": round(state.epoch, 2), **logs}
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                # Also print to stdout immediately
                loss_str = f"loss={logs.get('loss', 'N/A')}"
                lr_str = f"lr={logs.get('learning_rate', 'N/A')}"
                print(f"  [Step {state.global_step}/{state.max_steps}] {loss_str}, {lr_str}, epoch={round(state.epoch, 2)}", flush=True)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                entry = {"step": state.global_step, "type": "eval", **metrics}
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                print(f"  [EVAL Step {state.global_step}] eval_loss={metrics.get('eval_loss', 'N/A')}", flush=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=[LiveLogCallback()],
    )

    # Print full config
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model:              Llama 3 8B (4-bit QLoRA)")
    print(f"  Flash Attention 2:  ENABLED")
    print(f"  Sequence Packing:   DISABLED (OOM on 16GB with packing)")
    print(f"  Group by Length:    ENABLED")
    print(f"  NEFTune (alpha=5):  ENABLED")
    print(f"  rsLoRA (rank=32):   ENABLED")
    print(f"  Optimizer:          AdamW 8-bit")
    print(f"  LR Schedule:        Cosine ({sft_config.learning_rate})")
    print(f"  Precision:          BF16 + TF32")
    print(f"  Grad Checkpointing: ENABLED")
    print(f"  Batch:              {batch_size} x {grad_accum} (effective: {batch_size*grad_accum})")
    print(f"  Epochs:             {sft_config.num_train_epochs}")
    print(f"  Train examples:     {len(train_dataset)}")
    print(f"  Val examples:       {len(val_dataset)}")
    print(f"  Output:             {output_dir}")
    print("=" * 60 + "\n")

    # Train
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    stats = trainer.train()

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Time:          {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"  Train loss:    {stats.training_loss:.4f}")
    print(f"  Peak VRAM:     {peak_mem:.1f} GB / 17.2 GB ({peak_mem/17.2*100:.0f}%)")
    print(f"  Steps:         {stats.global_step}")
    print(f"  Samples/sec:   {stats.global_step * batch_size * grad_accum / elapsed:.1f}")
    print(f"{'=' * 60}")

    # Save final model
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")

    # Save training stats
    stats_dict = {
        "training_loss": stats.training_loss,
        "global_step": stats.global_step,
        "elapsed_minutes": round(elapsed / 60, 1),
        "peak_vram_gb": round(peak_mem, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    return trainer, stats


# ============================================================================
# Testing / Evaluation
# ============================================================================
def test_model(model, tokenizer, output_dir=OUTPUT_DIR, num_samples=20):
    """Test the fine-tuned model on test set examples and compute metrics."""
    import random
    from rouge_score import rouge_scorer

    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60 + "\n")

    # Load test set
    test_path = DATASET_DIR / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: Test set not found at {test_path}")
        return

    test_data = load_dataset("json", data_files=str(test_path), split="train")
    print(f"Test set: {len(test_data)} examples")

    # Sample subset for qualitative eval
    indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))
    samples = [test_data[i] for i in indices]

    # Enable inference mode
    model.eval()
    model.config.use_cache = True

    results = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    print(f"Running inference on {len(samples)} test samples...\n")

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        expected = sample["output"]

        # Format prompt like training data
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Compute ROUGE scores
        scores = scorer.score(expected, generated)

        result = {
            "instruction": instruction,
            "expected": expected[:200] + "..." if len(expected) > 200 else expected,
            "generated": generated[:200] + "..." if len(generated) > 200 else generated,
            "rouge1_f": round(scores['rouge1'].fmeasure, 4),
            "rouge2_f": round(scores['rouge2'].fmeasure, 4),
            "rougeL_f": round(scores['rougeL'].fmeasure, 4),
        }
        results.append(result)

        print(f"  [{i+1}/{len(samples)}] ROUGE-L: {result['rougeL_f']:.3f}")
        if i < 3:  # Show first 3 examples in detail
            print(f"    Q: {instruction[:80]}")
            print(f"    A: {generated[:120]}...")
            print()

    # Aggregate metrics
    avg_rouge1 = sum(r["rouge1_f"] for r in results) / len(results)
    avg_rouge2 = sum(r["rouge2_f"] for r in results) / len(results)
    avg_rougeL = sum(r["rougeL_f"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated:  {len(results)}")
    print(f"  ROUGE-1 (avg):      {avg_rouge1:.4f}")
    print(f"  ROUGE-2 (avg):      {avg_rouge2:.4f}")
    print(f"  ROUGE-L (avg):      {avg_rougeL:.4f}")
    print("=" * 60 + "\n")

    # Also compute test set loss using trainer eval
    print("Computing test set loss...")
    torch.cuda.empty_cache()  # Free VRAM from generation

    test_loss = "N/A"
    test_acc = "N/A"
    try:
        test_formatted = test_data.map(
            lambda x: {"text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{x['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{x['output']}<|eot_id|>"},
            num_proc=2,
        )

        from trl import SFTTrainer, SFTConfig as _SFTConfig
        test_config = _SFTConfig(
            output_dir=str(output_dir / "test_eval"),
            per_device_eval_batch_size=1,
            bf16=True,
            dataset_text_field="text",
            max_seq_length=512,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
        )
        test_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=test_formatted,   # SFTTrainer requires train_dataset
            eval_dataset=test_formatted,
            args=test_config,
        )
        test_metrics = test_trainer.evaluate()
        test_loss = test_metrics.get("eval_loss", "N/A")
        test_acc = test_metrics.get("eval_mean_token_accuracy", "N/A")
    except Exception as e:
        print(f"  Warning: Test loss computation failed ({e}), skipping.")

    print(f"\n  Test Set Loss:      {test_loss}")
    print(f"  Test Token Accuracy: {test_acc}")
    print("=" * 60 + "\n")

    # Save results
    eval_results = {
        "num_samples": len(results),
        "avg_rouge1": round(avg_rouge1, 4),
        "avg_rouge2": round(avg_rouge2, 4),
        "avg_rougeL": round(avg_rougeL, 4),
        "test_loss": float(test_loss) if isinstance(test_loss, float) else test_loss,
        "test_token_accuracy": float(test_acc) if isinstance(test_acc, float) else test_acc,
        "samples": results,
    }
    eval_path = output_dir / "test_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to: {eval_path}")

    return eval_results


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Skip training, only run test evaluation")
    parser.add_argument("--test-samples", type=int, default=20, help="Number of test samples for generation eval")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MOVIE DISCOVERY ASSISTANT - OPTIMIZED TRAINING")
    print("=" * 60 + "\n")

    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: No GPU found. Training requires CUDA GPU.")
        sys.exit(1)

    gpu_mem, compute_major = print_gpu_info()

    if args.test_only:
        # Load the fine-tuned model for testing
        final_model_dir = OUTPUT_DIR / "final_model"
        if not final_model_dir.exists():
            print(f"ERROR: No trained model found at {final_model_dir}")
            print("Run training first (without --test-only)")
            sys.exit(1)

        print(f"Loading fine-tuned model from {final_model_dir}...")
        from peft import PeftModel

        # Load base model
        model, tokenizer = load_model(MODEL_NAME)
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, str(final_model_dir))
        print("Fine-tuned model loaded.\n")

        test_model(model, tokenizer, num_samples=args.test_samples)
    else:
        # Full training + testing pipeline
        # Load model
        try:
            model, tokenizer = load_model(MODEL_NAME)
        except Exception as e:
            logger.warning(f"Could not load {MODEL_NAME}: {e}")
            logger.info(f"Trying fallback: {FALLBACK_MODEL}")
            model, tokenizer = load_model(FALLBACK_MODEL)

        # Apply LoRA
        lora_rank = 64 if gpu_mem >= 40 else 32 if gpu_mem >= 15 else 16
        model = apply_lora(model, rank=lora_rank)

        # Prepare data
        train_dataset, val_dataset = prepare_dataset(tokenizer)

        # Train
        trainer, stats = train(model, tokenizer, train_dataset, val_dataset)

        print(f"\nTraining finished! Model saved to: {OUTPUT_DIR / 'final_model'}")

        # Run test evaluation
        test_model(model, tokenizer, num_samples=args.test_samples)

        print("Next steps:")
        print("  1. Run DPO alignment:  python dpo_train_local.py")
        print("  2. Run GRPO alignment: python grpo_train_local.py")
        print("  3. Export to GGUF for Ollama deployment")


if __name__ == "__main__":
    main()
