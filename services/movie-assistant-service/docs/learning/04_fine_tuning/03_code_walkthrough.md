# Code Walkthrough: Fine-Tuning with Unsloth

This document explains the `llama_finetune_colab.ipynb` notebook, which uses **Unsloth** to fine-tune Llama 3 on a free Tesla T4 GPU.

## 1. Setup & Installation
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```
**Why?**
- **Unsloth:** A library that rewrites PyTorch's backpropagation to be 2x faster and use 70% less memory.
- **xformers:** Optimizes attention mechanisms (Flash Attention).
- **bitsandbytes:** Enables 4-bit quantization (loading a huge model in small memory).

## 2. Loading the Model (4-bit Quantization)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```
**Key Concepts:**
- **`load_in_4bit=True`**: Loads the 8 Billion parameters compressed into 4-bit integers (instead of 16-bit floats). This reduces VRAM usage from ~16GB to ~5.5GB.
- **`max_seq_length=2048`**: The maximum number of tokens the model can read at once.

## 3. Adding LoRA Adapters
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ...],
    lora_alpha = 16,
)
```
**Key Parameters:**
- **`r=16` (Rank):** The size of the "adapter" matrices. Higher = smarter but slower/larger. 16 is a sweet spot.
- **`target_modules`**: We attach adapters to ALL attention layers (`q, k, v, o`) and feed-forward layers (`gate, up, down`). This gives better quality than just doing attention layers.

## 4. Data Formatting
```python
alpaca_prompt = """Below is an instruction...
### Instruction:
{}
### Response:
{}"""
```
**Why?**
The model needs a consistent structure to understand where the user ends and the assistant begins. We use the standard **Alpaca Format** which maps perfectly to our `train.jsonl`.

## 5. Training (SFTTrainer)
```python
trainer = SFTTrainer(
    model = model,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        learning_rate = 2e-4,
        optim = "adamw_8bit",
    ),
)
```
**Optimization Tricks:**
- **`per_device_train_batch_size = 2`**: Small batch size to fit in memory.
- **`gradient_accumulation_steps = 4`**: We pretend to have a batch size of 8 (2 * 4) by accumulating gradients before updating weights.
- **`optim = "adamw_8bit"`**: Uses 8-bit optimizer states to save even more memory.

## 6. Inference (Testing)
```python
FastLanguageModel.for_inference(model)
model.generate(..., repetition_penalty = 1.3)
```
**Why `for_inference`?**
This enables Unsloth's native inference engine, which is 2x faster than standard Hugging Face generation. We also added `repetition_penalty` to fix the looping issues we saw in testing.
