# Interview Prep: LLM Fine-Tuning

## Common Questions & Answers

### Q1: Why did you use PEFT/LoRA instead of full fine-tuning?
**Answer:** Full fine-tuning of an 8B parameter model requires updating all weights, which would need ~100GB+ of VRAM (impossible on consumer hardware).
**LoRA (Low-Rank Adaptation)** allows us to freeze the main model and only train tiny "adapter" matrices (rank 16). This reduces trainable parameters by 99% and memory usage by ~70%, allowing me to train on a free Tesla T4 (16GB VRAM) while achieving comparable performance.

### Q2: What is Quantization and why did you use 4-bit?
**Answer:** Quantization reduces the precision of model weights to save memory.
- **Float16 (Standard):** 2 bytes per weight. 8B model = ~16GB VRAM.
- **4-bit (NF4):** 0.5 bytes per weight. 8B model = ~5.5GB VRAM.
I used 4-bit quantization (QLoRA) to fit the model into the Colab GPU memory, leaving space for the training gradients and batch overhead.

### Q3: How did you handle the "Catastrophic Forgetting" problem?
**Answer:** Catastrophic forgetting happens when a model learns new data but forgets its original knowledge.
**My Approach:**
1.  **LoRA:** Since we freeze the base model, the core knowledge (English, logic) is preserved. We only add a layer on top.
2.  **Low Epochs:** I trained for only 1 epoch (60 steps) to teach *style* without overwriting *knowledge*.
3.  **Low Learning Rate:** Used `2e-4` to make gentle updates.

### Q4: What is the difference between SFT and RLHF?
**Answer:**
- **SFT (Supervised Fine-Tuning):** What I did here. We show the model "Input -> Output" pairs and say "Copy this style." It's good for formatting and tone.
- **RLHF (Reinforcement Learning from Human Feedback):** A more complex step where the model generates multiple answers, a human ranks them, and a Reward Model trains the LLM to maximize that score. This is used for safety and alignment (e.g., ChatGPT), but was overkill for my movie recommender.

### Q5: Why did you use Unsloth?
**Answer:** Unsloth is a library that manually optimizes the PyTorch backward pass (backpropagation). It provides:
1.  **2x Faster Training:** By reducing GPU memory fragmentation.
2.  **Less Memory:** Allows larger batch sizes.
It was essential for making the training feasible on the free tier constraints.

### Q6: How did you evaluate the model?
**Answer:**
1.  **Qualitative Stress Testing:** I ran complex queries ("Sci-fi under 2 hours", "Abstract vibes") to check if it followed instructions.
2.  **Format Verification:** Checked if it consistently used bold titles and bullet points (which the base model didn't do).
3.  **Loss Curves:** Monitored the training loss to ensure it was converging (decreasing) and not overfitting.
