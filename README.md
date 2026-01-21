# GPT-2 Implementation from Scratch in PyTorch

This repository contains a comprehensive, educational implementation of the GPT-2 model, trained on the FineWeb-Edu dataset. Inspired by the "Attention is All You Need" paper and Andrej Karpathy's "nanoGPT" series, this project breaks down the complexities of Large Language Models (LLMs) into digestible layers.

---

## 🚀 Easy: Quick Start & High-Level Overview

### What is GPT-2?
GPT-2 (Generative Pre-trained Transformer 2) is a decoder-only transformer model designed to predict the next token in a sequence. By training on a massive corpus of text, it learns the patterns of human language.

### Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AroopGit/GPT-2-Model-Implementation-from-Scratch.git
   cd GPT-2-Model-Implementation-from-Scratch
   ```
2. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Dataset:**
   ```bash
   python src/prepare_dataset.py
   ```
4. **Run Training:**
   ```bash
   python src/train.py --num_epochs=1
   ```

### Core Repository Structure
- `src/model.py`: The "brain" — contains the transformer architecture.
- `src/train.py`: The "teacher" — handles the training loop and optimization.
- `src/dataloader.py`: The "librarian" — manages data feeding into the model.
- `src/inference.py`: The "interactor" — generates text from a trained model.

### Generating Text (Inference)
Once trained, use the inference script to generate text. Note that the script expects a model checkpoint in the `./logs/` directory (e.g., `model_95365.pt`).
```bash
python src/inference.py --prompt="The future of AI is" --num_seq=3 --max_tokens=100
```

---

## 🧠 Medium: Understanding the Transformer Architecture

If you're comfortable with basic PyTorch, here’s how the magic happens under the hood:

### 1. Multi-Head Self-Attention (`CausalSelfAttention`)
Instead of looking at a sequence as a whole, the model splits its "attention" into multiple heads. This allows it to simultaneously focus on different aspects of the text (e.g., one head for grammar, another for context). We use **Causal Masking** to ensure the model doesn't "cheat" by looking at future tokens during training.

### 2. MLP and Activations
Each transformer block contains a Feed-Forward Network (MLP). We use the **GELU (Gaussian Error Linear Unit)** activation function, specifically the `tanh` approximation used in the original GPT-2 paper.

### 3. Layer Normalization
GPT-2 uses **Pre-LayerNorm**, meaning normalization is applied *before* the attention and MLP layers. This helps in stabilizing the gradients during deep training.

### 4. Weight Tying
To save memory and improve performance, the input embedding weights (`wte`) are shared with the output language modeling head (`lm_head`). This "tying" ensures that the model's representation of tokens remains consistent across input and output.

---

## 🔥 Tough: Advanced Optimizations & Scalability

For those looking to understand production-grade training techniques:

### 1. Distributed Data Parallel (DDP)
We support training across multiple GPUs using PyTorch's DDP. This involves:
- **Gradient Synchronization:** Gradients are averaged across all GPUs using `AllReduce`.
- **Master Process:** Orchestrates logging and checkpointing while other processes focus on computation.

### 2. Computational Speedups
- **Mixed Precision (Bfloat16):** Reduces memory usage and speeds up training without significant loss in accuracy.
- **FlashAttention:** Uses an optimized kernel (`F.scaled_dot_product_attention`) to compute attention in $O(N)$ memory instead of $O(N^2)$.
- **Torch Compile:** Leverages the PyTorch 2.0 compiler to fuse kernels and reduce Python overhead.

### 3. Learning Rate & Scheduling
We implement a **Cosine Decay schedule with Linear Warmup**. 
- **Warmup:** Gradually increases the learning rate to prevent early-stage divergence.
- **Cosine Decay:** Smoothly reduces the learning rate to help the model converge to a better local minimum.

### 4. Gradient Accumulation
Since high-quality training requires large batch sizes (e.g., 0.5M tokens), we use gradient accumulation to simulate these large batches on hardware with limited VRAM.

### 5. Initialization Scaling
As the model gets deeper, residual connections can cause activations to grow. We scale the weights of the projection layers (`c_proj`) by $1/\sqrt{2 \times \text{number of layers}}$ to maintain variance across the network.

---

## 📊 Results & Evaluation
The model is evaluated using:
- **Validation Loss:** Monitored every 250 steps.
- **HellaSwag Accuracy:** A benchmark that tests the model's commonsense reasoning by asking it to complete sentences.

![Training Loss](./assets/loss_eval.png)

---

## 📚 References
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanogpt)
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## 🎖 Acknowledgments
Special thanks to the Open Source community and Andrej Karpathy for making LLM education accessible to everyone.