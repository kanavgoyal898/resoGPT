# resoGPT: GPT-2 Style Language Model

## Overview

This project is a clean, from-scratch implementation of a **GPT-2 style autoregressive transformer** built using PyTorch. Unlike character-level toy models, this implementation operates on **tokenized text (via `tiktoken`)** and supports **modern training features** such as:

* Multi-head causal self-attention (with Flash Attention)
* Residual connections and layer normalization
* Weight tying between embeddings and output layer
* Distributed Data Parallel (DDP) training
* Gradient accumulation for large effective batch sizes
* Cosine learning rate scheduling with warmup
* Mixed precision training (bfloat16)
* Optional `torch.compile` acceleration

**resoGPT** is trained on `tiny_shakespeare.txt` and can also **load pretrained GPT-2 weights** from Hugging Face.

<div style="text-align: center;">
  <img src="./image.png" alt="Preview" style="width: 64%;">
</div>

## Architecture

This implementation closely follows the original GPT-2 design:

* **Token Embeddings (`wte`)**
* **Positional Embeddings (`wpe`)**
* **Stack of Transformer Blocks**

  * LayerNorm
  * Causal Self-Attention
  * Feedforward (MLP)
* **Final LayerNorm (`ln_f`)**
* **Language Modeling Head (`lm_head`)**

### Key Design Choices

* **Causal Masking** ensures tokens only attend to past context
* **Flash Attention (`scaled_dot_product_attention`)** for efficient computation
* **Weight Sharing** between embedding and output projection
* **Residual Connections** for stable deep training

## Core Components

### 1. `GPTConfig`

Defines model hyperparameters:

* `block_size`: context length
* `vocab_size`: tokenizer vocabulary size
* `n_layer`: number of transformer blocks
* `n_head`: number of attention heads
* `n_embd`: embedding dimension

### 2. `CausalSelfAttention`

Implements multi-head masked self-attention:

* Projects input into **Q, K, V**
* Uses **scaled dot-product attention**
* Applies causal masking via `is_causal=True`
* Merges heads and applies output projection

### 3. `MLP`

Feedforward network:

* Expands embedding dimension by 4×
* Uses **GELU activation**
* Projects back to original size

### 4. `Block`

Transformer block:

* LayerNorm → Attention → Residual
* LayerNorm → MLP → Residual

### 5. `resoGPT`

Main model:

* Embedding layers + stacked blocks
* Final normalization + output head
* Computes logits and cross-entropy loss
* Includes:

  * Custom weight initialization
  * Optimizer configuration
  * Pretrained weight loading (GPT-2)

### 6. `DataLoader`

Simple sequential data loader:

* Tokenizes `tiny_shakespeare.txt` using `tiktoken`
* Produces `(x, y)` pairs for next-token prediction
* Supports multi-process sharding for DDP

## Training Pipeline

### Features

* **Gradient Accumulation**

  * Simulates very large batch sizes (~500K tokens)

* **Distributed Training (DDP)**

  * Multi-GPU support via `torchrun`

* **Mixed Precision**

  * Uses `torch.autocast` with `bfloat16`

* **Learning Rate Scheduler**

  * Linear warmup
  * Cosine decay

* **Gradient Clipping**

  * Prevents exploding gradients

### Training Loop

For each step:

1. Fetch batch
2. Forward pass
3. Compute loss
4. Backpropagate (accumulated)
5. Clip gradients
6. Update learning rate
7. Optimizer step

## Usage

### 1. Install Requirements

```bash
pip install torch tiktoken transformers
```

### 2. Prepare Dataset

Place:

```
tiny_shakespeare.txt
```

in the root directory.

### 3. Train (Single GPU / CPU)

```bash
python gpt.py
```

### 4. Train with DDP (Multi-GPU)

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS gpt.py
```

### 5. Load Pretrained GPT-2

```python
model = GPT.from_pretrained('gpt2')
```

Supported:

* `gpt2`
* `gpt2-medium`
* `gpt2-large`
* `gpt2-xl`

### 6. Text Generation

After training, **resoGPT**:

* Samples tokens using **top-k sampling (k=50)**
* Generates multiple sequences

## Hyperparameters

* `batch_size`: ~524K tokens (effective)
* `minbatch_size`: micro-batch size
* `block_size`: 1024 tokens
* `max_steps`: training iterations
* `learning_rate`: dynamic (cosine schedule)
* `weight_decay`: 0.1
* `betas`: (0.9, 0.95)

## Performance Features

* **Flash Attention** → faster + memory efficient
* **torch.compile** → graph optimization (CUDA)
* **Fused AdamW** → faster optimizer (if available)
* **DDP scaling** → near-linear multi-GPU speedup

## References

1. **Vaswani et al. (2017)** — *Attention is All You Need*
   [Link](https://arxiv.org/abs/1706.03762)
   Introduces the transformer architecture, replacing recurrence and convolution with self-attention, enabling parallel processing of sequences and forming the foundation of modern LLMs like resoGPT.

2. **Radford et al. (OpenAI)** — *GPT & GPT-2 Papers*
   [Link](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
   Presents the Generative Pre-trained Transformer paradigm, demonstrating how large-scale unsupervised pretraining followed by fine-tuning can achieve strong performance across diverse NLP tasks.

3. **PyTorch Documentation** — Scaled Dot Product Attention
   [Link](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
   Details the optimized attention primitive used in resoGPT, including efficient implementations (Flash Attention) that improve speed and memory usage during training and inference.

4. **Hugging Face Transformers** — GPT-2 Implementation
   [Link](https://github.com/huggingface/transformers)
   Provides a production-grade reference implementation of GPT-2, which resoGPT interoperates with for loading pretrained weights and validating architectural correctness.


## Example Output

```
> KING:
  What shall we do now, my lord?

> ROMEO:
  The night is young, and so are we...

> JULIET:
  If love be blind, it best agrees with night...
```

## Notes

* **resoGPT** is a **minimal, educational implementation**, not production-optimized
* Designed to clearly show how GPT-style models work end-to-end
* Easily extensible for:

  * Better datasets
  * Larger models
  * Advanced sampling (top-p, temperature)
  * Fine-tuning pipelines

## Conclusion

**resoGPT** demonstrates that a modern GPT-style language model can be implemented in **a few hundred lines of code** while still supporting:

* Pretraining
* Distributed training
* Efficient attention
* Real-world tokenization

It serves as a strong foundation for understanding and building more advanced LLM systems.
