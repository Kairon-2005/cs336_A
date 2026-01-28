# CS336 Assignment 1 — Transformer Language Model from Scratch

This repository contains my implementation of Stanford CS336 Assignment 1 (Spring 2025).
Following the assignment specification strictly, I implement a full Transformer language model from first principles, including tokenizer training, model architecture, optimization, training loop, and decoding.

All components are implemented using low-level PyTorch primitives only, in accordance with the “from-scratch” requirement of the course.

## Project Structure 

cs336_basics/
├── __init__.py
├── bpe.py          # Byte-level BPE tokenizer training
├── tokenizer.py    # Tokenizer encode / decode logic
├── model.py        # Transformer LM architecture
├── optimizer.py    # AdamW optimizer (from scratch)
├── utils.py        # Core utilities (loss, softmax, LR schedule, clipping)
├── train.py        # Training loop & checkpointing
└── decoding.py     # Autoregressive text generation

## Component Overview 

### 1. Byte-Level BPE Tokenizer (bpe.py, tokenizer.py)

Implements a byte-level Byte Pair Encoding (BPE) tokenizer, following Section 2 of the PDF.

Key features
	•	UTF-8 byte encoding (initial vocab size = 256)
	•	GPT-2 style regex pre-tokenization
	•	Deterministic BPE merge procedure
	•	No merges across pre-token or document boundaries
	•	Full support for special tokens (e.g. <|endoftext|>)
	•	Memory-efficient streaming tokenization (encode_iterable)

Responsibilities
	•	bpe.py: BPE training (vocab + merges)
	•	tokenizer.py: encoding / decoding using trained merges


### 2. Transformer Language Model (model.py)

Implements a decoder-only Transformer LM.

Architecture
	•	Token embedding
	•	Stack of pre-norm Transformer blocks
	•	Final RMSNorm
	•	Output projection (LM head)

Transformer block design
	•	RMSNorm → Causal multi-head self-attention → residual
	•	RMSNorm → SwiGLU feed-forward network → residual

Design choices 
	•	Pre-norm architecture
	•	RMSNorm instead of LayerNorm
	•	SwiGLU feed-forward (SiLU + GLU)
	•	Rotary Positional Embeddings (RoPE)
	•	No bias terms in linear layers
	•	Explicit causal masking

### 3. Optimization (optimizer.py, utils.py)

Implements all training utilities from scratch.

Loss
	•	Numerically stable cross-entropy loss
	•	Handles arbitrary batch dimensions

Optimizer
	•	Full AdamW implementation
	•	Correct moment tracking, bias correction, and decoupled weight decay
	•	Per-parameter optimizer state

Learning rate
	•	Cosine annealing schedule with warmup

Stability
	•	Gradient clipping by global $l_2$ norm

### 4. Training Loop (train.py)

Implements the full training pipeline described in Section 5 of the PDF.

Features
	•	Random subsequence sampling from a single token stream
	•	Memory-efficient dataset loading (numpy.memmap)
	•	Device-agnostic training (CPU / MPS / CUDA)
	•	Periodic validation evaluation
	•	Robust checkpoint save / resume
	•	Clean separation of model, optimizer, and scheduler state

### 5. Text Generation (decoding.py)

Implements autoregressive decoding as described in Section 6.

Supported features
	•	Temperature scaling
	•	Top-p (nucleus) sampling
	•	Early stopping on <|endoftext|>
	•	Configurable maximum generation length

This allows qualitative inspection of trained language models.

## Correctness & Testing
	•	All components are implemented to pass the official CS336 test suite
	•	Numerical stability and shape invariants are explicitly handled
	•	Adapter functions isolate test glue from core logic, as intended by the assignment


📌 Notes
	•	This repository follows Stanford CS336 academic guidelines.
	•	No high-level PyTorch abstractions (nn.Linear, nn.Embedding, torch.optim.Adam, etc.) are used.
	•	The implementation is suitable for small- to medium-scale experiments (TinyStories / OpenWebText).
