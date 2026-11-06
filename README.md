# GPT from Scratch: First-Principles Transformer Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)%20%7C%20CUDA%20%7C%20CPU-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interactive Demo](https://img.shields.io/badge/Interactive%20Walkthrough-Live%20Demo-brightgreen.svg)](https://yourusername.github.io/your-repo-name/)

> **A complete, modular Decoder-only Transformer Language Model built from scratch in pure PyTorch.**  
> Inspired by Andrej Karpathy's *Zero to Hero* series, enhanced with a modular package architecture, advanced decoding strategies (Temperature, Top-$K$, Top-$P$), attention heatmap visualization, multi-dataset training, and an interactive browser-based visual simulation.

---

## 🌟 Key Highlights

* **100% From-Scratch Implementation**: Zero reliance on HuggingFace, LangChain, or external LLM libraries. Every Query, Key, Value matrix, Causal Mask, and LayerNorm is implemented in pure PyTorch.
* **Empirical Validation**: Solves the "Bigram memory ceiling" ($2.46$ loss wall) by driving validation loss down to **$1.61$** in under 90 seconds.
* **Production Decoding Strategies**: Configurable **Temperature**, **Top-$K$ truncation**, and **Top-$P$ (Nucleus) sampling**.
* **Attention Map Visualizer**: In-terminal ASCII heatmap rendering of multi-head causal attention affinities.
* **Multi-Dataset Training**: Pre-configured support for **Shakespeare** and **Python CodeGPT**.
* **Interactive Visual Walkthrough**: Fully self-contained web animation in `docs/index.html` ready for GitHub Pages.

---

## 🧠 Architecture Overview

The model is a modern autoregressive decoder-only Transformer adhering to the pre-LayerNorm formulation (GPT-2 style):

```
 Input Prompt: "ROMEO:\nO, wilt thou leave me"
       │
       ▼
 ┌───────────────┐
 │ Tokenizer     │  Converts characters to discrete IDs via stoi dictionary
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐     ┌──────────────────────┐
 │ Token Embed   │  +  │ Positional Embed     │  (Adds seat coordinates to word meanings)
 └───────┬───────┘     └──────────┬───────────┘
         │                        │
         └───────────► ⨁ ◄────────┘
                       │
                       ▼   x (Residual Stream)
         ┌─────────────────────────────────────┐  ▲
         │ ┌─────────────────────────────────┐ │  │
         │ │ LayerNorm (Calm numbers)        │ │  │
         │ └───────────────┬─────────────────┘ │  │
         │                 ▼                   │  │
         │   Multi-Head Causal Self-Attention  │  │  Stacked N times
         │   (Q, K, V dot product + tril mask) │  │  (Assembly Line)
         │ └───────────────┬─────────────────┘ │  │
         │                 ▼                   │  │
         │       ⨁ (Residual Highway 1)        │  │
         │ └───────────────┬─────────────────┘ │  │
         │                 ▼                   │  │
         │   Feed-Forward Network (FFN)        │  │
         │   (Expand 4x -> ReLU -> Compress)   │  │
         │ └───────────────┬─────────────────┘ │  │
         │                 ▼                   │  │
         │       ⨁ (Residual Highway 2)        │  │
         └─────────────────┬───────────────────┘  ▼
                           │
                           ▼
                  ┌─────────────────┐
                  │ Final LayerNorm │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Linear LM Head  │  Projects vector to vocab_size Logits
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Top-K / Top-P   │  Samples next token & loops autoregressively
                  └─────────────────┘
```

---

## 🔬 Mathematical Mechanisms (Spelled Out)

### 1. Scaled Dot-Product Self-Attention
For a sequence matrix $X \in \mathbb{R}^{B \times T \times C}$, linear transformations project $X$ into Queries ($Q$), Keys ($K$), and Values ($V$):
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

The attention affinity matrix before softmax is computed as:
$$\text{Scores} = \frac{Q K^T}{\sqrt{d_k}} + M$$

Where:
* $\frac{1}{\sqrt{d_k}}$ preserves unit variance and prevents vanishing gradients during softmax.
* $M$ is the **Causal Mask**: an upper-triangular matrix filled with $-\infty$ where $j > i$ to ensure no token looks into future time steps.

$$\text{Attention Weights} = \text{Softmax}(\text{Scores})$$
$$\text{Output} = \text{Attention Weights} \times V$$

### 2. Pre-LayerNorm Residual Highways
Unlike the original 2017 Transformer paper which used Post-LN, this implementation uses the stable modern **Pre-LN formulation**:
$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$
$$x_{l+2} = x_{l+1} + \text{FFN}(\text{LN}(x_{l+1}))$$

This guarantees clean gradient propagation back through the identity path ($x_l$) without vanishing or exploding.

---

## 📊 Empirical Results: Breaking the Bigram Ceiling

During research, a baseline Bigram language model was built to evaluate next-token predictability without context memory:

| Model Architecture | Context Window ($T$) | Trainable Dials | Final Val Loss | Sample Output |
| :--- | :---: | :---: | :---: | :--- |
| **Bigram Model** | $1$ character | $4,225$ | **$2.461$** *(hard plateau)* | `Trdndoun ave t. Thtl! LApu is yfond es...` |
| **Our Transformer** | $64$ characters | $816,705$ | **$1.615$** *(converged)* | `ROMEO:\nO, wilt thou leave me such with of a word...` |

The Bigram model hit a hard statistical wall at **$2.46$** because single-character memory lacks grammatical context. Stacking 4 multi-head Transformer blocks smashed through this wall to **$1.61$**, learning dialogue tags, iambic rhythm, and archaic English structure.

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch
pip install -r requirements.txt
```

### 2. Interactive Text Generation
Generate text using the pre-trained Shakespeare checkpoint:
```bash
python generate.py \
  --prompt "ROMEO:\nO, wilt thou leave me" \
  --temperature 0.8 \
  --top_k 15 \
  --top_p 0.9 \
  --visualize_attention
```

### 3. Training a Model from Scratch
Train on Shakespeare or custom Python code:
```bash
# Train on Shakespeare
python train.py --data data/shakespeare.txt --checkpoint checkpoints/shakespeare_model.pt --iters 2000

# Train a mini CodeGPT on Python algorithms
python train.py --data data/python_code.txt --checkpoint checkpoints/codegpt_model.pt --iters 2500
```

---

## 🔍 In-Terminal Attention Visualization

Pass `--visualize_attention` to inspect how tokens dynamically route attention:

```text
--- Visualizing Self-Attention (Layer 0, Head 0) ---
           h    o    u    ␣    l    e    a    v    e    ␣    m    e
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    h ┃    █    ·    ·    ·    ·    ·    ·    ·    ·    ·    ·    ·
    o ┃    █         ·    ·    ·    ·    ·    ·    ·    ·    ·    ·
    u ┃         █         ·    ·    ·    ·    ·    ·    ·    ·    ·
    ␣ ┃              █         ·    ·    ·    ·    ·    ·    ·    ·
    l ┃                   █         ·    ·    ·    ·    ·    ·    ·
    e ┃                        █         ·    ·    ·    ·    ·    ·
    a ┃                             ▓         ·    ·    ·    ·    ·
    v ┃                                  ▓         ·    ·    ·    ·
    e ┃              ·                        █         ·    ·    ·
    ␣ ┃                                            ░    ░    ·    ·
    m ┃                                                 █         ·
    e ┃                                                      █     

Legend: [·: 0%] [░: 20%] [▒: 40%] [▓: 60%] [█: 80-100%]
```
*(Notice the lower-triangular causal mask: future tokens are strictly masked out with zeros).*

---

## 📂 Project Structure

```
├── data/
│   ├── shakespeare.txt            # 1.1MB Tiny Shakespeare corpus
│   └── python_code.txt            # 320KB real Python data structures/algorithms
├── src/
│   ├── __init__.py
│   ├── model.py                   # Modular GPTLanguageModel, Block, MultiHeadAttention, Head
│   ├── tokenizer.py               # Character Tokenizer with serialization
│   ├── sampling.py                # Temperature, Top-K, Top-P decoding logic
│   └── visualize.py               # In-terminal ASCII attention matrix plotter
├── checkpoints/
│   └── shakespeare_model.pt       # Pre-trained weights
├── docs/
│   └── index.html                 # Interactive visual walkthrough (GitHub Pages ready)
├── train.py                       # CLI training script
├── generate.py                    # CLI generation script
├── requirements.txt
└── README.md
```

---

## 📜 License
MIT License. Feel free to use, modify, and build upon this project for educational or commercial purposes.
