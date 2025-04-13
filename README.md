# <img src="images/logo_crop.png" alt="RoadRunner Logo" width="75" /> RoadRunner: Large Matmul Free Transformer Inference via SVD Adaptive Routing and Dot Products

> A novel architecture for accelerating transformer inference without retraining, using SVD-based adaptive routing and dot product prediction. Near 100% accuracy, all without any model weight modifications required.

---

## What is RoadRunner?

**RoadRunner** is a high-efficiency inference engine for transformer models like GPT-2 and LLaMA-3.2-1B. It bypasses large matrix multiplications in MLP blocks and LM heads using **Singular Value Decomposition (SVD)** and **adaptive residual routing** — all while preserving near-perfect output quality.

---

## Key Insights

- **SVD-Based Routing**: Decomposes MLP weight matrices to create efficient, low-rank computation paths.
- **Token Embedding Alignment**: Shows transformer hidden states naturally align with correct token embeddings (>0.99 cosine similarity).
- **Matrix-Free LM Head**: Replaces expensive vocabulary projection with lightweight dot-product prediction and reranking.
- **Layerwise Alpha Blending**: Uses minimal routing contributions (as low as 5%) to maintain output fidelity.

---

## Results

| Model          | Speedup | Token Match | Cosine Similarity |
|----------------|---------|-------------|-------------------|
| GPT-2          | 1.57×   | 99%         | >0.99             |
| LLaMA-3.2-1B   | 1.57×   | 99%         | >0.99             |

> Achieved in a simple PoC without quantization, compiling, or any weight modification.
