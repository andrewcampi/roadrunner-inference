import torch
import faiss
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==== Config ====
model_name = "gpt2"
num_tokens_to_test = 100
k = 5  # top-k to consider for match

use_ip = True  # Use dot-product (True) or L2 (False)
normalize = False  # Normalize if using inner product

# ==== Load model ====
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# ==== Build FAISS index on LM head weight ====
W = model.lm_head.weight.detach().cpu().numpy().astype("float32")  # [vocab, 768]
vocab_size, dim = W.shape

if normalize and use_ip:
    faiss.normalize_L2(W)

index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
index.add(W)

# ==== Prepare test inputs ====
prompt = "The moon is bright and the stars are visible in the night sky. This makes it a perfect"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids[:, -num_tokens_to_test:]  # Trim if too long
num_tokens = input_ids.shape[1]

# ==== Run benchmark ====
exact_matches = 0
topk_matches = 0
total_faiss_time = 0.0
total_matmul_time = 0.0

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-1]  # [1, seq_len, 768]
    W_T = model.lm_head.weight.data.T  # [768, vocab]

for i in range(num_tokens):
    h = hidden_states[0, i, :].cpu().numpy().astype("float32").reshape(1, -1)
    h_torch = torch.tensor(h).float()

    # ==== Ground Truth (matmul) ====
    start = time.time()
    logits = h_torch @ W_T
    gold_id = torch.argmax(logits, dim=-1).item()
    total_matmul_time += time.time() - start

    # ==== FAISS Routing ====
    if normalize and use_ip:
        faiss.normalize_L2(h)

    start = time.time()
    D, I = index.search(h, k)
    total_faiss_time += time.time() - start

    pred_ids = I[0]
    if gold_id == pred_ids[0]:
        exact_matches += 1
    if gold_id in pred_ids:
        topk_matches += 1

# ==== Results ====
print("\n======= 🧪 ANN Routing Benchmark =======")
print(f"Tested tokens       : {num_tokens}")
print(f"Top-1 match rate    : {exact_matches}/{num_tokens} ({exact_matches / num_tokens:.2%})")
print(f"Top-{k} match rate  : {topk_matches}/{num_tokens} ({topk_matches / num_tokens:.2%})")
print(f"Avg FAISS time      : {1000 * total_faiss_time / num_tokens:.3f} ms/token")
print(f"Avg Matmul time     : {1000 * total_matmul_time / num_tokens:.3f} ms/token")
print(f"Speedup (Matmul/ANN): {total_matmul_time / total_faiss_time:.2f}×")


""" Output:
======= 🧪 ANN Routing Benchmark =======
Tested tokens       : 19
Top-1 match rate    : 19/19 (100.00%)
Top-5 match rate  : 19/19 (100.00%)
Avg FAISS time      : 2.678 ms/token
Avg Matmul time     : 2.349 ms/token
Speedup (Matmul/ANN): 0.88×
"""

""" Analysis:
🧠 Interpretation
Inner Product (IP) is the correct similarity metric.

Normalization hurts performance and accuracy in this case.

So argmax(h @ Wᵗ) ≈ ANN with raw IP, as long as you don’t normalize.

You're replicating the full LM head behavior exactly using FAISS + IP routing. This means:

💡 The LM head is just a token router over dot products — and you just replaced it.
"""