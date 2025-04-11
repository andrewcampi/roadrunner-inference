import torch
import faiss
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==== Config ====
model_name = "gpt2"
k = 32  # Number of candidates from FAISS
num_tokens_to_test = 100

# ==== Load model ====
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# ==== Prepare weight matrix ====
W = model.lm_head.weight.detach().cpu().numpy().astype("float32")  # [vocab, 768]
vocab_size, dim = W.shape

index = faiss.IndexFlatIP(dim)
index.add(W)

# ==== Prepare test input ====
prompt = "The moon is bright and the stars are visible in the night sky. This makes it a perfect"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids[:, -num_tokens_to_test:]  # Trim if needed
num_tokens = input_ids.shape[1]

# ==== Forward pass ====
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-1]  # [1, seq_len, 768]
    W_tensor = model.lm_head.weight.data  # [vocab, 768]

# ==== Benchmark ====
matches = 0
faiss_time = 0.0
rerank_time = 0.0
matmul_time = 0.0

for i in range(num_tokens):
    h = hidden_states[0, i, :].cpu().numpy().astype("float32").reshape(1, -1)
    h_torch = torch.tensor(h).float()

    # ==== Full Matmul for Reference ====
    start = time.time()
    logits_full = h_torch @ W_tensor.T  # [1, vocab]
    gold_id = torch.argmax(logits_full, dim=-1).item()
    matmul_time += time.time() - start

    # ==== FAISS top-k search ====
    start = time.time()
    D, I = index.search(h, k)  # I: [1, k]
    faiss_time += time.time() - start

    # ==== Local rerank ====
    topk_ids = I[0]
    W_topk = W_tensor[topk_ids, :]  # [k, 768]
    h_torch_flat = h_torch.squeeze(0)  # [768]

    start = time.time()
    local_logits = torch.matmul(W_topk, h_torch_flat)  # [k]
    reranked_id = topk_ids[torch.argmax(local_logits).item()]
    rerank_time += time.time() - start

    if reranked_id == gold_id:
        matches += 1

# ==== Report ====
print("\n======= 🏁 Top-K ANN Rerank Benchmark =======")
print(f"Tested tokens       : {num_tokens}")
print(f"Exact match rate    : {matches}/{num_tokens} ({matches / num_tokens:.2%})")
print(f"Avg FAISS time      : {1000 * faiss_time / num_tokens:.3f} ms")
print(f"Avg Rerank time     : {1000 * rerank_time / num_tokens:.3f} ms")
print(f"Avg Full Matmul     : {1000 * matmul_time / num_tokens:.3f} ms")
print(f"Speedup vs Matmul   : {matmul_time / (faiss_time + rerank_time):.2f}×")
