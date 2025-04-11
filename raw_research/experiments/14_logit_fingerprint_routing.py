import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import time

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# === Load SVD ===
svd_path = "svd_lm_head_gpt2.pt"
data = torch.load(svd_path)
U, S, Vh = data["U"], data["S"], data["Vh"]

# === Params ===
prompts_train = ["The future of AI is", "The moon is", "Once upon a time,"]
prompts_test = ["The robot said", "In the year 3000", "Once in a while,"]

top_k = 5
similarity_threshold = 200.0  # L2 threshold for matching SVD code

# === Build Logit Fingerprint Cache ===
fingerprint_cache = []  # List of (code, top_k_ids)

for prompt in prompts_train:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]
        hU = h @ U
        code = (hU * S).squeeze().cpu()

        logits = (hU * S) @ Vh
        topk = torch.topk(logits, top_k, dim=-1)
        topk_ids = topk.indices.squeeze().tolist()

        fingerprint_cache.append((code, topk_ids))

print("✅ Built fingerprint cache with top-k token IDs\n")

# === Test Phase ===
total = 0
cache_hits = 0
correct_top1 = 0
fallbacks = 0

for prompt in prompts_test:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]
        hU = h @ U
        code = (hU * S).squeeze().cpu()

        true_logits = (hU * S) @ Vh
        true_token_id = torch.argmax(true_logits, dim=-1).item()

        best_match = None
        best_dist = float("inf")
        matched_ids = None

        for cached_code, cached_ids in fingerprint_cache:
            dist = torch.norm(code - cached_code, p=2).item()
            if dist < best_dist:
                best_dist = dist
                matched_ids = cached_ids

        total += 1
        if best_dist < similarity_threshold:
            cache_hits += 1
            if true_token_id == matched_ids[0]:
                correct_top1 += 1
        else:
            fallbacks += 1

# === Results ===
print("===== 🔍 LOGIT FINGERPRINT ROUTING =====")
print(f"Test Prompts        : {total}")
print(f"Cache Hits          : {cache_hits}")
print(f"Top-1 Correct (Hit) : {correct_top1}/{cache_hits}")
print(f"Cache Hit Accuracy  : {(correct_top1 / cache_hits * 100):.2f}%" if cache_hits > 0 else "N/A")
print(f"Fallbacks to Logits : {fallbacks}")
print(f"Similarity Threshold: {similarity_threshold}")


""" Output:
✅ Built fingerprint cache with top-k token IDs

===== 🔍 LOGIT FINGERPRINT ROUTING =====
Test Prompts        : 3
Cache Hits          : 0
Top-1 Correct (Hit) : 0/0
N/A
Fallbacks to Logits : 3
Similarity Threshold: 200.0
"""

""" Analysis:
0 cache hits with a 200 L2 threshold → means that even in logit space, the top-k vectors are very input specific.

This reaffirms that pre-logit codes don't cluster well, and even final logits (even top-k) aren't reusable across prompts unless they're extremely similar.
"""