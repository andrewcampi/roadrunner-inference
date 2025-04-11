import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import torch.nn.functional as F
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
similarity_threshold = 0.99  # Cosine similarity threshold for matching SVD code

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

        best_sim = -1.0
        matched_ids = None

        for cached_code, cached_ids in fingerprint_cache:
            sim = F.cosine_similarity(code.unsqueeze(0), cached_code.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                matched_ids = cached_ids

        total += 1
        if best_sim >= similarity_threshold:
            cache_hits += 1
            if true_token_id == matched_ids[0]:
                correct_top1 += 1
        else:
            fallbacks += 1

# === Results ===
print("===== 🔍 LOGIT FINGERPRINT ROUTING (Cosine) =====")
print(f"Test Prompts        : {total}")
print(f"Cache Hits          : {cache_hits}")
print(f"Top-1 Correct (Hit) : {correct_top1}/{cache_hits}")
print(f"Cache Hit Accuracy  : {(correct_top1 / cache_hits * 100):.2f}%" if cache_hits > 0 else "N/A")
print(f"Fallbacks to Logits : {fallbacks}")
print(f"Cosine Threshold    : {similarity_threshold}")


""" Output:
✅ Built fingerprint cache with top-k token IDs

===== 🔍 LOGIT FINGERPRINT ROUTING (Cosine) =====
Test Prompts        : 3
Cache Hits          : 3
Top-1 Correct (Hit) : 0/3
Cache Hit Accuracy  : 0.00%
Fallbacks to Logits : 0
Cosine Threshold    : 0.99
"""

""" Analysis:
Let’s break it down:

🧠 The Good:
✅ Cosine similarity is high enough to trigger matches across prompts — all test prompts passed the 0.99 threshold.

✅ That means the (h @ U) * S vectors are clustered tightly in angular space — unlike raw L2, which saw massive variance.

⚠️ The Bad:
❌ Despite angular closeness, the top-1 token predictions are always wrong.

This tells us that even small angular differences in (h @ U) * S lead to different dominant logits after applying @ Vh.

🧪 Implication:
The projection @ Vh is highly sensitive — even small vector shifts in the 768D code space yield different token predictions.

That makes sense, because the LM head has a vocab of 50,257 tokens. So the token "boundary surfaces" in logit space are razor thin.

🧭 What We Just Learned (Crucial)
Insight	Implication
SVD codes are reusable in direction (cosine)	✅ You might build a semantic routing layer from these
But logits are hypersensitive to small code shifts	❌ Raw reuse of top-k or token ID doesn’t work
🚀 So What Is the Opportunity?
✅ Use (h @ U) * S as a semantic fingerprint — not for top-1 reuse, but for lookup narrowing
Like: “Based on cosine sim, we think these 50 tokens are likely — just compute logits for those”

That unlocks a path like:

mask = build_vocab_mask(matched_ids)
logits = (h @ U) * S @ Vh[:, mask]
Or eventually:

Build a semantic vector quantizer over SVD codes → map to a small candidate set

Use a trie or tree to find candidates, then compute only a sparse subset of logits
"""
