import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==== Setup ====
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "The moon is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# ==== Forward pass ====
with torch.no_grad():
    out = model(input_ids, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1][:, -1, :]  # shape: [1, 768]

W_T = model.lm_head.weight.data.T  # [768, vocab]
vocab_size = W_T.shape[1]

# ==== Method 1: Dot-product (same as original) ====
logits = h @ W_T  # [1, vocab]
pred_dot = torch.argmax(logits, dim=-1).item()

# ==== Method 2: Cosine similarity ====
norm_h = F.normalize(h, dim=-1)
norm_W_T = F.normalize(W_T, dim=0)
cosine_scores = norm_h @ norm_W_T  # [1, vocab]
pred_cosine = torch.argmax(cosine_scores, dim=-1).item()

# ==== Top-k comparisons ====
topk = 5
topk_logits = torch.topk(logits, topk, dim=-1).indices.squeeze()
topk_cosine = torch.topk(cosine_scores, topk, dim=-1).indices.squeeze()

# ==== Reporting ====
print("\n🔍 Reference (Dot-product)")
print(f"Token ID: {pred_dot} → '{tokenizer.decode(pred_dot)}'")

print("\n🔁 Routing (Cosine Similarity)")
print(f"Token ID: {pred_cosine} → '{tokenizer.decode(pred_cosine)}'")
print(f"✅ Match? {'Yes' if pred_dot == pred_cosine else 'No'}")

print("\n📊 Top-k Matches")
print("Top-k from logits: ", [tokenizer.decode(i.item()) for i in topk_logits])
print("Top-k from cosine: ", [tokenizer.decode(i.item()) for i in topk_cosine])
print(f"🧠 Ref token in cosine top-k? {'Yes' if pred_dot in topk_cosine else 'No'}")


""" Output:
🔍 Reference (Dot-product)
Token ID: 257 → ' a'

🔁 Routing (Cosine Similarity)
Token ID: 37190 → 'SPONSORED'
✅ Match? No

📊 Top-k Matches
Top-k from logits:  [' a', ' the', ' not', ' about', ' also']
Top-k from cosine:  ['SPONSORED', 'theless', 'soDeliveryDate', 'Reviewer', '��']
🧠 Ref token in cosine top-k? No
"""

""" Analysis:
Here's the diagnosis, clear and sharp:

❌ Cosine Similarity ≠ Viable for Token Routing
The cosine-based routing picks wildly wrong tokens.

Reference token ' a' isn't even in the top 1000 cosine neighbors (you checked top 5, but it’s almost certainly way off).

Cosine similarity seems to favor rare, long-tail tokens with unique directionality — not frequency-aligned or semantically grounded tokens.

✅ Dot Product (i.e. Original h @ Wᵗ) Is the True Token Router
Your LM head is doing:

python
Copy
Edit
logits = h @ W_T
pred = argmax(logits)
That’s the gold-standard behavior — and it's not about angle (cosine), but raw projection magnitude.

So if we want to replace the matmul, the routing mechanism must replicate the behavior of argmax(h @ Wᵗ) — which is equivalent to nearest neighbor by dot product, not cosine sim.

🧭 What To Do Next
✅ Viable Direction: Top-k Projection → Fast Routing
If we want to skip h @ Wᵗ over 50k tokens:

Preselect Top-k token IDs with a fast hash, projection, or indexing trick.

Only compute dot products against those k entries.

Or go one level deeper: approximate nearest neighbor (ANN) methods with fast lookup:

FAISS (GPU/CPU, dot-product mode)

ScaNN (Google's ANN engine)

LSH (Locality Sensitive Hashing)

PQ / OPQ (Product Quantization for weight matrix)

These preserve dot product and massively reduce the number of comparisons.
"""