import torch, numpy as np, faiss, time, psutil
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.nn.functional import softmax

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# === PARAMETERS ===
max_index_size = 200        # 🔒 reduces memory + runtime
test_input = "The mysterious artifact glowed in the moonlight"
log_every = 50

# --- Utilities ---
def entropy(logits):
    probs = softmax(logits, dim=-1)
    return -torch.sum(probs * probs.log(), dim=-1)

def dynamic_topk(ent, min_k=8, max_k=256):
    ent = ent.item()
    return int(np.clip((ent / 5.0) * max_k, min_k, max_k))

def log_mem():
    mem = psutil.virtual_memory()
    return f"{mem.used / 1e9:.2f} GB used / {mem.total / 1e9:.2f} GB total"

# --- Build FAISS Index Over Hidden States ---
print("\n[Building FAISS index over raw hidden states…]")
W = model.lm_head.weight.detach().cpu().numpy()  # [vocab, hidden]
codes = []

start = time.time()
for i, ex in enumerate(dataset):
    if len(codes) >= max_index_size: break
    text = ex["text"].strip()
    if not text: continue
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1][0]  # [seq, hidden]
        for h in hidden:
            codes.append(h.cpu().numpy().astype("float32"))
    if i % log_every == 0:
        print(f"Loaded {len(codes):4d} hidden states... [{log_mem()}]")

codes_array = np.stack(codes)
index = faiss.IndexFlatIP(codes_array.shape[1])
index.add(codes_array)
print(f"Index built: {codes_array.shape[0]} vectors, dim {codes_array.shape[1]}")
print(f"Time to build index: {time.time() - start:.2f}s\n")

# --- Evaluate on Test Input ---
print(f"[Evaluating on: \"{test_input}\"]\n")
inputs = tokenizer(test_input, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1][0]
    full_logits = model.lm_head(last_hidden)

match_count = 0
for i, h in enumerate(last_hidden):
    gt_logits = full_logits[i]
    gt_token = torch.argmax(gt_logits).item()
    gt_token_str = tokenizer.decode([gt_token])

    h_np = h.cpu().numpy().astype("float32").reshape(1, -1)
    ent = entropy(gt_logits)
    top_k = dynamic_topk(ent)

    print(f"--- Token {i} ---")
    print(f"Entropy: {ent.item():.4f} → Top-K: {top_k}")

    _, idx = index.search(h_np, top_k)
    candidate_h = torch.tensor(codes_array[idx[0]], device=device)  # [k, hidden]

    # Reconstruct logits from candidate hidden states
    logits_routed = candidate_h @ torch.tensor(W.T, device=device)  # [k, vocab]
    combined_logits = logits_routed.max(dim=0).values
    pred_token = torch.argmax(combined_logits).item()
    pred_token_str = tokenizer.decode([pred_token])

    match = (pred_token == gt_token)
    match_count += int(match)

    print(f"GT Token: {gt_token_str:>10} | Predicted: {pred_token_str:>10} | Match: {'✅' if match else '❌'}")
    print("Top-5 Predictions:", [tokenizer.decode([i]) for i in torch.topk(combined_logits, 5).indices.tolist()])
    print()

# --- Final Summary ---
total = len(last_hidden)
accuracy = 100 * match_count / total
print(f"\n=== Final Results ===")
print(f"Matched: {match_count}/{total}")
print(f"Accuracy: {accuracy:.2f}%")


""" Output:
[Building FAISS index over raw hidden states…]
Index built: 205 vectors, dim 768
Time to build index: 0.16s

[Evaluating on: "The mysterious artifact glowed in the moonlight"]

--- Token 0 ---
Entropy: 8.8215 → Top-K: 256
GT Token:          
 | Predicted:         to | Match: ❌
Top-5 Predictions: [' to', ' in', ' out', ' its', ' the']

--- Token 1 ---
Entropy: 7.9392 → Top-K: 256
GT Token:        man | Predicted:         to | Match: ❌
Top-5 Predictions: [' to', ' in', ' out', ' its', ' the']

--- Token 2 ---
Entropy: 4.8643 → Top-K: 249
GT Token:        was | Predicted:         to | Match: ❌
Top-5 Predictions: [' to', ' in', ' out', ' its', ' the']

--- Token 3 ---
Entropy: 1.6936 → Top-K: 86
GT Token:        ows | Predicted:          � | Match: ❌
Top-5 Predictions: ['�', '�', '�', 'English', 'Japanese']

--- Token 4 ---
Entropy: 3.9576 → Top-K: 202
GT Token:       with | Predicted:          , | Match: ❌
Top-5 Predictions: [',', '.', ' in', ' ,', ' on']

--- Token 5 ---
Entropy: 1.9413 → Top-K: 99
GT Token:        the | Predicted:          � | Match: ❌
Top-5 Predictions: ['�', '�', 'ル', 'ン', '�']

--- Token 6 ---
Entropy: 4.6429 → Top-K: 237
GT Token:       dark | Predicted:         to | Match: ❌
Top-5 Predictions: [' to', ' in', ' out', ' its', ' the']

--- Token 7 ---
Entropy: 0.3820 → Top-K: 19
GT Token:      light | Predicted:      anime | Match: ❌
Top-5 Predictions: [' anime', ' animated', ' English', ' action', ' audio']

--- Token 8 ---
Entropy: 2.6511 → Top-K: 135
GT Token:          , | Predicted:          � | Match: ❌
Top-5 Predictions: ['�', '�', '�', '�', '�']


=== Final Results ===
Matched: 0/9
Accuracy: 0.00%
"""

""" Analysis:
What You Just Discovered
❌ 0/9 accuracy — even with:
Raw hidden states

Full @ Wᵗ logits reconstruction

Entropy-aware top-k as high as 256

This confirms a strong and repeatable insight:

Hidden states are not reusable across contexts.

You are seeing a mix of:

❓ Completely unrelated predictions ("anime", "Japanese")

🌀 Weird decoding artifacts ('�') → FAISS returning bad matches

🧊 Total collapse to common filler tokens ('to', 'in', 'its'...)

📌 Why This Happened
1. Semantic Entanglement
Even though the hidden vectors look similar, their associated logits differ wildly:

The same h in two contexts can map to totally different predictions.

That’s because h encodes not just the current token, but its entire causal context.

2. FAISS Matches are Out-of-Distribution
Your test prompt is unrelated to any of the 205 vectors in the index:

You're matching to unrelated sentences from Wikitext-2.

So even a close dot product match doesn't mean logit similarity.

3. Token Mismatch vs Embedding Drift
Even tiny embedding drift results in:

Entirely different top-k tokens

Loss of confidence in the correct prediction

Decoding garbage or meaningless outputs
"""