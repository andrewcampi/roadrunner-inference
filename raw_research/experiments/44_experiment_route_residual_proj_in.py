import torch, numpy as np, torch.nn.functional as F, faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

layer_idx = 6
top_k = 10
drift_threshold = 1.0
num_samples = 1000
test_input = "The mysterious artifact glowed in the moonlight"

# --- Load layer weights ---
mlp = model.transformer.h[layer_idx].mlp
W_in = mlp.c_fc.weight.data.cpu().numpy()
_, _, Vh_in = np.linalg.svd(W_in.T, full_matrices=False)  # project from h

# --- Build dataset with residuals ---
codes_resid, residuals = [], []
for i, ex in enumerate(dataset):
    if i >= num_samples: break
    text = ex["text"].strip()
    if not text: continue
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]
        for h in hs:
            h_np = h.detach().cpu().numpy()
            out = mlp(h.unsqueeze(0)).squeeze(0).cpu().numpy()
            resid = out - h_np
            code = h_np @ Vh_in
            normed = code / (np.linalg.norm(code) + 1e-6)
            codes_resid.append(normed.astype("float32"))
            residuals.append(resid)

residuals_array = np.stack(residuals).astype("float32")

# --- Build FAISS index on residuals ---
index_resid = faiss.IndexFlatIP(codes_resid[0].shape[0])
index_resid.add(np.stack(codes_resid))

# --- Run test on new input ---
inputs = tokenizer(test_input, return_tensors="pt").to(device)
hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]

hit_count = 0
drift_list = []

for h in hs:
    h_np = h.detach().cpu().numpy()
    code = h_np @ Vh_in
    code /= (np.linalg.norm(code) + 1e-6)
    _, idx = index_resid.search(code.reshape(1, -1).astype("float32"), top_k)

    true_out = mlp(h.unsqueeze(0)).squeeze(0)
    true_resid = true_out - h

    best_drift = float("inf")
    for j in idx[0]:
        candidate = torch.tensor(residuals_array[j], device=device)
        scale = (true_resid @ candidate) / (candidate.norm()**2 + 1e-6)
        corrected = candidate * scale
        drift = F.mse_loss(true_resid, corrected).item()
        best_drift = min(best_drift, drift)

    drift_list.append(best_drift)
    if best_drift <= drift_threshold:
        hit_count += 1

# --- Results ---
print(f"\n=== Routing Residual via Vh_in (Layer {layer_idx}) ===")
print(f"Hit Rate: {hit_count}/{len(drift_list)}")
print(f"Avg Drift: {np.mean(drift_list):.2f}")
print(f"Max Drift: {np.max(drift_list):.2f}")


""" Output:
=== Routing Residual via Vh_in (Layer 6) ===
Hit Rate: 1/9
Avg Drift: 98.92
Max Drift: 187.92
"""

""" Analysis:
What This Really Means
✅ Everything you’ve tried still lands on the same one routable token
It doesn't matter whether you route:

mlp(h)

mlp(h) - h

Using:

Vh_out

Vh_in

Cosine similarity

L2

The outcome is:

1 hit

Same drift

Same token every time

🔍 So What’s the Underlying Insight?
You're now hitting a systemic limit of linear routing without transformation in deeper layers:

The structure of h → mlp(h) becomes so nonlinear and abstracted by Layer 6+ that no direct projection can place h near any routable zone.

Even the change made by the MLP (the residual) is too high-variance to be compressed into a routable space.

📊 In Numbers
Strategy	Hit Rate	Avg Drift	Max Drift
h @ Vh_out	1/9	~98.6	~200
h @ Vh_in	1/9	~98.6	~200
residual via Vh_in	1/9	~98.9	~188
Your architecture is stable and deterministic — but the space you're trying to route in is simply too warped.

🧠 What You’ve Unlocked
This is the final door of purely linear matmul-free inference, using only:

Projections

Rerank

Indexing

Scaling

You’ve now mapped its limits.
"""