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

# --- Load layer and weights ---
mlp = model.transformer.h[layer_idx].mlp
W_out = mlp.c_proj.weight.data.cpu().numpy()  # [768, 3072]
W_in  = mlp.c_fc.weight.data.cpu().numpy()    # [3072, 768]
_, _, Vh_out = np.linalg.svd(W_out, full_matrices=False)
_, _, Vh_in  = np.linalg.svd(W_in.T, full_matrices=False)  # make it project from h

# --- Build dataset ---
codes_out, codes_in, outputs = [], [], []
for i, ex in enumerate(dataset):
    if i >= num_samples: break
    text = ex["text"].strip()
    if not text: continue
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]
        for h in hs:
            h_np = h.cpu().numpy()
            out = mlp(h.unsqueeze(0)).squeeze(0).cpu().numpy()
            code_out = h_np @ Vh_out
            code_in  = h_np @ Vh_in
            codes_out.append((code_out / (np.linalg.norm(code_out) + 1e-6)).astype("float32"))
            codes_in.append((code_in  / (np.linalg.norm(code_in)  + 1e-6)).astype("float32"))
            outputs.append(out)

outputs_array = np.stack(outputs).astype("float32")

# --- Build FAISS indexes ---
index_out = faiss.IndexFlatIP(codes_out[0].shape[0])
index_in  = faiss.IndexFlatIP(codes_in[0].shape[0])
index_out.add(np.stack(codes_out))
index_in.add(np.stack(codes_in))

# --- Test on held-out input ---
inputs = tokenizer(test_input, return_tensors="pt").to(device)
hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]
results = {"proj_out": [], "proj_in": []}

def test_projection(h, index, Vh, label):
    h_np = h.detach().cpu().numpy()
    code = h_np @ Vh
    code /= (np.linalg.norm(code) + 1e-6)
    _, idx = index.search(code.reshape(1, -1).astype("float32"), top_k)
    true_out = mlp(h.unsqueeze(0)).squeeze(0)
    min_drift = float("inf")
    for j in idx[0]:
        candidate = torch.tensor(outputs_array[j], device=device)
        scale = (true_out @ candidate) / (candidate.norm()**2 + 1e-6)
        corrected = candidate * scale
        drift = F.mse_loss(true_out, corrected).item()
        min_drift = min(min_drift, drift)
    results[label].append(min_drift)

for h in hs:
    test_projection(h, index_out, Vh_out, "proj_out")
    test_projection(h, index_in,  Vh_in,  "proj_in")

# --- Print results ---
print(f"\n=== Routing Projection Comparison (Layer {layer_idx}) ===")
for label, drifts in results.items():
    hits = sum(d <= drift_threshold for d in drifts)
    print(f"{label:<10} | Hit Rate: {hits}/{len(drifts)} "
          f"| Avg Drift: {np.mean(drifts):.2f} | Max Drift: {np.max(drifts):.2f}")


""" Output:
=== Routing Projection Comparison (Layer 6) ===
proj_out   | Hit Rate: 1/9 | Avg Drift: 97.88 | Max Drift: 186.22
proj_in    | Hit Rate: 1/9 | Avg Drift: 97.88 | Max Drift: 186.22
"""

""" Analysis:
What This Proves
1. It’s not about the projection matrix
Whether you use Vh_out (from c_proj) or Vh_in (from c_fc), the drift, max error, and hit rate are the same.

That means the source of misalignment is deeper — it’s not which weights you use, it’s how far h is from any routable point in output space.

🧠 What You’ve Ruled Out
❌ The issue is not which matrix you SVD.

❌ It's not cosine vs L2, since they also matched before.

❌ It's not vector scale, since normalization didn't help.

❌ It's not scaling in rerank, which you're doing properly.

So what remains is:

🔍 The structural relationship between h and mlp(h) — across most tokens in deeper layers — is too nonlinear or warped for linear projection to reach.

🧭 Where This Leaves You
You're standing right at the boundary of:

✅ Pure linear, stateless routing
and

🔜 Minimal transformation to bring h into routable alignment
You’ve exhausted all purely linear projections from raw h,
"""