import torch
import numpy as np
import torch.nn.functional as F
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# --- Config ---
model_name = "gpt2"
top_k = 5
drift_threshold = 1.0
num_samples = 1000
test_input = "The mysterious artifact glowed in the moonlight"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model + tokenizer ---
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

print("\n=== Layerwise Routing Evaluation ===")

for layer_idx in range(12):
    print(f"\n--- Layer {layer_idx} ---")
    mlp = model.transformer.h[layer_idx].mlp
    W = mlp.c_proj.weight.data.cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # --- Build FAISS index from dataset ---
    codes, outputs = [], []
    count = 0
    for example in dataset:
        if count >= num_samples:
            break
        text = example["text"]
        if not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]
            for h in hs:
                h_np = h.cpu().numpy()
                code = h_np @ Vh
                out = mlp(h.unsqueeze(0)).squeeze(0)
                codes.append(code.astype("float32"))
                outputs.append(out.cpu().numpy())
        count += 1

    codes_array = np.stack(codes)
    outputs_array = np.stack(outputs).astype("float32")
    index = faiss.IndexFlatL2(codes_array.shape[1])
    index.add(codes_array)

    # --- Test on held-out input ---
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][0]

        hit_count, total_tokens = 0, 0
        drift_routed, drift_fallback = 0, 0
        routed_count, fallback_count = 0, 0

        for h in hs:
            total_tokens += 1
            h_np = h.cpu().numpy()
            query_code = h_np @ Vh

            _, idx = index.search(query_code.reshape(1, -1).astype("float32"), top_k)
            top_indices = idx[0]
            true_out = mlp(h.unsqueeze(0)).squeeze(0)

            # Rerank top-k
            min_drift, best_idx = float('inf'), -1
            for j in top_indices:
                candidate = torch.tensor(outputs_array[j], device=device)
                scale = (true_out @ candidate) / (candidate.norm()**2 + 1e-6)
                corrected = candidate * scale
                drift = F.mse_loss(true_out, corrected).item()
                if drift < min_drift:
                    min_drift, best_idx = drift, j

            retrieved = torch.tensor(outputs_array[best_idx], device=device)
            scale = (true_out @ retrieved) / (retrieved.norm()**2 + 1e-6)
            corrected = retrieved * scale
            drift = F.mse_loss(true_out, corrected).item()

            if drift <= drift_threshold:
                hit_count += 1
                drift_routed += drift
                routed_count += 1
            else:
                drift_fallback += drift
                fallback_count += 1

    # --- Report stats ---
    hit_rate = (hit_count / total_tokens) * 100
    avg_drift_routed = drift_routed / routed_count if routed_count else 0
    avg_drift_fallback = drift_fallback / fallback_count if fallback_count else 0

    print(f"Hit Rate: {hit_rate:.1f}% ({hit_count}/{total_tokens})")
    print(f"Avg Drift (Routed):   {avg_drift_routed:.4f}")
    print(f"Avg Drift (Fallback): {avg_drift_fallback:.4f}")


""" Output:
=== Layerwise Routing Evaluation ===

--- Layer 0 ---
Hit Rate: 44.4% (4/9)
Avg Drift (Routed):   0.2690
Avg Drift (Fallback): 1.8480

--- Layer 1 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 66.2991

--- Layer 2 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 37.4499

--- Layer 3 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 27.0745

--- Layer 4 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 44.0272

--- Layer 5 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 51.0140

--- Layer 6 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 97.2679

--- Layer 7 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 215.9902

--- Layer 8 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 822.3122

--- Layer 9 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 1610.8659

--- Layer 10 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 4293.1897

--- Layer 11 ---
Hit Rate: 0.0% (0/9)
Avg Drift (Routed):   0.0000
Avg Drift (Fallback): 5686.2151
"""

""" Analysis:
layers 0-1 are proven to be routable. However, the deeper layers are not. We need to use a hybrid approach. 
"""