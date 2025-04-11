import torch
import time
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.linalg import svd

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

block = model.transformer.h[0]  # MLP block 0

# === Extract weights ===
fc_weight = block.mlp.c_fc.weight.data.clone().T  # [3072, 768]
fc_bias = block.mlp.c_fc.bias.data.clone()
proj_weight = block.mlp.c_proj.weight.data.clone().T  # [3072, 768]
proj_bias = block.mlp.c_proj.bias.data.clone()

# === SVD ===
U, S, Vh = svd(fc_weight, full_matrices=False)

# === Input ===
prompt = "The sky was clear and stars were bright. "
inputs = tokenizer(prompt, return_tensors="pt").to(device)
hidden = model.transformer.wte(inputs.input_ids)[0][-1].unsqueeze(0)  # [1, 768]

# === Full MLP output for comparison ===
def full_mlp(x):
    return block.mlp.c_proj(F.gelu(block.mlp.c_fc(x)))

baseline_out = full_mlp(hidden)
baseline_tok = torch.argmax(baseline_out).item()

# === Routed MLP with alpha blending ===
def routed_mlp(x, alpha):
    code = x @ Vh
    code_scaled = code * S
    routed_hidden = F.gelu(code_scaled @ U.T + fc_bias)
    routed_out = routed_hidden @ proj_weight.T + proj_bias
    return alpha * routed_out + (1 - alpha) * baseline_out

# === Alpha sweep ===
results = []
alphas = [round(a, 2) for a in torch.linspace(0, 1, 11).tolist()]

print("\n🔬 Alpha Sweep Results:")
print(f"{'α':>5} | {'Token Match':>12} | {'L2 Drift':>9} | {'Cos Sim':>9} | {'Time (ms)':>9}")
print("-" * 50)

best_alpha = None
best_cos = -1.0

for alpha in alphas:
    with torch.no_grad():
        t0 = time.time()
        routed_out = routed_mlp(hidden, alpha)
        t_routed = (time.time() - t0) * 1000

        l2 = torch.norm(routed_out - baseline_out).item()
        cos = F.cosine_similarity(routed_out, baseline_out).item()
        match = torch.argmax(routed_out).item() == baseline_tok

        results.append((alpha, match, l2, cos, t_routed))

        print(f"{alpha:>5.2f} | {'✅' if match else '❌':>12} | {l2:>9.4f} | {cos:>9.6f} | {t_routed:>9.3f}")

        if match and cos > best_cos:
            best_cos = cos
            best_alpha = alpha

# === Conclusion ===
print("\n✅ Optimal Alpha Selected:")
if best_alpha is not None:
    print(f"  α = {best_alpha:.2f} with cosine similarity = {best_cos:.6f}")
else:
    print("  No alpha maintained token match. Consider using lower α or fallback routing.")



""" Output:
🔬 Alpha Sweep Results:
    α |  Token Match |  L2 Drift |   Cos Sim | Time (ms)
--------------------------------------------------
 0.00 |            ✅ |    0.0000 |  1.000000 |     0.853
 0.10 |            ✅ |    7.2503 |  0.998729 |     0.354
 0.20 |            ✅ |   14.5006 |  0.993603 |     0.346
 0.30 |            ✅ |   21.7509 |  0.981511 |     0.341
 0.40 |            ✅ |   29.0012 |  0.956842 |     0.342
 0.50 |            ✅ |   36.2515 |  0.909751 |     0.338
 0.60 |            ✅ |   43.5018 |  0.824528 |     0.337
 0.70 |            ✅ |   50.7521 |  0.682126 |     0.335
 0.80 |            ✅ |   58.0024 |  0.474846 |     0.334
 0.90 |            ❌ |   65.2527 |  0.227537 |     0.333
 1.00 |            ❌ |   72.5030 | -0.011045 |     0.331
"""


""" Analysis:
Here’s what this experiment tells us loud and clear:

🔍 What We Learned
✅ Token Match stays perfect up to α = 0.80
That’s incredibly promising — even with 80% of the output coming from the routed path, you’re still getting the exact same token prediction.

📉 Cosine similarity declines smoothly
You get >0.9 similarity at α = 0.5

Even at α = 0.7, it holds a 68% cosine with baseline

It only collapses past α = 0.8 (and dies completely by α = 1.0)

⏱️ Inference time is almost constant
That means routing is dominated by overhead from PyTorch ops and not α-specific math. Good for batching or fusion later.

🧠 What’s the Right α?
The script selected α = 0.0 (baseline) because it gives perfect cosine similarity.

But that’s not what you actually want — you're trying to find the **highest α that:

maintains token match ✅

maximizes cosine similarity**

From your data, that would be:

✅ α = 0.5 — Token match + 0.91 cosine similarity + 36.25 drift

This is your optimal routing strength for MLP block 0.

🎯 Recommendation
Use α = 0.5 for now as your starting point in:
Layerwise routing

Full model routing

Future routing block modules

Later, you can:

Run this sweep for each layer to get an adaptive α_map

Use smaller α in deeper layers (since drift accumulates)

Fit a curve or heuristic (e.g. α[layer] = 0.6 - 0.02 * layer_idx)
"""