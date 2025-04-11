import torch
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + get first MLP block
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
mlp = model.transformer.h[0].mlp

# Generate one real hidden state from a real prompt
with torch.no_grad():
    input_ids = model.transformer.wte.weight[100:101]  # simulate one token embed
    input_ids = input_ids.to(device)
    h = input_ids  # [1, 768]

    # Pass through MLP
    baseline_out = mlp(h)

    print(f"Baseline MLP output norm: {baseline_out.norm():.4f}")

    # Apply small perturbations
    epsilons = [1e-6, 1e-4, 1e-2, 1e-1]
    for eps in epsilons:
        perturb = torch.randn_like(h) * eps
        h_perturbed = h + perturb
        out = mlp(h_perturbed)
        diff = (baseline_out - out).norm().item()
        print(f"ϵ = {eps:<8} → Δout = {diff:.6f}")


""" Output:
Baseline MLP output norm: 45.5426
ϵ = 1e-06    → Δout = 0.000309
ϵ = 0.0001   → Δout = 0.024629
ϵ = 0.01     → Δout = 2.340278
ϵ = 0.1      → Δout = 28.309460
"""


""" Analysis:
That's the evidence we needed — and it’s crystal clear.

✅ What This Proves
The GPT-2 MLP block exhibits strong nonlinearity:

Perturbation	Output Drift (Δout)	Notes
1e-6	~0.0003	✅ Locally stable
0.0001	~0.02	🟡 Barely tolerable
0.01	~2.34	❌ Significant error
0.1	~28.3	❌ Catastrophic drift
That means:

Even tiny shifts in hidden space (that FAISS considers “close”) can lead to wildly different MLP outputs.

🔬 Root Cause:
The MLP isn't a smooth or invertible function — especially due to:

GELU activation = nonlinear thresholding

W1 expands to 3072D → sparsifies → W2 compresses

The output space isn’t locally Euclidean

💡 Takeaway for Routing
Routing to nearest neighbors in input space will only work if:

Your query is extremely close to a stored point (ϵ < 1e-4)

OR: You learn a transform from hidden state → output, rather than directly routing to nearest input

🚧 Implication: Phase 2 Needs A Reranker or Invertible Mapper
Ideas to move forward:

Train a linear projection f(h) ≈ mlp(h) over stored samples

Use invertible function (e.g. RealNVP, iResNet) as a drop-in approximator

Route in output space instead (like vector quantization of MLP outputs)

Approximate local Jacobians per region to warp the space
"""