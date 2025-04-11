import torch
import time
import torch.nn.functional as F
from torch.nn import Linear
from torch.linalg import svd

torch.set_printoptions(precision=4, sci_mode=False)

# --- Configurable Params ---
hidden_dim = 768
intermediate_dim = 3072
batch_size = 1
alpha = 0.7  # blending factor (0 = full matmul, 1 = full routed)
top_k = 128  # top-k reranking (optional future extension)

# --- Fake Pretrained MLP Weights (as placeholders) ---
W_in = torch.randn(intermediate_dim, hidden_dim)
b_in = torch.randn(intermediate_dim)
W_out = torch.randn(hidden_dim, intermediate_dim)
b_out = torch.randn(hidden_dim)

# --- Random Hidden State (input) ---
x = torch.randn(batch_size, hidden_dim)

# --- Baseline MLP Forward ---
def mlp_full(x):
    return F.gelu(F.linear(x, W_in, b_in)) @ W_out.T + b_out

# --- Residual Routing using SVD on Input Projection ---
def mlp_routed(x, alpha=1.0):
    # SVD: W_in = U @ S @ Vh
    U, S, Vh = svd(W_in)
    Vh_in = Vh  # right singular vectors (input directions)
    
    # Project x into SVD code space
    code = x @ Vh_in.T  # shape: [B, hidden]
    routed_hidden = F.gelu(code @ torch.diag(S) @ U.T)  # Intermediate activation
    routed_output = routed_hidden @ W_out.T + b_out

    # Full path for blending
    full_output = mlp_full(x)

    # Blend outputs
    return alpha * routed_output + (1 - alpha) * full_output, full_output

# --- Run + Time Both ---
start = time.time()
routed_out, full_out = mlp_routed(x, alpha)
routed_time = time.time() - start

start = time.time()
baseline_out = mlp_full(x)
baseline_time = time.time() - start

# --- Metrics ---
l2_drift = torch.norm(routed_out - baseline_out).item()
cos_sim = F.cosine_similarity(routed_out, baseline_out).mean().item()
token_match = torch.argmax(routed_out, dim=-1).eq(torch.argmax(baseline_out, dim=-1)).sum().item()

# --- Print Results ---
print("\n🔬 Drift Metrics:")
print(f"  L2 Drift:        {l2_drift:.4f}")
print(f"  Cosine Sim:      {cos_sim:.6f}")
print(f"  Token Match:     {token_match}/{batch_size}")

print("\n⏱️ Inference Time:")
print(f"  Routed Output:   {routed_time*1000:.3f} ms")
print(f"  Full Matmul:     {baseline_time*1000:.3f} ms")

print(f"\n⚙️  Blend Factor α = {alpha}")
