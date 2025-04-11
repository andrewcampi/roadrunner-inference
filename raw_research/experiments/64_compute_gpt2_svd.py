import torch
import os
from transformers import GPT2LMHeadModel

# === CONFIG ===
MODEL_NAME = "gpt2"
OUTPUT_DIR = "svd"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.float32  # You can change to float16 or bfloat16 if needed

# === Load model ===
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"Loaded model: {MODEL_NAME} on {DEVICE}")

# === Utility ===
def compute_svd(matrix, k=None):
    """
    Computes economy SVD: W ≈ U S Vh
    If k is None, uses full rank.
    Returns U, S, Vh where Vh is the right singular vectors.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    if k:
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
    return U.to(PRECISION), S.to(PRECISION), Vh.to(PRECISION)

# === Process Layers ===
for i, block in enumerate(model.transformer.h):
    print(f"🔍 Processing layer {i}")

    # --- MLP Projections ---
    fc_weight = block.mlp.c_fc.weight.data.to(DEVICE)
    fc_bias = block.mlp.c_fc.bias.data.to(DEVICE)
    proj_weight = block.mlp.c_proj.weight.data.to(DEVICE)
    proj_bias = block.mlp.c_proj.bias.data.to(DEVICE)

    # Compute SVD on transposed weights to get correct shapes for inference
    U_fc, S_fc, Vh_fc = compute_svd(fc_weight.T)
    U_proj, S_proj, Vh_proj = compute_svd(proj_weight.T)  # proj_weight: [768, 3072] -> [3072, 768]

    mlp_svd = {
        "fc": (U_fc, S_fc, Vh_fc, fc_bias),
        "proj": (U_proj, S_proj, Vh_proj, proj_bias)
    }

    # --- Attention Projections ---
    c_attn_weight = block.attn.c_attn.weight.data.to(DEVICE)
    c_attn_bias = block.attn.c_attn.bias.data.to(DEVICE)
    c_proj_weight = block.attn.c_proj.weight.data.to(DEVICE)
    c_proj_bias = block.attn.c_proj.bias.data.to(DEVICE)

    # Split Q, K, V from c_attn
    hidden_dim = model.config.n_embd
    W_q, W_k, W_v = torch.chunk(c_attn_weight, 3, dim=1)  # Split along hidden_dim dimension
    b_q, b_k, b_v = torch.chunk(c_attn_bias, 3, dim=0)

    # Compute SVD for each component separately
    U_q, S_q, Vh_q = compute_svd(W_q)  # [768, 768] -> [768, 768]
    U_k, S_k, Vh_k = compute_svd(W_k)  # [768, 768] -> [768, 768]
    U_v, S_v, Vh_v = compute_svd(W_v)  # [768, 768] -> [768, 768]
    U_o, S_o, Vh_o = compute_svd(c_proj_weight)  # [768, 768] -> [768, 768]

    attn_svd = {
        "q": (U_q, S_q, Vh_q, b_q),
        "k": (U_k, S_k, Vh_k, b_k),
        "v": (U_v, S_v, Vh_v, b_v),
        "o": (U_o, S_o, Vh_o, c_proj_bias)
    }

    torch.save(
        {"mlp": mlp_svd, "attention": attn_svd},
        os.path.join(OUTPUT_DIR, f"layer_{i}.pt")
    )

print(f"\n✅ All SVDs saved to: {OUTPUT_DIR}/layer_*.pt")
