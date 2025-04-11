import torch
import time
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.linalg import svd

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

block = model.transformer.h[0]  # Focus on MLP block 0


# ✅ Correct weight shape for SVD: [3072, 768]
fc_weight = block.mlp.c_fc.weight.data.clone().T  # ✅ force [3072, 768]
fc_bias = block.mlp.c_fc.bias.data.clone()
proj_weight = block.mlp.c_proj.weight.data.clone().T  # [3072, 768]
proj_bias = block.mlp.c_proj.bias.data.clone()

# === SVD on W_fc ===
U, S, Vh = svd(fc_weight, full_matrices=False)  # [3072, 768] = U @ diag(S) @ Vh


# === Input ===
prompt = "The sky was clear and stars were bright. "
inputs = tokenizer(prompt, return_tensors="pt").to(device)
hidden = model.transformer.wte(inputs.input_ids)[0][-1].unsqueeze(0)  # [1, 768]

# === Full MLP forward ===
def full_mlp(x):
    return block.mlp.c_proj(F.gelu(block.mlp.c_fc(x)))

# === Routed MLP forward with α blending ===
def routed_mlp(x, alpha=1.0):
    code = x @ Vh  # [1, 768] @ [768, 768] → [1, 768]
    code_scaled = code * S  # Elementwise scale
    routed_hidden = F.gelu(code_scaled @ U.T + fc_bias)  # [1, 768] @ [768, 3072] → [1, 3072]
    routed_out = routed_hidden @ proj_weight.T + proj_bias  # [1, 3072] @ [768, 3072].T → [1, 768]

    full_out = full_mlp(x)
    blended = alpha * routed_out + (1 - alpha) * full_out
    return blended, full_out


print(f"W_fc shape : {fc_weight.shape}")
print(f"U shape    : {U.shape}")
print(f"S shape    : {S.shape}")
print(f"Vh shape   : {Vh.shape}")

# === Run + Time ===
alpha = 0.7
with torch.no_grad():
    t0 = time.time()
    routed_out, full_out = routed_mlp(hidden, alpha)
    t_routed = time.time() - t0

    t0 = time.time()
    baseline_out = full_mlp(hidden)
    t_full = time.time() - t0

# === Metrics ===
l2 = torch.norm(routed_out - baseline_out).item()
cos = F.cosine_similarity(routed_out, baseline_out).item()
tok_match = torch.argmax(routed_out) == torch.argmax(baseline_out)

# === Output ===
print("\n🧪 Adaptive MLP Residual Routing (Layer 0)")
print(f"Blend factor α          : {alpha}")
print(f"Token match             : {'✅' if tok_match else '❌'}")
print(f"L2 drift                : {l2:.4f}")
print(f"Cosine similarity       : {cos:.6f}")
print(f"Full matmul time        : {t_full * 1000:.3f} ms")
print(f"Routed output time      : {t_routed * 1000:.3f} ms")

""" Output:
W_fc shape : torch.Size([3072, 768])
U shape    : torch.Size([3072, 768])
S shape    : torch.Size([768])
Vh shape   : torch.Size([768, 768])

🧪 Adaptive MLP Residual Routing (Layer 0)
Blend factor α          : 0.7
Token match             : ✅
L2 drift                : 50.7521
Cosine similarity       : 0.682126
Full matmul time        : 0.258 ms
Routed output time      : 1.980 ms
"""
