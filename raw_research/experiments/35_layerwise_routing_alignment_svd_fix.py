import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# === Fast Routing Linear with SVD alignment and weighted logits ===
class WeightedSVDAlignedRoutingLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, top_k: int = 64, svd_components: int = 32, blend_factor: float = 0.5):
        super().__init__()
        self.top_k = top_k
        self.out_dim, self.in_dim = weight.shape
        self.svd_components = svd_components
        self.blend_factor = blend_factor  # Weighting factor to reduce drift

        self.weight_raw = weight.detach()
        self.bias = bias.detach() if bias is not None else None

        # Perform SVD on weights to reduce dimensionality (align them)
        U, S, V = torch.svd(self.weight_raw)  # U = [out_dim, svd_components], V = [in_dim, svd_components]
        self.U = U[:, :self.svd_components]  # [out_dim, svd_components]
        self.S = torch.diag(S[:self.svd_components])  # [svd_components, svd_components]
        self.V = V[:, :self.svd_components]  # [in_dim, svd_components]

    def forward(self, x: torch.Tensor):
        x_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_dim)

        # Apply SVD-based alignment (reduce dimensions using U)
        x_svd = torch.matmul(x_flat, self.V)  # [B*T, svd_components]
        
        # Cosine similarity for top-k selection
        logits = torch.matmul(x_svd, self.S)  # [B*T, svd_components]
        logits = torch.matmul(logits, self.U.T)  # [B*T, out_dim]
        topk_vals, topk_ids = torch.topk(logits, self.top_k, dim=1)

        # Full dot product for comparison
        dense_logits = torch.matmul(x_flat, self.weight_raw.T)

        # Soft blending (weighted logits instead of hard masking)
        mask = torch.zeros_like(dense_logits)
        mask.scatter_(1, topk_ids, 1.0)
        weighted_output = dense_logits * (1 - self.blend_factor) + mask * self.blend_factor

        if self.bias is not None:
            weighted_output += self.bias

        return weighted_output.view(*x_shape, self.out_dim)

# === Replace MLPs in GPT2 block ===
def route_mlp_only_with_reduced_drift(block, top_k=64, svd_components=32, blend_factor=0.5):
    block.mlp.c_fc = WeightedSVDAlignedRoutingLinear(block.mlp.c_fc.weight.T, block.mlp.c_fc.bias, top_k, svd_components, blend_factor).to(block.mlp.c_fc.weight.device)
    block.mlp.c_proj = WeightedSVDAlignedRoutingLinear(block.mlp.c_proj.weight.T, block.mlp.c_proj.bias, top_k, svd_components, blend_factor).to(block.mlp.c_proj.weight.device)

# === Apply SVD-based routing with reduced drift to final N blocks ===
def route_last_n_mlp_blocks_with_reduced_drift(model, n_blocks=4, top_k=64, svd_components=32, blend_factor=0.5):
    total = len(model.transformer.h)
    for i in range(total - n_blocks, total):
        route_mlp_only_with_reduced_drift(model.transformer.h[i], top_k, svd_components, blend_factor)

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prompt = "The moon is full and the sky is clear. " * 30
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# ==== Ground truth ====
model_gt = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
with torch.no_grad():
    out_gt = model_gt(input_ids)
    tok_gt = torch.argmax(out_gt.logits[:, -1, :], dim=-1)

# ==== Block-by-block scan ====
print("\n======= 🔎 Routing Fidelity Scanner (MLP with reduced drift per block) =======")
for n in range(1, 13):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    route_last_n_mlp_blocks_with_reduced_drift(model, n_blocks=n, top_k=64, svd_components=32, blend_factor=0.5)

    with torch.no_grad():
        out = model(input_ids)
        tok = torch.argmax(out.logits[:, -1, :], dim=-1)

    match = torch.equal(tok, tok_gt)
    print(f"Blocks routed: {n:2d} | Token match: {'✅' if match else '❌'}")
