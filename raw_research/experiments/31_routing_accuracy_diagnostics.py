import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# === Routing MLPs only ===
class MlpRoutingLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, top_k: int = 64):
        super().__init__()
        self.top_k = top_k
        self.out_dim, self.in_dim = weight.shape

        self.weight_raw = weight.detach()
        self.weight_norm = torch.nn.functional.normalize(weight, dim=1).detach()
        self.bias = bias.detach() if bias is not None else None

    def forward(self, x: torch.Tensor):
        x_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_dim)
        x_norm = torch.nn.functional.normalize(x_flat, dim=1)

        # Cosine similarity for top-k
        logits = torch.matmul(x_norm, self.weight_norm.T)
        topk_vals, topk_ids = torch.topk(logits, self.top_k, dim=1)

        # Full dot product
        dense_logits = torch.matmul(x_flat, self.weight_raw.T)

        # Soft masking
        mask = torch.zeros_like(dense_logits)
        mask.scatter_(1, topk_ids, 1.0)
        output = dense_logits * mask

        if self.bias is not None:
            output += self.bias

        return output.view(*x_shape, self.out_dim)

# === Replace MLP in block ===
def route_mlp_only(block, top_k_mlp=64):
    block.mlp.c_fc = MlpRoutingLinear(block.mlp.c_fc.weight.T, block.mlp.c_fc.bias, top_k=top_k_mlp).to(block.mlp.c_fc.weight.device)
    block.mlp.c_proj = MlpRoutingLinear(block.mlp.c_proj.weight.T, block.mlp.c_proj.bias, top_k=top_k_mlp).to(block.mlp.c_proj.weight.device)

# === Route final N blocks (MLP only) ===
def route_last_n_mlp_blocks(model, n_blocks=4, top_k_mlp=64):
    total = len(model.transformer.h)
    for i in range(total - n_blocks, total):
        route_mlp_only(model.transformer.h[i], top_k_mlp)

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
print("\n======= 🔎 Routing Fidelity Scanner (MLP only) =======")
for n in range(1, 13):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    route_last_n_mlp_blocks(model, n_blocks=n, top_k_mlp=256)

    with torch.no_grad():
        out = model(input_ids)
        tok = torch.argmax(out.logits[:, -1, :], dim=-1)

    match = torch.equal(tok, tok_gt)
    print(f"Blocks routed: {n:2d} | Token match: {'✅' if match else '❌'}")


""" Output:
======= 🔎 Routing Fidelity Scanner (MLP only) =======
Blocks routed:  1 | Token match: ✅
Blocks routed:  2 | Token match: ✅
Blocks routed:  3 | Token match: ✅
Blocks routed:  4 | Token match: ✅
Blocks routed:  5 | Token match: ✅
Blocks routed:  6 | Token match: ❌
Blocks routed:  7 | Token match: ❌
Blocks routed:  8 | Token match: ❌
Blocks routed:  9 | Token match: ❌
Blocks routed: 10 | Token match: ❌
Blocks routed: 11 | Token match: ❌
Blocks routed: 12 | Token match: ❌
"""

""" Analysis:
Adjusting top_k_mlp from 32 to 256 improves the token match, but still is not perfect. Adjusting approach in the next script to try to maintain accuracy.
"""