import torch
import torch.nn as nn
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import numpy as np

class RoutingLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, top_k: int = 32):
        super().__init__()
        self.top_k = top_k
        self.out_dim, self.in_dim = weight.shape
        self.weight = weight.detach().cpu()
        self.bias = bias.detach().cpu() if bias is not None else None

        self.index = faiss.IndexFlatIP(self.in_dim)
        self.index.add(self.weight.numpy().astype("float32"))

    def forward(self, x: torch.Tensor):
        x_flat = x.view(-1, self.in_dim)
        outputs = []

        for i in range(x_flat.size(0)):
            h_np = x_flat[i].detach().cpu().numpy().astype("float32")[None, :]
            _, topk_ids = self.index.search(h_np, self.top_k)
            topk_ids = topk_ids[0]

            # Step 1: Full fallback matmul
            full = torch.matmul(self.weight.to(x.device), x_flat[i])  # [out_dim]
            if self.bias is not None:
                full += self.bias.to(x.device)

            # Step 2: Overwrite top-k slice with reranked values
            W_topk = self.weight[topk_ids]
            reranked = torch.matmul(W_topk.to(x.device), x_flat[i])  # [top_k]
            if self.bias is not None:
                reranked += self.bias[topk_ids].to(x.device)
            full[topk_ids] = reranked

            outputs.append(full)

        return torch.stack(outputs, dim=0).view(*x.shape[:-1], self.out_dim)

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ==== MLP block to test ====
block = model.transformer.h[0]
original_fc = block.mlp.c_fc
original_proj = block.mlp.c_proj

# ==== Input ====
prompt = "The moon is full and the sky is clear."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1)).to(device))
    hidden = model.transformer.drop(hidden)

# ==== Original MLP output ====
with torch.no_grad():
    t0 = time.time()
    a = original_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_orig = original_proj(a)
    t_orig = time.time() - t0

# ==== Replace with hybrid RoutingLinear ====
routing_fc = RoutingLinear(original_fc.weight.T, original_fc.bias, top_k=32).to(device)
routing_proj = RoutingLinear(original_proj.weight.T, original_proj.bias, top_k=32).to(device)

block.mlp.c_fc = routing_fc
block.mlp.c_proj = routing_proj

# ==== Routed MLP output ====
with torch.no_grad():
    t0 = time.time()
    a = routing_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_route = routing_proj(a)
    t_route = time.time() - t0

# ==== Compare ====
diff = (out_orig - out_route).abs().max().item()

print("\n======= 🔁 Hybrid Routing MLP Benchmark =======")
print(f"Max absolute diff   : {diff:.4e}")
print(f"Original MLP time   : {1000 * t_orig:.3f} ms")
print(f"Routed MLP time     : {1000 * t_route:.3f} ms")
print(f"Speedup             : {t_orig / t_route:.2f}×")
