import torch
import time
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BatchedRoutingLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, top_k: int = 32):
        super().__init__()
        self.top_k = top_k
        self.out_dim, self.in_dim = weight.shape
        self.weight = weight.detach().cpu()
        self.bias = bias.detach().cpu() if bias is not None else None

        self.index = faiss.IndexFlatL2(self.in_dim)
        weight_np = self.weight.numpy().astype("float32", copy=False)
        weight_np = np.ascontiguousarray(weight_np)  # ✅ ensure C-contiguous
        faiss.normalize_L2(weight_np)
        self.index.add(weight_np)

    def forward(self, x: torch.Tensor):
        x_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_dim)  # [B*T, in_dim]
        x_np = x_flat.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(x_np)  # normalize in-place

        print("FAISS input shape:", x_np.shape, "| dtype:", x_np.dtype)

        assert x_np.dtype == np.float32, "FAISS input must be float32"
        assert x_np.ndim == 2 and x_np.shape[1] == self.in_dim, "FAISS input must be [batch, in_dim]"


        # Batch FAISS search
        I = []
        batch_size = 32
        for i in range(0, x_np.shape[0], batch_size):
            chunk = x_np[i:i+batch_size]
            _, I_chunk = self.index.search(chunk, self.top_k)
            I.append(I_chunk)
        I = np.vstack(I)  # [B*T, k]

        outputs = []
        for i in range(x_flat.size(0)):
            topk_ids = I[i]
            W_topk = self.weight[topk_ids].to(x.device)  # [k, in_dim]
            local_logits = torch.matmul(W_topk, x_flat[i])  # [k]
            if self.bias is not None:
                local_logits += self.bias[topk_ids].to(x.device)
            full = torch.matmul(self.weight.to(x.device), x_flat[i])  # fallback
            full[topk_ids] = local_logits
            outputs.append(full)

        out = torch.stack(outputs, dim=0).view(*x_shape, self.out_dim)
        return out

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ==== Target MLP ====
block = model.transformer.h[0]
original_fc = block.mlp.c_fc
original_proj = block.mlp.c_proj

# ==== Prompt ====
prompt = "The moon is full and the sky is clear. " * 20
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1)).to(device))
    hidden = model.transformer.drop(hidden)  # [1, seq_len, 768]

# ==== Original MLP ====
with torch.no_grad():
    t0 = time.time()
    a = original_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_orig = original_proj(a)
    t_orig = time.time() - t0

# ==== Routing MLP (Batched FAISS) ====
routing_fc = BatchedRoutingLinear(original_fc.weight.T, original_fc.bias, top_k=32).to(device)
routing_proj = BatchedRoutingLinear(original_proj.weight.T, original_proj.bias, top_k=32).to(device)
block.mlp.c_fc = routing_fc
block.mlp.c_proj = routing_proj

# ==== Routed MLP Output ====
with torch.no_grad():
    t0 = time.time()
    a = routing_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_routed = routing_proj(a)
    t_routed = time.time() - t0

# ==== Compare ====
diff = (out_orig - out_routed).abs().max().item()

print("\n======= ⚡ Batched FAISS RoutingLinear Benchmark =======")
print(f"Max absolute diff   : {diff:.4e}")
print(f"Original MLP time   : {1000 * t_orig:.3f} ms")
print(f"Routed MLP time     : {1000 * t_routed:.3f} ms")
print(f"Speedup             : {t_orig / t_routed:.2f}×")


""" Output:
FAISS input shape: (201, 768) | dtype: float32
zsh: segmentation fault  python3 26_batch_faiss_routing_linear.py
"""

""" Analysis:
Low level issue with FAISS on MacOS ARM. Trying a more stable approach next.
"""