import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from routing_linear import RoutingLinear  # <- from previous step

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ==== Get target block ====
block = model.transformer.h[0]
original_fc = block.mlp.c_fc
original_proj = block.mlp.c_proj

# ==== Prepare input ====
prompt = "The moon is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    hidden = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(1)).to(device))
    hidden = model.transformer.drop(hidden)  # [1, seq_len, 768]

# ==== Run original MLP ====
with torch.no_grad():
    t0 = time.time()
    a = original_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_orig = original_proj(a)
    t_orig = time.time() - t0

# ==== Replace MLP layers with RoutingLinear ====
routing_fc = RoutingLinear(original_fc.weight.data.T, original_fc.bias.data, top_k=32).to(device)
routing_proj = RoutingLinear(original_proj.weight.data.T, original_proj.bias.data, top_k=32).to(device)

block.mlp.c_fc = routing_fc
block.mlp.c_proj = routing_proj

# ==== Run routed MLP ====
with torch.no_grad():
    t0 = time.time()
    a = routing_fc(hidden)
    a = torch.nn.functional.gelu(a)
    out_route = routing_proj(a)
    t_route = time.time() - t0

# ==== Compare ====
diff = (out_orig - out_route).abs().max().item()

print("\n======= 🔁 MLP Routing Benchmark =======")
print(f"Max absolute diff   : {diff:.4e}")
print(f"Original MLP time   : {1000 * t_orig:.3f} ms")
print(f"Routed MLP time     : {1000 * t_route:.3f} ms")
print(f"Speedup             : {t_orig / t_route:.2f}×")


""" Output:
======= 🔁 MLP Routing Benchmark =======
Max absolute diff   : 2.3632e+02
Original MLP time   : 0.825 ms
Routed MLP time     : 3.114 ms
Speedup             : 0.26×
"""

""" Analysis:
Here's the breakdown:

🧪 Result Recap
Metric	Value	Diagnosis
Max diff	236.32 😬	🚨 Very large
Routing time	3.114 ms	🟡 Reasonable for now
Original time	0.825 ms	✅ Fast baseline
Speedup	0.26×	🔁 Expected without PQ
🧠 What’s Going On
The huge diff means that while we fixed the NaN, we didn’t fix the actual numerical fidelity.

Why?

You're zero-filling ~99% of the output neurons, so the model is missing most of its internal representation.

That’s fine when you're routing to the top token, but not okay inside the MLP, which relies on dense activation mixing.

✅ Options to Fix This
🚀 Option A: Do Full Output, Just Sparse Matmul
Instead of filling 99% with zeros, we compute only the necessary rows, and return dense outputs:

Top-k projection: W_topk @ h

Sparse output placement: out[topk_ids] = result

But then return full tensor, without filling the rest with zeros
"""