import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model + MLP block ===
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
mlp = model.transformer.h[0].mlp

# === Step 1: Build output-space index ===
N = 1000  # Number of cached MLP outputs
real_hiddens = []
mlp_outputs = []

print("📦 Building output cache...")
with torch.no_grad():
    for _ in range(N):
        tok = torch.randint(0, tokenizer.vocab_size, (1,))
        h = model.transformer.wte(tok.to(device))  # single token embed
        out = mlp(h)
        real_hiddens.append(h.squeeze())
        mlp_outputs.append(out.squeeze())

real_hiddens = torch.stack(real_hiddens)
mlp_outputs = torch.stack(mlp_outputs)

# === Step 2: Build FAISS index over output space ===
output_dim = mlp_outputs.size(1)
index = faiss.IndexFlatIP(output_dim)
normed_outputs = torch.nn.functional.normalize(mlp_outputs, dim=1)
index.add(normed_outputs.cpu().numpy())

# === Step 3: Test routing by output vector ===
with torch.no_grad():
    # Use a real prompt to get a new hidden state
    input_ids = tokenizer.encode("The future of AI is", return_tensors="pt").to(device)
    out = model.transformer(input_ids, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[1][:, -1, :]  # after block 0
    true_out = mlp(h)

    # Search for nearest precomputed MLP output
    query = torch.nn.functional.normalize(true_out, dim=1).cpu().numpy()
    D, I = index.search(query, 1)  # top-1 match
    routed_out = mlp_outputs[I[0][0]].unsqueeze(0).to(device)

    # Compare routed vs true using normalized vectors
    true_out_normed = torch.nn.functional.normalize(true_out, dim=1)
    routed_out_normed = torch.nn.functional.normalize(routed_out, dim=1)
    cosine_diff = (true_out_normed - routed_out_normed).abs().max().item()
    print(f"\n🔍 Search distance: {D[0][0]:.6f}")
    print(f"🔬 Cosine diff: {cosine_diff:.6f}")

    # Print both vectors' norms for reference
    print(f"✅ true_out.norm():   {true_out.norm().item():.2f}")
    print(f"📦 routed_out.norm(): {routed_out.norm().item():.2f}")



""" Output:
📦 Building output cache...

🔍 Search distance: 0.043199
🔬 Cosine diff: 0.000000
✅ true_out.norm():   2081.92
📦 routed_out.norm(): 47.89
"""

""" Analysis:
Updated Diagnosis: The MLP Output Space Is Non-Homogeneous
What this proves:

Even directionally close vectors in the MLP output space can differ drastically in magnitude and absolute content.

The MLP outputs:

Span multiple orders of magnitude

Are not normalized or scale-invariant

Contain nonlinear spikes in certain dimensions
"""