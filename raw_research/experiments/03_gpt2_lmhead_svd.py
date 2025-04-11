import torch
import torch.nn.functional as F
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize input
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Run input through model to get hidden state for final token
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_state = outputs.hidden_states[-1][:, -1, :]  # shape: [1, 768]

# LM head weights
W = model.lm_head.weight.data  # shape: [50257, 768]
W_t = W.T  # shape: [768, 50257]

print("Running full SVD on lm_head weight...")
start = time.time()
U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
svd_time = time.time() - start
print(f"Done. SVD time: {svd_time:.3f} seconds")
print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")

# Manual logits computation: h @ Wᵗ = (h @ U) @ S @ Vᵗ
timings = {}

with torch.no_grad():
    # Step 1: h @ U
    start = time.time()
    hu = hidden_state @ U  # [1 x 768] @ [768 x 768] = [1 x 768]
    timings["h @ U"] = time.time() - start

    # Step 2: (h @ U) * S
    start = time.time()
    hus = hu * S  # element-wise [1 x 768] * [768]
    timings["(h @ U) * S"] = time.time() - start

    # Step 3: @ Vh
    start = time.time()
    logits_svd = hus @ Vh  # [1 x 768] @ [768 x 50257] = [1 x 50257]
    timings["@ Vh"] = time.time() - start

    # Compare to normal lm_head output
    logits_orig = hidden_state @ W_t  # [1 x 768] @ [768 x 50257]

# Check difference
max_diff = (logits_orig - logits_svd).abs().max().item()
print(f"\n✅ Max difference between original and SVD logits: {max_diff:.6e}")

# Token prediction
pred_id_orig = torch.argmax(logits_orig, dim=-1)
pred_id_svd = torch.argmax(logits_svd, dim=-1)
pred_token_orig = tokenizer.decode(pred_id_orig.item())
pred_token_svd = tokenizer.decode(pred_id_svd.item())

print(f"Original Prediction: {pred_token_orig}")
print(f"SVD Prediction:     {pred_token_svd}")

# Print timings
print("\n===== SVD-based LM Head Timings =====")
for key, t in timings.items():
    print(f"{key:20s}: {t * 1000:.3f} ms")


""" Output:
Running full SVD on lm_head weight...
Done. SVD time: 1.644 seconds
U: torch.Size([768, 768]), S: torch.Size([768]), Vh: torch.Size([768, 50257])

✅ Max difference between original and SVD logits: 3.743744e-02
Original Prediction:  uncertain
SVD Prediction:      uncertain

===== SVD-based LM Head Timings =====
h @ U               : 0.057 ms
(h @ U) * S         : 0.007 ms
@ Vh                : 9.042 ms
"""
