import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

class SVDLMHead(nn.Module):
    def __init__(self, svd_path: str):
        super().__init__()
        data = torch.load(svd_path)
        self.register_buffer("U", data["U"])
        self.register_buffer("S", data["S"])
        self.register_buffer("Vh", data["Vh"])

    def forward(self, hidden_state: torch.Tensor):
        hU = hidden_state @ self.U
        hUS = hU * self.S
        return hUS @ self.Vh

# ========== Inference ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Swap in cached SVD version of LM head
svd_head = SVDLMHead("svd_lm_head_gpt2.pt").to(device)
model.lm_head = svd_head

# Prompt
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Manual token generation step
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_state = outputs.hidden_states[-1][:, -1, :]

    # Time SVD-based logits with cached weights
    start = time.time()
    logits = svd_head(hidden_state)
    elapsed = time.time() - start

# Prediction
pred_id = torch.argmax(logits, dim=-1)
pred_token = tokenizer.decode(pred_id.item())

print("\n====== Cached SVD Inference ======")
print(f"Predicted token: {pred_token}")
print(f"Inference time (no SVD overhead): {elapsed * 1000:.3f} ms")


""" Output:
====== Cached SVD Inference ======
Predicted token:  uncertain
Inference time (no SVD overhead): 2.326 ms
"""
