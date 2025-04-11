import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

class SVDLMHead(nn.Module):
    def __init__(self, weight_matrix: torch.Tensor):
        """
        weight_matrix: [vocab_size, hidden_dim]
        This is typically model.lm_head.weight
        """
        super().__init__()
        with torch.no_grad():
            W_t = weight_matrix.T  # [hidden_dim, vocab_size]
            print("Performing full SVD on LM head weight...")
            U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
            print("SVD complete.")

        # Register components as buffers so they move with .to(device)
        self.register_buffer("U", U)           # [hidden_dim, hidden_dim]
        self.register_buffer("S", S)           # [hidden_dim]
        self.register_buffer("Vh", Vh)         # [hidden_dim, vocab_size]

    def forward(self, hidden_state: torch.Tensor):
        # hidden_state: [batch_size, hidden_dim]
        hU = hidden_state @ self.U            # [batch_size, hidden_dim]
        hUS = hU * self.S                     # Element-wise scale
        logits = hUS @ self.Vh                # [batch_size, vocab_size]
        return logits


# ========== Run GPT-2 with SVD LM Head ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Replace LM head with SVDLMHead
original_weight = model.lm_head.weight.data
svd_lm_head = SVDLMHead(original_weight).to(device)
model.lm_head = svd_lm_head  # Drop-in replacement!

# Tokenize input
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate next token manually
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_state = outputs.hidden_states[-1][:, -1, :]  # [1, 768]

    # Time original matmul
    original_logits = hidden_state @ original_weight.T

    # Time SVD path
    start = time.time()
    svd_logits = svd_lm_head(hidden_state)
    svd_time = time.time() - start

# Verify correctness
max_diff = (original_logits - svd_logits).abs().max().item()
pred_id_orig = torch.argmax(original_logits, dim=-1).item()
pred_id_svd = torch.argmax(svd_logits, dim=-1).item()
pred_token_orig = tokenizer.decode(pred_id_orig)
pred_token_svd = tokenizer.decode(pred_id_svd)

print("\n====== VERIFICATION ======")
print(f"Max absolute difference in logits: {max_diff:.6e}")
print(f"Original token: {pred_token_orig}")
print(f"SVD token:     {pred_token_svd}")
print(f"SVD inference time: {svd_time * 1000:.3f} ms")
print(f"Prediction match? {'✅' if pred_id_orig == pred_id_svd else '❌'}")


""" Output:
Performing full SVD on LM head weight...
SVD complete.

====== VERIFICATION ======
Max absolute difference in logits: 3.743744e-02
Original token:  uncertain
SVD token:      uncertain
SVD inference time: 2.308 ms
Prediction match? ✅
"""