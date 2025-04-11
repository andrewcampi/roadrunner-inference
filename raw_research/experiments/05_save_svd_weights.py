import torch
from transformers import GPT2LMHeadModel

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
W_t = model.lm_head.weight.data.T  # [768, 50257]

print("Computing SVD...")
U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
print("Done. Saving...")

torch.save({
    "U": U,
    "S": S,
    "Vh": Vh
}, "svd_lm_head_gpt2.pt")

print("Saved as 'svd_lm_head_gpt2.pt'")


""" Output:
Computing SVD...
Done. Saving...
Saved as 'svd_lm_head_gpt2.pt'
"""
