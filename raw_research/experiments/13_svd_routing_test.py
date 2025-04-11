import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import time

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# === Load SVD ===
svd_path = "svd_lm_head_gpt2.pt"
data = torch.load(svd_path)
U, S, Vh = data["U"], data["S"], data["Vh"]

# === Params ===
prompts_train = ["The future of AI is", "The moon is", "Once upon a time,"]
prompts_test = ["The robot said", "In the year 3000", "Once in a while,"]

codebook = {}  # key: code (h@U)*S, value: token_id
failures = []

# === Encode training prompts ===
for prompt in prompts_train:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]  # [1, hidden]
        logits = out.logits[:, -1, :]
        true_token_id = torch.argmax(logits, dim=-1).item()

        hU = h @ U  # [1 x 768]
        code = (hU * S).squeeze().cpu()  # [768]
        codebook[prompt] = (code, true_token_id)

print("✅ Built routing codebook from training prompts\n")

# === Test prompts ===
total = 0
correct = 0
max_l2_diff = 0

for prompt in prompts_test:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]  # [1, hidden]
        logits = out.logits[:, -1, :]
        true_token_id = torch.argmax(logits, dim=-1).item()

        hU = h @ U
        query_code = (hU * S).squeeze().cpu()

        # Find nearest code in codebook (L2 distance)
        best_match = None
        best_dist = float("inf")
        best_token = None

        for base_code, token_id in codebook.values():
            dist = torch.norm(query_code - base_code, p=2).item()
            if dist < best_dist:
                best_dist = dist
                best_token = token_id

        total += 1
        if best_token == true_token_id:
            correct += 1
        else:
            failures.append((prompt, tokenizer.decode(best_token), tokenizer.decode(true_token_id), best_dist))

        max_l2_diff = max(max_l2_diff, best_dist)

# === Results ===
print("===== 🔍 ROUTING TEST RESULTS =====")
print(f"Test Prompts        : {len(prompts_test)}")
print(f"Correct Predictions : {correct}/{total}")
print(f"Accuracy            : {correct/total*100:.2f}%")
print(f"Max L2 Distance     : {max_l2_diff:.4f}\n")

if failures:
    print("❌ Mismatches:")
    for p, pred, gold, dist in failures:
        print(f"Prompt: '{p}'\n  Pred: '{pred}' | Gold: '{gold}' | ΔL2: {dist:.4f}\n")
else:
    print("✅ All test prompts routed to correct token ID")


""" Output:
✅ Built routing codebook from training prompts

===== 🔍 ROUTING TEST RESULTS =====
Test Prompts        : 3
Correct Predictions : 0/3
Accuracy            : 0.00%
Max L2 Distance     : 5279.2373

❌ Mismatches:
Prompt: 'The robot said'
  Pred: ' the' | Gold: ' it' | ΔL2: 3222.7466

Prompt: 'In the year 3000'
  Pred: ' the' | Gold: ',' | ΔL2: 5279.2373

Prompt: 'Once in a while,'
  Pred: ' the' | Gold: ' you' | ΔL2: 214.0399
"""

""" Analysis:
This test shows raw code routing is not viable — which is a great result because it clarifies the boundary.
"""