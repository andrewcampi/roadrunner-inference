import torch
import time
from transformers import GPT2Tokenizer
from sparse_logit_computation import SparseLogitTestSuite  # Assumes the class is saved in this file
import torch.nn.functional as F

# === Initialize the routing engine ===
suite = SparseLogitTestSuite(svd_path="svd_lm_head_gpt2.pt")  # Make sure this file is present
train_prompts, _ = suite.generate_diverse_prompts(num_train=50, num_test=0)
suite.build_code_cache(train_prompts)

# === Routing-based generation ===
def single_neighbor_router(code):
    """Returns the top-1 token ID from the nearest cached code."""
    nearest = suite.find_similar_codes(code, n=3)
    best_entry = nearest[0][1]
    return best_entry["top_ids_dict"][10][0]  # top-10 cache, use top-1 token

def generate_with_routing(prompt, max_new_tokens=20):
    tokenizer = suite.tokenizer
    model = suite.model
    device = suite.device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    past_key_values = None

    predicted_tokens = []
    matches = 0
    start_time = time.time()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:],  # last token
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

            # Compute routing code: (h @ U) * S
            hU = hidden @ suite.U
            code = (hU * suite.S).squeeze()

            # Get routed token ID
            routed_token_id = single_neighbor_router(code)

            # For verification: get actual top-1 from full logits
            full_logits = (hU * suite.S) @ suite.Vh
            true_top1 = torch.argmax(full_logits).item()

            if routed_token_id == true_top1:
                matches += 1

            predicted_tokens.append(tokenizer.decode([routed_token_id], clean_up_tokenization_spaces=True))
            generated = torch.cat([generated, torch.tensor([[routed_token_id]], device=device)], dim=1)

    total_time = time.time() - start_time

    print("\n=== 🚀 Routing-Based Inference Benchmark ===")
    print(f"Prompt: {prompt!r}")
    print("Generated:", ''.join(predicted_tokens))
    print(f"Accuracy vs baseline logits: {matches}/{max_new_tokens}")
    print(f"Total time: {total_time:.4f} s")
    print(f"Avg time per token: {1000 * total_time / max_new_tokens:.2f} ms")

# === Run benchmark ===
generate_with_routing("Once upon a time, there ", max_new_tokens=20)


""" Output:
✅ Built code cache with 50 entries

=== 🚀 Routing-Based Inference Benchmark ===
Prompt: 'Once upon a time, there '
Generated:  is in we have we have we are are are are are are are are are we are we are
Accuracy vs baseline logits: 8/20
Total time: 0.2320 s
Avg time per token: 11.60 ms
"""

""" Analysis:
Let's unpack what probably happened and how to fix it:

❗️ Symptoms
Repeating tokens: "we have we have we are are are are..."

Low accuracy: 8/20 correct vs full logits

Fast speed: ~11ms/token ✅

Prompt gets ignored: Seems like it "forgot" the context quickly

🧠 Diagnosis
This is a cache generalization issue. Here's likely what's happening:

1. Cache is too small / underfit
You only trained the SVD code cache on 50 prompts, and now you're generating from a novel one:

"Once upon a time, there "

The model is computing its SVD code per step, but the nearest neighbors from the 50-prompt cache:

Don’t reflect typical next-token contexts for that input

May produce incorrect top-10 token candidates

Still generate a valid token, but not the one GPT would prefer

🔎 The fact that you got 8/20 right shows that the mechanism kind of works — but your cache just isn't robust enough for open-ended generation.

2. Token-level routing vs Sequence-level training
Your cache was built on single-token completions, not multi-token generations. That means the routing system doesn’t know how to evolve the sequence over time.

Instead of adjusting dynamically, it keeps hitting the same or similar cached patterns, causing:

"are are are are..."

3. Overconfident nearest neighbor
Using just the top-1 neighbor (n=1) with fixed top_ids_dict[10][0] is very brittle.

Even a small mismatch in semantics (e.g., “Once upon a time” vs “In the future”) can throw off routing.

"""