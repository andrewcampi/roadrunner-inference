import torch
import torch.nn as nn
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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


def generate(model, tokenizer, prompt, max_new_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated = input_ids.clone()
    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:],  # last token
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_drag_race():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load baseline model
    baseline_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    # Load SVD model (with LM head replacement)
    svd_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    svd_model.lm_head = SVDLMHead("svd_lm_head_gpt2.pt").to(device)

    prompt = "Once upon a time, "
    max_tokens = 100

    print("🚗 Running baseline (standard LM head)...")
    start = time.time()
    baseline_output = generate(baseline_model, tokenizer, prompt, max_new_tokens=max_tokens)
    baseline_time = time.time() - start
    print(f"✅ Baseline done in {baseline_time:.2f} seconds.\n")

    print("🚀 Running SVD LM head...")
    start = time.time()
    svd_output = generate(svd_model, tokenizer, prompt, max_new_tokens=max_tokens)
    svd_time = time.time() - start
    print(f"✅ SVD version done in {svd_time:.2f} seconds.\n")

    print("======= 🏁 DRAG RACE RESULTS =======")
    print(f"Baseline Time : {baseline_time:.2f} s")
    print(f"SVD Time      : {svd_time:.2f} s")
    print(f"Speedup       : {baseline_time / svd_time:.2f}x" if svd_time < baseline_time else "No speedup")

    print("\n=== Completion Output (SVD) ===")
    print(svd_output)


if __name__ == "__main__":
    run_drag_race()


""" Output:
🚗 Running baseline (standard LM head)...
✅ Baseline done in 0.99 seconds.

🚀 Running SVD LM head...
✅ SVD version done in 0.98 seconds.

======= 🏁 DRAG RACE RESULTS =======
Baseline Time : 0.99 s
SVD Time      : 0.98 s
Speedup       : 1.01x

=== Completion Output (SVD) ===
Once upon a time, 

The following is a list of the most common errors that occur when you try to use the "set" command.

The following is a list of the most common errors that occur when you try to use the "set" command. The following is a list of the most common errors that occur when you try to use the "set" command. The following is a list of the most common errors that occur when you try to use the "set" command. The following is a
"""