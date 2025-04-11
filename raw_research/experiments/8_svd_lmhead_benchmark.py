import torch
import torch.nn as nn
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict


# ===== LM Head Implementations =====

class BaselineLMHead(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight.T  # [768, vocab]
    
    def forward(self, h):
        return h @ self.weight


class SVDLMHead(nn.Module):
    def __init__(self, svd_path):
        super().__init__()
        data = torch.load(svd_path)
        self.register_buffer("U", data["U"])
        self.register_buffer("S", data["S"])
        self.register_buffer("Vh", data["Vh"])

    def forward(self, h):
        hU = h @ self.U
        hUS = hU * self.S
        return hUS @ self.Vh


# ====== Test Harness ======

def run_benchmark(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    lm_heads: Dict[str, nn.Module],
    baseline_head: nn.Module,
    prompt: str,
    max_tokens: int = 10
):
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    results = {}

    for name, lm_head in lm_heads.items():
        print(f"\n🔬 Testing: {name}")
        model.lm_head = lm_head.to(device)
        generated = input_ids.clone()
        past_key_values = None
        total_time = 0.0
        match_count = 0
        max_logit_diff = 0.0

        for _ in range(max_tokens):
            with torch.no_grad():
                # Forward pass through transformer
                out = model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]  # [1, 768]
                past_key_values = out.past_key_values

                # Compute logits using custom LM head
                start = time.time()
                logits = lm_head(hidden)
                elapsed = time.time() - start
                total_time += elapsed

                # Baseline logits for verification
                baseline_logits = baseline_head(hidden)
                diff = (logits - baseline_logits).abs().max().item()
                max_logit_diff = max(max_logit_diff, diff)

                # Prediction check
                pred_custom = torch.argmax(logits, dim=-1)
                pred_baseline = torch.argmax(baseline_logits, dim=-1)
                if pred_custom.item() == pred_baseline.item():
                    match_count += 1

                # Append next token
                generated = torch.cat([generated, pred_custom.unsqueeze(0)], dim=1)

        results[name] = {
            "avg_time_ms": (total_time / max_tokens) * 1000,
            "match_count": match_count,
            "max_logit_diff": max_logit_diff,
        }

    return results


if __name__ == "__main__":
    model_name = "gpt2"
    prompt = "Once upon a time, "
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load LM heads
    baseline_lm = BaselineLMHead(model.lm_head.weight.data)
    svd_lm = SVDLMHead("svd_lm_head_gpt2.pt")

    lm_heads = {
        "baseline": baseline_lm,
        "svd": svd_lm,
        # Add more here
    }

    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        lm_heads=lm_heads,
        baseline_head=baseline_lm,
        prompt=prompt
    )

    print("\n======= 🧪 BENCHMARK RESULTS =======")
    for name, r in results.items():
        print(f"\n🔹 {name}")
        print(f"  Avg time/token   : {r['avg_time_ms']:.3f} ms")
        print(f"  Token match rate : {r['match_count']}/10")
        print(f"  Max logit diff   : {r['max_logit_diff']:.6f}")


""" Output:
🔬 Testing: baseline

🔬 Testing: svd

======= 🧪 BENCHMARK RESULTS =======

🔹 baseline
  Avg time/token   : 2.259 ms
  Token match rate : 10/10
  Max logit diff   : 0.000000

🔹 svd
  Avg time/token   : 2.324 ms
  Token match rate : 10/10
  Max logit diff   : 0.040161
"""