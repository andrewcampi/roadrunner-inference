import torch
import torch.nn as nn
import torch.fft
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict


# ========== Structured SVD Head Variants ==========

class SVDLMHeadTruncated(nn.Module):
    def __init__(self, svd_path: str, top_k: int):
        super().__init__()
        data = torch.load(svd_path)
        U, S, Vh = data["U"], data["S"], data["Vh"]
        self.register_buffer("U", U[:, :top_k])
        self.register_buffer("S", S[:top_k])
        self.register_buffer("Vh", Vh[:top_k, :])

    def forward(self, h):
        hU = h @ self.U
        hUS = hU * self.S
        return hUS @ self.Vh


class FFTLMHead(nn.Module):
    def __init__(self, svd_path: str):
        super().__init__()
        data = torch.load(svd_path)
        Vh = data["Vh"]  # [768, vocab_size]
        self.register_buffer("Vh_fft", torch.fft.fft(Vh, dim=0))
        self.register_buffer("U", data["U"])
        self.register_buffer("S", data["S"])

    def forward(self, h):
        hU = h @ self.U
        hUS = hU * self.S
        h_fft = torch.fft.fft(hUS, dim=-1)
        logits_fft = h_fft.unsqueeze(-1) * self.Vh_fft  # broadcasting
        logits = torch.fft.ifft(logits_fft, dim=-2).real.sum(dim=-2)
        return logits


# ========== Matrix Analysis ==========

def analyze_vh(svd_path: str):
    data = torch.load(svd_path)
    Vh = data["Vh"]
    S = data["S"]

    rank_threshold = 1e-3
    rank = (S > rank_threshold).sum().item()
    energy = (S**2).cumsum(0) / (S**2).sum()
    energy_at_128 = energy[127].item()
    print(f"🔍 SVD Analysis:")
    print(f"  Vh shape           : {Vh.shape}")
    print(f"  Effective rank     : {rank} / {len(S)}")
    print(f"  Energy @ top-128   : {energy_at_128:.4f}")
    print(f"  Spectrum tail      : min={S[-1]:.2e}, max={S[0]:.2e}")


# ========== Benchmark Runner ==========

def benchmark(model, tokenizer, lm_heads, baseline_head, prompt="Once upon a time", max_tokens=10):
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    results = {}

    for name, lm_head in lm_heads.items():
        print(f"\n🚀 Testing: {name}")
        model.lm_head = lm_head.to(device)
        generated = input_ids.clone()
        past_key_values = None
        total_time = 0.0
        match_count = 0
        max_diff = 0.0

        for _ in range(max_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]
                past_key_values = out.past_key_values

                # Timed forward
                start = time.time()
                logits = lm_head(hidden)
                total_time += time.time() - start

                # Compare
                baseline_logits = baseline_head(hidden)
                diff = (logits - baseline_logits).abs().max().item()
                max_diff = max(max_diff, diff)

                pred_svd = torch.argmax(logits, dim=-1)
                pred_base = torch.argmax(baseline_logits, dim=-1)
                if pred_svd.item() == pred_base.item():
                    match_count += 1

                generated = torch.cat([generated, pred_svd.unsqueeze(0)], dim=1)

        results[name] = {
            "avg_time_ms": (total_time / max_tokens) * 1000,
            "match_count": match_count,
            "max_logit_diff": max_diff
        }

    return results


# ========== Main Launcher ==========

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=True)
    model_name = "gpt2"
    svd_path = "svd_lm_head_gpt2.pt"
    prompt = "Once upon a time, "
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🔎 Running Matrix Structure Analysis")
    analyze_vh(svd_path)

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    baseline_head = nn.Linear(768, 50257, bias=False)
    baseline_head.weight.data.copy_(model.lm_head.weight.data)

    lm_heads = {
        "baseline": baseline_head,
        "svd_k256": SVDLMHeadTruncated(svd_path, top_k=256),
        "svd_k128": SVDLMHeadTruncated(svd_path, top_k=128),
        "svd_k64": SVDLMHeadTruncated(svd_path, top_k=64),
        "fft_sim": FFTLMHead(svd_path),
    }

    print("\n⚙️ Running Benchmarks...")
    results = benchmark(model, tokenizer, lm_heads, baseline_head, prompt)

    print("\n======= 🧪 RESULTS =======")
    for name, r in results.items():
        print(f"\n🔹 {name}")
        print(f"  Avg time/token   : {r['avg_time_ms']:.3f} ms")
        print(f"  Match count      : {r['match_count']}/10")
        print(f"  Max logit diff   : {r['max_logit_diff']:.6f}")
