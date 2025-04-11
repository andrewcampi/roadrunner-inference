import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time


def run_with_recomputed_kv(model, tokenizer, input_ids, max_new_tokens):
    model.eval()
    past_key_values = None
    generated = input_ids.clone()

    t0 = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:],  # Only new token
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated, time.time() - t0


def run_with_frozen_kv(model, tokenizer, input_ids, max_new_tokens):
    model.eval()
    generated = input_ids.clone()

    # Run once to compute K/V for the prompt
    with torch.no_grad():
        init = model(
            input_ids=generated,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = init.past_key_values

    t0 = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Only update Q from new token, reuse old K/V
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            # Keep old past_key_values (don't update with new K/V)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated, time.time() - t0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    prompt = "The moon is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_new_tokens = 10

    print("🚀 Running with full K/V recomputation...")
    output1, time1 = run_with_recomputed_kv(model, tokenizer, input_ids, max_new_tokens)

    print("🧊 Running with frozen K/V cache...")
    output2, time2 = run_with_frozen_kv(model, tokenizer, input_ids, max_new_tokens)

    text1 = tokenizer.decode(output1[0])
    text2 = tokenizer.decode(output2[0])

    print("\n=== Comparison ===")
    print(f"Time (recompute): {time1:.3f}s")
    print(f"Time (frozen)   : {time2:.3f}s")
    print(f"Speedup         : {time1 / time2:.2f}×")
    print(f"Exact match     : {'✅' if text1 == text2 else '❌'}")
    if text1 != text2:
        print("\nOutput mismatch:")
        print(f"[Recomputed] {text1}")
        print(f"[Frozen KV]  {text2}")


""" Output:
🚀 Running with full K/V recomputation...
🧊 Running with frozen K/V cache...

=== Comparison ===
Time (recompute): 0.109s
Time (frozen)   : 0.091s
Speedup         : 1.20×
Exact match     : ❌

Output mismatch:
[Recomputed] The moon is the most important thing to do.

The
[Frozen KV]  The moon is a very close to beaming, in the only
"""