import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time


def run_incremental_kv(model, tokenizer, input_ids, max_new_tokens):
    model.eval()
    generated = input_ids.clone()

    # Run once to get full past for prompt
    with torch.no_grad():
        init = model(
            input_ids=generated,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = list(init.past_key_values)

    t0 = time.time()
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=tuple(past_key_values),
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits
            new_past = out.past_key_values

            # Append only the latest K/V slice to our running cache
            for l in range(len(past_key_values)):
                past_k, past_v = past_key_values[l]
                new_k, new_v = new_past[l]
                past_key_values[l] = (
                    torch.cat([past_k, new_k[:, :, -1:, :]], dim=2),
                    torch.cat([past_v, new_v[:, :, -1:, :]], dim=2),
                )

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated, time.time() - t0


def run_reference(model, tokenizer, input_ids, max_new_tokens):
    model.eval()
    past = None
    generated = input_ids.clone()

    t0 = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits
            past = out.past_key_values
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
    max_tokens = 10

    print("🔁 Running Reference Inference...")
    out_ref, t_ref = run_reference(model, tokenizer, input_ids, max_tokens)

    print("🧠 Running Incremental K/V...")
    out_inc, t_inc = run_incremental_kv(model, tokenizer, input_ids, max_tokens)

    txt_ref = tokenizer.decode(out_ref[0])
    txt_inc = tokenizer.decode(out_inc[0])

    print("\n=== Comparison ===")
    print(f"Time (ref):     {t_ref:.3f}s")
    print(f"Time (incremental): {t_inc:.3f}s")
    print(f"Speedup        : {t_ref / t_inc:.2f}×")
    print(f"Exact match    : {'✅' if txt_ref == txt_inc else '❌'}")

    if txt_ref != txt_inc:
        print("\n[Reference]   ", txt_ref)
        print("[Incremental] ", txt_inc)


""" Output:
🔁 Running Reference Inference...
🧠 Running Incremental K/V...

=== Comparison ===
Time (ref):     0.121s
Time (incremental): 0.092s
Speedup        : 1.32×
Exact match    : ❌

[Reference]    The moon is the most important thing to do.

The
[Incremental]  The moon is a very bright star, and it is very bright
"""