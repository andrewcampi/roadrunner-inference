import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F


def get_reference_kv(model, input_ids, max_new_tokens):
    generated = input_ids.clone()
    kvs = []

    with torch.no_grad():
        out = model(
            input_ids=generated,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        kvs.append([tuple(t.clone() for t in pair) for pair in past])

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            kvs.append([tuple(t.clone() for t in pair) for pair in past])
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return kvs[1:]  # to match incremental (after prompt)



def get_incremental_kv(model, input_ids, max_new_tokens):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past = list(out.past_key_values)
        generated = input_ids.clone()

    kvs = [[tuple(t.clone() for t in pair) for pair in past]]

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=tuple(past),
                use_cache=True,
                return_dict=True,
            )
            new_past = out.past_key_values

            # Append only the new K/V to previous
            for l in range(len(past)):
                pk, pv = past[l]
                nk, nv = new_past[l]
                past[l] = (
                    torch.cat([pk, nk[:, :, -1:, :]], dim=2),
                    torch.cat([pv, nv[:, :, -1:, :]], dim=2),
                )
            kvs.append([tuple(t.clone() for t in pair) for pair in past]) 

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return kvs[1:]  # list of past_key_values after each step


def compare_kv_sets(ref_kvs, inc_kvs):
    assert len(ref_kvs) == len(inc_kvs), "KV sequence length mismatch!"

    for step, (ref, inc) in enumerate(zip(ref_kvs, inc_kvs)):
        print(f"\n🔍 Step {step + 1}")
        for l, (r, i) in enumerate(zip(ref, inc)):
            rk, rv = r
            ik, iv = i

            assert rk.shape == ik.shape, f"[Layer {l}] Key shape mismatch: {rk.shape} vs {ik.shape}"
            assert rv.shape == iv.shape, f"[Layer {l}] Value shape mismatch: {rv.shape} vs {iv.shape}"

            k_diff = (rk - ik).abs().mean().item()
            v_diff = (rv - iv).abs().mean().item()

            print(f"[Layer {l}] ΔKey mean abs diff: {k_diff:.6e} | ΔValue mean abs diff: {v_diff:.6e}")

            # Optional: cosine similarity check
            cos_k = F.cosine_similarity(rk.flatten(), ik.flatten(), dim=0).item()
            cos_v = F.cosine_similarity(rv.flatten(), iv.flatten(), dim=0).item()
            print(f"           CosSim Keys: {cos_k:.6f} | CosSim Values: {cos_v:.6f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    prompt = "The moon is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_tokens = 5  # Keep small for clarity

    print("⏳ Capturing reference past_key_values...")
    ref_kvs = get_reference_kv(model, input_ids, max_tokens)

    print("⚙️  Building manual incremental past_key_values...")
    inc_kvs = get_incremental_kv(model, input_ids, max_tokens)

    print("\n📊 Comparing reference vs incremental...")
    compare_kv_sets(ref_kvs, inc_kvs)


""" Output:
⏳ Capturing reference past_key_values...
⚙️  Building manual incremental past_key_values...

📊 Comparing reference vs incremental...

🔍 Step 1
[Layer 0] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 1] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 2] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 3] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 4] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 5] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 6] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 7] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 8] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 9] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 10] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 11] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000

🔍 Step 2
[Layer 0] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 1] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 2] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 3] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 4] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 5] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 6] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000001
[Layer 7] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 8] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 9] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 10] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 11] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000

🔍 Step 3
[Layer 0] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 1] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 2] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 3] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 4] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 5] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 6] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000001
[Layer 7] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 8] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 9] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 10] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 11] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000

🔍 Step 4
[Layer 0] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 1] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 2] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 3] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 4] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 5] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 6] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000001
[Layer 7] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 8] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 9] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 10] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 11] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000

🔍 Step 5
[Layer 0] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 1] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 2] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 3] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 4] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 5] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 6] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 7] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 8] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 9] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 10] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
[Layer 11] ΔKey mean abs diff: 0.000000e+00 | ΔValue mean abs diff: 0.000000e+00
           CosSim Keys: 1.000000 | CosSim Values: 1.000000
"""