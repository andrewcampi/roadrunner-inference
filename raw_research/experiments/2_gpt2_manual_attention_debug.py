import torch
import torch.nn.functional as F
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize input
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Choose first transformer block to inspect
block = model.transformer.h[0]
ln_f = model.transformer.ln_f
lm_head = model.lm_head
wte = model.transformer.wte
wpe = model.transformer.wpe

# Collect timing info
timings = {
    "qkv_projection": [],
    "qk_matmul": [],
    "softmax": [],
    "av_matmul": [],
    "output_proj": [],
    "layernorm_final": [],
    "lm_head": [],
}

# Inference loop
past_key_values = [None] * len(model.transformer.h)
generated = input_ids
max_new_tokens = 10

for step in range(max_new_tokens):
    print(f"\n--- Step {step+1} ---")

    input_token = generated[:, -1:]

    # Input embeddings
    input_embeds = wte(input_token) + wpe(torch.tensor([[generated.size(1) - 1]], device=device))

    # Run through manual block
    x = input_embeds
    attn = block.attn
    ln_1 = block.ln_1
    ln_2 = block.ln_2
    mlp = block.mlp

    # LayerNorm before attention
    x_ln = ln_1(x)

    # Q, K, V projection
    start = time.time()
    qkv = attn.c_attn(x_ln)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    end = time.time()
    timings["qkv_projection"].append(end - start)
    print(f"[qkv_projection] q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

    # Get past keys and values
    layer_past = past_key_values[0]
    if layer_past is None:
        past_k = k
        past_v = v
    else:
        past_k = torch.cat([layer_past[0], k], dim=1)
        past_v = torch.cat([layer_past[1], v], dim=1)

    past_key_values[0] = (past_k, past_v)

    # Reshape for multi-head attention
    def split_heads(t):
        b, t_len, d = t.size()
        return t.view(b, t_len, attn.num_heads, d // attn.num_heads).transpose(1, 2)

    q = split_heads(q)
    k = split_heads(past_k)
    v = split_heads(past_v)

    # QKᵀ
    start = time.time()
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    end = time.time()
    timings["qk_matmul"].append(end - start)
    print(f"[qk_matmul] scores shape: {attn_scores.shape}")

    # Softmax
    start = time.time()
    attn_probs = F.softmax(attn_scores, dim=-1)
    end = time.time()
    timings["softmax"].append(end - start)
    print(f"[softmax] probs shape: {attn_probs.shape}, sum: {attn_probs.sum(dim=-1)}")

    # AV
    start = time.time()
    attn_output = torch.matmul(attn_probs, v)
    end = time.time()
    timings["av_matmul"].append(end - start)
    print(f"[av_matmul] attn_output shape: {attn_output.shape}")

    # Combine heads
    def merge_heads(t):
        b, h, t_len, d = t.size()
        return t.transpose(1, 2).contiguous().view(b, t_len, h * d)

    attn_output = merge_heads(attn_output)

    # Output projection
    start = time.time()
    attn_output = attn.c_proj(attn_output)
    end = time.time()
    timings["output_proj"].append(end - start)
    print(f"[output_proj] projected shape: {attn_output.shape}")

    # Residual + MLP
    x = x + attn_output
    x_ln2 = ln_2(x)

    x_mlp = mlp(x_ln2)
    x = x + x_mlp

    # Final layer norm
    start = time.time()
    x = ln_f(x)
    end = time.time()
    timings["layernorm_final"].append(end - start)

    # LM head
    start = time.time()
    logits = lm_head(x)
    end = time.time()
    timings["lm_head"].append(end - start)

    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    generated = torch.cat((generated, next_token_id), dim=1)
    print(f"[Generated token] {tokenizer.decode(next_token_id.squeeze())}")

# Final report
print("\n===== Average Component Timings =====")
for key, times in timings.items():
    avg_time = sum(times) / len(times)
    print(f"{key:20s}: {avg_time * 1000:.3f} ms")


""" Output:
--- Step 1 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 1])
[softmax] probs shape: torch.Size([1, 12, 1, 1]), sum: tensor([[[1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.],
         [1.]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  not

--- Step 2 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 2])
[softmax] probs shape: torch.Size([1, 12, 1, 2]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 3 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 3])
[softmax] probs shape: torch.Size([1, 12, 1, 3]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 4 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 4])
[softmax] probs shape: torch.Size([1, 12, 1, 4]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 5 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 5])
[softmax] probs shape: torch.Size([1, 12, 1, 5]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 6 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 6])
[softmax] probs shape: torch.Size([1, 12, 1, 6]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 7 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 7])
[softmax] probs shape: torch.Size([1, 12, 1, 7]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 8 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 8])
[softmax] probs shape: torch.Size([1, 12, 1, 8]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 9 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 9])
[softmax] probs shape: torch.Size([1, 12, 1, 9]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

--- Step 10 ---
[qkv_projection] q shape: torch.Size([1, 1, 768]), k shape: torch.Size([1, 1, 768]), v shape: torch.Size([1, 1, 768])
[qk_matmul] scores shape: torch.Size([1, 12, 1, 10])
[softmax] probs shape: torch.Size([1, 12, 1, 10]), sum: tensor([[[1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000],
         [1.0000]]], grad_fn=<SumBackward1>)
[av_matmul] attn_output shape: torch.Size([1, 12, 1, 64])
[output_proj] projected shape: torch.Size([1, 1, 768])
[Generated token]  yet

===== Average Component Timings =====
qkv_projection      : 0.198 ms
qk_matmul           : 0.043 ms
softmax             : 0.025 ms
av_matmul           : 0.018 ms
output_proj         : 0.069 ms
layernorm_final     : 0.010 ms
lm_head             : 2.847 ms
"""