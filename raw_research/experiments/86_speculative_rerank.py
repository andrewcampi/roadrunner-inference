import torch
import time
import torch.nn.functional as F
import numpy as np

# === Settings ===
vocab_size = 50000
hidden_dim = 4096
proj_dim = 1024
repeats = 200
chunk_size = 16

# === Simulated Tensors ===
torch.manual_seed(42)
device = "mps"
hidden_states = torch.randn(repeats, hidden_dim).to(device)
vocab_weight = torch.randn(vocab_size, hidden_dim).to(device)

# === PCA-like Projection via SVD ===
with torch.no_grad():
    _, _, v = torch.svd(vocab_weight)
    projection_matrix = v[:, :proj_dim].to(device)
    projected_vocab = torch.matmul(vocab_weight, projection_matrix)

# === Naive Full Rerank (Ground Truth) ===
def rerank_naive_batched(hidden_batch, weight, top_k):
    tokens = []
    scores = []
    for hidden in hidden_batch:
        sims = torch.matmul(hidden.unsqueeze(0), weight.T)
        topk_vals, topk_idxs = torch.topk(sims, top_k, dim=-1)
        topk_vectors = weight[topk_idxs[0]]
        rerank_scores = torch.matmul(hidden.unsqueeze(0), topk_vectors.T)
        best_idx = torch.argmax(rerank_scores, dim=-1)
        best_token = topk_idxs[0, best_idx].item()
        tokens.append(best_token)
        scores.append(sims[0, best_token].item())
    return tokens, scores

# === Speculative Reranking with Dot Product Threshold Verification ===
def speculative_rerank_with_dot_verification(hidden_batch, weight, proj_weight, proj_mat, top_k_rerank, top_k_route, threshold):
    accepted_tokens = []
    accepted_flags = []
    for i in range(0, hidden_batch.shape[0], chunk_size):
        chunk = hidden_batch[i:i+chunk_size]
        proj_hidden = torch.matmul(chunk, proj_mat)
        sims = torch.matmul(proj_hidden, proj_weight.T)
        topk_vals, topk_idxs = sims.topk(top_k_route, dim=-1)

        topk_vectors = torch.stack([weight[idx] for idx in topk_idxs])
        hidden_exp = chunk.unsqueeze(1)
        scores = torch.matmul(hidden_exp, topk_vectors.transpose(1, 2)).squeeze(1)

        rerank_topk_vals, rerank_topk_idxs = scores.topk(top_k_rerank, dim=-1)
        best_token_indices = torch.gather(topk_idxs, 1, rerank_topk_idxs[:, :1]).squeeze(1)

        # Dot-product verification
        chosen_vecs = weight[best_token_indices]
        dot_scores = torch.einsum('bd,bd->b', chunk, chosen_vecs)
        accepted = dot_scores >= threshold
        final_tokens = best_token_indices.masked_fill(~accepted, -1)

        accepted_tokens.extend(final_tokens.tolist())
        accepted_flags.extend(accepted.tolist())
    return accepted_tokens, accepted_flags

# === Benchmark ===
def benchmark(fn, *args):
    start = time.time()
    result = fn(*args)
    return result, time.time() - start

# === Run Speculative Test ===
print("\n🔬 Speculative Rerank w/ Dot Product Verification")
tok_naive, naive_scores = rerank_naive_batched(hidden_states, vocab_weight, 64)
_, baseline_time = benchmark(lambda: rerank_naive_batched(hidden_states, vocab_weight, 64))
print(f"Naive (batched)           | Time: {baseline_time:.4f}s | Speed: {repeats / baseline_time:.2f} tok/s")

# === Sweep configuration ===
best_speed = 0
best_config = None
for percentile in [0, 5, 10, 15, 20]:
    for route_k in [512, 1024, 2048]:
        for rerank_k in [64, 32, 16]:
            threshold = np.percentile(naive_scores, percentile)
            (tok_spec, accepted_flags), spec_time = benchmark(
                lambda: speculative_rerank_with_dot_verification(
                    hidden_states, vocab_weight, projected_vocab, projection_matrix,
                    rerank_k, route_k, threshold
                )
            )
            valid_indices = [i for i, flag in enumerate(accepted_flags) if flag]
            accepted = [tok_spec[i] for i in valid_indices]
            correct = sum(1 for i in valid_indices if tok_spec[i] == tok_naive[i])
            accepted_rate = len(accepted) / repeats
            accuracy = correct / len(accepted) if accepted else 0
            speed = repeats / spec_time
            print(f"Route={route_k:<4} Rerank={rerank_k:<4} P{percentile:<2} | Speed: {speed:6.2f} tok/s | Accepted: {accepted_rate*100:5.1f}% | Match: {accuracy*100:5.1f}%")

            if accuracy >= 0.95 and speed > best_speed:
                best_speed = speed
                best_config = (route_k, rerank_k, percentile, speed, accuracy, accepted_rate)

if best_config:
    print(f"\n✅ Best Config => Route={best_config[0]}, Rerank={best_config[1]}, P{best_config[2]} | Speed: {best_config[3]:.2f} tok/s | Match: {best_config[4]*100:.2f}% | Accepted: {best_config[5]*100:.2f}%")
else:
    print("\n❌ No configuration reached 95% match accuracy.")


""" Output:
🔬 Speculative Rerank w/ Dot Product Verification
Naive (batched)           | Time: 1.8709s | Speed: 106.90 tok/s
Route=512  Rerank=64   P0  | Speed: 406.17 tok/s | Accepted:  96.5% | Match:  62.7%
Route=512  Rerank=32   P0  | Speed: 441.53 tok/s | Accepted:  96.5% | Match:  62.7%
Route=512  Rerank=16   P0  | Speed: 442.43 tok/s | Accepted:  96.5% | Match:  62.7%
Route=1024 Rerank=64   P0  | Speed: 256.51 tok/s | Accepted:  97.5% | Match:  74.4%
Route=1024 Rerank=32   P0  | Speed: 254.71 tok/s | Accepted:  97.5% | Match:  74.4%
Route=1024 Rerank=16   P0  | Speed: 255.34 tok/s | Accepted:  97.5% | Match:  74.4%
Route=2048 Rerank=64   P0  | Speed: 136.54 tok/s | Accepted:  98.5% | Match:  84.3%
Route=2048 Rerank=32   P0  | Speed: 137.84 tok/s | Accepted:  98.5% | Match:  84.3%
Route=2048 Rerank=16   P0  | Speed: 139.72 tok/s | Accepted:  98.5% | Match:  84.3%
Route=512  Rerank=64   P5  | Speed: 439.67 tok/s | Accepted:  82.5% | Match:  70.9%
Route=512  Rerank=32   P5  | Speed: 441.55 tok/s | Accepted:  82.5% | Match:  70.9%
Route=512  Rerank=16   P5  | Speed: 439.61 tok/s | Accepted:  82.5% | Match:  70.9%
Route=1024 Rerank=64   P5  | Speed: 260.15 tok/s | Accepted:  88.0% | Match:  79.5%
Route=1024 Rerank=32   P5  | Speed: 260.00 tok/s | Accepted:  88.0% | Match:  79.5%
Route=1024 Rerank=16   P5  | Speed: 260.18 tok/s | Accepted:  88.0% | Match:  79.5%
Route=2048 Rerank=64   P5  | Speed: 141.33 tok/s | Accepted:  92.0% | Match:  86.4%
Route=2048 Rerank=32   P5  | Speed: 141.52 tok/s | Accepted:  92.0% | Match:  86.4%
Route=2048 Rerank=16   P5  | Speed: 141.54 tok/s | Accepted:  92.0% | Match:  86.4%
Route=512  Rerank=64   P10 | Speed: 440.95 tok/s | Accepted:  73.0% | Match:  76.0%
Route=512  Rerank=32   P10 | Speed: 440.72 tok/s | Accepted:  73.0% | Match:  76.0%
Route=512  Rerank=16   P10 | Speed: 440.25 tok/s | Accepted:  73.0% | Match:  76.0%
Route=1024 Rerank=64   P10 | Speed: 260.21 tok/s | Accepted:  79.5% | Match:  83.6%
Route=1024 Rerank=32   P10 | Speed: 259.99 tok/s | Accepted:  79.5% | Match:  83.6%
Route=1024 Rerank=16   P10 | Speed: 259.84 tok/s | Accepted:  79.5% | Match:  83.6%
Route=2048 Rerank=64   P10 | Speed: 141.47 tok/s | Accepted:  84.0% | Match:  89.9%
Route=2048 Rerank=32   P10 | Speed: 141.37 tok/s | Accepted:  84.0% | Match:  89.9%
Route=2048 Rerank=16   P10 | Speed: 141.57 tok/s | Accepted:  84.0% | Match:  89.9%
Route=512  Rerank=64   P15 | Speed: 440.87 tok/s | Accepted:  65.5% | Match:  80.2%
Route=512  Rerank=32   P15 | Speed: 442.75 tok/s | Accepted:  65.5% | Match:  80.2%
Route=512  Rerank=16   P15 | Speed: 442.67 tok/s | Accepted:  65.5% | Match:  80.2%
Route=1024 Rerank=64   P15 | Speed: 260.06 tok/s | Accepted:  73.5% | Match:  86.4%
Route=1024 Rerank=32   P15 | Speed: 260.31 tok/s | Accepted:  73.5% | Match:  86.4%
Route=1024 Rerank=16   P15 | Speed: 256.80 tok/s | Accepted:  73.5% | Match:  86.4%
Route=2048 Rerank=64   P15 | Speed: 141.31 tok/s | Accepted:  78.5% | Match:  92.4%
Route=2048 Rerank=32   P15 | Speed: 141.42 tok/s | Accepted:  78.5% | Match:  92.4%
Route=2048 Rerank=16   P15 | Speed: 141.56 tok/s | Accepted:  78.5% | Match:  92.4%
Route=512  Rerank=64   P20 | Speed: 439.66 tok/s | Accepted:  61.5% | Match:  79.7%
Route=512  Rerank=32   P20 | Speed: 441.74 tok/s | Accepted:  61.5% | Match:  79.7%
Route=512  Rerank=16   P20 | Speed: 443.09 tok/s | Accepted:  61.5% | Match:  79.7%
Route=1024 Rerank=64   P20 | Speed: 259.08 tok/s | Accepted:  68.5% | Match:  86.1%
Route=1024 Rerank=32   P20 | Speed: 254.89 tok/s | Accepted:  68.5% | Match:  86.1%
Route=1024 Rerank=16   P20 | Speed: 250.74 tok/s | Accepted:  68.5% | Match:  86.1%
Route=2048 Rerank=64   P20 | Speed: 138.40 tok/s | Accepted:  73.0% | Match:  92.5%
Route=2048 Rerank=32   P20 | Speed: 139.02 tok/s | Accepted:  73.0% | Match:  92.5%
Route=2048 Rerank=16   P20 | Speed: 141.47 tok/s | Accepted:  73.0% | Match:  92.5%

❌ No configuration reached 95% match accuracy.
"""