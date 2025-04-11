import torch
import time

# === Settings ===
vocab_size = 50000
hidden_dim = 4096
proj_dim = 1024  # Compressed dimension for routing
batch_size = 64
repeats = 200
chunk_size = 16  # Safe chunk for memory

# === Simulated Tensors ===
torch.manual_seed(42)
device = "mps"
hidden_states = torch.randn(repeats, hidden_dim).to(device)
vocab_weight = torch.randn(vocab_size, hidden_dim).to(device)

# === PCA-like Projection via SVD ===
with torch.no_grad():
    u, s, v = torch.svd(vocab_weight)
    projection_matrix = v[:, :proj_dim].to(device)  # [hidden_dim, proj_dim]
    projected_vocab = torch.matmul(vocab_weight, projection_matrix)  # [vocab_size, proj_dim]

# === Baseline (Naive, Batched) ===
def rerank_naive_batched(hidden_batch, weight, top_k):
    tokens = []
    for hidden in hidden_batch:
        sims = torch.matmul(hidden.unsqueeze(0), weight.T)
        topk_vals, topk_idxs = torch.topk(sims, top_k, dim=-1)
        topk_vectors = weight[topk_idxs[0]]
        rerank_scores = torch.matmul(hidden.unsqueeze(0), topk_vectors.T)
        best_idx = torch.argmax(rerank_scores, dim=-1)
        tokens.append(topk_idxs[0, best_idx].item())
    return tokens

# === Chunked Routing + Accurate Rerank ===
def rerank_projected_batched(hidden_batch, weight, proj_weight, proj_mat, top_k_rerank, top_k_route):
    tokens = []
    for i in range(0, hidden_batch.shape[0], chunk_size):
        chunk = hidden_batch[i:i+chunk_size]
        proj_hidden = torch.matmul(chunk, proj_mat)  # [chunk, proj_dim]
        sims = torch.matmul(proj_hidden, proj_weight.T)  # [chunk, vocab_size]
        topk_vals, topk_idxs = sims.topk(top_k_route, dim=-1)

        # Use full precision vectors for reranking
        topk_vectors = torch.stack([weight[idx] for idx in topk_idxs])  # [chunk, top_k_route, hidden_dim]
        hidden_expanded = chunk.unsqueeze(1)  # [chunk, 1, hidden_dim]
        scores = torch.matmul(hidden_expanded, topk_vectors.transpose(1, 2)).squeeze(1)  # [chunk, top_k_route]

        rerank_topk_vals, rerank_topk_idxs = scores.topk(top_k_rerank, dim=-1)
        best_token_indices = torch.gather(topk_idxs, 1, rerank_topk_idxs[:, :1])
        tokens.extend(best_token_indices.squeeze(1).tolist())
    return tokens

# === Benchmark ===
def benchmark(fn, *args):
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start = time.time()
    result = fn(*args)
    total = time.time() - start
    return result, total

# === Run Optimized Sweep ===
print("\n🔬 Optimizing Projected Token Selection")
tok_naive, baseline_time = benchmark(lambda: rerank_naive_batched(hidden_states, vocab_weight, 64))
print(f"Naive (batched)           | Time: {baseline_time:.4f}s | Speed: {repeats / baseline_time:.2f} tok/s")

best_match = 0
best_config = None

for top_k_route in [512, 1024, 2048, 4096]:
    for top_k_rerank in [64, 32, 16]:
        try:
            tok_proj, time_proj = benchmark(
                lambda: rerank_projected_batched(hidden_states, vocab_weight, projected_vocab, projection_matrix, top_k_rerank, top_k_route)
            )
            match_count = sum(1 for a, b in zip(tok_naive, tok_proj) if a == b)
            match_rate = (match_count / repeats) * 100
            speed = repeats / time_proj
            print(f"Route={top_k_route:<5} Rerank={top_k_rerank:<5} | Time: {time_proj:.4f}s | Speed: {speed:.2f} tok/s | Match: {match_rate:.2f}%")

            if match_rate > best_match:
                best_match = match_rate
                best_config = (top_k_route, top_k_rerank, speed)
        except RuntimeError as e:
            print(f"Route={top_k_route:<5} Rerank={top_k_rerank:<5} | FAILED: {str(e)}")

print(f"\n✅ Best Config: Route={best_config[0]}, Rerank={best_config[1]}, Speed={best_config[2]:.2f} tok/s, Match={best_match:.2f}%\n")


""" Output:
🔬 Optimizing Projected Token Selection
Naive (batched)           | Time: 2.0366s | Speed: 98.20 tok/s
Route=512   Rerank=64    | Time: 0.7493s | Speed: 266.92 tok/s | Match: 60.50%
Route=512   Rerank=32    | Time: 0.4501s | Speed: 444.32 tok/s | Match: 60.50%
Route=512   Rerank=16    | Time: 0.4496s | Speed: 444.89 tok/s | Match: 60.50%
Route=1024  Rerank=64    | Time: 0.7759s | Speed: 257.78 tok/s | Match: 72.50%
Route=1024  Rerank=32    | Time: 0.7709s | Speed: 259.45 tok/s | Match: 72.50%
Route=1024  Rerank=16    | Time: 0.7712s | Speed: 259.34 tok/s | Match: 72.50%
Route=2048  Rerank=64    | Time: 2.0287s | Speed: 98.58 tok/s | Match: 83.00%
Route=2048  Rerank=32    | Time: 1.4058s | Speed: 142.27 tok/s | Match: 83.00%
Route=2048  Rerank=16    | Time: 1.4119s | Speed: 141.65 tok/s | Match: 83.00%
Route=4096  Rerank=64    | Time: 3.2837s | Speed: 60.91 tok/s | Match: 91.50%
Route=4096  Rerank=32    | Time: 2.6970s | Speed: 74.16 tok/s | Match: 91.50%
Route=4096  Rerank=16    | Time: 2.7169s | Speed: 73.61 tok/s | Match: 91.50%

✅ Best Config: Route=4096, Rerank=64, Speed=60.91 tok/s, Match=91.50%
"""