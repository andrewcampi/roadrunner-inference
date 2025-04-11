import torch
import time
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

# === Speculative Beam Search-lite ===
def speculative_beam_lite(hidden_batch, weight, proj_weight, proj_mat, beam_width, threshold):
    selected_tokens = []
    accepted_flags = []
    for i in range(0, hidden_batch.shape[0], chunk_size):
        chunk = hidden_batch[i:i+chunk_size]  # [B, D]
        proj_hidden = torch.matmul(chunk, proj_mat)  # [B, P]
        sims = torch.matmul(proj_hidden, proj_weight.T)  # [B, V]
        topk_vals, topk_idxs = torch.topk(sims, beam_width, dim=-1)  # [B, beam_width]

        token_vectors = vocab_weight[topk_idxs]  # [B, beam_width, D]
        dot_scores = torch.einsum("bd,bkd->bk", chunk, token_vectors)  # [B, beam_width]

        accepted_mask = dot_scores >= threshold  # [B, beam_width]
        dot_scores_masked = dot_scores.masked_fill(~accepted_mask, float('-inf'))
        best_indices = torch.argmax(dot_scores_masked, dim=-1)  # [B]

        best_tokens = topk_idxs[torch.arange(chunk.size(0)), best_indices]  # [B]
        was_accepted = accepted_mask[torch.arange(chunk.size(0)), best_indices]

        selected_tokens.extend(best_tokens.tolist())
        accepted_flags.extend(was_accepted.tolist())

    return selected_tokens, accepted_flags

# === Benchmark ===
def benchmark(fn, *args):
    start = time.time()
    result = fn(*args)
    return result, time.time() - start

# === Run Baseline ===
print("\n🔬 Speculative Beam Search-lite")
tok_naive, naive_scores = rerank_naive_batched(hidden_states, vocab_weight, 64)
_, baseline_time = benchmark(lambda: rerank_naive_batched(hidden_states, vocab_weight, 64))
print(f"Naive (batched)           | Time: {baseline_time:.4f}s | Speed: {repeats / baseline_time:.2f} tok/s")

# === Beam Sweep (extended) ===
percentiles = list(range(0, 40, 5))  # P0 to P35
beam_widths = [2, 4, 8, 16, 32, 64]  # Extended beam sizes

best_accuracy = 0
best_config = None

for percentile in percentiles:
    threshold = np.percentile(naive_scores, percentile)
    for beam_width in beam_widths:
        (tok_spec, accepted_flags), spec_time = benchmark(
            lambda: speculative_beam_lite(
                hidden_states, vocab_weight, projected_vocab, projection_matrix,
                beam_width, threshold
            )
        )
        valid_indices = [i for i, flag in enumerate(accepted_flags) if flag]
        accepted = [tok_spec[i] for i in valid_indices]
        correct = sum(1 for i in valid_indices if tok_spec[i] == tok_naive[i])
        accepted_rate = len(accepted) / repeats
        accuracy = correct / len(accepted) if accepted else 0
        speed = repeats / spec_time
        print(f"Beam={beam_width:<2} P{percentile:<2} | Speed: {speed:6.2f} tok/s | Accepted: {accepted_rate*100:5.1f}% | Match: {accuracy*100:5.1f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = (beam_width, percentile, speed, accepted_rate, accuracy)

if best_config:
    print(f"\n✅ Best Config => Beam={best_config[0]}, P{best_config[1]} | Speed: {best_config[2]:.2f} tok/s | Accepted: {best_config[3]*100:.2f}% | Match: {best_config[4]*100:.2f}%")
else:
    print("\n❌ No configuration reached improved accuracy.")


""" Output:
🔬 Speculative Beam Search-lite
Naive (batched)           | Time: 1.8710s | Speed: 106.89 tok/s
Beam=2  P0  | Speed: 1129.42 tok/s | Accepted:  14.0% | Match:  32.1%
Beam=4  P0  | Speed: 1381.02 tok/s | Accepted:  18.0% | Match:  30.6%
Beam=8  P0  | Speed: 1382.38 tok/s | Accepted:  29.0% | Match:  36.2%
Beam=16 P0  | Speed: 1354.71 tok/s | Accepted:  41.5% | Match:  30.1%
Beam=32 P0  | Speed: 1277.78 tok/s | Accepted:  55.5% | Match:  35.1%
Beam=64 P0  | Speed: 1140.05 tok/s | Accepted:  71.5% | Match:  37.8%
Beam=2  P5  | Speed: 1544.84 tok/s | Accepted:   9.5% | Match:  47.4%
Beam=4  P5  | Speed: 1532.55 tok/s | Accepted:  12.5% | Match:  44.0%
Beam=8  P5  | Speed: 1505.20 tok/s | Accepted:  22.5% | Match:  46.7%
Beam=16 P5  | Speed: 1459.90 tok/s | Accepted:  29.5% | Match:  40.7%
Beam=32 P5  | Speed: 1375.60 tok/s | Accepted:  39.5% | Match:  48.1%
Beam=64 P5  | Speed: 1213.53 tok/s | Accepted:  52.5% | Match:  50.5%
Beam=2  P10 | Speed: 1538.59 tok/s | Accepted:   8.5% | Match:  52.9%
Beam=4  P10 | Speed: 1535.31 tok/s | Accepted:  11.5% | Match:  47.8%
Beam=8  P10 | Speed: 1481.13 tok/s | Accepted:  19.5% | Match:  51.3%
Beam=16 P10 | Speed: 1457.50 tok/s | Accepted:  25.5% | Match:  45.1%
Beam=32 P10 | Speed: 1364.70 tok/s | Accepted:  34.5% | Match:  50.7%
Beam=64 P10 | Speed: 1215.86 tok/s | Accepted:  43.0% | Match:  58.1%
Beam=2  P15 | Speed: 1549.68 tok/s | Accepted:   7.5% | Match:  60.0%
Beam=4  P15 | Speed: 1536.68 tok/s | Accepted:  10.5% | Match:  52.4%
Beam=8  P15 | Speed: 1496.85 tok/s | Accepted:  17.5% | Match:  57.1%
Beam=16 P15 | Speed: 1458.63 tok/s | Accepted:  23.0% | Match:  50.0%
Beam=32 P15 | Speed: 1375.11 tok/s | Accepted:  30.5% | Match:  55.7%
Beam=64 P15 | Speed: 1211.58 tok/s | Accepted:  38.0% | Match:  63.2%
Beam=2  P20 | Speed: 1545.09 tok/s | Accepted:   7.5% | Match:  60.0%
Beam=4  P20 | Speed: 1536.59 tok/s | Accepted:  10.5% | Match:  52.4%
Beam=8  P20 | Speed: 1508.57 tok/s | Accepted:  16.0% | Match:  56.2%
Beam=16 P20 | Speed: 1461.98 tok/s | Accepted:  21.5% | Match:  48.8%
Beam=32 P20 | Speed: 1375.78 tok/s | Accepted:  28.5% | Match:  56.1%
Beam=64 P20 | Speed: 1218.38 tok/s | Accepted:  35.0% | Match:  64.3%
Beam=2  P25 | Speed: 1549.63 tok/s | Accepted:   6.5% | Match:  69.2%
Beam=4  P25 | Speed: 1538.64 tok/s | Accepted:   8.5% | Match:  64.7%
Beam=8  P25 | Speed: 1507.05 tok/s | Accepted:  13.5% | Match:  66.7%
Beam=16 P25 | Speed: 1459.67 tok/s | Accepted:  18.0% | Match:  58.3%
Beam=32 P25 | Speed: 1373.01 tok/s | Accepted:  24.5% | Match:  65.3%
Beam=64 P25 | Speed: 1217.28 tok/s | Accepted:  31.5% | Match:  71.4%
Beam=2  P30 | Speed: 1548.12 tok/s | Accepted:   5.5% | Match:  72.7%
Beam=4  P30 | Speed: 1533.79 tok/s | Accepted:   7.5% | Match:  66.7%
Beam=8  P30 | Speed: 1510.02 tok/s | Accepted:  12.0% | Match:  70.8%
Beam=16 P30 | Speed: 1462.45 tok/s | Accepted:  16.0% | Match:  62.5%
Beam=32 P30 | Speed: 1374.96 tok/s | Accepted:  20.5% | Match:  73.2%
Beam=64 P30 | Speed: 1213.10 tok/s | Accepted:  26.5% | Match:  79.2%
Beam=2  P35 | Speed: 1538.33 tok/s | Accepted:   4.0% | Match:  87.5%
Beam=4  P35 | Speed: 1536.32 tok/s | Accepted:   6.0% | Match:  75.0%
Beam=8  P35 | Speed: 1507.00 tok/s | Accepted:  10.5% | Match:  76.2%
Beam=16 P35 | Speed: 1460.45 tok/s | Accepted:  14.5% | Match:  65.5%
Beam=32 P35 | Speed: 1377.02 tok/s | Accepted:  19.0% | Match:  76.3%
Beam=64 P35 | Speed: 1220.37 tok/s | Accepted:  25.0% | Match:  82.0%

✅ Best Config => Beam=2, P35 | Speed: 1538.33 tok/s | Accepted: 4.00% | Match: 87.50%
"""