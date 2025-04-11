import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import os

SVD_DIR = "svd"  # Must exist

# === Load SVD data ===
def load_svd(layer):
    path = os.path.join(SVD_DIR, f"layer_{layer}.pt")
    data = torch.load(path, map_location="cpu")
    return data["mlp"], data["attention"]


def layerwise_alpha_mlp(layer_idx):
    # Based on experiment #53 & #52: Safe blend values per layer
    alpha_map = {
        0: 0.10, 1: 0.08, 2: 0.06, 3: 0.05,
        4: 0.04, 5: 0.03, 6: 0.02, 7: 0.01,
        8: 0.01, 9: 0.01, 10: 0.01, 11: 0.01
    }
    return alpha_map.get(layer_idx, 0.01)  # Default to safe low α


def layerwise_alpha_attn(layer_idx):
    # Similar strategy for attention, slightly more tolerant
    alpha_map = {
        0: 0.10, 1: 0.10, 2: 0.08, 3: 0.05,
        4: 0.04, 5: 0.03, 6: 0.02, 7: 0.01,
        8: 0.01, 9: 0.01, 10: 0.005, 11: 0.005
    }
    return alpha_map.get(layer_idx, 0.005)


# === MLP ===
class RoadrunnerMLP(torch.nn.Module):
    def __init__(self, original, svd, alpha=0.8):  # Increased alpha for MLP
        super().__init__()
        self.alpha = alpha
        self.original = original
        self.U_fc, self.S_fc, self.Vh_fc, self.b_fc = svd["fc"]
        self.U_proj, self.S_proj, self.Vh_proj, self.b_proj = svd["proj"]

    def forward(self, x):
        # Route through SVD path
        routed = torch.matmul(x, self.Vh_fc) * self.S_fc
        hidden = F.gelu(torch.matmul(routed, self.U_fc.T) + self.b_fc)
        out = torch.matmul(hidden, self.Vh_proj.T) * self.S_proj
        out = torch.matmul(out, self.U_proj.T) + self.b_proj

        # Get fallback output
        fallback_out = self.original(x)

        # Combine with higher weight on routed path
        return self.alpha * out + (1 - self.alpha) * fallback_out

# === Attention ===
class RoadrunnerAttention(torch.nn.Module):
    def __init__(self, original, svd, alpha=0.6):  # Reduced alpha for attention
        super().__init__()
        self.alpha = alpha
        self.original = original
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.hidden_dim = self.num_heads * self.head_dim

        self.U_q, self.S_q, self.Vh_q, self.b_q = svd["q"]
        self.U_k, self.S_k, self.Vh_k, self.b_k = svd["k"]
        self.U_v, self.S_v, self.Vh_v, self.b_v = svd["v"]
        self.U_o, self.S_o, self.Vh_o, self.b_o = svd["o"]

    def _reshape(self, x):
        B, T, H = x.size()
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine(self, x):
        B, nH, T, dH = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nH * dH)

    def forward(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False
    ):
        # Project input to Q, K, V
        Q = torch.matmul(x, self.Vh_q) * self.S_q
        Q = torch.matmul(Q, self.U_q.T) + self.b_q
        K = torch.matmul(x, self.Vh_k) * self.S_k
        K = torch.matmul(K, self.U_k.T) + self.b_k
        V = torch.matmul(x, self.Vh_v) * self.S_v
        V = torch.matmul(V, self.U_v.T) + self.b_v

        # Reshape for attention
        Q = self._reshape(Q)  # [batch, num_heads, seq_len, head_dim]
        K = self._reshape(K)
        V = self._reshape(V)

        # Handle past key/values
        if layer_past is not None:
            past_key, past_value = layer_past
            # Ensure past keys/values are on the same device
            past_key = past_key.to(K.device)
            past_value = past_value.to(V.device)
            K = torch.cat([past_key, K], dim=2)
            V = torch.cat([past_value, V], dim=2)

        # Store current K,V for next iteration
        present = (K, V) if use_cache else None

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Compute attention weights and context
        weights = torch.softmax(scores, dim=-1)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, V)
        
        # Reshape and project output
        context = self._combine(context)
        out = torch.matmul(context, self.Vh_o) * self.S_o
        out = torch.matmul(out, self.U_o.T) + self.b_o

        # Get fallback output
        fallback_out = self.original(
            x,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )[0]  # Only take the output, not the past

        # Combine routed and fallback paths
        combined_out = self.alpha * out + (1 - self.alpha) * fallback_out

        if output_attentions:
            return combined_out, present, weights
        return combined_out, present


# === LM Head with Top-k routing + fallback ===
class RoadrunnerLMHead(torch.nn.Module):
    def __init__(self, model, k=5, min_gap_threshold=1.0, rerank=True, temperature=0.8, 
             rerank_full_every=10, nucleus_p=0.9):
        super().__init__() 
        self.weight = model.lm_head.weight.data
        self.bias = model.lm_head.bias.data if model.lm_head.bias is not None else None
        self.k = k
        self.min_gap_threshold = min_gap_threshold
        self.rerank = rerank
        self.temperature = temperature
        self.rerank_full_every = rerank_full_every
        self.nucleus_p = nucleus_p
        self.token_count = 0
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.history = TokenHistory(self.tokenizer)

    def nucleus_sample(self, logits):
        """Nucleus (top-p) sampling over logits"""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumsum_probs <= self.nucleus_p
        nucleus_mask[0] = True  # Always include top token
        
        # Filter logits and renormalize
        filtered_logits = logits.clone()
        filtered_logits[~nucleus_mask] = float('-inf')
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        
        # Sample from filtered distribution
        sampled = torch.multinomial(filtered_probs, 1)
        return sampled, 'nucleus'

    def forward(self, hidden_state):
        return self.predict(hidden_state)[0]  # just return token tensor

    def predict(self, hidden_state):
        # Get logits and apply temperature scaling
        logits = hidden_state @ self.weight.T  # [1, 768] x [768, vocab]
        # Keep batch dimension, don't squeeze
        if self.bias is not None:
            logits = logits + self.bias
        
        # Apply temperature scaling
        logits = logits / self.temperature

        # Apply history-based penalties
        logits = logits + self.history.get_token_penalty(logits)

        # Get top-k tokens and scores for gap analysis
        logits = logits.view(-1)  # Flatten to [vocab_size]
        topk_scores, topk_indices = torch.topk(logits, self.k)
        # Get first and last scores as individual scalars
        highest_score = topk_scores[0].item()  # Convert to scalar immediately
        lowest_score = topk_scores[-1].item()  # Convert to scalar immediately
        gap = highest_score - lowest_score  # Now working with Python floats

        # Debug information
        print("\n=== Token Selection Debug ===")
        print(f"Top-{self.k} scores: {topk_scores.tolist()}")
        print(f"Score gap: {gap:.2f}")
        
        tokens = []
        for idx in topk_indices:
            token_text = self.tokenizer.decode([idx.item()])
            tokens.append(token_text)
        print(f"Top-{self.k} tokens: {tokens}")

        # Print diversity stats
        stats = self.history.get_diversity_stats()
        print(f"Diversity Stats:")
        print(f"  Unique tokens: {stats['unique_tokens']}/{stats['total_tokens']}")
        print(f"  Diversity score: {stats['diversity']:.3f}")
        print(f"  'the' frequency: {stats['the_frequency']:.3f}")

        self.token_count += 1
        do_full_rerank = (self.token_count % self.rerank_full_every) == 0

        # Sampling strategy
        if gap >= self.min_gap_threshold:
            if do_full_rerank or self.token_count % 2 == 1:
                # Use nucleus sampling
                selected_token, mode = self.nucleus_sample(logits)
            else:
                # Use top token
                selected_token = topk_indices[0].unsqueeze(0)
                mode = 'top1'
                
            # Check for repetition and 'the' threshold
            token_id = selected_token.item()
            if self.history.is_ngram_repeat(token_id) or self.history.would_exceed_the_threshold(token_id):
                # Try nucleus sampling instead
                selected_token, mode = self.nucleus_sample(logits)
                
            print(f"Selected token: '{self.tokenizer.decode([selected_token.item()])}'")
            self.history.update(selected_token.item())
            return selected_token, mode

        # Fallback routing
        fallback_idx = torch.argmax(logits).unsqueeze(0)
        print(f"Selected token (fallback): '{self.tokenizer.decode([fallback_idx.item()])}'")
        print("=========================")
        self.history.update(fallback_idx.item())
        return fallback_idx, 'fallback'


# === Token History and Diversity Tracking ===
class TokenHistory:
    def __init__(self, tokenizer, ngram_size=3, the_threshold=4):
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        self.the_threshold = the_threshold
        self.reset()
    
    def reset(self):
        self.token_history = []
        self.the_count = 0
        self.unique_tokens = set()
        
    def update(self, token_id):
        token_text = self.tokenizer.decode([token_id])
        self.token_history.append(token_id)
        self.unique_tokens.add(token_text.strip())
        if token_text.strip() == "the":
            self.the_count += 1
            
    def get_diversity_stats(self):
        total_tokens = len(self.token_history)
        unique_count = len(self.unique_tokens)
        diversity = unique_count / max(1, total_tokens)
        return {
            "unique_tokens": unique_count,
            "total_tokens": total_tokens,
            "diversity": diversity,
            "the_frequency": self.the_count / max(1, total_tokens)
        }
        
    def is_ngram_repeat(self, candidate_id):
        """Check if adding this token would create an n-gram repeat"""
        if len(self.token_history) < self.ngram_size:
            return False
            
        candidate_history = self.token_history + [candidate_id]
        for n in range(2, self.ngram_size + 1):
            # Get the potential new n-gram
            new_ngram = candidate_history[-n:]
            # Check against all n-grams in history
            for i in range(len(self.token_history) - n + 1):
                if self.token_history[i:i+n] == new_ngram:
                    return True
        return False
        
    def would_exceed_the_threshold(self, candidate_id):
        """Check if adding this token would exceed 'the' threshold"""
        if self.tokenizer.decode([candidate_id]).strip() == "the":
            return (self.the_count + 1) > self.the_threshold
        return False
        
    def get_token_penalty(self, logits):
        """Apply penalties to logits based on history"""
        penalty = torch.zeros_like(logits)
        
        # Penalize recent tokens
        recent_tokens = self.token_history[-5:]  # Last 5 tokens
        for token_id in recent_tokens:
            token_id = int(token_id)  # Convert to int to avoid tensor indexing issues
            if 0 <= token_id < penalty.size(-1):  # Check bounds
                penalty[..., token_id] -= 2.0  # Use ... to handle any number of leading dimensions
            
        # Extra penalty for 'the' if over threshold
        if self.the_count >= self.the_threshold:
            the_token_id = int(self.tokenizer.encode(" the")[0])  # Convert to int
            if 0 <= the_token_id < penalty.size(-1):  # Check bounds
                penalty[..., the_token_id] -= 5.0
            
        return penalty


# === Engine ===
class RoadrunnerEngine:
    def __init__(self, model_name="gpt2", mlp_alpha=0.8, attn_alpha=0.6, k=5, 
                 temperature=1.0, nucleus_p=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.mlp_alpha = mlp_alpha
        self.attn_alpha = attn_alpha
        self.k = k
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self._inject_svd()
        self.lm_head = RoadrunnerLMHead(
            self.model, 
            k=k,
            min_gap_threshold=1.0,
            rerank=True,
            temperature=temperature,
            nucleus_p=nucleus_p,
            rerank_full_every=10
        )

    def _inject_svd(self):
        for i, block in enumerate(self.model.transformer.h):
            mlp_svd, attn_svd = load_svd(i)
            block.mlp = RoadrunnerMLP(block.mlp, mlp_svd, self.mlp_alpha)
            block.attn = RoadrunnerAttention(block.attn, attn_svd, self.attn_alpha)
        print("✅ SVD routing injected into all layers.")

    def generate(self, prompt, max_new_tokens=30):
        # Reset token history for new generation
        self.lm_head.history.reset()
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = input_ids.clone()
        past = None
        start = time.time()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = output_ids[:, -1:] if output_ids.size(1) > 1 else output_ids

                if past is None:
                    output = self.model.transformer(x, use_cache=True)
                else:
                    output = self.model.transformer(x, past_key_values=past, use_cache=True)

                hidden_state = output.last_hidden_state[:, -1:, :]
                past = output.past_key_values

                next_token, mode = self.lm_head.predict(hidden_state)
                output_ids = torch.cat([output_ids, next_token.view(1, 1)], dim=1)

        duration = time.time() - start
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print("\n=== 🧠 Roadrunner Output ===")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Time: {duration:.2f} sec | Speed: {max_new_tokens / duration:.2f} tokens/sec")
        
        # Print final diversity stats
        stats = self.lm_head.history.get_diversity_stats()
        print("\n=== Final Diversity Stats ===")
        print(f"Total unique tokens: {stats['unique_tokens']}")
        print(f"Total tokens generated: {stats['total_tokens']}")
        print(f"Diversity score: {stats['diversity']:.3f}")
        print(f"'the' frequency: {stats['the_frequency']:.3f}")
        print("=============================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoadrunnerGPT2Model(torch.nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._inject_routing()

    def _inject_routing(self):
        # Replace all blocks with routed attention + MLP
        for i, block in enumerate(self.model.transformer.h):
            mlp_svd, attn_svd = load_svd(i)
            block.mlp = RoadrunnerMLP(block.mlp, mlp_svd, alpha=layerwise_alpha_mlp(i))
            block.attn = RoadrunnerAttention(block.attn, attn_svd, alpha=layerwise_alpha_attn(i))
        self.model.lm_head = RoadrunnerLMHead(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def lm_head(self):
        return self.model.lm_head


def generate_tokens(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = input_ids.clone()
    past = None

    start = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = output_ids[:, -1:] if output_ids.size(1) > 1 else output_ids

            if past is None:
                output = model.transformer(x, use_cache=True)
            else:
                output = model.transformer(x, past_key_values=past, use_cache=True)

            hidden_state = output.last_hidden_state[:, -1:, :]
            past = output.past_key_values

            logits = model.lm_head(hidden_state)
            next_token = torch.argmax(logits, dim=-1)
            output_ids = torch.cat([output_ids, next_token.view(1, 1)], dim=1)


    duration = time.time() - start
    return output_ids[0], duration


def compare_outputs(prompt, max_new_tokens=30):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load baseline GPT-2
    baseline = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print("🧠 Loaded baseline GPT-2")

    # Load Roadrunner model (uses SVD-routed modules)
    roadrunner = RoadrunnerGPT2Model("gpt2").to(device)
    print("🚀 Loaded Roadrunner Inference Engine")

    # Run inference
    print("\n== Running Baseline ==")
    baseline_output, baseline_time = generate_tokens(baseline, tokenizer, prompt, max_new_tokens)
    print("Baseline generation done.")

    print("\n== Running Roadrunner ==")
    roadrunner_output, roadrunner_time = generate_tokens(roadrunner, tokenizer, prompt, max_new_tokens)
    print("Roadrunner generation done.")

    # Decode outputs
    baseline_text = tokenizer.decode(baseline_output, skip_special_tokens=True)
    roadrunner_text = tokenizer.decode(roadrunner_output, skip_special_tokens=True)

    # Compare tokens
    match_count = (baseline_output == roadrunner_output).sum().item()
    total = baseline_output.size(0)
    match_percent = 100 * match_count / total

    # Results
    print("\n=== 📊 Comparison Results ===")
    print(f"Token match: {match_percent:.2f}% ({match_count}/{total})")
    print(f"⏱️ Baseline time: {baseline_time:.4f} sec")
    print(f"⏱️ Roadrunner time: {roadrunner_time:.4f} sec")
    if roadrunner_time > 0:
        print(f"⚡ Speedup: {baseline_time / roadrunner_time:.2f}×")

    print("\n=== 🧠 Baseline Output ===")
    print(baseline_text)

    print("\n=== 🚀 Roadrunner Output ===")
    print(roadrunner_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()

    compare_outputs(prompt=args.prompt, max_new_tokens=args.tokens)
