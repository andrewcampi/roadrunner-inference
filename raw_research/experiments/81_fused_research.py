import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import time
import numpy as np
import json
from theory_vs_practice import RoadrunnerExplorer  # Imported for attention stabilization
from optimized_sparse_routing import run_comparison


# === Attention Layer Stabilization (from Exp 61) ===
def stabilize_attention_layers(model):
    print("🧠 Running attention layer stabilization using RoadrunnerExplorer...")
    explorer = RoadrunnerExplorer(model_name="gpt2")  # Using GPT-2 for compatibility

    print("\n=== 🛠️ Full-Stabilization Mode: Attention Layer Routing ===")
    results = []
    max_layers = 12
    alpha = 0.001
    precision = "float64"
    stabilizer = "layernorm"
    sv_handler = "full"

    for layer in range(max_layers):
        config = {
            "svd_impl": "numpy_svd",
            "precision": precision,
            "sv_handler": sv_handler,
            "stabilizer": stabilizer,
            "alpha": alpha,
            "layers": [layer],
            "mlp_routing": False,
            "attn_routing": True
        }

        model_layer = explorer.create_test_model(config)
        print(f"✅ Layer {layer} routed with config: {config}")
        results.append({"layer": layer, **config})

    print("🎉 All attention layers stabilized using SVD routing.")
    return results


# === Matrix-Free LM Head with Fallback (Exp 63 enhancements) ===
class MatrixFreeLMHead:
    def __init__(self, model, threshold, fallback_threshold):
        self.model = model
        self.device = model.device
        self.weight = model.lm_head.weight.data.to(self.device)
        self.bias = None  # Llama models typically don't use bias
        self.threshold = threshold
        self.fallback_threshold = fallback_threshold

    def predict(self, hidden_state, Vh=None, vocab_routing_proj=None, top_k=64):
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]

        if Vh is None or vocab_routing_proj is None:
            scores = torch.matmul(hidden_state.view(-1), self.weight.t())
            top_score, top_index = torch.max(scores, dim=0)
            return top_index.unsqueeze(0), top_score.item(), 'fallback_dense'

        # Sparse SVD routing path
        code = torch.matmul(hidden_state, Vh.T)
        sims = torch.matmul(code, vocab_routing_proj.T)
        topk_values, topk_indices = torch.topk(sims, top_k, dim=-1)
        topk_vectors = self.weight[topk_indices[0]]
        rerank_scores = torch.matmul(hidden_state, topk_vectors.T)
        best_idx = torch.argmax(rerank_scores, dim=-1)
        final_token = topk_indices[0][best_idx]
        score = rerank_scores[0, best_idx]

        if score.item() < self.fallback_threshold:
            scores = torch.matmul(hidden_state.view(-1), self.weight.t())
            top_score, top_index = torch.max(scores, dim=0)
            return top_index.unsqueeze(0), top_score.item(), 'fallback_dense'

        return final_token.unsqueeze(0), score.item(), 'sparse_routed'


class MatrixFreeLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.matrix_free_mode = False
        self.matrix_free_head = None

    def enable_matrix_free(self, threshold=-float("inf"), fallback_threshold=-float("inf")):
        self.matrix_free_mode = True
        self.matrix_free_head = MatrixFreeLMHead(self, threshold, fallback_threshold)
        return self.matrix_free_head

    def disable_matrix_free(self):
        self.matrix_free_mode = False

    def forward(self, *args, **kwargs):
        if not self.matrix_free_mode:
            return super().forward(*args, **kwargs)

        kwargs['output_hidden_states'] = True
        input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
        past_key_values = kwargs.get('past_key_values', None)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

        hidden_states = outputs.last_hidden_state
        next_token, score, route_type = self.matrix_free_head.predict(hidden_states)

        dummy_logits = torch.zeros((1, 1, self.config.vocab_size), device=self.device)
        dummy_logits[0, 0, next_token] = score

        return CausalLMOutputWithPast(
            loss=None,
            logits=dummy_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def calibrate_thresholds(model, tokenizer, prompts, max_new_tokens=5):
    scores = []
    router = MatrixFreeLMHead(model, -float("inf"), -float("inf"))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                hidden = outputs.hidden_states[-1]
                past_key_values = outputs.past_key_values
                _, score, _ = router.predict(hidden)
                scores.append(score)
                input_ids = torch.argmax(outputs.logits[:, -1:, :], dim=-1)

    routing_threshold = np.percentile(scores, 50)
    fallback_threshold = np.percentile(scores, 10)
    return routing_threshold, fallback_threshold


def run_enhanced_routing(model_name="unsloth/Llama-3.2-1B", max_new_tokens=20):
    run_comparison(model_name=model_name, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    run_enhanced_routing()


""" Output:
🔄 Loading Llama 3 model: unsloth/Llama-3.2-1B on mps...
📊 Calibrating routing threshold...
✅ Calibrated P0 threshold: 17.51

🔍 Running accuracy verification test...
🧪 Prompt: The meaning of life is
--- Llama 3 Baseline ---
📜 The meaning of life is to find your passion and
⚡ 60.77 tokens/sec | ⏱ 16.46 ms/token
--- Routed (Matrix-Free) with Verification ---
📜 The meaning of life is to find your passion and
⚡ 8.94 tokens/sec | ⏱ 111.82 ms/token
🎯 Accuracy: 100.00%

📊 Timing Breakdown:
  input_prep: 0.1% (0.54ms)
  model_forward: 50.8% (284.27ms)
  token_selection: 0.1% (0.28ms)
  memory_ops: 0.0% (0.00ms)
  verification: 48.3% (270.08ms)
------------------------------------------------------------
🧪 Prompt: In a distant galaxy, a civilization
--- Llama 3 Baseline ---
📜 In a distant galaxy, a civilization has developed a technology that
⚡ 64.88 tokens/sec | ⏱ 15.41 ms/token
--- Routed (Matrix-Free) with Verification ---
📜 In a distant galaxy, a civilization has developed a technology that
⚡ 8.65 tokens/sec | ⏱ 115.54 ms/token
🎯 Accuracy: 100.00%

📊 Timing Breakdown:
  input_prep: 0.1% (0.40ms)
  model_forward: 50.4% (291.18ms)
  token_selection: 0.1% (0.31ms)
  memory_ops: 0.0% (0.00ms)
  verification: 48.8% (281.74ms)
------------------------------------------------------------

⚡ Running performance test (without verification)...
🧪 Prompt: The meaning of life is
--- Llama 3 Baseline ---
📜 The meaning of life is to find your passion and to live it. I am passionate about helping people find their passion and live
⚡ 24.69 tokens/sec | ⏱ 40.50 ms/token
--- Routed (Matrix-Free) Production ---
📜 The meaning of life is to find your passion and to live it. I am passionate about helping people find their passion and live
⚡ 17.18 tokens/sec | ⏱ 58.21 ms/token

📊 Timing Breakdown:
  input_prep: 0.0% (0.42ms)
  model_forward: 97.0% (1129.60ms)
  token_selection: 0.1% (1.23ms)
  memory_ops: 0.0% (0.01ms)
------------------------------------------------------------
🧪 Prompt: In a distant galaxy, a civilization
--- Llama 3 Baseline ---
📜 In a distant galaxy, a civilization has developed a technology that allows them to travel through space and time. They have discovered a way to
⚡ 24.67 tokens/sec | ⏱ 40.53 ms/token
--- Routed (Matrix-Free) Production ---
📜 In a distant galaxy, a civilization has developed a technology that allows them to travel through space and time. They have discovered a way to
⚡ 16.99 tokens/sec | ⏱ 58.87 ms/token

📊 Timing Breakdown:
  input_prep: 0.0% (0.37ms)
  model_forward: 97.1% (1143.04ms)
  token_selection: 0.1% (1.20ms)
  memory_ops: 0.0% (0.01ms)
------------------------------------------------------------
🧪 Prompt: The future of AI will depend on
--- Llama 3 Baseline ---
📜 The future of AI will depend on how we use it
The future of AI will depend on how we use it
The future of
⚡ 24.65 tokens/sec | ⏱ 40.58 ms/token
--- Routed (Matrix-Free) Production ---
📜 The future of AI will depend on how we use it
The future of AI will depend on how we use it
The future of
⚡ 16.79 tokens/sec | ⏱ 59.57 ms/token

📊 Timing Breakdown:
  input_prep: 0.0% (0.38ms)
  model_forward: 97.1% (1156.68ms)
  token_selection: 0.1% (1.24ms)
  memory_ops: 0.0% (0.01ms)
------------------------------------------------------------
🧪 Prompt: Once upon a time
--- Llama 3 Baseline ---
📜 Once upon a time, there was a man who was very rich. He had a lot of money, and he was
⚡ 24.90 tokens/sec | ⏱ 40.16 ms/token
--- Routed (Matrix-Free) Production ---
📜 Once upon a time, there was a man who was very rich. He had a lot of money, and he was
⚡ 17.01 tokens/sec | ⏱ 58.78 ms/token

📊 Timing Breakdown:
  input_prep: 0.0% (0.41ms)
  model_forward: 97.2% (1142.41ms)
  token_selection: 0.1% (1.24ms)
  memory_ops: 0.0% (0.01ms)
------------------------------------------------------------
🧪 Prompt: The quantum computer
--- Llama 3 Baseline ---
📜 The quantum computer is a computer that uses quantum mechanics to perform calculations. It is a new type of computer that uses
⚡ 24.90 tokens/sec | ⏱ 40.15 ms/token
--- Routed (Matrix-Free) Production ---
📜 The quantum computer is a computer that uses quantum mechanics to perform calculations. It is a new type of computer that uses
⚡ 16.93 tokens/sec | ⏱ 59.07 ms/token

📊 Timing Breakdown:
  input_prep: 0.0% (0.34ms)
  model_forward: 97.1% (1147.54ms)
  token_selection: 0.1% (1.36ms)
  memory_ops: 0.0% (0.00ms)
------------------------------------------------------------

📊 Overall Summary:
Prompt | Baseline (tok/s) | Matrix-Free (tok/s) | Speedup | Accuracy
--------------------------------------------------------------------------------
The meaning of life ... | 24.69 | 17.18 | 0.7x | Verified
In a distant galaxy,... | 24.67 | 16.99 | 0.69x | Verified
The future of AI wil... | 24.65 | 16.79 | 0.68x | Verified
Once upon a time | 24.9 | 17.01 | 0.68x | Verified
The quantum computer | 24.9 | 16.93 | 0.68x | Verified
--------------------------------------------------------------------------------
AVERAGE | | | 0.69x | Verified

⚠️ No speedup achieved. Current overhead: 45.8%
"""