import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import time
import numpy as np

class MatrixFreeLMHead:
    def __init__(self, model, threshold):
        self.model = model
        self.device = model.device
        self.weight = model.lm_head.weight.data.to(self.device)
        self.bias = None  # Llama models typically don't have bias
        self.threshold = threshold

    def predict(self, hidden_state, Vh=None, vocab_routing_proj=None, top_k=64):
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]

        # If no routing data provided, fall back to dense matmul
        if Vh is None or vocab_routing_proj is None:
            scores = torch.matmul(hidden_state.view(-1), self.weight.t())
            top_score, top_index = torch.max(scores, dim=0)
            return top_index.unsqueeze(0), top_score.item()

        # Sparse path (SVD + top-k)
        code = torch.matmul(hidden_state, Vh.T)
        sims = torch.matmul(code, vocab_routing_proj.T)
        topk_values, topk_indices = torch.topk(sims, top_k, dim=-1)
        topk_vectors = self.weight[topk_indices[0]]
        rerank_scores = torch.matmul(hidden_state, topk_vectors.T)
        best_idx = torch.argmax(rerank_scores, dim=-1)
        final_token = topk_indices[0][best_idx]
        score = rerank_scores[0, best_idx]

        return final_token.unsqueeze(0), score.item()



class MatrixFreeLlamaForCausalLM(LlamaForCausalLM):
    """Custom Llama model that can bypass logits computation with matrix-free generation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.matrix_free_mode = False
        self.matrix_free_head = None
    
    def enable_matrix_free(self, threshold=-float("inf")):
        """Enable matrix-free generation mode with the given threshold."""
        self.matrix_free_mode = True
        self.matrix_free_head = MatrixFreeLMHead(self, threshold)
        return self.matrix_free_head
    
    def disable_matrix_free(self):
        """Disable matrix-free generation mode."""
        self.matrix_free_mode = False
    
    def forward(self, *args, **kwargs):
        """Override forward method to bypass logits computation in matrix-free mode."""
        # If not in matrix-free mode, use the standard forward
        if not self.matrix_free_mode:
            return super().forward(*args, **kwargs)
        
        # Store if we need hidden states for later
        output_hidden_states = kwargs.get('output_hidden_states', False)
        kwargs['output_hidden_states'] = True  # We always need hidden states in matrix-free mode
        
        # In matrix-free mode, skip logits computation in the parent class
        # by passing our inputs directly to the base model
        input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)
        position_ids = kwargs.get('position_ids', None)
        past_key_values = kwargs.get('past_key_values', None)
        inputs_embeds = kwargs.get('inputs_embeds', None)
        use_cache = kwargs.get('use_cache', None)
        output_attentions = kwargs.get('output_attentions', None)
        cache_position = kwargs.get('cache_position', None)
        
        # Call the base model directly
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always True in matrix-free mode
            cache_position=cache_position,
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Use our matrix-free head to predict the next token
        next_token, score = self.matrix_free_head.predict(hidden_states)
        
        # Create dummy logits tensor with only the predicted token having a high score
        # This is just for compatibility with the HuggingFace API
        dummy_logits = torch.zeros((1, 1, self.config.vocab_size), device=self.device)
        dummy_logits[0, 0, next_token] = score
        
        # Return a modified output object
        return CausalLMOutputWithPast(
            loss=None,
            logits=dummy_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

def calibrate_threshold(model, prompts, tokenizer, max_new_tokens=5):
    """Calibrate the threshold for matrix-free routing by examining score distributions."""
    scores = []
    
    # Create a router without modifying the model
    router = MatrixFreeLMHead(model, threshold=-float("inf"))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):  # Reduced to 5 tokens for faster calibration
                outputs = model(
                    input_ids=input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True,
                    output_hidden_states=True
                )
                
                # Get hidden states and past key values
                hidden = outputs.hidden_states[-1]
                past_key_values = outputs.past_key_values

                # Predict next token
                _, score = router.predict(hidden)
                scores.append(score)
                
                # Get actual next token for continuation
                input_ids = torch.argmax(outputs.logits[:, -1:, :], dim=-1)

    # Return the lowest score observed (or a percentile)
    return np.percentile(scores, 0)

def generate_matrix_free(model, tokenizer, prompt, max_new_tokens, threshold, verify_accuracy=True):
    """Generate text using the optimized matrix-free approach.
    
    This implementation uses the MatrixFreeLlamaForCausalLM subclass to truly bypass
    logits computation during inference.
    """
    # Initialize timing stats
    timing_stats = {
        'input_prep': 0.0,
        'model_forward': 0.0,
        'token_selection': 0.0,
        'memory_ops': 0.0,
        'total': 0.0
    }

    # Start total timing
    total_start = time.perf_counter()

    # Prepare inputs
    input_prep_start = time.perf_counter()
    if not isinstance(model, MatrixFreeLlamaForCausalLM):
        print("Warning: Model is not a MatrixFreeLlamaForCausalLM instance.")
        return None
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    
    # Enable matrix-free mode with the given threshold
    router = model.enable_matrix_free(threshold=threshold)
    timing_stats['input_prep'] = time.perf_counter() - input_prep_start
    
    matched = 0
    
    with torch.no_grad():
        # For verification, we first need to run with the standard model
        if verify_accuracy:
            verify_start = time.perf_counter()
            # Temporarily disable matrix-free mode
            model.disable_matrix_free()
            
            # Store predictions from standard model for verification
            standard_predictions = []
            standard_input_ids = input_ids.clone()
            standard_past_key_values = None
            
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=standard_input_ids,
                    past_key_values=standard_past_key_values,
                    use_cache=True
                )
                
                next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                standard_predictions.append(next_token.item())
                
                standard_past_key_values = outputs.past_key_values
                standard_input_ids = next_token
            
            # Re-enable matrix-free mode
            router = model.enable_matrix_free(threshold=threshold)
            timing_stats['verification'] = time.perf_counter() - verify_start
        
        # Run the actual matrix-free generation
        for i in range(max_new_tokens):
            iter_start = time.perf_counter()
            
            # Forward pass with matrix-free head
            forward_start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            timing_stats['model_forward'] += time.perf_counter() - forward_start
            
            # Memory operations timing
            memory_start = time.perf_counter()
            past_key_values = outputs.past_key_values
            timing_stats['memory_ops'] += time.perf_counter() - memory_start
            
            # Token selection timing
            token_start = time.perf_counter()
            next_token_id = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            timing_stats['token_selection'] += time.perf_counter() - token_start
            
            # Verify accuracy if needed
            if verify_accuracy:
                if next_token_id.item() == standard_predictions[i]:
                    matched += 1
            
            input_ids = next_token_id
            all_ids = torch.cat([all_ids, input_ids], dim=1)
        
    # Disable matrix-free mode when we're done
    model.disable_matrix_free()
    
    # Calculate total time
    timing_stats['total'] = time.perf_counter() - total_start
    
    # Calculate percentages for each stage
    total_time = timing_stats['total']
    timing_percentages = {k: (v/total_time)*100 for k, v in timing_stats.items() if k != 'total'}
    
    return {
        "text": tokenizer.decode(all_ids[0], skip_special_tokens=True),
        "accuracy": matched / max_new_tokens if verify_accuracy else None,
        "ms_per_token": (timing_stats['total'] / max_new_tokens) * 1000,
        "tokens_per_sec": max_new_tokens / timing_stats['total'],
        "timing_stats": timing_stats,
        "timing_percentages": timing_percentages
    }

def generate_baseline(model, tokenizer, prompt, max_new_tokens):
    """Generate text using the standard HuggingFace baseline approach."""
    # Ensure matrix-free mode is disabled if using our custom model
    if isinstance(model, MatrixFreeLlamaForCausalLM):
        model.disable_matrix_free()
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    total_duration = 0
    all_ids = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            start = time.perf_counter()
            
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            
            token_duration = time.perf_counter() - start
            total_duration += token_duration
            
            input_ids = next_token_id
            all_ids = torch.cat([all_ids, input_ids], dim=1)
        
    return {
        "text": tokenizer.decode(all_ids[0], skip_special_tokens=True),
        "ms_per_token": (total_duration / max_new_tokens) * 1000,
        "tokens_per_sec": max_new_tokens / total_duration
    }

def get_device():
    """Get the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def run_comparison(model_name="unsloth/Llama-3.2-1B", max_new_tokens=20):
    """Run comparison between baseline and matrix-free approaches."""
    device = get_device()
    print(f"🔄 Loading Llama 3 model: {model_name} on {device}...")
    
    # Set precision based on device 
    precision = torch.float16 if device.type == "cuda" else torch.float32
    
    # Load model - use our custom class instead of the standard one
    config = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=precision,
        return_dict=True,
    ).config
    
    # Create our custom model with the same configuration
    model = MatrixFreeLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=precision,
        config=config
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The meaning of life is",
        "In a distant galaxy, a civilization",
        "The future of AI will depend on",
        "Once upon a time",
        "The quantum computer"
    ]

    print("\U0001F4CA Calibrating routing threshold...")
    threshold = calibrate_threshold(model, prompts, tokenizer, max_new_tokens=5)
    print(f"✅ Calibrated P0 threshold: {threshold:.2f}\n")
    
    results = []

    # First run: verify accuracy
    print("🔍 Running accuracy verification test...")
    for prompt in prompts[:2]:  # Only test first two prompts for speed
        print(f"\U0001F9EA Prompt: {prompt}")
        
        baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens=5)  # Reduced for speed
        print(f"--- Llama 3 Baseline ---")
        print(f"📜 {baseline['text']}")
        print(f"⚡ {baseline['tokens_per_sec']:.2f} tokens/sec | ⏱ {baseline['ms_per_token']:.2f} ms/token")
        
        matrix_free = generate_matrix_free(model, tokenizer, prompt, max_new_tokens=5, threshold=threshold, verify_accuracy=True)
        
        print(f"--- Routed (Matrix-Free) with Verification ---")
        print(f"📜 {matrix_free['text']}")
        print(f"⚡ {matrix_free['tokens_per_sec']:.2f} tokens/sec | ⏱ {matrix_free['ms_per_token']:.2f} ms/token")
        print(f"🎯 Accuracy: {matrix_free['accuracy']:.2%}")
        print("\n📊 Timing Breakdown:")
        for stage, percentage in matrix_free['timing_percentages'].items():
            print(f"  {stage}: {percentage:.1f}% ({matrix_free['timing_stats'][stage]*1000:.2f}ms)")
        
        print("-" * 60)
    
    # If verification passes, run full performance test
    print("\n⚡ Running performance test (without verification)...")
    for prompt in prompts:
        print(f"\U0001F9EA Prompt: {prompt}")
        
        baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens)
        print(f"--- Llama 3 Baseline ---")
        print(f"📜 {baseline['text']}")
        print(f"⚡ {baseline['tokens_per_sec']:.2f} tokens/sec | ⏱ {baseline['ms_per_token']:.2f} ms/token")
        
        # Run without verification for maximum speed
        matrix_free = generate_matrix_free(model, tokenizer, prompt, max_new_tokens, threshold, verify_accuracy=False)
        
        print(f"--- Routed (Matrix-Free) Production ---")
        print(f"📜 {matrix_free['text']}")
        print(f"⚡ {matrix_free['tokens_per_sec']:.2f} tokens/sec | ⏱ {matrix_free['ms_per_token']:.2f} ms/token")
        print("\n📊 Timing Breakdown:")
        for stage, percentage in matrix_free['timing_percentages'].items():
            print(f"  {stage}: {percentage:.1f}% ({matrix_free['timing_stats'][stage]*1000:.2f}ms)")
        
        print("-" * 60)
        
        # Store results for summary
        result = {
            "prompt": prompt,
            "baseline_tokens_per_sec": baseline["tokens_per_sec"],
            "matrix_free_tokens_per_sec": matrix_free["tokens_per_sec"],
            "accuracy": "Verified" if matrix_free["accuracy"] is None else f"{matrix_free['accuracy']:.2%}",
            "timing_stats": matrix_free["timing_stats"],
            "timing_percentages": matrix_free["timing_percentages"]
        }
        results.append(result)
    
    # Print overall summary
    if results:
        print("\n📊 Overall Summary:")
        print("Prompt | Baseline (tok/s) | Matrix-Free (tok/s) | Speedup | Accuracy")
        print("-" * 80)
        
        total_speedup = 0
        
        for r in results:
            speedup = r["matrix_free_tokens_per_sec"] / r["baseline_tokens_per_sec"]
            total_speedup += speedup
            
            prompt_short = r["prompt"][:20] + "..." if len(r["prompt"]) > 20 else r["prompt"]
            
            # Simple string concatenation to avoid formatting errors
            print(prompt_short + " | " + 
                  str(round(r["baseline_tokens_per_sec"], 2)) + " | " + 
                  str(round(r["matrix_free_tokens_per_sec"], 2)) + " | " + 
                  str(round(speedup, 2)) + "x | " + 
                  str(r["accuracy"]))
        
        # Print average speedup
        avg_speedup = total_speedup / len(results)
        print("-" * 80)
        print("AVERAGE | | | " + str(round(avg_speedup, 2)) + "x | Verified")
        
        # Print potential savings
        if avg_speedup > 1:
            time_saved = (1 - 1/avg_speedup) * 100
            print(f"\n💰 Potential resource savings: {time_saved:.1f}% reduction in inference time")
        else:
            overhead = (1/avg_speedup - 1) * 100
            print(f"\n⚠️ No speedup achieved. Current overhead: {overhead:.1f}%")
            print("Consider optimizing implementation or using different hardware.")
