import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity

# === Configuration ===
MODEL_NAME = "unsloth/Llama-3.2-1B"
BEAM_WIDTH = 16
THRESHOLD_PERCENTILE = 30
PROJ_DIM = 1024
MAX_NEW_TOKENS = 20
NUM_TEST_PROMPTS = 5
USE_FP16 = True  # Set to False to use FP32 for entire model
DEFAULT_THRESHOLD = 0.5  # Fallback threshold value if calibration fails

# Set up device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")

class RoadRunnerDecoder:
    def __init__(self, model, tokenizer, proj_dim=1024, beam_width=16, threshold_percentile=30):
        self.model = model
        self.tokenizer = tokenizer
        self.proj_dim = proj_dim
        self.beam_width = beam_width
        self.threshold_percentile = threshold_percentile
        self.device = next(model.parameters()).device
        
        # Get model dimensions
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Extract and prepare LM head
        self.lm_head = model.lm_head
        self.lm_head_weight = self.lm_head.weight
        
        # Precompute SVD projection matrix
        with torch.no_grad():
            print("Computing SVD projection matrix...")
            # Convert to float32 for SVD computation
            weight_fp32 = self.lm_head_weight.float()
            
            # Note: SVD on MPS will fall back to CPU - this is expected behavior
            print("Computing SVD (may fall back to CPU if using MPS)...")
            try:
                _, _, v = torch.svd(weight_fp32)
                # Convert projection matrix back to model dtype
                self.projection_matrix = v[:, :proj_dim].to(self.device).to(self.lm_head_weight.dtype)
                self.projected_vocab = torch.matmul(self.lm_head_weight, self.projection_matrix)
                
                # Calibrate threshold
                print("Calibrating threshold...")
                self.threshold = self._calibrate_threshold()
                print(f"Using threshold: {self.threshold:.4f}")
            except Exception as e:
                print(f"SVD computation failed: {e}. Using fallback projection.")
                # Fallback: use random projection if SVD fails
                self.projection_matrix = torch.randn(self.hidden_dim, proj_dim).to(self.device).to(self.lm_head_weight.dtype)
                self.projection_matrix = torch.nn.functional.normalize(self.projection_matrix, dim=0)
                self.projected_vocab = torch.matmul(self.lm_head_weight, self.projection_matrix)
                self.threshold = DEFAULT_THRESHOLD
                print(f"Using default threshold: {self.threshold}")
    
    def _calibrate_threshold(self):
        try:
            # Use a smaller sample to avoid memory issues
            n_samples = 500
            rand_indices = torch.randint(0, self.vocab_size, (n_samples,))
            sample_vecs = self.lm_head_weight[rand_indices]
            
            # Normalize vectors for more stable dot products
            sample_vecs = torch.nn.functional.normalize(sample_vecs, dim=1)
            
            # Get dot products for random vectors
            dot_products = torch.matmul(sample_vecs, sample_vecs.T)
            
            # Flatten and get percentile
            flat_dots = dot_products.view(-1).cpu().numpy()
            
            # Filter out non-finite values and handle empty results
            finite_mask = np.isfinite(flat_dots)
            if not np.any(finite_mask) or flat_dots.size == 0:
                return DEFAULT_THRESHOLD
                
            # Remove diagonal elements (self-similarity = 1.0) to get better distribution
            filtered_dots = flat_dots[finite_mask]
            
            # Use a more robust percentile calculation with explicit interpolation
            with np.errstate(all='ignore'):  # Suppress numpy warnings
                try:
                    threshold = np.percentile(filtered_dots, self.threshold_percentile, 
                                             interpolation='linear')
                    
                    # Safety check
                    if not np.isfinite(threshold):
                        return DEFAULT_THRESHOLD
                        
                    # Ensure positive threshold (dot products should be positive for similar vectors)
                    return max(threshold, 0.1)
                except Exception:
                    return DEFAULT_THRESHOLD
        except Exception as e:
            print(f"Threshold calibration failed: {e}. Using default threshold.")
            return DEFAULT_THRESHOLD
    
    @torch.no_grad()
    def generate_baseline(self, prompt, max_new_tokens=20):
        """Generate text using standard generation (baseline)"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Record starting time
        start_time = time.time()
        
        # Tokenize and generate
        outputs = []
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Forward pass
            model_out = self.model(input_ids)
            next_token_logits = model_out.logits[:, -1, :]
            
            # Get next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to outputs
            outputs.append(next_token.item())
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Calculate time and tokens per second
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_sec = max_new_tokens / time_taken
        
        generated_text = self.tokenizer.decode(outputs)
        
        return {
            "generated_tokens": outputs,
            "generated_text": generated_text,
            "time_taken": time_taken,
            "tokens_per_sec": tokens_per_sec
        }
    
    @torch.no_grad()
    def generate_roadrunner(self, prompt, max_new_tokens=20):
        """Generate text using RoadRunner optimized generation"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Record starting time
        start_time = time.time()
        
        # Initial forward pass
        outputs = self.model(input_ids, return_dict=True)
        
        # Generate tokens one by one
        generated_tokens = []
        speculative_hits = 0
        speculative_misses = 0
        
        for _ in range(max_new_tokens):
            try:
                # For LLaMA models, we work directly with the logits from the output
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                
                # For speculative decoding with RoadRunner, we reuse the LM head matrix
                # Matrix is [vocab_size, hidden_dim] for the full vocab
                # First, project logits to a normalized hidden state representation
                hidden_state = torch.matmul(torch.softmax(logits, dim=-1), self.lm_head_weight)
                
                # Use RoadRunner projection for fast candidate selection
                proj_hidden = torch.matmul(hidden_state, self.projection_matrix)
                sims = torch.matmul(proj_hidden, self.projected_vocab.T)
                topk_vals, topk_idxs = torch.topk(sims, self.beam_width, dim=-1)
                
                # Rerank candidates with full logits
                token_vectors = self.lm_head_weight[topk_idxs[0]]
                dot_scores = torch.matmul(hidden_state, token_vectors.T)
                
                # Check if any score passes threshold
                max_score = torch.max(dot_scores)
                if max_score >= self.threshold:
                    # Use speculative result
                    next_token = topk_idxs[0, torch.argmax(dot_scores, dim=-1)].unsqueeze(0)
                    speculative_hits += 1
                else:
                    # Fallback to using the original logits
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    speculative_misses += 1
                
                # Append to generated tokens
                generated_tokens.append(next_token.item())
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Forward pass for next token
                outputs = self.model(input_ids[:, -1:], use_cache=True, past_key_values=outputs.past_key_values)
            except Exception as e:
                # Fallback to get at least some token
                print(f"Error during generation: {e}. Falling back to simple generation.")
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                outputs = self.model(input_ids, return_dict=True)
                speculative_misses += 1
        
        # Calculate time and tokens per second
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_sec = max_new_tokens / time_taken
        speculation_rate = (speculative_hits / max_new_tokens) * 100 if max_new_tokens > 0 else 0
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return {
            "generated_tokens": generated_tokens,
            "generated_text": generated_text,
            "time_taken": time_taken,
            "tokens_per_sec": tokens_per_sec,
            "speculation_rate": speculation_rate,
            "speculative_hits": speculative_hits,
            "speculative_misses": speculative_misses
        }

def run_comparison():
    """Run comparison between baseline and RoadRunner inference"""
    # Load model and tokenizer
    print(f"Loading {MODEL_NAME}...")
    dtype = torch.float16 if USE_FP16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialize RoadRunner
    roadrunner = RoadRunnerDecoder(
        model=model,
        tokenizer=tokenizer,
        proj_dim=PROJ_DIM,
        beam_width=BEAM_WIDTH,
        threshold_percentile=THRESHOLD_PERCENTILE
    )
    
    # Test prompts
    test_prompts = [
        "The best way to predict the future is to",
        "In machine learning, attention mechanisms",
        "The key to efficient inference is",
        "Large language models can be optimized by",
        "Matrix factorization techniques help with"
    ]
    
    # Results tracking
    results = {
        "baseline": {"times": [], "speeds": [], "outputs": [], "texts": []},
        "roadrunner": {"times": [], "speeds": [], "outputs": [], "matches": [], 
                      "speculation_rates": [], "texts": []}
    }
    
    print("\n=== Running Comparison Tests ===")
    
    # Run tests for each prompt
    for i, prompt in enumerate(test_prompts[:NUM_TEST_PROMPTS]):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Run baseline generation
        print("Running baseline generation...")
        baseline_result = roadrunner.generate_baseline(prompt, MAX_NEW_TOKENS)
        results["baseline"]["times"].append(baseline_result["time_taken"])
        results["baseline"]["speeds"].append(baseline_result["tokens_per_sec"])
        results["baseline"]["outputs"].append(baseline_result["generated_tokens"])
        results["baseline"]["texts"].append(baseline_result["generated_text"])
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Run RoadRunner generation
        print("Running RoadRunner generation...")
        roadrunner_result = roadrunner.generate_roadrunner(prompt, MAX_NEW_TOKENS)
        results["roadrunner"]["times"].append(roadrunner_result["time_taken"])
        results["roadrunner"]["speeds"].append(roadrunner_result["tokens_per_sec"])
        results["roadrunner"]["outputs"].append(roadrunner_result["generated_tokens"])
        results["roadrunner"]["texts"].append(roadrunner_result["generated_text"])
        results["roadrunner"]["speculation_rates"].append(roadrunner_result["speculation_rate"])
        
        # Calculate token match accuracy
        token_matches = sum(1 for i in range(MAX_NEW_TOKENS) if 
                           roadrunner_result["generated_tokens"][i] == baseline_result["generated_tokens"][i])
        match_accuracy = token_matches / MAX_NEW_TOKENS * 100
        results["roadrunner"]["matches"].append(match_accuracy)
        
        # Print results for this prompt
        print(f"  Baseline: {baseline_result['tokens_per_sec']:.2f} tokens/sec")
        print(f"  RoadRunner: {roadrunner_result['tokens_per_sec']:.2f} tokens/sec (match: {match_accuracy:.1f}%)")
        print(f"  Speculation Success: {roadrunner_result['speculation_rate']:.1f}%")
        print(f"  Speedup: {roadrunner_result['tokens_per_sec'] / baseline_result['tokens_per_sec']:.2f}x")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate aggregate results
    avg_baseline_speed = np.mean(results["baseline"]["speeds"])
    avg_roadrunner_speed = np.mean(results["roadrunner"]["speeds"])
    avg_match_accuracy = np.mean(results["roadrunner"]["matches"])
    avg_speculation_rate = np.mean(results["roadrunner"]["speculation_rates"])
    avg_speedup = avg_roadrunner_speed / avg_baseline_speed
    
    print("\n=== Summary Results ===")
    print(f"Average Baseline Speed: {avg_baseline_speed:.2f} tokens/sec")
    print(f"Average RoadRunner Speed: {avg_roadrunner_speed:.2f} tokens/sec")
    print(f"Average Token Match Accuracy: {avg_match_accuracy:.2f}%")
    print(f"Average Speculation Success Rate: {avg_speculation_rate:.2f}%")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    # Create visualization
    create_results_visualization(results)
    
    # Output text samples
    print("\n=== Text Generation Samples ===")
    for i in range(min(3, len(results["baseline"]["texts"]))):
        print(f"\nPrompt {i+1}: {test_prompts[i]}")
        print(f"  Baseline: {results['baseline']['texts'][i]}")
        print(f"  RoadRunner: {results['roadrunner']['texts'][i]}")
    
    return results

def create_results_visualization(results):
    """Create visualization of comparison results"""
    baseline_speeds = results["baseline"]["speeds"]
    roadrunner_speeds = results["roadrunner"]["speeds"]
    match_accuracies = results["roadrunner"]["matches"]
    speculation_rates = results["roadrunner"]["speculation_rates"]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Speed comparison
    x = np.arange(len(baseline_speeds))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_speeds, width, label='Baseline')
    ax1.bar(x + width/2, roadrunner_speeds, width, label='RoadRunner')
    
    ax1.set_ylabel('Tokens per second')
    ax1.set_title('Generation Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Prompt {i+1}' for i in range(len(baseline_speeds))])
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(x, match_accuracies, 'o-', color='green', label='Token Match Accuracy')
    ax2.axhline(y=np.mean(match_accuracies), color='green', linestyle='--', 
               label=f'Avg: {np.mean(match_accuracies):.1f}%')
    
    ax2.set_ylabel('Match Accuracy (%)')
    ax2.set_title('Token Match Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Prompt {i+1}' for i in range(len(match_accuracies))])
    ax2.set_ylim(0, 110)
    ax2.legend()
    
    # Speculation rate plot
    ax3.plot(x, speculation_rates, 'o-', color='purple', label='Speculation Rate')
    ax3.axhline(y=np.mean(speculation_rates), color='purple', linestyle='--', 
               label=f'Avg: {np.mean(speculation_rates):.1f}%')
    
    ax3.set_ylabel('Speculation Success (%)')
    ax3.set_title('RoadRunner Speculation Success Rate')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Prompt {i+1}' for i in range(len(speculation_rates))])
    ax3.set_ylim(0, 110)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('roadrunner_results.png')
    print("Results visualization saved to roadrunner_results.png")

if __name__ == "__main__":
    run_comparison()

""" Ouptut:
Using device: mps
Loading unsloth/Llama-3.2-1B...
Computing SVD projection matrix...
Computing SVD (may fall back to CPU if using MPS)...
/Users/andrewcampi/Desktop/Projects/current/road_runner_inference_engine/89_roadrunner.py:60: UserWarning: The operator 'aten::linalg_svd' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:14.)
  _, _, v = torch.svd(weight_fp32)
Calibrating threshold...
/Users/andrewcampi/Desktop/Projects/current/road_runner_inference_engine/89_roadrunner.py:67: DeprecationWarning: the `interpolation=` argument to percentile was renamed to `method=`, which has additional options.
Users of the modes 'nearest', 'lower', 'higher', or 'midpoint' are encouraged to review the method they used. (Deprecated NumPy 1.22)
  self.threshold = self._calibrate_threshold()
Using threshold: 0.5000

=== Running Comparison Tests ===

Prompt 1: The best way to predict the future is to
Running baseline generation...
Running RoadRunner generation...
  Baseline: 10.03 tokens/sec
  RoadRunner: 21.35 tokens/sec (match: 100.0%)
  Speculation Success: 30.0%
  Speedup: 2.13x

Prompt 2: In machine learning, attention mechanisms
Running baseline generation...
Running RoadRunner generation...
  Baseline: 15.43 tokens/sec
  RoadRunner: 24.29 tokens/sec (match: 100.0%)
  Speculation Success: 25.0%
  Speedup: 1.57x

Prompt 3: The key to efficient inference is
Running baseline generation...
Running RoadRunner generation...
  Baseline: 16.99 tokens/sec
  RoadRunner: 24.53 tokens/sec (match: 95.0%)
  Speculation Success: 15.0%
  Speedup: 1.44x

Prompt 4: Large language models can be optimized by
Running baseline generation...
Running RoadRunner generation...
  Baseline: 16.57 tokens/sec
  RoadRunner: 24.55 tokens/sec (match: 100.0%)
  Speculation Success: 40.0%
  Speedup: 1.48x

Prompt 5: Matrix factorization techniques help with
Running baseline generation...
Running RoadRunner generation...
  Baseline: 17.08 tokens/sec
  RoadRunner: 24.50 tokens/sec (match: 100.0%)
  Speculation Success: 35.0%
  Speedup: 1.43x

=== Summary Results ===
Average Baseline Speed: 15.22 tokens/sec
Average RoadRunner Speed: 23.84 tokens/sec
Average Token Match Accuracy: 99.00%
Average Speculation Success Rate: 29.00%
Average Speedup: 1.57x
Results visualization saved to roadrunner_results.png

=== Text Generation Samples ===

Prompt 1: The best way to predict the future is to
  Baseline:  create it. That's the philosophy behind the new 2019 Ford F-150 Raptor.
  RoadRunner:  create it. That's the philosophy behind the new 2019 Ford F-150 Raptor.

Prompt 2: In machine learning, attention mechanisms
  Baseline:  are used to focus on specific parts of the input data. They are used in a wide range of
  RoadRunner:  are used to focus on specific parts of the input data. They are used in a wide range of

Prompt 3: The key to efficient inference is
  Baseline:  to use the right model. In this post, we will discuss the difference between the two most popular
  RoadRunner:  to use the right model. In this post, we will discuss the difference between the two most common
"""