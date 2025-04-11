import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime
import argparse

class MatrixFreeLMHead:
    """Matrix-free implementation of the LM head for token prediction."""
    
    def __init__(self, model, threshold=-float('inf')):
        self.model = model
        self.device = model.device
        self.weight = model.lm_head.weight.data.to(self.device)
        self.bias = None  # Llama models typically don't have bias
        self.threshold = threshold
        
        # Store vocabulary size for analytics
        self.vocab_size = self.weight.shape[0]
        self.embed_dim = self.weight.shape[1]
        
        print(f"✅ Matrix-free head initialized with vocab size: {self.vocab_size}, embedding dim: {self.embed_dim}")
    
    def predict(self, hidden_state, measure_time=False):
        """Predict the next token using direct matrix multiplication with optional time measurement."""
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # Ensure we're working with the last token's hidden state
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        
        if measure_time:
            extract_time = time.perf_counter()
            times['extract_hidden'] = (extract_time - start) * 1000  # in ms
        
        # Simple, reliable matrix multiplication
        scores = torch.matmul(hidden_state.view(-1), self.weight.t())
        
        if measure_time:
            matmul_time = time.perf_counter()
            times['matrix_multiply'] = (matmul_time - extract_time) * 1000  # in ms
        
        # Get the token with the highest score
        top_score, top_index = torch.max(scores, dim=0)
        
        if measure_time:
            argmax_time = time.perf_counter()
            times['argmax'] = (argmax_time - matmul_time) * 1000  # in ms
            total_time = argmax_time - start
            times['total'] = total_time * 1000  # in ms
        
        return top_index.unsqueeze(0), top_score.item(), times if measure_time else None


class MatrixFreeLlamaForCausalLM(LlamaForCausalLM):
    """Custom Llama model that can bypass logits computation with matrix-free generation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.matrix_free_mode = False
        self.matrix_free_head = None
        self.timing_stats = {
            'transformer_time': [],
            'lm_head_time': [],
            'setup_time': [],
            'total_time': []
        }
    
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
        measure_time = kwargs.pop('measure_time', False)
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # If not in matrix-free mode, use the standard forward
        if not self.matrix_free_mode:
            result = super().forward(*args, **kwargs)
            
            if measure_time:
                end = time.perf_counter()
                times['total'] = (end - start) * 1000  # in ms
                return result, times
            return result
        
        # Store if we need hidden states for later
        output_hidden_states = kwargs.get('output_hidden_states', False)
        kwargs['output_hidden_states'] = True  # We always need hidden states in matrix-free mode
        
        if measure_time:
            setup_start = time.perf_counter()
        
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
        
        if measure_time:
            setup_end = time.perf_counter()
            times['setup'] = (setup_end - setup_start) * 1000  # in ms
            transformer_start = time.perf_counter()
        
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
        
        if measure_time:
            transformer_end = time.perf_counter()
            times['transformer'] = (transformer_end - transformer_start) * 1000  # in ms
            lm_head_start = time.perf_counter()
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Use our matrix-free head to predict the next token
        next_token, score, head_times = self.matrix_free_head.predict(hidden_states, measure_time=measure_time)
        
        if measure_time and head_times:
            times['lm_head'] = head_times
        
        # Create dummy logits tensor with only the predicted token having a high score
        # This is just for compatibility with the HuggingFace API
        dummy_logits = torch.zeros((1, 1, self.config.vocab_size), device=self.device)
        dummy_logits[0, 0, next_token] = score
        
        result = CausalLMOutputWithPast(
            loss=None,
            logits=dummy_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
        
        if measure_time:
            end = time.perf_counter()
            times['total'] = (end - start) * 1000  # in ms
            
            # Store timing stats
            self.timing_stats['transformer_time'].append(times['transformer'])
            self.timing_stats['lm_head_time'].append(times['lm_head']['total'] if 'lm_head' in times else 0)
            self.timing_stats['setup_time'].append(times['setup'])
            self.timing_stats['total_time'].append(times['total'])
            
            return result, times
        
        return result

def calibrate_threshold(model, prompts, tokenizer, max_new_tokens=5):
    """Calibrate the threshold for matrix-free routing by examining score distributions."""
    scores = []
    
    # Create a router without modifying the model
    router = MatrixFreeLMHead(model, threshold=-float("inf"))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):  # Reduced tokens for faster calibration
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
                _, score, _ = router.predict(hidden, measure_time=False)
                scores.append(score)
                
                # Get actual next token for continuation
                input_ids = torch.argmax(outputs.logits[:, -1:, :], dim=-1)

    # Return the lowest score observed (or a percentile)
    return np.percentile(scores, 0)

def benchmark_token_generation(model, tokenizer, prompt, num_tokens=20, verify_accuracy=True, measure_time=True):
    """Detailed benchmark of token generation with per-component timing."""
    # Prepare inputs
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    
    # Timing statistics
    times = {
        'standard': {
            'transformer': [],
            'lm_head': [],
            'total': []
        },
        'matrix_free': {
            'transformer': [],
            'lm_head': [],
            'setup': [],
            'total': []
        }
    }
    
    # Verification data
    standard_tokens = []
    matrix_free_tokens = []
    matched = 0
    
    # Run standard baseline first
    with torch.no_grad():
        baseline_start = time.perf_counter()
        
        for i in range(num_tokens):
            token_start = time.perf_counter()
            
            # Standard forward pass
            outputs, token_times = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                measure_time=True
            )
            
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            standard_tokens.append(next_token.item())
            
            # Update for next iteration
            past_key_values = outputs.past_key_values
            input_ids = next_token
            all_ids = torch.cat([all_ids, input_ids], dim=1)
            
            # Record timing
            times['standard']['total'].append(token_times['total'])
            
        baseline_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
        baseline_duration = time.perf_counter() - baseline_start
    
    # Enable matrix-free mode
    model.enable_matrix_free()
    
    # Reset for matrix-free run
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        matrix_free_start = time.perf_counter()
        
        for i in range(num_tokens):
            # Matrix-free forward pass
            outputs, token_times = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                measure_time=True
            )
            
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            matrix_free_tokens.append(next_token.item())
            
            # Check accuracy
            if i < len(standard_tokens) and next_token.item() == standard_tokens[i]:
                matched += 1
            
            # Update for next iteration
            past_key_values = outputs.past_key_values
            input_ids = next_token
            all_ids = torch.cat([all_ids, input_ids], dim=1)
            
            # Record detailed timing
            times['matrix_free']['transformer'].append(token_times['transformer'])
            times['matrix_free']['lm_head'].append(token_times['lm_head']['total'] if 'lm_head' in token_times else 0)
            times['matrix_free']['setup'].append(token_times['setup'])
            times['matrix_free']['total'].append(token_times['total'])
            
        matrix_free_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
        matrix_free_duration = time.perf_counter() - matrix_free_start
    
    # Disable matrix-free mode when done
    model.disable_matrix_free()
    
    return {
        'standard': {
            'text': baseline_text,
            'tokens_per_sec': num_tokens / baseline_duration,
            'ms_per_token': (baseline_duration / num_tokens) * 1000,
            'times': times['standard']
        },
        'matrix_free': {
            'text': matrix_free_text,
            'tokens_per_sec': num_tokens / matrix_free_duration,
            'ms_per_token': (matrix_free_duration / num_tokens) * 1000,
            'times': times['matrix_free']
        },
        'accuracy': matched / num_tokens if verify_accuracy else None,
        'tokens': {
            'standard': standard_tokens,
            'matrix_free': matrix_free_tokens
        }
    }

def run_detailed_analysis(model_name, max_tokens=50, output_dir='results'):
    """Run a detailed analysis with extensive metrics and visualizations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    run_dir = os.path.join(output_dir, f"{model_short_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(run_dir, "benchmark_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Matrix-Free Token Generation Analysis\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens per prompt: {max_tokens}\n\n")
    
    # Get device
    device = get_device()
    
    # Log device information
    with open(log_file, 'a') as f:
        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        f.write("\n")
    
    print(f"🔄 Loading model: {model_name} on {device}...")
    
    # Set precision based on device
    precision = torch.float16 if device.type == "cuda" else torch.float32
    
    # Load model
    config = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
    ).config
    
    # Create custom model
    model = MatrixFreeLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        config=config
    ).to(device).eval()
    
    # Log model information
    with open(log_file, 'a') as f:
        f.write(f"Model configuration:\n")
        f.write(f"  Vocab size: {model.config.vocab_size}\n")
        f.write(f"  Hidden size: {model.config.hidden_size}\n")
        f.write(f"  Intermediate size: {model.config.intermediate_size}\n")
        f.write(f"  Num hidden layers: {model.config.num_hidden_layers}\n")
        f.write(f"  Num attention heads: {model.config.num_attention_heads}\n")
        f.write(f"  Precision: {precision}\n\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sample prompts for testing
    prompts = [
        "The meaning of life is",
        "In a distant galaxy, a civilization",
        "The future of AI will depend on",
        "Once upon a time there was",
        "The quantum computer revolutionized",
        "Scientists have discovered a new",
        "The theory of relativity explains",
        "Climate change has led to",
        "Machine learning algorithms can",
        "The human brain processes information"
    ]
    
    # Calibrate threshold
    print("\n📊 Calibrating threshold...")
    threshold = calibrate_threshold(model, prompts[:3], tokenizer, max_new_tokens=5)
    print(f"✅ Calibrated threshold: {threshold:.2f}")
    
    # Log threshold
    with open(log_file, 'a') as f:
        f.write(f"Calibrated threshold: {threshold:.2f}\n\n")
    
    # Run detailed benchmarks
    print("\n🔬 Running detailed benchmark analysis...")
    
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Analyzing prompts")):
        print(f"\n📝 Prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        # Perform detailed analysis
        benchmark = benchmark_token_generation(
            model, 
            tokenizer, 
            prompt, 
            num_tokens=max_tokens,
            verify_accuracy=True,
            measure_time=True
        )
        
        # Add to results
        results.append({
            'prompt': prompt,
            'benchmark': benchmark
        })
        
        # Log individual prompt results
        with open(log_file, 'a') as f:
            f.write(f"Prompt: '{prompt}'\n")
            f.write(f"  Standard: {benchmark['standard']['tokens_per_sec']:.2f} tokens/sec, {benchmark['standard']['ms_per_token']:.2f} ms/token\n")
            f.write(f"  Matrix-Free: {benchmark['matrix_free']['tokens_per_sec']:.2f} tokens/sec, {benchmark['matrix_free']['ms_per_token']:.2f} ms/token\n")
            f.write(f"  Accuracy: {benchmark['accuracy']:.2%}\n")
            f.write(f"  Speedup: {benchmark['matrix_free']['tokens_per_sec'] / benchmark['standard']['tokens_per_sec']:.2f}x\n\n")
            
            # Add time breakdowns
            f.write(f"  Matrix-Free time breakdown (averages):\n")
            f.write(f"    Transformer: {np.mean(benchmark['matrix_free']['times']['transformer']):.2f} ms ({np.mean(benchmark['matrix_free']['times']['transformer']) / np.mean(benchmark['matrix_free']['times']['total']) * 100:.1f}%)\n")
            f.write(f"    LM Head: {np.mean(benchmark['matrix_free']['times']['lm_head']):.2f} ms ({np.mean(benchmark['matrix_free']['times']['lm_head']) / np.mean(benchmark['matrix_free']['times']['total']) * 100:.1f}%)\n")
            f.write(f"    Setup: {np.mean(benchmark['matrix_free']['times']['setup']):.2f} ms ({np.mean(benchmark['matrix_free']['times']['setup']) / np.mean(benchmark['matrix_free']['times']['total']) * 100:.1f}%)\n\n")
        
        # Create visualization for this prompt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Token generation time comparison
        axes[0, 0].plot(benchmark['standard']['times']['total'], label='Standard', marker='o')
        axes[0, 0].plot(benchmark['matrix_free']['times']['total'], label='Matrix-Free', marker='x')
        axes[0, 0].set_title('Token Generation Time')
        axes[0, 0].set_xlabel('Token #')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Matrix-free time breakdown
        df = pd.DataFrame({
            'Transformer': benchmark['matrix_free']['times']['transformer'],
            'LM Head': benchmark['matrix_free']['times']['lm_head'],
            'Setup': benchmark['matrix_free']['times']['setup']
        })
        df.plot.area(ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Matrix-Free Time Breakdown')
        axes[0, 1].set_xlabel('Token #')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].grid(True)
        
        # 3. Cumulative time
        std_cumulative = np.cumsum(benchmark['standard']['times']['total'])
        mf_cumulative = np.cumsum(benchmark['matrix_free']['times']['total'])
        axes[1, 0].plot(std_cumulative, label='Standard', marker='o')
        axes[1, 0].plot(mf_cumulative, label='Matrix-Free', marker='x')
        axes[1, 0].set_title('Cumulative Generation Time')
        axes[1, 0].set_xlabel('Token #')
        axes[1, 0].set_ylabel('Cumulative Time (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Time ratio
        ratio = np.array(benchmark['matrix_free']['times']['total']) / np.array(benchmark['standard']['times']['total'])
        axes[1, 1].plot(ratio, marker='o', color='purple')
        axes[1, 1].axhline(y=1.0, linestyle='--', color='r')
        axes[1, 1].set_title('Matrix-Free / Standard Time Ratio')
        axes[1, 1].set_xlabel('Token #')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_ylim(0, 2)  # Adjust as needed
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"prompt_{i+1}_analysis.png"))
        plt.close()
    
    # Aggregate results
    print("\n📊 Aggregating results...")
    
    standard_tps = [r['benchmark']['standard']['tokens_per_sec'] for r in results]
    matrix_free_tps = [r['benchmark']['matrix_free']['tokens_per_sec'] for r in results]
    accuracies = [r['benchmark']['accuracy'] for r in results]
    speedups = [mf / std for mf, std in zip(matrix_free_tps, standard_tps)]
    
    # Calculate averages
    avg_standard_tps = np.mean(standard_tps)
    avg_matrix_free_tps = np.mean(matrix_free_tps)
    avg_accuracy = np.mean(accuracies)
    avg_speedup = np.mean(speedups)
    
    # Log summary
    with open(log_file, 'a') as f:
        f.write("\nSUMMARY\n")
        f.write("=======\n\n")
        f.write(f"Average Standard: {avg_standard_tps:.2f} tokens/sec\n")
        f.write(f"Average Matrix-Free: {avg_matrix_free_tps:.2f} tokens/sec\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4%}\n")
        f.write(f"Average Speedup: {avg_speedup:.2f}x\n")
        
        if avg_speedup > 1:
            f.write(f"Resource savings: {(1 - 1/avg_speedup) * 100:.1f}% reduction in inference time\n")
        else:
            f.write(f"Overhead: {(1/avg_speedup - 1) * 100:.1f}%\n")
    
    # Create summary visualizations
    # 1. Tokens per second comparison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(prompts))
    width = 0.35
    
    plt.bar(x - width/2, standard_tps, width, label='Standard')
    plt.bar(x + width/2, matrix_free_tps, width, label='Matrix-Free')
    
    plt.xlabel('Prompt #')
    plt.ylabel('Tokens per Second')
    plt.title('Generation Speed Comparison')
    plt.xticks(x, [f"Prompt {i+1}" for i in range(len(prompts))])
    plt.legend()
    plt.grid(True, axis='y')
    
    for i, (std, mf) in enumerate(zip(standard_tps, matrix_free_tps)):
        plt.text(i - width/2, std + 0.5, f"{std:.1f}", ha='center')
        plt.text(i + width/2, mf + 0.5, f"{mf:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "speed_comparison.png"))
    plt.close()
    
    # 2. Speedup ratio
    plt.figure(figsize=(14, 6))
    
    plt.bar(x, speedups, color='purple')
    plt.axhline(y=1.0, linestyle='--', color='r')
    
    plt.xlabel('Prompt #')
    plt.ylabel('Speedup Ratio')
    plt.title('Matrix-Free / Standard Speedup Ratio')
    plt.xticks(x, [f"Prompt {i+1}" for i in range(len(prompts))])
    plt.grid(True, axis='y')
    
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup + 0.02, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "speedup_ratio.png"))
    plt.close()
    
    # 3. Time breakdown pie chart (average across all prompts)
    setup_times = []
    transformer_times = []
    lm_head_times = []
    
    for r in results:
        setup_times.extend(r['benchmark']['matrix_free']['times']['setup'])
        transformer_times.extend(r['benchmark']['matrix_free']['times']['transformer'])
        lm_head_times.extend(r['benchmark']['matrix_free']['times']['lm_head'])
    
    avg_setup = np.mean(setup_times)
    avg_transformer = np.mean(transformer_times)
    avg_lm_head = np.mean(lm_head_times)
    
    plt.figure(figsize=(10, 8))
    
    plt.pie(
        [avg_setup, avg_transformer, avg_lm_head],
        labels=['Setup', 'Transformer', 'LM Head'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff9999','#66b3ff','#99ff99']
    )
    plt.title('Matrix-Free Time Distribution')
    plt.axis('equal')
    
    plt.savefig(os.path.join(run_dir, "time_distribution.png"))
    plt.close()
    
    # Print final summary
    print("\n📋 SUMMARY")
    print("==========")
    print(f"Average Standard: {avg_standard_tps:.2f} tokens/sec")
    print(f"Average Matrix-Free: {avg_matrix_free_tps:.2f} tokens/sec")
    print(f"Average Accuracy: {avg_accuracy:.4%}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1:
        print(f"💰 Resource savings: {(1 - 1/avg_speedup) * 100:.1f}% reduction in inference time")
    else:
        print(f"⚠️ Overhead: {(1/avg_speedup - 1) * 100:.1f}%")
    
    # Print time breakdown
    total_avg = avg_setup + avg_transformer + avg_lm_head
    print("\n⏱️ Matrix-Free Time Breakdown:")
    print(f"  Setup: {avg_setup:.2f} ms ({avg_setup/total_avg*100:.1f}%)")
    print(f"  Transformer: {avg_transformer:.2f} ms ({avg_transformer/total_avg*100:.1f}%)")
    print(f"  LM Head: {avg_lm_head:.2f} ms ({avg_lm_head/total_avg*100:.1f}%)")
    
    print(f"\n✅ Results saved to: {run_dir}")

def get_device():
    """Get the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix-Free Token Generation Analysis")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate per prompt")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    run_detailed_analysis(
        model_name=args.model,
        max_tokens=args.tokens,
        output_dir=args.output
    )

""" Output:
🔄 Loading Llama 3 model: unsloth/Llama-3.1-8B-Instruct on cuda...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.81it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.80it/s]

🔍 Profiling baseline generation...

🔍 Profiling matrix-free generation...

📊 Baseline Generation Profile Summary:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                              baseline_token_generation        31.77%      77.160ms        99.66%     242.067ms      48.413ms       0.000us         0.00%      90.127ms      18.025ms           0 b           0 b     383.50 Kb    -135.86 Mb             5  
                                           aten::linear         1.19%       2.900ms        20.14%      48.906ms      43.472us       0.000us         0.00%      81.522ms      72.464us           0 b           0 b      28.70 Mb           0 b          1125  
                                           aten::matmul         2.52%       6.118ms        16.69%      40.538ms      35.874us       0.000us         0.00%      81.528ms      72.149us           0 b           0 b      28.70 Mb           0 b          1130  
                                       cudaLaunchKernel        16.30%      39.602ms        16.32%      39.646ms       5.949us       0.000us         0.00%       1.823us       0.000us           0 b           0 b           0 b           0 b          6664  
                                               aten::mm         9.07%      22.020ms        12.47%      30.293ms      26.927us      81.398ms        90.44%      81.522ms      72.464us           0 b           0 b      28.70 Mb      28.70 Mb          1125  
                                              aten::mul         5.24%      12.732ms         8.16%      19.820ms      13.576us       2.076ms         2.31%       2.076ms       1.422us           0 b           0 b      30.24 Mb      30.24 Mb          1460  
                                               aten::to         0.39%     943.849us         6.38%      15.506ms      15.352us       0.000us         0.00%     897.042us       0.888us           0 b           0 b      15.24 Mb      -8.00 Kb          1010  
                                              aten::add         3.25%       7.898ms         5.97%      14.503ms      15.029us     960.368us         1.07%     960.368us       0.995us           0 b           0 b       8.28 Mb       8.28 Mb           965  
                                         aten::_to_copy         1.32%       3.209ms         5.96%      14.484ms      21.780us       0.000us         0.00%     897.042us       1.349us           0 b           0 b      15.24 Mb           0 b           665  
                                            aten::copy_         2.58%       6.258ms         5.53%      13.430ms      12.803us       1.551ms         1.72%       1.551ms       1.479us           0 b           0 b           0 b           0 b          1049  
                                          aten::reshape         1.19%       2.899ms         5.39%      13.104ms       8.089us       0.000us         0.00%     547.092us       0.338us           0 b           0 b      20.00 Mb           0 b          1620  
                     aten::scaled_dot_product_attention         0.55%       1.340ms         4.17%      10.123ms      63.270us       0.000us         0.00%     632.882us       3.956us           0 b      -2.50 Kb       2.50 Mb     -96.00 Kb           160  
                                              aten::cat         2.63%       6.387ms         3.94%       9.581ms      16.491us       1.182ms         1.31%       1.184ms       2.038us           0 b           0 b       7.38 Mb       7.38 Mb           581  
                                            aten::clone         0.47%       1.153ms         3.83%       9.304ms      24.229us       0.000us         0.00%     653.937us       1.703us           0 b           0 b      23.00 Mb           0 b           384  
              aten::_scaled_dot_product_flash_attention         0.52%       1.271ms         3.62%       8.783ms      54.894us       0.000us         0.00%     632.882us       3.956us       2.50 Kb           0 b       2.59 Mb           0 b           160  
                                        aten::transpose         2.02%       4.902ms         2.80%       6.807ms       2.825us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2410  
                                            aten::slice         2.26%       5.478ms         2.74%       6.654ms       3.386us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1965  
                         aten::_flash_attention_forward         0.90%       2.184ms         2.54%       6.168ms      38.548us     632.882us         0.70%     632.882us       3.956us       2.50 Kb          56 b       2.59 Mb           0 b           160  
                                                aten::t         1.00%       2.420ms         2.36%       5.737ms       5.099us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1125  
                                              aten::pow         1.39%       3.382ms         2.11%       5.130ms      15.784us     297.498us         0.33%     297.498us       0.915us           0 b           0 b      10.16 Mb      10.16 Mb           325  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 242.884ms
Self CUDA time total: 90.002ms


📊 Matrix-Free Generation Profile Summary:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           matrix_free_token_generation        31.03%      73.765ms       100.00%     237.713ms      47.543ms       0.000us         0.00%      90.641ms      18.128ms           0 b         -16 b     383.50 Kb    -137.11 Mb             5  
                                           aten::linear         0.94%       2.240ms        20.16%      47.930ms      42.795us       0.000us         0.00%      75.924ms      67.790us           0 b           0 b      26.25 Mb           0 b          1120  
                                           aten::matmul         2.33%       5.538ms        17.17%      40.809ms      36.114us       0.000us         0.00%      81.423ms      72.056us           0 b           0 b      27.48 Mb           0 b          1130  
                                       cudaLaunchKernel        16.83%      39.998ms        16.83%      39.998ms       5.988us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          6680  
                                               aten::mm         9.01%      21.419ms        13.22%      31.426ms      27.935us      81.417ms        89.79%      81.417ms      72.371us           0 b           0 b      27.47 Mb      27.47 Mb          1125  
                                              aten::mul         5.05%      11.997ms         8.18%      19.438ms      13.314us       2.185ms         2.41%       2.185ms       1.496us           0 b           0 b      30.24 Mb      30.24 Mb          1460  
                                               aten::to         0.41%     968.856us         6.35%      15.088ms      14.865us       0.000us         0.00%     955.816us       0.942us           0 b           0 b      15.25 Mb           0 b          1015  
                                         aten::_to_copy         1.29%       3.069ms         5.94%      14.120ms      21.074us       0.000us         0.00%     955.816us       1.427us           0 b           0 b      15.25 Mb           0 b           670  
                                            aten::copy_         2.57%       6.104ms         5.54%      13.169ms      12.495us       1.643ms         1.81%       1.643ms       1.558us           0 b           0 b           0 b           0 b          1054  
                                              aten::add         3.23%       7.688ms         5.30%      12.589ms      13.046us       1.013ms         1.12%       1.013ms       1.050us           0 b           0 b       8.28 Mb       8.28 Mb           965  
                                          aten::reshape         1.12%       2.654ms         5.19%      12.347ms       7.621us       0.000us         0.00%     580.758us       0.358us           0 b           0 b      20.00 Mb           0 b          1620  
                     aten::scaled_dot_product_attention         0.54%       1.285ms         3.97%       9.443ms      59.019us       0.000us         0.00%     673.502us       4.209us          16 b      -2.48 Kb       2.50 Mb     -96.00 Kb           160  
                                              aten::cat         2.54%       6.039ms         3.88%       9.212ms      15.856us       1.254ms         1.38%       1.254ms       2.159us           0 b           0 b       7.38 Mb       7.38 Mb           581  
                                            aten::clone         0.44%       1.045ms         3.76%       8.944ms      23.291us       0.000us         0.00%     686.738us       1.788us           0 b           0 b      23.00 Mb           0 b           384  
              aten::_scaled_dot_product_flash_attention         0.45%       1.066ms         3.43%       8.158ms      50.985us       0.000us         0.00%     673.502us       4.209us       2.50 Kb           0 b       2.59 Mb           0 b           160  
                                        aten::transpose         1.84%       4.381ms         2.55%       6.061ms       2.515us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2410  
                         aten::_flash_attention_forward         0.88%       2.098ms         2.50%       5.940ms      37.122us     673.502us         0.74%     673.502us       4.209us       2.50 Kb          64 b       2.59 Mb           0 b           160  
                                            aten::slice         2.05%       4.867ms         2.47%       5.883ms       3.001us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1960  
                                                aten::t         1.00%       2.370ms         2.27%       5.394ms       4.795us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1125  
                                             aten::item         0.00%       8.641us         2.26%       5.373ms       1.075ms       0.000us         0.00%       5.439us       1.088us           0 b           0 b           0 b           0 b             5  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 237.719ms
Self CUDA time total: 90.679ms


✅ Profiling traces exported to baseline_trace.json and matrix_free_trace.json

🔎 Analyzing operations related to logits computation:
Baseline logits computation time: 0.30 ms
Matrix-free predict operation time: 0.21 ms

💾 Memory Usage Analysis:
Baseline Memory Usage (Top Allocations):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::empty         1.40%       3.389ms         1.40%       3.389ms       3.277us       0.000us         0.00%       0.000us       0.000us       2.45 Kb       2.45 Kb      23.09 Mb      23.09 Mb          1034  
                         aten::_flash_attention_forward         0.90%       2.184ms         2.54%       6.168ms      38.548us     632.882us         0.70%     632.882us       3.956us       2.50 Kb          56 b       2.59 Mb           0 b           160  
                              baseline_token_generation        31.77%      77.160ms        99.66%     242.067ms      48.413ms       0.000us         0.00%      90.127ms      18.025ms           0 b           0 b     383.50 Kb    -135.86 Mb             5  
                                        aten::embedding         0.02%      46.060us         0.26%     637.631us     127.526us       0.000us         0.00%      10.784us       2.157us           0 b           0 b      80.00 Kb           0 b             5  
                                          aten::reshape         1.19%       2.899ms         5.39%      13.104ms       8.089us       0.000us         0.00%     547.092us       0.338us           0 b           0 b      20.00 Mb           0 b          1620  
                                             aten::view         0.93%       2.261ms         0.93%       2.261ms       1.267us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1785  
                                     aten::index_select         0.19%     460.030us         0.23%     568.321us     113.664us      10.784us         0.01%      10.784us       2.157us           0 b           0 b      80.00 Kb           0 b             5  
                                          aten::resize_         0.02%      45.761us         0.02%      45.761us       4.576us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      82.50 Kb      82.50 Kb            10  
                                       cudaLaunchKernel        16.30%      39.602ms        16.32%      39.646ms       5.949us       0.000us         0.00%       1.823us       0.000us           0 b           0 b           0 b           0 b          6664  
                              baseline_token_generation         0.00%       0.000us         0.00%       0.000us       0.000us     242.416ms       269.35%     242.416ms      48.483ms           0 b           0 b           0 b           0 b             5  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 242.884ms
Self CUDA time total: 90.002ms


Matrix-Free Memory Usage (Top Allocations):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::empty         1.39%       3.294ms         1.39%       3.294ms       3.171us       0.000us         0.00%       0.000us       0.000us       2.44 Kb       2.44 Kb      25.54 Mb      25.54 Mb          1039  
                         aten::_flash_attention_forward         0.88%       2.098ms         2.50%       5.940ms      37.122us     673.502us         0.74%     673.502us       4.209us       2.50 Kb          64 b       2.59 Mb           0 b           160  
                                        aten::embedding         0.02%      39.490us         0.21%     502.128us     100.426us       0.000us         0.00%      10.976us       2.195us           0 b           0 b      80.00 Kb           0 b             5  
                                          aten::reshape         1.12%       2.654ms         5.19%      12.347ms       7.621us       0.000us         0.00%     580.758us       0.358us           0 b           0 b      20.00 Mb           0 b          1620  
                                             aten::view         0.90%       2.131ms         0.90%       2.131ms       1.191us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          1790  
                                     aten::index_select         0.15%     348.087us         0.19%     444.357us      88.871us      10.976us         0.01%      10.976us       2.195us           0 b           0 b      80.00 Kb           0 b             5  
                                          aten::resize_         0.01%      34.300us         0.01%      34.300us       3.430us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      82.50 Kb      82.50 Kb            10  
                                       cudaLaunchKernel        16.83%      39.998ms        16.83%      39.998ms       5.988us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          6680  
                           matrix_free_token_generation         0.00%       0.000us         0.00%       0.000us       0.000us     236.802ms       261.14%     236.802ms      47.360ms           0 b           0 b           0 b           0 b             5  
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us      10.976us         0.01%      10.976us       2.195us           0 b           0 b           0 b           0 b             5  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 237.719ms
Self CUDA time total: 90.679ms


⏱️ Time breakdown by model components:
Baseline time breakdown:
  - attention: 4.15%
  - mlp: 0.00%
  - lm_head/predict: 0.05%
  - other: 95.80%

Matrix-free time breakdown:
  - attention: 3.87%
  - mlp: 0.00%
  - lm_head/predict: 0.04%
  - other: 96.09%
"""