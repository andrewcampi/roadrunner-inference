import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import gc

@dataclass
class TimingStats:
    total_time: float = 0.0
    transformer_time: float = 0.0
    lm_head_time: float = 0.0
    router_time: float = 0.0
    misc_time: float = 0.0
    token_count: int = 0
    
    def get_tokens_per_second(self):
        return self.token_count / self.total_time if self.total_time > 0 else 0
    
    def get_breakdown(self):
        if self.total_time == 0:
            return {}
        
        return {
            "transformer_pct": (self.transformer_time / self.total_time) * 100,
            "lm_head_pct": (self.lm_head_time / self.total_time) * 100,
            "router_pct": (self.router_time / self.total_time) * 100,
            "misc_pct": (self.misc_time / self.total_time) * 100,
            "tokens_per_sec": self.get_tokens_per_second(),
            "ms_per_token": (self.total_time / self.token_count) * 1000 if self.token_count > 0 else 0
        }


class DotProductRoutedLMHead:
    def __init__(self, model, threshold):
        self.model = model
        self.device = model.device
        self.weight = model.lm_head.weight.data.to(self.device)
        self.bias = model.lm_head.bias.data.to(self.device) if model.lm_head.bias is not None else None
        self.threshold = threshold
        self.vocab_size = self.weight.shape[0]
        self.hidden_size = self.weight.shape[1]
        
        # Pre-normalize weights for faster dot product
        self.weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        
        # Initialize timing stats
        self.stats = TimingStats()

    def predict(self, hidden_state):
        start = time.perf_counter()
        
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
            
        # Fast path: Compute just the score for top tokens
        scores = torch.matmul(self.weight, hidden_state.view(-1))  # [vocab]
        
        if self.bias is not None:
            scores += self.bias
            
        top1_score, top1_index = torch.max(scores, dim=0)
        
        self.stats.router_time += time.perf_counter() - start
        return top1_index.unsqueeze(0), top1_score.item()
    
    def reset_stats(self):
        self.stats = TimingStats()


def detailed_generate_baseline(model, tokenizer, prompt, max_new_tokens):
    """Generate with full metrics breakdown using standard HF generation"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    
    stats = TimingStats()
    stats.token_count = max_new_tokens
    
    start_total = time.perf_counter()
    with torch.no_grad():
        # Measure transformer forward pass time
        start_transformer = time.perf_counter()
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        stats.transformer_time = time.perf_counter() - start_transformer
        
    stats.total_time = time.perf_counter() - start_total
    
    # Calculate remaining time as misc
    stats.misc_time = stats.total_time - stats.transformer_time
    
    return {
        "text": tokenizer.decode(outputs.sequences[0], skip_special_tokens=True),
        "stats": stats
    }


def detailed_generate_routed(model, tokenizer, prompt, max_new_tokens, threshold):
    """Generate with full metrics breakdown using the matrix-free approach"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    router = DotProductRoutedLMHead(model, threshold=threshold)
    router.reset_stats()
    
    stats = TimingStats()
    stats.token_count = max_new_tokens
    
    matched = 0
    start_total = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Measure transformer time
            start_transformer = time.perf_counter()
            out = model.transformer(input_ids, past_key_values=past_key_values, use_cache=True)
            hidden = out[0][:, -1:, :]
            past_key_values = out[1]
            end_transformer = time.perf_counter()
            stats.transformer_time += end_transformer - start_transformer
            
            # Measure routing time
            start_router = time.perf_counter()
            pred_id, score = router.predict(hidden)
            end_router = time.perf_counter()
            
            # Measure LM head time
            start_lm = time.perf_counter()
            baseline = torch.argmax(model.lm_head(hidden), dim=-1).item()
            end_lm = time.perf_counter()
            stats.lm_head_time += end_lm - start_lm
            
            if pred_id.item() == baseline:
                matched += 1
                
            # Track timing
            input_ids = pred_id.unsqueeze(0)
            all_ids = torch.cat([all_ids, input_ids], dim=1)
            
    stats.total_time = time.perf_counter() - start_total
    stats.router_time = router.stats.router_time
    
    # Calculate remaining time as misc
    stats.misc_time = stats.total_time - (stats.transformer_time + stats.lm_head_time + stats.router_time)
    
    return {
        "text": tokenizer.decode(all_ids[0], skip_special_tokens=True),
        "accuracy": matched / max_new_tokens,
        "stats": stats
    }


def calibrate_threshold(model, prompts, tokenizer, max_new_tokens=20):
    """Same calibration as original, just for threshold"""
    scores = []
    router = DotProductRoutedLMHead(model, threshold=-float("inf"))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model.transformer(input_ids, past_key_values=past_key_values, use_cache=True)
                hidden = out[0][:, -1:, :]
                past_key_values = out[1]

                _, score = router.predict(hidden)
                scores.append(score)
                input_ids = torch.argmax(model.lm_head(hidden), dim=-1).unsqueeze(0)
    return np.percentile(scores, 0)


def optimize_memory():
    """Free unused memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_full_analysis(model_name="gpt2", max_new_tokens=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    print("🔄 Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {model_name} - {total_params/1e6:.2f}M parameters")
    lm_head_params = model.lm_head.weight.numel()
    print(f"📊 LM Head size: {lm_head_params/1e6:.2f}M parameters")
    print(f"📊 Vocab size: {len(tokenizer)}")
    print(f"📊 Hidden size: {model.config.n_embd}")

    prompts = [
        "The meaning of life is",
        "In a distant galaxy, a civilization",
        "The future of AI will depend on",
        "Once upon a time",
        "The quantum computer"
    ]

    print("\n📊 Calibrating routing threshold...")
    threshold = calibrate_threshold(model, prompts, tokenizer, max_new_tokens)
    print(f"✅ Calibrated P0 threshold: {threshold:.2f}\n")

    baseline_stats = TimingStats()
    routed_stats = TimingStats()
    
    for i, prompt in enumerate(prompts):
        print(f"🧪 Prompt {i+1}/{len(prompts)}: {prompt}")
        
        # Run baseline
        optimize_memory()
        baseline = detailed_generate_baseline(model, tokenizer, prompt, max_new_tokens)
        baseline_stats.transformer_time += baseline["stats"].transformer_time
        baseline_stats.misc_time += baseline["stats"].misc_time
        baseline_stats.total_time += baseline["stats"].total_time
        baseline_stats.token_count += baseline["stats"].token_count
        
        # Run routed
        optimize_memory()
        routed = detailed_generate_routed(model, tokenizer, prompt, max_new_tokens, threshold)
        routed_stats.transformer_time += routed["stats"].transformer_time
        routed_stats.lm_head_time += routed["stats"].lm_head_time
        routed_stats.router_time += routed["stats"].router_time
        routed_stats.misc_time += routed["stats"].misc_time
        routed_stats.total_time += routed["stats"].total_time
        routed_stats.token_count += routed["stats"].token_count
        
        print(f"--- GPT-2 Baseline ---")
        print(f"📜 {baseline['text']}")
        baseline_breakdown = baseline["stats"].get_breakdown()
        print(f"⚡ {baseline_breakdown.get('tokens_per_sec', 0):.2f} tokens/sec | ⏱ {baseline_breakdown.get('ms_per_token', 0):.2f} ms/token")

        print(f"--- Routed (Matrix-Free) ---")
        print(f"📜 {routed['text']}")
        routed_breakdown = routed["stats"].get_breakdown()
        print(f"⚡ {routed_breakdown.get('tokens_per_sec', 0):.2f} tokens/sec | ⏱ {routed_breakdown.get('ms_per_token', 0):.2f} ms/token")
        print(f"🎯 Accuracy: {routed['accuracy']:.2%}")
        print("-" * 60)
    
    # Final stats
    print("\n📊 OVERALL PERFORMANCE SUMMARY 📊")
    print("-" * 60)
    
    baseline_breakdown = baseline_stats.get_breakdown()
    routed_breakdown = routed_stats.get_breakdown()
    
    print(f"--- GPT-2 Baseline Performance ---")
    print(f"⚡ {baseline_breakdown.get('tokens_per_sec', 0):.2f} tokens/sec | ⏱ {baseline_breakdown.get('ms_per_token', 0):.2f} ms/token")
    print(f"🔍 Transformer: {baseline_breakdown.get('transformer_pct', 0):.1f}% of time")
    print(f"🔍 Misc (incl. LM Head): {baseline_breakdown.get('misc_pct', 0):.1f}% of time")
    
    print(f"\n--- Routed (Matrix-Free) Performance ---")
    print(f"⚡ {routed_breakdown.get('tokens_per_sec', 0):.2f} tokens/sec | ⏱ {routed_breakdown.get('ms_per_token', 0):.2f} ms/token")
    print(f"🔍 Transformer: {routed_breakdown.get('transformer_pct', 0):.1f}% of time")
    print(f"🔍 LM Head: {routed_breakdown.get('lm_head_pct', 0):.1f}% of time")
    print(f"🔍 Router: {routed_breakdown.get('router_pct', 0):.1f}% of time")
    print(f"🔍 Misc: {routed_breakdown.get('misc_pct', 0):.1f}% of time")
    
    print("-" * 60)
    print(f"⚡ Speed ratio: {routed_breakdown.get('tokens_per_sec', 0) / baseline_breakdown.get('tokens_per_sec', 1):.2f}x")
    
    # Calculate theoretical maximum speedup
    if routed_breakdown.get('transformer_pct', 0) > 0:
        theoretical_max = 100 / routed_breakdown.get('transformer_pct', 100)
        print(f"🚀 Theoretical maximum speedup with Transformer bottleneck: {theoretical_max:.2f}x")
    
    return baseline_stats, routed_stats