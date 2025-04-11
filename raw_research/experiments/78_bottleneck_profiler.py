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
from sklearn.decomposition import TruncatedSVD
import json
import math

class SVDRoutedLMHead:
    """SVD-based dimensional reduction routing for next token prediction."""
    
    def __init__(self, model, code_dim=256, top_k=64, alpha=0.001, stabilizer="layernorm", cached_data=None):
        self.model = model
        self.device = model.device
        self.weight = model.lm_head.weight.data.to(self.device)
        self.bias = None  # Llama models typically don't have bias
        
        # Store vocabulary size for analytics
        self.vocab_size = self.weight.shape[0]
        self.embed_dim = self.weight.shape[1]
        self.code_dim = min(code_dim, self.embed_dim)  # Don't exceed original dim
        self.top_k = top_k
        self.alpha = alpha
        self.stabilizer = stabilizer
        
        if cached_data is not None:
            print(f"🔄 Loading cached SVD projections for vocabulary...")
            self.Vh = cached_data['Vh'].to(self.device)
            self.vocab_routing_proj = cached_data['vocab_routing_proj'].to(self.device)
            print(f"✅ SVD-routed head initialized with vocab size: {self.vocab_size}, " 
                  f"embedding dim: {self.embed_dim}, code dim: {self.code_dim}, top_k: {self.top_k}")
        else:
            print(f"🔄 Computing SVD projection for vocabulary...")
            self._compute_svd_projection()
            print(f"✅ SVD-routed head initialized with vocab size: {self.vocab_size}, " 
                  f"embedding dim: {self.embed_dim}, code dim: {self.code_dim}, top_k: {self.top_k}")
    
    def _compute_svd_projection(self):
        """Compute SVD projection matrices for the vocabulary."""
        print(f"🔄 Computing SVD projection for vocabulary ({self.vocab_size:,} tokens)...")
        print(f"Computing SVD decomposition (reducing from {self.embed_dim} to {self.code_dim} dimensions)...")
        
        # Compute SVD directly on GPU using PyTorch
        try:
            # Convert to float32 for SVD computation if needed
            weight_for_svd = self.weight.float() if self.weight.dtype == torch.float16 else self.weight
            
            # Use PyTorch's SVD which can run on GPU
            U, S, Vh = torch.linalg.svd(weight_for_svd, full_matrices=False)
            
            # Take only the top code_dim components
            Vh = Vh[:self.code_dim]
            
            # Convert back to original dtype if needed
            Vh = Vh.to(dtype=self.weight.dtype)
            
            # Store the right singular vectors (Vh) for dimensionality reduction
            self.Vh = Vh
            
            # Precompute the projected vocabulary matrix
            self.vocab_routing_proj = torch.matmul(self.weight, self.Vh.T)
            
            # Report compression rate
            original_size = self.vocab_size * self.embed_dim
            compressed_size = self.vocab_size * self.code_dim + self.code_dim * self.embed_dim
            compression_ratio = original_size / compressed_size
            print(f"📊 SVD compression ratio: {compression_ratio:.2f}x")
            
        except RuntimeError as e:
            print(f"⚠️ GPU computation failed ({str(e)}). Falling back to CPU...")
            
            # Fallback to CPU computation using sklearn's TruncatedSVD
            print("Step 1/3: Moving weights to CPU...")
            weight_cpu = self.weight.cpu().numpy()
            
            print(f"Step 2/3: Computing SVD decomposition...")
            svd = TruncatedSVD(n_components=self.code_dim, random_state=42)
            
            # Create a wrapper to show progress
            total_size = weight_cpu.shape[0]
            with tqdm(total=100, desc="SVD Progress") as pbar:
                last_progress = [0]
                def progress_callback(progress):
                    update = progress - last_progress[0]
                    if update > 0:
                        pbar.update(update)
                        last_progress[0] = progress
                
                # Monkey patch the SVD object to report progress
                original_fit = svd.fit
                def fit_with_progress(X):
                    for i in range(0, 100, 10):
                        progress_callback(i)
                        if i == 0:
                            result = original_fit(X)
                    progress_callback(100)
                    return result
                svd.fit = fit_with_progress
            
            # Perform the SVD
            svd.fit(weight_cpu)
            
            print("Step 3/3: Moving results back to GPU...")
            # Store the right singular vectors (Vh) for dimensionality reduction
            self.Vh = torch.tensor(svd.components_, device=self.device, dtype=self.weight.dtype)
            
            # Precompute the projected vocabulary matrix
            self.vocab_routing_proj = torch.matmul(self.weight, self.Vh.T)
            
            # Report compression rate
            original_size = self.vocab_size * self.embed_dim
            compressed_size = self.vocab_size * self.code_dim + self.code_dim * self.embed_dim
            compression_ratio = original_size / compressed_size
            print(f"📊 SVD compression ratio: {compression_ratio:.2f}x")
    
    def predict(self, hidden_state, measure_time=False, blend_with_standard=False):
        """Predict the next token using SVD-based routing with optional time measurement."""
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # Ensure we're working with the last token's hidden state
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        
        if measure_time:
            extract_time = time.perf_counter()
            times['extract_hidden'] = (extract_time - start) * 1000  # in ms
        
        # Step 1: Project hidden state into routing space
        svd_code = torch.matmul(hidden_state, self.Vh.T)
        
        if measure_time:
            proj_time = time.perf_counter()
            times['svd_projection'] = (proj_time - extract_time) * 1000  # in ms
        
        # Step 2: Dot-product routing for top-k candidates
        sims = torch.matmul(svd_code, self.vocab_routing_proj.T)
        topk_values, topk_indices = torch.topk(sims, self.top_k, dim=-1)
        
        if measure_time:
            routing_time = time.perf_counter()
            times['routing_topk'] = (routing_time - proj_time) * 1000  # in ms
        
        # Step 3: Rerank using original vocabulary
        topk_vectors = self.weight[topk_indices[0]]
        logits_topk = torch.matmul(hidden_state, topk_vectors.T)
        
        # Get the final prediction from top-k candidates
        topk_token_idx = torch.argmax(logits_topk, dim=-1)
        final_token = topk_indices[0][topk_token_idx]
        score = logits_topk[0, topk_token_idx]
        
        if measure_time:
            rerank_time = time.perf_counter()
            times['rerank_logits'] = (rerank_time - routing_time) * 1000  # in ms
        
        # Create dummy logits tensor with the predicted token having a high score
        dummy_logits = torch.zeros((1, 1, self.vocab_size), device=self.device)
        dummy_logits[0, 0, final_token] = torch.tensor(score, dtype=dummy_logits.dtype, device=dummy_logits.device)
        
        # Optional: Blend with standard output for stability
        if blend_with_standard:
            # Standard output (full matmul)
            standard_logits = torch.matmul(hidden_state, self.weight.T)
            
            # Apply stability measure (layernorm or epsilon)
            if self.stabilizer == "layernorm":
                # Apply layer normalization for stabilization
                standard_logits = F.layer_norm(standard_logits, [standard_logits.size(-1)])
                dummy_logits = F.layer_norm(dummy_logits[0], [dummy_logits.size(-1)]).unsqueeze(0)
            else:  # epsilon stabilizer
                eps = 1e-5
                standard_logits = standard_logits + eps
                dummy_logits = dummy_logits + eps
            
            # Blend outputs with alpha
            blended_logits = self.alpha * dummy_logits + (1 - self.alpha) * standard_logits.unsqueeze(1)
            
            # Get final token from blended logits
            final_token = torch.argmax(blended_logits[0, 0])
            score = blended_logits[0, 0, final_token]
            
            # Update dummy logits to show the blended decision
            dummy_logits = torch.zeros((1, 1, self.vocab_size), device=self.device)
            dummy_logits[0, 0, final_token] = score
            
        if measure_time:
            argmax_time = time.perf_counter()
            times['argmax'] = (argmax_time - rerank_time) * 1000  # in ms
            total_time = argmax_time - start
            times['total'] = total_time * 1000  # in ms
        
        return final_token.unsqueeze(0), score.item(), times if measure_time else None


class SVDRoutedTransformerMLP:
    """SVD-based routing for transformer MLP blocks."""
    
    def __init__(self, mlp_block, alpha=0.05, cached_data=None):
        self.original_block = mlp_block
        self.device = next(mlp_block.parameters()).device
        self.alpha = alpha
        
        # Extract weights from MLP block using Llama's attribute names
        self.gate_weight = mlp_block.gate_proj.weight.data.clone()  # [hidden_dim, intermediate_dim]
        self.up_weight = mlp_block.up_proj.weight.data.clone()      # [hidden_dim, intermediate_dim]
        self.down_weight = mlp_block.down_proj.weight.data.clone()  # [intermediate_dim, hidden_dim]
        
        # Store original dtype
        self.orig_dtype = self.gate_weight.dtype
        
        if cached_data is not None:
            print("🔄 Loading cached SVD projections for MLP...")
            # Load cached projections and move to correct device
            self.U_gate = cached_data['U_gate'].to(self.device)
            self.U_up = cached_data['U_up'].to(self.device)
            self.S = cached_data['S'].to(self.device)
            self.Vh = cached_data['Vh'].to(self.device)
            self.U_down = cached_data['U_down'].to(self.device)
            self.S_down = cached_data['S_down'].to(self.device)
            self.Vh_down = cached_data['Vh_down'].to(self.device)
            print(f"✅ Loaded cached SVD projections with shapes: U_gate={self.U_gate.shape}, U_up={self.U_up.shape}, Vh={self.Vh.shape}")
        else:
            print(f"🔄 Computing SVD for MLP block...")
            self._compute_svd()
        
        # Free original weights after initialization
        del self.gate_weight
        del self.up_weight
        del self.down_weight
        torch.cuda.empty_cache()
    
    def _compute_svd(self):
        """Compute SVD decomposition of the MLP weights."""
        try:
            # Process gate and up projections
            gate_weight_float = self.gate_weight.to(torch.float32)
            up_weight_float = self.up_weight.to(torch.float32)
            
            # Compute SVD for combined weight
            combined_weight = torch.cat([gate_weight_float, up_weight_float], dim=0)
            del gate_weight_float, up_weight_float
            torch.cuda.empty_cache()
            
            U, S, Vh = torch.linalg.svd(combined_weight, full_matrices=False)
            del combined_weight
            torch.cuda.empty_cache()
            
            # Split U back into gate and up components
            hidden_dim = self.gate_weight.shape[0]
            self.U_gate = U[:hidden_dim].to(self.orig_dtype)
            self.U_up = U[hidden_dim:].to(self.orig_dtype)
            self.S = S.to(self.orig_dtype)
            self.Vh = Vh.to(self.orig_dtype)
            del U, S, Vh
            torch.cuda.empty_cache()
            
            # Process down projection
            down_weight_float = self.down_weight.to(torch.float32)
            U_down, S_down, Vh_down = torch.linalg.svd(down_weight_float, full_matrices=False)
            del down_weight_float
            torch.cuda.empty_cache()
            
            self.U_down = U_down.to(self.orig_dtype)
            self.S_down = S_down.to(self.orig_dtype)
            self.Vh_down = Vh_down.to(self.orig_dtype)
            del U_down, S_down, Vh_down
            torch.cuda.empty_cache()
            
            print(f"✅ SVD computed for MLP with shapes: U_gate={self.U_gate.shape}, U_up={self.U_up.shape}, Vh={self.Vh.shape}")
            
        except RuntimeError as e:
            print(f"⚠️ GPU SVD computation failed ({str(e)}). Falling back to CPU...")
            
            # Process gate and up projections on CPU
            gate_weight_cpu = self.gate_weight.cpu().to(torch.float32)
            up_weight_cpu = self.up_weight.cpu().to(torch.float32)
            
            # Compute SVD for combined weight
            combined_weight_cpu = torch.cat([gate_weight_cpu, up_weight_cpu], dim=0)
            del gate_weight_cpu, up_weight_cpu
            torch.cuda.empty_cache()
            
            U, S, Vh = torch.linalg.svd(combined_weight_cpu, full_matrices=False)
            del combined_weight_cpu
            torch.cuda.empty_cache()
            
            # Split U back into gate and up components and move to GPU
            hidden_dim = self.gate_weight.shape[0]
            self.U_gate = U[:hidden_dim].to(self.device, dtype=self.orig_dtype)
            self.U_up = U[hidden_dim:].to(self.device, dtype=self.orig_dtype)
            self.S = S.to(self.device, dtype=self.orig_dtype)
            self.Vh = Vh.to(self.device, dtype=self.orig_dtype)
            del U, S, Vh
            torch.cuda.empty_cache()
            
            # Process down projection on CPU
            down_weight_cpu = self.down_weight.cpu().to(torch.float32)
            U_down, S_down, Vh_down = torch.linalg.svd(down_weight_cpu, full_matrices=False)
            del down_weight_cpu
            torch.cuda.empty_cache()
            
            self.U_down = U_down.to(self.device, dtype=self.orig_dtype)
            self.S_down = S_down.to(self.device, dtype=self.orig_dtype)
            self.Vh_down = Vh_down.to(self.device, dtype=self.orig_dtype)
            del U_down, S_down, Vh_down
            torch.cuda.empty_cache()
            
            print(f"✅ SVD computed for MLP with shapes: U_gate={self.U_gate.shape}, U_up={self.U_up.shape}, Vh={self.Vh.shape}")
    
    def forward(self, x, measure_time=False):
        """Forward pass with SVD-based routing and standard MLP blending."""
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # SVD-routed path
        # Project input through gate and up paths
        code = x @ self.Vh  # Project input to SVD space
        code_scaled = code * self.S  # Scale by singular values
        
        # Split into gate and up projections
        gate_hidden = F.silu(code_scaled @ self.U_gate.T)
        up_hidden = code_scaled @ self.U_up.T
        
        # Combine with SwiGLU activation
        routed_hidden = gate_hidden * up_hidden
        
        # Project through down path
        code_down = routed_hidden @ self.Vh_down.T  # transpose
        code_down_scaled = code_down * self.S_down  # Scale by singular values
        routed_out = code_down_scaled @ self.U_down.T  # Project back to hidden space
        
        if measure_time:
            routed_time = time.perf_counter()
            times['routed_path'] = (routed_time - start) * 1000  # in ms
        
        # Standard path (original MLP)
        standard_out = self.original_block(x)
        
        if measure_time:
            standard_time = time.perf_counter()
            times['standard_path'] = (standard_time - routed_time) * 1000  # in ms
        
        # Blend outputs
        blended_out = self.alpha * routed_out + (1 - self.alpha) * standard_out
        
        if measure_time:
            blend_time = time.perf_counter()
            times['blend'] = (blend_time - standard_time) * 1000  # in ms
            times['total'] = (blend_time - start) * 1000  # in mss
        
        return blended_out, times if measure_time else None


class SVDRoutedAttention:
    """SVD-based routing for transformer attention blocks."""
    
    def __init__(self, attn_block, alpha=0.001, stabilizer="layernorm", cached_data=None):
        self.original_block = attn_block
        self.device = next(attn_block.parameters()).device
        self.alpha = alpha
        self.stabilizer = stabilizer
        
        # Extract weights from attention block
        self.q_weight = attn_block.q_proj.weight.data.clone()
        self.k_weight = attn_block.k_proj.weight.data.clone()
        self.v_weight = attn_block.v_proj.weight.data.clone()
        self.o_weight = attn_block.o_proj.weight.data.clone()
        
        # Store original dtype
        self.orig_dtype = self.q_weight.dtype
        
        # Add biases if present
        self.q_bias = getattr(attn_block.q_proj, 'bias', None)
        self.k_bias = getattr(attn_block.k_proj, 'bias', None)
        self.v_bias = getattr(attn_block.v_proj, 'bias', None)
        self.o_bias = getattr(attn_block.o_proj, 'bias', None)
        
        if cached_data is not None:
            print("🔄 Loading cached SVD projections for attention...")
            # Load cached projections and move to correct device
            self.U_q = cached_data['U_q'].to(self.device)
            self.S_q = cached_data['S_q'].to(self.device)
            self.Vh_q = cached_data['Vh_q'].to(self.device)
            self.U_k = cached_data['U_k'].to(self.device)
            self.S_k = cached_data['S_k'].to(self.device)
            self.Vh_k = cached_data['Vh_k'].to(self.device)
            self.U_v = cached_data['U_v'].to(self.device)
            self.S_v = cached_data['S_v'].to(self.device)
            self.Vh_v = cached_data['Vh_v'].to(self.device)
            print(f"✅ Loaded cached SVD projections with shapes: U={self.U_q.shape}, S={self.S_q.shape}, Vh={self.Vh_q.shape}")
        else:
            print(f"🔄 Computing SVD for attention block...")
            self._compute_svd()
        
        # Free original weights after initialization
        del self.q_weight
        del self.k_weight
        del self.v_weight
        torch.cuda.empty_cache()
    
    def _compute_svd(self):
        """Compute SVD decomposition of the attention weights."""
        try:
            # Process one matrix at a time to save memory
            # Query projection
            q_weight_float = self.q_weight.to(torch.float32)
            U_q, S_q, Vh_q = torch.linalg.svd(q_weight_float, full_matrices=False)
            self.U_q = U_q.to(self.orig_dtype)
            self.S_q = S_q.to(self.orig_dtype)
            self.Vh_q = Vh_q.to(self.orig_dtype)
            del q_weight_float, U_q, S_q, Vh_q
            torch.cuda.empty_cache()
            
            # Key projection
            k_weight_float = self.k_weight.to(torch.float32)
            U_k, S_k, Vh_k = torch.linalg.svd(k_weight_float, full_matrices=False)
            self.U_k = U_k.to(self.orig_dtype)
            self.S_k = S_k.to(self.orig_dtype)
            self.Vh_k = Vh_k.to(self.orig_dtype)
            del k_weight_float, U_k, S_k, Vh_k
            torch.cuda.empty_cache()
            
            # Value projection
            v_weight_float = self.v_weight.to(torch.float32)
            U_v, S_v, Vh_v = torch.linalg.svd(v_weight_float, full_matrices=False)
            self.U_v = U_v.to(self.orig_dtype)
            self.S_v = S_v.to(self.orig_dtype)
            self.Vh_v = Vh_v.to(self.orig_dtype)
            del v_weight_float, U_v, S_v, Vh_v
            torch.cuda.empty_cache()
            
            print(f"✅ SVD computed for attention with shapes: U={self.U_q.shape}, S={self.S_q.shape}, Vh={self.Vh_q.shape}")
            
        except RuntimeError as e:
            print(f"⚠️ GPU SVD computation failed ({str(e)}). Falling back to CPU...")
            
            # Process one matrix at a time on CPU
            # Query projection
            q_weight_cpu = self.q_weight.cpu().to(torch.float32)
            U_q, S_q, Vh_q = torch.linalg.svd(q_weight_cpu, full_matrices=False)
            self.U_q = U_q.to(self.device, dtype=self.orig_dtype)
            self.S_q = S_q.to(self.device, dtype=self.orig_dtype)
            self.Vh_q = Vh_q.to(self.device, dtype=self.orig_dtype)
            del q_weight_cpu, U_q, S_q, Vh_q
            torch.cuda.empty_cache()
            
            # Key projection
            k_weight_cpu = self.k_weight.cpu().to(torch.float32)
            U_k, S_k, Vh_k = torch.linalg.svd(k_weight_cpu, full_matrices=False)
            self.U_k = U_k.to(self.device, dtype=self.orig_dtype)
            self.S_k = S_k.to(self.device, dtype=self.orig_dtype)
            self.Vh_k = Vh_k.to(self.device, dtype=self.orig_dtype)
            del k_weight_cpu, U_k, S_k, Vh_k
            torch.cuda.empty_cache()
            
            # Value projection
            v_weight_cpu = self.v_weight.cpu().to(torch.float32)
            U_v, S_v, Vh_v = torch.linalg.svd(v_weight_cpu, full_matrices=False)
            self.U_v = U_v.to(self.device, dtype=self.orig_dtype)
            self.S_v = S_v.to(self.device, dtype=self.orig_dtype)
            self.Vh_v = Vh_v.to(self.device, dtype=self.orig_dtype)
            del v_weight_cpu, U_v, S_v, Vh_v
            torch.cuda.empty_cache()
            
            print(f"✅ SVD computed for attention with shapes: U={self.U_q.shape}, S={self.S_q.shape}, Vh={self.Vh_q.shape}")
    
    def forward(self, hidden_states, attention_mask=None, layer_past=None, position_embeddings=None, measure_time=False):
        """Forward pass with SVD-based routing for attention."""
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # Standard path (original attention)
        # Create a copy of layer_past if it's a tuple to avoid modifying the original
        if isinstance(layer_past, tuple):
            layer_past_copy = layer_past
            layer_past = None
        else:
            layer_past_copy = None
            
        standard_output = self.original_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=layer_past,
        )
        
        if measure_time:
            standard_time = time.perf_counter()
            times['standard_path'] = (standard_time - start) * 1000  # in ms
        
        # SVD-routed path for query, key, value projections
        q_code = hidden_states @ self.Vh_q.T
        q_code_scaled = q_code * self.S_q
        q_routed = q_code_scaled @ self.U_q.T
        
        k_code = hidden_states @ self.Vh_k.T
        k_code_scaled = k_code * self.S_k
        k_routed = k_code_scaled @ self.U_k.T
        
        v_code = hidden_states @ self.Vh_v.T
        v_code_scaled = v_code * self.S_v
        v_routed = v_code_scaled @ self.U_v.T
        
        # Add biases if present
        if self.q_bias is not None:
            q_routed = q_routed + self.q_bias
        if self.k_bias is not None:
            k_routed = k_routed + self.k_bias
        if self.v_bias is not None:
            v_routed = v_routed + self.v_bias
        
        # Compute attention
        attn_output = self._compute_attention(
            q_routed, k_routed, v_routed, 
            attention_mask=attention_mask, 
            layer_past=layer_past_copy,  # Use the copied layer_past
            cos_sin=position_embeddings
        )
        
        # Output projection
        routed_output = attn_output @ self.o_weight.T
        if self.o_bias is not None:
            routed_output = routed_output + self.o_bias
        
        # Handle tuple output
        if isinstance(standard_output, tuple):
            standard_output = standard_output[0]
        
        # Apply stabilization
        if self.stabilizer == "layernorm":
            standard_output = F.layer_norm(standard_output, [standard_output.size(-1)])
            routed_output = F.layer_norm(routed_output, [routed_output.size(-1)])
        
        # Blend outputs
        blended_output = self.alpha * routed_output + (1 - self.alpha) * standard_output
        
        if measure_time:
            blend_time = time.perf_counter()
            times['blend'] = (blend_time - standard_time) * 1000  # in ms
            times['total'] = (blend_time - start) * 1000  # in ms
        
        return blended_output, times if measure_time else None
    
    def _compute_attention(self, query, key, value, attention_mask=None, layer_past=None, cos_sin=None):
        """Compute attention scores and weighted sum."""
        batch_size, seq_len, _ = query.size()

        # Infer dimensions
        hidden_size = key.size(-1)
        hidden_size_query = query.size(-1)

        # Try to retrieve or infer head_dim and num_key_value_heads
        num_key_value_heads = getattr(self.original_block, 'num_key_value_heads', None)
        head_dim = getattr(self.original_block, 'head_dim', None)

        if num_key_value_heads is None or head_dim is None:
            for possible_head_dim in [64, 128, 32]:
                if hidden_size % possible_head_dim == 0:
                    num_key_value_heads = hidden_size // possible_head_dim
                    head_dim = possible_head_dim
                    break
            else:
                raise ValueError(f"Cannot determine num_key_value_heads from hidden_size={hidden_size}")

        # Compute number of query heads
        num_heads = hidden_size_query // head_dim

        # Reshape inputs
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, Hq, S, D]
        key = key.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)  # [B, Hkv, S, D]
        value = value.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)  # [B, Hkv, S, D]

        # Expand key/value heads if fewer than query heads (multi-query attention support)
        if num_key_value_heads != num_heads:
            if num_heads % num_key_value_heads != 0:
                raise ValueError(f"Cannot evenly expand {num_key_value_heads} KV heads to {num_heads} Q heads.")
            repeat_factor = num_heads // num_key_value_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Append past keys/values if present
        if layer_past is not None:
            if isinstance(layer_past, tuple):
                past_key, past_value = layer_past
                # Ensure past states are in the correct shape
                if past_key.dim() == 3:
                    # Reshape 3D past states to 4D
                    past_seq_len = past_key.size(1)
                    past_key = past_key.view(batch_size, num_key_value_heads, past_seq_len, head_dim)
                    past_value = past_value.view(batch_size, num_key_value_heads, past_seq_len, head_dim)
                    
                    # Expand if needed for multi-query attention
                    if num_key_value_heads != num_heads:
                        past_key = past_key.repeat_interleave(repeat_factor, dim=1)
                        past_value = past_value.repeat_interleave(repeat_factor, dim=1)
                
                key = torch.cat((past_key, key), dim=2)
                value = torch.cat((past_value, value), dim=2)

        # Apply rotary embeddings
        if cos_sin is not None:
            cos, sin = cos_sin
            query = self._apply_rotary_pos_emb(query, cos, sin)
            key = self._apply_rotary_pos_emb(key, cos, sin)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)

        # Combine heads and return
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size_query)

        return context

    def _apply_rotary_pos_emb(self, x, cos, sin):
        """
        x: [batch, num_heads, seq_len, head_dim]
        cos/sin: [batch, 1, seq_len, max_rotary_dim]
        """
        # Determine the rotary dimension from x itself
        head_dim = x.size(-1)
        rotary_dim = min(cos.size(-1), head_dim)

        # Split x into rotary and pass-through parts
        x_rotary, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

        # Align cos/sin dimensions to x_rotary
        cos = cos[..., :rotary_dim]
        sin = sin[..., :rotary_dim]

        x1, x2 = x_rotary.chunk(2, dim=-1)
        x_rotated = torch.cat([-x2, x1], dim=-1)

        x_rotated = (x_rotary * cos) + (x_rotated * sin)
        return torch.cat([x_rotated, x_pass], dim=-1)



class SVDRoutedLlamaForCausalLM(LlamaForCausalLM):
    """Custom Llama model with SVD-based routing for attention, MLP, and LM head."""
    
    def __init__(self, config):
        super().__init__(config)
        self.svd_routing_mode = False
        self.lm_head_router = None
        self.attn_routers = []
        self.mlp_routers = []
        
        self.timing_stats = {
            'transformer_time': [],
            'routing_time': [],
            'setup_time': [],
            'total_time': []
        }
    
    def enable_svd_routing(self, 
                          lm_head=True, 
                          attention=True, 
                          mlp=True, 
                          lm_head_config=None,
                          attn_config=None,
                          mlp_config=None,
                          cache_dir="svd_projections"):
        """Enable SVD-based routing with given parameters."""
        self.svd_routing_mode = True
        
        # Default configurations
        default_lm_head_config = {
            'code_dim': 256,
            'top_k': 64,
            'alpha': 0.5,
            'stabilizer': 'layernorm'
        }
        
        default_attn_config = {
            'alpha': 0.001,
            'stabilizer': 'layernorm'
        }
        
        default_mlp_config = {
            'alpha': 0.05
        }
        
        # Apply provided configs or use defaults
        lm_head_config = lm_head_config or default_lm_head_config
        attn_config = attn_config or default_attn_config
        mlp_config = mlp_config or default_mlp_config
        
        # Try to load cached SVD projections
        cached_data = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cached_data = load_svd_projections(cache_dir, self.config._name_or_path)
        
        # Check if we have complete cached data
        have_complete_cache = False
        if cached_data:
            have_complete_cache = True
            if lm_head and 'lm_head' not in cached_data:
                have_complete_cache = False
            if attention:
                for i in range(len(self.model.layers)):
                    if f'attention_{i}' not in cached_data:
                        have_complete_cache = False
                        break
            if mlp:
                for i in range(len(self.model.layers)):
                    if f'mlp_{i}' not in cached_data:
                        have_complete_cache = False
                        break
        
        if have_complete_cache:
            print("🔄 Loading all SVD projections from cache...")
        else:
            print("🔄 Computing SVD projections (this may take a while)...")
            cached_data = {}
        
        # Initialize LM head router
        if lm_head:
            if have_complete_cache:
                print("🔄 Initializing LM head router from cache...")
                self.lm_head_router = SVDRoutedLMHead(
                    self, 
                    code_dim=lm_head_config['code_dim'],
                    top_k=lm_head_config['top_k'],
                    alpha=lm_head_config['alpha'],
                    stabilizer=lm_head_config['stabilizer'],
                    cached_data=cached_data['lm_head']
                )
            else:
                self.lm_head_router = SVDRoutedLMHead(
                    self, 
                    code_dim=lm_head_config['code_dim'],
                    top_k=lm_head_config['top_k'],
                    alpha=lm_head_config['alpha'],
                    stabilizer=lm_head_config['stabilizer']
                )
                cached_data['lm_head'] = {
                    'Vh': self.lm_head_router.Vh,
                    'vocab_routing_proj': self.lm_head_router.vocab_routing_proj
                }
        
        # Initialize attention routers
        if attention:
            self.attn_routers = []
            for i, layer in enumerate(self.model.layers):
                if have_complete_cache:
                    print(f"🔄 Initializing attention router {i} from cache...")
                    router = SVDRoutedAttention(
                        layer.self_attn,
                        alpha=attn_config['alpha'],
                        stabilizer=attn_config['stabilizer'],
                        cached_data=cached_data[f'attention_{i}']
                    )
                else:
                    router = SVDRoutedAttention(
                        layer.self_attn,
                        alpha=attn_config['alpha'],
                        stabilizer=attn_config['stabilizer']
                    )
                    cached_data[f'attention_{i}'] = {
                        'U_q': router.U_q,
                        'S_q': router.S_q,
                        'Vh_q': router.Vh_q,
                        'U_k': router.U_k,
                        'S_k': router.S_k,
                        'Vh_k': router.Vh_k,
                        'U_v': router.U_v,
                        'S_v': router.S_v,
                        'Vh_v': router.Vh_v
                    }
                self.attn_routers.append(router)
                print(f"✅ Initialized attention router for layer {i}")
        
        # Initialize MLP routers
        if mlp:
            self.mlp_routers = []
            for i, layer in enumerate(self.model.layers):
                if have_complete_cache:
                    print(f"🔄 Initializing MLP router {i} from cache...")
                    router = SVDRoutedTransformerMLP(
                        layer.mlp,
                        alpha=mlp_config['alpha'],
                        cached_data=cached_data[f'mlp_{i}']
                    )
                else:
                    router = SVDRoutedTransformerMLP(
                        layer.mlp,
                        alpha=mlp_config['alpha']
                    )
                    cached_data[f'mlp_{i}'] = {
                        'U_gate': router.U_gate,
                        'U_up': router.U_up,
                        'S': router.S,
                        'Vh': router.Vh,
                        'U_down': router.U_down,
                        'S_down': router.S_down,
                        'Vh_down': router.Vh_down
                    }
                self.mlp_routers.append(router)
                print(f"✅ Initialized MLP router for layer {i}")
        
        # Save SVD projections if they weren't loaded from cache
        if cache_dir and not have_complete_cache:
            print("💾 Saving SVD projections to cache...")
            save_svd_projections(cache_dir, self.config._name_or_path, cached_data)
        
        return {
            'lm_head_router': self.lm_head_router,
            'attn_routers': self.attn_routers,
            'mlp_routers': self.mlp_routers
        }
    
    def disable_svd_routing(self):
        """Disable SVD-based routing."""
        self.svd_routing_mode = False
    
    def forward(self, *args, **kwargs):
        """Override forward method for SVD-based routing."""
        measure_time = kwargs.pop('measure_time', False)
        times = {}
        
        if measure_time:
            start = time.perf_counter()
        
        # If not in SVD routing mode, use the standard forward pass
        if not self.svd_routing_mode:
            result = super().forward(*args, **kwargs)
            
            if measure_time:
                end = time.perf_counter()
                times['total'] = (end - start) * 1000  # in ms
                return result, times
            return result
        
        # Store if we need hidden states for later
        output_hidden_states = kwargs.get('output_hidden_states', False)
        kwargs['output_hidden_states'] = True  # We always need hidden states for routing
        
        if measure_time:
            setup_start = time.perf_counter()
        
        # Extract inputs
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
        
        # Process embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.model.layers)
        
        # Generate position IDs if not provided
        if position_ids is None:
            if input_ids is not None:
                # Create position_ids from input_ids
                position_ids = torch.arange(
                    0, input_ids.shape[-1], 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0)
            else:
                # Create position_ids from inputs_embeds
                position_ids = torch.arange(
                    0, inputs_embeds.shape[1], 
                    dtype=torch.long, 
                    device=inputs_embeds.device
                ).unsqueeze(0)
        
        # Compute rotary position embeddings
        # Get rotary embeddings parameters from first attention layer and config
        first_attn = self.model.layers[0].self_attn
        head_dim = first_attn.head_dim
        num_heads = self.config.num_attention_heads
        num_key_value_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        max_seq_len = position_ids.shape[-1]
        
        # Compute rotary embeddings following LLaMA implementation
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float().to(position_ids.device) / head_dim))
        t = position_ids.float().unsqueeze(-1) @ inv_freq.unsqueeze(0)
        
        # Create rotary embeddings
        cos = torch.cos(t).to(inputs_embeds.dtype)
        sin = torch.sin(t).to(inputs_embeds.dtype)
        
        # Duplicate to match head_dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        # Store as single tuple - let the attention block handle the expansion
        position_embeddings = (cos, sin)
        
        # Forward through layers with SVD routing
        hidden_states = inputs_embeds
        new_past_key_values = []
        
        for i, layer in enumerate(self.model.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Apply attention with SVD routing if enabled
            if self.attn_routers and i < len(self.attn_routers):
                # First apply normalization
                norm_hidden_states = layer.input_layernorm(hidden_states)
                
                # Check if we're using the newer KV cache format
                if layer_past is not None and not isinstance(layer_past, tuple):
                    # For newer API versions, use standard attention
                    attention_output = layer.self_attn(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=layer_past,
                        use_cache=use_cache,
                        output_attentions=output_attentions
                    )
                    
                    # Handle output format
                    if isinstance(attention_output, tuple):
                        attn_output = attention_output[0]
                        if use_cache:
                            new_past_key_values.append(attention_output[1])
                    else:
                        attn_output = attention_output
                        if use_cache:
                            new_past_key_values.append(None)
                else:
                    # For older tuple-based API or first generation, use SVD routing
                    attn_output, _ = self.attn_routers[i].forward(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        layer_past=layer_past,
                        position_embeddings=position_embeddings,
                        measure_time=measure_time
                    )
                    
                    # Handle key-value caching
                    if use_cache:
                        # Project current hidden states to get new key and value states
                        new_key = layer.self_attn.k_proj(norm_hidden_states)
                        new_value = layer.self_attn.v_proj(norm_hidden_states)
                        
                        if layer_past is not None and isinstance(layer_past, tuple):
                            # If we have past key-values, concatenate with new ones
                            past_key, past_value = layer_past
                            new_key = torch.cat([past_key, new_key], dim=1)
                            new_value = torch.cat([past_value, new_value], dim=1)
                        
                        # Store as tuple for next iteration
                        new_past_key_values.append((new_key, new_value))
                    else:
                        new_past_key_values.append(None)
                
                # Apply residual connection
                hidden_states = hidden_states + attn_output
            else:
                # Standard attention
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    new_past_key_values.append(layer_outputs[1])
                else:
                    new_past_key_values.append(None)
            
            # Apply MLP with SVD routing if enabled
            if self.mlp_routers and i < len(self.mlp_routers):
                # Apply normalization
                norm_hidden_states = layer.post_attention_layernorm(hidden_states)
                
                # Apply SVD-routed MLP
                mlp_output, _ = self.mlp_routers[i].forward(
                    norm_hidden_states,
                    measure_time=measure_time
                )
                
                # Residual connection
                hidden_states = hidden_states + mlp_output
        
        # Apply final normalization
        hidden_states = self.model.norm(hidden_states)
        
        if measure_time:
            transformer_end = time.perf_counter()
            times['transformer'] = (transformer_end - transformer_start) * 1000  # in ms
            routing_start = time.perf_counter()
        
        # Use SVD router for LM head if enabled
        if self.lm_head_router:
            next_token, score, router_times = self.lm_head_router.predict(
                hidden_states,
                measure_time=measure_time,
                blend_with_standard=True
            )
            
            if measure_time and router_times:
                times['routing'] = router_times
            
            # Create dummy logits tensor with the predicted token
            dummy_logits = torch.zeros((1, 1, self.config.vocab_size), device=self.device)
            dummy_logits[0, 0, next_token] = score
            
            result = CausalLMOutputWithPast(
                loss=None,
                logits=dummy_logits,
                past_key_values=new_past_key_values,
                hidden_states=hidden_states if output_hidden_states else None,
                attentions=None,
            )
        else:
            # Standard LM head
            logits = self.lm_head(hidden_states)
            
            result = CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=new_past_key_values,
                hidden_states=hidden_states if output_hidden_states else None,
                attentions=None,
            )
        
        if measure_time:
            end = time.perf_counter()
            times['total'] = (end - start) * 1000  # in ms
            
            # Store timing stats
            self.timing_stats['transformer_time'].append(times['transformer'])
            self.timing_stats['routing_time'].append(times['routing']['total'] if 'routing' in times else 0)
            self.timing_stats['setup_time'].append(times['setup'])
            self.timing_stats['total_time'].append(times['total'])
            
            return result, times
        
        return result

def save_svd_projections(save_dir, model_name, data_dict):
    """Save SVD projections to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a safe filename from model name
    safe_name = model_name.replace('/', '_').replace('-', '_')
    save_path = os.path.join(save_dir, f"{safe_name}_svd_projections.pt")
    
    # Save tensors
    torch.save(data_dict, save_path)
    
    print(f"✅ Saved SVD projections to {save_path}")
    return save_path

def load_svd_projections(save_dir, model_name):
    """Load SVD projections from disk if they exist."""
    safe_name = model_name.replace('/', '_').replace('-', '_')
    save_path = os.path.join(save_dir, f"{safe_name}_svd_projections.pt")
    
    if os.path.exists(save_path):
        data_dict = torch.load(save_path)
        print(f"✅ Loaded SVD projections from {save_path}")
        return data_dict
    return None

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
        'svd_routed': {
            'transformer': [],
            'routing': {
                'svd_projection': [],
                'routing_topk': [],
                'rerank_logits': [],
                'argmax': [],
                'total': []
            },
            'setup': [],
            'total': []
        }
    }
    
    # Verification data
    standard_tokens = []
    svd_tokens = []
    matched = 0
    
    # Run standard baseline first
    with torch.no_grad():
        baseline_start = time.perf_counter()
        
        for i in range(num_tokens):
            # Standard forward pass with original model
            model.disable_svd_routing()
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
            if 'transformer' in token_times:
                times['standard']['transformer'].append(token_times['transformer'])
        
        baseline_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
        baseline_duration = time.perf_counter() - baseline_start
    
    # Reset for SVD-routed run
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    
    # Enable SVD routing
    model.enable_svd_routing(
        lm_head=True,
        attention=True,
        mlp=True
    )
    
    with torch.no_grad():
        svd_start = time.perf_counter()
        
        for i in range(num_tokens):
            # SVD-routed forward pass
            outputs, token_times = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                measure_time=True
            )
            
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            svd_tokens.append(next_token.item())
            
            # Check accuracy
            if i < len(standard_tokens) and next_token.item() == standard_tokens[i]:
                matched += 1
            
            # Update for next iteration
            past_key_values = outputs.past_key_values
            input_ids = next_token
            all_ids = torch.cat([all_ids, input_ids], dim=1)
            
            # Record detailed timing
            times['svd_routed']['transformer'].append(token_times['transformer'])
            times['svd_routed']['setup'].append(token_times.get('setup', 0))
            
            # Save routing component timings
            if 'routing' in token_times:
                route_times = token_times['routing']
                times['svd_routed']['routing']['svd_projection'].append(route_times.get('svd_projection', 0))
                times['svd_routed']['routing']['routing_topk'].append(route_times.get('routing_topk', 0))
                times['svd_routed']['routing']['rerank_logits'].append(route_times.get('rerank_logits', 0))
                times['svd_routed']['routing']['argmax'].append(route_times.get('argmax', 0))
                times['svd_routed']['routing']['total'].append(route_times.get('total', 0))
            
            times['svd_routed']['total'].append(token_times['total'])
            
        svd_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
        svd_duration = time.perf_counter() - svd_start
    
    # Disable routing when done
    model.disable_svd_routing()
    
    return {
        'standard': {
            'text': baseline_text,
            'tokens_per_sec': num_tokens / baseline_duration,
            'ms_per_token': (baseline_duration / num_tokens) * 1000,
            'times': times['standard']
        },
        'svd_routed': {
            'text': svd_text,
            'tokens_per_sec': num_tokens / svd_duration,
            'ms_per_token': (svd_duration / num_tokens) * 1000,
            'times': times['svd_routed']
        },
        'accuracy': matched / num_tokens if verify_accuracy else None,
        'tokens': {
            'standard': standard_tokens,
            'svd_routed': svd_tokens
        }
    }

def run_detailed_analysis(model_name, max_tokens=50, output_dir='results', code_dim=256, top_k=64):
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
        f.write(f"SVD-Routed Token Generation Analysis\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens per prompt: {max_tokens}\n")
        f.write(f"SVD code dimension: {code_dim}\n")
        f.write(f"Top-k routing: {top_k}\n\n")
    
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
    model = SVDRoutedLlamaForCausalLM.from_pretrained(
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
            f.write(f"  SVD-Routed: {benchmark['svd_routed']['tokens_per_sec']:.2f} tokens/sec, {benchmark['svd_routed']['ms_per_token']:.2f} ms/token\n")
            f.write(f"  Accuracy: {benchmark['accuracy']:.2%}\n")
            f.write(f"  Speedup: {benchmark['svd_routed']['tokens_per_sec'] / benchmark['standard']['tokens_per_sec']:.2f}x\n\n")
            
            # Add time breakdowns
            routing_times = benchmark['svd_routed']['times']['routing']
            if routing_times:
                f.write(f"  SVD-Routing time breakdown (averages):\n")
                f.write(f"    Transformer: {np.mean(benchmark['svd_routed']['times']['transformer']):.2f} ms\n")
                f.write(f"    SVD Projection: {np.mean(routing_times['svd_projection']):.2f} ms\n")
                f.write(f"    Routing Top-k: {np.mean(routing_times['routing_topk']):.2f} ms\n")
                f.write(f"    Rerank Logits: {np.mean(routing_times['rerank_logits']):.2f} ms\n")
                f.write(f"    ArgMax: {np.mean(routing_times['argmax']):.2f} ms\n")
                f.write(f"    Total Routing: {np.mean(routing_times['total']):.2f} ms\n\n")
        
        # Create visualization for this prompt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Token generation time comparison
        axes[0, 0].plot(benchmark['standard']['times']['total'], label='Standard', marker='o')
        axes[0, 0].plot(benchmark['svd_routed']['times']['total'], label='SVD-Routed', marker='x')
        axes[0, 0].set_title('Token Generation Time')
        axes[0, 0].set_xlabel('Token #')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. SVD-routing time breakdown
        if benchmark['svd_routed']['times']['routing']:
            df = pd.DataFrame({
                'SVD Projection': benchmark['svd_routed']['times']['routing']['svd_projection'],
                'Routing Top-k': benchmark['svd_routed']['times']['routing']['routing_topk'],
                'Rerank Logits': benchmark['svd_routed']['times']['routing']['rerank_logits'],
                'ArgMax': benchmark['svd_routed']['times']['routing']['argmax']
            })
            df.plot.area(ax=axes[0, 1], stacked=True)
            axes[0, 1].set_title('SVD-Routing Time Breakdown')
            axes[0, 1].set_xlabel('Token #')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].grid(True)
        
        # 3. Cumulative time
        std_cumulative = np.cumsum(benchmark['standard']['times']['total'])
        svd_cumulative = np.cumsum(benchmark['svd_routed']['times']['total'])
        axes[1, 0].plot(std_cumulative, label='Standard', marker='o')
        axes[1, 0].plot(svd_cumulative, label='SVD-Routed', marker='x')
        axes[1, 0].set_title('Cumulative Generation Time')
        axes[1, 0].set_xlabel('Token #')
        axes[1, 0].set_ylabel('Cumulative Time (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Time ratio
        ratio = np.array(benchmark['svd_routed']['times']['total']) / np.array(benchmark['standard']['times']['total'])
        axes[1, 1].plot(ratio, marker='o', color='purple')
        axes[1, 1].axhline(y=1.0, linestyle='--', color='r')
        axes[1, 1].set_title('SVD-Routed / Standard Time Ratio')
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
    svd_tps = [r['benchmark']['svd_routed']['tokens_per_sec'] for r in results]
    accuracies = [r['benchmark']['accuracy'] for r in results]
    speedups = [svd / std for svd, std in zip(svd_tps, standard_tps)]
    
    # Calculate averages
    avg_standard_tps = np.mean(standard_tps)
    avg_svd_tps = np.mean(svd_tps)
    avg_accuracy = np.mean(accuracies)
    avg_speedup = np.mean(speedups)
    
    # Log summary
    with open(log_file, 'a') as f:
        f.write("\nSUMMARY\n")
        f.write("=======\n\n")
        f.write(f"Average Standard: {avg_standard_tps:.2f} tokens/sec\n")
        f.write(f"Average SVD-Routed: {avg_svd_tps:.2f} tokens/sec\n")
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
    plt.bar(x + width/2, svd_tps, width, label='SVD-Routed')
    
    plt.xlabel('Prompt #')
    plt.ylabel('Tokens per Second')
    plt.title('Generation Speed Comparison')
    plt.xticks(x, [f"Prompt {i+1}" for i in range(len(prompts))])
    plt.legend()
    plt.grid(True, axis='y')
    
    for i, (std, svd) in enumerate(zip(standard_tps, svd_tps)):
        plt.text(i - width/2, std + 0.5, f"{std:.1f}", ha='center')
        plt.text(i + width/2, svd + 0.5, f"{svd:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "speed_comparison.png"))
    plt.close()
    
    # 2. Speedup ratio
    plt.figure(figsize=(14, 6))
    
    plt.bar(x, speedups, color='purple')
    plt.axhline(y=1.0, linestyle='--', color='r')
    
    plt.xlabel('Prompt #')
    plt.ylabel('Speedup Ratio')
    plt.title('SVD-Routed / Standard Speedup Ratio')
    plt.xticks(x, [f"Prompt {i+1}" for i in range(len(prompts))])
    plt.grid(True, axis='y')
    
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup + 0.02, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "speedup_ratio.png"))
    plt.close()
    
    # 3. Time breakdown pie chart for SVD routing
    # Collect all routing component timings
    all_components = {
        'Transformer': [],
        'SVD Projection': [],
        'Routing Top-k': [],
        'Rerank Logits': [],
        'ArgMax': [],
        'Setup': []
    }
    
    for r in results:
        all_components['Transformer'].extend(r['benchmark']['svd_routed']['times']['transformer'])
        all_components['Setup'].extend(r['benchmark']['svd_routed']['times']['setup'])
        
        routing = r['benchmark']['svd_routed']['times']['routing']
        all_components['SVD Projection'].extend(routing['svd_projection'])
        all_components['Routing Top-k'].extend(routing['routing_topk'])
        all_components['Rerank Logits'].extend(routing['rerank_logits'])
        all_components['ArgMax'].extend(routing['argmax'])
    
    # Average each component
    avg_components = {k: np.mean(v) for k, v in all_components.items()}
    
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.pie(
        avg_components.values(),
        labels=avg_components.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999', '#ffcc99', '#99ff99', '#c2c2f0', '#ffb3e6']
    )
    plt.title('SVD-Routed Time Distribution')
    plt.axis('equal')
    
    plt.savefig(os.path.join(run_dir, "time_distribution.png"))
    plt.close()
    
    # 4. Component efficiency chart (transformer vs routing)
    plt.figure(figsize=(10, 6))
    
    # Get average times
    transformer_times = []
    routing_times = []
    for r in results:
        transformer_times.extend(r['benchmark']['svd_routed']['times']['transformer'])
        routing = r['benchmark']['svd_routed']['times']['routing']
        routing_times.extend(routing['total'])
    
    avg_transformer = np.mean(transformer_times)
    avg_routing = np.mean(routing_times)
    other_time = np.mean([t for r in results for t in r['benchmark']['svd_routed']['times']['setup']])
    
    # Create stacked bar chart
    components = ['Standard', 'SVD-Routed']
    standard_breakdown = [avg_standard_tps, 0]  # Placeholder for standard model
    router_breakdown = [0, avg_routing]
    transformer_breakdown = [0, avg_transformer]
    
    plt.bar(components, standard_breakdown, label='Standard Processing')
    plt.bar(components, router_breakdown, bottom=standard_breakdown, label='Routing Processing')
    plt.bar(components, transformer_breakdown, bottom=[a + b for a, b in zip(standard_breakdown, router_breakdown)], 
            label='Transformer Processing')
    
    plt.xlabel('Approach')
    plt.ylabel('Processing Time (ms)')
    plt.title('Processing Time Breakdown')
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "component_efficiency.png"))
    plt.close()
    
    # Print final summary
    print("\n📋 SUMMARY")
    print("==========")
    print(f"Average Standard: {avg_standard_tps:.2f} tokens/sec")
    print(f"Average SVD-Routed: {avg_svd_tps:.2f} tokens/sec")
    print(f"Average Accuracy: {avg_accuracy:.4%}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    # Calculate projected speedup at different scales
    if avg_speedup > 1:
        print(f"💰 Resource savings: {(1 - 1/avg_speedup) * 100:.1f}% reduction in inference time")
        
        # Project scaled performance to larger vocabulary sizes
        print("\n🔮 Projected performance for larger vocabulary sizes:")
        vocab_sizes = [100_000, 250_000, 500_000, 1_000_000]
        for vocab in vocab_sizes:
            scale_factor = vocab / model.config.vocab_size
            # Only LM head scales with vocabulary size while transformer stays constant
            transformer_ratio = np.mean(transformer_times) / (np.mean(transformer_times) + np.mean(routing_times))
            lm_head_ratio = 1 - transformer_ratio
            # Project speedup by scaling only the LM head portion
            proj_speedup = 1 / ((transformer_ratio) + (lm_head_ratio/avg_speedup)*scale_factor)
            print(f"  - {vocab:,} vocab size: {proj_speedup:.2f}x speedup")
    else:
        print(f"⚠️ Overhead: {(1/avg_speedup - 1) * 100:.1f}%")
    
    # Print time breakdown
    print("\n⏱️ SVD-Routing Time Breakdown:")
    total_avg = sum(avg_components.values())
    for component, value in avg_components.items():
        print(f"  {component}: {value:.3f} ms ({value/total_avg*100:.1f}%)")
    
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
    parser = argparse.ArgumentParser(description="Comprehensive SVD-Routed Token Generation Analysis")
    
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate per prompt")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    
    # SVD-routing configurations
    parser.add_argument("--lm_head_code_dim", type=int, default=256, help="Dimensionality of SVD projection for LM head")
    parser.add_argument("--lm_head_top_k", type=int, default=64, help="Number of top-k candidates for reranking in LM head")
    parser.add_argument("--lm_head_alpha", type=float, default=0.5, help="Blending factor for LM head router")
    
    parser.add_argument("--attention_alpha", type=float, default=0.001, help="Blending factor for attention routers")
    parser.add_argument("--attention_stabilizer", type=str, default="layernorm", choices=["layernorm", "epsilon"], 
                       help="Stabilization method for attention routers")
    
    parser.add_argument("--mlp_alpha", type=float, default=0.05, help="Blending factor for MLP routers")
    
    # Component selection
    parser.add_argument("--route_lm_head", action="store_true", help="Apply routing to LM head")
    parser.add_argument("--route_attention", action="store_true", help="Apply routing to attention layers")
    parser.add_argument("--route_mlp", action="store_true", help="Apply routing to MLP layers")
    
    args = parser.parse_args()
    
    # Configure routing parameters
    lm_head_config = {
        'code_dim': args.lm_head_code_dim,
        'top_k': args.lm_head_top_k,
        'alpha': args.lm_head_alpha,
        'stabilizer': 'layernorm'
    }
    
    attn_config = {
        'alpha': args.attention_alpha,
        'stabilizer': args.attention_stabilizer
    }
    
    mlp_config = {
        'alpha': args.mlp_alpha
    }
    
    # Run detailed analysis with specified routing components
    run_detailed_analysis(
        model_name=args.model,
        max_tokens=args.tokens,
        output_dir=args.output,
        code_dim=args.lm_head_code_dim,
        top_k=args.lm_head_top_k
    )


""" Output:
SVD-Routed Token Generation Analysis
Model: unsloth/Llama-3.1-8B-Instruct
Date: 2025-04-05 15:31:52
Tokens per prompt: 50
SVD code dimension: 256
Top-k routing: 64

Device: cuda
CUDA Device: NVIDIA H100 80GB HBM3
CUDA Memory: 85.05 GB

Model configuration:
  Vocab size: 128256
  Hidden size: 4096
  Intermediate size: 14336
  Num hidden layers: 32
  Num attention heads: 32
  Precision: torch.float16

Prompt: 'The meaning of life is'
  Standard: 38.81 tokens/sec, 25.77 ms/token
  SVD-Routed: 24.45 tokens/sec, 40.90 ms/token
  Accuracy: 0.00%
  Speedup: 0.63x

  SVD-Routing time breakdown (averages):
    Transformer: 38.95 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.47 ms
    Rerank Logits: 0.30 ms
    ArgMax: 0.89 ms
    Total Routing: 1.69 ms

Prompt: 'In a distant galaxy, a civilization'
  Standard: 57.59 tokens/sec, 17.36 ms/token
  SVD-Routed: 24.69 tokens/sec, 40.50 ms/token
  Accuracy: 0.00%
  Speedup: 0.43x

  SVD-Routing time breakdown (averages):
    Transformer: 39.17 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.88 ms
    Total Routing: 1.06 ms

Prompt: 'The future of AI will depend on'
  Standard: 53.78 tokens/sec, 18.59 ms/token
  SVD-Routed: 24.43 tokens/sec, 40.94 ms/token
  Accuracy: 0.00%
  Speedup: 0.45x

  SVD-Routing time breakdown (averages):
    Transformer: 39.56 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.91 ms
    Total Routing: 1.08 ms

Prompt: 'Once upon a time there was'
  Standard: 55.45 tokens/sec, 18.03 ms/token
  SVD-Routed: 25.14 tokens/sec, 39.78 ms/token
  Accuracy: 0.00%
  Speedup: 0.45x

  SVD-Routing time breakdown (averages):
    Transformer: 38.43 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.91 ms
    Total Routing: 1.08 ms

Prompt: 'The quantum computer revolutionized'
  Standard: 56.55 tokens/sec, 17.69 ms/token
  SVD-Routed: 24.74 tokens/sec, 40.42 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 39.02 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.09 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.91 ms
    Total Routing: 1.11 ms

Prompt: 'Scientists have discovered a new'
  Standard: 57.39 tokens/sec, 17.42 ms/token
  SVD-Routed: 25.34 tokens/sec, 39.47 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 38.11 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.91 ms
    Total Routing: 1.08 ms

Prompt: 'The theory of relativity explains'
  Standard: 57.33 tokens/sec, 17.44 ms/token
  SVD-Routed: 25.35 tokens/sec, 39.44 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 38.10 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.89 ms
    Total Routing: 1.06 ms

Prompt: 'Climate change has led to'
  Standard: 57.22 tokens/sec, 17.47 ms/token
  SVD-Routed: 25.10 tokens/sec, 39.84 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 38.46 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.91 ms
    Total Routing: 1.08 ms

Prompt: 'Machine learning algorithms can'
  Standard: 57.33 tokens/sec, 17.44 ms/token
  SVD-Routed: 25.08 tokens/sec, 39.87 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 38.55 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.89 ms
    Total Routing: 1.06 ms

Prompt: 'The human brain processes information'
  Standard: 57.32 tokens/sec, 17.45 ms/token
  SVD-Routed: 25.00 tokens/sec, 40.01 ms/token
  Accuracy: 0.00%
  Speedup: 0.44x

  SVD-Routing time breakdown (averages):
    Transformer: 38.65 ms
    SVD Projection: 0.01 ms
    Routing Top-k: 0.08 ms
    Rerank Logits: 0.07 ms
    ArgMax: 0.92 ms
    Total Routing: 1.09 ms


SUMMARY
=======

Average Standard: 54.88 tokens/sec
Average SVD-Routed: 24.93 tokens/sec
Average Accuracy: 0.0000%
Average Speedup: 0.46x
Overhead: 117.4%

"""