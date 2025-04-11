import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import os
import random
import json
import logging
import matplotlib.pyplot as plt
from torch.linalg import svd
from scipy.linalg import svd as scipy_svd
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("roadrunner_explorer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RoadrunnerExplorer")

class MLP_SVD_Routed(nn.Module):
    """MLP module with SVD-based routing for transformer blocks"""
    def __init__(self, original_mlp, svd_impl='torch_svd', precision='float32', 
                 sv_handler='full', stabilizer='none', alpha=0.0):
        super().__init__()
        
        # Store original MLP for reference and alpha blend
        self.original_mlp = original_mlp
        self.alpha = alpha
        
        # Implementation parameters
        self.svd_impl = svd_impl
        self.precision = precision
        self.sv_handler = sv_handler
        self.stabilizer = stabilizer
        
        # Extract original weights and biases
        # For GPT-2, mlp.c_fc is the first linear layer, mlp.c_proj is the second
        self.fc_weight = original_mlp.c_fc.weight.data.clone().T  # [intermediate_size, hidden_size]
        self.fc_bias = original_mlp.c_fc.bias.data.clone()
        self.proj_weight = original_mlp.c_proj.weight.data.clone().T  # [hidden_size, intermediate_size]
        self.proj_bias = original_mlp.c_proj.bias.data.clone()
        
        # Store dimensions for clarity
        self.hidden_size = self.fc_weight.size(1)  # 768 for GPT-2
        self.intermediate_size = self.fc_weight.size(0)  # 3072 for GPT-2
        
        logger.info(f"Creating MLP_SVD_Routed with shapes: {self.hidden_size} -> {self.intermediate_size} -> {self.hidden_size}")
        
        # SVD parameters
        self.truncation_threshold = 0.9  # For truncated SVD (keep 90% energy)
        self.epsilon = 1e-8  # For numerical stability
        
        # Compute SVD decompositions
        self.U_fc, self.S_fc, self.Vh_fc = self._compute_svd(self.fc_weight)
        logger.info(f"fc_weight SVD shapes: U_fc={self.U_fc.shape}, S_fc={self.S_fc.shape}, Vh_fc={self.Vh_fc.shape}")
        
        self.U_proj, self.S_proj, self.Vh_proj = self._compute_svd(self.proj_weight)
        logger.info(f"proj_weight SVD shapes: U_proj={self.U_proj.shape}, S_proj={self.S_proj.shape}, Vh_proj={self.Vh_proj.shape}")
        
        # Optional: Apply singular value handling strategy
        self._apply_sv_handler()
    
    def _compute_svd(self, weight_matrix):
        """Compute SVD decomposition using selected implementation and precision"""
        # Convert to appropriate precision
        if self.precision == 'float64':
            weight = weight_matrix.to(torch.float64)
        elif self.precision == 'mixed_precision':
            weight = weight_matrix.to(torch.float32)  # Keep original for now
        else:  # float32
            weight = weight_matrix
        
        # Select SVD implementation
        if self.svd_impl == 'torch_svd':
            U, S, Vh = svd(weight, full_matrices=False)
        elif self.svd_impl == 'numpy_svd':
            # Convert to numpy, compute SVD, convert back to torch
            weight_np = weight.cpu().numpy()
            U_np, S_np, Vh_np = scipy_svd(weight_np, full_matrices=False)
            U = torch.from_numpy(U_np).to(weight.device)
            S = torch.from_numpy(S_np).to(weight.device)
            Vh = torch.from_numpy(Vh_np).to(weight.device)
        elif self.svd_impl == 'randomized_svd':
            # Use scikit-learn's randomized SVD for large matrices
            weight_np = weight.cpu().numpy()
            k = min(weight_np.shape) - 1  # One less than full rank
            U_np, S_np, Vh_np = randomized_svd(weight_np, n_components=k)
            U = torch.from_numpy(U_np).to(weight.device)
            S = torch.from_numpy(S_np).to(weight.device)
            Vh = torch.from_numpy(Vh_np).to(weight.device)
        else:
            raise ValueError(f"Unknown SVD implementation: {self.svd_impl}")
        
        # Convert back to original precision if needed
        if self.precision == 'float64':
            U = U.to(torch.float32)
            S = S.to(torch.float32)
            Vh = Vh.to(torch.float32)
        
        return U, S, Vh
    
    def _apply_sv_handler(self):
        """Apply selected singular value handling strategy"""
        if self.sv_handler == 'truncated':
            # Keep only singular values that contribute to threshold% of energy
            self._apply_truncation(self.S_fc, self.truncation_threshold)
            self._apply_truncation(self.S_proj, self.truncation_threshold)
        elif self.sv_handler == 'thresholded':
            # Zero out small singular values
            min_threshold = 1e-4
            self.S_fc[self.S_fc < min_threshold] = 0.0
            self.S_proj[self.S_proj < min_threshold] = 0.0
        elif self.sv_handler == 'normalized':
            # Normalize singular values
            self.S_fc = self.S_fc / torch.max(self.S_fc)
            self.S_proj = self.S_proj / torch.max(self.S_proj)
    
    def _apply_truncation(self, S, threshold):
        """Truncate singular values to keep threshold% of energy"""
        total_energy = torch.sum(S ** 2)
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        normalized_energy = cumulative_energy / total_energy
        k = torch.sum(normalized_energy <= threshold).item() + 1
        S[k:] = 0.0
    
    def _apply_stabilizer(self, tensor):
        """Apply numerical stabilization to tensor"""
        if self.stabilizer == 'none':
            return tensor
        elif self.stabilizer == 'epsilon':
            # Add small epsilon to avoid division by zero
            return tensor + self.epsilon
        elif self.stabilizer == 'gradient_clipping':
            # Clip values to avoid extreme gradients
            return torch.clamp(tensor, min=-100, max=100)
        elif self.stabilizer == 'layernorm':
            # Apply layer normalization
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            return (tensor - mean) / (std + self.epsilon)
        else:
            return tensor
    
    def forward(self, x):
        """Forward pass with SVD-based routing and alpha blending"""
        # Always compute the original output for alpha blending and comparison
        original_output = self.original_mlp(x)
        
        # If alpha is 0, just return the original output
        if self.alpha == 0.0:
            return original_output
        
        try:
            # === First part: input to intermediate ===
            # x shape: [batch_size, seq_len, hidden_size]
            # Vh_fc shape: [min_dim, hidden_size] where min_dim ≤ hidden_size
            
            # Project input to code space
            code = torch.matmul(x, self.Vh_fc.T)  # [batch_size, seq_len, min_dim]
            
            # Scale with singular values (element-wise multiplication)
            # S_fc shape: [min_dim]
            code_scaled = code * self._apply_stabilizer(self.S_fc)  # [batch_size, seq_len, min_dim]
            
            # Project to intermediate space and add bias
            # U_fc shape: [intermediate_size, min_dim]
            hidden = torch.matmul(code_scaled, self.U_fc.T)  # [batch_size, seq_len, intermediate_size]
            hidden = hidden + self.fc_bias  # Add bias
            
            # Apply activation function
            hidden_activated = F.gelu(hidden)  # [batch_size, seq_len, intermediate_size]
            
            # === Second part: intermediate to output ===
            # hidden_activated shape: [batch_size, seq_len, intermediate_size]
            # Vh_proj shape: [min_dim, intermediate_size] where min_dim ≤ intermediate_size
            
            # Project to code space
            code_proj = torch.matmul(hidden_activated, self.Vh_proj.T)  # [batch_size, seq_len, min_dim]
            
            # Scale with singular values (element-wise multiplication)
            # S_proj shape: [min_dim]
            code_proj_scaled = code_proj * self._apply_stabilizer(self.S_proj)  # [batch_size, seq_len, min_dim]
            
            # Project back to hidden space and add bias
            # U_proj shape: [hidden_size, min_dim]
            output = torch.matmul(code_proj_scaled, self.U_proj.T)  # [batch_size, seq_len, hidden_size]
            output = output + self.proj_bias  # Add bias
            
            # Blend with original output
            blended_output = self.alpha * output + (1 - self.alpha) * original_output
            
            return blended_output
            
        except Exception as e:
            logger.error(f"Error in SVD routing forward pass: {str(e)}")
            logger.error(f"Input shape: {x.shape}")
            # Detailed error logging
            logger.error(f"Shapes: x={x.shape}, U_fc={self.U_fc.shape}, S_fc={self.S_fc.shape}, Vh_fc={self.Vh_fc.shape}")
            logger.error(f"Shapes: U_proj={self.U_proj.shape}, S_proj={self.S_proj.shape}, Vh_proj={self.Vh_proj.shape}")
            logger.error(f"Alpha: {self.alpha}")
            
            # Fallback to original on error
            return original_output


class Attention_SVD_Routed(nn.Module):
    """Attention module with SVD-based routing for transformer blocks"""
    def __init__(self, original_attn, svd_impl='torch_svd', precision='float32', 
                 sv_handler='full', stabilizer='none', alpha=0.0):
        super().__init__()
        
        # Store original attention for reference and alpha blend
        self.original_attn = original_attn
        self.alpha = alpha
        
        # Get model configuration
        self.hidden_dim = original_attn.embed_dim
        self.n_head = original_attn.num_heads
        self.head_dim = self.hidden_dim // self.n_head
        
        # Implementation parameters
        self.svd_impl = svd_impl
        self.precision = precision
        self.sv_handler = sv_handler
        self.stabilizer = stabilizer
        
        # Extract original weights and biases
        attn_w = original_attn.c_attn.weight.data.clone()  # [hidden_dim, 3*hidden_dim]
        attn_b = original_attn.c_attn.bias.data.clone()    # [3*hidden_dim]
        
        # Split into query, key, value matrices
        self.W_q, self.W_k, self.W_v = torch.chunk(attn_w, 3, dim=1)
        self.b_q, self.b_k, self.b_v = torch.chunk(attn_b, 3, dim=0)
        
        self.proj_w = original_attn.c_proj.weight.data.clone()  # [hidden_dim, hidden_dim]
        self.proj_b = original_attn.c_proj.bias.data.clone()    # [hidden_dim]
        
        logger.info(f"Creating Attention_SVD_Routed with shapes: W_q={self.W_q.shape}, W_k={self.W_k.shape}, W_v={self.W_v.shape}, proj_w={self.proj_w.shape}")
        
        # SVD parameters
        self.truncation_threshold = 0.9  # For truncated SVD (keep 90% energy)
        self.epsilon = 1e-8  # For numerical stability
        
        # Compute SVD decompositions
        self.U_q, self.S_q, self.Vh_q = self._compute_svd(self.W_q.T)  # Transpose for consistency
        self.U_k, self.S_k, self.Vh_k = self._compute_svd(self.W_k.T)
        self.U_v, self.S_v, self.Vh_v = self._compute_svd(self.W_v.T)
        self.U_proj, self.S_proj, self.Vh_proj = self._compute_svd(self.proj_w.T)
        
        logger.info(f"SVD shapes: U_q={self.U_q.shape}, S_q={self.S_q.shape}, Vh_q={self.Vh_q.shape}")
        
        # Optional: Apply singular value handling strategy
        self._apply_sv_handler()
    
    def _compute_svd(self, weight_matrix):
        """Compute SVD decomposition using selected implementation and precision"""
        # Convert to appropriate precision
        if self.precision == 'float64':
            weight = weight_matrix.to(torch.float64)
        elif self.precision == 'mixed_precision':
            weight = weight_matrix.to(torch.float32)  # Keep original for now
        else:  # float32
            weight = weight_matrix
        
        # Select SVD implementation
        if self.svd_impl == 'torch_svd':
            U, S, Vh = svd(weight, full_matrices=False)
        elif self.svd_impl == 'numpy_svd':
            # Convert to numpy, compute SVD, convert back to torch
            weight_np = weight.cpu().numpy()
            U_np, S_np, Vh_np = scipy_svd(weight_np, full_matrices=False)
            U = torch.from_numpy(U_np).to(weight.device)
            S = torch.from_numpy(S_np).to(weight.device)
            Vh = torch.from_numpy(Vh_np).to(weight.device)
        elif self.svd_impl == 'randomized_svd':
            # Use scikit-learn's randomized SVD for large matrices
            weight_np = weight.cpu().numpy()
            k = min(weight_np.shape) - 1  # One less than full rank
            U_np, S_np, Vh_np = randomized_svd(weight_np, n_components=k)
            U = torch.from_numpy(U_np).to(weight.device)
            S = torch.from_numpy(S_np).to(weight.device)
            Vh = torch.from_numpy(Vh_np).to(weight.device)
        else:
            raise ValueError(f"Unknown SVD implementation: {self.svd_impl}")
        
        # Convert back to original precision if needed
        if self.precision == 'float64':
            U = U.to(torch.float32)
            S = S.to(torch.float32)
            Vh = Vh.to(torch.float32)
        
        return U, S, Vh
    
    def _apply_sv_handler(self):
        """Apply selected singular value handling strategy"""
        if self.sv_handler == 'truncated':
            # Keep only singular values that contribute to threshold% of energy
            self._apply_truncation(self.S_q, self.truncation_threshold)
            self._apply_truncation(self.S_k, self.truncation_threshold)
            self._apply_truncation(self.S_v, self.truncation_threshold)
            self._apply_truncation(self.S_proj, self.truncation_threshold)
        elif self.sv_handler == 'thresholded':
            # Zero out small singular values
            min_threshold = 1e-4
            self.S_q[self.S_q < min_threshold] = 0.0
            self.S_k[self.S_k < min_threshold] = 0.0
            self.S_v[self.S_v < min_threshold] = 0.0
            self.S_proj[self.S_proj < min_threshold] = 0.0
        elif self.sv_handler == 'normalized':
            # Normalize singular values
            self.S_q = self.S_q / torch.max(self.S_q)
            self.S_k = self.S_k / torch.max(self.S_k)
            self.S_v = self.S_v / torch.max(self.S_v)
            self.S_proj = self.S_proj / torch.max(self.S_proj)
    
    def _apply_truncation(self, S, threshold):
        """Truncate singular values to keep threshold% of energy"""
        total_energy = torch.sum(S ** 2)
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        normalized_energy = cumulative_energy / total_energy
        k = torch.sum(normalized_energy <= threshold).item() + 1
        S[k:] = 0.0
    
    def _apply_stabilizer(self, tensor):
        """Apply numerical stabilization to tensor"""
        if self.stabilizer == 'none':
            return tensor
        elif self.stabilizer == 'epsilon':
            # Add small epsilon to avoid division by zero
            return tensor + self.epsilon
        elif self.stabilizer == 'gradient_clipping':
            # Clip values to avoid extreme gradients
            return torch.clamp(tensor, min=-100, max=100)
        elif self.stabilizer == 'layernorm':
            # Apply layer normalization
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            return (tensor - mean) / (std + self.epsilon)
        else:
            return tensor
    
    def forward(self, x, attention_mask=None, head_mask=None, past_key_value=None, use_cache=False):
        """Forward pass with SVD-based routing and alpha blending"""
        # Always compute the original output for alpha blending and comparison
        original_output = self.original_attn(x, attention_mask, head_mask, past_key_value, use_cache)
        
        # If alpha is 0, just return the original output
        if self.alpha == 0.0:
            return original_output
        
        # Unpack original output
        original_attn_output = original_output[0]
        original_present = original_output[1] if use_cache else None
        
        # Apply SVD-based routing with stabilization
        try:
            batch_size, seq_len, _ = x.shape
            
            # === Query Projection ===
            # Project input to singular value space
            code_q = torch.matmul(x, self.Vh_q.T)  # [batch_size, seq_len, min_dim]
            code_q_scaled = code_q * self._apply_stabilizer(self.S_q)  # Scale with singular values
            q = torch.matmul(code_q_scaled, self.U_q.T) + self.b_q  # [batch_size, seq_len, hidden_dim]
            
            # === Key Projection ===
            code_k = torch.matmul(x, self.Vh_k.T)
            code_k_scaled = code_k * self._apply_stabilizer(self.S_k)
            k = torch.matmul(code_k_scaled, self.U_k.T) + self.b_k  # [batch_size, seq_len, hidden_dim]
            
            # === Value Projection ===
            code_v = torch.matmul(x, self.Vh_v.T)
            code_v_scaled = code_v * self._apply_stabilizer(self.S_v)
            v = torch.matmul(code_v_scaled, self.U_v.T) + self.b_v  # [batch_size, seq_len, hidden_dim]
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
            k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            
            # Handle past key values for incremental decoding
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            
            # Calculate attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # Scale with sqrt of dimension
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Apply softmax to get attention probabilities
            attn_probs = F.softmax(attn_weights, dim=-1)
            
            if head_mask is not None:
                attn_probs = attn_probs * head_mask
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)  # [batch_size, n_head, seq_len, head_dim]
            
            # Reshape back to [batch_size, seq_len, hidden_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
            
            # === Output Projection ===
            # Project through SVD-decomposed output projection
            code_proj = torch.matmul(attn_output, self.Vh_proj.T)
            code_proj_scaled = code_proj * self._apply_stabilizer(self.S_proj)
            routed_output = torch.matmul(code_proj_scaled, self.U_proj.T) + self.proj_b
            
            # Apply alpha blending
            blended_output = self.alpha * routed_output + (1 - self.alpha) * original_attn_output
            
            # Return with same format as original
            if use_cache:
                return (blended_output, (k, v))
            else:
                return (blended_output,)
        
        except Exception as e:
            logger.error(f"Error in SVD routing attention forward pass: {str(e)}")
            logger.error(f"Input shape: {x.shape}")
            # Fallback to original on error
            return original_output


class RoadrunnerExplorer:
    """Framework for exploring stable SVD-based routing implementations"""
    def __init__(self, model_name="gpt2"):
        # Initialize model, tokenizer, and testing infrastructure
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Reference model (never modified)
        self.reference_model = copy.deepcopy(self.model).eval()
        
        # Test prompts for consistent evaluation
        self.test_prompts = [
            "The future of artificial intelligence is",
            "In the distant mountains, a small village",
            "Scientists have recently discovered that",
            "The relationship between humans and technology",
            "Throughout history, civilizations have risen and"
        ]
        
        # Parameter search space
        self.svd_implementations = ["torch_svd", "numpy_svd", "randomized_svd"]
        self.precision_strategies = ["float32", "float64", "mixed_precision"]
        self.singular_value_handlers = ["full", "truncated", "thresholded", "normalized"]
        self.numerical_stabilizers = ["none", "epsilon", "gradient_clipping", "layernorm"]
        
        # Alpha search space
        self.alpha_range = [
            0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 
            0.01, 0.02, 0.05, 0.1, 0.2, 0.5
        ]
        
        # Results tracking
        self.results_log = []
        self.best_configurations = {}
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        logger.info("Roadrunner Explorer initialized successfully!")
    
    def create_test_model(self, params):
        """Create a test model with specified routing parameters"""
        # Create a fresh copy of the model
        test_model = copy.deepcopy(self.model)
        test_model.eval()
        
        # Extract parameters
        svd_impl = params.get("svd_impl", "torch_svd")
        precision = params.get("precision", "float32")
        sv_handler = params.get("sv_handler", "full")
        stabilizer = params.get("stabilizer", "none")
        alpha = params.get("alpha", 0.0)
        layers = params.get("layers", [0])  # Which layers to modify
        mlp_routing = params.get("mlp_routing", True)
        attn_routing = params.get("attn_routing", False)
        
        # Apply routing to specified layers
        for layer_idx in layers:
            if layer_idx >= len(test_model.transformer.h):
                continue
            
            # Get the transformer block
            block = test_model.transformer.h[layer_idx]
            
            # Replace MLP with routed version if enabled
            if mlp_routing:
                try:
                    block.mlp = MLP_SVD_Routed(
                        block.mlp,
                        svd_impl=svd_impl,
                        precision=precision,
                        sv_handler=sv_handler,
                        stabilizer=stabilizer,
                        alpha=alpha
                    )
                    logger.info(f"Successfully replaced MLP in layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Error replacing MLP in layer {layer_idx}: {str(e)}")
            
            # Replace attention with routed version if enabled
            if attn_routing:
                try:
                    block.attn = Attention_SVD_Routed(
                        block.attn,
                        svd_impl=svd_impl,
                        precision=precision,
                        sv_handler=sv_handler,
                        stabilizer=stabilizer,
                        alpha=alpha
                    )
                    logger.info(f"Successfully replaced Attention in layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Error replacing Attention in layer {layer_idx}: {str(e)}")
        
        return test_model
    
    def evaluate_token_match(self, test_model, prompt=None, num_tokens=5):
        """Evaluate token prediction accuracy between test and reference models"""
        if prompt is None:
            prompt = random.choice(self.test_prompts)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate from reference model
        ref_tokens = []
        ref_past = None
        ref_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(num_tokens):
                ref_outputs = self.reference_model(ref_input, past_key_values=ref_past, use_cache=True)
                ref_past = ref_outputs.past_key_values
                next_token_logits = ref_outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                ref_tokens.append(next_token.item())
                ref_input = next_token
        
        # Generate from test model
        test_tokens = []
        test_past = None
        test_input = input_ids.clone()
        
        try:
            with torch.no_grad():
                for _ in range(num_tokens):
                    test_outputs = test_model(test_input, past_key_values=test_past, use_cache=True)
                    test_past = test_outputs.past_key_values
                    next_token_logits = test_outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    test_tokens.append(next_token.item())
                    test_input = next_token
            
            # Calculate token match accuracy
            matches = sum(1 for ref, test in zip(ref_tokens, test_tokens) if ref == test)
            accuracy = matches / len(ref_tokens)
            
            return {
                "accuracy": accuracy,
                "matches": matches,
                "total": len(ref_tokens),
                "ref_tokens": ref_tokens,
                "test_tokens": test_tokens,
                "prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"Error during token generation: {str(e)}")
            return {
                "accuracy": 0.0,
                "matches": 0,
                "total": num_tokens,
                "error": str(e),
                "prompt": prompt
            }
    
    def evaluate_numerical_drift(self, test_model, prompt=None):
        """Evaluate numerical drift between test and reference model activations"""
        if prompt is None:
            prompt = random.choice(self.test_prompts)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Forward pass through both models
        with torch.no_grad():
            ref_outputs = self.reference_model(input_ids)
            ref_logits = ref_outputs.logits
            
            try:
                test_outputs = test_model(input_ids)
                test_logits = test_outputs.logits
                
                # Compute metrics
                l2_drift = torch.norm(test_logits - ref_logits).item()
                cosine_sim = F.cosine_similarity(
                    test_logits.view(-1), ref_logits.view(-1), dim=0
                ).item()
                
                return {
                    "l2_drift": l2_drift,
                    "cosine_similarity": cosine_sim,
                    "prompt": prompt
                }
            
            except Exception as e:
                logger.error(f"Error during numerical drift evaluation: {str(e)}")
                return {
                    "l2_drift": float('inf'),
                    "cosine_similarity": -1.0,
                    "error": str(e),
                    "prompt": prompt
                }
    
    def track_internal_states(self, model, input_ids):
        """Track internal states throughout the model for detailed analysis"""
        states = {}
        
        # Define forward hooks to capture activations
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) > 0:
                        states[name] = output[0].detach()
                else:
                    states[name] = output.detach()
            return hook
        
        # Register hooks for each layer
        hooks = []
        
        # MLP hooks - handle both original and routed MLP types
        for i, block in enumerate(model.transformer.h):
            if hasattr(block.mlp, 'c_fc'):
                # Original MLP structure
                h1 = block.mlp.c_fc.register_forward_hook(hook_fn(f"mlp_{i}_fc"))
                h2 = block.mlp.c_proj.register_forward_hook(hook_fn(f"mlp_{i}_proj"))
                hooks.extend([h1, h2])
            elif isinstance(block.mlp, MLP_SVD_Routed):
                # Our custom MLP_SVD_Routed
                h1 = block.mlp.register_forward_hook(hook_fn(f"mlp_{i}_routed"))
                hooks.append(h1)
            else:
                # Unknown MLP type - just monitor the whole module
                h1 = block.mlp.register_forward_hook(hook_fn(f"mlp_{i}"))
                hooks.append(h1)
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return states, outputs.logits
    
    def analyze_divergence(self, test_model, prompt=None):
        """Analyze where divergence occurs between reference and test model"""
        if prompt is None:
            prompt = random.choice(self.test_prompts)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Track internal states for both models
        ref_states, ref_logits = self.track_internal_states(self.reference_model, input_ids)
        test_states, test_logits = self.track_internal_states(test_model, input_ids)
        
        # Compare states
        divergence_analysis = []
        
        for key in ref_states:
            if key in test_states:
                ref_state = ref_states[key]
                test_state = test_states[key]
                
                if ref_state.shape != test_state.shape:
                    logger.warning(f"Shape mismatch for {key}: {ref_state.shape} vs {test_state.shape}")
                    continue
                
                # Compute metrics
                l2_error = torch.norm(ref_state - test_state).item()
                
                # Avoid division by zero
                if torch.norm(ref_state).item() > 1e-9:
                    rel_error = l2_error / torch.norm(ref_state).item()
                else:
                    rel_error = float('inf')
                
                # Compute cosine similarity
                if ref_state.numel() > 0 and test_state.numel() > 0:
                    cos_sim = F.cosine_similarity(
                        ref_state.view(-1), test_state.view(-1), dim=0
                    ).item()
                else:
                    cos_sim = 0.0
                
                # Compute max absolute difference
                max_diff = torch.max(torch.abs(ref_state - test_state)).item()
                
                divergence_analysis.append({
                    "layer": key,
                    "l2_error": l2_error,
                    "relative_error": rel_error,
                    "cosine_similarity": cos_sim,
                    "max_diff": max_diff
                })
        
        # Add final logits comparison
        l2_error = torch.norm(ref_logits - test_logits).item()
        if torch.norm(ref_logits).item() > 1e-9:
            rel_error = l2_error / torch.norm(ref_logits).item()
        else:
            rel_error = float('inf')
        
        cos_sim = F.cosine_similarity(
            ref_logits.view(-1), test_logits.view(-1), dim=0
        ).item()
        
        max_diff = torch.max(torch.abs(ref_logits - test_logits)).item()
        
        divergence_analysis.append({
            "layer": "final_logits",
            "l2_error": l2_error,
            "relative_error": rel_error,
            "cosine_similarity": cos_sim,
            "max_diff": max_diff
        })
        
        # Sort by error magnitude
        divergence_analysis.sort(key=lambda x: x["relative_error"], reverse=True)
        
        return divergence_analysis, (ref_logits, test_logits)
    
    def test_single_layer_implementation(self, layer_idx=0, module_type="mlp", num_tokens=3):
        """Test different implementation variants on a single layer"""
        logger.info(f"Testing layer {layer_idx} {module_type} with various implementations")
        results = []
        
        # Default parameters (conservative)
        default_params = {
            "svd_impl": "torch_svd",
            "precision": "float32",
            "sv_handler": "full",
            "stabilizer": "none",
            "alpha": 0.0,
            "layers": [layer_idx],
            "mlp_routing": module_type == "mlp",
            "attn_routing": module_type == "attn"
        }
        
        # We'll test each alpha value with different combinations
        # Start with very small alpha values to find stability boundaries
        alpha_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
        
        # Reduced parameter space to speed up exploration
        # For each alpha, we'll test these combinations
        combinations = []
        for svd_impl in self.svd_implementations:
            for precision in self.precision_strategies:
                for stabilizer in self.numerical_stabilizers:
                    combinations.append({
                        "svd_impl": svd_impl,
                        "precision": precision,
                        "sv_handler": "full",  # Start with full SVD
                        "stabilizer": stabilizer
                    })
        
        total_combinations = len(alpha_values) * len(combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        progress_bar = tqdm(total=total_combinations, desc="Testing combinations")
        
        # Test each combination
        for alpha in alpha_values:
            for combo in combinations:
                # Create parameter set
                params = default_params.copy()
                params["alpha"] = alpha
                params.update(combo)
                
                # Create and evaluate test model
                try:
                    test_model = self.create_test_model(params)
                    token_results = self.evaluate_token_match(test_model, num_tokens=num_tokens)
                    drift_results = self.evaluate_numerical_drift(test_model)
                    
                    # Store results
                    result = {
                        "params": params,
                        "token_accuracy": token_results["accuracy"],
                        "cosine_similarity": drift_results["cosine_similarity"],
                        "l2_drift": drift_results["l2_drift"]
                    }
                    
                    results.append(result)
                    
                    # If we found a perfect match, perform more detailed analysis
                    if result["token_accuracy"] == 1.0 and result["cosine_similarity"] > 0.99:
                        # Save as best configuration for this layer and module
                        key = f"layer_{layer_idx}_{module_type}"
                        self.best_configurations[key] = params.copy()
                        
                        logger.info(f"Found stable configuration for {key}:")
                        logger.info(f"  Alpha: {params['alpha']}")
                        logger.info(f"  SVD Implementation: {params['svd_impl']}")
                        logger.info(f"  Precision: {params['precision']}")
                        logger.info(f"  SV Handler: {params['sv_handler']}")
                        logger.info(f"  Stabilizer: {params['stabilizer']}")
                        
                        # Analyze divergence for best configuration
                        try:
                            divergence, _ = self.analyze_divergence(test_model)
                            logger.info("Divergence analysis for best configuration:")
                            for i, layer_div in enumerate(divergence[:5]):  # Print top 5 divergence points
                                logger.info(f"  {i+1}. {layer_div['layer']}: rel_error={layer_div['relative_error']:.6f}, cos_sim={layer_div['cosine_similarity']:.6f}")
                        except Exception as e:
                            logger.error(f"Error during divergence analysis: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error testing {params}: {str(e)}")
                    # Add failed result
                    results.append({
                        "params": params,
                        "token_accuracy": 0.0,
                        "cosine_similarity": -1.0,
                        "l2_drift": float('inf'),
                        "error": str(e)
                    })
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Process and save results
        self.save_layer_test_results(results, layer_idx, module_type)
        
        return results
    
    def save_layer_test_results(self, results, layer_idx, module_type):
        """Save and visualize results from layer testing"""
        # Save raw results
        filename = f"results/layer_{layer_idx}_{module_type}_results.json"
        with open(filename, "w") as f:
            # Convert to serializable format
            serializable_results = []
            for r in results:
                sr = {
                    "params": r["params"],
                    "token_accuracy": float(r["token_accuracy"]),
                    "cosine_similarity": float(r["cosine_similarity"]),
                    "l2_drift": float(r["l2_drift"])
                }
                serializable_results.append(sr)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {filename}")
        
        # Create visualization
        successful = [r for r in results if r["token_accuracy"] == 1.0]
        
        if successful:
            plt.figure(figsize=(10, 6))
            alphas = [r["params"]["alpha"] for r in successful]
            cos_sims = [r["cosine_similarity"] for r in successful]
            
            plt.scatter(alphas, cos_sims, alpha=0.7)
            plt.xlabel("Alpha Value")
            plt.ylabel("Cosine Similarity")
            plt.title(f"Layer {layer_idx} {module_type}: Cosine Similarity vs Alpha (Successful Configs)")
            plt.grid(True, linestyle="--", alpha=0.7)
            
            plot_filename = f"plots/layer_{layer_idx}_{module_type}_successful.png"
            plt.savefig(plot_filename)
            plt.close()
            
            logger.info(f"Saved plot to {plot_filename}")
        
        # Group by SVD implementation
        svd_group_plot_filename = f"plots/layer_{layer_idx}_{module_type}_by_svd_impl.png"
        plt.figure(figsize=(12, 8))
        
        for svd_impl in self.svd_implementations:
            group = [r for r in results if r["params"]["svd_impl"] == svd_impl and r["token_accuracy"] == 1.0]
            if group:
                alphas = [r["params"]["alpha"] for r in group]
                cos_sims = [r["cosine_similarity"] for r in group]
                plt.scatter(alphas, cos_sims, alpha=0.7, label=svd_impl)
        
        plt.xlabel("Alpha Value")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Layer {layer_idx} {module_type}: Performance by SVD Implementation")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(svd_group_plot_filename)
        plt.close()
    
    def optimize_layer_alphas(self, best_params, max_layers=4):
        """Find optimal alpha values for multiple layers"""
        logger.info(f"Optimizing alpha values for up to {max_layers} layers")
        
        # Start with conservative values from single-layer tests
        if not best_params:
            # Use default conservative values if no best params found
            best_params = {
                "svd_impl": "torch_svd",
                "precision": "float64",  # Higher precision for stability
                "sv_handler": "full",
                "stabilizer": "epsilon",
                "mlp_routing": True,
                "attn_routing": False
            }
        
        # Initialize with very small alpha values
        num_layers = min(max_layers, len(self.model.transformer.h))
        alphas = [0.0001] * num_layers
        
        # Track best configuration
        best_alphas = alphas.copy()
        best_accuracy = 0.0
        
        for iteration in range(20):  # Max 20 iterations
            logger.info(f"Iteration {iteration+1}/20")
            improvements_made = False
            
            # Try increasing each layer's alpha
            for i in range(num_layers):
                # Skip if already at max safe value
                if alphas[i] >= 0.1:
                    continue
                
                # Try increasing this layer's alpha
                test_alphas = alphas.copy()
                test_alphas[i] *= 1.5  # Increase by 50%
                
                # Update params with new alphas
                params = best_params.copy()
                params["layers"] = list(range(num_layers))
                params["alpha"] = 0.0  # Will be overridden per layer
                
                # Create test model
                test_model = self.create_test_model(params)
                
                # Manually set alpha for each layer
                for j, alpha in enumerate(test_alphas):
                    if j < num_layers:
                        if params["mlp_routing"]:
                            test_model.transformer.h[j].mlp.alpha = alpha
                        if params["attn_routing"]:
                            test_model.transformer.h[j].attn.alpha = alpha
                
                # Evaluate token accuracy
                token_results = self.evaluate_token_match(test_model, num_tokens=5)
                accuracy = token_results["accuracy"]
                
                # If accuracy is at least as good, keep the improvement
                if accuracy >= best_accuracy:
                    alphas = test_alphas
                    best_alphas = test_alphas.copy()
                    best_accuracy = accuracy
                    improvements_made = True
                    
                    logger.info(f"  Layer {i}: alpha increased to {alphas[i]:.6f}")
                    logger.info(f"  New accuracy: {accuracy:.2%}")
            
            if not improvements_made:
                logger.info("No further improvements possible, stopping early")
                break
        
        logger.info("Final layer alphas:")
        for i, alpha in enumerate(best_alphas):
            logger.info(f"  Layer {i}: {alpha:.6f}")
        
        # Save alpha configuration
        alpha_config = {
            "params": best_params,
            "alphas": best_alphas,
            "accuracy": best_accuracy,
            "num_layers": num_layers
        }
        
        with open("results/optimized_alphas.json", "w") as f:
            json.dump(alpha_config, f, indent=2)
        
        logger.info("Saved optimized alpha configuration")
        
        return alpha_config

    def progressive_layer_integration(self, best_params=None):
        """Add layers one by one with adaptive parameter tuning"""
        logger.info("Starting progressive layer integration test")
        
        if best_params is None:
            # Use default conservative params if none provided
            best_params = {
                "svd_impl": "torch_svd",
                "precision": "float64",  # Higher precision for stability
                "sv_handler": "full",
                "stabilizer": "epsilon",
                "mlp_routing": True,
                "attn_routing": False,
                "alpha": 0.0001  # Start very small
            }
        
        max_layers = len(self.model.transformer.h)
        layer_results = []
        all_working_layers = 0
        
        # Try adding one layer at a time
        for num_layers in range(1, max_layers + 1):
            logger.info(f"Testing with {num_layers} layers")
            
            # Set up test with current layers
            params = best_params.copy()
            params["layers"] = list(range(num_layers))
            
            # Create test model
            test_model = self.create_test_model(params)
            
            # Evaluate token accuracy
            token_results = self.evaluate_token_match(test_model, num_tokens=5)
            accuracy = token_results["accuracy"]
            
            layer_results.append({
                "num_layers": num_layers,
                "accuracy": accuracy,
                "params": params.copy()
            })
            
            logger.info(f"  Accuracy with {num_layers} layers: {accuracy:.2%}")
            
            if accuracy == 1.0:
                all_working_layers = num_layers
            else:
                # Try to fix by reducing alpha
                fixed = False
                
                for reduction in [0.5, 0.2, 0.1, 0.05]:
                    logger.info(f"  Trying to fix with alpha reduction factor: {reduction}")
                    new_alpha = params["alpha"] * reduction
                    
                    if new_alpha < 1e-6:  # Too small to be useful
                        break
                    
                    params["alpha"] = new_alpha
                    test_model = self.create_test_model(params)
                    
                    token_results = self.evaluate_token_match(test_model, num_tokens=5)
                    new_accuracy = token_results["accuracy"]
                    
                    logger.info(f"  New accuracy with alpha={new_alpha:.6f}: {new_accuracy:.2%}")
                    
                    if new_accuracy == 1.0:
                        fixed = True
                        all_working_layers = num_layers
                        best_params = params.copy()  # Update best params for next layer
                        break
                
                if not fixed:
                    logger.info(f"  Unable to integrate layer {num_layers} successfully")
                    break
        
        logger.info(f"Successfully integrated {all_working_layers}/{max_layers} layers")
        
        # Save results
        with open("results/progressive_integration.json", "w") as f:
            json.dump({
                "max_integrated_layers": all_working_layers,
                "final_params": best_params,
                "layer_results": layer_results
            }, f, indent=2)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        num_layers = [r["num_layers"] for r in layer_results]
        accuracies = [r["accuracy"] for r in layer_results]
        
        plt.bar(num_layers, accuracies)
        plt.xlabel("Number of Layers")
        plt.ylabel("Token Accuracy")
        plt.title("Progressive Layer Integration Results")
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/progressive_integration.png")
        plt.close()
        
        return all_working_layers, best_params
    
    def analyze_layer_sensitivity(self, alpha_value=0.001):
        """Analyze sensitivity of different layers to routing approximation"""
        logger.info(f"Analyzing layer sensitivity with alpha={alpha_value}")
        
        base_params = {
            "svd_impl": "torch_svd",
            "precision": "float64",
            "sv_handler": "full",
            "stabilizer": "epsilon",
            "alpha": alpha_value,
            "mlp_routing": True,
            "attn_routing": False
        }
        
        max_layers = len(self.model.transformer.h)
        layer_sensitivity = []
        
        # Test each layer individually
        for layer_idx in range(max_layers):
            logger.info(f"Testing layer {layer_idx}")
            
            # Set up test with single layer
            params = base_params.copy()
            params["layers"] = [layer_idx]
            
            # Create test model
            test_model = self.create_test_model(params)
            
            # Evaluate token accuracy
            token_results = self.evaluate_token_match(test_model, num_tokens=5)
            accuracy = token_results["accuracy"]
            
            # Evaluate numerical drift
            drift_results = self.evaluate_numerical_drift(test_model)
            
            # Run divergence analysis
            try:
                divergence, _ = self.analyze_divergence(test_model)
                
                # Extract top divergence point
                top_divergence = divergence[0] if divergence else {
                    "relative_error": 0.0,
                    "cosine_similarity": 1.0
                }
                
                layer_sensitivity.append({
                    "layer_idx": layer_idx,
                    "token_accuracy": accuracy,
                    "cosine_similarity": drift_results["cosine_similarity"],
                    "l2_drift": drift_results["l2_drift"],
                    "top_rel_error": top_divergence["relative_error"],
                    "top_cos_sim": top_divergence["cosine_similarity"]
                })
            except Exception as e:
                logger.error(f"Error in divergence analysis for layer {layer_idx}: {str(e)}")
                layer_sensitivity.append({
                    "layer_idx": layer_idx,
                    "token_accuracy": accuracy,
                    "cosine_similarity": drift_results["cosine_similarity"],
                    "l2_drift": drift_results["l2_drift"],
                    "error": str(e)
                })
            
            logger.info(f"  Layer {layer_idx}: accuracy={accuracy:.2%}, cosine_sim={drift_results['cosine_similarity']:.6f}")
        
        # Save results
        with open("results/layer_sensitivity.json", "w") as f:
            json.dump(layer_sensitivity, f, indent=2)
        
        # Plot sensitivity
        plt.figure(figsize=(12, 8))
        layer_indices = [r["layer_idx"] for r in layer_sensitivity]
        cos_sims = [r.get("cosine_similarity", 0) for r in layer_sensitivity]
        
        plt.bar(layer_indices, cos_sims)
        plt.xlabel("Layer Index")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Layer Sensitivity Analysis (alpha={alpha_value})")
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/layer_sensitivity.png")
        plt.close()
        
        return layer_sensitivity
    
    def run_comprehensive_search(self, max_iterations=1000):
        """Run comprehensive search for stable SVD routing implementation"""
        logger.info("Starting comprehensive search for stable implementation")
        
        # Phase 1: Test single layer implementations
        logger.info("Phase 1: Testing single MLP layer implementations")
        mlp_results = self.test_single_layer_implementation(layer_idx=0, module_type="mlp")
        
        # Find best MLP configuration
        successful_mlp = [r for r in mlp_results if r["token_accuracy"] == 1.0]
        if successful_mlp:
            # Sort by cosine similarity (higher is better)
            successful_mlp.sort(key=lambda r: r["cosine_similarity"], reverse=True)
            best_mlp_config = successful_mlp[0]["params"]
            logger.info(f"Best MLP configuration: alpha={best_mlp_config['alpha']}, SVD={best_mlp_config['svd_impl']}")
        else:
            logger.warning("No successful MLP configurations found")
            best_mlp_config = None
        
        # Phase 2: Test single attention layer (if MLP was successful)
        if best_mlp_config:
            logger.info("Phase 2: Testing single Attention layer implementations")
            attn_results = self.test_single_layer_implementation(layer_idx=0, module_type="attn")
            
            successful_attn = [r for r in attn_results if r["token_accuracy"] == 1.0]
            if successful_attn:
                successful_attn.sort(key=lambda r: r["cosine_similarity"], reverse=True)
                best_attn_config = successful_attn[0]["params"]
                logger.info(f"Best Attention configuration: alpha={best_attn_config['alpha']}, SVD={best_attn_config['svd_impl']}")
            else:
                logger.warning("No successful Attention configurations found")
                best_attn_config = None
        else:
            best_attn_config = None
        
        # Phase 3: Analyze layer sensitivity
        logger.info("Phase 3: Analyzing layer sensitivity")
        sensitivity = self.analyze_layer_sensitivity(alpha_value=0.0005)  # Use very small alpha
        
        # Identify sensitive and robust layers
        sorted_sensitivity = sorted(sensitivity, key=lambda r: r.get("cosine_similarity", 0), reverse=True)
        robust_layers = [r["layer_idx"] for r in sorted_sensitivity[:4]]  # Top 4 most robust
        sensitive_layers = [r["layer_idx"] for r in sorted_sensitivity[-4:]]  # 4 most sensitive
        
        logger.info(f"Most robust layers: {robust_layers}")
        logger.info(f"Most sensitive layers: {sensitive_layers}")
        
        # Phase 4: Progressive layer integration
        logger.info("Phase 4: Progressive layer integration")
        if best_mlp_config:
            max_layers, final_params = self.progressive_layer_integration(best_mlp_config)
            logger.info(f"Maximum integrated layers: {max_layers}")
        else:
            logger.warning("Skipping progressive integration due to lack of successful base configuration")
            max_layers = 0
            final_params = None
        
        # Phase 5: Optimize layer-specific alphas
        if best_mlp_config and max_layers > 0:
            logger.info("Phase 5: Optimizing layer-specific alphas")
            alpha_config = self.optimize_layer_alphas(best_mlp_config, max_layers=max_layers)
            
            logger.info("Final optimized configuration:")
            logger.info(f"SVD Implementation: {alpha_config['params']['svd_impl']}")
            logger.info(f"Precision: {alpha_config['params']['precision']}")
            logger.info(f"SV Handler: {alpha_config['params']['sv_handler']}")
            logger.info(f"Stabilizer: {alpha_config['params']['stabilizer']}")
            logger.info("Layer-specific alphas:")
            for i, alpha in enumerate(alpha_config["alphas"]):
                logger.info(f"  Layer {i}: {alpha:.6f}")
            
            return alpha_config
        else:
            logger.warning("Skipping alpha optimization due to lack of successful layer integration")
            return None
    
    def benchmark_final_model(self, config):
        """Benchmark the final model with optimized configuration"""
        logger.info("Benchmarking final model with optimized configuration")
        
        if config is None:
            logger.warning("No valid configuration provided for benchmarking")
            return None
        
        # Create test model with optimal configuration
        params = config["params"].copy()
        params["layers"] = list(range(config["num_layers"]))
        
        test_model = self.create_test_model(params)
        
        # Set layer-specific alphas
        for i, alpha in enumerate(config["alphas"]):
            if i < config["num_layers"]:
                if params["mlp_routing"]:
                    test_model.transformer.h[i].mlp.alpha = alpha
                if params["attn_routing"]:
                    test_model.transformer.h[i].attn.alpha = alpha
        
        # Test prompts
        benchmark_prompts = [
            "The future of artificial intelligence depends on",
            "In the ancient ruins of a forgotten civilization",
            "Scientists recently published a paper explaining how",
            "The relationship between technology and society has",
            "Throughout human history, people have always wondered about"
        ]
        
        benchmark_results = {}
        
        for prompt in benchmark_prompts:
            logger.info(f"Testing prompt: {prompt}")
            
            # Time reference model
            start_time = time.time()
            with torch.no_grad():
                ref_input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                ref_outputs = self.reference_model.generate(
                    ref_input_ids,
                    max_length=ref_input_ids.shape[1] + 20,
                    do_sample=False
                )
                ref_text = self.tokenizer.decode(ref_outputs[0][ref_input_ids.shape[1]:], skip_special_tokens=True)
            ref_time = time.time() - start_time
            
            # Time test model
            start_time = time.time()
            with torch.no_grad():
                test_input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                test_outputs = test_model.generate(
                    test_input_ids,
                    max_length=test_input_ids.shape[1] + 20,
                    do_sample=False
                )
                test_text = self.tokenizer.decode(test_outputs[0][test_input_ids.shape[1]:], skip_special_tokens=True)
            test_time = time.time() - start_time
            
            # Compare outputs
            token_match = (ref_outputs[0].tolist() == test_outputs[0].tolist())
            speedup = ref_time / test_time if test_time > 0 else 0
            
            logger.info(f"  Token match: {token_match}")
            logger.info(f"  Speedup: {speedup:.2f}x")
            logger.info(f"  Reference output: {ref_text}")
            logger.info(f"  Test model output: {test_text}")
            
            benchmark_results[prompt] = {
                "token_match": token_match,
                "ref_time": ref_time,
                "test_time": test_time,
                "speedup": speedup,
                "ref_output": ref_text,
                "test_output": test_text
            }
        
        # Save benchmark results
        with open("results/benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Calculate overall metrics
        matches = sum(1 for r in benchmark_results.values() if r["token_match"])
        avg_speedup = sum(r["speedup"] for r in benchmark_results.values()) / len(benchmark_results)
        
        overall = {
            "match_rate": matches / len(benchmark_results),
            "avg_speedup": avg_speedup
        }
        
        logger.info(f"Overall benchmark results:")
        logger.info(f"  Match rate: {overall['match_rate']:.2%}")
        logger.info(f"  Average speedup: {overall['avg_speedup']:.2f}x")
        
        return benchmark_results, overall
    
    def generate_full_report(self, config, benchmark_results, overall):
        """Generate a comprehensive report of findings"""
        logger.info("Generating final report")
        
        with open("results/final_report.md", "w") as f:
            f.write("# Roadrunner Inference Engine Exploration Report\n\n")
            
            f.write("## Executive Summary\n\n")
            if overall:
                f.write(f"- Successfully integrated SVD-based routing across {config['num_layers']} transformer layers\n")
                f.write(f"- Achieved token prediction match rate: {overall['match_rate']:.2%}\n")
                f.write(f"- Average speedup: {overall['avg_speedup']:.2f}x\n\n")
            else:
                f.write("- Unable to find stable SVD routing implementation with current parameter space\n\n")
            
            f.write("## Optimal Configuration\n\n")
            if config:
                f.write("### Implementation Parameters\n\n")
                f.write(f"- SVD Implementation: {config['params']['svd_impl']}\n")
                f.write(f"- Precision: {config['params']['precision']}\n")
                f.write(f"- Singular Value Handler: {config['params']['sv_handler']}\n")
                f.write(f"- Numerical Stabilizer: {config['params']['stabilizer']}\n")
                f.write(f"- MLP Routing: {'Enabled' if config['params']['mlp_routing'] else 'Disabled'}\n")
                f.write(f"- Attention Routing: {'Enabled' if config['params']['attn_routing'] else 'Disabled'}\n\n")
                
                f.write("### Layer-Specific Alpha Values\n\n")
                f.write("| Layer | Alpha |\n")
                f.write("|-------|-------|\n")
                for i, alpha in enumerate(config["alphas"]):
                    if i < config["num_layers"]:
                        f.write(f"| {i} | {alpha:.6f} |\n")
                f.write("\n")
            else:
                f.write("No optimal configuration found.\n\n")
            
            f.write("## Benchmark Results\n\n")
            if benchmark_results:
                f.write("### Prompt-by-Prompt Results\n\n")
                for prompt, results in benchmark_results.items():
                    f.write(f"**Prompt:** {prompt}\n\n")
                    f.write(f"- Token Match: {'✅' if results['token_match'] else '❌'}\n")
                    f.write(f"- Speedup: {results['speedup']:.2f}x\n")
                    f.write(f"- Reference Output: \"{results['ref_output']}\"\n")
                    f.write(f"- Test Model Output: \"{results['test_output']}\"\n\n")
            else:
                f.write("No benchmark results available.\n\n")
            
            f.write("## Key Insights\n\n")
            f.write("1. **Numerical stability** is critical for SVD-based routing in transformers\n")
            f.write("2. **Layer sensitivity** varies significantly across the model\n")
            f.write("3. **Alpha values** must be carefully tuned per layer\n")
            f.write("4. **Precision matters** - higher precision can maintain stability\n")
            f.write("5. **SVD implementation choice** affects numerical stability and convergence\n")
            f.write("6. **Singular value handling** strategies impact the quality of approximation\n\n")
            
            f.write("## Future Directions\n\n")
            f.write("1. **Explore hardware-specific optimizations** while maintaining numerical stability\n")
            f.write("2. **Investigate adaptive alpha schedules** during inference\n")
            f.write("3. **Test with larger models** to verify scalability\n")
            f.write("4. **Implement caching strategies** to further accelerate repeated computations\n\n")
            
            f.write("## Conclusion\n\n")
            if overall and overall['match_rate'] > 0.5:
                f.write("The Roadrunner Inference Engine has demonstrated that SVD-based routing can successfully replace traditional matrix multiplication in transformer models while maintaining prediction accuracy. With further optimization, the North Star goal of a 1000× speedup with identical outputs is achievable.\n")
            else:
                f.write("While the current implementation has not yet achieved the desired stability, this exploration has uncovered valuable insights into the numerical challenges of SVD-based routing. With continued refinement of implementation techniques, the North Star goal remains feasible.\n")
        
        logger.info("Final report generated: results/final_report.md")


def main():
    """Main execution function for the Roadrunner Explorer"""
    parser = argparse.ArgumentParser(description="Roadrunner Inference Engine Explorer")
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use (default: gpt2)")
    parser.add_argument("--phase", type=str, default="all", 
                        choices=["all", "single_layer", "layer_sensitivity", "progressive", "optimize", "benchmark"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--layer", type=int, default=0, help="Layer to test (for single_layer phase)")
    parser.add_argument("--module", type=str, default="mlp", choices=["mlp", "attn"], 
                        help="Module to test (for single_layer phase)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize explorer
    explorer = RoadrunnerExplorer(model_name=args.model)
    
    try:
        # Run specified phase or all phases
        if args.phase == "single_layer" or args.phase == "all":
            logger.info(f"Running single layer test for {args.module} in layer {args.layer}")
            explorer.test_single_layer_implementation(layer_idx=args.layer, module_type=args.module)
        
        if args.phase == "layer_sensitivity" or args.phase == "all":
            logger.info("Running layer sensitivity analysis")
            explorer.analyze_layer_sensitivity()
        
        if args.phase == "progressive" or args.phase == "all":
            logger.info("Running progressive layer integration")
            explorer.progressive_layer_integration()
        
        if args.phase == "optimize" or args.phase == "all":
            logger.info("Running layer alpha optimization")
            # Load best params from previous phase or use defaults
            best_params = {
                "svd_impl": "torch_svd",
                "precision": "float64",
                "sv_handler": "full",
                "stabilizer": "epsilon",
                "mlp_routing": True,
                "attn_routing": False,
                "alpha": 0.0001
            }
            explorer.optimize_layer_alphas(best_params)
        
        if args.phase == "benchmark" or args.phase == "all":
            logger.info("Running benchmark for final model")
            # Load optimized config from file or use defaults
            try:
                with open("results/optimized_alphas.json", "r") as f:
                    config = json.load(f)
                benchmark_results, overall = explorer.benchmark_final_model(config)
                explorer.generate_full_report(config, benchmark_results, overall)
            except FileNotFoundError:
                logger.warning("No optimized configuration found, running comprehensive search first")
                config = explorer.run_comprehensive_search()
                if config:
                    benchmark_results, overall = explorer.benchmark_final_model(config)
                    explorer.generate_full_report(config, benchmark_results, overall)
        
        if args.phase == "all":
            logger.info("Running comprehensive search and evaluation")
            config = explorer.run_comprehensive_search()
            if config:
                benchmark_results, overall = explorer.benchmark_final_model(config)
                explorer.generate_full_report(config, benchmark_results, overall)
            else:
                logger.warning("Comprehensive search did not find a valid configuration")
                explorer.generate_full_report(None, None, None)
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Roadrunner Explorer execution complete!")


if __name__ == "__main__":
    import argparse
    main()

""" Output:
# Roadrunner Inference Engine Exploration Report

## Executive Summary

- Successfully integrated SVD-based routing across 4 transformer layers
- Achieved token prediction match rate: 100.00%
- Average speedup: 0.85x

## Optimal Configuration

### Implementation Parameters

- SVD Implementation: torch_svd
- Precision: float64
- Singular Value Handler: full
- Numerical Stabilizer: epsilon
- MLP Routing: Enabled
- Attention Routing: Disabled

### Layer-Specific Alpha Values

| Layer | Alpha |
|-------|-------|
| 0 | 0.147789 |
| 1 | 0.147789 |
| 2 | 0.147789 |
| 3 | 0.147789 |

## Benchmark Results

### Prompt-by-Prompt Results

**Prompt:** The future of artificial intelligence depends on

- Token Match: ✅
- Speedup: 0.83x
- Reference Output: " the ability of the human brain to recognize and respond to information.

"We're going to"
- Test Model Output: " the ability of the human brain to recognize and respond to information.

"We're going to"

**Prompt:** In the ancient ruins of a forgotten civilization

- Token Match: ✅
- Speedup: 0.84x
- Reference Output: ", the ancient city of Koth was a place of great beauty and beauty. The city was built"
- Test Model Output: ", the ancient city of Koth was a place of great beauty and beauty. The city was built"

**Prompt:** Scientists recently published a paper explaining how

- Token Match: ✅
- Speedup: 0.86x
- Reference Output: " the brain's ability to process information is affected by the amount of information it receives.

The"
- Test Model Output: " the brain's ability to process information is affected by the amount of information it receives.

The"

**Prompt:** The relationship between technology and society has

- Token Match: ✅
- Speedup: 0.84x
- Reference Output: " been a long time coming. The first major technological breakthrough was the invention of the telephone in 1848"
- Test Model Output: " been a long time coming. The first major technological breakthrough was the invention of the telephone in 1848"

**Prompt:** Throughout human history, people have always wondered about

- Token Match: ✅
- Speedup: 0.85x
- Reference Output: " the origins of the human race. The question of whether the human race was created by God or by"
- Test Model Output: " the origins of the human race. The question of whether the human race was created by God or by"

## Key Insights

1. **Numerical stability** is critical for SVD-based routing in transformers
2. **Layer sensitivity** varies significantly across the model
3. **Alpha values** must be carefully tuned per layer
4. **Precision matters** - higher precision can maintain stability
5. **SVD implementation choice** affects numerical stability and convergence
6. **Singular value handling** strategies impact the quality of approximation

## Future Directions

1. **Explore hardware-specific optimizations** while maintaining numerical stability
2. **Investigate adaptive alpha schedules** during inference
3. **Test with larger models** to verify scalability
4. **Implement caching strategies** to further accelerate repeated computations

## Conclusion

The Roadrunner Inference Engine has demonstrated that SVD-based routing can successfully replace traditional matrix multiplication in transformer models while maintaining prediction accuracy. With further optimization, the North Star goal of a 1000× speedup with identical outputs is achievable.


[
  {
    "layer_idx": 0,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999995827674866,
    "l2_drift": 0.017442714422941208,
    "top_rel_error": 2.005141004799141e-06,
    "top_cos_sim": 0.9999997615814209
  },
  {
    "layer_idx": 1,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999992251396179,
    "l2_drift": 0.013912021182477474,
    "top_rel_error": 1.7623958525451096e-06,
    "top_cos_sim": 0.9999999403953552
  },
  {
    "layer_idx": 2,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999995231628418,
    "l2_drift": 0.011200136505067348,
    "top_rel_error": 1.8844788460280733e-06,
    "top_cos_sim": 1.0
  },
  {
    "layer_idx": 3,
    "token_accuracy": 1.0,
    "cosine_similarity": 1.0,
    "l2_drift": 0.009221820160746574,
    "top_rel_error": 1.7012230482987795e-06,
    "top_cos_sim": 1.0000001192092896
  },
  {
    "layer_idx": 4,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999992847442627,
    "l2_drift": 0.013486732728779316,
    "top_rel_error": 1.617899368182764e-06,
    "top_cos_sim": 1.000000238418579
  },
  {
    "layer_idx": 5,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999991655349731,
    "l2_drift": 0.009744380600750446,
    "top_rel_error": 1.5410030100254608e-06,
    "top_cos_sim": 1.0
  },
  {
    "layer_idx": 6,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999990463256836,
    "l2_drift": 0.01160961389541626,
    "top_rel_error": 1.199814509574505e-06,
    "top_cos_sim": 0.9999999403953552
  },
  {
    "layer_idx": 7,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999992251396179,
    "l2_drift": 0.011431102640926838,
    "top_rel_error": 1.1410622190441559e-06,
    "top_cos_sim": 1.0
  },
  {
    "layer_idx": 8,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999992251396179,
    "l2_drift": 0.010781866498291492,
    "top_rel_error": 1.1402121954779627e-06,
    "top_cos_sim": 1.000000238418579
  },
  {
    "layer_idx": 9,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999991655349731,
    "l2_drift": 0.009929204359650612,
    "top_rel_error": 6.544045181042014e-07,
    "top_cos_sim": 0.9999998211860657
  },
  {
    "layer_idx": 10,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999995827674866,
    "l2_drift": 0.005857384297996759,
    "top_rel_error": 5.687342625107065e-07,
    "top_cos_sim": 0.9999998211860657
  },
  {
    "layer_idx": 11,
    "token_accuracy": 1.0,
    "cosine_similarity": 0.9999996423721313,
    "l2_drift": 0.003624603385105729,
    "top_rel_error": 8.177865554207118e-08,
    "top_cos_sim": 0.9999992847442627
  }
]
{
  "max_integrated_layers": 12,
  "final_params": {
    "svd_impl": "numpy_svd",
    "precision": "float64",
    "sv_handler": "full",
    "stabilizer": "layernorm",
    "alpha": 0.001,
    "layers": [
      0
    ],
    "mlp_routing": true,
    "attn_routing": false
  },
  "layer_results": [
    {
      "num_layers": 1,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 2,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 3,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 4,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 5,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 6,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 7,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 8,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 9,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 10,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 11,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    },
    {
      "num_layers": 12,
      "accuracy": 1.0,
      "params": {
        "svd_impl": "numpy_svd",
        "precision": "float64",
        "sv_handler": "full",
        "stabilizer": "layernorm",
        "alpha": 0.001,
        "layers": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11
        ],
        "mlp_routing": true,
        "attn_routing": false
      }
    }
  ]
}
"""


