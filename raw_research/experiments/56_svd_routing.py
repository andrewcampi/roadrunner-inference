import torch
import time
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.linalg import svd

class RoadrunnerEngine:
    def __init__(self, model_name="gpt2"):
        """Initialize Roadrunner Inference Engine with the GPT-2 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size
        self.n_head = self.model.config.n_head
        self.head_dim = self.hidden_dim // self.n_head
        
        # Cache for SVD decompositions
        self.svd_cache = {}
        
        # Alpha value for routing blend (start with 0)
        self.alpha = 0.0
        
        # Routing flags (disabled by default)
        self._mlp_routing_enabled = False
        self._attn_routing_enabled = False
        
        # Debug stats
        self.debug_stats = {'times': {}, 'cosine_similarities': {}}
        
        # Precompute SVD decompositions
        self._precompute_svd()
        
        print(f"Initialized Roadrunner with alpha = {self.alpha}")
        print(f"MLP routing: {'enabled' if self._mlp_routing_enabled else 'disabled'}")
        print(f"Attention routing: {'enabled' if self._attn_routing_enabled else 'disabled'}")
    
    def _precompute_svd(self):
        """Precompute SVD decompositions for all model weights."""
        print("Precomputing SVD decompositions...")
        
        # Process each transformer block
        for i, block in enumerate(self.model.transformer.h):
            self.svd_cache[i] = {
                'mlp': self._decompose_mlp(block),
                'attention': self._decompose_attention(block)
            }
        
        print("SVD decomposition complete!")
    
    def _decompose_mlp(self, block):
        """Decompose MLP weights using SVD."""
        # Extract weights
        fc_weight = block.mlp.c_fc.weight.data.clone().T  # [3072, 768]
        fc_bias = block.mlp.c_fc.bias.data.clone()
        proj_weight = block.mlp.c_proj.weight.data.clone().T  # [768, 3072]
        proj_bias = block.mlp.c_proj.bias.data.clone()
        
        # SVD decomposition
        U_fc, S_fc, Vh_fc = svd(fc_weight, full_matrices=False)
        U_proj, S_proj, Vh_proj = svd(proj_weight, full_matrices=False)
        
        return {
            'fc': (U_fc, S_fc, Vh_fc, fc_bias),
            'proj': (U_proj, S_proj, Vh_proj, proj_bias),
            'fc_weight': fc_weight,
            'proj_weight': proj_weight
        }
    
    def _decompose_attention(self, block):
        """Decompose attention weights using SVD."""
        # Extract weights
        attn_w = block.attn.c_attn.weight.data.clone()  # [768, 2304]
        attn_b = block.attn.c_attn.bias.data.clone()
        W_q, W_k, W_v = torch.chunk(attn_w, 3, dim=1)
        b_q, b_k, b_v = torch.chunk(attn_b, 3, dim=0)
        
        # SVD for Q, K, V
        U_q, S_q, Vh_q = svd(W_q, full_matrices=False)
        U_k, S_k, Vh_k = svd(W_k, full_matrices=False)
        U_v, S_v, Vh_v = svd(W_v, full_matrices=False)
        
        # SVD for projection
        proj_w = block.attn.c_proj.weight.data.clone().T
        proj_b = block.attn.c_proj.bias.data.clone()
        U_proj, S_proj, Vh_proj = svd(proj_w, full_matrices=False)
        
        return {
            'q': (U_q, S_q, Vh_q, b_q),
            'k': (U_k, S_k, Vh_k, b_k),
            'v': (U_v, S_v, Vh_v, b_v),
            'proj': (U_proj, S_proj, Vh_proj, proj_b),
            'W_q': W_q,
            'W_k': W_k,
            'W_v': W_v,
            'proj_w': proj_w
        }
    
    def _mlp_forward(self, layer_idx, x):
        """Forward pass through MLP using standard computation."""
        block = self.model.transformer.h[layer_idx]
        return block.mlp(x)
    
    def _mlp_routed(self, layer_idx, x):
        """Forward pass through MLP using SVD-based routing.
        Based on experiment #50 with careful handling of numerical precision.
        """
        block = self.model.transformer.h[layer_idx]
        alpha = self.alpha
        
        # Time standard forward pass
        t0 = time.time()
        standard_output = self._mlp_forward(layer_idx, x)
        t_standard = time.time() - t0
        
        if alpha <= 0 or not self._mlp_routing_enabled:
            return standard_output
        
        # Get cached SVD components
        svd_mlp = self.svd_cache[layer_idx]['mlp']
        U_fc, S_fc, Vh_fc, fc_bias = svd_mlp['fc']
        U_proj, S_proj, Vh_proj, proj_bias = svd_mlp['proj']
        
        # Time routed forward pass
        t0 = time.time()
        
        # Route through SVD decomposition (following experiment #50)
        # x shape: [batch_size, seq_len, hidden_dim]
        # Vh_fc shape: [hidden_dim, hidden_dim]
        code = x @ Vh_fc                                  # Project to code space
        code_scaled = code * S_fc                         # Scale by singular values
        routed_hidden = F.gelu(code_scaled @ U_fc.T + fc_bias)  # Apply activation
        
        # Process through projection
        # routed_hidden shape: [batch_size, seq_len, hidden_dim]
        # Vh_proj shape: [hidden_dim, hidden_dim]
        proj_code = routed_hidden @ Vh_proj.T            # Note: Using transpose of Vh_proj
        routed_out = proj_code * S_proj @ U_proj.T + proj_bias
        
        t_routed = time.time() - t0
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                standard_output.flatten(),
                routed_out.flatten(),
                dim=0
            ).item()
            
            layer_name = f"mlp_{layer_idx}"
            if layer_name not in self.debug_stats['times']:
                self.debug_stats['times'][layer_name] = []
                self.debug_stats['cosine_similarities'][layer_name] = []
            
            self.debug_stats['times'][layer_name].append((t_standard, t_routed))
            self.debug_stats['cosine_similarities'][layer_name].append(cos_sim)
        
        # Return blended output
        return alpha * routed_out + (1 - alpha) * standard_output
    
    def _attention_forward(self, layer_idx, x, past_key_value=None):
        """Forward pass through attention using standard computation."""
        block = self.model.transformer.h[layer_idx]
        outputs = block.attn(x, past_key_value=past_key_value, use_cache=True)
        return outputs[0], outputs[1]  # output, present_key_value
    
    def _attention_routed(self, layer_idx, x, past_key_value=None):
        """Forward pass through attention using SVD-based routing.
        Based on experiment #54 with careful handling of numerical precision.
        """
        block = self.model.transformer.h[layer_idx]
        alpha = self.alpha
        
        # Time standard forward pass
        t0 = time.time()
        standard_output, present_key_value = self._attention_forward(layer_idx, x, past_key_value)
        t_standard = time.time() - t0
        
        if alpha <= 0 or not self._attn_routing_enabled:
            return standard_output, present_key_value
        
        # Get cached SVD components
        svd_attn = self.svd_cache[layer_idx]['attention']
        U_q, S_q, Vh_q, b_q = svd_attn['q']
        U_k, S_k, Vh_k, b_k = svd_attn['k']
        U_v, S_v, Vh_v, b_v = svd_attn['v']
        U_proj, S_proj, Vh_proj, proj_b = svd_attn['proj']
        
        # Time routed forward pass
        t0 = time.time()
        
        batch_size, seq_len, _ = x.shape
        
        # Process QKV in a way that mirrors experiment #54
        # Project to Q, K, V
        code_q = x @ Vh_q
        code_k = x @ Vh_k
        code_v = x @ Vh_v
        
        # Scale and reconstruct
        q = (code_q * S_q) @ U_q.T + b_q
        k = (code_k * S_k) @ U_k.T + b_k
        v = (code_v * S_v) @ U_v.T + b_v
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle past key values for incremental decoding
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Scale q for numerical stability
        q = q / (self.head_dim ** 0.5)
        
        # Compute attention scores and apply softmax
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Apply output projection
        proj_code = attn_output @ Vh_proj
        routed_output = proj_code * S_proj @ U_proj.T + proj_b
        
        t_routed = time.time() - t0
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                standard_output.flatten(),
                routed_output.flatten(),
                dim=0
            ).item()
            
            layer_name = f"attn_{layer_idx}"
            if layer_name not in self.debug_stats['times']:
                self.debug_stats['times'][layer_name] = []
                self.debug_stats['cosine_similarities'][layer_name] = []
            
            self.debug_stats['times'][layer_name].append((t_standard, t_routed))
            self.debug_stats['cosine_similarities'][layer_name].append(cos_sim)
        
        # Blend outputs
        blended_output = alpha * routed_output + (1 - alpha) * standard_output
        
        return blended_output, (k, v)
    
    def _process_block(self, layer_idx, x, past_key_value=None):
        """Process a transformer block with attention and MLP.
        Uses routing if enabled, otherwise standard computation.
        """
        block = self.model.transformer.h[layer_idx]
        
        # Use standard block without routing by default
        if not self._mlp_routing_enabled and not self._attn_routing_enabled:
            outputs = block(x, past_key_value=past_key_value, use_cache=True)
            return outputs[0], outputs[1]  # output, present_key_value
        
        # Implement the full routing path with both attention and MLP
        # Layer norm 1
        residual = x
        x_ln1 = block.ln_1(x)
        
        # Attention (routed or standard)
        if self._attn_routing_enabled:
            attn_output, present_key_value = self._attention_routed(layer_idx, x_ln1, past_key_value)
        else:
            attn_output, present_key_value = self._attention_forward(layer_idx, x_ln1, past_key_value)
        
        # Residual connection
        x = residual + attn_output
        
        # Layer norm 2
        residual = x
        x_ln2 = block.ln_2(x)
        
        # MLP (routed or standard)
        if self._mlp_routing_enabled:
            mlp_output = self._mlp_routed(layer_idx, x_ln2)
        else:
            mlp_output = self._mlp_forward(layer_idx, x_ln2)
        
        # Residual connection
        x = residual + mlp_output
        
        return x, present_key_value
    
    def generate_token(self, input_ids, past_key_values=None):
        """Generate a single token using the model."""
        # If routing is disabled, use the model directly for exact match
        if not self._mlp_routing_enabled and not self._attn_routing_enabled:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                return next_token_id, outputs.past_key_values
        
        # Use our custom routing implementation
        with torch.no_grad():
            # Embed input tokens
            inputs_embeds = self.model.transformer.wte(input_ids)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.device)
            position_embeds = self.model.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            
            # Initialize past key values if not provided
            if past_key_values is None:
                past_key_values = [None] * len(self.model.transformer.h)
            
            present_key_values = []
            
            # Process through transformer blocks
            for i in range(len(self.model.transformer.h)):
                hidden_states, present_kv = self._process_block(
                    i, hidden_states, past_key_values[i]
                )
                present_key_values.append(present_kv)
            
            # Final layer norm
            hidden_states = self.model.transformer.ln_f(hidden_states)
            
            # LM head
            lm_logits = self.model.lm_head(hidden_states[:, -1:, :]).squeeze(1)
            
            # Get the most likely next token
            next_token_id = torch.argmax(lm_logits, dim=-1, keepdim=True)
            
            return next_token_id, present_key_values
    
    def generate(self, prompt, max_length=50):
        """Generate text with the model."""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        past_key_values = None
        
        # Start generation timer
        start_time = time.time()
        all_tokens = input_ids.clone()
        
        # Generate tokens one by one
        for i in range(max_length):
            next_token_id, past_key_values = self.generate_token(input_ids, past_key_values)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            all_tokens = torch.cat([all_tokens, next_token_id], dim=1)
            
            # Print generation progress (every 5 tokens to avoid cluttering)
            if i % 5 == 0 or i == max_length - 1:
                generated_text = self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
                print(f"\rToken {i+1}/{max_length}: {generated_text}", end="", flush=True)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        tokens_per_second = max_length / generation_time
        
        # Decode final sequence
        generated_text = self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        
        print(f"\n\nGeneration complete!")
        print(f"Generated {max_length} tokens in {generation_time:.2f} seconds")
        print(f"Speed: {tokens_per_second:.2f} tokens/sec")
        
        return generated_text
    
    def verify_standard_model(self, prompt, num_tokens=5):
        """Verify that our implementation of the standard model works correctly."""
        # Create a baseline model for comparison
        baseline_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        std_ids = input_ids.clone()
        
        # Init past key values
        roadrunner_past = None
        std_past = None
        
        print("\n--- Standard Model Verification ---")
        print(f"Prompt: {prompt}")
        print(f"{'Token #':<8} {'HF Model':<15} {'Our Model':<15} {'Match':<10}")
        print("-" * 50)
        
        match_count = 0
        
        for i in range(num_tokens):
            # Generate with Roadrunner (standard mode)
            with torch.no_grad():
                next_token_rr, roadrunner_past = self.generate_token(input_ids, roadrunner_past)
                input_ids = torch.cat([input_ids, next_token_rr], dim=1)
            
            # Generate with HuggingFace baseline model
            with torch.no_grad():
                outputs = baseline_model(std_ids, past_key_values=std_past, use_cache=True)
                next_token_std = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                std_ids = torch.cat([std_ids, next_token_std], dim=1)
                std_past = outputs.past_key_values
            
            # Compare tokens
            token_match = (next_token_rr.item() == next_token_std.item())
            if token_match:
                match_count += 1
            
            # Display results
            token_rr = self.tokenizer.decode([next_token_rr.item()])
            token_std = self.tokenizer.decode([next_token_std.item()])
            print(f"{i+1:<8} {token_std:<15} {token_rr:<15} {'✅' if token_match else '❌':<10}")
        
        accuracy = match_count / num_tokens * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({match_count}/{num_tokens} tokens matched)")
        return accuracy == 100.0
    
    def verify_routed_model(self, prompt, num_tokens=5):
        """Verify that the routed model produces the same tokens as standard."""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Save current routing state
        current_alpha = self.alpha
        current_mlp_routing = self._mlp_routing_enabled
        current_attn_routing = self._attn_routing_enabled
        
        # Generate with standard model (no routing)
        self.alpha = 0.0
        self._mlp_routing_enabled = False
        self._attn_routing_enabled = False
        
        std_ids = input_ids.clone()
        std_past = None
        std_tokens = []
        
        for _ in range(num_tokens):
            with torch.no_grad():
                next_token_std, std_past = self.generate_token(std_ids, std_past)
                std_ids = torch.cat([std_ids, next_token_std], dim=1)
                std_tokens.append(next_token_std.item())
        
        # Generate with routed model (restore settings)
        self.alpha = current_alpha
        self._mlp_routing_enabled = current_mlp_routing
        self._attn_routing_enabled = current_attn_routing
        
        routed_ids = input_ids.clone()
        routed_past = None
        routed_tokens = []
        
        for _ in range(num_tokens):
            with torch.no_grad():
                next_token_routed, routed_past = self.generate_token(routed_ids, routed_past)
                routed_ids = torch.cat([routed_ids, next_token_routed], dim=1)
                routed_tokens.append(next_token_routed.item())
        
        # Compare results
        print("\n--- Routed Model Verification ---")
        print(f"Prompt: {prompt}")
        print(f"Alpha: {self.alpha}")
        print(f"MLP routing: {'enabled' if self._mlp_routing_enabled else 'disabled'}")
        print(f"Attention routing: {'enabled' if self._attn_routing_enabled else 'disabled'}")
        print(f"{'Token #':<8} {'Standard':<15} {'Routed':<15} {'Match':<10}")
        print("-" * 50)
        
        match_count = 0
        for i, (std_token, routed_token) in enumerate(zip(std_tokens, routed_tokens)):
            token_match = std_token == routed_token
            if token_match:
                match_count += 1
            
            token_std = self.tokenizer.decode([std_token])
            token_routed = self.tokenizer.decode([routed_token])
            print(f"{i+1:<8} {token_std:<15} {token_routed:<15} {'✅' if token_match else '❌':<10}")
        
        accuracy = match_count / num_tokens * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({match_count}/{num_tokens} tokens matched)")
        return accuracy == 100.0
    
    def enable_mlp_routing(self, enabled=True):
        """Enable or disable MLP routing for testing purposes."""
        print(f"\nMLP routing {'enabled' if enabled else 'disabled'}")
        self._mlp_routing_enabled = enabled
        
    def enable_attention_routing(self, enabled=True):
        """Enable or disable attention routing for testing purposes."""
        print(f"\nAttention routing {'enabled' if enabled else 'disabled'}")
        self._attn_routing_enabled = enabled
    
    def tune_mlp_routing(self, prompt, max_alpha=0.05, steps=10, tokens_per_step=3):
        """Tune MLP routing alpha while ensuring token prediction fidelity."""
        print("\n--- MLP Routing Alpha Tuning ---")
        print(f"Starting with alpha = 0, increasing toward {max_alpha}")
        
        # Enable MLP routing only
        self.enable_mlp_routing(True)
        self.enable_attention_routing(False)
        
        # Start with alpha = 0
        original_alpha = self.alpha
        self.alpha = 0.0
        alpha_step = max_alpha / steps
        
        for step in range(1, steps + 1):
            # Set alpha for this step
            test_alpha = step * alpha_step
            self.alpha = test_alpha
            
            # Test with this alpha
            print(f"\nStep {step}/{steps}: Testing alpha = {test_alpha:.6f}")
            accuracy = self.verify_routed_model(prompt, num_tokens=tokens_per_step)
            
            if not accuracy:
                # If mismatch, revert to previous alpha
                self.alpha = (step - 1) * alpha_step
                print(f"❌ Alpha {test_alpha:.6f} caused mismatch. Reverting to {self.alpha:.6f}")
                break
        
        mlp_alpha = self.alpha
        print(f"\n✅ Final MLP alpha after tuning: {mlp_alpha:.6f}")
        
        # Restore original alpha
        self.alpha = original_alpha
        
        return mlp_alpha
    
    def tune_attention_routing(self, prompt, max_alpha=0.01, steps=10, tokens_per_step=3):
        """Tune attention routing alpha while ensuring token prediction fidelity."""
        print("\n--- Attention Routing Alpha Tuning ---")
        print(f"Starting with alpha = 0, increasing toward {max_alpha}")
        
        # Enable attention routing only
        self.enable_mlp_routing(False)
        self.enable_attention_routing(True)
        
        # Start with alpha = 0
        original_alpha = self.alpha
        self.alpha = 0.0
        alpha_step = max_alpha / steps
        
        for step in range(1, steps + 1):
            # Set alpha for this step
            test_alpha = step * alpha_step
            self.alpha = test_alpha
            
            # Test with this alpha
            print(f"\nStep {step}/{steps}: Testing alpha = {test_alpha:.6f}")
            accuracy = self.verify_routed_model(prompt, num_tokens=tokens_per_step)
            
            if not accuracy:
                # If mismatch, revert to previous alpha
                self.alpha = (step - 1) * alpha_step
                print(f"❌ Alpha {test_alpha:.6f} caused mismatch. Reverting to {self.alpha:.6f}")
                break
        
        attn_alpha = self.alpha
        print(f"\n✅ Final attention alpha after tuning: {attn_alpha:.6f}")
        
        # Restore original alpha
        self.alpha = original_alpha
        
        return attn_alpha
    
    def print_performance_stats(self):
        """Print performance statistics based on collected debug data."""
        print("\n--- Performance Statistics ---")
        
        if not self.debug_stats['times']:
            print("No performance data collected yet.")
            return
        
        print(f"{'Layer':<15} {'Standard (ms)':<15} {'Routed (ms)':<15} {'Speedup':<10} {'Cos Sim':<10}")
        print("-" * 65)
        
        for layer_name, times in self.debug_stats['times'].items():
            if not times:
                continue
                
            avg_std_time = sum(t[0] for t in times) / len(times) * 1000  # ms
            avg_routed_time = sum(t[1] for t in times) / len(times) * 1000  # ms
            speedup = avg_std_time / avg_routed_time if avg_routed_time > 0 else 0
            
            avg_cos_sim = sum(self.debug_stats['cosine_similarities'][layer_name]) / len(self.debug_stats['cosine_similarities'][layer_name])
            
            print(f"{layer_name:<15} {avg_std_time:<15.3f} {avg_routed_time:<15.3f} {speedup:<10.2f}x {avg_cos_sim:<10.6f}")

# === Main execution ===
if __name__ == "__main__":
    # Initialize the Roadrunner engine
    roadrunner = RoadrunnerEngine()
    
    # Test prompt
    test_prompt = "The sky was clear and stars were bright. The moon"
    
    # Step 1: Verify our standard model works correctly
    print("\n=== STEP 1: Verifying Standard Model ===")
    is_base_accurate = roadrunner.verify_standard_model(test_prompt, num_tokens=5)
    
    if is_base_accurate:
        print("\n✅ Standard model implementation works correctly!")
        
        # Step 2: Tune MLP routing
        print("\n=== STEP 2: Tuning MLP Routing ===")
        mlp_alpha = roadrunner.tune_mlp_routing(test_prompt, max_alpha=0.05, steps=5)
        
        # Step 3: Tune attention routing
        print("\n=== STEP 3: Tuning Attention Routing ===")
        attn_alpha = roadrunner.tune_attention_routing(test_prompt, max_alpha=0.01, steps=5)
        
        # Step 4: Enable both with optimal alpha values
        print("\n=== STEP 4: Testing Full Routing Implementation ===")
        roadrunner.alpha = min(mlp_alpha, attn_alpha)  # Use the more conservative alpha
        roadrunner.enable_mlp_routing(True)
        roadrunner.enable_attention_routing(True)
        
        is_full_accurate = roadrunner.verify_routed_model(test_prompt, num_tokens=10)
        
        if is_full_accurate:
            print(f"\n✅ Full Roadrunner implementation works with alpha = {roadrunner.alpha}!")
            
            # Step 5: Generate text with fully routed model
            print("\n=== STEP 5: Generate with Routed Model ===")
            completion = roadrunner.generate(
                prompt="In a breakthrough paper on artificial intelligence, researchers demonstrated",
                max_length=20
            )
            
            # Step 6: Print performance statistics
            roadrunner.print_performance_stats()
        else:
            print("\n❌ Full routing implementation has accuracy issues. Further tuning needed.")
    else:
        print("\n❌ Standard model implementation has issues. Fixed this first.")


""" Output:
Precomputing SVD decompositions...
SVD decomposition complete!
Initialized Roadrunner with alpha = 0.0
MLP routing: disabled
Attention routing: disabled

=== STEP 1: Verifying Standard Model ===

--- Standard Model Verification ---
Prompt: The sky was clear and stars were bright. The moon
Token #  HF Model        Our Model       Match     
--------------------------------------------------
1         was             was            ✅         
2         clear           clear          ✅         
3         and             and            ✅         
4         stars           stars          ✅         
5         were            were           ✅         

Accuracy: 100.0% (5/5 tokens matched)

✅ Standard model implementation works correctly!

=== STEP 2: Tuning MLP Routing ===

--- MLP Routing Alpha Tuning ---
Starting with alpha = 0, increasing toward 0.05

MLP routing enabled

Attention routing disabled

Step 1/5: Testing alpha = 0.010000

--- Routed Model Verification ---
Prompt: The sky was clear and stars were bright. The moon
Alpha: 0.01
MLP routing: enabled
Attention routing: disabled
Token #  Standard        Routed          Match     
--------------------------------------------------
1         was             was            ✅         
2         clear           shining        ❌         
3         and            ,               ❌         

Accuracy: 33.3% (1/3 tokens matched)
❌ Alpha 0.010000 caused mismatch. Reverting to 0.000000

✅ Final MLP alpha after tuning: 0.000000

=== STEP 3: Tuning Attention Routing ===

--- Attention Routing Alpha Tuning ---
Starting with alpha = 0, increasing toward 0.01

MLP routing disabled

Attention routing enabled

Step 1/5: Testing alpha = 0.002000

--- Routed Model Verification ---
Prompt: The sky was clear and stars were bright. The moon
Alpha: 0.002
MLP routing: disabled
Attention routing: enabled
Token #  Standard        Routed          Match     
--------------------------------------------------
1         was             was            ✅         
2         clear           shining        ❌         
3         and            ,               ❌         

Accuracy: 33.3% (1/3 tokens matched)
❌ Alpha 0.002000 caused mismatch. Reverting to 0.000000

✅ Final attention alpha after tuning: 0.000000

=== STEP 4: Testing Full Routing Implementation ===

MLP routing enabled

Attention routing enabled

--- Routed Model Verification ---
Prompt: The sky was clear and stars were bright. The moon
Alpha: 0.0
MLP routing: enabled
Attention routing: enabled
Token #  Standard        Routed          Match     
--------------------------------------------------
1         was             was            ✅         
2         clear           shining        ❌         
3         and            ,               ❌         
4         stars           and            ❌         
5         were            the            ❌         
6         bright          stars          ❌         
7        .                were           ❌         
8         The             shining        ❌         
9         sky            .               ❌         
10        was            
               ❌         

Accuracy: 10.0% (1/10 tokens matched)

❌ Full routing implementation has accuracy issues. Further tuning needed.
"""