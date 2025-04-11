import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

SVD_DIR = "svd" # Must run 65_compute_gpt2_svd.py first

def load_svd_for_mlp(i):
    svd_path = os.path.join(SVD_DIR, f"layer_{i}.pt")
    data = torch.load(svd_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return data["mlp"]

def load_svd_for_attention(i):
    svd_path = os.path.join(SVD_DIR, f"layer_{i}.pt")
    data = torch.load(svd_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return data["attention"]

def normalize_singular_values(S):
    """
    Normalizes singular values to prevent exploding activations.
    Uses log1p scaling for better numerical stability.
    """
    return torch.log1p(S) / torch.log1p(S).max()

class RoadrunnerMLP:
    """
    Implements SVD-routed MLP with alpha blending and fallback.

    Parameters:
        model: The GPT-2 transformer block containing the MLP module.
        svd_mlp (dict): Dictionary containing SVD components:
            {
                'fc': (U_fc, S_fc, Vh_fc, bias_fc),
                'proj': (U_proj, S_proj, Vh_proj, bias_proj)
            }
        alpha (float): Blending coefficient between routed and standard output (0.0 = no routing).
        enabled (bool): Whether to apply SVD routing. If False, runs standard MLP.
    """
    def __init__(self, model, svd_mlp, alpha=0.5, enabled=True):
        self.block = model  # The full transformer block
        self.svd = svd_mlp
        self.alpha = alpha
        self.enabled = enabled

    def forward(self, x, verbose=False):
        """
        Computes MLP output using SVD routing if enabled, otherwise defaults.

        Args:
            x (Tensor): Input hidden state [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor: Output tensor of same shape
        """
        if not self.enabled or self.alpha <= 0:
            return self.block.mlp(x)

        # Standard MLP output
        standard_out = self.block.mlp(x)

        # SVD components
        U_fc, S_fc, Vh_fc, bias_fc = self.svd['fc']
        U_proj, S_proj, Vh_proj, bias_proj = self.svd['proj']

        # Normalize singular values
        S_fc_scaled = normalize_singular_values(S_fc)
        S_proj_scaled = normalize_singular_values(S_proj)

        # Input projection: x @ Vh_fc
        code = torch.matmul(x, Vh_fc)  # [B, T, 768]
        code_scaled = code * S_fc_scaled.unsqueeze(0).unsqueeze(0)  # [B, T, 768]
        
        # Add clamping before GELU for numerical stability
        hidden_pre = torch.matmul(code_scaled, U_fc.T) + bias_fc
        hidden_pre = torch.clamp(hidden_pre, -10, 10)  # Safety net to prevent extreme values
        hidden = F.gelu(hidden_pre)  # [B, T, 3072]

        # Output projection
        proj_code = torch.matmul(hidden, Vh_proj.T)
        routed_out = torch.matmul(proj_code * S_proj_scaled, U_proj.T) + bias_proj

        # Debug info
        if verbose:
            diff = (routed_out - standard_out).norm().item()
            print(f"MLP routed/std diff: {diff:.4f}")

        # Blend routed and standard output
        return self.alpha * routed_out + (1 - self.alpha) * standard_out

class RoadrunnerAttention:
    """
    Implements SVD-routed attention with support for past key/value caching and alpha blending.

    Parameters:
        model: The GPT-2 attention block.
        svd_attn (dict): Dictionary containing SVD components for query, key, value, output projections:
            {
                'q': (U_q, S_q, Vh_q, bias_q),
                'k': (U_k, S_k, Vh_k, bias_k),
                'v': (U_v, S_v, Vh_v, bias_v),
                'o': (U_o, S_o, Vh_o, bias_o)
            }
        alpha (float): Blending coefficient between routed and standard output (0.0 = full fallback).
        enabled (bool): Whether to apply SVD routing.
    """
    def __init__(self, model, svd_attn, hidden_dim, alpha=0.5, enabled=True):
        self.block = model
        self.svd = svd_attn
        self.alpha = alpha
        self.enabled = enabled

        self.num_heads = getattr(model.attn, "num_heads", hidden_dim // 64)
        self.head_dim = hidden_dim // self.num_heads

    def _reshape_for_heads(self, x):
        """
        Reshapes tensor for multi-head attention.

        Args:
            x (Tensor): [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor: [batch_size, num_heads, seq_len, head_dim]
        """
        B, T, H = x.size()
        
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x):
        """
        Combines multi-head tensor back to [batch, seq_len, hidden].

        Args:
            x (Tensor): [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Tensor: [batch_size, seq_len, hidden_dim]
        """
        B, nH, T, dH = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nH * dH)

    def forward(self, x, past_key_value=None, verbose=False):
        """
        Computes attention output using routed SVD or defaults to original attention.

        Args:
            x (Tensor): Input hidden state [batch, seq_len, hidden_dim]
            past_key_value (tuple): Optional cached key/value for faster decoding

        Returns:
            Tuple:
                attn_output (Tensor): [batch, seq_len, hidden_dim]
                new_past_key_value (tuple): updated cached key/value
        """
        if not self.enabled or self.alpha <= 0:
            attn_output, new_past = self.block.attn(x, layer_past=past_key_value, use_cache=True)
            return attn_output, new_past

        # --- Routing: Apply SVD to query/key/value/o_proj ---
        U_q, S_q, Vh_q, bias_q = self.svd['q']
        U_k, S_k, Vh_k, bias_k = self.svd['k']
        U_v, S_v, Vh_v, bias_v = self.svd['v']
        U_o, S_o, Vh_o, bias_o = self.svd['o']

        # Normalize singular values
        S_q_scaled = normalize_singular_values(S_q)
        S_k_scaled = normalize_singular_values(S_k)
        S_v_scaled = normalize_singular_values(S_v)
        S_o_scaled = normalize_singular_values(S_o)

        # Compute queries, keys, values
        Q = torch.matmul(x, Vh_q) * S_q_scaled
        Q = torch.matmul(Q, U_q.T) + bias_q
        K = torch.matmul(x, Vh_k) * S_k_scaled
        K = torch.matmul(K, U_k.T) + bias_k
        V = torch.matmul(x, Vh_v) * S_v_scaled
        V = torch.matmul(V, U_v.T) + bias_v

        # Reshape for multi-head attention
        Q = self._reshape_for_heads(Q)
        K = self._reshape_for_heads(K)
        V = self._reshape_for_heads(V)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        new_past = (K, V)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Combine heads
        context = self._combine_heads(context)

        # Output projection (SVD-routed)
        out = torch.matmul(context, Vh_o) * S_o_scaled
        out = torch.matmul(out, U_o.T) + bias_o

        # Fallback / original attention
        fallback_out, _ = self.block.attn(x, layer_past=past_key_value, use_cache=True)

        # Debug info
        if verbose:
            diff = (out - fallback_out).norm().item()
            print(f"Attention routed/std diff: {diff:.4f}")

        # Blend outputs
        blended_out = self.alpha * out + (1 - self.alpha) * fallback_out
        return blended_out, new_past

class RoadrunnerLMHead:
    """
    Replaces dense LM head projection with top-k dot-product routing and optional fallback.

    Parameters:
        model: The GPT-2 model, used to access lm_head weight and bias.
        k (int): Number of top tokens to consider during routing.
        fallback_threshold (float): Minimum score to accept routed prediction; otherwise fallback.
        rerank (bool): If True, reranks top-k using logits for better precision.
        fallback (bool): If True, allows full matmul fallback when threshold not met.
    """
    def __init__(self, model, k=5, fallback_threshold=0.0, rerank=True, fallback=True):
        self.model = model
        self.device = model.device
        self.k = k
        self.threshold = fallback_threshold
        self.rerank = rerank
        self.fallback = fallback

        self.weight = model.lm_head.weight.data.to(self.device)  # [vocab_size, hidden_dim]
        self.bias = model.lm_head.bias
        if self.bias is not None:
            self.bias = self.bias.data.to(self.device)

    def predict(self, hidden_state, verbose=False):
        """
        Predicts the next token using top-k dot product routing.

        Args:
            hidden_state (Tensor): Final hidden state from transformer [1, hidden_dim]

        Returns:
            Tensor: predicted token ID [int]
        """
        # Dot product routing: scores = W @ h
        scores = torch.matmul(self.weight, hidden_state.squeeze(0))  # [vocab]
        if self.bias is not None:
            scores += self.bias

        # Get top-k tokens
        topk_scores, topk_indices = torch.topk(scores, self.k)
        top_score = topk_scores[0].item()

        if verbose:
            print(f"Top token score: {top_score:.4f}")

        if top_score >= self.threshold:
            if self.rerank:
                reranked_idx = torch.argmax(topk_scores).item()
                return topk_indices[reranked_idx].clone().detach().to(hidden_state.device)
            else:
                return topk_indices[0].clone().detach().to(hidden_state.device)
        elif self.fallback:
            # Full logits fallback
            logits = torch.matmul(hidden_state, self.weight.T)
            if self.bias is not None:
                logits += self.bias
            return torch.argmax(logits.squeeze()).to(hidden_state.device)
        else:
            # No fallback allowed — force routing choice
            reranked_idx = torch.argmax(topk_scores).item()
            return topk_indices[reranked_idx].clone().detach().to(hidden_state.device)


class RoadrunnerInferenceEngine:
    """
    Wraps a GPT-2 model with routed MLP, attention, and LM head modules.
    Provides token-by-token generation using matrix-free routing.

    Parameters:
        model_name (str): Name of the pretrained GPT-2 model to load.
        use_routing (bool): Whether to enable routing for MLP and attention layers.
        alpha (float): Blend ratio between routed and baseline outputs (1.0 = fully routed).
    """
    def __init__(self, model_name="gpt2", use_routing=True, alpha=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.alpha = alpha
        self.use_routing = use_routing
        self.svd_cache = {}  # SVD components per layer

        self._precompute_svd()

        self.mlp_blocks = []
        self.attn_blocks = []

        hidden_dim = self.model.config.n_embd

        for i, block in enumerate(self.model.transformer.h):
            mlp = RoadrunnerMLP(
                model=block,
                svd_mlp=self.svd_cache[i]["mlp"],
                alpha=self.alpha,
                enabled=use_routing
            )
            attn = RoadrunnerAttention(
                model=block,
                svd_attn=self.svd_cache[i]["attention"],
                hidden_dim=hidden_dim,
                alpha=self.alpha,
                enabled=use_routing
            )
            self.mlp_blocks.append(mlp)
            self.attn_blocks.append(attn)

        self.routed_lm_head = RoadrunnerLMHead(self.model)

    def _precompute_svd(self):
        """
        TEMP: Fills self.svd_cache with identity SVD components to test the routing pipeline.
        Replace with real SVD loading when available.
        """
        hidden_dim = self.model.config.n_embd
        head_dim = self.model.config.n_embd // self.model.config.n_head

        identity = torch.eye(hidden_dim).to(self.device)
        ones = torch.ones(hidden_dim).to(self.device)
        zeros = torch.zeros(hidden_dim).to(self.device)

        for i in range(len(self.model.transformer.h)):
            self.svd_cache[i] = {
                "mlp": load_svd_for_mlp(i),
                "attention": load_svd_for_attention(i)
            }


    def generate(self, prompt, max_new_tokens=20, verbose=False):
        """
        Generates text using token-by-token inference via routed MLP, attention, and LM head.

        Args:
            prompt (str): The initial text prompt.
            max_new_tokens (int): Number of tokens to generate.
            verbose (bool): If True, prints per-token debug info.

        Returns:
            str: Final decoded string after generation.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = input_ids.clone()

        past_key_values = [None] * len(self.model.transformer.h)

        for _ in range(max_new_tokens):
            x = output_ids[:, -1:] if output_ids.size(1) > 1 else output_ids

            # Run input through embeddings and position encodings
            hidden_states = self.model.transformer.wte(x) + self.model.transformer.wpe(
                torch.arange(output_ids.size(1), device=self.device)
            ).unsqueeze(0)

            for i, (attn, mlp) in enumerate(zip(self.attn_blocks, self.mlp_blocks)):
                hidden_states, past = attn.forward(hidden_states, past_key_values[i], verbose=True)
                past_key_values[i] = past
                hidden_states = mlp.forward(hidden_states, verbose=True)

            # Final layernorm
            hidden_states = self.model.transformer.ln_f(hidden_states)

            # Predict next token (dot-product LM head)
            next_token = self.routed_lm_head.predict(hidden_states[:, -1, :])

            # Append next token
            output_ids = torch.cat([output_ids, next_token.view(1, 1)], dim=-1)

            if verbose:
                print(f"[Token {output_ids.shape[1] - 1}] → {self.tokenizer.decode([next_token.item()])}")

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    import time

    # --- Config ---
    prompt = "The universe is"
    max_new_tokens = 5
    model_name = "gpt2"
    use_routing = True
    alpha = 0.5  # Reduced from 1.0 to 0.5 for better blending
    verbose = True

    # --- Initialize Engine ---
    print("🔧 Initializing RoadrunnerInferenceEngine...")
    engine = RoadrunnerInferenceEngine(
        model_name=model_name,
        use_routing=use_routing,
        alpha=alpha
    )

    print(f"🚀 Generating from prompt: \"{prompt}\"")
    start = time.time()
    output = engine.generate(prompt, max_new_tokens=max_new_tokens, verbose=verbose)
    end = time.time()

    print("\n🧾 === Final Output ===")
    print(output)
    print("======================")
    print(f"⌛ Time: {end - start:.2f} seconds")
    print(f"📏 Tokens generated: {max_new_tokens}")
    print(f"⚡ Avg token time: {(end - start)/max_new_tokens * 1000:.2f} ms/token")

if __name__ == "__main__":
    main()


""" Output:
🔧 Initializing RoadrunnerInferenceEngine...
🚀 Generating from prompt: "The universe is"
Attention routed/std diff: 166.3774
MLP routed/std diff: 2290.7380
Attention routed/std diff: 7552.2129
MLP routed/std diff: 174508.8125
Attention routed/std diff: 868487.7500
MLP routed/std diff: 38305600.0000
Attention routed/std diff: 63802660.0000
MLP routed/std diff: 2938678016.0000
Attention routed/std diff: 8957815808.0000
MLP routed/std diff: 293867487232.0000
Attention routed/std diff: 675440230400.0000
MLP routed/std diff: 5252364369920.0000
Attention routed/std diff: 24673540636672.0000
MLP routed/std diff: 235981005389824.0000
Attention routed/std diff: 1383850408148992.0000
MLP routed/std diff: 16642723394093056.0000
Attention routed/std diff: 57685761587150848.0000
MLP routed/std diff: 525971728111763456.0000
Attention routed/std diff: 2782073106162778112.0000
MLP routed/std diff: inf
Attention routed/std diff: inf
MLP routed/std diff: inf
Attention routed/std diff: nan
MLP routed/std diff: nan
[Token 3] → !
Attention routed/std diff: 89.0029
MLP routed/std diff: 2294.0310
Attention routed/std diff: 2010.4418
MLP routed/std diff: 23931.1016
Attention routed/std diff: 146872.3750
MLP routed/std diff: 4328860.5000
Attention routed/std diff: 27403438.0000
MLP routed/std diff: 383081824.0000
Attention routed/std diff: 2246363648.0000
MLP routed/std diff: 50542833664.0000
Attention routed/std diff: 226768371712.0000
MLP routed/std diff: 4298898931712.0000
Attention routed/std diff: 10591375720448.0000
MLP routed/std diff: 133187850207232.0000
Attention routed/std diff: 223776335724544.0000
MLP routed/std diff: 3230841635340288.0000
Attention routed/std diff: 19142415835201536.0000
MLP routed/std diff: 201073395087966208.0000
Attention routed/std diff: 745213659495530496.0000
MLP routed/std diff: 10021002851620225024.0000
Attention routed/std diff: inf
MLP routed/std diff: inf
Attention routed/std diff: nan
MLP routed/std diff: nan
[Token 4] → !
Attention routed/std diff: 39.7217
MLP routed/std diff: 973.7902
Attention routed/std diff: 1978.0308
MLP routed/std diff: 19549.8887
Attention routed/std diff: 163569.6719
MLP routed/std diff: 5237063.5000
Attention routed/std diff: 29091570.0000
MLP routed/std diff: 401983552.0000
Attention routed/std diff: 2305244160.0000
MLP routed/std diff: 51523719168.0000
Attention routed/std diff: 240918446080.0000
MLP routed/std diff: 4691707559936.0000
Attention routed/std diff: 5098313875456.0000
MLP routed/std diff: 74486376300544.0000
Attention routed/std diff: 240932968464384.0000
MLP routed/std diff: 3551503893659648.0000
Attention routed/std diff: 19047314521849856.0000
MLP routed/std diff: 207421906507268096.0000
Attention routed/std diff: 708459322241187840.0000
MLP routed/std diff: 10104522854377717760.0000
Attention routed/std diff: inf
MLP routed/std diff: inf
Attention routed/std diff: nan
MLP routed/std diff: nan
[Token 5] → !
Attention routed/std diff: 43.2535
MLP routed/std diff: 1077.7180
Attention routed/std diff: 2105.1973
MLP routed/std diff: 20566.4414
Attention routed/std diff: 176660.1094
MLP routed/std diff: 5800877.0000
Attention routed/std diff: 31106574.0000
MLP routed/std diff: 427036064.0000
Attention routed/std diff: 2436691200.0000
MLP routed/std diff: 55233871872.0000
Attention routed/std diff: 260368760832.0000
MLP routed/std diff: 5107625754624.0000
Attention routed/std diff: 5482489053184.0000
MLP routed/std diff: 79606304473088.0000
Attention routed/std diff: 261394326880256.0000
MLP routed/std diff: 3871865504268288.0000
Attention routed/std diff: 20168859331854336.0000
MLP routed/std diff: 222241141406302208.0000
Attention routed/std diff: 759188314845609984.0000
MLP routed/std diff: 11089542336353402880.0000
Attention routed/std diff: inf
MLP routed/std diff: inf
Attention routed/std diff: nan
MLP routed/std diff: nan
[Token 6] → !
Attention routed/std diff: 46.4271
MLP routed/std diff: 1172.7640
Attention routed/std diff: 2380.7771
MLP routed/std diff: 23278.9590
Attention routed/std diff: 185882.7812
MLP routed/std diff: 6036753.0000
Attention routed/std diff: 33041012.0000
MLP routed/std diff: 450127680.0000
Attention routed/std diff: 2561285376.0000
MLP routed/std diff: 58778226688.0000
Attention routed/std diff: 278420652032.0000
MLP routed/std diff: 5486585839616.0000
Attention routed/std diff: 5841357897728.0000
MLP routed/std diff: 84418932768768.0000
Attention routed/std diff: 283231718998016.0000
MLP routed/std diff: 4208921384321024.0000
Attention routed/std diff: 21436484569530368.0000
MLP routed/std diff: 236990900914356224.0000
Attention routed/std diff: 804226372703617024.0000
MLP routed/std diff: 11878436431724544000.0000
Attention routed/std diff: inf
MLP routed/std diff: inf
Attention routed/std diff: nan
MLP routed/std diff: nan
[Token 7] → !

🧾 === Final Output ===
The universe is!!!!!
======================
⌛ Time: 0.21 seconds
📏 Tokens generated: 5
⚡ Avg token time: 41.14 ms/token
"""