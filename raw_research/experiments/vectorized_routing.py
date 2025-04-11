import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FastRoutingLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, top_k: int = 32):
        super().__init__()
        self.top_k = top_k
        self.out_dim, self.in_dim = weight.shape

        self.weight_raw = weight.detach()  # [out_dim, in_dim]
        self.weight_norm = torch.nn.functional.normalize(weight, dim=1).detach()

        self.bias = bias.detach() if bias is not None else None

    def forward(self, x: torch.Tensor):
        x_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_dim)  # [B*T, in_dim]
        x_norm = torch.nn.functional.normalize(x_flat, dim=1)  # [B*T, in_dim]

        # Step 1: Top-k token IDs from cosine similarity
        logits = torch.matmul(x_norm, self.weight_norm.T)  # [B*T, out_dim]
        topk_vals, topk_ids = torch.topk(logits, self.top_k, dim=1)  # [B*T, k]

        # Step 2: Gather weights and rerank in parallel
        W_topk = self.weight_raw[topk_ids]  # [B*T, k, in_dim]
        x_exp = x_flat.unsqueeze(1)         # [B*T, 1, in_dim]
        local_logits = torch.sum(W_topk * x_exp, dim=-1)  # [B*T, k]

        if self.bias is not None:
            local_logits += self.bias[topk_ids]  # [B*T, k]

        # Step 3: Sparse full vector with only top-k filled
        output = torch.zeros(x_flat.size(0), self.out_dim, device=x.device)  # [B*T, out_dim]
        output.scatter_(1, topk_ids, local_logits)

        return output.view(*x_shape, self.out_dim)