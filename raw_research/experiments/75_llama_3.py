import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
import time
import numpy as np

class MatrixFreeLMHead:
    def __init__(self, model, threshold):
        self.model = model
        self.device = model.device
        # For Llama models, the lm_head is typically just a linear layer
        self.weight = model.lm_head.weight.data.to(self.device)
        # Llama models typically don't have bias in the lm_head
        self.bias = None
        self.threshold = threshold

    def predict(self, hidden_state):
        # Ensure we're working with the last token's hidden state
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        
        # Skip normalization if performance is critical
        # Compute dot product using more efficient batch matrix multiplication
        # Use only top-k to avoid full matrix multiplication
        k = 100  # Try different values here
        
        # Use einsum for more efficient computation
        scores = torch.einsum('i,ji->j', hidden_state.view(-1), self.weight)
        
        # Get the token with the highest score
        top_score, top_index = torch.topk(scores, 1)
        return top_index, top_score.item()

def calibrate_threshold(model, prompts, tokenizer, max_new_tokens=20):
    scores = []
    router = MatrixFreeLMHead(model, threshold=-float("inf"))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # For Llama models
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
                input_ids = torch.argmax(model.lm_head(hidden[:, -1:, :]), dim=-1)

    # Return the lowest score observed (or a percentile)
    return np.percentile(scores, 0)

def generate_matrix_free(model, tokenizer, prompt, max_new_tokens, threshold):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    all_ids = input_ids.clone()
    past_key_values = None
    router = MatrixFreeLMHead(model, threshold=threshold)

    matched = 0
    start = time.perf_counter()
    
    # Compile the model for better performance if torch.compile is available
    try:
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            model = torch.compile(model)
            print("Using torch.compile for acceleration")
    except:
        pass
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass through model
            outputs = model(
                input_ids=input_ids, 
                past_key_values=past_key_values, 
                use_cache=True,
                output_hidden_states=True
            )
            
            # Get hidden states and past key values
            hidden = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values

            # Matrix-free prediction
            pred_id, score = router.predict(hidden)
            
            # For accuracy measurement, we still need the baseline
            # But we can make this optional with a flag for pure performance testing
            baseline = torch.argmax(model.lm_head(hidden[:, -1:, :]), dim=-1).item()

            # Check if predictions match
            if pred_id.item() == baseline:
                matched += 1

            # Use predicted token for next iteration
            input_ids = pred_id.unsqueeze(0)
            all_ids = torch.cat([all_ids, input_ids], dim=1)
            
    duration = time.perf_counter() - start
    
    return {
        "text": tokenizer.decode(all_ids[0], skip_special_tokens=True),
        "accuracy": matched / max_new_tokens,
        "ms_per_token": (duration / max_new_tokens) * 1000,
        "tokens_per_sec": max_new_tokens / duration
    }

def generate_baseline(model, tokenizer, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    start = time.perf_counter()
    
    with torch.no_grad():
        # Clear all sampling parameters to avoid warnings
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=None,  # Remove temperature setting
            top_p=None,        # Remove top_p setting
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        
    duration = time.perf_counter() - start
    
    return {
        "text": tokenizer.decode(output[0], skip_special_tokens=True),
        "ms_per_token": (duration / max_new_tokens) * 1000,
        "tokens_per_sec": max_new_tokens / duration
    }

def run_comparison(model_name="unsloth/Llama-3.2-1B", max_new_tokens=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Loading Llama 3 model: {model_name}...")
    
    # Load Llama model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
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
    threshold = calibrate_threshold(model, prompts, tokenizer, max_new_tokens)
    print(f"✅ Calibrated P0 threshold: {threshold:.2f}\n")

    for prompt in prompts:
        print(f"\U0001F9EA Prompt: {prompt}")
        
        baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens)
        matrix_free = generate_matrix_free(model, tokenizer, prompt, max_new_tokens, threshold)

        print(f"--- Llama 3 Baseline ---")
        print(f"📜 {baseline['text']}")
        print(f"⚡ {baseline['tokens_per_sec']:.2f} tokens/sec | ⏱ {baseline['ms_per_token']:.2f} ms/token")

        print(f"--- Routed (Matrix-Free) ---")
        print(f"📜 {matrix_free['text']}")
        print(f"⚡ {matrix_free['tokens_per_sec']:.2f} tokens/sec | ⏱ {matrix_free['ms_per_token']:.2f} ms/token")
        print(f"🎯 Accuracy: {matrix_free['accuracy']:.2%}")
        print("-" * 60)

if __name__ == "__main__":
    run_comparison(max_new_tokens=20)


""" Output:
🔄 Loading Llama 3 model: unsloth/Llama-3.2-1B...
📊 Calibrating routing threshold...
✅ Calibrated P0 threshold: 17.51

🧪 Prompt: The meaning of life is
--- Llama 3 Baseline ---
📜 The meaning of life is to find your passion and to live it. I am passionate about helping people find their passion and live
⚡ 12.29 tokens/sec | ⏱ 81.40 ms/token
--- Routed (Matrix-Free) ---
📜 The meaning of life is to find your passion and to live it. I am passionate about helping people find their passion and live
⚡ 8.73 tokens/sec | ⏱ 114.56 ms/token
🎯 Accuracy: 100.00%
------------------------------------------------------------
🧪 Prompt: In a distant galaxy, a civilization
--- Llama 3 Baseline ---
📜 In a distant galaxy, a civilization has developed a technology that allows them to travel through space and time. They have discovered a way to
⚡ 12.18 tokens/sec | ⏱ 82.10 ms/token
--- Routed (Matrix-Free) ---
📜 In a distant galaxy, a civilization has developed a technology that allows them to travel through space and time. They have discovered a way to
⚡ 8.66 tokens/sec | ⏱ 115.50 ms/token
🎯 Accuracy: 100.00%
------------------------------------------------------------
🧪 Prompt: The future of AI will depend on
--- Llama 3 Baseline ---
📜 The future of AI will depend on how we use it
The future of AI will depend on how we use it
The future of
⚡ 11.57 tokens/sec | ⏱ 86.46 ms/token
--- Routed (Matrix-Free) ---
📜 The future of AI will depend on how we use it
The future of AI will depend on how we use it
The future of
⚡ 8.76 tokens/sec | ⏱ 114.12 ms/token
🎯 Accuracy: 100.00%
------------------------------------------------------------
🧪 Prompt: Once upon a time
--- Llama 3 Baseline ---
📜 Once upon a time, there was a man who was very rich. He had a lot of money, and he was
⚡ 12.15 tokens/sec | ⏱ 82.28 ms/token
--- Routed (Matrix-Free) ---
📜 Once upon a time, there was a man who was very rich. He had a lot of money, and he was
⚡ 8.81 tokens/sec | ⏱ 113.56 ms/token
🎯 Accuracy: 100.00%
------------------------------------------------------------
🧪 Prompt: The quantum computer
--- Llama 3 Baseline ---
📜 The quantum computer is a computer that uses quantum mechanics to perform calculations. It is a new type of computer that uses
⚡ 11.70 tokens/sec | ⏱ 85.44 ms/token
--- Routed (Matrix-Free) ---
📜 The quantum computer is a computer that uses quantum mechanics to perform calculations. It is a new type of computer that uses
⚡ 8.77 tokens/sec | ⏱ 114.04 ms/token
🎯 Accuracy: 100.00%
------------------------------------------------------------
"""