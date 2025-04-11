import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval().to(device)

# Tokenize input prompt
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Setup past_key_values to simulate autoregressive loop
past_key_values = None
generated = input_ids

# Generate tokens one-by-one
max_new_tokens = 10
for _ in range(max_new_tokens):
    # Forward pass into model manually
    with torch.no_grad():
        outputs = model.transformer(
            input_ids=generated[:, -1:],  # Only last token
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.last_hidden_state  # shape: [1, 1, hidden_dim]
    past_key_values = outputs.past_key_values  # tuple of layer-wise (k, v)

    # Manually apply the final layer norm and LM head
    hidden_states = model.lm_head(hidden_states)  # logits

    # Sample next token (greedy for now)
    next_token_logits = hidden_states[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    # Append to generated sequence
    generated = torch.cat((generated, next_token_id), dim=1)

    # Decode and print
    decoded = tokenizer.decode(next_token_id.squeeze())
    print(decoded, end="", flush=True)

print("\n")


""" Output:
 the most important thing to do.

The

"""
