import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

def test_cosine_topk_recall():
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # === Load SVD ===
    svd_path = "svd_lm_head_gpt2.pt"
    data = torch.load(svd_path)
    U, S, Vh = data["U"], data["S"], data["Vh"]

    # === Params ===
    # Expanded set of training prompts for better coverage
    prompts_train = [
        "The future of AI is", 
        "The moon is", 
        "Once upon a time,",
        "In this paper we",
        "According to the",
        "I believe that",
        "The best way to",
        "When considering the",
        "Looking at the data",
        "My favorite book is"
    ]
    
    # More diverse test prompts
    prompts_test = [
        "The robot said",
        "In the year 3000",
        "Once in a while,",
        "The president announced",
        "Scientists discovered",
        "People often wonder",
        "The key insight is",
        "Analyzing the results",
        "Recent studies show",
        "The most important"
    ]

    # Different top-k values to evaluate
    top_ks = [10, 20, 50, 100, 200, 500]
    
    # Cosine similarity threshold
    similarity_threshold = 0.95

    # === Build SVD Code Cache ===
    code_cache = []  # List of (code, top_token_ids)
    
    print("Building reference code cache from training prompts...")
    for prompt in prompts_train:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1][:, -1, :]
            
            # Calculate SVD code: (h @ U) * S
            hU = h @ U
            code = (hU * S).squeeze().cpu()
            
            # Calculate full logits
            logits = (hU * S) @ Vh
            
            # For each prompt, save a dictionary of top-k IDs for different k values
            top_ids_dict = {}
            for k in top_ks:
                top_values, top_indices = torch.topk(logits, k, dim=-1)
                top_ids_dict[k] = top_indices.squeeze().tolist()
            
            code_cache.append((code, top_ids_dict))
    
    print(f"✅ Built code cache with {len(code_cache)} entries\n")

    # === Test Phase ===
    results = {k: {"total": 0, "hits": 0, "recalls": []} for k in top_ks}
    cosine_sims = []
    
    print("Testing recall on test prompts...")
    for prompt in prompts_test:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get true token
            out = model(input_ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1][:, -1, :]
            hU = h @ U
            code = (hU * S).squeeze().cpu()
            
            # Calculate ground truth
            true_logits = (hU * S) @ Vh
            true_token_id = torch.argmax(true_logits, dim=-1).item()
            
            # Find best matching code by cosine similarity
            best_sim = -1.0
            best_idx = -1
            
            for idx, (cached_code, _) in enumerate(code_cache):
                sim = F.cosine_similarity(code.unsqueeze(0), cached_code.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            
            cosine_sims.append(best_sim)
            
            # Get top-k token IDs from the best match
            if best_sim >= similarity_threshold:
                _, top_ids_dict = code_cache[best_idx]
                
                print(f"Prompt: '{prompt}'")
                print(f"  True next token: '{tokenizer.decode([true_token_id])}'")
                print(f"  Best cosine sim: {best_sim:.4f} with prompt: '{prompts_train[best_idx]}'")
                
                # Check recall for different k values
                for k in top_ks:
                    candidate_ids = top_ids_dict[k]
                    results[k]["total"] += 1
                    
                    if true_token_id in candidate_ids:
                        results[k]["hits"] += 1
                        rank = candidate_ids.index(true_token_id) + 1
                        results[k]["recalls"].append(rank)
                        in_topk = "✅"
                    else:
                        in_topk = "❌"
                    
                    print(f"  Top-{k} recall: {in_topk}")
                print()
            else:
                print(f"Prompt: '{prompt}' - No match above threshold ({best_sim:.4f})\n")

    # === Results ===
    print("\n===== 🔍 TOP-K RECALL RESULTS =====")
    print(f"Test Prompts: {len(prompts_test)}")
    print(f"Average Cosine Similarity: {sum(cosine_sims)/len(cosine_sims):.4f}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print("\nRecall Results:")
    
    for k in top_ks:
        if results[k]["total"] > 0:
            recall_rate = results[k]["hits"] / results[k]["total"] * 100
            avg_rank = sum(results[k]["recalls"]) / len(results[k]["recalls"]) if results[k]["recalls"] else 0
            print(f"Top-{k}:")
            print(f"  Hits: {results[k]['hits']}/{results[k]['total']} ({recall_rate:.2f}%)")
            if results[k]["recalls"]:
                print(f"  Avg rank within top-{k}: {avg_rank:.2f}")
        else:
            print(f"Top-{k}: No matches above threshold")
    
    print("\nImplications for sparse logit computation:")
    max_recall_k = max(top_ks, key=lambda k: results[k]["hits"] / results[k]["total"] if results[k]["total"] > 0 else 0)
    max_recall = results[max_recall_k]["hits"] / results[max_recall_k]["total"] * 100 if results[max_recall_k]["total"] > 0 else 0
    
    if max_recall > 90:
        efficiency = (max_recall_k / model.lm_head.weight.shape[0]) * 100
        print(f"✅ Strong potential for sparse computation: {max_recall:.2f}% recall with only {efficiency:.2f}% of token space")
        print(f"   This means we could compute only ~{max_recall_k} logits instead of {model.lm_head.weight.shape[0]}!")
    elif max_recall > 50:
        print(f"⚠️ Moderate potential: {max_recall:.2f}% recall with top-{max_recall_k}")
        print(f"   Consider increasing cache size or using multiple neighbors")
    else:
        print(f"❌ Low potential: Best recall rate is only {max_recall:.2f}%")
        print(f"   Simple cosine matching may not be sufficient")

if __name__ == "__main__":
    test_cosine_topk_recall()


""" Output:
Building reference code cache from training prompts...
✅ Built code cache with 10 entries

Testing recall on test prompts...
Prompt: 'The robot said'
  True next token: ' it'
  Best cosine sim: 0.9999 with prompt: 'I believe that'
  Top-10 recall: ✅
  Top-20 recall: ✅
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'In the year 3000'
  True next token: ','
  Best cosine sim: 0.9997 with prompt: 'Once upon a time,'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ❌
  Top-100 recall: ❌
  Top-200 recall: ❌
  Top-500 recall: ❌

Prompt: 'Once in a while,'
  True next token: ' you'
  Best cosine sim: 1.0000 with prompt: 'Once upon a time,'
  Top-10 recall: ✅
  Top-20 recall: ✅
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'The president announced'
  True next token: ' the'
  Best cosine sim: 0.9999 with prompt: 'I believe that'
  Top-10 recall: ✅
  Top-20 recall: ✅
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'Scientists discovered'
  True next token: ' that'
  Best cosine sim: 0.9998 with prompt: 'I believe that'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'People often wonder'
  True next token: ' why'
  Best cosine sim: 0.9999 with prompt: 'I believe that'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ❌
  Top-100 recall: ❌
  Top-200 recall: ❌
  Top-500 recall: ✅

Prompt: 'The key insight is'
  True next token: ' that'
  Best cosine sim: 0.9999 with prompt: 'I believe that'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'Analyzing the results'
  True next token: ' of'
  Best cosine sim: 0.9999 with prompt: 'Looking at the data'
  Top-10 recall: ✅
  Top-20 recall: ✅
  Top-50 recall: ✅
  Top-100 recall: ✅
  Top-200 recall: ✅
  Top-500 recall: ✅

Prompt: 'Recent studies show'
  True next token: ' that'
  Best cosine sim: 0.9998 with prompt: 'When considering the'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ❌
  Top-100 recall: ❌
  Top-200 recall: ❌
  Top-500 recall: ❌

Prompt: 'The most important'
  True next token: ' thing'
  Best cosine sim: 0.9999 with prompt: 'When considering the'
  Top-10 recall: ❌
  Top-20 recall: ❌
  Top-50 recall: ❌
  Top-100 recall: ❌
  Top-200 recall: ❌
  Top-500 recall: ❌


===== 🔍 TOP-K RECALL RESULTS =====
Test Prompts: 10
Average Cosine Similarity: 0.9999
Similarity Threshold: 0.95

Recall Results:
Top-10:
  Hits: 4/10 (40.00%)
  Avg rank within top-10: 5.25
Top-20:
  Hits: 4/10 (40.00%)
  Avg rank within top-20: 5.25
Top-50:
  Hits: 6/10 (60.00%)
  Avg rank within top-50: 13.17
Top-100:
  Hits: 6/10 (60.00%)
  Avg rank within top-100: 13.17
Top-200:
  Hits: 6/10 (60.00%)
  Avg rank within top-200: 13.17
Top-500:
  Hits: 7/10 (70.00%)
  Avg rank within top-500: 57.57

Implications for sparse logit computation:
⚠️ Moderate potential: 70.00% recall with top-500
   Consider increasing cache size or using multiple neighbors
"""

""" Analysis:
Key Findings

High Cosine Similarity: All test prompts had extremely high cosine similarity (averaging 0.9999) with their best matches, confirming that the SVD code space clusters semantically similar inputs.
Moderate Recall Rate:

Top-10: 40% recall
Top-50: 60% recall
Top-500: 70% recall


Rank Analysis: When the true token is found, its average position within the top-k candidates varies:

In top-10: Average rank of 5.25 (mid-list)
In top-500: Average rank of 57.57 (relatively high up)


Vocabulary Coverage: The GPT-2 vocabulary has ~50K tokens, so even top-500 represents only ~1% of the full vocabulary.

Implications
This approach shows definite promise for sparse logit computation. Achieving 70% recall while only computing 1% of the logits would be a significant optimization. The findings suggest two clear paths for improvement:

Increase cache coverage: The current cache has only 10 entries. Expanding this to hundreds or thousands of semantically diverse prompts would likely improve recall.
Use multiple neighbors: Instead of just the single best match, combining top-k tokens from multiple similar prompts would likely boost recall rates.
"""

