import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json

class SparseLogitTestSuite:
    def __init__(self, model_name="gpt2", svd_path="svd_lm_head_gpt2.pt", device=None):
        """Initialize the test suite with model and SVD components."""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vocab_size = self.model.lm_head.weight.shape[0]
        
        # Load SVD components
        print(f"Loading SVD from {svd_path}...")
        data = torch.load(svd_path)
        self.U = data["U"].to(self.device)
        self.S = data["S"].to(self.device)
        self.Vh = data["Vh"].to(self.device)
        
        # Cache and results storage
        self.code_cache = []  # [(code, prompt, top_ids_dict), ...]
        self.test_results = {}
        
        print(f"Initialized with vocabulary size: {self.vocab_size}")
    
    def generate_diverse_prompts(self, num_train=50, num_test=20, seed=42):
        """Generate diverse training and test prompts."""
        random.seed(seed)
        
        # Templates for generating diverse prompts
        templates = [
            "The future of {topic} is",
            "When {person} thinks about {topic}, they",
            "{Person} believes that {topic}",
            "The best way to {action} is",
            "In the context of {topic}, we",
            "Many people wonder why {topic}",
            "The relationship between {topic} and {topic2} suggests",
            "Looking at the data on {topic}, we see",
            "The main challenge with {topic} is",
            "Scientists studying {topic} have discovered",
            "If we consider {topic} from the perspective of {topic2},",
            "The key insight about {topic} is",
            "Most experts agree that {topic}",
            "Analyzing the results of {topic} research shows",
            "The historical development of {topic} indicates",
        ]
        
        topics = ["AI", "technology", "science", "education", "economics", "politics", 
                  "climate change", "healthcare", "social media", "psychology", 
                  "philosophy", "art", "literature", "history", "mathematics",
                  "biology", "physics", "chemistry", "engineering", "business"]
        
        actions = ["learn", "solve problems", "improve efficiency", "understand concepts",
                   "innovate", "collaborate", "analyze data", "make decisions", 
                   "communicate", "design solutions", "implement changes", "evaluate outcomes"]
        
        people = ["a researcher", "a student", "an expert", "a beginner", "a critic", 
                  "a professional", "a teacher", "an enthusiast", "a skeptic",
                  "a leader", "a manager", "a policymaker", "a consumer"]
        
        # Generate prompts
        all_prompts = []
        for _ in range(max(num_train + num_test, 100)):  # Generate extras to ensure diversity
            template = random.choice(templates)
            format_args = {}
            
            if "{topic}" in template:
                format_args["topic"] = random.choice(topics)
            
            if "{topic2}" in template:
                format_args["topic2"] = random.choice([t for t in topics if t != format_args.get("topic")])
            
            if "{action}" in template:
                format_args["action"] = random.choice(actions)
            
            if "{person}" in template:
                format_args["person"] = random.choice(people)
            
            if "{Person}" in template:
                format_args["Person"] = random.choice(people).capitalize()
            
            prompt = template.format(**format_args)
            if prompt not in all_prompts:
                all_prompts.append(prompt)
        
        # Ensure we have enough unique prompts
        assert len(all_prompts) >= num_train + num_test, "Not enough unique prompts generated"
        
        # Split into train and test sets
        train_prompts = all_prompts[:num_train]
        test_prompts = all_prompts[num_train:num_train + num_test]
        
        print(f"Generated {len(train_prompts)} training prompts and {len(test_prompts)} test prompts")
        return train_prompts, test_prompts
    
    def compute_svd_code(self, prompt):
        """Compute the SVD code for a prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1][:, -1, :]
            
            # Calculate SVD code: (h @ U) * S
            hU = h @ self.U
            code = (hU * self.S).squeeze()
            
            # Calculate logits
            logits = (hU * self.S) @ self.Vh
            true_token_id = torch.argmax(logits, dim=-1).item()
            
            return {
                "code": code,
                "logits": logits.squeeze(),
                "true_token_id": true_token_id,
                "true_token": self.tokenizer.decode([true_token_id])
            }
    
    def build_code_cache(self, prompts, top_ks=[10, 20, 50, 100, 200, 500, 1000]):
        """Build cache of SVD codes and top-k token IDs for training prompts."""
        self.code_cache = []
        self.top_ks = top_ks
        
        print(f"Building code cache from {len(prompts)} prompts...")
        for prompt in tqdm(prompts):
            result = self.compute_svd_code(prompt)
            
            # Store top-k token IDs for various k values
            top_ids_dict = {}
            for k in top_ks:
                top_values, top_indices = torch.topk(result["logits"], k)
                top_ids_dict[k] = top_indices.cpu().tolist()
            
            self.code_cache.append({
                "code": result["code"],
                "prompt": prompt,
                "top_ids_dict": top_ids_dict,
                "true_token_id": result["true_token_id"],
                "true_token": result["true_token"]
            })
        
        print(f"✅ Built code cache with {len(self.code_cache)} entries")
    
    def find_similar_codes(self, code, n=5):
        """Find the n most similar codes in the cache by cosine similarity."""
        similarities = []
        
        for entry in self.code_cache:
            cached_code = entry["code"]
            sim = F.cosine_similarity(code.unsqueeze(0), cached_code.unsqueeze(0)).item()
            similarities.append((sim, entry))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:n]
    
    def test_single_neighbor(self, test_prompts):
        """Test approach 1: Use the single most similar cached prompt."""
        results = {k: {"total": 0, "hits": 0, "recalls": []} for k in self.top_ks}
        cosine_sims = []
        details = []
        
        print("Testing single neighbor approach...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            
            # Find most similar cached code
            similarities = self.find_similar_codes(code, n=1)
            best_sim, best_entry = similarities[0]
            cosine_sims.append(best_sim)
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "best_match": best_entry["prompt"],
                "similarity": best_sim,
                "recall": {}
            }
            
            # Check recall for different k values
            for k in self.top_ks:
                results[k]["total"] += 1
                candidate_ids = best_entry["top_ids_dict"][k]
                
                if true_token_id in candidate_ids:
                    results[k]["hits"] += 1
                    rank = candidate_ids.index(true_token_id) + 1
                    results[k]["recalls"].append(rank)
                    prompt_detail["recall"][k] = rank
                else:
                    prompt_detail["recall"][k] = None
            
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "single_neighbor",
            "avg_cosine_sim": sum(cosine_sims) / len(cosine_sims),
            "recall_rates": {},
            "avg_ranks": {}
        }
        
        for k in self.top_ks:
            if results[k]["total"] > 0:
                recall_rate = results[k]["hits"] / results[k]["total"] * 100
                summary["recall_rates"][k] = recall_rate
                
                if results[k]["recalls"]:
                    avg_rank = sum(results[k]["recalls"]) / len(results[k]["recalls"])
                    summary["avg_ranks"][k] = avg_rank
                else:
                    summary["avg_ranks"][k] = None
        
        return {"summary": summary, "details": details}
    
    def test_ensemble_approach(self, test_prompts, n_neighbors=3):
        """Test approach 2: Take union of top-k tokens from N neighbors."""
        results = {k: {"total": 0, "hits": 0} for k in self.top_ks}
        cosine_sims = []
        details = []
        
        print(f"Testing ensemble approach with {n_neighbors} neighbors...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            
            # Find N most similar cached codes
            similarities = self.find_similar_codes(code, n=n_neighbors)
            avg_sim = sum(sim for sim, _ in similarities) / len(similarities)
            cosine_sims.append(avg_sim)
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "neighbors": [{"prompt": entry["prompt"], "similarity": sim} 
                             for sim, entry in similarities],
                "avg_similarity": avg_sim,
                "recall": {}
            }
            
            # Check recall for different k values
            for k in self.top_ks:
                results[k]["total"] += 1
                
                # Union of top-k tokens from all neighbors
                all_candidates = set()
                for _, entry in similarities:
                    all_candidates.update(entry["top_ids_dict"][k])
                
                if true_token_id in all_candidates:
                    results[k]["hits"] += 1
                    prompt_detail["recall"][k] = True
                else:
                    prompt_detail["recall"][k] = False
                
                # Also store the ensemble size
                prompt_detail[f"ensemble_size_{k}"] = len(all_candidates)
            
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "ensemble",
            "n_neighbors": n_neighbors,
            "avg_cosine_sim": sum(cosine_sims) / len(cosine_sims),
            "recall_rates": {},
            "avg_ensemble_sizes": {}
        }
        
        for k in self.top_ks:
            if results[k]["total"] > 0:
                recall_rate = results[k]["hits"] / results[k]["total"] * 100
                summary["recall_rates"][k] = recall_rate
                
                avg_size = sum(detail[f"ensemble_size_{k}"] for detail in details) / len(details)
                summary["avg_ensemble_sizes"][k] = avg_size
        
        return {"summary": summary, "details": details}
    
    def test_weighted_voting(self, test_prompts, n_neighbors=3, weight_factor=1.0):
        """Test approach 3: Weight token contributions by cosine similarity."""
        results = {k: {"total": 0, "hits": 0, "recalls": []} for k in self.top_ks}
        details = []
        
        print(f"Testing weighted voting with {n_neighbors} neighbors...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            
            # Find N most similar cached codes
            similarities = self.find_similar_codes(code, n=n_neighbors)
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "neighbors": [{"prompt": entry["prompt"], "similarity": sim} 
                             for sim, entry in similarities],
                "recall": {}
            }
            
            # Check recall for different k values
            for k in self.top_ks:
                results[k]["total"] += 1
                
                # Apply weighted voting to all neighbors' top-k tokens
                token_scores = defaultdict(float)
                
                for sim, entry in similarities:
                    # Apply weight factor to similarity for more aggressive weighting
                    weight = sim ** weight_factor
                    
                    for token_id in entry["top_ids_dict"][k]:
                        token_scores[token_id] += weight
                
                # Get top-k tokens by weighted score
                sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
                top_tokens = [token_id for token_id, _ in sorted_tokens[:k]]
                
                if true_token_id in top_tokens:
                    results[k]["hits"] += 1
                    rank = top_tokens.index(true_token_id) + 1
                    results[k]["recalls"].append(rank)
                    prompt_detail["recall"][k] = rank
                else:
                    prompt_detail["recall"][k] = None
            
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "weighted_voting",
            "n_neighbors": n_neighbors,
            "weight_factor": weight_factor,
            "recall_rates": {},
            "avg_ranks": {}
        }
        
        for k in self.top_ks:
            if results[k]["total"] > 0:
                recall_rate = results[k]["hits"] / results[k]["total"] * 100
                summary["recall_rates"][k] = recall_rate
                
                if results[k]["recalls"]:
                    avg_rank = sum(results[k]["recalls"]) / len(results[k]["recalls"])
                    summary["avg_ranks"][k] = avg_rank
                else:
                    summary["avg_ranks"][k] = None
        
        return {"summary": summary, "details": details}
    
    def test_adaptive_token_selection(self, test_prompts, similarity_thresholds=[0.995, 0.99, 0.98, 0.97, 0.96, 0.95]):
        """Test approach 4: Adaptively adjust k based on similarity."""
        results = {"total": 0, "hits": 0, "tokens_evaluated": []}
        details = []
        
        print("Testing adaptive token selection...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            
            # Find most similar cached code
            similarities = self.find_similar_codes(code, n=1)
            best_sim, best_entry = similarities[0]
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "best_match": best_entry["prompt"],
                "similarity": best_sim
            }
            
            # Determine k based on similarity
            selected_k = None
            for i, threshold in enumerate(similarity_thresholds):
                if best_sim >= threshold:
                    selected_k = self.top_ks[min(i, len(self.top_ks)-1)]
                    break
            
            # If none of the thresholds matched, use the largest k
            if selected_k is None:
                selected_k = self.top_ks[-1]
            
            results["total"] += 1
            results["tokens_evaluated"].append(selected_k)
            
            # Check if true token is in the selected top-k
            candidate_ids = best_entry["top_ids_dict"][selected_k]
            
            if true_token_id in candidate_ids:
                results["hits"] += 1
                rank = candidate_ids.index(true_token_id) + 1
                prompt_detail["hit"] = True
                prompt_detail["rank"] = rank
            else:
                prompt_detail["hit"] = False
                prompt_detail["rank"] = None
            
            prompt_detail["selected_k"] = selected_k
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "adaptive_selection",
            "similarity_thresholds": similarity_thresholds,
            "recall_rate": results["hits"] / results["total"] * 100 if results["total"] > 0 else 0,
            "avg_tokens_evaluated": sum(results["tokens_evaluated"]) / len(results["tokens_evaluated"]) if results["tokens_evaluated"] else 0
        }
        
        return {"summary": summary, "details": details}
    
    def test_vocabulary_clustering(self, test_prompts, n_clusters=50):
        """Test approach 5: Pre-cluster the vocabulary and select relevant clusters."""
        # This is a simplified version that simulates vocabulary clustering
        # In practice, you would cluster the Vh matrix columns to group similar tokens
        
        results = {"total": 0, "hits": 0, "clusters_evaluated": []}
        details = []
        
        # Create simulated clusters (in practice, you'd use K-means or similar)
        # For this test, we'll create random clusters
        clusters = {}
        token_to_cluster = {}
        
        # Assign each token to a random cluster
        for token_id in range(self.vocab_size):
            cluster_id = random.randint(0, n_clusters - 1)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(token_id)
            token_to_cluster[token_id] = cluster_id
        
        print(f"Testing vocabulary clustering with {n_clusters} clusters...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            true_cluster = token_to_cluster[true_token_id]
            
            # Find most similar cached code
            similarities = self.find_similar_codes(code, n=1)
            best_sim, best_entry = similarities[0]
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "best_match": best_entry["prompt"],
                "similarity": best_sim
            }
            
            # Find which clusters are represented in the top tokens
            top_tokens = best_entry["top_ids_dict"][self.top_ks[-1]]  # Use largest k
            cluster_counts = Counter(token_to_cluster[token_id] for token_id in top_tokens)
            
            # Select the top-3 most represented clusters
            top_clusters = [cluster for cluster, _ in cluster_counts.most_common(3)]
            
            # Collect all tokens from the selected clusters
            candidate_tokens = []
            for cluster_id in top_clusters:
                candidate_tokens.extend(clusters[cluster_id])
            
            results["total"] += 1
            results["clusters_evaluated"].append(len(top_clusters))
            
            # Check if true token is in the candidate set
            if true_token_id in candidate_tokens:
                results["hits"] += 1
                prompt_detail["hit"] = True
                prompt_detail["true_cluster_selected"] = true_cluster in top_clusters
            else:
                prompt_detail["hit"] = False
                prompt_detail["true_cluster_selected"] = False
            
            prompt_detail["top_clusters"] = top_clusters
            prompt_detail["candidate_set_size"] = len(candidate_tokens)
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "vocabulary_clustering",
            "n_clusters": n_clusters,
            "recall_rate": results["hits"] / results["total"] * 100 if results["total"] > 0 else 0,
            "avg_clusters_evaluated": sum(results["clusters_evaluated"]) / len(results["clusters_evaluated"]),
            "avg_candidate_set_size": sum(d["candidate_set_size"] for d in details) / len(details)
        }
        
        return {"summary": summary, "details": details}
    
    def test_two_stage_filtering(self, test_prompts, first_stage_k=500, second_stage_k=50):
        """Test approach 6: Two-stage filtering process."""
        results = {"total": 0, "hits": 0, "first_stage_hits": 0, "second_stage_hits": 0}
        details = []
        
        print("Testing two-stage filtering...")
        for prompt in tqdm(test_prompts):
            # Get true token and code
            test_result = self.compute_svd_code(prompt)
            code = test_result["code"]
            true_token_id = test_result["true_token_id"]
            
            # Find most similar cached code
            similarities = self.find_similar_codes(code, n=1)
            best_sim, best_entry = similarities[0]
            
            prompt_detail = {
                "prompt": prompt,
                "true_token": test_result["true_token"],
                "best_match": best_entry["prompt"],
                "similarity": best_sim
            }
            
            # Stage 1: Get top-k candidates from the cache
            first_stage_candidates = best_entry["top_ids_dict"][first_stage_k]
            first_stage_hit = true_token_id in first_stage_candidates
            
            if first_stage_hit:
                results["first_stage_hits"] += 1
            
            # Stage 2: Compute exact logits for the first-stage candidates
            # In a real implementation, you'd compute h @ W[:, candidates]
            # Here we'll simulate by getting the true logits and masking
            
            # Get the logits for these candidates from the true distribution
            true_logits = test_result["logits"]
            
            # Create a mask for the first-stage candidates
            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            mask[first_stage_candidates] = True
            
            # Apply mask and get top-k from the filtered logits
            masked_logits = true_logits.clone()
            masked_logits[~mask] = -float('inf')
            
            _, second_stage_indices = torch.topk(masked_logits, second_stage_k)
            second_stage_candidates = second_stage_indices.cpu().tolist()
            
            second_stage_hit = true_token_id in second_stage_candidates
            
            results["total"] += 1
            if second_stage_hit:
                results["hits"] += 1
                results["second_stage_hits"] += 1
            
            prompt_detail["first_stage_hit"] = first_stage_hit
            prompt_detail["second_stage_hit"] = second_stage_hit
            
            details.append(prompt_detail)
        
        # Calculate summary metrics
        summary = {
            "method": "two_stage_filtering",
            "first_stage_k": first_stage_k,
            "second_stage_k": second_stage_k,
            "first_stage_recall": results["first_stage_hits"] / results["total"] * 100,
            "second_stage_recall": results["second_stage_hits"] / results["total"] * 100,
            "overall_recall": results["hits"] / results["total"] * 100,
            "computation_ratio": (first_stage_k + second_stage_k) / self.vocab_size * 100
        }
        
        return {"summary": summary, "details": details}
    
    def run_all_tests(self, train_prompts=None, test_prompts=None, num_train=50, num_test=20):
        """Run all test approaches and compare results."""
        # Generate prompts if not provided
        if train_prompts is None or test_prompts is None:
            train_prompts, test_prompts = self.generate_diverse_prompts(num_train, num_test)
        
        # Build code cache
        self.build_code_cache(train_prompts)
        
        # Run all tests
        self.test_results["single_neighbor"] = self.test_single_neighbor(test_prompts)
        
        # Test ensemble approach with different numbers of neighbors
        for n in [3, 5, 10]:
            self.test_results[f"ensemble_{n}"] = self.test_ensemble_approach(test_prompts, n_neighbors=n)
        
        # Test weighted voting with different settings
        for n, w in [(3, 1.0), (3, 2.0), (5, 1.0), (5, 2.0)]:
            self.test_results[f"weighted_{n}_{w}"] = self.test_weighted_voting(test_prompts, n_neighbors=n, weight_factor=w)
        
        # Test adaptive token selection
        self.test_results["adaptive"] = self.test_adaptive_token_selection(test_prompts)
        
        # Test vocabulary clustering
        self.test_results["clustering"] = self.test_vocabulary_clustering(test_prompts, n_clusters=50)
        
        # Test two-stage filtering
        self.test_results["two_stage"] = self.test_two_stage_filtering(test_prompts, first_stage_k=500, second_stage_k=50)
        
        return self.test_results
    
    def print_results_summary(self):
        """Print a summary of all test results."""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        print("\n===== 🔍 SPARSE LOGIT TEST RESULTS =====")
        
        # Print single neighbor results
        single_result = self.test_results["single_neighbor"]["summary"]
        print(f"\nBase Approach (Single Neighbor)")
        print(f"  Average Cosine Similarity: {single_result['avg_cosine_sim']:.4f}")
        print("  Recall Rates:")
        for k, rate in single_result["recall_rates"].items():
            print(f"    Top-{k}: {rate:.2f}%")
        
        # Compare all approaches by best recall
        print("\nComparison of All Approaches:")
        approach_results = {}
        
        for name, result in self.test_results.items():
            summary = result["summary"]
            method = summary["method"]
            
            if method == "single_neighbor":
                best_k = max(summary["recall_rates"].items(), key=lambda x: x[1])
                approach_results[name] = {
                    "best_recall": best_k[1],
                    "k_or_size": best_k[0],
                    "computation": best_k[0] / self.vocab_size * 100
                }
            
            elif method == "ensemble":
                best_k = max(summary["recall_rates"].items(), key=lambda x: x[1])
                k = best_k[0]
                approach_results[name] = {
                    "best_recall": best_k[1],
                    "k_or_size": k,
                    "computation": summary["avg_ensemble_sizes"][k] / self.vocab_size * 100
                }
            
            elif method == "weighted_voting":
                best_k = max(summary["recall_rates"].items(), key=lambda x: x[1])
                approach_results[name] = {
                    "best_recall": best_k[1],
                    "k_or_size": best_k[0],
                    "computation": best_k[0] / self.vocab_size * 100
                }
            
            elif method == "adaptive_selection":
                approach_results[name] = {
                    "best_recall": summary["recall_rate"],
                    "k_or_size": summary["avg_tokens_evaluated"],
                    "computation": summary["avg_tokens_evaluated"] / self.vocab_size * 100
                }
            
            elif method == "vocabulary_clustering":
                approach_results[name] = {
                    "best_recall": summary["recall_rate"],
                    "k_or_size": summary["avg_candidate_set_size"],
                    "computation": summary["avg_candidate_set_size"] / self.vocab_size * 100
                }
            
            elif method == "two_stage_filtering":
                approach_results[name] = {
                    "best_recall": summary["overall_recall"],
                    "k_or_size": f"{summary['first_stage_k']}+{summary['second_stage_k']}",
                    "computation": summary["computation_ratio"]
                }
        
        # Sort by recall (descending)
        sorted_approaches = sorted(approach_results.items(), key=lambda x: x[1]["best_recall"], reverse=True)
        
        for name, metrics in sorted_approaches:
            print(f"  {name}:")
            print(f"    Recall Rate: {metrics['best_recall']:.2f}%")
            print(f"    Token Set Size: {metrics['k_or_size']}")
            print(f"    Computation: {metrics['computation']:.2f}% of full vocab")
        
        # Find the best overall approach
        best_approach = max(approach_results.items(), key=lambda x: x[1]["best_recall"])
        
        print("\n🏆 Best Overall Approach:")
        print(f"  {best_approach[0]}")
        print(f"  Recall Rate: {best_approach[1]['best_recall']:.2f}%")
        print(f"  Computation: {best_approach[1]['computation']:.2f}% of full vocab")
        
        # Find the most efficient approach with >80% recall
        efficient_approaches = [(name, metrics) for name, metrics in approach_results.items() 
                              if metrics["best_recall"] >= 80.0]
        
        if efficient_approaches:
            most_efficient = min(efficient_approaches, key=lambda x: x[1]["computation"])
            print("\n💡 Most Efficient Approach with >80% Recall:")
            print(f"  {most_efficient[0]}")
            print(f"  Recall Rate: {most_efficient[1]['best_recall']:.2f}%")
            print(f"  Computation: {most_efficient[1]['computation']:.2f}% of full vocab")
            print(f"  Theoretical Speedup: {100 / most_efficient[1]['computation']:.2f}x")
    
    def plot_results(self, save_path=None):
        """Plot comparative results of all approaches."""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        methods = []
        recalls = []
        computations = []
        
        for name, result in self.test_results.items():
            summary = result["summary"]
            method = summary["method"]
            
            if method == "single_neighbor":
                for k, rate in summary["recall_rates"].items():
                    methods.append(f"single_{k}")
                    recalls.append(rate)
                    computations.append(k / self.vocab_size * 100)
            
            elif method == "ensemble":
                for k, rate in summary["recall_rates"].items():
                    methods.append(f"ensemble{summary['n_neighbors']}_{k}")
                    recalls.append(rate)
                    computations.append(summary["avg_ensemble_sizes"][k] / self.vocab_size * 100)
            
            elif method == "weighted_voting":
                for k, rate in summary["recall_rates"].items():
                    methods.append(f"weighted{summary['n_neighbors']}_{k}")
                    recalls.append(rate)
                    computations.append(k / self.vocab_size * 100)
            
            elif method == "adaptive_selection":
                methods.append("adaptive")
                recalls.append(summary["recall_rate"])
                computations.append(summary["avg_tokens_evaluated"] / self.vocab_size * 100)
            
            elif method == "vocabulary_clustering":
                methods.append("clustering")
                recalls.append(summary["recall_rate"])
                computations.append(summary["avg_candidate_set_size"] / self.vocab_size * 100)
            
            elif method == "two_stage_filtering":
                methods.append("two_stage")
                recalls.append(summary["overall_recall"])
                computations.append(summary["computation_ratio"])
        
        # Create scatter plot
        plt.scatter(computations, recalls, s=100, alpha=0.7)
        
        # Add labels
        for i, method in enumerate(methods):
            plt.annotate(method, (computations[i], recalls[i]), fontsize=8)
        
        # Add reference lines
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.3, label='80% Recall')
        plt.axhline(y=90, color='g', linestyle='--', alpha=0.3, label='90% Recall')
        
        # Add labels and title
        plt.xlabel('Computation (% of full vocabulary)')
        plt.ylabel('Recall Rate (%)')
        plt.title('Sparse Logit Computation: Recall vs. Computation')
        
        # Add grid and legend
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Adjust axis limits
        plt.xlim(0, max(computations) * 1.1)
        plt.ylim(min(recalls) * 0.9, 100)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, file_path="sparse_logit_results.json"):
        """Save test results to a JSON file."""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        # Convert all results to JSON-serializable format
        serializable_results = {}
        
        for name, result in self.test_results.items():
            # Convert tensors and other non-serializable objects to standard Python types
            serializable_summary = {}
            for k, v in result["summary"].items():
                if isinstance(v, dict):
                    serializable_summary[k] = {str(kk): float(vv) if hasattr(vv, "item") else vv 
                                              for kk, vv in v.items()}
                elif hasattr(v, "item"):
                    serializable_summary[k] = float(v)
                else:
                    serializable_summary[k] = v
            
            serializable_details = []
            for detail in result["details"]:
                serializable_detail = {}
                for k, v in detail.items():
                    if isinstance(v, dict):
                        serializable_detail[k] = {str(kk): float(vv) if hasattr(vv, "item") else vv 
                                                for kk, vv in v.items()}
                    elif hasattr(v, "item"):
                        serializable_detail[k] = float(v)
                    else:
                        serializable_detail[k] = v
                serializable_details.append(serializable_detail)
            
            serializable_results[name] = {
                "summary": serializable_summary,
                "details": serializable_details
            }
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {file_path}")

def main():
    """Run the complete test suite with default parameters."""
    test_suite = SparseLogitTestSuite()
    
    # Generate prompts
    train_prompts, test_prompts = test_suite.generate_diverse_prompts(num_train=50, num_test=20)
    
    # Run all tests
    results = test_suite.run_all_tests(train_prompts, test_prompts)
    
    # Print summary
    test_suite.print_results_summary()
    
    # Save results
    test_suite.save_results()
    
    # Plot results
    test_suite.plot_results(save_path="sparse_logit_results.png")
    
    return test_suite

if __name__ == "__main__":
    main()

""" Output:
Using device: cpu
Loading SVD from svd_lm_head_gpt2.pt...
Initialized with vocabulary size: 50257
Generated 50 training prompts and 20 test prompts
Building code cache from 50 prompts...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 56.38it/s]
✅ Built code cache with 50 entries
Testing single neighbor approach...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 57.67it/s]
Testing ensemble approach with 3 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 56.80it/s]
Testing ensemble approach with 5 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 54.84it/s]
Testing ensemble approach with 10 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 56.39it/s]
Testing weighted voting with 3 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 55.41it/s]
Testing weighted voting with 3 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 54.93it/s]
Testing weighted voting with 5 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 55.28it/s]
Testing weighted voting with 5 neighbors...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 55.68it/s]
Testing adaptive token selection...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 56.09it/s]
Testing vocabulary clustering with 50 clusters...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 56.93it/s]
Testing two-stage filtering...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 57.00it/s]

===== 🔍 SPARSE LOGIT TEST RESULTS =====

Base Approach (Single Neighbor)
  Average Cosine Similarity: 1.0000
  Recall Rates:
    Top-10: 100.00%
    Top-20: 100.00%
    Top-50: 100.00%
    Top-100: 100.00%
    Top-200: 100.00%
    Top-500: 100.00%
    Top-1000: 100.00%

Comparison of All Approaches:
  single_neighbor:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.02% of full vocab
  ensemble_3:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.03% of full vocab
  ensemble_5:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.03% of full vocab
  ensemble_10:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.06% of full vocab
  weighted_3_1.0:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.02% of full vocab
  weighted_3_2.0:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.02% of full vocab
  weighted_5_1.0:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.02% of full vocab
  weighted_5_2.0:
    Recall Rate: 100.00%
    Token Set Size: 10
    Computation: 0.02% of full vocab
  adaptive:
    Recall Rate: 100.00%
    Token Set Size: 10.0
    Computation: 0.02% of full vocab
  two_stage:
    Recall Rate: 100.00%
    Token Set Size: 500+50
    Computation: 1.09% of full vocab
  clustering:
    Recall Rate: 0.00%
    Token Set Size: 3098.75
    Computation: 6.17% of full vocab

🏆 Best Overall Approach:
  single_neighbor
  Recall Rate: 100.00%
  Computation: 0.02% of full vocab

💡 Most Efficient Approach with >80% Recall:
  single_neighbor
  Recall Rate: 100.00%
  Computation: 0.02% of full vocab
  Theoretical Speedup: 5025.70x
"""