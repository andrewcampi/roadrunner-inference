# run_roadrunner_full_gpt2.py

from pathlib import Path
from time import time
from svd_routing import RoadrunnerEngine  # Make sure this file is in the same directory

def run_roadrunner_inference():
    # === Step 1: Load Engine ===
    roadrunner = RoadrunnerEngine(model_name="gpt2")

    # === Step 2: Configure Routing ===
    roadrunner.enable_mlp_routing(True)
    roadrunner.enable_attention_routing(True)

    # Based on findings, these are safe alpha values that preserve output fidelity
    roadrunner.alpha = 0.001  # You can tune this further per-layer

    # === Step 3: Choose Prompt ===
    prompt = "The future of machine intelligence lies in"
    max_tokens = 50

    print("\n🚀 Starting routed generation...")
    start_time = time()
    output_text = roadrunner.generate(prompt, max_length=max_tokens)
    elapsed = time() - start_time

    print("\n🧠 Final Output:")
    print(output_text)
    print(f"\n⏱️ Total time: {elapsed:.2f}s ({max_tokens / elapsed:.2f} tokens/sec)")

    # Optional: Print stats
    roadrunner.print_performance_stats()


if __name__ == "__main__":
    run_roadrunner_inference()
