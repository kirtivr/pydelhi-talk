import anthropic
import os
import time
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY environment variable not set. "
        "Please export it: export ANTHROPIC_API_KEY='your-key-here'"
    )
client = anthropic.Anthropic(api_key=api_key)

with open("large_shakespearean_text_dump", "r", encoding="utf-8") as f:
    large_context = f.read()


def generate_needle_prompt(needle: str, index: int) -> str:
    """Generate a 5000-character prompt asking to find a specific needle in the haystack."""
    base_prompt = f"""Find the following exact text in the provided Shakespearean play and provide:
1. The exact location (Act, Scene, and approximate line context)
2. The surrounding dialogue (at least 3 lines before and after)
3. Which character speaks this line

The text to find is: "{needle}"

Please search through the entire document carefully and provide a detailed answer with context."""

    padding_needed = 5000 - len(base_prompt)
    if padding_needed > 0:
        padding = " " * padding_needed
        base_prompt += padding

    return base_prompt[:5000]


needles = [
    "Who's there?",
    "Long live the king!",
    "Not a mouse stirring.",
    "What, has this thing appear'd again to-night?",
    "Most like: it harrows me with fear and wonder.",
    "O, that this too too solid flesh would melt",
    "Frailty, thy name is woman!",
    "I know not 'seems.'",
    "My father's spirit in arms! all is not well",
    "A little more than kin, and less than kind."
]

user_prompts = [generate_needle_prompt(
    needle, i) for i, needle in enumerate(needles)]

system_message = f"""You are a helpful AI assistant. Analyze the following Shakespearean text carefully.

{large_context}"""


def approach_1_parallel() -> Dict[str, Any]:
    """
    Approach 1: Send all 10 prompts in parallel using ThreadPoolExecutor.
    """
    print("\n" + "="*70)
    print("APPROACH 1: Parallel Requests")
    print("="*70)

    metrics: Dict[str, Any] = {
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "avg_token_throughput": 0.0,
        "num_requests": len(user_prompts)
    }

    start_time = time.perf_counter()

    def send_request(prompt: str, index: int) -> Dict[str, Any]:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return {
            "index": index,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_prompt = {
            executor.submit(send_request, prompt, i): (prompt, i)
            for i, prompt in enumerate(user_prompts)
        }

        completed = 0
        for future in as_completed(future_to_prompt):
            completed += 1
            result = future.result()
            metrics["total_tokens_processed"] += result["total_tokens"]
            print(f"  Request {result['index']+1}/10 completed: "
                  f"{result['input_tokens']} input + {result['output_tokens']} output tokens")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]
    else:
        metrics["avg_token_throughput"] = 0.0

    return metrics


def approach_2_sequential() -> Dict[str, Any]:
    """
    Approach 2: Send 10 prompts sequentially, one after another.
    """
    print("\n" + "="*70)
    print("APPROACH 2: Sequential Requests")
    print("="*70)

    metrics: Dict[str, Any] = {
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "avg_token_throughput": 0.0,
        "num_requests": len(user_prompts)
    }

    start_time = time.perf_counter()

    for i, prompt in enumerate(user_prompts):
        print(f"\nSending request {i+1}/10...")
        request_start = time.perf_counter()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        request_end = time.perf_counter()
        request_time = request_end - request_start

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        metrics["total_tokens_processed"] += input_tokens + output_tokens

        print(f"  Completed in {request_time:.2f}s: "
              f"{input_tokens} input + {output_tokens} output tokens")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]
    else:
        metrics["avg_token_throughput"] = 0.0

    return metrics


def print_comparison(metrics1: Dict[str, Any], metrics2: Dict[str, Any]):
    """
    Print a formatted comparison table of metrics from both approaches.
    """
    print("\n" + "="*70)
    print("METRICS COMPARISON")
    print("="*70)
    print(f"{'Metric':<35} {'Approach 1 (Parallel)':<30} {'Approach 2 (Sequential)':<30}")
    print("-"*70)
    print(
        f"{'Number of Requests':<35} {metrics1['num_requests']:<30} {metrics2['num_requests']:<30}")
    print(
        f"{'Total Tokens Processed':<35} {metrics1['total_tokens_processed']:<30} {metrics2['total_tokens_processed']:<30}")
    print(
        f"{'Execution Time':<35} {metrics1['execution_time']:.3f}s{'':<25} {metrics2['execution_time']:.3f}s{'':<25}")
    print(
        f"{'Avg Token Throughput':<35} {metrics1['avg_token_throughput']:.2f} tokens/s{'':<15} {metrics2['avg_token_throughput']:.2f} tokens/s{'':<15}")
    print("="*70)

    if metrics2['execution_time'] > 0 and metrics1['execution_time'] > 0:
        speedup_factor = metrics2['execution_time'] / \
            metrics1['execution_time']
        time_improvement = (speedup_factor - 1) * 100
        print(
            f"\nExecution Time Improvement: {time_improvement:.1f}% faster with parallel requests")
        print(
            f"   Parallel: {metrics1['execution_time']:.2f}s vs Sequential: {metrics2['execution_time']:.2f}s")
        print(
            f"   Speedup: {speedup_factor:.2f}x faster")
        print(
            f"   Time saved: {metrics2['execution_time'] - metrics1['execution_time']:.2f}s")

    if metrics1['avg_token_throughput'] > 0 and metrics2['avg_token_throughput'] > 0:
        throughput_improvement = (
            (metrics1['avg_token_throughput'] - metrics2['avg_token_throughput']) / metrics2['avg_token_throughput']) * 100
        print(
            f"\nThroughput Improvement: {throughput_improvement:.1f}% higher with parallel requests")
        print(
            f"   Parallel: {metrics1['avg_token_throughput']:.2f} tokens/s vs Sequential: {metrics2['avg_token_throughput']:.2f} tokens/s")

    if metrics1['total_tokens_processed'] == metrics2['total_tokens_processed']:
        print(
            f"\nBoth approaches processed the same number of tokens: {metrics1['total_tokens_processed']}")
        print(f"   The performance difference is purely due to parallelization efficiency.")


if __name__ == "__main__":
    print("Parallel vs Sequential Requests Comparison")
    print("="*70)
    print("Finding needles in a haystack: 10 prompts, each ~5000 characters")
    print("Each prompt asks to find a specific quote in the Shakespearean text")

    metrics1 = approach_1_parallel()
    metrics2 = approach_2_sequential()
    print_comparison(metrics1, metrics2)
