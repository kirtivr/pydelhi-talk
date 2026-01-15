import openai
import os
import time
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable not set. "
        "Please export it: export OPENROUTER_API_KEY='your-key-here'"
    )

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

with open("large_shakespearean_text_dump", "r", encoding="utf-8") as f:
    large_context = f.read()

user_prompts = [
    "Summarize the main events and characters introduced in Act I, Scene I.",
    "What is the relationship between Hamlet and King Claudius, and how does Hamlet feel about his mother's remarriage?",
    "Describe the appearance and behavior of the ghost that appears to the guards, and explain what Horatio thinks it might signify."
]


def approach_1_non_streaming() -> Dict[str, Any]:
    """
    Approach 1: Non-streaming requests without cache control.
    Sends 3 prompts one after another without streaming.
    """
    print("\n" + "="*70)
    print("APPROACH 1: Non-Streaming Requests (No Cache Control)")
    print("="*70)

    # System message without cache_control - no caching
    system_message_no_cache: List[Dict[str, Any]] = [
        {"type": "text", "text": "You are a helpful AI assistant."},
        {"type": "text", "text": large_context}
    ]

    metrics: Dict[str, Any] = {
        "ttft": None,
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "avg_token_throughput": 0.0
    }

    start_time = time.perf_counter()

    for i, prompt in enumerate(user_prompts):
        print(f"\nSending request {i+1}/3: {prompt[:50]}...")
        request_start = time.perf_counter()

        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": system_message_no_cache
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        request_end = time.perf_counter()

        if i == 0:
            metrics["ttft"] = request_end - request_start
            print(f"  First response at {metrics['ttft']:.3f}s")

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # OpenRouter returns cache info in prompt_tokens_details
        cache_read = 0
        cache_creation = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cache_read = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        metrics["total_tokens_processed"] += input_tokens + output_tokens
        metrics["cache_read_tokens"] += cache_read
        metrics["cache_creation_tokens"] += cache_creation

        print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"  Cache read: {cache_read}, Cache creation: {cache_creation}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]

    return metrics


def approach_2_non_streaming_with_cache() -> Dict[str, Any]:
    """
    Approach 2: Non-streaming requests with cache control.
    Sends 3 prompts one after another without streaming, but with caching enabled.
    Uses OpenRouter's cache_control format for Anthropic models.
    """
    print("\n" + "="*70)
    print("APPROACH 2: Non-Streaming Requests with Cache Control")
    print("="*70)

    # System message WITH cache_control - enables prompt caching
    system_message_with_cache: List[Dict[str, Any]] = [
        {"type": "text", "text": "You are a helpful AI assistant."},
        {"type": "text", "text": large_context, "cache_control": {"type": "ephemeral"}}
    ]

    metrics: Dict[str, Any] = {
        "ttft": None,
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "avg_token_throughput": 0.0
    }

    start_time = time.perf_counter()

    for i, prompt in enumerate(user_prompts):
        print(f"\nSending request {i+1}/3: {prompt[:50]}...")
        request_start = time.perf_counter()

        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": system_message_with_cache
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        request_end = time.perf_counter()

        if i == 0:
            metrics["ttft"] = request_end - request_start
            print(f"  First response at {metrics['ttft']:.3f}s")

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # OpenRouter returns cache info in prompt_tokens_details
        cache_read = 0
        cache_creation = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cache_read = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        metrics["total_tokens_processed"] += input_tokens + output_tokens
        metrics["cache_read_tokens"] += cache_read
        metrics["cache_creation_tokens"] += cache_creation

        print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"  Cache read: {cache_read}, Cache creation: {cache_creation}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]

    return metrics


def approach_3_streaming_with_cache() -> Dict[str, Any]:
    """
    Approach 3: Streaming requests with cache control.
    Measures actual TTFT (time to first token) with streaming.
    """
    print("\n" + "="*70)
    print("APPROACH 3: Streaming Requests with Cache Control")
    print("="*70)

    # System message WITH cache_control - enables prompt caching
    system_message_with_cache: List[Dict[str, Any]] = [
        {"type": "text", "text": "You are a helpful AI assistant."},
        {"type": "text", "text": large_context, "cache_control": {"type": "ephemeral"}}
    ]

    metrics: Dict[str, Any] = {
        "ttft": None,
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "avg_token_throughput": 0.0
    }

    start_time = time.perf_counter()

    for i, prompt in enumerate(user_prompts):
        print(f"\nSending request {i+1}/3 (streaming): {prompt[:50]}...")
        request_start = time.perf_counter()
        first_token_time = None
        output_tokens = 0

        stream = client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            max_tokens=1024,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": system_message_with_cache
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        for chunk in stream:
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = time.perf_counter()
                if i == 0:
                    metrics["ttft"] = first_token_time - request_start
                    print(f"  First token at {metrics['ttft']:.3f}s")
            if chunk.choices and chunk.choices[0].delta.content:
                output_tokens += 1  # Approximate token count from chunks

        # Make a follow-up call to get accurate usage stats
        usage_response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": system_message_with_cache
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        usage = usage_response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        actual_output_tokens = usage.completion_tokens if usage else 0

        cache_read = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cache_read = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        metrics["total_tokens_processed"] += input_tokens + actual_output_tokens
        metrics["cache_read_tokens"] += cache_read

        print(f"  Input tokens: {input_tokens}, Output tokens: {actual_output_tokens}")
        print(f"  Cache read: {cache_read}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]

    return metrics


def print_comparison(metrics1: Dict[str, Any], metrics2: Dict[str, Any], metrics3: Dict[str, Any]):
    """
    Print a formatted comparison table of metrics from all three approaches.
    """
    print("\n" + "="*90)
    print("METRICS COMPARISON")
    print("="*90)
    print(f"{'Metric':<25} {'Approach 1 (No Cache)':<22} {'Approach 2 (Cache)':<22} {'Approach 3 (Streaming)':<22}")
    print("-"*90)

    ttft1_str = f"{metrics1['ttft']:.3f}s" if metrics1['ttft'] is not None else "N/A"
    ttft2_str = f"{metrics2['ttft']:.3f}s" if metrics2['ttft'] is not None else "N/A"
    ttft3_str = f"{metrics3['ttft']:.3f}s" if metrics3['ttft'] is not None else "N/A"

    print(f"{'TTFT (first request)':<25} {ttft1_str:<22} {ttft2_str:<22} {ttft3_str:<22}")
    print(f"{'Total Tokens Processed':<25} {metrics1['total_tokens_processed']:<22} {metrics2['total_tokens_processed']:<22} {metrics3['total_tokens_processed']:<22}")
    print(f"{'Execution Time':<25} {metrics1['execution_time']:.3f}s{'':<18} {metrics2['execution_time']:.3f}s{'':<18} {metrics3['execution_time']:.3f}s{'':<18}")
    print(f"{'Avg Token Throughput':<25} {metrics1['avg_token_throughput']:.2f} tok/s{'':<13} {metrics2['avg_token_throughput']:.2f} tok/s{'':<13} {metrics3['avg_token_throughput']:.2f} tok/s{'':<13}")
    print(f"{'Cache Read Tokens':<25} {metrics1['cache_read_tokens']:<22} {metrics2['cache_read_tokens']:<22} {metrics3['cache_read_tokens']:<22}")
    print("="*90)

    # Show performance improvement from caching
    if metrics1['ttft'] and metrics2['ttft']:
        improvement = ((metrics1['ttft'] - metrics2['ttft']) / metrics1['ttft']) * 100
        if improvement > 0:
            print(f"\nCache benefit: Approach 2 TTFT is {improvement:.1f}% faster than Approach 1")

    if metrics2['cache_read_tokens'] > 0:
        print(f"Cache hit confirmed: {metrics2['cache_read_tokens']} tokens read from cache in Approach 2")


if __name__ == "__main__":
    print("Prefix Caching Comparison: Three Approaches (OpenRouter)")
    print("="*70)
    print("Approach 1: Non-streaming, NO cache control (baseline)")
    print("Approach 2: Non-streaming WITH cache control (should show cache hits)")
    print("Approach 3: Streaming WITH cache control (measures true TTFT)")

    metrics1 = approach_1_non_streaming()
    metrics2 = approach_2_non_streaming_with_cache()
    metrics3 = approach_3_streaming_with_cache()
    print_comparison(metrics1, metrics2, metrics3)
