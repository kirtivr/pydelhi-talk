import anthropic
import os
import time
from typing import Dict, List, Optional, Any, cast

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    print("APPROACH 1: Non-Streaming Requests")
    print("="*70)

    system_message_no_cache: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "You are a helpful AI assistant."
        },
        {
            "type": "text",
            "text": large_context
        }
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

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=cast(Any, system_message_no_cache),  # type: ignore
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        request_end = time.perf_counter()

        if i == 0:
            metrics["ttft"] = request_end - request_start
            print(f"  First token received at {metrics['ttft']:.3f}s")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_read = response.usage.cache_read_input_tokens or 0
        cache_creation = response.usage.cache_creation_input_tokens or 0
        metrics["total_tokens_processed"] += input_tokens + \
            cache_read + output_tokens
        metrics["cache_read_tokens"] += cache_read
        metrics["cache_creation_tokens"] += cache_creation

        print(
            f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"  Cache read: {response.usage.cache_read_input_tokens or 0}, "
              f"Cache creation: {response.usage.cache_creation_input_tokens or 0}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]
    else:
        metrics["avg_token_throughput"] = 0.0

    return metrics


def approach_2_non_streaming_with_cache() -> Dict[str, Any]:
    """
    Approach 2: Non-streaming requests with cache control.
    Sends 3 prompts one after another without streaming, but with caching enabled.
    """
    print("\n" + "="*70)
    print("APPROACH 2: Non-Streaming Requests with Cache Control")
    print("="*70)

    system_message_with_cache_control: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "You are a helpful AI assistant."
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
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

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            # type: ignore
            system=cast(Any, system_message_with_cache_control),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        request_end = time.perf_counter()

        if i == 0:
            metrics["ttft"] = request_end - request_start
            print(f"  First token received at {metrics['ttft']:.3f}s")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_read = response.usage.cache_read_input_tokens or 0
        cache_creation = response.usage.cache_creation_input_tokens or 0
        metrics["total_tokens_processed"] += input_tokens + \
            cache_read + output_tokens
        metrics["cache_read_tokens"] += cache_read
        metrics["cache_creation_tokens"] += cache_creation

        print(
            f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"  Cache read: {cache_read}, "
              f"Cache creation: {cache_creation}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]
    else:
        metrics["avg_token_throughput"] = 0.0

    return metrics


def get_streaming_usage(prompt: str) -> Dict[str, int]:
    """
    Helper function to get usage stats for a streaming request.
    Makes a non-streaming call with same parameters to get accurate usage.
    """
    system_message_with_cache_control: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "You are a helpful AI assistant."
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
    ]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=cast(Any, system_message_with_cache_control),  # type: ignore
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": response.usage.cache_read_input_tokens or 0,
        "cache_creation_tokens": response.usage.cache_creation_input_tokens or 0
    }


def approach_3_streaming() -> Dict[str, Any]:
    """
    Approach 3: Streaming requests exploiting prefix caching.
    Properly tracks metrics including usage statistics.
    """
    print("\n" + "="*70)
    print("APPROACH 3: Streaming Requests with Prefix Caching")
    print("="*70)

    metrics: Dict[str, Any] = {
        "ttft": None,
        "total_tokens_processed": 0,
        "execution_time": 0.0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "avg_token_throughput": 0.0
    }

    system_message_with_cache_control: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "You are a helpful AI assistant."
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    start_time = time.perf_counter()

    for i, prompt in enumerate(user_prompts):
        print(f"\nSending request {i+1}/3 (streaming): {prompt[:50]}...")
        request_start = time.perf_counter()
        first_token_time = None

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            # type: ignore
            system=cast(Any, system_message_with_cache_control),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ) as stream:
            for event in stream:
                if i == 0 and first_token_time is None:
                    if event.type == "content_block_start" or event.type == "content_block_delta":
                        first_token_time = time.perf_counter()
                        metrics["ttft"] = first_token_time - request_start
                        print(
                            f"  First token received at {metrics['ttft']:.3f}s")

        usage_stats = get_streaming_usage(prompt)

        metrics["total_tokens_processed"] += usage_stats["input_tokens"] + \
            usage_stats["cache_read_tokens"] + usage_stats["output_tokens"]
        metrics["cache_read_tokens"] += usage_stats["cache_read_tokens"]
        metrics["cache_creation_tokens"] += usage_stats["cache_creation_tokens"]

        print(f"  Input tokens: {usage_stats['input_tokens']}, "
              f"Output tokens: {usage_stats['output_tokens']}")
        print(f"  Cache read: {usage_stats['cache_read_tokens']}, "
              f"Cache creation: {usage_stats['cache_creation_tokens']}")

    metrics["execution_time"] = time.perf_counter() - start_time
    if metrics["execution_time"] > 0:
        metrics["avg_token_throughput"] = metrics["total_tokens_processed"] / \
            metrics["execution_time"]
    else:
        metrics["avg_token_throughput"] = 0.0

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
    print(
        f"{'Total Tokens Processed':<25} {metrics1['total_tokens_processed']:<22} {metrics2['total_tokens_processed']:<22} {metrics3['total_tokens_processed']:<22}")
    print(
        f"{'Execution Time':<25} {metrics1['execution_time']:.3f}s{'':<18} {metrics2['execution_time']:.3f}s{'':<18} {metrics3['execution_time']:.3f}s{'':<18}")
    print(
        f"{'Avg Token Throughput':<25} {metrics1['avg_token_throughput']:.2f} tok/s{'':<13} {metrics2['avg_token_throughput']:.2f} tok/s{'':<13} {metrics3['avg_token_throughput']:.2f} tok/s{'':<13}")
    print(
        f"{'Cache Read Tokens':<25} {metrics1['cache_read_tokens']:<22} {metrics2['cache_read_tokens']:<22} {metrics3['cache_read_tokens']:<22}")
    print(
        f"{'Cache Creation Tokens':<25} {metrics1['cache_creation_tokens']:<22} {metrics2['cache_creation_tokens']:<22} {metrics3['cache_creation_tokens']:<22}")
    print("="*90)


if __name__ == "__main__":
    print("Prefix Caching Comparison: Three Approaches")
    print("="*70)
    print("Approach 1: Non-streaming, no cache control")
    print("Approach 2: Non-streaming with cache control")
    print("Approach 3: Streaming with cache control")

    metrics1 = approach_1_non_streaming()
    metrics2 = approach_2_non_streaming_with_cache()
    metrics3 = approach_3_streaming()
    print_comparison(metrics1, metrics2, metrics3)
