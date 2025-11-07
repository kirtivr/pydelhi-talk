import os
import time
from typing import List, Dict, Any, Tuple, cast
from mem0 import MemoryClient  # type: ignore
from openai import OpenAI

# Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.environ.get(
    "DEEPSEEK_API_BASE", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")
USER_ID = "developer_alice"

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "DEEPSEEK_API_KEY not set. Export it: export DEEPSEEK_API_KEY='your-key'"
    )

if not MEM0_API_KEY:
    raise ValueError(
        "MEM0_API_KEY not set. Export it: export MEM0_API_KEY='your-key'"
    )

# Client (DeepSeek uses OpenAI-compatible API)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

# Pricing (USD per 1M tokens)
PRICE_INPUT_CACHE_HIT = 0.028
PRICE_INPUT_CACHE_MISS = 0.28
PRICE_OUTPUT = 0.42


def generate_developer_conversation() -> List[Dict[str, str]]:
    """
    Generate ~50 messages from a developer describing architectural preferences.
    """
    themes = [
        "I prefer event-driven microservices with idempotent handlers",
        "All services expose gRPC plus a REST facade for public traffic",
        "Use protobuf for contracts and buf for linting and breaking-change checks",
        "Zero-downtime deploys via blue/green with health probes",
        "Observability first: OpenTelemetry traces, structured logs, RED metrics",
        "Backpressure over retries; circuit breakers with exponential backoff",
        "State in Postgres, search in OpenSearch, cache in Redis",
        "Use S3 for immutable artifacts and backups with lifecycle policies",
        "Prefer CQRS for complex read paths and materialized views",
        "Outbox pattern for reliable event emission",
        "Service auth via mTLS; user auth via OIDC and JWT",
        "Feature flags for gradual rollouts and A/B testing",
        "Kill switches for risky features",
        "Prefer infra-as-code with Terraform and policy-as-code with OPA",
        "Use Skaffold locally; GitOps with ArgoCD for clusters",
        "Horizontal scaling first; vertical only when justified",
        "Hard multi-tenancy with per-tenant encryption keys",
        "PII tokenization and field-level encryption",
        "Schema migrations via Atlas or Flyway with shadow DB",
        "Async workflows modelled with temporal-like orchestrators",
        "Use LRU caches and request coalescing for hot keys",
        "SLOs: p99 latency and error budgets tracked weekly",
        "Retry budgets to avoid thundering herds",
        "Chaos experiments in staging weekly",
        "Prefer Go for services, Python for data pipelines",
        "Static code analysis in CI and pre-commit hooks",
        "Contract tests to prevent integration regressions",
        "Canary deploys with automated rollback",
        "Prefer immutable container images and SBOMs",
        "Rate limit per user and per token",
        "Shadow traffic to test new versions",
        "Use priority queues for critical events",
        "Batch writes where latency allows",
        "Pagination by cursor not offset",
        "Leverage read replicas for heavy analytics",
        "Secrets managed in Vault and rotated",
        "Use RFC3339 timestamps and UTC everywhere",
        "Partitioning strategy reviewed quarterly",
        "Bloom filters to pre-check likely misses",
        "Content-addressable storage for dedupe",
        "Use protobuf enums not magic numbers",
        "Dead-letter queues with reprocessing",
        "Schema versioning with adapters",
        "Runbooks for all critical alerts",
        "Prefer domain events over CRUD events",
        "Snapshotting for long-lived aggregates",
        "Deterministic IDs (ULIDs) for time-order",
        "Shard by tenant and hot key hashing",
        "Avoid distributed transactions; favor sagas",
        "Keep services small but independently valuable",
        "Document decisions with ADRs",
    ]
    messages: List[Dict[str, str]] = []
    # Simulate a user-only history (developer notes)
    for i, t in enumerate(themes[:50]):
        messages.append({"role": "user", "content": f"Pref {i+1}: {t}."})
    return messages


def concat_history_as_text(history: List[Dict[str, str]]) -> str:
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])


def token_estimate_from_text(text: str) -> int:
    # Rough estimate: 4 chars per token
    return max(1, len(text) // 4)


def run_chat(messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], float, str]:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=cast(Any, messages),  # type: ignore
        temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    usage = getattr(resp, "usage", None)
    input_tokens = 0
    output_tokens = 0
    if usage is not None:
        # OpenAI-compatible usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
            usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
            usage, "output_tokens", 0) or 0
    else:
        # Fallback estimate if provider doesn't return usage
        full_text = "\n".join([m.get("content", "") for m in messages])
        input_tokens = token_estimate_from_text(full_text)
        output_tokens = 300

    # Extract response text
    response_text = ""
    if hasattr(resp, "choices") and len(resp.choices) > 0:
        choice = resp.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            response_text = choice.message.content or ""

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}, elapsed, response_text


def estimate_cost(input_tokens: int, output_tokens: int, cache_hit_ratio: float = 0.0) -> float:
    cache_hit_tokens = int(input_tokens * max(0.0, min(1.0, cache_hit_ratio)))
    cache_miss_tokens = input_tokens - cache_hit_tokens
    input_cost = (cache_hit_tokens * PRICE_INPUT_CACHE_HIT +
                  cache_miss_tokens * PRICE_INPUT_CACHE_MISS) / 1_000_000
    output_cost = (output_tokens * PRICE_OUTPUT) / 1_000_000
    return input_cost + output_cost


def approach_1_full_context(history: List[Dict[str, str]], query: str) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("APPROACH 1: Naive - Full Conversation Context")
    print("="*70)
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {"role": "user",
                "content": f"Conversation History:\n{concat_history_as_text(history)}\n\nUser Query: {query}"}

    print("\nUser Message (Full Context):")
    print("-" * 70)
    print(user_msg["content"])
    print("-" * 70)

    usage, elapsed, response_text = run_chat([system_msg, user_msg])

    print("\nResponse (Full Context):")
    print("-" * 70)
    print(response_text)
    print("-" * 70)

    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    metrics = {
        "num_requests": 1,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens_processed": total_tokens,
        "execution_time": elapsed,
        "avg_token_throughput": (total_tokens / elapsed) if elapsed > 0 else 0.0,
        "estimated_cost": estimate_cost(usage["input_tokens"], usage["output_tokens"], cache_hit_ratio=0.0),
        "response": response_text,
    }
    print(
        f"  Input tokens: {usage['input_tokens']}, Output tokens: {usage['output_tokens']}")
    print(
        f"  Execution time: {elapsed:.3f}s, Avg throughput: {metrics['avg_token_throughput']:.2f} tok/s")
    print(f"  Estimated cost: ${metrics['estimated_cost']:.6f}")
    return metrics


def approach_2_with_mem0(memory_client: MemoryClient, user_id: str, query: str) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("APPROACH 2: With Mem0 - Retrieve Only Relevant Context")
    print("="*70)
    # Search for relevant memories using Mem0
    # Filters are required - user_id must be in filters dict
    search_results = memory_client.search(query, filters={"user_id": user_id})
    # Extract memory content from search results
    if search_results and isinstance(search_results, dict) and "results" in search_results:
        memories = search_results["results"]
    elif isinstance(search_results, list):
        memories = search_results
    else:
        memories = []

    relevant_text = "\n".join([
        str(m.get("data", {}).get("memory", m)
            ) if isinstance(m, dict) else str(m)
        for m in memories
    ]) if memories else ""

    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {"role": "user",
                "content": f"User Context (from mem0):\n{relevant_text}\n\nUser Query: {query}"}

    print("\nUser Message (Mem0 - Relevant Context Only):")
    print("-" * 70)
    print(user_msg["content"])
    print("-" * 70)

    usage, elapsed, response_text = run_chat([system_msg, user_msg])

    print("\nResponse (Mem0 - Relevant Context Only):")
    print("-" * 70)
    print(response_text)
    print("-" * 70)

    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    metrics = {
        "num_requests": 1,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens_processed": total_tokens,
        "execution_time": elapsed,
        "avg_token_throughput": (total_tokens / elapsed) if elapsed > 0 else 0.0,
        "estimated_cost": estimate_cost(usage["input_tokens"], usage["output_tokens"], cache_hit_ratio=0.0),
        "response": response_text,
    }
    print(
        f"  Input tokens: {usage['input_tokens']}, Output tokens: {usage['output_tokens']}")
    print(
        f"  Execution time: {elapsed:.3f}s, Avg throughput: {metrics['avg_token_throughput']:.2f} tok/s")
    print(f"  Estimated cost: ${metrics['estimated_cost']:.6f}")
    return metrics


def print_comparison(metrics1: Dict[str, Any], metrics2: Dict[str, Any]):
    # Display metrics comparison
    print("\n" + "="*70)
    print("METRICS COMPARISON")
    print("="*70)
    print(f"{'Metric':<35} {'Approach 1 (Full Context)':<30} {'Approach 2 (Mem0)':<30}")
    print("-"*70)
    print(
        f"{'Number of Requests':<35} {metrics1['num_requests']:<30} {metrics2['num_requests']:<30}")
    print(
        f"{'Total Tokens Processed':<35} {metrics1['total_tokens_processed']:<30} {metrics2['total_tokens_processed']:<30}")
    print(
        f"{'Execution Time':<35} {metrics1['execution_time']:.3f}s{'':<25} {metrics2['execution_time']:.3f}s{'':<25}")
    print(
        f"{'Avg Token Throughput':<35} {metrics1['avg_token_throughput']:.2f} tok/s{'':<15} {metrics2['avg_token_throughput']:.2f} tok/s{'':<15}")
    print(
        f"{'Estimated Cost (USD)':<35} ${metrics1['estimated_cost']:.6f}{'':<20} ${metrics2['estimated_cost']:.6f}{'':<20}")
    print("="*70)

    if metrics2['execution_time'] > 0 and metrics1['execution_time'] > 0:
        speedup_factor = metrics1['execution_time'] / \
            metrics2['execution_time']
        time_improvement = (speedup_factor - 1) * 100
        print(
            f"\nExecution Time Improvement: {time_improvement:.1f}% faster with Mem0")
        print(
            f"   Full: {metrics1['execution_time']:.2f}s vs Mem0: {metrics2['execution_time']:.2f}s")
        print(f"   Speedup: {speedup_factor:.2f}x faster")
        print(
            f"   Time saved: {metrics1['execution_time'] - metrics2['execution_time']:.2f}s")

    if metrics1['estimated_cost'] > 0 and metrics2['estimated_cost'] >= 0:
        cost_reduction_pct = (
            (metrics1['estimated_cost'] - metrics2['estimated_cost']) / metrics1['estimated_cost']) * 100
        print(f"\nCost Reduction: {cost_reduction_pct:.1f}% lower with Mem0")
        print(
            f"   Full: ${metrics1['estimated_cost']:.6f} vs Mem0: ${metrics2['estimated_cost']:.6f}")


if __name__ == "__main__":
    # Build and store conversation history
    conversation_history = generate_developer_conversation()

    # Initialize Mem0 client (hosted API)
    mem0_client = MemoryClient(api_key=MEM0_API_KEY)

    # Add conversation history to Mem0
    mem0_client.add(conversation_history, user_id=USER_ID, version="v2")

    query = "How should I structure a resilient async workflow with retries and idempotency?"

    metrics_full = approach_1_full_context(conversation_history, query)
    metrics_mem0 = approach_2_with_mem0(mem0_client, USER_ID, query)
    print_comparison(metrics_full, metrics_mem0)
