from __future__ import annotations

import concurrent.futures
import statistics
import time
import tracemalloc
from typing import Dict, List


def run_mock_load_test(
    prompts: int = 100,
    concurrency: int = 4,
    provider_latency_ms: int = 50,
    jury_size: int = 4,
    rounds: int = 1,
) -> Dict[str, object]:
    total_calls = prompts * max(1, jury_size) * max(1, rounds)
    submitted_at = time.monotonic()

    def fake_provider(_: int) -> tuple[float, float]:
        started = time.monotonic()
        queue_ms = (started - submitted_at) * 1000
        time.sleep(provider_latency_ms / 1000.0)
        return (time.monotonic() - started) * 1000, queue_ms

    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        measurements = list(executor.map(fake_provider, range(total_calls)))
    wall = time.monotonic() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latencies: List[float] = [item[0] for item in measurements]
    queue_latencies: List[float] = [item[1] for item in measurements]
    ordered = sorted(latencies)

    def percentile(value: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * value)))
        return ordered[index]

    return {
        "mode": "deterministic_mock_provider",
        "prompts": prompts,
        "jury_size": jury_size,
        "rounds": rounds,
        "simulated_provider_calls": total_calls,
        "concurrency": concurrency,
        "provider_latency_ms": provider_latency_ms,
        "throughput_prompts_per_second": prompts / wall if wall else 0.0,
        "throughput_calls_per_second": total_calls / wall if wall else 0.0,
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
        "p99_latency_ms": percentile(0.99),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "mean_queue_ms": statistics.mean(queue_latencies) if queue_latencies else 0.0,
        "p95_queue_ms": sorted(queue_latencies)[min(len(queue_latencies) - 1, int(round((len(queue_latencies) - 1) * 0.95)))] if queue_latencies else 0.0,
        "process_cpu_seconds": cpu_seconds,
        "tracemalloc_peak_bytes": peak_memory_bytes,
        "error_rate": 0.0,
        "warning": "Mock orchestration throughput is not live-provider scalability.",
    }
