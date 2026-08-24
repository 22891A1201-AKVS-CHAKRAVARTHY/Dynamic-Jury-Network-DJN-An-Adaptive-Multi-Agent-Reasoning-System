from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

PRICE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "model_pricing.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass
class UsageRecord:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    unavailable_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _first_int(source: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = source.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def extract_usage(message: Any) -> UsageRecord:
    candidates = []
    usage_metadata = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(usage_metadata, dict):
        candidates.append(usage_metadata)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            nested = response_metadata.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)

    for usage in candidates:
        prompt = _first_int(usage, "input_tokens", "prompt_tokens", "prompt_token_count")
        completion = _first_int(
            usage,
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
        )
        total = _first_int(usage, "total_tokens", "total_token_count")
        cached = _first_int(
            usage,
            "cached_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
        )
        if prompt is not None or completion is not None or total is not None:
            if total is None and prompt is not None and completion is not None:
                total = prompt + completion
            return UsageRecord(prompt, completion, total, cached, "")

    return UsageRecord(unavailable_reason="PROVIDER_USAGE_NOT_EXPOSED")


def load_price_config(path: Path = PRICE_CONFIG_PATH) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return {"version": "unavailable", "currency": "USD", "models": {}}


def estimate_cost(model_id: str, usage: UsageRecord) -> Dict[str, Any]:
    config = load_price_config()
    price = (config.get("models") or {}).get(model_id)
    result = {
        "amount": None,
        "currency": config.get("currency", "USD"),
        "price_version": config.get("version", "unavailable"),
        "unavailable_reason": "",
    }
    if not price:
        result["unavailable_reason"] = "MODEL_PRICE_NOT_CONFIGURED"
        return result
    if usage.prompt_tokens is None or usage.completion_tokens is None:
        result["unavailable_reason"] = usage.unavailable_reason or "TOKEN_USAGE_UNAVAILABLE"
        return result

    input_rate = float(price.get("input_per_million", 0.0))
    output_rate = float(price.get("output_per_million", 0.0))
    result["amount"] = (
        usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate
    ) / 1_000_000.0
    return result


_LOCK = threading.Lock()
_SEMAPHORES: Dict[str, threading.BoundedSemaphore] = {}
_FAILURES: Dict[str, int] = {}
_CIRCUIT_OPENED_AT: Dict[str, float] = {}
_LAST_CALL_AT: Dict[str, float] = {}


def _provider_semaphore(provider: str) -> threading.BoundedSemaphore:
    limit = max(1, int(os.getenv(f"DJN_{provider.upper()}_MAX_CONCURRENCY", os.getenv("DJN_MAX_CONCURRENCY", "4"))))
    key = f"{provider}:{limit}"
    with _LOCK:
        if key not in _SEMAPHORES:
            _SEMAPHORES[key] = threading.BoundedSemaphore(limit)
        return _SEMAPHORES[key]


def _circuit_is_open(provider: str) -> bool:
    threshold = max(1, int(os.getenv("DJN_CIRCUIT_BREAKER_FAILURES", "5")))
    cooldown = max(1.0, float(os.getenv("DJN_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30")))
    with _LOCK:
        if _FAILURES.get(provider, 0) < threshold:
            return False
        opened = _CIRCUIT_OPENED_AT.get(provider, 0.0)
        if time.monotonic() - opened >= cooldown:
            _FAILURES[provider] = 0
            return False
        return True


def _respect_provider_rate_limit(provider: str) -> None:
    interval_ms = max(
        0.0,
        float(os.getenv(
            f"DJN_{provider.upper()}_MIN_INTERVAL_MS",
            os.getenv("DJN_PROVIDER_MIN_INTERVAL_MS", "0"),
        )),
    )
    if interval_ms <= 0:
        return
    with _LOCK:
        now = time.monotonic()
        delay = interval_ms / 1000.0 - (now - _LAST_CALL_AT.get(provider, 0.0))
        _LAST_CALL_AT[provider] = now + max(0.0, delay)
    if delay > 0:
        time.sleep(delay)


def invoke_with_telemetry(
    invoke: Callable[[Dict[str, Any]], Any],
    payload: Dict[str, Any],
    *,
    call_type: str,
    provider: str,
    model_id: str,
    max_retries: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if _circuit_is_open(provider):
        raise RuntimeError(f"Provider circuit is open: {provider}")

    retries = max_retries if max_retries is not None else int(os.getenv("DJN_MAX_RETRIES", "2"))
    semaphore = _provider_semaphore(provider)
    queued_at = time.monotonic()
    semaphore.acquire()
    _respect_provider_rate_limit(provider)
    queue_ms = int((time.monotonic() - queued_at) * 1000)
    try:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            started = time.monotonic()
            try:
                message = invoke(payload)
                latency_ms = int((time.monotonic() - started) * 1000)
                usage = extract_usage(message)
                cost = estimate_cost(model_id, usage)
                with _LOCK:
                    _FAILURES[provider] = 0
                return message, {
                    "call_type": call_type,
                    "provider": provider,
                    "model_id": model_id,
                    "latency_ms": latency_ms,
                    "queue_ms": queue_ms,
                    "retry_count": attempt,
                    "usage": usage.as_dict(),
                    "cost": cost,
                    "status": "OK",
                    "error": "",
                }
            except Exception as exc:  # provider exceptions vary by SDK
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 4) + random.random() * 0.05)
        with _LOCK:
            _FAILURES[provider] = _FAILURES.get(provider, 0) + 1
            _CIRCUIT_OPENED_AT[provider] = time.monotonic()
        raise last_error or RuntimeError("Model invocation failed")
    finally:
        semaphore.release()
