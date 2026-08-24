from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from djn_db.models import LLMPool, ModelRollingStat

JURY_SIZE = 4
SUPPORTED_CATEGORIES = {
    "coding", "career", "planning", "factual", "opinion", "mathematical", "general",
}
ROLE_MAP: Dict[str, str] = {
    "J1": "PROPOSER", "J2": "CRITIC", "J3": "REFINER", "J4": "RISK",
}
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"


def normalize_category(category: str) -> str:
    normalized = (category or "general").strip().lower()
    return normalized if normalized in SUPPORTED_CATEGORIES else "general"


def _load_json(name: str) -> Dict[str, Any]:
    try:
        with (CONFIG_ROOT / name).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, TypeError, ValueError):
        return {}


def _shrink(observed: float, samples: int, prior: float, minimum_samples: int) -> float:
    weight = min(1.0, max(0, samples) / max(1, minimum_samples))
    return observed * weight + prior * (1.0 - weight)


def score_model(model_row: LLMPool, category: str) -> Dict[str, Any]:
    category = normalize_category(category)
    selector_config = _load_json("selector.json")
    capabilities = _load_json("model_capabilities.json")
    weights = selector_config.get("weights") or {}
    prior = float(selector_config.get("cold_start_prior", 0.5))
    minimum_samples = int(selector_config.get("minimum_samples", 10))
    latency_reference = max(1.0, float(selector_config.get("latency_reference_ms", 4000)))

    stat = (
        ModelRollingStat.objects.filter(model=model_row, category=category).first()
        or ModelRollingStat.objects.filter(model=model_row, category="general").first()
    )
    samples = int(getattr(stat, "completed_total", 0) or 0)
    appearances = int(getattr(stat, "appearances_total", 0) or 0)
    registry_caps = (capabilities.get("models") or {}).get(model_row.model_id, {})
    model_caps = model_row.capabilities_json or registry_caps
    capability = float(model_caps.get(category, model_caps.get("general", prior)) or 0.0)

    schema = _shrink(float(getattr(stat, "schema_valid_rate", prior) or 0.0), samples, prior, minimum_samples)
    acceptance = _shrink(
        float(getattr(stat, "user_acceptance_rate", prior) or 0.0),
        int(getattr(stat, "feedback_events_total", 0) or 0),
        prior,
        minimum_samples,
    )
    majority = _shrink(float(getattr(stat, "win_rate_in_majority", prior) or 0.0), samples, prior, minimum_samples)
    errors = int(getattr(stat, "error_total", 0) or 0)
    reliability = _shrink(1.0 - (errors / appearances if appearances else 0.0), appearances, prior, minimum_samples)
    latency = float(getattr(stat, "avg_latency_ms", 0.0) or 0.0)
    latency_score = prior if latency <= 0 else 1.0 / (1.0 + math.log1p(latency / latency_reference))

    components = {
        "capability": max(0.0, min(1.0, capability)),
        "schema_validity": max(0.0, min(1.0, schema)),
        "acceptance": max(0.0, min(1.0, acceptance)),
        "majority_alignment": max(0.0, min(1.0, majority)),
        "reliability": max(0.0, min(1.0, reliability)),
        "latency": max(0.0, min(1.0, latency_score)),
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    return {
        "model_id": model_row.model_id,
        "provider": model_row.provider,
        "score": round(total, 8),
        "components": components,
        "samples": samples,
        "appearances": appearances,
        "selector_version": selector_config.get("version", "selector-v2"),
        "capability_version": capabilities.get("version", "capabilities-v1"),
    }


def select_jury_roster(
    category: str,
    k: int = JURY_SIZE,
    *,
    seed: int = 0,
    return_trace: bool = False,
    allowed_model_ids: List[str] | None = None,
) -> Tuple[List[dict], Dict[str, str]] | Tuple[List[dict], Dict[str, str], Dict[str, Any]]:
    category = normalize_category(category)
    base_queryset = LLMPool.objects.filter(enabled=True)
    if allowed_model_ids:
        base_queryset = base_queryset.filter(model_id__in=allowed_model_ids)
    enabled_count = base_queryset.count()
    queryset = base_queryset.exclude(health_status="UNHEALTHY")
    enabled = list(queryset.order_by("model_id"))
    scored = [(model, score_model(model, category)) for model in enabled]
    rng = random.Random(seed)
    tie_breakers = {model.model_id: rng.random() for model, _ in scored}
    scored.sort(key=lambda item: (-item[1]["score"], tie_breakers[item[0].model_id], item[0].model_id))
    for rank, (_, breakdown) in enumerate(scored, start=1):
        breakdown["rank"] = rank

    picked: List[Tuple[LLMPool, Dict[str, Any]]] = []
    used_providers = set()
    for item in scored:
        provider = (item[0].provider or "unknown").lower()
        if provider not in used_providers:
            picked.append(item)
            used_providers.add(provider)
        if len(picked) == k:
            break
    if len(picked) < k:
        for item in scored:
            if item not in picked:
                picked.append(item)
            if len(picked) == k:
                break

    fallback_reason = "" if len(picked) >= k else f"ONLY_{len(picked)}_HEALTHY_ENABLED_CANDIDATES"
    roster = [
        {
            "juror_id": f"J{index}",
            "model_id": model.model_id,
            "provider": model.provider,
            "name": model.name,
            "selection_score": score["score"],
        }
        for index, (model, score) in enumerate(picked, start=1)
    ]
    role_map = {f"J{i}": ROLE_MAP.get(f"J{i}", "GENERALIST") for i in range(1, len(roster) + 1)}
    trace = {
        "category": category,
        "seed": seed,
        "candidate_count": len(enabled),
        "enabled_count": enabled_count,
        "excluded_unhealthy_count": enabled_count - len(enabled),
        "candidates": [score for _, score in scored],
        "selected_model_ids": [item["model_id"] for item in roster],
        "fallback_reason": fallback_reason,
        "selector_version": scored[0][1]["selector_version"] if scored else "selector-v2",
        "capability_version": scored[0][1]["capability_version"] if scored else "capabilities-v1",
    }
    if return_trace:
        return roster, role_map, trace
    return roster, role_map
