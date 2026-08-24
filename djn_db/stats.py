from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Tuple

from django.db import transaction

from .models import (
    DJNRun,
    JurorResponse,
    LLMPool,
    ModelRollingStat,
    RunFeedback,
)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


@transaction.atomic
def rebuild_all_stats() -> int:
    """Recompute rolling statistics from source rows; safe to run repeatedly."""
    accumulators: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
        "appearances": 0,
        "completed": 0,
        "errors": 0,
        "schema_valid": 0,
        "wins": 0,
        "disagreements": 0,
        "latency_count": 0,
        "latency_sum": 0,
        "feedback_events": 0,
        "accepts": 0,
    })

    responses = JurorResponse.objects.select_related("round__run").all()
    models_by_run: Dict[int, set[str]] = defaultdict(set)
    for response in responses:
        model_id = (response.model_id_snapshot or "").strip()
        if not model_id:
            continue
        category = response.round.run.category or "general"
        values = accumulators[(model_id, category)]
        values["appearances"] += 1
        models_by_run[response.round.run_id].add(model_id)
        if response.status == "OK":
            values["completed"] += 1
        else:
            values["errors"] += 1
        if response.schema_valid:
            values["schema_valid"] += 1
        if response.verdict_label and response.round.majority_label:
            if response.verdict_label == response.round.majority_label:
                values["wins"] += 1
            else:
                values["disagreements"] += 1
        if response.latency_ms is not None:
            values["latency_count"] += 1
            values["latency_sum"] += int(response.latency_ms)

    for feedback in RunFeedback.objects.select_related("run").all():
        category = feedback.run.category or "general"
        for model_id in models_by_run.get(feedback.run_id, set()):
            values = accumulators[(model_id, category)]
            values["feedback_events"] += 1
            if feedback.value == 1:
                values["accepts"] += 1

    ModelRollingStat.objects.all().delete()
    rows = []
    model_rows = {row.model_id: row for row in LLMPool.objects.all()}
    for (model_id, category), values in accumulators.items():
        model = model_rows.get(model_id)
        if model is None:
            continue
        judged = values["wins"] + values["disagreements"]
        rows.append(ModelRollingStat(
            model=model,
            category=category,
            appearances_total=values["appearances"],
            completed_total=values["completed"],
            error_total=values["errors"],
            schema_valid_total=values["schema_valid"],
            majority_win_total=values["wins"],
            disagreement_total=values["disagreements"],
            latency_sample_count=values["latency_count"],
            latency_sum_ms=values["latency_sum"],
            feedback_events_total=values["feedback_events"],
            user_accepts_total=values["accepts"],
            user_acceptance_rate=_safe_rate(values["accepts"], values["feedback_events"]),
            win_rate_in_majority=_safe_rate(values["wins"], judged),
            disagreement_rate=_safe_rate(values["disagreements"], judged),
            avg_latency_ms=_safe_rate(values["latency_sum"], values["latency_count"]),
            schema_valid_rate=_safe_rate(values["schema_valid"], values["appearances"]),
        ))
    ModelRollingStat.objects.bulk_create(rows)
    return len(rows)


def update_stats_for_run(run_id: str) -> None:
    if DJNRun.objects.filter(session_id=run_id).exists():
        rebuild_all_stats()


@transaction.atomic
def apply_feedback(run_id: str, voter_session: str, value: int) -> RunFeedback:
    if value not in (-1, 1):
        raise ValueError("Feedback value must be -1 or 1")
    run = DJNRun.objects.select_for_update().get(session_id=run_id)
    if not run.final_answer or not run.rounds.exists():
        raise ValueError("Feedback is allowed only for a completed persisted run")
    feedback, _ = RunFeedback.objects.update_or_create(
        run=run,
        voter_session=voter_session or "anonymous",
        defaults={"value": value},
    )
    run.user_feedback = value
    run.save(update_fields=["user_feedback"])
    rebuild_all_stats()
    return feedback
