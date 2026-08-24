from __future__ import annotations

from typing import Any, Dict, Optional

from django.db import transaction
from django.utils import timezone

from .models import DJNRun, DJNRound, JurorResponse, LLMPool


def _get_model_row(model_id: str):
    try:
        return LLMPool.objects.get(model_id=model_id)
    except LLMPool.DoesNotExist:
        return None


@transaction.atomic
def upsert_run(payload: Dict[str, Any]) -> DJNRun:
    sid = payload["session_id"]
    run, _ = DJNRun.objects.get_or_create(
        session_id=sid,
        defaults={"created_at": timezone.now(), "q_raw": payload.get("q_raw", payload.get("q_final", ""))},
    )
    simple_fields = {
        "q_raw": "q_raw", "q_final": "q_final", "category": "category",
        "selector_version": "selector_version", "selection_seed": "selection_seed",
        "capability_version": "capability_version", "experiment_config_id": "experiment_config_id",
        "benchmark_task_id": "benchmark_task_id", "scorer_version": "scorer_version",
        "duration_ms": "duration_ms",
    }
    for payload_key, field_name in simple_fields.items():
        if payload_key in payload:
            setattr(run, field_name, payload[payload_key])
    run.category_confidence = payload.get("category_confidence", run.category_confidence) or 0.0
    run.user_constraints_json = payload.get("user_constraints", run.user_constraints_json) or {}
    run.missing_fields_json = payload.get("missing_fields", run.missing_fields_json) or []
    run.assumptions_json = payload.get("assumptions", run.assumptions_json) or []
    run.jury_roster_json = payload.get("jury_roster", run.jury_roster_json) or []
    run.role_map_json = payload.get("role_map", run.role_map_json) or {}
    run.clarifier_used = bool(payload.get("clarifier_used", run.clarifier_used))
    run.clarifier_questions_json = payload.get("clarifier_questions", run.clarifier_questions_json) or []
    run.clarifier_answers_json = payload.get("clarifier_answers", run.clarifier_answers_json) or []
    run.clarifier_skipped = bool(payload.get("clarifier_skipped", run.clarifier_skipped))
    run.selection_trace_json = payload.get("selection_trace", run.selection_trace_json) or {}
    run.experiment_config_json = payload.get("experiment_config", run.experiment_config_json) or {}
    run.prompt_versions_json = payload.get("prompt_versions", run.prompt_versions_json) or {}
    run.pre_run_calls_json = payload.get("pre_run_calls", run.pre_run_calls_json) or []
    final = payload.get("final") or {}
    run.final_label = final.get("final_label", run.final_label)
    run.final_answer = final.get("final_answer", run.final_answer)
    run.final_confidence = final.get("confidence", run.final_confidence)
    run.raw_judge_confidence = final.get("raw_judge_confidence", run.raw_judge_confidence)
    run.consensus_confidence = final.get("consensus_confidence", run.consensus_confidence)
    run.confidence_policy_version = final.get("confidence_policy_version", run.confidence_policy_version)
    run.stop_reason = final.get("stop_reason", run.stop_reason)
    run.save()
    return run


@transaction.atomic
def write_round(run: DJNRun, round_payload: Dict[str, Any]) -> DJNRound:
    index = int(round_payload["round"])
    row, _ = DJNRound.objects.get_or_create(run=run, round_index=index)
    row.agreement = round_payload.get("agreement")
    row.majority_label = round_payload.get("majority_label", "") or ""
    row.improvement = round_payload.get("improvement")
    row.stagnation_flag = bool(round_payload.get("stagnation_flag", False))
    row.verdict_distribution_json = round_payload.get("verdict_distribution", {}) or {}
    row.tldr_similarity_score = round_payload.get("tldr_similarity_score")
    row.effective_agreement_score = round_payload.get("effective_agreement_score")
    row.handoff_tldr_json = round_payload.get("handoff_tldr", {}) or {}
    row.handoff_hash = round_payload.get("handoff_hash", "") or ""
    row.handoff_prompt_version = round_payload.get("handoff_prompt_version", "") or ""
    row.handoff_schema_version = round_payload.get("handoff_schema_version", "") or ""
    row.handoff_schema_valid = round_payload.get("handoff_schema_valid")
    row.handoff_error = round_payload.get("handoff_error", "") or ""
    row.handoff_model_id = round_payload.get("handoff_model_id", "") or ""
    row.handoff_latency_ms = round_payload.get("handoff_latency_ms")
    row.handoff_queue_ms = round_payload.get("handoff_queue_ms")
    row.handoff_usage_json = round_payload.get("handoff_usage", {}) or {}
    row.handoff_cost_estimate = round_payload.get("handoff_cost_estimate")
    row.judge_output_json = round_payload.get("judge_output", {}) or {}
    row.judge_schema_valid = round_payload.get("judge_schema_valid")
    row.judge_error = round_payload.get("judge_error", "") or ""
    row.judge_model_id = round_payload.get("judge_model_id", "") or ""
    row.judge_latency_ms = round_payload.get("judge_latency_ms")
    row.judge_queue_ms = round_payload.get("judge_queue_ms")
    row.judge_usage_json = round_payload.get("judge_usage", {}) or {}
    row.judge_cost_estimate = round_payload.get("judge_cost_estimate")
    row.stop_reason = round_payload.get("stop_reason", "") or ""
    row.latency_ms = round_payload.get("latency_ms")
    row.save()

    for output in round_payload.get("outputs") or []:
        response, _ = JurorResponse.objects.get_or_create(
            round=row, juror_id=output.get("juror_id", ""),
        )
        response.role = output.get("role", response.role) or ""
        response.role_instruction = output.get("role_instruction", response.role_instruction) or ""
        response.role_instruction_version = output.get("role_instruction_version", response.role_instruction_version) or ""
        response.juror_prompt_version = output.get("juror_prompt_version", response.juror_prompt_version) or ""
        model_id = output.get("model_id", "") or ""
        response.model = _get_model_row(model_id)
        response.model_id_snapshot = model_id
        response.verdict_label = output.get("verdict_label", "") or ""
        response.tldr = output.get("tldr", "") or ""
        response.reasoning_json = output.get("reasoning", []) or []
        response.status = output.get("status", "OK") or "OK"
        response.schema_valid = bool(output.get("schema_valid", True))
        response.error_msg = output.get("error_msg", "") or ""
        response.latency_ms = output.get("latency_ms")
        response.queue_ms = output.get("queue_ms")
        response.retry_count = int(output.get("retry_count", 0) or 0)
        response.token_in = output.get("token_in")
        response.token_out = output.get("token_out")
        response.token_total = output.get("token_total")
        response.cached_tokens = output.get("cached_tokens")
        response.usage_unavailable_reason = output.get("usage_unavailable_reason", "") or ""
        response.cost_estimate = output.get("cost_estimate")
        response.cost_currency = output.get("cost_currency", "USD") or "USD"
        response.price_version = output.get("price_version", "") or ""
        response.save()
    return row


@transaction.atomic
def persist_engine_result(
    result: Dict[str, Any], *, q_raw: str, q_final: str, category: str,
    extra: Optional[Dict[str, Any]] = None,
) -> DJNRun:
    extra = extra or {}
    stop = result.get("run_stop") or {}
    rounds = result.get("rounds") or []
    final_label = (rounds[-1].get("majority_label") if rounds else "") or ""
    run = upsert_run({
        "session_id": result["run_id"], "q_raw": q_raw, "q_final": q_final,
        "category": category, "jury_roster": result.get("jury_roster") or [],
        "role_map": result.get("role_map") or {}, "selector_version": result.get("selector_version", ""),
        "selection_seed": result.get("selection_seed"), "selection_trace": result.get("selection_trace") or {},
        "capability_version": result.get("capability_version", ""),
        "experiment_config": result.get("experiment_config") or {},
        "experiment_config_id": result.get("experiment_config_id", ""),
        "prompt_versions": result.get("prompt_versions") or {},
        "benchmark_task_id": extra.get("benchmark_task_id", ""),
        "scorer_version": extra.get("scorer_version", ""), "duration_ms": result.get("duration_ms"),
        "final": {
            "final_label": final_label,
            "final_answer": result.get("final", ""),
            "confidence": stop.get("consensus_confidence") or "",
            "raw_judge_confidence": stop.get("raw_judge_confidence") or "",
            "consensus_confidence": stop.get("consensus_confidence") or "",
            "confidence_policy_version": stop.get("confidence_policy_version") or "",
            "stop_reason": stop.get("stop_reason") or "",
        },
    })
    for round_data in rounds:
        mapped = dict(round_data)
        mapped.update({
            "round": round_data.get("round"), "agreement": round_data.get("agreement_score"),
            "improvement": round_data.get("improvement_score"),
            "latency_ms": round_data.get("latency_ms_per_round"),
        })
        write_round(run, mapped)
    return run
