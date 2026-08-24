from __future__ import annotations

from typing import Any, Dict

from .models import DJNRun


def build_run_audit(run: DJNRun) -> Dict[str, Any]:
    rounds = []
    for round_row in run.rounds.prefetch_related("juror_responses").all():
        outputs = []
        for response in round_row.juror_responses.all():
            outputs.append({
                "juror_id": response.juror_id,
                "model_id": response.model_id_snapshot,
                "role": response.role,
                "role_instruction": response.role_instruction,
                "role_instruction_version": response.role_instruction_version,
                "juror_prompt_version": response.juror_prompt_version,
                "verdict_label": response.verdict_label,
                "tldr": response.tldr,
                "reasoning": response.reasoning_json,
                "status": response.status,
                "schema_valid": response.schema_valid,
                "error_msg": response.error_msg,
                "latency_ms": response.latency_ms,
                "queue_ms": response.queue_ms,
                "retry_count": response.retry_count,
                "token_in": response.token_in,
                "token_out": response.token_out,
                "token_total": response.token_total,
                "cached_tokens": response.cached_tokens,
                "usage_unavailable_reason": response.usage_unavailable_reason,
                "cost_estimate": response.cost_estimate,
                "cost_currency": response.cost_currency,
                "price_version": response.price_version,
            })
        rounds.append({
            "round": round_row.round_index,
            "agreement_score": round_row.agreement,
            "majority_label": round_row.majority_label,
            "improvement_score": round_row.improvement,
            "stagnation_flag": round_row.stagnation_flag,
            "stop_reason": round_row.stop_reason,
            "verdict_distribution": round_row.verdict_distribution_json,
            "latency_ms_per_round": round_row.latency_ms,
            "judge_output": round_row.judge_output_json,
            "judge_schema_valid": round_row.judge_schema_valid,
            "judge_error": round_row.judge_error,
            "judge_model_id": round_row.judge_model_id,
            "judge_latency_ms": round_row.judge_latency_ms,
            "judge_queue_ms": round_row.judge_queue_ms,
            "judge_usage": round_row.judge_usage_json,
            "judge_cost_estimate": round_row.judge_cost_estimate,
            "handoff_tldr": round_row.handoff_tldr_json,
            "handoff_hash": round_row.handoff_hash,
            "handoff_prompt_version": round_row.handoff_prompt_version,
            "handoff_schema_version": round_row.handoff_schema_version,
            "handoff_schema_valid": round_row.handoff_schema_valid,
            "handoff_error": round_row.handoff_error,
            "handoff_model_id": round_row.handoff_model_id,
            "handoff_latency_ms": round_row.handoff_latency_ms,
            "handoff_queue_ms": round_row.handoff_queue_ms,
            "handoff_usage": round_row.handoff_usage_json,
            "handoff_cost_estimate": round_row.handoff_cost_estimate,
            "outputs": outputs,
        })
    last_round = rounds[-1] if rounds else {}
    return {
        "run_id": run.session_id,
        "ts_utc": run.created_at.isoformat(),
        "q_raw": run.q_raw,
        "q_final": run.q_final,
        "category": run.category,
        "category_confidence": run.category_confidence,
        "missing_fields": run.missing_fields_json,
        "clarifier_used": run.clarifier_used,
        "clarifier_questions": run.clarifier_questions_json,
        "clarifier_answers": run.clarifier_answers_json,
        "clarifier_skipped": run.clarifier_skipped,
        "assumptions": run.assumptions_json,
        "jury_roster": run.jury_roster_json,
        "role_map": run.role_map_json,
        "selector_version": run.selector_version,
        "selection_seed": run.selection_seed,
        "selection_trace": run.selection_trace_json,
        "capability_version": run.capability_version,
        "experiment_config": run.experiment_config_json,
        "experiment_config_id": run.experiment_config_id,
        "prompt_versions": run.prompt_versions_json,
        "pre_run_calls": run.pre_run_calls_json,
        "rounds": rounds,
        "final": run.final_answer,
        "final_display": run.final_answer,
        "run_stop": {
            "stop_reason": run.stop_reason,
            "raw_judge_confidence": run.raw_judge_confidence,
            "consensus_confidence": run.consensus_confidence,
            "confidence_policy_version": run.confidence_policy_version,
        },
        "run_metrics": {
            "agreement_last_round": last_round.get("agreement_score"),
            "schema_valid_rate_last_round": (
                sum(1 for item in last_round.get("outputs", []) if item.get("schema_valid"))
                / len(last_round.get("outputs", []))
                if last_round.get("outputs") else 0.0
            ),
            "n_ok_last_round": sum(
                1 for item in last_round.get("outputs", []) if item.get("status") == "OK"
            ),
        },
        "duration_ms": run.duration_ms,
    }
