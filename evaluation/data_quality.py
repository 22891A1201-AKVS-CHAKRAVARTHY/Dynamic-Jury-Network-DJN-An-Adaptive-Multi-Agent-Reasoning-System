from __future__ import annotations

from typing import Any, Dict, List

from djn_db.models import DJNRun, JurorResponse, ModelRollingStat


def check_run(run: DJNRun) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    rounds = list(run.rounds.prefetch_related("juror_responses").all())
    roster_size = len(run.jury_roster_json or [])
    indices = [row.round_index for row in rounds]
    if indices != list(range(1, len(indices) + 1)):
        issues.append({"severity": "ERROR", "code": "NON_CONTIGUOUS_ROUND_INDICES"})
    if not run.experiment_config_id or not run.experiment_config_json:
        issues.append({"severity": "ERROR", "code": "MISSING_EXPERIMENT_CONFIG"})
    if not run.selector_version or not run.prompt_versions_json:
        issues.append({"severity": "ERROR", "code": "MISSING_VERSION_METADATA"})
    if not (run.final_answer or "").strip():
        issues.append({"severity": "ERROR", "code": "MISSING_FINAL_ANSWER"})
    for index, round_row in enumerate(rounds):
        responses = list(round_row.juror_responses.all())
        if roster_size and len(responses) != roster_size:
            issues.append({"severity": "ERROR", "code": "ROSTER_RESPONSE_COUNT_MISMATCH", "round": round_row.round_index})
        if not 0.0 <= float(round_row.agreement or 0.0) <= 1.0:
            issues.append({"severity": "ERROR", "code": "AGREEMENT_OUT_OF_RANGE", "round": round_row.round_index})
        continuing = index < len(rounds) - 1
        if continuing and not round_row.handoff_tldr_json:
            issues.append({"severity": "ERROR", "code": "MISSING_CONTINUING_HANDOFF", "round": round_row.round_index})
        if continuing and round_row.handoff_tldr_json and not round_row.handoff_hash:
            issues.append({"severity": "ERROR", "code": "MISSING_HANDOFF_HASH", "round": round_row.round_index})
        if round_row.judge_schema_valid is False and not round_row.judge_error:
            issues.append({"severity": "WARNING", "code": "MISSING_JUDGE_ERROR", "round": round_row.round_index})
        latencies = [response.latency_ms for response in responses if response.latency_ms is not None]
        if len(latencies) > 1 and len(set(latencies)) == 1:
            issues.append({"severity": "WARNING", "code": "SUSPICIOUS_DUPLICATE_JUROR_LATENCIES", "round": round_row.round_index})
        valid = sum(1 for response in responses if response.schema_valid)
        if round_row.agreement is not None and valid == 0 and round_row.agreement > 0:
            issues.append({"severity": "ERROR", "code": "AGREEMENT_WITHOUT_VALID_OUTPUTS", "round": round_row.round_index})
        for response in responses:
            if response.token_total is None and not response.usage_unavailable_reason:
                issues.append({"severity": "WARNING", "code": "MISSING_USAGE_REASON", "round": round_row.round_index, "juror": response.juror_id})
    if run.stop_reason == "THRESHOLD_MET" and rounds and rounds[-1].agreement is not None:
        threshold = float((run.experiment_config_json or {}).get("threshold", 0.75))
        if rounds[-1].agreement < threshold:
            issues.append({"severity": "ERROR", "code": "INVALID_THRESHOLD_STOP"})
    if run.stop_reason not in {"THRESHOLD_MET", "STAGNATION", "MAX_ROUNDS"}:
        issues.append({"severity": "ERROR", "code": "INVALID_OR_MISSING_STOP_REASON"})
    confidence_rank = {"": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    if confidence_rank.get(run.consensus_confidence, 99) > confidence_rank.get(run.raw_judge_confidence, 99):
        issues.append({"severity": "ERROR", "code": "CONFIDENCE_CAP_INCREASED_CONFIDENCE"})
    return issues


def check_all_runs() -> Dict[str, Any]:
    results = []
    for run in DJNRun.objects.prefetch_related("rounds__juror_responses"):
        issues = check_run(run)
        if issues:
            results.append({"run_id": run.session_id, "issues": issues})
    stat_issues = []
    for stat in ModelRollingStat.objects.select_related("model"):
        source = JurorResponse.objects.filter(
            model_id_snapshot=stat.model.model_id,
            round__run__category=stat.category,
        )
        source_count = source.count()
        valid_count = source.filter(schema_valid=True).count()
        if stat.appearances_total != source_count:
            stat_issues.append({
                "severity": "ERROR", "code": "ROLLING_APPEARANCE_MISMATCH",
                "model": stat.model.model_id, "category": stat.category,
            })
        if stat.schema_valid_total != valid_count:
            stat_issues.append({
                "severity": "ERROR", "code": "ROLLING_SCHEMA_TOTAL_MISMATCH",
                "model": stat.model.model_id, "category": stat.category,
            })
    if stat_issues:
        results.append({"run_id": "__rolling_statistics__", "issues": stat_issues})
    critical = sum(1 for result in results for issue in result["issues"] if issue["severity"] == "ERROR")
    return {"runs_checked": DJNRun.objects.count(), "runs_with_issues": len(results), "critical_count": critical, "results": results}
