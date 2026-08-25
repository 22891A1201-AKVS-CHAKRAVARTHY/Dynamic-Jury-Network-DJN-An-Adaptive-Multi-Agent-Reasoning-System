from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from .audit import invoke_with_telemetry, stable_hash
from .llms import LLMConfig, build_llm
from .pool import JUDGE, JURORS
from .run import (
    DEFAULT_ROLE_INSTRUCTION,
    ROLE_INSTRUCTIONS,
    ROLE_INSTRUCTIONS_VERSION,
    RoundState,
    _agreement_from_ok,
    _build_round_context,
    _cap_confidence,
    _compute_schema_valid_rate,
    _format_final_display,
    _safe_parse_judge,
    _safe_parse_juror,
    _safe_parse_round_summary,
    _verdict_distribution,
)
from .schemas import CallStatus, ExperimentConfig, JurorResult

try:
    from djn_db.selector import select_jury_roster
except Exception:
    select_jury_roster = None

JUROR_PROMPT_VERSION = "juror-role-prompt-v2"
JUDGE_PROMPT_VERSION = "judge-prompt-v2"
ROUND_SUMMARY_PROMPT_VERSION = "round-summary-prompt-v1"
ROUND_SUMMARY_SCHEMA_VERSION = "round-summary-schema-v1"
CONFIDENCE_POLICY_VERSION = "consensus-cap-v1"
SELECTOR_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "selector.json"

JUROR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a DJN juror.\nAssigned role: {role}\nRole responsibilities: {role_instruction}\n"
     "Reach an independent conclusion and do not blindly repeat prior context.\n"
     "Output only valid JSON with exactly these keys:\n"
     '{{"verdict_label":"STRING","tldr":"STRING","reasoning":["STRING","STRING","STRING"]}}'),
    ("user", "User query:\n{query}\n\nRound context:\n{round_context}\n\nReturn JSON now."),
])
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the DJN Moderator/Judge. Output only valid JSON with exactly these keys:\n"
     '{{"final_recommendation":"STRING","why":["STRING","STRING"],"confidence":"HIGH|MEDIUM|LOW",'
     '"common_ground":["STRING"],"main_disagreement":["STRING"],"conditional_guidance":["STRING"]}}'),
    ("user",
     "User query:\n{query}\n\nNumeric agreement: {agreement}\nOperational stop reason: {stop_reason}\n\n"
     "Validated juror outputs:\n{juror_text}\n\nReturn JSON now."),
])
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Summarize juror outputs for the next round. Output only valid JSON with exactly these keys:\n"
     '{{"common_ground":["STRING"],"key_disagreements":["STRING"],"open_questions":["STRING"],'
     '"current_best_label":"YES|NO|CONDITIONAL|UNKNOWN","why_this_label":"STRING"}}'),
    ("user", "User query:\n{query}\n\nValidated juror outputs:\n{juror_text}\n\nReturn JSON now."),
])


def default_experiment_config() -> ExperimentConfig:
    import os
    return ExperimentConfig(
        threshold=float(os.getenv("DJN_THRESHOLD", "0.75")),
        max_rounds=int(os.getenv("DJN_MAX_ROUNDS", "3")),
        min_ok_jurors=int(os.getenv("DJN_MIN_OK_JURORS", "2")),
        min_improvement=float(os.getenv("DJN_MIN_IMPROVEMENT", "0.05")),
        stagnation_rounds=int(os.getenv("DJN_STAGNATION_ROUNDS", "1")),
        max_concurrency=int(os.getenv("DJN_MAX_CONCURRENCY", "4")),
        seed=int(os.getenv("DJN_SELECTION_SEED", "0")),
    )


def _juror_prompt(role: str, role_mode: str):
    actual_role = role if role_mode == "conditioned" else "GENERALIST"
    return JUROR_PROMPT.partial(
        role=actual_role,
        role_instruction=ROLE_INSTRUCTIONS.get(actual_role, DEFAULT_ROLE_INSTRUCTION),
    )


def _failure_telemetry(call_type: str, config: LLMConfig, error: Exception) -> Dict[str, Any]:
    return {
        "call_type": call_type, "provider": config.provider, "model_id": config.model,
        "latency_ms": None, "queue_ms": None, "retry_count": 0,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                  "cached_tokens": None, "unavailable_reason": "CALL_FAILED"},
        "cost": {"amount": None, "currency": "USD", "price_version": "prices-v1",
                 "unavailable_reason": "CALL_FAILED"},
        "status": "FAILED", "error": f"{type(error).__name__}: {error}",
    }


def _select(config: ExperimentConfig, category: str):
    size = config.jury_size
    configured = {item.model: item for item in JURORS}
    selected: List[LLMConfig] = []
    trace: Dict[str, Any] = {}
    roles = {"J1": "PROPOSER", "J2": "CRITIC", "J3": "REFINER", "J4": "RISK"}
    if config.selector_mode == "fixed":
        for index, model_id in enumerate(config.fixed_roster[:size], start=1):
            source = configured.get(model_id)
            if source is None:
                raise ValueError(f"Fixed roster model is not configured: {model_id}")
            selected.append(LLMConfig(
                name=f"J{index}", provider=source.provider, model=source.model,
                temperature=config.temperature, base_url=source.base_url,
                api_key_env=source.api_key_env,
            ))
        trace = {
            "selector_version": config.selector_version, "capability_version": config.capability_version,
            "seed": config.seed, "selected_model_ids": config.fixed_roster[:size],
            "fallback_reason": "FIXED_ROSTER", "candidates": [],
        }
    elif select_jury_roster:
        try:
            roster, roles, trace = select_jury_roster(
                category, k=size, seed=config.seed, return_trace=True,
                allowed_model_ids=config.model_pool or None,
            )
            selected = []
            for item in roster:
                source = configured.get(item["model_id"])
                selected.append(LLMConfig(
                    name=item["juror_id"], provider=(item.get("provider") or "ollama_cloud").lower(),
                    model=item["model_id"], temperature=config.temperature,
                    base_url=getattr(source, "base_url", None),
                    api_key_env=getattr(source, "api_key_env", None),
                ))
        except Exception as exc:
            trace = {"fallback_reason": f"SELECTOR_ERROR:{type(exc).__name__}"}
    if (
        config.selector_mode == "dynamic"
        and trace.get("enabled_count", trace.get("candidate_count", 0)) > 0
        and len(selected) < size
    ):
        raise RuntimeError(
            f"Only {len(selected)} of {size} required healthy enabled candidates are available; "
            f"selector decision: {trace.get('fallback_reason') or 'INSUFFICIENT_CANDIDATES'}."
        )
    if len(selected) < size:
        try:
            fallback_ids = json.loads(SELECTOR_CONFIG_PATH.read_text(encoding="utf-8")).get("fallback_model_ids", [])
        except (OSError, ValueError, TypeError):
            fallback_ids = []
        selected_ids = {item.model for item in selected}
        allowed = set(config.model_pool) if config.model_pool else None
        for model_id in fallback_ids:
            source = configured.get(model_id)
            if source is None or model_id in selected_ids or (allowed is not None and model_id not in allowed):
                continue
            selected.append(LLMConfig(
                name=f"J{len(selected) + 1}", provider=source.provider, model=source.model,
                temperature=config.temperature, base_url=source.base_url,
                api_key_env=source.api_key_env,
            ))
            selected_ids.add(model_id)
            if len(selected) == size:
                break
        trace.update({
            "selector_version": config.selector_version, "capability_version": config.capability_version,
            "seed": config.seed, "selected_model_ids": [item.model for item in selected],
            "fallback_reason": trace.get("fallback_reason") or "DETERMINISTIC_NAMED_FALLBACK",
        })
    if len(selected) != size:
        raise RuntimeError(
            f"Requested jury size {size}, but only {len(selected)} enabled/configured models are available."
        )
    role_cycle = ["PROPOSER", "CRITIC", "REFINER", "RISK"]
    roles = {f"J{index}": role_cycle[(index - 1) % len(role_cycle)] for index in range(1, size + 1)}
    roster = [{"juror_id": item.name, "model_id": item.model, "provider": item.provider, "name": item.name} for item in selected]
    return selected, roster, roles, trace


def run_djn_once(
    query: str,
    category: str = "general",
    experiment_config: Optional[ExperimentConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_started = time.monotonic()
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "Empty query."}
    clarification_skipped = "[MODERATOR NOTE: The user skipped clarifications." in query
    config = experiment_config if isinstance(experiment_config, ExperimentConfig) else (
        ExperimentConfig.model_validate(experiment_config) if experiment_config is not None else default_experiment_config()
    )
    config.validate_combination()
    max_rounds = 1 if config.mode in {"single_model", "static_jury_one_round"} else config.max_rounds
    min_ok = 1 if config.mode == "single_model" else config.min_ok_jurors
    selected, roster, role_map, selection_trace = _select(config, category)

    def runnable_for(model_config: LLMConfig):
        role = role_map.get(model_config.name, "GENERALIST")
        chain = _juror_prompt(role, config.role_mode) | build_llm(model_config)

        def call(payload):
            try:
                message, telemetry = invoke_with_telemetry(
                    chain.invoke, payload, call_type="juror",
                    provider=model_config.provider, model_id=model_config.model,
                )
                result = _safe_parse_juror(model_config.name, model_config.model, message)
            except Exception as exc:
                result = JurorResult(
                    juror_id=model_config.name, model_id=model_config.model, output=None,
                    status=CallStatus(ok=False, err=f"{type(exc).__name__}: {exc}", raw=""),
                )
                telemetry = _failure_telemetry("juror", model_config, exc)
            return {"result": result, "telemetry": telemetry}
        return RunnableLambda(call)

    parallel = RunnableParallel({item.name: runnable_for(item) for item in selected})
    # Disabled ablations must not require credentials for model calls they do not make.
    judge_chain = (JUDGE_PROMPT | build_llm(JUDGE)) if config.synthesis_mode == "judge" else None
    summary_chain = (
        SUMMARY_PROMPT | build_llm(JUDGE)
        if config.handoff_mode == "structured" and max_rounds > 1
        else None
    )
    rounds: List[Dict[str, Any]] = []
    previous_agreement = None
    stagnation_hits = 0
    round_context = ""
    last_results = []
    last_judge_message = None
    last_judge = None
    last_judge_parse: Dict[str, Any] = {}
    raw_judge_confidence = ""
    stop_reason = "MAX_ROUNDS"
    best_available = False

    for round_index in range(1, max_rounds + 1):
        round_started = time.monotonic()
        juror_started = time.monotonic()
        bundled = parallel.invoke(
            {"query": query, "round_context": round_context},
            config={"max_concurrency": config.max_concurrency},
        )
        juror_wall = int((time.monotonic() - juror_started) * 1000)
        last_results = [item["result"] for item in bundled.values()]
        telemetry = {key: item["telemetry"] for key, item in bundled.items()}
        distribution, majority, agreement, valid_count = _verdict_distribution(last_results)
        schema_rate = _compute_schema_valid_rate(last_results, len(selected))
        improvement = None if previous_agreement is None else agreement - previous_agreement
        stagnated = improvement is not None and improvement < config.min_improvement
        stagnation_hits = stagnation_hits + 1 if stagnated else 0
        valid = [item for item in last_results if item.status.ok and item.status.raw]
        juror_text = "\n\n".join(f"[{item.juror_id}]\n{item.status.raw}" for item in valid)

        if config.stopping_mode == "dynamic" and valid_count >= min_ok and agreement >= config.threshold:
            stop_reason, best_available = "THRESHOLD_MET", False
        elif config.stopping_mode == "dynamic" and round_index >= 2 and stagnation_hits >= config.stagnation_rounds:
            stop_reason, best_available = "STAGNATION", True
        else:
            stop_reason, best_available = "MAX_ROUNDS", False
        if round_index == max_rounds and stop_reason == "MAX_ROUNDS":
            best_available = True
        continuing = round_index < max_rounds and stop_reason not in {"THRESHOLD_MET", "STAGNATION"}
        judge_stop_reason = "CONTINUE" if continuing else stop_reason

        judge_telemetry = {}
        if config.synthesis_mode == "judge" and valid_count > 0:
            try:
                last_judge_message, judge_telemetry = invoke_with_telemetry(
                    judge_chain.invoke,
                    {"query": query, "agreement": f"{agreement:.6f}",
                     "stop_reason": judge_stop_reason, "juror_text": juror_text},
                    call_type="judge", provider=JUDGE.provider, model_id=JUDGE.model,
                )
                last_judge_parse = _safe_parse_judge(last_judge_message)
                last_judge = last_judge_parse["output"].model_dump() if last_judge_parse.get("ok") else None
            except Exception as exc:
                last_judge_parse = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "raw": ""}
                last_judge = None
                judge_telemetry = _failure_telemetry("judge", JUDGE, exc)
        elif config.synthesis_mode == "judge":
            last_judge = None
            last_judge_parse = {
                "ok": False,
                "error": "Judge was not called because no valid juror outputs were available.",
                "raw": "",
            }
        else:
            last_judge = {
                "final_recommendation": f"Deterministic jury majority: {majority}.",
                "why": [f"Agreement was {agreement:.2f} across {valid_count} valid jurors.", "No judge model was called."],
                "confidence": "HIGH" if agreement >= config.threshold else "MEDIUM",
                "common_ground": [], "main_disagreement": [], "conditional_guidance": [],
            }
            last_judge_parse = {"ok": True, "raw": ""}
        if last_judge is not None:
            raw_judge_confidence = (last_judge.get("confidence") or "").upper()
            _cap_confidence(last_judge, agreement, config.threshold, stop_reason)
            if clarification_skipped or valid_count < min_ok:
                last_judge["confidence"] = "LOW"

        state = RoundState(
            round=round_index, n_ok=valid_count, verdict_distribution=distribution,
            majority_label=majority, agreement_score=agreement, improvement_score=improvement,
            stagnation_flag=stagnated,
            stop_reason=stop_reason if stop_reason in {"THRESHOLD_MET", "STAGNATION"} or round_index == max_rounds else None,
            best_available_used=best_available, latency_ms_per_round=0,
            model_latency_ms={key: value.get("latency_ms") for key, value in telemetry.items()},
            schema_valid_rate=schema_rate,
        )
        record = asdict(state)
        record.update({
            "juror_wall_latency_ms": juror_wall, "consensus_threshold": config.threshold,
            "judge_output": last_judge or {}, "judge_schema_valid": bool(last_judge_parse.get("ok")),
            "judge_error": last_judge_parse.get("error", ""),
            "judge_model_id": JUDGE.model if config.synthesis_mode == "judge" else "deterministic-majority",
            "judge_latency_ms": judge_telemetry.get("latency_ms"), "judge_queue_ms": judge_telemetry.get("queue_ms"),
            "judge_usage": judge_telemetry.get("usage", {}),
            "judge_cost_estimate": (judge_telemetry.get("cost") or {}).get("amount"),
            "outputs": [],
        })
        for result in last_results:
            call = telemetry.get(result.juror_id, {})
            usage, cost = call.get("usage") or {}, call.get("cost") or {}
            role = role_map.get(result.juror_id, "GENERALIST")
            record["outputs"].append({
                "juror_id": result.juror_id, "model_id": result.model_id, "role": role,
                "role_instruction": ROLE_INSTRUCTIONS.get(role, DEFAULT_ROLE_INSTRUCTION),
                "role_instruction_version": ROLE_INSTRUCTIONS_VERSION,
                "juror_prompt_version": JUROR_PROMPT_VERSION,
                "verdict_label": result.output.verdict_label if result.status.ok and result.output else "",
                "tldr": result.output.tldr if result.status.ok and result.output else "",
                "reasoning": result.output.reasoning if result.status.ok and result.output else [],
                "status": "OK" if result.status.ok else "FAILED", "schema_valid": bool(result.status.ok and result.output),
                "error_msg": result.status.err or "", "latency_ms": call.get("latency_ms"),
                "queue_ms": call.get("queue_ms"), "retry_count": call.get("retry_count", 0),
                "token_in": usage.get("prompt_tokens"), "token_out": usage.get("completion_tokens"),
                "token_total": usage.get("total_tokens"), "cached_tokens": usage.get("cached_tokens"),
                "usage_unavailable_reason": usage.get("unavailable_reason", ""),
                "cost_estimate": cost.get("amount"), "cost_currency": cost.get("currency", "USD"),
                "price_version": cost.get("price_version", config.price_version),
            })

        record.update({
            "handoff_tldr": {}, "handoff_hash": "", "handoff_prompt_version": ROUND_SUMMARY_PROMPT_VERSION,
            "handoff_schema_version": ROUND_SUMMARY_SCHEMA_VERSION, "handoff_schema_valid": None,
            "handoff_error": "", "handoff_model_id": "", "handoff_latency_ms": None,
            "handoff_queue_ms": None, "handoff_usage": {}, "handoff_cost_estimate": None,
        })
        if continuing and config.handoff_mode == "structured":
            try:
                message, call = invoke_with_telemetry(
                    summary_chain.invoke, {"query": query, "juror_text": juror_text},
                    call_type="summary", provider=JUDGE.provider, model_id=JUDGE.model,
                )
                parsed = _safe_parse_round_summary(message)
                record.update({
                    "handoff_schema_valid": bool(parsed.get("ok")), "handoff_error": parsed.get("error", ""),
                    "handoff_model_id": JUDGE.model, "handoff_latency_ms": call.get("latency_ms"),
                    "handoff_queue_ms": call.get("queue_ms"), "handoff_usage": call.get("usage", {}),
                    "handoff_cost_estimate": (call.get("cost") or {}).get("amount"),
                })
                if parsed.get("ok"):
                    summary = parsed["output"]
                    payload = summary.model_dump()
                    record["handoff_tldr"], record["handoff_hash"] = payload, stable_hash(payload)
                    round_context = _build_round_context(summary)
                else:
                    round_context = f"Current majority: {majority}\nAgreement: {agreement:.2f}\n"
            except Exception as exc:
                record["handoff_schema_valid"] = False
                record["handoff_error"] = f"{type(exc).__name__}: {exc}"
                round_context = f"Current majority: {majority}\nAgreement: {agreement:.2f}\n"
        elif continuing and config.handoff_mode == "raw":
            payload = {"previous_juror_responses": [item.status.raw for item in valid]}
            record.update({"handoff_tldr": payload, "handoff_hash": stable_hash(payload), "handoff_schema_valid": True})
            round_context = "\n\n".join(payload["previous_juror_responses"])
        else:
            round_context = ""
        record["latency_ms_per_round"] = int((time.monotonic() - round_started) * 1000)
        rounds.append(record)
        previous_agreement = agreement
        if stop_reason in {"THRESHOLD_MET", "STAGNATION"}:
            break

    if stop_reason not in {"THRESHOLD_MET", "STAGNATION"}:
        stop_reason, best_available = "MAX_ROUNDS", True
    if last_judge is None:
        final_display = "DJN could not synthesize a recommendation because no valid juror output was available. Inspect the run audit for provider and schema errors."
        final_text = ""
    else:
        final_display = _format_final_display(last_judge, last_judge_message, query)
        final_text = last_judge.get("final_recommendation", "") or ""
    consensus_confidence = (last_judge or {}).get("confidence")
    return {
        "ok": True, "q_raw": query, "q_final": query, "query": query,
        "run_id": str(uuid.uuid4()), "category": category, "jury_roster": roster,
        "role_map": role_map, "selection_trace": selection_trace,
        "selector_version": selection_trace.get("selector_version", config.selector_version),
        "selection_seed": config.seed,
        "capability_version": selection_trace.get("capability_version", config.capability_version),
        "experiment_config": config.snapshot(), "experiment_config_id": config.config_id,
        "prompt_versions": {
            "roles": ROLE_INSTRUCTIONS_VERSION, "juror": JUROR_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION, "summary": ROUND_SUMMARY_PROMPT_VERSION,
            "summary_schema": ROUND_SUMMARY_SCHEMA_VERSION,
        },
        "jurors": [{
            "juror_id": item.juror_id, "model_id": item.model_id, "ok": item.status.ok,
            "err": item.status.err, "raw": item.status.raw,
            "parsed": item.output.model_dump() if item.output else None,
        } for item in last_results],
        "judge": last_judge or {"ok": False, "error": last_judge_parse.get("error"), "raw": last_judge_parse.get("raw")},
        "final": final_text, "final_display": final_display,
        "metrics": _agreement_from_ok(last_results), "meta": config.snapshot(), "rounds": rounds,
        "duration_ms": int((time.monotonic() - run_started) * 1000),
        "run_stop": {
            "stop_reason": stop_reason, "best_available_used": best_available,
            "raw_judge_confidence": raw_judge_confidence,
            "consensus_confidence": consensus_confidence,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "final_confidence_level": consensus_confidence,
        },
        "run_metrics": {
            "schema_valid_rate_last_round": rounds[-1].get("schema_valid_rate") if rounds else 0.0,
            "agreement_last_round": rounds[-1].get("agreement_score") if rounds else 0.0,
            "n_ok_last_round": rounds[-1].get("n_ok") if rounds else 0,
        },
    }
