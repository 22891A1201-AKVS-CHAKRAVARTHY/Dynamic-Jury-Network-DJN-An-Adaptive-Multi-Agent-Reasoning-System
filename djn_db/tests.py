import json
from unittest.mock import patch

from django.test import TestCase
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from djn_engine.audit import UsageRecord, estimate_cost, extract_usage, stable_hash
from djn_engine.orchestration import run_djn_once
from djn_engine.privacy import sanitize
from djn_engine.run import (
    JUROR_PROMPT_VERSION,
    ROLE_INSTRUCTIONS,
    ROLE_INSTRUCTIONS_VERSION,
    build_role_aware_juror_prompt,
    _cap_confidence,
)
from djn_engine.schemas import ExperimentConfig, RoundSummary
from evaluation.scorers import score_answer
from evaluation.data_quality import check_run

from .db_writer import upsert_run, write_round
from .models import DJNRound, JurorResponse, LLMPool, ModelRollingStat, RunFeedback
from .selector import normalize_category, select_jury_roster
from .stats import apply_feedback, rebuild_all_stats


class RolePromptAndAuditTests(TestCase):
    def test_all_roles_render_distinct_instructions(self):
        rendered = {}
        for role, instruction in ROLE_INSTRUCTIONS.items():
            messages = build_role_aware_juror_prompt(role).format_messages(
                query="Should this system be deployed?", round_context="No previous round.",
            )
            system = messages[0].content
            self.assertIn(f"Assigned role: {role}", system)
            self.assertIn(instruction, system)
            rendered[role] = system
        self.assertEqual(len(set(rendered.values())), len(ROLE_INSTRUCTIONS))

    def test_writer_persists_role_audit_metadata(self):
        run = upsert_run({
            "session_id": "test-role-audit", "q_raw": "Role audit verification",
            "q_final": "Role audit verification", "category": "general",
            "jury_roster": [], "role_map": {"J1": "PROPOSER"}, "final": {},
        })
        round_row = write_round(run, {
            "round": 1, "agreement": 1.0, "majority_label": "YES",
            "outputs": [{
                "juror_id": "J1", "model_id": "", "role": "PROPOSER",
                "role_instruction": ROLE_INSTRUCTIONS["PROPOSER"],
                "role_instruction_version": ROLE_INSTRUCTIONS_VERSION,
                "juror_prompt_version": JUROR_PROMPT_VERSION,
                "verdict_label": "YES", "tldr": "Verification output.",
                "reasoning": ["Test one", "Test two", "Test three"],
                "status": "OK", "schema_valid": True,
            }],
        })
        response = JurorResponse.objects.get(round=round_row, juror_id="J1")
        self.assertEqual(response.role, "PROPOSER")
        self.assertEqual(response.role_instruction, ROLE_INSTRUCTIONS["PROPOSER"])
        self.assertEqual(response.role_instruction_version, ROLE_INSTRUCTIONS_VERSION)
        self.assertEqual(response.juror_prompt_version, JUROR_PROMPT_VERSION)

    def test_writer_persists_inter_round_handoff(self):
        handoff = RoundSummary(
            common_ground=["Monitoring is required."],
            key_disagreements=["Deployment timing remains disputed."],
            open_questions=["What threshold is acceptable?"],
            current_best_label="CONDITIONAL",
            why_this_label="Safeguards are required before deployment.",
        ).model_dump()
        run = upsert_run({
            "session_id": "test-handoff", "q_raw": "Test", "q_final": "Test",
            "category": "general", "jury_roster": [], "role_map": {}, "final": {},
        })
        write_round(run, {
            "round": 1, "agreement": 0.75, "majority_label": "CONDITIONAL",
            "handoff_tldr": handoff, "handoff_hash": stable_hash(handoff),
            "handoff_prompt_version": "round-summary-prompt-v1",
            "handoff_schema_version": "round-summary-schema-v1",
            "handoff_schema_valid": True, "outputs": [],
        })
        saved = DJNRound.objects.get(run=run, round_index=1)
        self.assertEqual(saved.handoff_tldr_json, handoff)
        self.assertEqual(saved.handoff_hash, stable_hash(handoff))
        self.assertTrue(saved.handoff_schema_valid)

    def test_usage_extraction_is_provider_neutral(self):
        message = AIMessage(
            content="{}",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        usage = extract_usage(message)
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.total_tokens), (10, 5, 15))
        self.assertEqual(usage.unavailable_reason, "")

    @patch("djn_engine.audit.load_price_config")
    def test_cost_calculation_uses_versioned_price_config(self, load_prices):
        load_prices.return_value = {
            "version": "test-prices-v1", "currency": "USD",
            "models": {"mock:model": {"input_per_million": 1.0, "output_per_million": 2.0}},
        }
        cost = estimate_cost("mock:model", UsageRecord(prompt_tokens=1_000_000, completion_tokens=500_000))
        self.assertEqual(cost["amount"], 2.0)
        self.assertEqual(cost["price_version"], "test-prices-v1")

    def test_confidence_cap_never_presents_weak_consensus_as_high(self):
        judge = {"confidence": "HIGH"}
        _cap_confidence(judge, agreement=0.5, threshold=0.75, stop_reason="MAX_ROUNDS")
        self.assertNotEqual(judge["confidence"], "HIGH")

    def test_category_normalization_is_explicit(self):
        self.assertEqual(normalize_category("MATHEMATICAL"), "mathematical")
        self.assertEqual(normalize_category("not-a-category"), "general")

    def test_feedback_is_idempotent_and_reversible(self):
        model = LLMPool.objects.create(name="Model", provider="mock", model_id="mock:model")
        run = upsert_run({
            "session_id": "feedback-run", "q_raw": "Test", "q_final": "Test",
            "category": "general", "jury_roster": [], "role_map": {},
            "final": {"final_answer": "Completed answer"},
        })
        write_round(run, {
            "round": 1, "agreement": 1.0, "majority_label": "YES",
            "outputs": [{
                "juror_id": "J1", "model_id": model.model_id, "verdict_label": "YES",
                "tldr": "ok", "reasoning": [], "status": "OK", "schema_valid": True,
                "latency_ms": 100,
            }],
        })
        apply_feedback(run.session_id, "voter-1", 1)
        apply_feedback(run.session_id, "voter-1", 1)
        self.assertEqual(RunFeedback.objects.count(), 1)
        stat = ModelRollingStat.objects.get(model=model, category="general")
        self.assertEqual((stat.feedback_events_total, stat.user_accepts_total), (1, 1))
        apply_feedback(run.session_id, "voter-1", -1)
        stat = ModelRollingStat.objects.get(model=model, category="general")
        self.assertEqual((stat.feedback_events_total, stat.user_accepts_total), (1, 0))

    def test_stats_rebuild_handles_failures_and_missing_latency(self):
        model = LLMPool.objects.create(name="Stats model", provider="mock", model_id="mock:stats")
        run = upsert_run({
            "session_id": "stats-run", "q_raw": "Test", "q_final": "Test",
            "category": "general", "final": {"final_answer": "Complete"},
        })
        write_round(run, {
            "round": 1, "agreement": 0.0, "majority_label": "UNKNOWN",
            "outputs": [{
                "juror_id": "J1", "model_id": model.model_id, "status": "FAILED",
                "schema_valid": False, "latency_ms": None, "error_msg": "mock failure",
            }],
        })
        rebuild_all_stats()
        first = ModelRollingStat.objects.get(model=model, category="general")
        self.assertEqual((first.appearances_total, first.error_total, first.latency_sample_count), (1, 1, 0))
        rebuild_all_stats()
        second = ModelRollingStat.objects.get(model=model, category="general")
        self.assertEqual((second.appearances_total, second.error_total), (1, 1))

    def test_clarification_audit_fields_persist_for_all_paths(self):
        cases = [
            ("none", False, [], [], False),
            ("answered", True, ["Which scope?"], ["Pilot scope"], False),
            ("skipped", True, ["Which scope?"], [], True),
        ]
        for suffix, used, questions, answers, skipped in cases:
            run = upsert_run({
                "session_id": f"clarifier-{suffix}", "q_raw": "Raw", "q_final": "Normalized",
                "category": "general", "clarifier_used": used,
                "clarifier_questions": questions, "clarifier_answers": answers,
                "clarifier_skipped": skipped, "assumptions": ["Pilot only"], "final": {},
            })
            self.assertEqual(run.clarifier_used, used)
            self.assertEqual(run.clarifier_questions_json, questions)
            self.assertEqual(run.clarifier_answers_json, answers)
            self.assertEqual(run.clarifier_skipped, skipped)

    def test_data_quality_flags_incomplete_audit_record(self):
        run = upsert_run({
            "session_id": "quality-failure", "q_raw": "Raw", "q_final": "Raw",
            "category": "general", "final": {},
        })
        codes = {issue["code"] for issue in check_run(run)}
        self.assertIn("MISSING_EXPERIMENT_CONFIG", codes)
        self.assertIn("MISSING_FINAL_ANSWER", codes)

    def test_selector_is_reproducible_and_records_breakdowns(self):
        for index in range(5):
            LLMPool.objects.create(
                name=f"Model {index}", provider=f"provider-{index % 2}",
                model_id=f"mock:{index}", capabilities_json={"general": 0.5},
            )
        first = select_jury_roster("unknown-category", seed=42, return_trace=True)
        second = select_jury_roster("unknown-category", seed=42, return_trace=True)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2]["selected_model_ids"], second[2]["selected_model_ids"])
        self.assertIn("components", first[2]["candidates"][0])

    def test_config_ids_and_offline_scorers_are_deterministic(self):
        self.assertEqual(ExperimentConfig(seed=7).config_id, ExperimentConfig(seed=7).config_id)
        self.assertTrue(score_answer("Answer 42", "answer 42", "normalized_match")["correct"])

    def test_privacy_sanitizer_preserves_token_counts(self):
        output = sanitize({"authorization": "Bearer secret", "token_total": 15, "email": "a@example.com"})
        self.assertEqual(output["authorization"], "[REDACTED_SECRET]")
        self.assertEqual(output["token_total"], 15)
        self.assertEqual(output["email"], "[REDACTED_EMAIL]")

    @patch("djn_engine.orchestration.build_llm")
    def test_full_mocked_two_round_run_records_handoff_and_call_telemetry(self, mocked_build):
        labels = {"J1": "YES", "J2": "NO", "J3": "CONDITIONAL", "J4": "NO"}

        def factory(config):
            def respond(prompt_value):
                system = prompt_value.to_messages()[0].content
                if "next round" in system.lower():
                    content = json.dumps({
                        "common_ground": ["Shared point"], "key_disagreements": ["Disputed point"],
                        "open_questions": ["Open point"], "current_best_label": "NO",
                        "why_this_label": "Two of four jurors currently prefer NO.",
                    })
                elif "Moderator/Judge" in system:
                    content = json.dumps({
                        "final_recommendation": "The mocked recommendation is sufficiently long for schema validation and remains conditional on evidence.",
                        "why": ["First reason", "Second reason"], "confidence": "HIGH",
                        "common_ground": [], "main_disagreement": [], "conditional_guidance": [],
                    })
                else:
                    content = json.dumps({
                        "verdict_label": labels.get(config.name, "NO"), "tldr": "Mocked juror output.",
                        "reasoning": ["One", "Two", "Three"],
                    })
                return AIMessage(
                    content=content,
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
            return RunnableLambda(respond)

        mocked_build.side_effect = factory
        result = run_djn_once(
            "Should this mocked system be deployed?",
            experiment_config=ExperimentConfig(max_rounds=2, threshold=0.75),
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["rounds"]), 2)
        self.assertTrue(result["rounds"][0]["handoff_tldr"])
        self.assertEqual(len(result["rounds"][0]["handoff_hash"]), 64)
        self.assertEqual(result["rounds"][0]["outputs"][0]["token_total"], 15)
        self.assertEqual(result["rounds"][0]["outputs"][0]["juror_prompt_version"], "juror-role-prompt-v2")
        self.assertIsNotNone(result["duration_ms"])

    @patch("djn_engine.orchestration.build_llm")
    def test_invalid_judge_output_is_preserved_as_an_auditable_failure(self, mocked_build):
        def factory(config):
            def respond(_):
                if config.name == "judge_gemini":
                    return AIMessage(content="{}")
                return AIMessage(content=json.dumps({
                    "verdict_label": "YES", "tldr": "Valid juror output.",
                    "reasoning": ["One", "Two", "Three"],
                }))
            return RunnableLambda(respond)

        mocked_build.side_effect = factory
        result = run_djn_once(
            "A sufficiently specified mocked query for judge failure testing.",
            experiment_config=ExperimentConfig(max_rounds=1),
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["rounds"][0]["judge_schema_valid"])
        self.assertTrue(result["rounds"][0]["judge_error"])
        self.assertEqual(result["final"], "")
