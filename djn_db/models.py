from django.db import models
from django.utils import timezone


class LLMPool(models.Model):
    name = models.CharField(max_length=120)
    provider = models.CharField(max_length=60, blank=True, default="")
    model_id = models.CharField(max_length=160, unique=True)
    enabled = models.BooleanField(default=True)
    tags_json = models.JSONField(default=list, blank=True)
    category_weights_json = models.JSONField(default=dict, blank=True)
    capabilities_json = models.JSONField(default=dict, blank=True)
    capability_version = models.CharField(max_length=40, blank=True, default="")
    health_status = models.CharField(max_length=20, default="UNKNOWN")
    health_checked_at = models.DateTimeField(null=True, blank=True)
    cost_tier = models.CharField(max_length=30, blank=True, default="")
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.model_id})"


class DJNRun(models.Model):
    session_id = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    q_raw = models.TextField()
    user_constraints_json = models.JSONField(default=dict, blank=True)
    category = models.CharField(max_length=30, default="general")
    category_confidence = models.FloatField(default=0.0)
    missing_fields_json = models.JSONField(default=list, blank=True)
    clarifier_used = models.BooleanField(default=False)
    clarifier_questions_json = models.JSONField(default=list, blank=True)
    clarifier_answers_json = models.JSONField(default=list, blank=True)
    clarifier_skipped = models.BooleanField(default=False)
    q_final = models.TextField(blank=True, default="")
    assumptions_json = models.JSONField(default=list, blank=True)
    jury_roster_json = models.JSONField(default=list, blank=True)
    role_map_json = models.JSONField(default=dict, blank=True)
    selector_version = models.CharField(max_length=40, blank=True, default="")
    selection_seed = models.BigIntegerField(null=True, blank=True)
    selection_trace_json = models.JSONField(default=dict, blank=True)
    capability_version = models.CharField(max_length=40, blank=True, default="")
    experiment_config_json = models.JSONField(default=dict, blank=True)
    experiment_config_id = models.CharField(max_length=64, blank=True, default="")
    prompt_versions_json = models.JSONField(default=dict, blank=True)
    pre_run_calls_json = models.JSONField(default=list, blank=True)
    benchmark_task_id = models.CharField(max_length=120, blank=True, default="")
    scorer_version = models.CharField(max_length=40, blank=True, default="")
    final_label = models.CharField(max_length=80, blank=True, default="")
    final_answer = models.TextField(blank=True, default="")
    final_confidence = models.CharField(max_length=10, blank=True, default="")
    raw_judge_confidence = models.CharField(max_length=10, blank=True, default="")
    consensus_confidence = models.CharField(max_length=10, blank=True, default="")
    confidence_policy_version = models.CharField(max_length=40, blank=True, default="")
    stop_reason = models.CharField(max_length=40, blank=True, default="")
    user_feedback = models.SmallIntegerField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"DJNRun {self.session_id} [{self.category}]"

    @property
    def query(self):
        """Compatibility alias used by the legacy Google Docs export helper."""
        return self.q_final


class DJNRound(models.Model):
    run = models.ForeignKey(DJNRun, on_delete=models.CASCADE, related_name="rounds")
    round_index = models.PositiveSmallIntegerField()
    created_at = models.DateTimeField(default=timezone.now)
    agreement = models.FloatField(null=True, blank=True)
    majority_label = models.CharField(max_length=80, blank=True, default="")
    improvement = models.FloatField(null=True, blank=True)
    stagnation_flag = models.BooleanField(default=False)
    verdict_distribution_json = models.JSONField(default=dict, blank=True)
    tldr_similarity_score = models.FloatField(null=True, blank=True)
    effective_agreement_score = models.FloatField(null=True, blank=True)
    handoff_tldr_json = models.JSONField(default=dict, blank=True)
    handoff_hash = models.CharField(max_length=64, blank=True, default="")
    handoff_prompt_version = models.CharField(max_length=40, blank=True, default="")
    handoff_schema_version = models.CharField(max_length=40, blank=True, default="")
    handoff_schema_valid = models.BooleanField(null=True, blank=True)
    handoff_error = models.TextField(blank=True, default="")
    handoff_model_id = models.CharField(max_length=160, blank=True, default="")
    handoff_latency_ms = models.IntegerField(null=True, blank=True)
    handoff_queue_ms = models.IntegerField(null=True, blank=True)
    handoff_usage_json = models.JSONField(default=dict, blank=True)
    handoff_cost_estimate = models.FloatField(null=True, blank=True)
    judge_output_json = models.JSONField(default=dict, blank=True)
    judge_schema_valid = models.BooleanField(null=True, blank=True)
    judge_error = models.TextField(blank=True, default="")
    judge_model_id = models.CharField(max_length=160, blank=True, default="")
    judge_latency_ms = models.IntegerField(null=True, blank=True)
    judge_queue_ms = models.IntegerField(null=True, blank=True)
    judge_usage_json = models.JSONField(default=dict, blank=True)
    judge_cost_estimate = models.FloatField(null=True, blank=True)
    stop_reason = models.CharField(max_length=40, blank=True, default="")
    latency_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["round_index"]
        constraints = [
            models.UniqueConstraint(fields=["run", "round_index"], name="uniq_run_roundindex"),
        ]

    def __str__(self):
        return f"Round {self.round_index} ({self.run.session_id})"


class JurorResponse(models.Model):
    round = models.ForeignKey(DJNRound, on_delete=models.CASCADE, related_name="juror_responses")
    juror_id = models.CharField(max_length=4)
    role = models.CharField(max_length=16, blank=True, default="")
    role_instruction = models.TextField(blank=True, default="")
    role_instruction_version = models.CharField(max_length=40, blank=True, default="")
    juror_prompt_version = models.CharField(max_length=40, blank=True, default="")
    model = models.ForeignKey(LLMPool, on_delete=models.SET_NULL, null=True, blank=True)
    model_id_snapshot = models.CharField(max_length=160, blank=True, default="")
    verdict_label = models.CharField(max_length=80, blank=True, default="")
    tldr = models.TextField(blank=True, default="")
    reasoning_json = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=16, default="OK")
    schema_valid = models.BooleanField(default=True)
    error_msg = models.TextField(blank=True, default="")
    latency_ms = models.IntegerField(null=True, blank=True)
    queue_ms = models.IntegerField(null=True, blank=True)
    retry_count = models.PositiveSmallIntegerField(default=0)
    token_in = models.IntegerField(null=True, blank=True)
    token_out = models.IntegerField(null=True, blank=True)
    token_total = models.IntegerField(null=True, blank=True)
    cached_tokens = models.IntegerField(null=True, blank=True)
    usage_unavailable_reason = models.CharField(max_length=80, blank=True, default="")
    cost_estimate = models.FloatField(null=True, blank=True)
    cost_currency = models.CharField(max_length=8, blank=True, default="USD")
    price_version = models.CharField(max_length=40, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["juror_id"]
        constraints = [
            models.UniqueConstraint(fields=["round", "juror_id"], name="uniq_round_juror"),
        ]

    def __str__(self):
        return f"{self.juror_id} {self.verdict_label} ({self.round})"


class ModelRollingStat(models.Model):
    model = models.ForeignKey(LLMPool, on_delete=models.CASCADE, related_name="rolling_stats")
    category = models.CharField(max_length=30, default="general")
    appearances_total = models.IntegerField(default=0)
    completed_total = models.IntegerField(default=0)
    error_total = models.IntegerField(default=0)
    schema_valid_total = models.IntegerField(default=0)
    majority_win_total = models.IntegerField(default=0)
    disagreement_total = models.IntegerField(default=0)
    latency_sample_count = models.IntegerField(default=0)
    latency_sum_ms = models.BigIntegerField(default=0)
    feedback_events_total = models.IntegerField(default=0)
    user_accepts_total = models.IntegerField(default=0)
    user_acceptance_rate = models.FloatField(default=0.0)
    win_rate_in_majority = models.FloatField(default=0.0)
    disagreement_rate = models.FloatField(default=0.0)
    avg_latency_ms = models.FloatField(default=0.0)
    schema_valid_rate = models.FloatField(default=0.0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["model__name", "category"]
        constraints = [
            models.UniqueConstraint(fields=["model", "category"], name="uniq_model_category_stat"),
        ]

    def __str__(self):
        return f"{self.model.name} [{self.category}]"


class RunFeedback(models.Model):
    run = models.ForeignKey(DJNRun, on_delete=models.CASCADE, related_name="feedback_records")
    voter_session = models.CharField(max_length=64)
    value = models.SmallIntegerField()
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "voter_session"], name="uniq_run_voter_feedback"),
            models.CheckConstraint(condition=models.Q(value__in=[-1, 1]), name="feedback_value_up_or_down"),
        ]


class BenchmarkTask(models.Model):
    task_id = models.CharField(max_length=120, unique=True)
    dataset = models.CharField(max_length=120)
    split = models.CharField(max_length=40, default="test")
    category = models.CharField(max_length=30, default="general")
    prompt = models.TextField()
    reference_answer_json = models.JSONField(default=dict, blank=True)
    scorer = models.CharField(max_length=40)
    scorer_version = models.CharField(max_length=40)
    source_url = models.URLField(blank=True, default="")
    license = models.CharField(max_length=120, blank=True, default="")
    metadata_json = models.JSONField(default=dict, blank=True)


class EvaluationResult(models.Model):
    task = models.ForeignKey(BenchmarkTask, on_delete=models.PROTECT, related_name="results")
    run = models.ForeignKey(DJNRun, on_delete=models.CASCADE, related_name="evaluation_results")
    experiment_config_id = models.CharField(max_length=64)
    scorer_version = models.CharField(max_length=40)
    score = models.FloatField(null=True, blank=True)
    correct = models.BooleanField(null=True, blank=True)
    details_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["task", "experiment_config_id"],
                name="uniq_task_experiment_result",
            ),
        ]
