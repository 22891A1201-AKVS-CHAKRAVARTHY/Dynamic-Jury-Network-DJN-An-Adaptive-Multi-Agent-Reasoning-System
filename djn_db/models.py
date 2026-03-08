from django.db import models
from django.utils import timezone


# --------- LLM Pool (candidates that can be selected as jurors) ---------
class LLMPool(models.Model):
    # Human label for admin/debug: "Ollama: llama3.1" / "NIM: mistral-nemotron" etc.
    name = models.CharField(max_length=120)
    provider = models.CharField(max_length=60, blank=True, default="")  # ollama | nvidia | openrouter | etc.
    model_id = models.CharField(max_length=160, unique=True)  # canonical id you call with
    enabled = models.BooleanField(default=True)

    # Category tags used in roster select (v1: top tag match; fallback list)
    # store like: ["coding","planning","general"]
    tags_json = models.JSONField(default=list, blank=True)

    # v1-ready weights for tie-break (default 1.0)
    # store like: {"coding":1.1,"planning":0.9,"general":1.0}
    category_weights_json = models.JSONField(default=dict, blank=True)

    # ops hints (optional but useful for selection later)
    cost_tier = models.CharField(max_length=30, blank=True, default="")  # "free|cheap|mid|expensive"
    notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.model_id})"


# --------- One DJN run per user query (maps to session_id in v1 protocol) ---------
class DJNRun(models.Model):
    # v1: session_id is uuid in payload; store as string so you can keep compatibility with your engine
    session_id = models.CharField(max_length=64, unique=True)

    created_at = models.DateTimeField(default=timezone.now)

    # Intake
    q_raw = models.TextField()
    user_constraints_json = models.JSONField(default=dict, blank=True)

    # Classify + assumptions (v1)
    category = models.CharField(max_length=30, default="general")
    category_confidence = models.FloatField(default=0.0)
    missing_fields_json = models.JSONField(default=list, blank=True)

    clarifier_used = models.BooleanField(default=False)
    clarifier_questions_json = models.JSONField(default=list, blank=True)
    clarifier_answers_json = models.JSONField(default=list, blank=True)

    q_final = models.TextField(blank=True, default="")
    assumptions_json = models.JSONField(default=list, blank=True)

    # Roster snapshot + role map (v1)
    jury_roster_json = models.JSONField(default=list, blank=True)  # [{juror_id:"J1", model_id:"..."}]
    role_map_json = models.JSONField(default=dict, blank=True)     # {"J1":"PROPOSER",...}

    # Final (v1)
    final_label = models.CharField(max_length=80, blank=True, default="")
    final_answer = models.TextField(blank=True, default="")
    final_confidence = models.CharField(max_length=10, blank=True, default="")  # HIGH|MEDIUM|LOW
    stop_reason = models.CharField(max_length=40, blank=True, default="")       # THRESHOLD_R1|MAX_ROUNDS|...

    # Feedback (MVP truth metric)
    user_feedback = models.SmallIntegerField(null=True, blank=True)  # 1=up, -1=down, null=unset

    # convenience for history page
    duration_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"DJNRun {self.session_id} [{self.category}]"


# --------- Each round for a run (max 2 in v1) ---------
class DJNRound(models.Model):
    run = models.ForeignKey(DJNRun, on_delete=models.CASCADE, related_name="rounds")
    round_index = models.PositiveSmallIntegerField()  # 1 or 2

    created_at = models.DateTimeField(default=timezone.now)

    # v1 scoring
    agreement = models.FloatField(null=True, blank=True)
    majority_label = models.CharField(max_length=80, blank=True, default="")
    improvement = models.FloatField(null=True, blank=True)  # only meaningful for round 2
    stagnation_flag = models.BooleanField(default=False)

    # distributions + summaries (DB-ready metrics list)
    verdict_distribution_json = models.JSONField(default=dict, blank=True)  # {"A":3,"B":1}
    tldr_similarity_score = models.FloatField(null=True, blank=True)
    effective_agreement_score = models.FloatField(null=True, blank=True)

    # Round 2 handoff summary (RoundSummary1)
    handoff_tldr_json = models.JSONField(default=dict, blank=True)  # {common_ground:[],...}

    # ops
    latency_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        unique_together = [("run", "round_index")]
        ordering = ["round_index"]
        constraints = [
        models.UniqueConstraint(fields=["run", "round_index"], name="uniq_run_roundindex"),
    ]

    def __str__(self):
        return f"Round {self.round_index} ({self.run.session_id})"


# --------- Each juror response per round ---------
class JurorResponse(models.Model):
    round = models.ForeignKey(DJNRound, on_delete=models.CASCADE, related_name="juror_responses")

    juror_id = models.CharField(max_length=4)  # J1..J4
    role = models.CharField(max_length=16, blank=True, default="")  # PROPOSER|CRITIC|REFINER|RISK

    # which model actually answered
    model = models.ForeignKey(LLMPool, on_delete=models.SET_NULL, null=True, blank=True)
    model_id_snapshot = models.CharField(max_length=160, blank=True, default="")  # if model row missing later

    # strict JSON output fields
    verdict_label = models.CharField(max_length=80, blank=True, default="")
    tldr = models.TextField(blank=True, default="")
    reasoning_json = models.JSONField(default=list, blank=True)

    # validation + failure rules (v1)
    status = models.CharField(max_length=16, default="OK")  # OK|FAILED|TIMEOUT|RETRY_EXHAUSTED|PARSE_FAIL
    schema_valid = models.BooleanField(default=True)
    error_msg = models.TextField(blank=True, default="")

    # ops metrics (per model call)
    latency_ms = models.IntegerField(null=True, blank=True)
    token_in = models.IntegerField(null=True, blank=True)
    token_out = models.IntegerField(null=True, blank=True)
    cost_estimate = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [("round", "juror_id")]
        ordering = ["juror_id"]
        constraints = [
            models.UniqueConstraint(fields=["round", "juror_id"], name="uniq_round_juror"),
        ]


    def __str__(self):
        return f"{self.juror_id} {self.verdict_label} ({self.round})"


# --------- Rolling stats (per model, per category) ---------
class ModelRollingStat(models.Model):
    model = models.ForeignKey(LLMPool, on_delete=models.CASCADE, related_name="rolling_stats")
    category = models.CharField(max_length=30, default="general")

    appearances_total = models.IntegerField(default=0)
    user_accepts_total = models.IntegerField(default=0)

    # derived convenience fields (can be updated by code)
    user_acceptance_rate = models.FloatField(default=0.0)
    win_rate_in_majority = models.FloatField(default=0.0)
    disagreement_rate = models.FloatField(default=0.0)
    avg_latency_ms = models.FloatField(default=0.0)
    schema_valid_rate = models.FloatField(default=0.0)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("model", "category")]
        ordering = ["model__name", "category"]

    def __str__(self):
        return f"{self.model.name} [{self.category}]"
