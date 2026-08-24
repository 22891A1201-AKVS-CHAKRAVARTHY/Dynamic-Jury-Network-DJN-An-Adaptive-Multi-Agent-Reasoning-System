from django.contrib import admin
from django import forms
from djn_db.selector import SUPPORTED_CATEGORIES
from .models import (
    BenchmarkTask,
    DJNRun,
    DJNRound,
    EvaluationResult,
    JurorResponse,
    LLMPool,
    ModelRollingStat,
    RunFeedback,
)


class LLMPoolForm(forms.ModelForm):
    class Meta:
        model = LLMPool
        fields = "__all__"

    def clean_capabilities_json(self):
        capabilities = self.cleaned_data.get("capabilities_json") or {}
        if not isinstance(capabilities, dict):
            raise forms.ValidationError("Capabilities must be a category-to-weight object.")
        unknown = sorted(set(capabilities) - SUPPORTED_CATEGORIES)
        if unknown:
            raise forms.ValidationError(f"Unsupported categories: {', '.join(unknown)}")
        for category, weight in capabilities.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise forms.ValidationError(f"{category} must have a numeric weight from 0 to 1.")
        return capabilities


@admin.register(LLMPool)
class LLMPoolAdmin(admin.ModelAdmin):
    form = LLMPoolForm
    list_display = ("name", "provider", "model_id", "enabled")
    list_filter = ("enabled", "provider")
    search_fields = ("name", "model_id")
    readonly_fields = ("created_at", "updated_at")


@admin.register(DJNRun)
class DJNRunAdmin(admin.ModelAdmin):
    list_display = ("session_id", "created_at", "category", "final_confidence", "stop_reason", "user_feedback")
    list_filter = ("category", "final_confidence", "stop_reason")
    search_fields = ("session_id", "q_raw", "q_final")


@admin.register(DJNRound)
class DJNRoundAdmin(admin.ModelAdmin):
    list_display = ("run", "round_index", "agreement", "majority_label", "improvement", "stagnation_flag", "latency_ms")
    list_filter = ("round_index", "stagnation_flag")


@admin.register(JurorResponse)
class JurorResponseAdmin(admin.ModelAdmin):
    list_display = ("round", "juror_id", "role", "verdict_label", "status", "schema_valid", "latency_ms")
    list_filter = ("status", "schema_valid", "role")
    search_fields = ("verdict_label", "tldr", "model_id_snapshot")


@admin.register(ModelRollingStat)
class ModelRollingStatAdmin(admin.ModelAdmin):
    list_display = ("model", "category", "appearances_total", "user_acceptance_rate", "avg_latency_ms", "schema_valid_rate")
    list_filter = ("category",)
    readonly_fields = [field.name for field in ModelRollingStat._meta.fields]


@admin.register(RunFeedback)
class RunFeedbackAdmin(admin.ModelAdmin):
    list_display = ("run", "voter_session", "value", "updated_at")
    list_filter = ("value",)


admin.site.register(BenchmarkTask)
admin.site.register(EvaluationResult)
