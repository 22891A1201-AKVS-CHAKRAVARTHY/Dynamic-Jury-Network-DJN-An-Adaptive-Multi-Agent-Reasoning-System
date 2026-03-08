from __future__ import annotations
from typing import Dict, Any, List
from django.db import transaction
from django.db.models import Avg

from .models import DJNRun, DJNRound, JurorResponse, ModelRollingStat, LLMPool


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@transaction.atomic
def update_stats_for_run(run_id: str) -> None:
    """
    Updates ModelRollingStat rows for the models that appeared in a run.
    Uses:
      - appearances_total
      - schema_valid_rate
      - avg_latency_ms
      - disagreement_rate (vs majority label in that round)
      - win_rate_in_majority
      - user_acceptance_rate (based on DJNRun.user_feedback)
    """
    try:
        run = DJNRun.objects.get(session_id=run_id)
    except DJNRun.DoesNotExist:
        return

    rounds = DJNRound.objects.filter(run=run).prefetch_related("juror_responses")
    if not rounds.exists():
        return

    category = run.category or "general"
    fb = run.user_feedback  # None | 1 | -1

    # collect all juror responses in this run
    responses = JurorResponse.objects.filter(round__run=run)

    # group by model_id_snapshot (even if LLMPool row missing)
    by_mid: Dict[str, List[JurorResponse]] = {}
    for jr in responses:
        mid = (jr.model_id_snapshot or "").strip()
        if not mid:
            continue
        by_mid.setdefault(mid, []).append(jr)

    for mid, items in by_mid.items():
        model_row = None
        try:
            model_row = LLMPool.objects.get(model_id=mid)
        except LLMPool.DoesNotExist:
            # If you didn't seed pool yet, skip creating stats
            continue

        stat, _ = ModelRollingStat.objects.get_or_create(model=model_row, category=category)

        # appearances: count unique (round, juror_id) rows
        appearances = len(items)
        stat.appearances_total += appearances

        # schema valid
        schema_ok = sum(1 for x in items if x.schema_valid)
        stat.schema_valid_rate = _safe_rate(schema_ok, appearances)

        # avg latency (ignore nulls)
        lat_vals = [x.latency_ms for x in items if x.latency_ms is not None]
        stat.avg_latency_ms = (sum(lat_vals) / len(lat_vals)) if lat_vals else stat.avg_latency_ms

        # disagreement / win rate: compare juror verdict to majority_label per round
        wins = 0
        disagrees = 0
        for x in items:
            maj = x.round.majority_label
            if not maj or not x.verdict_label:
                continue
            if x.verdict_label == maj:
                wins += 1
            else:
                disagrees += 1

        judged = wins + disagrees
        stat.win_rate_in_majority = _safe_rate(wins, judged)
        stat.disagreement_rate = _safe_rate(disagrees, judged)

        # user acceptance: only update totals if feedback exists
        if fb in (1, -1):
            # count "accepts" as run-level positive feedback for any model that appeared
            stat.user_accepts_total += 1 if fb == 1 else 0
            stat.user_acceptance_rate = _safe_rate(stat.user_accepts_total, stat.appearances_total > 0 and stat.user_accepts_total + (stat.appearances_total - stat.user_accepts_total) or 1)

        stat.save()
