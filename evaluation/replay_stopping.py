from __future__ import annotations

from typing import Any, Dict, Iterable, List


def replay_thresholds(rounds: Iterable[Any], thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    observed = list(rounds)
    results = []
    for threshold in thresholds:
        stop_round = None
        for round_row in observed:
            agreement = getattr(round_row, "agreement", None)
            if agreement is not None and agreement >= threshold:
                stop_round = getattr(round_row, "round_index", None)
                break
        original_stopped_early = bool(observed) and getattr(observed[-1], "stop_reason", "") in {
            "THRESHOLD_MET", "STAGNATION",
        }
        censored = stop_round is None and original_stopped_early
        results.append({
            "threshold": threshold,
            "stop_round": stop_round,
            "would_stop": stop_round is not None,
            "censored": censored,
            "observed_rounds": len(observed),
            "interpretation": "retrospective operational replay; not accuracy validation",
        })
    return results
