from __future__ import annotations

import math
import re
from typing import Any, Dict

SCORER_VERSION = "scorers-v1"


def _normalize(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9.+-]+", " ", str(value).lower()).split())


def score_answer(answer: str, reference: Any, scorer: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = metadata or {}
    if scorer == "exact_match":
        correct = str(answer) == str(reference)
    elif scorer == "normalized_match":
        correct = _normalize(answer) == _normalize(reference)
    elif scorer == "numeric_tolerance":
        tolerance = float(metadata.get("tolerance", 1e-6))
        try:
            observed = float(re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer).group())
            correct = math.isclose(observed, float(reference), abs_tol=tolerance, rel_tol=tolerance)
        except (AttributeError, TypeError, ValueError):
            correct = False
    elif scorer == "unit_test":
        # Safe structured checks only; this deliberately does not execute untrusted code.
        checks = reference.get("required_substrings", []) if isinstance(reference, dict) else []
        correct = bool(checks) and all(str(item) in answer for item in checks)
    elif scorer in {"rubric", "manual"}:
        return {"score": None, "correct": None, "details": {"reason": "MANUAL_REVIEW_REQUIRED"}}
    else:
        raise ValueError(f"Unknown scorer: {scorer}")
    return {"score": 1.0 if correct else 0.0, "correct": correct, "details": {"scorer_version": SCORER_VERSION}}
