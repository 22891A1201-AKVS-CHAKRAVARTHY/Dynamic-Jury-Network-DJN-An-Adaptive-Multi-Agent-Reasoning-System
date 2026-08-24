from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def calibration_metrics(rows: Iterable[Tuple[float, int]], bins: int = 10) -> Dict[str, object]:
    values = [(max(0.0, min(1.0, float(conf))), int(correct)) for conf, correct in rows]
    if not values:
        return {"count": 0, "brier_score": None, "ece": None, "bins": []}
    brier = sum((confidence - correct) ** 2 for confidence, correct in values) / len(values)
    output_bins: List[Dict[str, float]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [(c, y) for c, y in values if lower <= c < upper or (index == bins - 1 and c == 1.0)]
        if not members:
            continue
        avg_conf = sum(c for c, _ in members) / len(members)
        accuracy = sum(y for _, y in members) / len(members)
        ece += len(members) / len(values) * abs(avg_conf - accuracy)
        output_bins.append({"lower": lower, "upper": upper, "count": len(members), "confidence": avg_conf, "accuracy": accuracy})
    selective = []
    for threshold in (0.0, 0.5, 0.75, 0.9):
        retained = [(confidence, correct) for confidence, correct in values if confidence >= threshold]
        selective.append({
            "threshold": threshold,
            "coverage": len(retained) / len(values),
            "accuracy": (sum(correct for _, correct in retained) / len(retained)) if retained else None,
            "count": len(retained),
        })
    return {
        "count": len(values), "brier_score": brier, "ece": ece,
        "bins": output_bins, "selective_accuracy": selective,
    }
