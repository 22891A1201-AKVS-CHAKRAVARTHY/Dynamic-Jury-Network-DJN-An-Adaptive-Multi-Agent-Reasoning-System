from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.db.models import Avg, Count, Sum

from djn_db.models import DJNRun, DJNRound, JurorResponse
from .data_quality import check_all_runs

METRIC_DEFINITION_VERSION = "paper-metrics-v1"


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=settings.BASE_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def _database_hash() -> str:
    path = Path(settings.DATABASES["default"]["NAME"])
    if not path.exists():
        return "UNAVAILABLE"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_metrics() -> Dict[str, Any]:
    runs = DJNRun.objects.all()
    responses = JurorResponse.objects.all()
    rounds = DJNRound.objects.all()
    response_count = responses.count()
    valid_count = responses.filter(schema_valid=True).count()
    jury_sizes: Dict[str, int] = {}
    experiment_modes: Dict[str, int] = {}
    for run in runs.only("jury_roster_json", "experiment_config_json"):
        size = str(len(run.jury_roster_json or []))
        mode = str((run.experiment_config_json or {}).get("mode", "UNSPECIFIED"))
        jury_sizes[size] = jury_sizes.get(size, 0) + 1
        experiment_modes[mode] = experiment_modes.get(mode, 0) + 1
    return {
        "run_count": runs.count(),
        "round_count": rounds.count(),
        "juror_call_count": response_count,
        "schema_valid_count": valid_count,
        "schema_valid_rate": valid_count / response_count if response_count else 0.0,
        "schema_valid_denominator": response_count,
        "average_agreement": rounds.aggregate(value=Avg("agreement"))["value"],
        "avg_run_duration_ms": runs.aggregate(value=Avg("duration_ms"))["value"],
        "avg_juror_latency_ms": responses.aggregate(value=Avg("latency_ms"))["value"],
        "prompt_tokens": responses.aggregate(value=Sum("token_in"))["value"],
        "completion_tokens": responses.aggregate(value=Sum("token_out"))["value"],
        "estimated_cost": responses.aggregate(value=Sum("cost_estimate"))["value"],
        "cost_available_count": responses.exclude(cost_estimate=None).count(),
        "cost_denominator": response_count,
        "usage_unavailable_reasons": list(
            responses.exclude(usage_unavailable_reason="")
            .values("usage_unavailable_reason").annotate(count=Count("id"))
            .order_by("usage_unavailable_reason")
        ),
        "stop_reasons": list(runs.values("stop_reason").annotate(count=Count("id")).order_by("stop_reason")),
        "jury_sizes": [{"jury_size": int(size), "count": count} for size, count in sorted(jury_sizes.items())],
        "experiment_modes": [{"mode": mode, "count": count} for mode, count in sorted(experiment_modes.items())],
    }


def export_metrics(output_directory: str | Path, fail_on_quality: bool = True) -> Dict[str, Path]:
    quality = check_all_runs()
    if fail_on_quality and quality["critical_count"]:
        raise ValueError(f"Critical data-quality failures: {quality['critical_count']}")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()
    metadata = {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "commit_sha": _commit_sha(),
        "database_sha256": _database_hash(),
        "query": "all DJNRun and JurorResponse rows",
        "quality": quality,
    }
    json_path = directory / "paper_metrics.json"
    json_path.write_text(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2, default=str), encoding="utf-8")
    csv_path = directory / "paper_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, json.dumps(value, default=str) if isinstance(value, (list, dict)) else value])
    markdown_path = directory / "paper_metrics.md"
    rows = ["| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        rows.append(f"| {key} | {json.dumps(value, default=str)} |")
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
