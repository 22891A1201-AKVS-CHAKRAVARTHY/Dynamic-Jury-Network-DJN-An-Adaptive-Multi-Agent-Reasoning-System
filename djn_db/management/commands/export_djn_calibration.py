import json
from pathlib import Path

from django.core.management.base import BaseCommand

from djn_db.models import EvaluationResult
from evaluation.calibration import calibration_metrics


class Command(BaseCommand):
    help = "Export calibration metrics from labeled benchmark results; never calls an LLM."

    def add_arguments(self, parser):
        parser.add_argument("--output", default="outputs/calibration.json")

    def handle(self, *args, **options):
        mapping = {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75}
        raw_rows = []
        consensus_rows = []
        for result in EvaluationResult.objects.select_related("run").exclude(correct=None):
            correct = 1 if result.correct else 0
            raw = mapping.get((result.run.raw_judge_confidence or "").upper())
            consensus = mapping.get((result.run.consensus_confidence or "").upper())
            if raw is not None:
                raw_rows.append((raw, correct))
            if consensus is not None:
                consensus_rows.append((consensus, correct))
        report = {
            "confidence_mapping_version": "categorical-confidence-map-v1",
            "warning": "Category-to-number mapping is an analysis convention, not a learned correctness probability.",
            "raw_judge_confidence": calibration_metrics(raw_rows),
            "capped_consensus_confidence": calibration_metrics(consensus_rows),
        }
        target = Path(options["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(str(target)))
