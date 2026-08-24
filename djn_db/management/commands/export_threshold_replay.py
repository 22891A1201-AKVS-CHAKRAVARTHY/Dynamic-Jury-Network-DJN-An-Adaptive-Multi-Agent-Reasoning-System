import json
from pathlib import Path

from django.core.management.base import BaseCommand

from djn_db.models import DJNRun
from evaluation.replay_stopping import replay_thresholds


class Command(BaseCommand):
    help = "Export aggregate retrospective stopping-threshold replay over stored rounds."

    def add_arguments(self, parser):
        parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.75,0.8,0.9")
        parser.add_argument("--output", default="outputs/threshold_replay.json")

    def handle(self, *args, **options):
        thresholds = [float(value) for value in options["thresholds"].split(",")]
        runs = list(DJNRun.objects.prefetch_related("rounds").all())
        rows = {threshold: [] for threshold in thresholds}
        for run in runs:
            for result in replay_thresholds(run.rounds.all(), thresholds):
                rows[result["threshold"]].append(result)
        summary = []
        for threshold, values in rows.items():
            observed_stops = [item["stop_round"] for item in values if item["stop_round"] is not None]
            summary.append({
                "threshold": threshold,
                "run_count": len(values),
                "observed_stop_count": len(observed_stops),
                "observed_stop_rate": len(observed_stops) / len(values) if values else 0.0,
                "average_stop_round_when_observed": sum(observed_stops) / len(observed_stops) if observed_stops else None,
                "censored_count": sum(1 for item in values if item["censored"]),
            })
        report = {
            "analysis_type": "retrospective operational replay; not accuracy validation",
            "summary": summary,
        }
        target = Path(options["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(str(target)))
