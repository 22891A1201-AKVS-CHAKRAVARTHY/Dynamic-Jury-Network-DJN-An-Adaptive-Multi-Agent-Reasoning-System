import json
from pathlib import Path

from django.core.management.base import BaseCommand

from djn_db.audit_export import build_run_audit
from djn_db.models import DJNRun
from djn_engine.privacy import sanitize


class Command(BaseCommand):
    help = "Export redacted audit JSONL without modifying the source database."

    def add_arguments(self, parser):
        parser.add_argument("--output", default="outputs/sanitized_runs.jsonl")

    def handle(self, *args, **options):
        target = Path(options["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for run in DJNRun.objects.prefetch_related("rounds__juror_responses").all():
                handle.write(json.dumps(sanitize(build_run_audit(run)), ensure_ascii=False, default=str) + "\n")
        self.stdout.write(self.style.SUCCESS(f"Redacted export written to {target}; manual review is still required."))
