import json

from django.core.management.base import BaseCommand, CommandError

from evaluation.data_quality import check_all_runs


class Command(BaseCommand):
    help = "Run DJN database consistency and auditability checks."

    def handle(self, *args, **options):
        report = check_all_runs()
        self.stdout.write(json.dumps(report, indent=2))
        if report["critical_count"]:
            raise CommandError(f"Detected {report['critical_count']} critical data-quality issues.")
