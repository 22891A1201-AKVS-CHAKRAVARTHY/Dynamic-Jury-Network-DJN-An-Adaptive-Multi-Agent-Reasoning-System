from django.core.management.base import BaseCommand, CommandError

from evaluation.export_paper_metrics import export_metrics


class Command(BaseCommand):
    help = "Export reproducible DJN paper metrics to JSON, CSV, and Markdown."

    def add_arguments(self, parser):
        parser.add_argument("--output", default="outputs/paper_metrics")
        parser.add_argument("--allow-quality-errors", action="store_true")

    def handle(self, *args, **options):
        try:
            paths = export_metrics(options["output"], not options["allow_quality_errors"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        for kind, path in paths.items():
            self.stdout.write(f"{kind}: {path}")
