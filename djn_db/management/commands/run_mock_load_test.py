import json

from django.core.management.base import BaseCommand

from evaluation.load_test import run_mock_load_test


class Command(BaseCommand):
    help = "Measure DJN orchestration with deterministic mock-provider latency."

    def add_arguments(self, parser):
        parser.add_argument("--prompts", type=int, default=100)
        parser.add_argument("--concurrency", type=int, default=4)
        parser.add_argument("--latency-ms", type=int, default=50)
        parser.add_argument("--jury-size", type=int, default=4)
        parser.add_argument("--rounds", type=int, default=1)

    def handle(self, *args, **options):
        report = run_mock_load_test(
            options["prompts"], options["concurrency"], options["latency_ms"],
            options["jury_size"], options["rounds"],
        )
        self.stdout.write(json.dumps(report, indent=2))
