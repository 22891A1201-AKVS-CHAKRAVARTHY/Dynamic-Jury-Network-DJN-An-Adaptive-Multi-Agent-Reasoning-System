import json

from django.core.management.base import BaseCommand, CommandError

from djn_db.models import DJNRun
from evaluation.replay_stopping import replay_thresholds


class Command(BaseCommand):
    help = "Replay alternative stopping thresholds over observed rounds only."

    def add_arguments(self, parser):
        parser.add_argument("run_id")
        parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.75,0.8,0.9")

    def handle(self, *args, **options):
        try:
            run = DJNRun.objects.get(session_id=options["run_id"])
        except DJNRun.DoesNotExist as exc:
            raise CommandError("Run not found") from exc
        thresholds = [float(value) for value in options["thresholds"].split(",")]
        self.stdout.write(json.dumps(replay_thresholds(run.rounds.all(), thresholds), indent=2))
