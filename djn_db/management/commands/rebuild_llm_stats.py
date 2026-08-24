from django.core.management.base import BaseCommand

from djn_db.stats import rebuild_all_stats


class Command(BaseCommand):
    help = "Rebuild all model rolling statistics from persisted source rows."

    def handle(self, *args, **options):
        count = rebuild_all_stats()
        self.stdout.write(self.style.SUCCESS(f"Rebuilt {count} model/category statistics."))
