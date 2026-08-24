from django.core.management.base import BaseCommand
from django.utils import timezone
import json
from pathlib import Path

from djn_db.models import LLMPool

from djn_engine.pool import JURORS


CAPABILITY_PATH = Path(__file__).resolve().parents[3] / "config" / "model_capabilities.json"


def _load_capabilities():
    with CAPABILITY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Command(BaseCommand):
    help = "Seed/refresh LLMPool from djn_engine.pool.JURORS (idempotent upsert)."

    def handle(self, *args, **options):
        registry = _load_capabilities()
        capability_version = registry.get("version", "capabilities-v1")
        registered_models = registry.get("models") or {}
        upserts = 0
        for cfg in JURORS:
            model_id = getattr(cfg, "model", "")
            if not model_id:
                continue

            name = getattr(cfg, "name", model_id)
            provider = getattr(cfg, "provider", "") or ""
            capabilities = registered_models.get(model_id, {"general": 0.5})
            tags = sorted(capabilities)

            row, created = LLMPool.objects.get_or_create(
                model_id=model_id,
                defaults={
                    "name": name[:120],
                    "provider": provider[:60],
                    "enabled": True,
                    "tags_json": tags,
                    "category_weights_json": {},
                    "capabilities_json": capabilities,
                    "capability_version": capability_version,
                    "cost_tier": "",
                    "notes": "Seeded by seed_llmpool command.",
                    "created_at": timezone.now(),
                }
            )

            if not created:
                row.name = name[:120]
                row.provider = provider[:60]
                row.tags_json = tags
                row.capabilities_json = capabilities
                row.capability_version = capability_version
                row.updated_at = timezone.now()
                row.save(update_fields=[
                    "name", "provider", "tags_json", "capabilities_json",
                    "capability_version", "updated_at",
                ])

            upserts += 1

        self.stdout.write(self.style.SUCCESS(f"LLMPool seeded/refreshed: {upserts} models"))
        self.stdout.write("Run: python manage.py seed_llmpool")
