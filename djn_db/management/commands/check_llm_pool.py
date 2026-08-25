import os

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from djn_db.models import LLMPool
from djn_engine.audit import invoke_with_telemetry
from djn_engine.llms import LLMConfig, build_llm, credential_env_for_config
from djn_engine.pool import JURORS


class Command(BaseCommand):
    help = "Validate enabled model configuration; live checks require explicit approval."

    def add_arguments(self, parser):
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--approve-live", action="store_true")
        parser.add_argument("--max-calls", type=int, default=10)

    def handle(self, *args, **options):
        rows = list(LLMPool.objects.filter(enabled=True).order_by("model_id"))
        configured = {cfg.model: cfg for cfg in JURORS}
        if len(rows) > options["max_calls"]:
            raise CommandError(f"Enabled models ({len(rows)}) exceed --max-calls guard.")
        if options["live"] and not options["approve_live"]:
            raise CommandError("Live checks may consume tokens; add --approve-live after reviewing the roster.")
        report = []
        for row in rows:
            provider = (row.provider or "").lower()
            source = configured.get(row.model_id)
            config = LLMConfig(
                name=row.name,
                provider=provider,
                model=row.model_id,
                temperature=getattr(source, "temperature", 0.2),
                base_url=getattr(source, "base_url", None),
                api_key_env=getattr(source, "api_key_env", None),
            )
            required = credential_env_for_config(config)
            status = "CONFIGURED" if not required or os.getenv(required) else "MISSING_CREDENTIAL"
            error = ""
            if options["live"]:
                try:
                    model = build_llm(config)
                    invoke_with_telemetry(
                        model.invoke, "Reply with OK only.",
                        call_type="health_check", provider=provider, model_id=row.model_id,
                        max_retries=0,
                    )
                    status = "HEALTHY"
                except Exception as exc:
                    status, error = "UNHEALTHY", f"{type(exc).__name__}: {exc}"
                row.health_status = status
                row.health_checked_at = timezone.now()
                row.save(update_fields=["health_status", "health_checked_at"])
            report.append((row.model_id, provider, status, error))
        for model_id, provider, status, error in report:
            self.stdout.write(f"{model_id} [{provider}] {status}{': ' + error if error else ''}")
        if not options["live"]:
            self.stdout.write("Dry validation only; no provider was contacted and health state was not changed.")
