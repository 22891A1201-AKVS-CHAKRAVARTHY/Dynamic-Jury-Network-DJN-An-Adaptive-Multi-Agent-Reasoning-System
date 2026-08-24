import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from djn_db.db_writer import persist_engine_result
from djn_db.models import BenchmarkTask, EvaluationResult
from djn_engine.experiment_runner import build_dry_run, load_config
from djn_engine.orchestration import run_djn_once
from evaluation.scorers import score_answer
from evaluation.tasks import load_tasks


class Command(BaseCommand):
    help = "Validate or execute a reproducible DJN benchmark evaluation."

    def add_arguments(self, parser):
        parser.add_argument("--tasks", required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--max-calls", type=int, default=100)
        parser.add_argument("--execute", action="store_true")
        parser.add_argument("--approve-live", action="store_true")
        parser.add_argument("--manifest", default="outputs/evaluation_manifest.json")

    def handle(self, *args, **options):
        tasks = load_tasks(options["tasks"])
        config = load_config(options["config"])
        try:
            manifest = build_dry_run(tasks, config, options["max_calls"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        if not options["execute"]:
            self.stdout.write(json.dumps(manifest, indent=2))
            return
        if not options["approve_live"]:
            raise CommandError("Live execution requires --approve-live after reviewing the dry run.")

        completed = set(EvaluationResult.objects.filter(
            experiment_config_id=config.config_id,
        ).values_list("task__task_id", flat=True))
        failures = []
        for task in tasks:
            if task.task_id in completed:
                continue
            try:
                result = run_djn_once(task.prompt, task.category, config)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "DJN run failed"))
                run = persist_engine_result(
                    result, q_raw=task.prompt, q_final=task.prompt, category=task.category,
                    extra={"benchmark_task_id": task.task_id, "scorer_version": task.scorer_version},
                )
                task_row, _ = BenchmarkTask.objects.update_or_create(
                    task_id=task.task_id,
                    defaults={
                        "dataset": task.dataset, "split": task.split, "category": task.category,
                        "prompt": task.prompt, "reference_answer_json": task.reference_answer,
                        "scorer": task.scorer, "scorer_version": task.scorer_version,
                        "source_url": task.source_url, "license": task.license,
                        "metadata_json": task.metadata,
                    },
                )
                scored = score_answer(result.get("final", ""), task.reference_answer, task.scorer, task.metadata)
                EvaluationResult.objects.update_or_create(
                    task=task_row, experiment_config_id=config.config_id,
                    defaults={
                        "run": run, "scorer_version": task.scorer_version,
                        "score": scored["score"], "correct": scored["correct"],
                        "details_json": scored["details"],
                    },
                )
            except Exception as exc:
                failures.append({"task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"})
                break
        manifest.update({"dry_run": False, "failures": failures})
        target = Path(options["manifest"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Evaluation manifest: {target}"))
