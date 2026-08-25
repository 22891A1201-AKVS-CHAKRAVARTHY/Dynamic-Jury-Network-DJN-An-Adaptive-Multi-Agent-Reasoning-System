from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .llms import credential_env_for_config
from .schemas import ExperimentConfig
from .pool import JUDGE, JURORS
from evaluation.tasks import TaskSpec, load_tasks


def projected_calls(task_count: int, config: ExperimentConfig) -> int:
    rounds = 1 if config.mode in {"single_model", "static_jury_one_round"} else config.max_rounds
    judge_calls = 0 if config.synthesis_mode == "majority" else rounds
    summary_calls = 0 if config.handoff_mode != "structured" else max(0, rounds - 1)
    return task_count * (config.jury_size * rounds + judge_calls + summary_calls)


def build_dry_run(tasks: List[TaskSpec], config: ExperimentConfig, max_calls: int) -> Dict[str, Any]:
    config.validate_combination()
    calls = projected_calls(len(tasks), config)
    if calls > max_calls:
        raise ValueError(f"Projected calls {calls} exceed maximum-call guard {max_calls}")
    candidates = [item for item in JURORS if not config.model_pool or item.model in config.model_pool]
    if config.selector_mode == "fixed":
        candidates = [item for item in candidates if item.model in config.fixed_roster]
        if len(candidates) != config.jury_size:
            known = {item.model for item in candidates}
            missing = [model_id for model_id in config.fixed_roster if model_id not in known]
            raise ValueError(f"Fixed roster contains unconfigured model IDs: {missing}")
    provider_configs = {provider: [] for provider in {item.provider for item in candidates}}
    for item in candidates:
        provider_configs[item.provider].append(item)
    if config.synthesis_mode == "judge" or config.handoff_mode == "structured":
        provider_configs.setdefault(JUDGE.provider, []).append(JUDGE)

    provider_validation = []
    for provider in sorted(provider_configs):
        credential_names = sorted({
            name
            for item in provider_configs[provider]
            if (name := credential_env_for_config(item))
        })
        provider_validation.append({
            "provider": provider,
            "credential_variable": ", ".join(credential_names) or None,
            "credential_present": all(bool(os.getenv(name)) for name in credential_names),
        })
    return {
        "dry_run": True,
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "experiment_config": config.snapshot(),
        "experiment_config_id": config.config_id,
        "maximum_projected_calls": calls,
        "models": [item.model for item in candidates],
        "provider_validation": provider_validation,
        "providers_contacted": False,
    }


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_evaluation(task_path: str | Path, config_path: str | Path, max_calls: int) -> Dict[str, Any]:
    return build_dry_run(load_tasks(task_path), load_config(config_path), max_calls)
