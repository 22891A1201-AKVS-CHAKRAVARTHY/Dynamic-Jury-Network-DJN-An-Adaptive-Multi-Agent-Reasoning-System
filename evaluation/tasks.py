from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=120)
    dataset: str = Field(min_length=1, max_length=120)
    split: str = "test"
    category: str = "general"
    prompt: str = Field(min_length=1)
    reference_answer: Any
    scorer: str
    scorer_version: str = "scorers-v1"
    source_url: str = ""
    license: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


def load_tasks(path: str | Path) -> List[TaskSpec]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    tasks = [TaskSpec.model_validate(row) for row in rows]
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark task IDs detected")
    for task in tasks:
        reference_text = json.dumps(task.reference_answer, sort_keys=True).strip('"')
        if reference_text and reference_text in task.prompt:
            raise ValueError(f"Potential reference-answer leakage in task {task.task_id}")
    return tasks
