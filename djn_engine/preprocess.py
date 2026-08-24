from __future__ import annotations

from typing import Any, Dict, Iterable

from .audit import invoke_with_telemetry
from .json_enforce import parse_with_repair
from .llms import build_llm
from .pool import JUDGE
from .run import ASSUMPTIONS_PROMPT, MODERATOR_PROMPT
from .schemas import AssumptionsOut, ModeratorOut

MODERATOR_PROMPT_VERSION = "moderator-prompt-v1"
ASSUMPTIONS_PROMPT_VERSION = "assumptions-prompt-v1"


def _invoke_structured(
    prompt, payload: Dict[str, Any], schema, call_type: str, prompt_version: str,
) -> Dict[str, Any]:
    """Invoke one preprocessing call and return parsing plus provider-neutral telemetry."""
    try:
        message, telemetry = invoke_with_telemetry(
            (prompt | build_llm(JUDGE)).invoke,
            payload,
            call_type=call_type,
            provider=JUDGE.provider,
            model_id=JUDGE.model,
        )
        telemetry["prompt_version"] = prompt_version
    except Exception as exc:
        return {
            "ok": False,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": {
                "call_type": call_type,
                "provider": JUDGE.provider,
                "model_id": JUDGE.model,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_version": prompt_version,
            },
            "prompt_version": prompt_version,
        }
    raw = message if isinstance(message, str) else str(getattr(message, "content", "") or "")
    try:
        output = parse_with_repair(schema, raw)
        return {
            "ok": True, "output": output, "raw": raw, "error": "",
            "telemetry": telemetry, "prompt_version": prompt_version,
        }
    except Exception as exc:
        return {
            "ok": False, "output": None, "raw": raw,
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": telemetry, "prompt_version": prompt_version,
        }


def moderator_check(query: str) -> Dict[str, Any]:
    return _invoke_structured(
        MODERATOR_PROMPT,
        {"query": (query or "").strip()},
        ModeratorOut,
        "moderator",
        MODERATOR_PROMPT_VERSION,
    )


def build_assumptions(q_raw: str, clarifier_answers: Iterable[str]) -> Dict[str, Any]:
    answers = [str(item).strip() for item in clarifier_answers if str(item).strip()]
    return _invoke_structured(
        ASSUMPTIONS_PROMPT,
        {
            "q_raw": (q_raw or "").strip(),
            "clarifier_answers": "\n- " + "\n- ".join(answers) if answers else "(none)",
        },
        AssumptionsOut,
        "assumption_builder",
        ASSUMPTIONS_PROMPT_VERSION,
    )
