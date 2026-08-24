"""Audited web integration layered over the original presentation/OAuth helpers.

Keeping this adapter separate makes the revision easy to review and preserves the
legacy Google Docs workflow while routing DJN execution and persistence through
the versioned orchestration engine.
"""
from __future__ import annotations

import threading

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from djn_db.audit_export import build_run_audit
from djn_db.db_writer import persist_engine_result, upsert_run
from djn_db.models import DJNRun
from djn_db.stats import apply_feedback
from djn_engine.logger import log_run
from djn_engine.orchestration import run_djn_once
from djn_engine.preprocess import (
    build_assumptions as audited_build_assumptions,
    moderator_check as audited_moderator_check,
)
from djn_engine.privacy import sanitize

from . import views as legacy

_REQUEST_CALLS = threading.local()
_PERSIST_FLAG = "djn_audited_persist_ok"
_FAILED_STATE = "djn_failed_clarification_state"
_PRE_RUN_CALLS = "djn_pre_run_calls"


def _calls():
    if not hasattr(_REQUEST_CALLS, "items"):
        _REQUEST_CALLS.items = []
    return _REQUEST_CALLS.items


def _record_call(telemetry):
    if not telemetry:
        return
    _calls().append(telemetry)
    request = getattr(_REQUEST_CALLS, "request", None)
    if request is not None:
        request.session[_PRE_RUN_CALLS] = list(_calls())
        request.session.modified = True


def moderator_check(query):
    cache = getattr(_REQUEST_CALLS, "moderator_cache", {})
    key = (query or "").strip()
    if key in cache:
        return cache[key]
    result = audited_moderator_check(query)
    _record_call(result.get("telemetry"))
    cache[key] = result
    _REQUEST_CALLS.moderator_cache = cache
    return result


def build_assumptions(q_raw, clarifier_answers=None):
    result = audited_build_assumptions(q_raw, clarifier_answers or [])
    _record_call(result.get("telemetry"))
    return result


def _clarification_snapshot(request):
    return {
        legacy.STATE_KEY: request.session.get(legacy.STATE_KEY, "idle"),
        legacy.PENDING_QUERY_KEY: request.session.get(legacy.PENDING_QUERY_KEY, ""),
        legacy.CLARIFY_QS_KEY: request.session.get(legacy.CLARIFY_QS_KEY, []) or [],
        legacy.CLARIFY_A_KEY: request.session.get(legacy.CLARIFY_A_KEY, []) or [],
        legacy.FORCE_LOW_CONF_KEY: bool(request.session.get(legacy.FORCE_LOW_CONF_KEY, False)),
    }


def _run_and_persist(request, q_raw, q_final, cat, cat_conf, missing, assumptions):
    request.session[_PERSIST_FLAG] = False
    request.session[_FAILED_STATE] = _clarification_snapshot(request)
    result = run_djn_once(q_final, category=cat)
    if not result.get("ok"):
        legacy._push(request, "assistant", f"DJN run failed: {result.get('error', 'unknown error')}")
        return result

    legacy._push(request, "assistant", result.get("final_display", result.get("final", "")))
    request.session[legacy.LAST_RUN_ID_KEY] = result.get("run_id")
    request.session[legacy.LAST_FINAL_IDX_KEY] = len(legacy._get_chat(request))
    request.session.modified = True
    try:
        log_run({
            "q_raw": q_raw, "q_final": q_final, "ok": True,
            "final_display": result.get("final_display"), "judge": result.get("judge"),
            "jurors": result.get("jurors"), "jury_roster": result.get("jury_roster", []),
            "role_map": result.get("role_map", {}), "selection_trace": result.get("selection_trace", {}),
            "experiment_config": result.get("experiment_config", {}),
            "rounds": result.get("rounds", []), "run_stop": result.get("run_stop", {}),
            "run_metrics": result.get("run_metrics", {}),
        })
    except Exception:
        pass

    try:
        persist_engine_result(result, q_raw=q_raw, q_final=q_final, category=cat)
        upsert_run({
            "session_id": result["run_id"], "q_raw": q_raw, "q_final": q_final,
            "category": cat, "category_confidence": cat_conf,
            "missing_fields": missing, "assumptions": assumptions,
            "clarifier_used": bool(request.session.get(legacy.CLARIFY_QS_KEY, [])),
            "clarifier_questions": request.session.get(legacy.CLARIFY_QS_KEY, []) or [],
            "clarifier_answers": request.session.get(legacy.CLARIFY_A_KEY, []) or [],
            "clarifier_skipped": bool(request.session.get(legacy.FORCE_LOW_CONF_KEY, False)),
            "pre_run_calls": list(_calls()), "final": {},
        })
        request.session[_PERSIST_FLAG] = True
        request.session.pop(_FAILED_STATE, None)
        request.session.pop(_PRE_RUN_CALLS, None)
    except Exception as exc:
        legacy._push(request, "assistant", f"The answer was generated, but its audit record was not saved: {type(exc).__name__}.")
    finally:
        _REQUEST_CALLS.items = []
        request.session.modified = True
    return result


# The legacy discussion function resolves these names in its module globals.
legacy.run_djn_once = run_djn_once
legacy.moderator_check = moderator_check
legacy.build_assumptions = build_assumptions
legacy._run_and_persist = _run_and_persist


def jury_discussion(request):
    _REQUEST_CALLS.request = request
    _REQUEST_CALLS.items = list(request.session.get(_PRE_RUN_CALLS, []) or [])
    _REQUEST_CALLS.moderator_cache = {}
    response = legacy.jury_discussion(request)
    if not request.session.get(_PERSIST_FLAG, True):
        snapshot = request.session.get(_FAILED_STATE) or {}
        for key, value in snapshot.items():
            request.session[key] = value
        request.session.modified = True
    request.session.pop(_PERSIST_FLAG, None)
    request.session.pop(_FAILED_STATE, None)
    _REQUEST_CALLS.request = None
    _REQUEST_CALLS.items = []
    _REQUEST_CALLS.moderator_cache = {}
    return response


def history(request):
    rows = DJNRun.objects.prefetch_related("rounds__juror_responses").all()[:30]
    return render(request, "webapp/history.html", {"runs": [build_run_audit(row) for row in rows]})


def run_audit_json(request, run_id):
    try:
        run = DJNRun.objects.prefetch_related("rounds__juror_responses").get(session_id=run_id)
    except DJNRun.DoesNotExist:
        return JsonResponse({"error": "Run not found."}, status=404)
    response = JsonResponse(sanitize(build_run_audit(run)), json_dumps_params={"indent": 2})
    response["Content-Disposition"] = f'attachment; filename="djn-audit-{run_id}.json"'
    return response


@require_POST
def jury_feedback(request):
    run_id = request.POST.get("run_id")
    value = request.POST.get("value")
    if not run_id or value not in {"up", "down"}:
        return redirect("jury_discussion")
    try:
        voter = request.session.session_key
        if not voter:
            request.session.create()
            voter = request.session.session_key
        apply_feedback(run_id, voter, 1 if value == "up" else -1)
    except (DJNRun.DoesNotExist, ValueError):
        pass
    return redirect("jury_discussion")


# Presentation and Google Docs helpers remain unchanged.
home = legacy.home
about = legacy.about
jury_clear = legacy.jury_clear
gdocs_share = legacy.gdocs_share
gdocs_callback = legacy.gdocs_callback
