# Model Pool Updates — 25 August 2026

## Summary

This update corrects the NVIDIA Nemotron model identifier and replaces the
closed-source GLM Ollama Cloud juror with NVIDIA-hosted
`meta/muse-glimmer-30b`.

## Changes and rationale

### Corrected the Nemotron model identifier

The NIM juror previously used the incomplete identifier
`nvidia/nemotron-3-super`. It now uses the NVIDIA model identifier
`nvidia/nemotron-3-super-120b-a12b`.

The capability registry and changelog were updated to use the same identifier.
This prevents the configured pool and model metadata from referring to
different models.

### Replaced GLM 5.2

The Ollama Cloud juror `glm-5.2:cloud` was removed because it did not meet the
project's open-source model requirement. It was replaced by
`meta/muse-glimmer-30b`, accessed through NVIDIA's NIM API.

A live request was sent to NVIDIA's OpenAI-compatible chat-completions
endpoint using the configured `META_API` credential. The endpoint returned the
requested model, an `OK` response, and `finish_reason=stop`, confirming that the
credential and model endpoint were operational at the time of testing.

### Added per-model NVIDIA credentials

The existing Nemotron juror continues to use `NVIDIA_API_KEY`, while Muse
Glimmer uses `META_API`. `LLMConfig` now supports an optional per-model
`api_key_env` setting so multiple NIM jurors can select their intended
credential without storing secrets in source code.

This setting is propagated through fixed selection, dynamic selection,
fallback selection, legacy execution, health checks, and experiment dry-run
validation. `.env.example` documents the variable name but contains no key.

### Synchronized removed models safely

`check_llm_pool` reads enabled models from the database rather than directly
from `pool.py`. Previously, `seed_llmpool` added current models but left removed
seeded models enabled. Consequently, obsolete model IDs continued to appear in
pool checks and remained eligible for selection.

The seeder now disables stale rows that it previously created. It does not
delete them, so historical runs and foreign-key references remain intact.
Custom database rows not created by the seeder are left unchanged.

## Files affected

| File | Purpose of change |
|---|---|
| `.env.example` | Documents the `META_API` variable. |
| `CHANGESLOG.md` | Records the corrected Nemotron ID and GLM replacement. |
| `config/model_capabilities.json` | Replaces obsolete model IDs and registers Muse Glimmer capabilities. |
| `djn_engine/pool.py` | Configures the corrected Nemotron model and Muse Glimmer juror. |
| `djn_engine/llms.py` | Adds per-model NIM credential selection. |
| `djn_engine/orchestration.py` | Preserves credential configuration during jury selection. |
| `djn_engine/run.py` | Preserves credential configuration in the legacy execution path. |
| `djn_engine/experiment_runner.py` | Validates all credentials required by selected models. |
| `djn_db/management/commands/check_llm_pool.py` | Uses model-specific credentials during dry and live health checks. |
| `djn_db/management/commands/seed_llmpool.py` | Disables obsolete seeded models without deleting history. |
| `djn_db/tests.py` | Adds regression coverage for stale-model disabling and current model seeding. |

## Deployment and verification

After pulling or deploying these changes, synchronize the database:

```powershell
python manage.py seed_llmpool
python manage.py check_llm_pool
```

The dry check should include these enabled NIM models:

```text
meta/muse-glimmer-30b [nim] CONFIGURED
nvidia/nemotron-3-super-120b-a12b [nim] CONFIGURED
```

The old `glm-5.2:cloud` and `nvidia/nemotron-3-super` rows may remain in the
database for historical integrity, but they should be disabled and therefore
absent from `check_llm_pool` output.

## Verification status

- The NVIDIA Muse Glimmer API endpoint was tested successfully with a minimal
  live request.
- `config/model_capabilities.json` was parsed successfully.
- `git diff --check` completed without whitespace errors.
- The Django test suite was not run in this workspace because its virtual
  environment points to a Python installation that is no longer present. Run
  the test suite in a working Python environment before merging or releasing.

