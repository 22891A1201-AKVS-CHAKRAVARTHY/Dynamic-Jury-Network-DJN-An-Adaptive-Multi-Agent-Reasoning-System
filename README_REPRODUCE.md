# DJN Reproducibility Guide

This release separates code validation and offline analysis from paid model inference.
Do not make paper claims from a run until data-quality checks pass.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py seed_llmpool
```

Populate provider keys only in `.env`; never commit that file. Populate verified
prices in `config/model_pricing.json` before reporting monetary cost.

## No-provider validation

```powershell
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test djn_db -v 2
python manage.py rebuild_llm_stats
python manage.py check_djn_data
python manage.py check_llm_pool
python manage.py run_mock_load_test --prompts 100 --concurrency 4 --latency-ms 50
```

## Evaluation dry run

```powershell
python manage.py run_djn_evaluation `
  --tasks evaluation/datasets/example_tasks.json `
  --config config/experiment.example.json `
  --max-calls 100
```

The dry run contacts no model provider. Replace the synthetic task fixture with a
licensed, frozen benchmark before evaluation. Actual calls require both `--execute`
and `--approve-live`.

## Paper tables

```powershell
python manage.py check_djn_data
python manage.py export_paper_metrics --output outputs/paper_metrics
python manage.py export_threshold_replay --output outputs/threshold_replay.json
python manage.py export_djn_calibration --output outputs/calibration.json
```

The export records the commit SHA, database SHA-256, query scope, metric-definition
version, denominators, and quality report. Keep four-juror and other configurations
separate when interpreting the exported configuration groups.

## Privacy review

Audit JSON downloads and JSONL logs apply automatic secret/email/phone redaction.
Before publishing any database or export, manually inspect it for personal or
licensed content. Never release `.env`, `credentials.json`, the working SQLite
database, provider headers, or raw user prompts without explicit permission.

```powershell
python manage.py export_sanitized_runs --output outputs/sanitized_runs.jsonl
```

Complete `docs/PRIVACY_REVIEW.md` before distribution.
