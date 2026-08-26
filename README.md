# Dynamic Jury Network (DJN)

**An Adaptive Multi-Agent Reasoning System**

Dynamic Jury Network (DJN) is a Django-based research system for structured, auditable deliberation across heterogeneous large language models. It selects a jury for each query, assigns complementary reasoning roles, validates structured responses, measures agreement over one or more rounds, and asks a moderator/judge to synthesize the final recommendation.

DJN is designed to investigate whether heterogeneous multi-model deliberation can improve robustness and expose disagreement more clearly than a single response. The repository provides the implementation and controlled experiment infrastructure needed to test that question. It does not, by itself, establish that DJN is more accurate than every single-model or multi-agent baseline.

## Contents

- [Motivation](#motivation)
- [How DJN works](#how-djn-works)
- [Architecture](#architecture)
- [Dynamic juror selection](#dynamic-juror-selection)
- [Roles, rounds, and handoff](#roles-rounds-and-handoff)
- [Consensus, stopping, and final synthesis](#consensus-stopping-and-final-synthesis)
- [Auditing, feedback, and statistics](#auditing-feedback-and-statistics)
- [Experiments and evaluation](#experiments-and-evaluation)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running DJN](#running-djn)
- [Validation and tests](#validation-and-tests)
- [Reproducibility workflow](#reproducibility-workflow)
- [Privacy and security](#privacy-and-security)
- [Known limitations and research caveats](#known-limitations-and-research-caveats)

## Motivation

A single model can be brittle, overconfident, or limited by one training and inference profile. DJN turns answer generation into an explicit protocol:

1. classify and normalize the query;
2. choose models using category-aware capability and observed operational statistics;
3. assign distinct juror responsibilities;
4. collect schema-validated independent verdicts;
5. measure agreement and decide whether another round is useful;
6. pass a structured summary between rounds when enabled; and
7. synthesize and audit a final recommendation.

The design emphasizes traceability. Model selection, prompts, outputs, failures, timing, token usage, confidence handling, and experiment settings are persisted so that a run can be inspected rather than treated as an opaque answer.

## How DJN works

The audited web path is implemented in `webapp/audited_views.py`; the authoritative deliberation engine is `djn_engine/orchestration.py`.

```text
User query
   |
   v
Moderation, classification, clarification, and assumptions
   |
   v
Dynamic or fixed jury selection ---- selector/capability config + rolling statistics
   |
   v
Role-conditioned juror calls in parallel
   |
   v
Schema validation + verdict distribution + agreement
   |
   +---- threshold met or stagnation detected ----+
   |                                              |
   v                                              v
Structured handoff to next round          Moderator/judge synthesis
   |                                              |
   +---------------- repeat as configured --------+
                                                  |
                                                  v
                                  Persisted result, audit, and feedback
```

### Query processing lifecycle

1. **Intake and preprocessing.** The web flow records the original query. `djn_engine/preprocess.py` supports query categorization, up to three clarification questions, and explicit assumptions. A skipped clarification is recorded and forces the final confidence level to `LOW`.
2. **Experiment resolution.** An immutable `ExperimentConfig` defines selection, roles, handoff, stopping, synthesis, jury size, thresholds, concurrency, seed, and version identifiers. Its stable hash becomes the experiment configuration ID.
3. **Jury selection.** DJN uses either the adaptive selector or a fixed roster, depending on the experiment mode. The complete candidate score breakdown and selection decision are captured in a trace.
4. **Parallel juror calls.** Jurors receive the same user query, a role-specific instruction, and any round context. They must return `verdict_label`, `tldr`, and at least three reasoning points as strict JSON.
5. **Round analysis.** Valid verdict labels are normalized, counted, and used to compute majority agreement. Schema failures and provider failures do not silently become valid votes.
6. **Synthesis and handoff.** The judge may synthesize every round. When another round is needed, a structured summary can carry forward common ground, disagreements, open questions, and the current best label.
7. **Stopping and persistence.** The run stops at the configured threshold, after stagnation, or at the maximum round count. The final output and its audit metadata are stored in Django models and optionally logged as JSONL.

## Architecture

DJN has four cooperating layers.

| Layer | Primary modules | Responsibility |
|---|---|---|
| Web application | `webapp/audited_views.py`, `webapp/views.py`, templates | Query/clarification flow, results, history, audit downloads, feedback, optional Google Docs integration |
| Reasoning engine | `djn_engine/orchestration.py`, `run.py`, `schemas.py`, `audit.py`, `llms.py` | Selection handoff, prompts, provider calls, validation, deliberation, telemetry, synthesis |
| Data and adaptation | `djn_db/models.py`, `db_writer.py`, `selector.py`, `stats.py`, `audit_export.py` | Persistence, model pool, selection scores, feedback, rolling statistics, audit construction |
| Evaluation | `djn_engine/experiment_runner.py`, `evaluation/`, management commands, `config/` | Experiment definitions, baselines, call guards, scoring, quality checks, replay, calibration, load testing, metrics export |

The legacy presentation and optional OAuth helpers remain in `webapp/views.py`, but URL routing uses the audited adapter. New architectural claims should be checked against `djn_engine/orchestration.py` and the current database models rather than inferred from older screenshots or diagrams.

## Dynamic juror selection

The adaptive selector in `djn_db/selector.py` normalizes a query into one of these categories:

`coding`, `career`, `planning`, `factual`, `opinion`, `mathematical`, or `general`.

Each enabled, non-unhealthy model receives a weighted score using:

- category capability;
- schema-validity rate;
- user acceptance rate;
- majority alignment;
- reliability derived from error history; and
- observed latency.

Weights and statistical safeguards live in `config/selector.json`. The current selector uses cold-start priors and shrinks observed rates until a minimum sample count is reached. Capabilities are versioned in `config/model_capabilities.json`.

Selection is deterministic for the same database state, category, configuration, and seed. A seeded tie-breaker resolves equal scores. The selector first prefers provider diversity, then fills remaining seats by rank. It excludes models marked `UNHEALTHY`, records every candidate component and rank, and uses a named deterministic fallback list when database selection is unavailable. A dynamic run fails clearly if an enabled pool exists but cannot supply the requested number of healthy jurors.

For controlled baselines, `selector_mode: fixed` bypasses adaptive ranking and requires a complete fixed roster.

## Roles, rounds, and handoff

### Juror roles

DJN cycles through four role-conditioned perspectives:

| Role | Responsibility |
|---|---|
| `PROPOSER` | Develop a direct, actionable candidate answer |
| `CRITIC` | Challenge assumptions, evidence, and failure modes |
| `REFINER` | Reconcile useful ideas and improve precision |
| `RISK` | Surface safety, uncertainty, and downside considerations |

Role text and role/prompt version identifiers are stored with each juror response. Experiments can use generic roles as an ablation.

### Multi-round deliberation

Juror calls run in parallel within a round, subject to the configured concurrency. After validation, DJN records the verdict distribution, majority label, valid response count, agreement score, schema-valid rate, latency, and improvement from the preceding round.

The default full configuration allows up to three rounds, but both the maximum and stopping policy are configurable. Static one-round and single-model modes are forced to one round.

### Structured inter-round handoff

When `handoff_mode` is `structured`, the judge model produces a strict `RoundSummary` containing:

- common ground;
- key disagreements;
- open questions;
- the current best verdict label; and
- an explanation of that label.

The summary, stable hash, prompt/schema versions, schema status, model, latency, queue time, usage, cost availability, and errors are audited. If summary generation fails, DJN records the failure and falls back to basic majority/agreement context. Experiments may instead use raw prior responses or no handoff.

## Consensus, stopping, and final synthesis

### Agreement and stopping

Agreement is the share of valid juror verdicts supporting the majority label. Dynamic stopping has three terminal reasons:

- `THRESHOLD_MET`: enough valid jurors responded and agreement reached the configured threshold;
- `STAGNATION`: agreement improvement stayed below the configured minimum for the required number of rounds; or
- `MAX_ROUNDS`: the configured round limit was reached.

Stagnation and maximum-round exits are explicitly marked as best-available outcomes. These operational measures describe jury convergence; they are not ground-truth correctness measures.

### Moderator/judge

The judge synthesizes validated juror responses into:

- a final recommendation;
- supporting reasons;
- a `HIGH`, `MEDIUM`, or `LOW` confidence level;
- common ground;
- main disagreements; and
- conditional guidance.

Judge output is schema-validated. If no valid juror output exists, the system returns an auditable failure message instead of presenting an unsupported recommendation. Experiment configurations can replace model synthesis with a deterministic majority summary.

### Consensus Confidence

DJN calls the displayed confidence **Consensus Confidence**. It is an ordinal summary constrained by observed jury agreement and stop conditions. The engine records the judge's raw level, applies a versioned cap, and stores both values.

Consensus Confidence is **not a calibrated probability that the answer is correct**. It must not be reported as prediction probability, expected accuracy, or statistical certainty unless a separate labeled calibration study supports that interpretation. Calibration exports operate only on labeled evaluation results.

## Auditing, feedback, and statistics

### Run telemetry

Depending on provider availability, DJN records:

- run, round, juror, role, provider, and model identifiers;
- query/clarification state and assumptions;
- experiment, selector, capability, pricing, prompt, role, and schema versions;
- candidate score breakdowns, selected roster, seed, and fallback reason;
- raw/parsed status, schema validity, normalized verdicts, and errors;
- provider latency, queue latency, retries, round wall time, and total duration;
- input, output, total, and cached token counts when exposed by the provider;
- cost estimates only when a versioned verified price exists;
- agreement, improvement, stop reason, and best-available status; and
- judge and handoff outputs, hashes, telemetry, and confidence handling.

Unknown usage remains `null` with a reason. `config/model_pricing.json` is intentionally empty by default; populate it with verified provider prices before reporting monetary cost. A missing price must remain an unavailable cost, not be interpreted as zero.

The history UI exposes recent persisted runs. `/history/<run_id>/audit.json` downloads a sanitized audit record.

### Database model

| Model | Purpose |
|---|---|
| `LLMPool` | Configured models, providers, capabilities, health, and enablement |
| `DJNRun` | Query, preprocessing, roster, experiment, final output, confidence, and stop metadata |
| `DJNRound` | Round agreement, handoff, judge output, stopping, and timing |
| `JurorResponse` | Role, model snapshot, structured response, status, usage, cost, and latency |
| `ModelRollingStat` | Rebuildable category/model operational statistics |
| `RunFeedback` | One reversible up/down vote per run and voter session |
| `BenchmarkTask` | Frozen evaluation task, reference, scorer, source, and license metadata |
| `EvaluationResult` | Score associated with a task, run, and experiment configuration |

Migrations in `djn_db/migrations/` are part of the reproducible schema history and must be retained.

### Feedback and rolling statistics

Feedback uses an idempotent `(run, voter_session)` record. Re-voting updates the existing value rather than double-counting it. Statistics are rebuilt from persisted source rows, so feedback changes are reversible. The rebuilt statistics feed future adaptive selection, but user acceptance and majority alignment are behavioral signals—not benchmark accuracy.

## Experiments and evaluation

### Supported experiment dimensions

`djn_engine/schemas.py` defines immutable experiment configurations across:

- mode: single model, static jury, or full DJN;
- adaptive versus fixed selection;
- conditioned versus generic roles;
- structured, raw, or disabled handoff;
- dynamic versus fixed-round stopping;
- judge versus deterministic-majority synthesis; and
- jury size, threshold, round count, concurrency, temperature, token limits, and seed.

Tracked configurations include:

| File | Condition |
|---|---|
| `config/paper_full_djn.json` | Full adaptive DJN with roles, structured handoff, and dynamic stopping |
| `config/paper_full_no_handoff.json` | Full DJN ablation without inter-round handoff |
| `config/paper_single.json` | Fixed single-model baseline |
| `config/paper_static_jury.json` | Fixed generic-role, one-round jury baseline |
| `config/experiment.example.json` | General full-DJN example |

The matching JSON files under `outputs/` are tracked experiment manifests and should not be removed as disposable runtime output.

### Evaluation framework

The `evaluation/` package provides:

- validated task loading with dataset, split, source, license, reference, and scorer metadata;
- exact, normalized, numeric-tolerance, safe substring, and manual/rubric scoring paths;
- database consistency and audit-quality checks;
- labeled calibration summaries;
- retrospective threshold replay over observed rounds;
- paper metric exports with commit and database hashes; and
- deterministic mock-provider load testing.

Evaluation execution is resumable by `(task, experiment_config_id)` and stops on the first task failure. Live provider calls require two explicit flags after reviewing a dry run and its maximum projected call count.

## Project structure

```text
.
|-- .github/workflows/tests.yml        # Django validation in CI
|-- config/                            # Selector, capability, price, and experiment definitions
|-- djn_db/
|   |-- management/commands/           # Validation, evaluation, export, replay, and load commands
|   |-- migrations/                    # Versioned database schema
|   |-- models.py                      # Runs, rounds, responses, stats, feedback, benchmarks
|   |-- selector.py                    # Adaptive juror scoring and selection
|   `-- stats.py                       # Rebuildable feedback/performance statistics
|-- djn_engine/
|   |-- orchestration.py               # Authoritative audited DJN execution path
|   |-- audit.py                       # Call telemetry, token and cost handling
|   |-- experiment_runner.py           # Dry-run and projected-call safeguards
|   |-- llms.py / pool.py              # Provider adapters and configured model pool
|   |-- preprocess.py                  # Moderation, classification, clarification, assumptions
|   |-- privacy.py                     # Recursive export redaction
|   |-- run.py                         # Shared parsing, roles, agreement, confidence helpers
|   `-- schemas.py                     # Strict outputs and experiment schema
|-- djn_site/                          # Django settings and root URLs
|-- docs/                              # Architecture assets, privacy review, experiment metadata
|-- evaluation/                        # Tasks, scorers, quality, calibration, replay, metrics, load tests
|-- outputs/                           # Tracked experiment manifests; generated exports may also appear here
|-- webapp/                            # Audited web adapter, legacy UI helpers, templates, static assets
|-- manage.py
`-- requirements.txt
```

## Setup

The CI workflow uses Python 3.12. The commands below target Windows PowerShell from the repository root.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py seed_llmpool
```

Keep `.env` local. At minimum, set a non-default `DJANGO_SECRET_KEY` and configure only the providers you intend to use.

### Environment configuration

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini moderator/judge access |
| `OLLAMA_API_KEY`, `OLLAMA_BASE_URL` | Ollama Cloud access |
| `OLLAMA_LOCAL_URL` | Local Ollama endpoint; defaults to `http://127.0.0.1:11434` |
| `NVIDIA_API_KEY` | Default NVIDIA NIM credential |
| `META_API` | Per-model NIM credential used by Muse Glimmer |
| `NVIDIA_NIM_BASE_URL` | Optional NIM endpoint override |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS` | Django security/runtime settings |
| `DJN_MAX_ROUNDS`, `DJN_THRESHOLD`, `DJN_MIN_IMPROVEMENT` | Default stopping policy |
| `DJN_STAGNATION_ROUNDS`, `DJN_MIN_OK_JURORS` | Stagnation and minimum-valid-response controls |
| `DJN_MAX_CONCURRENCY`, `DJN_PROVIDER_TIMEOUT_SECONDS` | Concurrency and timeout controls |
| `DJN_MAX_RETRIES`, circuit-breaker variables | Provider retry and health controls |
| `DJN_SELECTION_SEED` | Deterministic selector seed |
| `GOOGLE_CLIENT_SECRETS_FILE`, `GOOGLE_OAUTH_REDIRECT_URI` | Optional Google Docs integration |

See `.env.example` for the complete current list. Never commit real values. The seeded model registry currently includes Ollama Cloud, local Ollama, and NVIDIA NIM jurors; Gemini is used for judge/moderator functions. Model availability and provider names change over time, so validate the pool instead of assuming every configured model is reachable.

## Running DJN

Start the Django development server:

```powershell
python manage.py runserver
```

Open `http://127.0.0.1:8000/`. The application provides the query flow, jury discussion, run history, sanitized audit downloads, feedback, and optional Google Docs sharing.

Validate the seeded model pool without live calls:

```powershell
python manage.py check_llm_pool
```

A live pool check is deliberately gated and bounded:

```powershell
python manage.py check_llm_pool --live --approve-live --max-calls 10
```

Review provider credentials, expected cost, and the maximum call count before using the live form.

## Validation and tests

The standard offline validation sequence is:

```powershell
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test djn_db -v 2
python manage.py rebuild_llm_stats
python manage.py check_djn_data
python manage.py check_llm_pool
```

The first three commands also run in `.github/workflows/tests.yml`. Tests use mocks where appropriate and do not establish live-provider availability or answer quality.

## Reproducibility workflow

### 1. Freeze the inputs

Use a licensed, versioned task file with reference answers and scorer metadata. `evaluation/datasets/example_tasks.json` is a synthetic format example. The paper-oriented dataset and configurations in the repository are controlled artifacts, but their presence does not substitute for checking source, license, reference validity, and scorer suitability.

### 2. Review an evaluation dry run

```powershell
python manage.py run_djn_evaluation `
  --tasks evaluation/datasets/example_tasks.json `
  --config config/experiment.example.json `
  --max-calls 100
```

This command validates configuration and credentials, calculates the maximum projected provider calls, and contacts no model provider.

### 3. Execute only after explicit approval

```powershell
python manage.py run_djn_evaluation `
  --tasks evaluation/datasets/example_tasks.json `
  --config config/experiment.example.json `
  --max-calls 100 `
  --execute `
  --approve-live `
  --manifest outputs/evaluation_manifest.json
```

This can incur provider usage and cost. Use frozen task/config files, a clean source commit, and an archived manifest for every condition. Do not mix different jury sizes or experiment configurations in one reported result.

### 4. Check data before reporting it

```powershell
python manage.py rebuild_llm_stats
python manage.py check_djn_data
```

Operational run history is not automatically benchmark evidence. Paper or research claims require valid references, appropriate scorers, comparable conditions, and a quality report without unresolved critical errors.

### 5. Export metrics and analyses

```powershell
python manage.py export_paper_metrics --output outputs/paper_metrics
python manage.py export_threshold_replay --output outputs/threshold_replay.json
python manage.py export_djn_calibration --output outputs/calibration.json
```

The paper metrics export includes the metric-definition version, source commit SHA, database SHA-256, query scope, denominators, and data-quality report. It refuses critical data-quality errors by default. `--allow-quality-errors` is available for diagnosis, not for silently legitimizing invalid results.

Replay one stored run at alternative observed-round thresholds with:

```powershell
python manage.py replay_djn_thresholds <run_id> --thresholds 0.5,0.6,0.7,0.75,0.8,0.9
```

Threshold replay is retrospective over rounds that actually occurred; it does not simulate unobserved model responses.

### Mock load testing

```powershell
python manage.py run_mock_load_test `
  --prompts 20 `
  --concurrency 4 `
  --latency-ms 10 `
  --jury-size 4 `
  --rounds 2
```

This is a deterministic orchestration benchmark with simulated provider latency. It measures scheduling and orchestration overhead under the supplied assumptions. It is **not** evidence of live-provider throughput, production scalability, model latency, rate-limit behavior, or end-to-end cost.

## Privacy and security

DJN records prompts and model outputs, so audit data may contain personal, confidential, copyrighted, or licensed material. Sanitized exports recursively redact common secrets, email addresses, and Indian phone-number formats:

```powershell
python manage.py export_sanitized_runs --output outputs/sanitized_runs.jsonl
```

Automated redaction is a safeguard, not proof that an artifact is safe to publish. Complete `docs/PRIVACY_REVIEW.md` and manually inspect every export. Never distribute `.env`, `credentials.json`, OAuth tokens, provider headers, the working `db.sqlite3`, raw logs, or credentials. Local databases, logs, environments, and secret files are ignored by Git.

## Known limitations and research caveats

- Multi-model deliberation increases calls, latency, and potential cost relative to a single call.
- Provider availability, rate limits, model behavior, token metadata, and prices can change independently of this repository.
- Monetary cost is reportable only for models with verified, versioned pricing. Unknown cost is not zero cost.
- Majority agreement and Consensus Confidence measure convergence, not factual correctness.
- Rolling acceptance and majority-alignment statistics may encode user or jury biases; they are not ground-truth accuracy.
- Dynamic selection quality depends on capability metadata, health state, sample size, and the representativeness of stored feedback.
- Structured JSON enforcement detects malformed responses but does not prove that valid-looking reasoning is correct.
- Stopping-threshold replay cannot recover counterfactual later rounds that were never executed.
- The included mock load test excludes real network and provider behavior.
- Calibration and accuracy claims require labeled, licensed, quality-checked benchmark data and appropriate scorers.
- The implementation supports the research workflow; it does not prove that a conference manuscript was submitted, accepted, or reproduced independently.

Use DJN as an experimental decision-support and multi-agent reasoning artifact. For high-stakes domains, independently verify evidence and keep qualified human oversight in the loop.
