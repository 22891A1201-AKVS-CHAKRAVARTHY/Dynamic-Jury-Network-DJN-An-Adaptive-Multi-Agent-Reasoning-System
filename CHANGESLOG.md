# DJN Changelog

This changelog summarizes meaningful repository evolution by development phase. It is derived from the Git history and current implementation rather than being a one-to-one copy of commit messages.

## Current / August 2026 — Audited Research Revision

The August revision expanded DJN from its earlier application-oriented implementation into an auditable research artifact with explicit baselines, ablations, evaluation safeguards, and reproducibility metadata.

### Added

- Introduced the authoritative audited orchestration path in `djn_engine/orchestration.py` and routed the web application through `webapp/audited_views.py` while retaining legacy presentation and optional Google Docs helpers.
- Added role-conditioned juror prompts for `PROPOSER`, `CRITIC`, `REFINER`, and `RISK`, with persisted role instructions and version identifiers.
- Added strict juror, judge, and inter-round summary schemas, including auditable handling of malformed provider output and judge-schema failures.
- Added structured inter-round handoff containing common ground, disagreements, open questions, the current best label, and its rationale. Handoff content is hashed and stored with prompt/schema versions, validity, errors, timing, usage, and cost availability.
- Added dynamic stopping based on consensus threshold, minimum valid jurors, stagnation, and maximum rounds, with explicit stop reasons and best-available outcomes.
- Added provider-neutral call telemetry covering model/provider identity, latency, queue time, retries, token usage when available, and cost only when a verified versioned price exists.
- Added selection traces with category normalization, per-candidate score components and ranks, selected model IDs, versions, seed, health exclusions, and fallback reason.
- Added deterministic experiment configuration snapshots and stable configuration IDs.

### Changed

- Redesigned adaptive juror selection to combine category capability, schema validity, user acceptance, majority alignment, reliability, and latency.
- Added cold-start priors and minimum-sample shrinkage so sparse historical statistics do not dominate selection.
- Made tie-breaking deterministic from the selection seed, preferred provider diversity, excluded unhealthy models, and added deterministic named fallback behavior.
- Clarified final confidence as **Consensus Confidence**: an ordinal judge assessment capped by observed agreement and stop conditions, not a calibrated probability of correctness.
- Expanded final judge output to preserve the recommendation, reasons, confidence, common ground, main disagreements, and conditional guidance.
- Improved response normalization for providers that return typed content blocks, preventing provider metadata from leaking into user-visible text.
- Made feedback idempotent and reversible per run and voter session; rolling statistics are now rebuilt from persisted source rows.
- Expanded the model registry and provider adapter layer to separate local Ollama from Ollama Cloud and to support per-model NVIDIA NIM credentials.

### Evaluation and reproducibility

- Added immutable experiment modes for full DJN, full DJN without handoff, a single-model baseline, and a fixed one-round static jury baseline.
- Added tracked paper experiment configurations and matching manifests under `outputs/`.
- Added benchmark task and evaluation result persistence with dataset, split, source, license, scorer, reference, and experiment metadata.
- Added guarded evaluation dry runs that validate configurations, report provider credential state, calculate projected calls, and make no provider calls.
- Required both `--execute` and `--approve-live` for benchmark execution, with a configurable maximum-call guard and resumable results by task/configuration.
- Added exact, normalized, numeric-tolerance, safe structured-substring, and manual/rubric scoring paths; numeric scoring was updated to accept a reference value appearing within a longer answer.
- Added data-quality checks for audit completeness, round continuity, response counts, agreement validity, handoff integrity, confidence caps, usage availability, stop reasons, and rebuilt statistics.
- Added reproducible paper metric exports in JSON, CSV, and Markdown with metric version, source commit, database hash, query scope, denominators, and quality report.
- Added labeled calibration export and retrospective stopping-threshold replay at both run and aggregate levels.
- Added deterministic mock-provider load testing to measure orchestration behavior separately from live-provider performance.
- Added CI validation for Django checks, migration drift, and the `djn_db` test suite.
- Added a versioned environment template, Git attributes, selector/capability/pricing registries, experiment metadata, and a privacy review checklist.

### Database and auditing

- Extended the schema through migration `0004_benchmarktask_evaluationresult_runfeedback_and_more.py`.
- Expanded `DJNRun`, `DJNRound`, `JurorResponse`, `LLMPool`, and `ModelRollingStat` with experiment, prompt, selection, role, health, timing, usage, cost, handoff, confidence, and error metadata.
- Added `BenchmarkTask`, `EvaluationResult`, and `RunFeedback` entities.
- Added sanitized audit downloads and JSONL exports for persisted runs.
- Added commands to seed and validate the model pool, rebuild statistics, check data, run evaluations, export metrics/calibration/replay data, and perform mock load tests.

### Fixed

- Corrected the NVIDIA Nemotron juror identifier to `nvidia/nemotron-3-super-120b-a12b`.
- Replaced the unavailable GLM jury entry with NVIDIA-hosted `meta/muse-glimmer-30b` and added its dedicated `META_API` credential mapping.
- Updated capability data, orchestration, experiment-model synchronization, and pool seeding for the current NVIDIA models.
- Changed model-pool seeding to synchronize existing rows instead of leaving obsolete models enabled indefinitely; validation now reports unexpected enabled rows and missing configured rows.
- Added checks for insufficient healthy enabled candidates and preserved explicit failures rather than silently running an undersized dynamic jury.

### Privacy, security, and repository cleanup

- Added recursive redaction for common secrets, sensitive credential fields, email addresses, and Indian phone-number formats; documented the required manual privacy review before sharing artifacts.
- Restored and expanded `.env.example` without real credentials.
- Strengthened repository hygiene by removing obsolete local helper/archive artifacts and expanding ignore rules for credentials, environments, databases, logs, caches, temporary files, and generated runtime data.
- Removed a temporary credential-bearing local helper after credential rotation and removed other local analysis/archive helpers and Python cache directories.
- Removed the conference-workflow-only source ZIP helper from the project and consolidated architecture, operation, and reproducibility documentation into one authoritative README.

### Documentation

- Recorded versioned experiment and implementation metadata for the audited revision and merge.
- Added `docs/MODEL_POOL_UPDATES_2026-08-25.md` describing the NVIDIA model corrections, provider credential split, seed synchronization behavior, and validation procedure.
- Replaced the earlier MVP-oriented README and separate reproduction guide with a current, research-oriented README grounded in the audited engine and evaluation workflow.

## August 2026 — Initial Model and Output Modernization

Before the audited research revision, the initial application was cleaned up to address unavailable models and final-output problems.

### Changed

- Updated the configured judge and jury pool as earlier provider models became unavailable.
- Added a dedicated local Ollama provider path and `OLLAMA_LOCAL_URL` support for a locally hosted DeepSeek Coder juror, keeping local and cloud Ollama configuration independent.
- Improved final-response rendering so all six judge sections reached the Django UI: recommendation, confidence, reasons, common ground, disagreements, and conditional guidance.
- Normalized typed provider message content to extract user-visible text rather than rendering raw metadata blocks.

### Documentation

- Added the original model-configuration change record, then folded it into `CHANGESLOG.md` and removed the redundant source note.
- Updated the main JSONL log filename and related ignored-runtime configuration.

## March 2026 — Initial Public Application

### Added

- Introduced the Django web application, DJN engine, model pool, database persistence, selection/statistics layer, JSON enforcement, logging, templates, styling, screenshots, and architecture diagrams.
- Added the initial `LLMPool`, `DJNRun`, `DJNRound`, `JurorResponse`, and `ModelRollingStat` schema and model-pool seeding command.
- Added the original multi-model query, clarification, jury, judgment, history, feedback, and optional Google Docs sharing flows.
- Added the first repository README and later corrected its project-tree formatting.
- Added a complete project execution video under `docs/Execution Video/`.

### Changed

- Performed an early source cleanup that removed unused code and presentation remnants across the engine, database, settings, and web application.
- Updated Python dependencies required by the application and simplified the database writer.

## March 2026 — Repository Initialization

- Created the initial DJN source tree, Django project, database migration, provider integrations, user interface assets, environment template, and ignore rules.
