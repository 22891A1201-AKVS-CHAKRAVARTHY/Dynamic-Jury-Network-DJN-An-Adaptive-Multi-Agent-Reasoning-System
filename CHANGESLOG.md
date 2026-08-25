# Changes to the Initial DJN Model Configuration

`23rd Aug 2026`

This document summarizes the changes made to the initial Dynamic Jury Network (DJN) implementation. The updates improve the final response displayed in the Django UI, replace unavailable models in the jury pool, and add support for a locally hosted Ollama model.

## 1. Final-response formatting

### Previous behavior

The Judge prompt already required the following six output fields:

1. `final_recommendation`
2. `confidence`
3. `why`
4. `common_ground`
5. `main_disagreement`
6. `conditional_guidance`

However, the previous `_format_final_display()` implementation in `djn_engine/run.py` displayed only three sections in the Django UI:

1. Final Recommendation
2. Confidence Level
3. Reason

Although the Judge generated all six fields, the formatter discarded `common_ground`, `main_disagreement`, and `conditional_guidance` when preparing the user-visible response.

### Current behavior

The formatter now displays all six parts of the Judge output:

1. Final Recommendation
2. Confidence
3. Why
4. Common Ground
5. Main Disagreements
6. Conditional Guidance

List-based sections are presented as bullet points. If a section has no values, the UI displays `None identified` instead of leaving an unclear empty section.

### Provider-response normalization

The `_msg_text()` function in `djn_engine/run.py` was also updated. Some LangChain providers return `AIMessage.content` as a list of typed content blocks rather than a plain string. Previously, this list could be converted directly to text and displayed in the UI, exposing internal fields such as:

- `type`
- `text`
- `extras`
- provider signatures

The updated function extracts only the user-visible `text` values. This allows the JSON response to be parsed normally and prevents provider metadata from appearing in the final Django UI output.

## 2. Model-pool changes

Several models in the initial `djn_engine/pool.py` configuration became unavailable. They were replaced with currently selected alternatives, including one locally hosted model.

| Role/provider | Initial model | Current model | Change |
|---|---|---|---|
| Judge / Gemini | `gemini-2.5-flash-lite` | `gemini-3.5-flash-lite` | Judge model replaced |
| Juror / Ollama Cloud | `gpt-oss:20b-cloud` | `gpt-oss:20b-cloud` | Unchanged |
| Juror / Ollama Cloud | `gpt-oss:120b-cloud` | `gpt-oss:120b-cloud` | Unchanged |
| Juror / Ollama Cloud → Local Ollama | `deepseek-v3.1:671b-cloud` | `deepseek-coder:6.7b` | Replaced with a locally hosted DeepSeek model |
| Juror / Ollama Cloud | `qwen3-coder:480b-cloud` | `nemotron-3-ultra:cloud` | Cloud model replaced |
| Juror / Ollama Cloud | `qwen3-vl:235b-cloud` | `gemma4:31b-cloud` | Cloud model replaced |
| Juror / Ollama Cloud | `minimax-m2:cloud` | `minimax-m3:cloud` | Model version replaced |
| Juror / Ollama Cloud → NVIDIA NIM | `glm-4.6:cloud` | `meta/muse-glimmer-30b` | Replaced with an NVIDIA-hosted Meta model |
| Juror / NVIDIA NIM | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3-super-120b-a12b` | NIM juror replaced |

The local DeepSeek juror is configured as follows:

```python
LLMConfig(
    name="deepseek-coder:6.7b",
    provider="ollama",
    model="deepseek-coder:6.7b",
    temperature=0.35,
)
```

Using the distinct provider name `ollama` is important because it separates the local Ollama connection from models accessed through `ollama_cloud`.

## 3. Local Ollama support in `llms.py`

The initial `build_llm()` implementation supported these providers:

- `gemini`
- `ollama_cloud`
- `nim`

Because the new DeepSeek juror uses `provider="ollama"`, the initial implementation raised:

```text
ValueError: Unknown provider: ollama
```

A dedicated local-Ollama branch was added to `djn_engine/llms.py`:

```python
if cfg.provider == "ollama":
    local_url = cfg.base_url or os.getenv(
        "OLLAMA_LOCAL_URL",
        "http://127.0.0.1:11434",
    )
    if not isinstance(local_url, str) or not local_url.strip():
        raise RuntimeError(
            "OLLAMA_LOCAL_URL must be a URL string, for example "
            "http://127.0.0.1:11434"
        )
    return ChatOllama(
        model=cfg.model,
        temperature=cfg.temperature,
        base_url=local_url.strip(),
    )
```

This change provides the following behavior:

- Uses `cfg.base_url` when a model-specific URL is supplied.
- Otherwise reads `OLLAMA_LOCAL_URL` from the environment.
- Defaults to `http://127.0.0.1:11434` when the environment variable is absent.
- Verifies that the resolved URL is a non-empty string.
- Removes accidental surrounding whitespace before creating `ChatOllama`.
- Keeps local and cloud Ollama configuration independent.

The local environment can explicitly define the endpoint as:

```dotenv
OLLAMA_LOCAL_URL=http://127.0.0.1:11434
```

The existing `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` variables remain dedicated to Ollama Cloud. The local model does not use the cloud authorization header.

## Files affected

- `djn_engine/run.py` — normalizes provider content blocks and displays all six Judge output sections.
- `djn_engine/pool.py` — replaces unavailable models and registers the local DeepSeek juror.
- `djn_engine/llms.py` — adds validated local Ollama provider support.

---
