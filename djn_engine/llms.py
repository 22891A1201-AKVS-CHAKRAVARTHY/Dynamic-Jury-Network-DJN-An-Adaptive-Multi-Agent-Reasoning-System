import os
from dataclasses import dataclass
from typing import Optional, Any

# Jurors: Ollama Cloud
from langchain_ollama import ChatOllama

# Nemotron: NVIDIA NIM
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Judge: Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class LLMConfig:
    name: str                 # friendly ID used in logs
    provider: str             # "ollama_cloud" | "nim" | "gemini"
    model: str
    temperature: float = 0.2
    base_url: Optional[str] = None


def build_llm(cfg: LLMConfig) -> Any:
    """
    Returns a LangChain chat model instance.
    Provider behavior matches your txt skeletons (Ollama Cloud / NIM / Gemini).
    """

    if cfg.provider == "gemini":
        # Uses GOOGLE_API_KEY from env
        return ChatGoogleGenerativeAI(
            model=cfg.model,
            temperature=cfg.temperature,
        )

    if cfg.provider == "ollama_cloud":
        # Ollama Cloud typically uses base_url; auth depends on hosting.
        # If your cloud uses header keys, we pass them via client_kwargs where possible.
        base_url = cfg.base_url or os.getenv("OLLAMA_BASE_URL")
        api_key = os.getenv("OLLAMA_API_KEY")

        # ChatOllama supports base_url. Auth patterns vary; keep it ready via headers.
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
            base_url=base_url,
            client_kwargs={"headers": headers} if headers else None,
        )

    if cfg.provider == "nim":
        # NVIDIA NIM via langchain-nvidia-ai-endpoints
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is missing. Django likely isn't loading .env. "
                "Load dotenv in settings.py or set the env var in your OS."
            )
        # Base URL is optional. If you set it in .env, pass it directly instead of forcing env overrides.
        base_url = cfg.base_url or os.getenv("NVIDIA_NIM_BASE_URL") or None

        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key

        # Some versions support base_url; if yours doesn't, fall back safely.
        if base_url:
            try:
                return ChatNVIDIA(**kwargs, base_url=base_url)
            except TypeError:
                pass

        return ChatNVIDIA(**kwargs)


    raise ValueError(f"Unknown provider: {cfg.provider}")
