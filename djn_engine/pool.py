from .llms import LLMConfig

JUDGE = LLMConfig(
    name="judge_gemini",
    provider="gemini",
    model="gemini-3.5-flash-lite",
    temperature=0.2,
)

JURORS = [
    LLMConfig(name="gpt-oss-20b", provider="ollama_cloud", model="gpt-oss:20b-cloud", temperature=0.4),
    LLMConfig(name="gpt-oss-120b", provider="ollama_cloud", model="gpt-oss:120b-cloud", temperature=0.35),
    LLMConfig(name="deepseek-coder:6.7b", provider="ollama", model="deepseek-coder:6.7b", temperature=0.35),
    LLMConfig(name="nemotron-3-ultra", provider="ollama_cloud", model="nemotron-3-ultra:cloud", temperature=0.35),
    LLMConfig(name="gemma4:31b", provider="ollama_cloud", model="gemma4:31b-cloud", temperature=0.35),
    LLMConfig(name="minimax-m3", provider="ollama_cloud", model="minimax-m3:cloud", temperature=0.35),
    LLMConfig(name="glm-5.2", provider="ollama_cloud", model="glm-5.2:cloud", temperature=0.35),

    LLMConfig(
        name="nemotron-3-super",
        provider="nim",
        model="nvidia/nemotron-3-super",
        temperature=0.35,
    )

]
