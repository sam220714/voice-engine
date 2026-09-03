import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    # Stage 1: Whisper STT
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    whisper_beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

    # Stage 2: LLM via Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    llm_default_instruction: str = os.getenv(
        "LLM_DEFAULT_INSTRUCTION",
        "Clean up this raw speech transcript: fix punctuation and grammar, "
        "remove filler words, and keep the meaning unchanged.",
    )
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))


settings = Settings()
