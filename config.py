"""
config.py — Central configuration using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # LLM (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key", "")
    groq_model: str = "openai/gpt-oss-120b"

    # ASR (Whisper)
    whisper_model: str = "small.en"
    whisper_device: str = "auto"

    # TTS (Kokoro)
    kokoro_voice: str = "af_heart" #(default; other options: af_bella, bf_emma, am_adam, bm_lewis)
    kokoro_speed: float = 1.0

    # VAD (Silero)
    vad_threshold: float = 0.5
    vad_silence_ms: int = 400

    # ── Server ────────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "info"

    # ── Latency budgets (ms) ──────────────────────────────────────────────────
    budget_vad_ms: int = 30
    budget_asr_ms: int = 3000
    budget_llm_ms: int = 5000
    budget_tts_ms: int = 300
    budget_total_ms: int = 12000

    # Circuit breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_s: int = 30

    # Metrics
    metrics_port: int = 9090
    enable_metrics: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton settings instance"""
    return Settings()
