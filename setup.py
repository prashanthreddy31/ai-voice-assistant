"""
setup.py — First-run setup checker for the Voice Assistant.

Current stack:
    VAD  : Silero VAD       (open-source, MIT, local)
    ASR  : OpenAI Whisper   (open-source, MIT, local)
    LLM  : Groq API         (cloud, free tier available)
    TTS  : Kokoro TTS       (open-source, Apache 2.0, local)

Run this ONCE before starting the server for the first time:
    python setup.py

What it does:
    1.  Checks Python version (3.10+ required)
    2.  Checks all required packages are installed
    3.  Validates .env file exists and GROQ_API_KEY is set
    4.  Verifies Groq API key works with a live test call
    5.  Checks CUDA availability and recommends device setting
    6.  Downloads and warms up Whisper model
    7.  Downloads and warms up Kokoro TTS model
    8.  Downloads Silero VAD model
    9.  Verifies static/index.html exists (browser UI)
    10. Runs a quick pipeline import check
    11. Prints final run command
"""

from __future__ import annotations

import os
import sys
import shutil
import asyncio
from pathlib import Path

ROOT = Path(__file__).parent

# ── Terminal colours (Windows-safe) ───────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}!{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {CYAN}→{RESET}  {msg}")
def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")
    print("─" * (len(msg) + 2))

errors   : list[str] = []   # hard blockers — server won't start
warnings : list[str] = []   # soft issues — server might still work


def record_fail(msg: str) -> None:
    fail(msg)
    errors.append(msg)

def record_warn(msg: str) -> None:
    warn(msg)
    warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────────────────────────────────────

def check_python() -> None:
    header("1. Python version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor >= 10:
        ok(f"Python {version_str}")
    else:
        record_fail(
            f"Python {version_str} is too old. "
            "Python 3.10+ required. Download from https://python.org"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Required packages
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    # (import_name,       pip_name,                   role)
    ("fastapi",           "fastapi",                  "Web server"),
    ("uvicorn",           "uvicorn[standard]",        "ASGI server"),
    ("websockets",        "websockets",               "WebSocket transport"),
    ("pydantic",          "pydantic",                 "Settings validation"),
    ("pydantic_settings", "pydantic-settings",        "Env var loading"),
    ("dotenv",            "python-dotenv",            "Dotenv file loader"),
    ("groq",              "groq",                     "Groq LLM client"),
    ("whisper",           "openai-whisper",           "Whisper ASR"),
    ("torch",             "torch",                    "PyTorch (VAD + Whisper)"),
    ("torchaudio",        "torchaudio",               "Audio ops for torch"),
    ("kokoro",            "kokoro",                   "Kokoro TTS"),
    ("soundfile",         "soundfile",                "Audio file I/O"),
    ("numpy",             "numpy",                    "Audio array ops"),
    ("scipy",             "scipy",                    "Signal processing"),
    ("librosa",           "librosa",                  "Audio resampling"),
    ("prometheus_client", "prometheus-client",        "Metrics"),
    ("structlog",         "structlog",                "Structured logging"),
    ("httpx",             "httpx",                    "Async HTTP client"),
    ("tenacity",          "tenacity",                 "Retry logic"),
]

def check_packages() -> None:
    header("2. Required packages")
    missing = []
    for import_name, pip_name, role in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            ok(f"{pip_name:<30} ({role})")
        except ImportError:
            record_fail(f"{pip_name:<30} MISSING  →  pip install {pip_name}")
            missing.append(pip_name)

    if missing:
        print()
        info(f"Install all missing packages with:")
        print(f"\n    pip install {' '.join(missing)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. .env file and GROQ_API_KEY
# ─────────────────────────────────────────────────────────────────────────────

def check_env_file() -> dict:
    header("3. Environment / .env file")

    env_path  = ROOT / ".env"
    example   = ROOT / ".env.example"
    env_values: dict = {}

    # Create .env from example if missing
    if not env_path.exists():
        if example.exists():
            shutil.copy(example, env_path)
            warn(".env not found — created from .env.example")
            warn("Open .env and set your GROQ_API_KEY before running the server")
            warnings.append("GROQ_API_KEY not set in .env")
        else:
            record_fail(".env and .env.example both missing")
        return env_values

    ok(".env file found")

    # Parse .env manually (avoids importing dotenv before package check)
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            env_values[key.strip()] = val.strip()

    # Check GROQ_API_KEY
    groq_key = env_values.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        record_fail(
            "GROQ_API_KEY is not set in .env\n"
            "     Get a free key at https://console.groq.com → API Keys"
        )
    elif groq_key.startswith("gsk_") and len(groq_key) > 20:
        ok(f"GROQ_API_KEY set  (gsk_...{groq_key[-4:]})")
    else:
        record_warn(
            f"GROQ_API_KEY looks unusual (doesn't start with 'gsk_'). "
            "Double-check it at console.groq.com"
        )

    # Check GROQ_MODEL
    groq_model = env_values.get("GROQ_MODEL", "llama-3.1-8b-instant")
    valid_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama-3.2-3b-preview",
    ]
    if groq_model in valid_models:
        ok(f"GROQ_MODEL = {groq_model}")
    else:
        record_warn(
            f"GROQ_MODEL = '{groq_model}' is not a recognised Groq model ID. "
            f"Valid options: {', '.join(valid_models)}"
        )

    # Check WHISPER_MODEL
    whisper_model = env_values.get("WHISPER_MODEL", "base.en")
    valid_whisper = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
    if whisper_model in valid_whisper:
        ok(f"WHISPER_MODEL = {whisper_model}")
    else:
        record_warn(f"WHISPER_MODEL = '{whisper_model}' is not a standard Whisper model name")

    # Check KOKORO_VOICE
    kokoro_voice = env_values.get("KOKORO_VOICE", "af_heart")
    ok(f"KOKORO_VOICE = {kokoro_voice}")

    # Check WHISPER_DEVICE
    whisper_device = env_values.get("WHISPER_DEVICE", "cpu")
    ok(f"WHISPER_DEVICE = {whisper_device}")

    return env_values


# ─────────────────────────────────────────────────────────────────────────────
# 4. Live Groq API key test
# ─────────────────────────────────────────────────────────────────────────────

async def _test_groq_async(api_key: str, model: str) -> tuple[bool, str]:
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        # Cheapest possible call — list models
        models_resp = await client.models.list()
        available = [m.id for m in models_resp.data]
        await client.close()
        if model in available:
            return True, f"Key valid. Model '{model}' confirmed available."
        else:
            return True, (
                f"Key valid but '{model}' not listed. "
                f"Available: {', '.join(available[:4])}..."
            )
    except Exception as exc:
        return False, str(exc)


def check_groq_api(env_values: dict) -> None:
    header("4. Groq API key (live test)")

    api_key = env_values.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    model   = env_values.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        warn("Skipping live test — GROQ_API_KEY not set")
        return

    info("Testing Groq API key (requires internet)...")
    success, message = asyncio.run(_test_groq_async(api_key, model))
    if success:
        ok(message)
    else:
        record_fail(f"Groq API test failed: {message}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CUDA / GPU check
# ─────────────────────────────────────────────────────────────────────────────

def check_cuda(env_values: dict) -> None:
    header("5. CUDA / GPU")

    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_name  = torch.cuda.get_device_name(0)
            vram_total   = torch.cuda.get_device_properties(0).total_memory
            vram_gb      = vram_total / (1024 ** 3)
            ok(f"CUDA available  —  {device_name}  ({vram_gb:.1f} GB VRAM)")

            whisper_device = env_values.get("WHISPER_DEVICE", "cuda")
            if whisper_device == "cpu":
                warn(
                    "WHISPER_DEVICE=cpu in .env but CUDA is available. "
                    "Set WHISPER_DEVICE=cuda for ~10x faster transcription."
                )
            else:
                ok("WHISPER_DEVICE=cuda  — GPU inference enabled")

            # VRAM budgeting advice
            whisper_model = env_values.get("WHISPER_MODEL", "base.en")
            vram_map = {
                "tiny.en": 0.39, "base.en": 0.50,
                "small.en": 1.0, "medium.en": 2.6,
            }
            whisper_vram = vram_map.get(whisper_model, 0.5)
            kokoro_vram  = 0.4
            total_vram   = whisper_vram + kokoro_vram

            if total_vram <= vram_gb:
                ok(
                    f"VRAM budget OK — Whisper {whisper_model} "
                    f"({whisper_vram}GB) + Kokoro ({kokoro_vram}GB) "
                    f"= {total_vram}GB / {vram_gb:.1f}GB available"
                )
            else:
                record_warn(
                    f"VRAM may be tight — Whisper {whisper_model} "
                    f"({whisper_vram}GB) + Kokoro ({kokoro_vram}GB) "
                    f"= {total_vram}GB but only {vram_gb:.1f}GB available. "
                    f"Try a smaller Whisper model."
                )
        else:
            warn("CUDA not available — running on CPU (slower)")
            info(
                "If you have an NVIDIA GPU, reinstall PyTorch with CUDA:\n"
                "     pip install torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
    except ImportError:
        record_fail("PyTorch not installed — run: pip install torch torchaudio")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Whisper model download
# ─────────────────────────────────────────────────────────────────────────────

def check_whisper(env_values: dict) -> None:
    header("6. Whisper ASR model")

    try:
        import whisper
        model_name = env_values.get("WHISPER_MODEL", "base.en")
        device     = env_values.get("WHISPER_DEVICE", "cuda")

        info(f"Loading Whisper '{model_name}' on {device} (downloads if not cached)...")

        import time
        t0 = time.monotonic()
        model = whisper.load_model(model_name, device=device)
        elapsed = (time.monotonic() - t0) * 1000

        # Quick inference test on 1s of silence
        import numpy as np
        silence = np.zeros(16000, dtype=np.float32)
        result  = model.transcribe(silence, fp16=(device == "cuda"), language="en")
        ok(
            f"Whisper '{model_name}' ready on {device}  "
            f"(loaded in {elapsed:.0f}ms, test transcription: '{result['text'].strip() or '<silence>'}')"
        )

        del model   # free memory for next checks

    except Exception as exc:
        record_fail(f"Whisper failed: {exc}")
        info("Fix: pip install openai-whisper")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Kokoro TTS model download
# ─────────────────────────────────────────────────────────────────────────────

def check_kokoro(env_values: dict) -> None:
    header("7. Kokoro TTS model")

    try:
        from kokoro import KPipeline
        import numpy as np

        voice     = env_values.get("KOKORO_VOICE", "af_heart")
        lang_code = "b" if voice.startswith("b") else "a"

        info(f"Loading Kokoro (lang={lang_code}, voice={voice}) — downloads ~330MB first run...")

        import time
        t0       = time.monotonic()
        pipeline = KPipeline(lang_code=lang_code)
        elapsed  = (time.monotonic() - t0) * 1000

        # Warm-up synthesis
        info("Running warm-up synthesis...")
        audio_chunks = []
        for _, _, audio in pipeline("Hello.", voice=voice, speed=1.0, split_pattern=None):
            if audio is not None:
                audio_chunks.append(audio)

        if audio_chunks:
            total_samples = sum(len(a) for a in audio_chunks)
            duration_ms   = total_samples / 24000 * 1000
            ok(
                f"Kokoro ready — voice='{voice}', "
                f"loaded in {elapsed:.0f}ms, "
                f"warm-up produced {duration_ms:.0f}ms of audio at 24kHz"
            )
        else:
            record_warn("Kokoro loaded but warm-up produced no audio. Check voice name.")

        del pipeline

    except ImportError:
        record_fail("Kokoro not installed — run: pip install kokoro soundfile")
    except Exception as exc:
        record_fail(f"Kokoro failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Silero VAD model download
# ─────────────────────────────────────────────────────────────────────────────

def check_silero_vad() -> None:
    header("8. Silero VAD model")

    try:
        import torch

        info("Loading Silero VAD from torch hub (downloads ~2MB if not cached)...")
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        model.eval()

        # Quick inference test on 512 samples of silence
        test_audio = torch.zeros(1, 512)
        with torch.no_grad():
            prob = model(test_audio, 16000)

        ok(f"Silero VAD ready — test speech probability on silence: {float(prob):.3f} (expected ~0.0)")

        del model

    except Exception as exc:
        record_fail(f"Silero VAD failed: {exc}")
        info("Fix: pip install silero-vad  or check your internet connection")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Static files
# ─────────────────────────────────────────────────────────────────────────────

def check_static_files() -> None:
    header("9. Static files (browser UI)")

    index_path = ROOT / "static" / "index.html"
    if index_path.exists():
        size_kb = index_path.stat().st_size / 1024
        # Verify UTF-8 readable (the Windows cp1252 bug check)
        try:
            content = index_path.read_text(encoding="utf-8")
            # Check sample rate is correct for Kokoro
            if "24000" in content:
                ok(f"static/index.html  ({size_kb:.1f} KB, UTF-8 OK, SAMPLE_RATE=24000 ✓)")
            elif "22050" in content:
                record_warn(
                    "static/index.html has SAMPLE_RATE=22050 (Coqui default). "
                    "Change to 24000 for Kokoro or audio will play at wrong pitch."
                )
            else:
                ok(f"static/index.html  ({size_kb:.1f} KB, UTF-8 OK)")
        except UnicodeDecodeError:
            record_fail(
                "static/index.html is not valid UTF-8. "
                "This causes the 500 error on Windows. Re-save it as UTF-8."
            )
    else:
        record_fail(
            "static/index.html not found. "
            "The browser UI will return 404."
        )

    # Check static directory itself
    static_dir = ROOT / "static"
    if static_dir.exists():
        ok(f"static/ directory exists")
    else:
        record_fail("static/ directory missing — create it and add index.html")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Pipeline import check
# ─────────────────────────────────────────────────────────────────────────────

def check_pipeline_imports() -> None:
    header("10. Pipeline import check")

    sys.path.insert(0, str(ROOT))

    imports = [
        ("config",                    "get_settings"),
        ("core.frames",               "AudioRawFrame, TranscriptionFrame, LLMSentenceFrame"),
        ("core.pipeline",             "FrameProcessor, Pipeline, PipelineRunner"),
        ("core.circuit_breaker",      "CircuitBreaker"),
        ("services.vad_service",      "SileroVADProcessor"),
        ("services.asr_service",      "WhisperASRProcessor"),
        ("services.llm_service",      "GroqLLMProcessor"),
        ("services.tts_service",      "KokoroTTSProcessor"),
        ("services.transport",        "WebSocketTransportInput, WebSocketTransportOutput"),
        ("utils.metrics",             "start_metrics_server"),
    ]

    for module, symbols in imports:
        try:
            mod = __import__(module, fromlist=symbols.split(","))
            for sym in [s.strip() for s in symbols.split(",")]:
                getattr(mod, sym)
            ok(f"{module}")
        except ImportError as exc:
            record_fail(f"{module}  —  ImportError: {exc}")
        except AttributeError as exc:
            record_fail(f"{module}  —  AttributeError: {exc}")
        except Exception as exc:
            record_warn(f"{module}  —  {type(exc).__name__}: {exc}")

    # Also verify config instantiates cleanly
    try:
        from config import get_settings
        cfg = get_settings()
        _ = cfg.server_host, cfg.server_port, cfg.groq_model
        _ = cfg.whisper_model, cfg.kokoro_voice, cfg.vad_threshold
        ok("config.get_settings() instantiates cleanly")
    except Exception as exc:
        record_fail(f"config.get_settings() failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. espeak-ng fallback check (optional)
# ─────────────────────────────────────────────────────────────────────────────

def check_espeak() -> None:
    header("11. espeak-ng TTS fallback (optional)")

    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if binary:
        ok(f"espeak found at {binary}  — TTS fallback available")
    else:
        warn(
            "espeak-ng not found. TTS will fall back to silence if Kokoro fails.\n"
            "     Install (optional):\n"
            "       Windows : https://github.com/espeak-ng/espeak-ng/releases\n"
            "       Linux   : sudo apt install espeak-ng\n"
            "       macOS   : brew install espeak"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary and final instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(env_values: dict) -> None:
    print()
    print("═" * 56)
    print(f"{BOLD}  Setup Summary{RESET}")
    print("═" * 56)

    if errors:
        print(f"\n  {RED}{BOLD}✗ {len(errors)} error(s) must be fixed before starting:{RESET}")
        for i, e in enumerate(errors, 1):
            print(f"    {i}. {e}")
    else:
        print(f"\n  {GREEN}{BOLD}✓ All checks passed — ready to run!{RESET}")

    if warnings:
        print(f"\n  {YELLOW}! {len(warnings)} warning(s) (server may still start):{RESET}")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")

    port  = env_values.get("SERVER_PORT", "8000")
    model = env_values.get("GROQ_MODEL", "llama-3.1-8b-instant")
    voice = env_values.get("KOKORO_VOICE", "af_heart")
    wm    = env_values.get("WHISPER_MODEL", "base.en")
    wd    = env_values.get("WHISPER_DEVICE", "cuda")

    print(f"""
  {BOLD}Current configuration:{RESET}
    LLM   : Groq  →  {model}
    ASR   : Whisper {wm} on {wd}
    TTS   : Kokoro  →  voice: {voice}  (24kHz)
    VAD   : Silero VAD (local)

  {BOLD}To start the server:{RESET}
    python main.py

  {BOLD}Then open in your browser:{RESET}
    http://localhost:{port}

  {BOLD}Prometheus metrics:{RESET}
    http://localhost:9090/metrics

  {BOLD}Health check:{RESET}
    http://localhost:{port}/health

  {BOLD}Run tests:{RESET}
    pytest tests/ -v --timeout=30
""")
    print("═" * 56)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        os.system("color")

    print()
    print("═" * 56)
    print(f"{BOLD}  Real-Time Voice Assistant — Setup Checker{RESET}")
    print(f"  Silero VAD · Whisper ASR · Groq LLM · Kokoro TTS")
    print("═" * 56)

    check_python()
    check_packages()
    env_values = check_env_file()
    check_groq_api(env_values)
    check_cuda(env_values)
    check_whisper(env_values)
    check_kokoro(env_values)
    check_silero_vad()
    check_static_files()
    check_pipeline_imports()
    check_espeak()
    print_summary(env_values)

    # Exit with error code if blockers found (useful in CI)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
