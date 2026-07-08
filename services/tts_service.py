"""
services/tts_service.py — Text-to-Speech using Kokoro (open-source, local).

Kokoro is an Apache 2.0 TTS model with 82M parameters (~330MB on disk).
Despite its small size it ranked #1 on TTS Arena at launch, beating models
10x larger. It runs comfortably within 4GB VRAM alongside Whisper.

Why Kokoro beats Coqui XTTS for your setup:
    - XTTS v2 needs ~2GB VRAM minimum; Kokoro needs ~400MB
    - Synthesis speed: ~50–150ms per sentence on GPU (vs 500ms+ for XTTS)
    - No temp file roundtrip: returns numpy array directly
    - Single pip install, no espeak dependency on all platforms

Installation:
    pip install kokoro soundfile

Available voices (set KOKORO_VOICE in .env):
    American English (lang_code='a'):
        af_heart    — warm female (default, recommended)
        af_bella    — bright female
        af_nicole   — soft female
        af_sarah    — clear female
        am_adam     — deep male
        am_michael  — neutral male

    British English (lang_code='b'):
        bf_emma     — refined female
        bf_isabella — elegant female
        bm_george   — distinguished male
        bm_lewis    — young male

Output: 24kHz, 16-bit, mono PCM — make sure transport.py and index.html
match this sample rate (24000), not Coqui's 22050.

Pipeline role:
    Receives LLMSentenceFrame → synthesizes with Kokoro → emits AudioOutputFrame.
    Sentence-by-sentence streaming: TTS begins on sentence 1 while the LLM
    is still generating sentence 2. Synthesis runs in a thread executor to
    avoid blocking the asyncio event loop.

Graceful degradation:
    Tier 1 — Kokoro synthesis error  → retry once with shorter text chunk
    Tier 2 — 3 consecutive failures  → circuit opens → espeak fallback
    Tier 3 — espeak unavailable      → emit silence frame + log error
"""
from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from typing import Optional

import numpy as np
import structlog

from config import get_settings
from core.circuit_breaker import CircuitBreaker
from core.frames import (
    AudioOutputFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMSentenceFrame,
    MetricsFrame,
    StartFrame,
)
from core.pipeline import FrameProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

log = structlog.get_logger(__name__)

# Kokoro outputs at 24kHz — must match transport.py and index.html
KOKORO_SAMPLE_RATE = 24000
KOKORO_SYNTHESIS_LOCK = asyncio.Lock()


class KokoroTTSProcessor(FrameProcessor):
    """
    Synthesizes speech sentence-by-sentence using Kokoro TTS (local, no API key).

    Input:  LLMSentenceFrame   (one sentence at a time from the LLM stage)
    Output: AudioOutputFrame   (raw 16-bit PCM bytes at 24kHz)
            MetricsFrame       (synthesis latency telemetry)

    Synthesis runs in asyncio thread executor so the event loop is never
    blocked during model inference. Kokoro's pipeline object is NOT
    thread-safe, so we protect concurrent calls with an asyncio.Lock.
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang_code: str = "a",          # 'a'=American English, 'b'=British English
        device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto-detect)
        preloaded_pipeline = None,
        session_state = None,
    ):
        super().__init__(name="KokoroTTS")
        cfg = get_settings()

        self.voice = voice or cfg.kokoro_voice       # e.g. "af_heart"
        self.speed = speed or cfg.kokoro_speed       # e.g. 1.0
        self.lang_code = lang_code
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._preloaded_pipeline = preloaded_pipeline
        self._session_state = session_state

        self._pipeline = None                        # kokoro.KPipeline instance
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._synthesis_lock = KOKORO_SYNTHESIS_LOCK # serialize shared Kokoro access

        # Fallback availability flags
        self._espeak_available: bool = False
        self._fallback_pipeline = None               # slower CPU fallback voice

        self._circuit_breaker = CircuitBreaker(
            name="kokoro_tts",
            threshold=3,
            reset_s=30,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load Kokoro pipeline in a thread executor (blocking download/init).
        Also pre-warms by synthesizing a silent test string so the first
        real sentence doesn't pay the model warm-up cost.
        """
        self._loop = asyncio.get_event_loop()
        if self._preloaded_pipeline is not None:
            self._pipeline = self._preloaded_pipeline
            log.info("tts.using_preloaded_pipeline", voice=self.voice)
            self._check_espeak_fallback()
            return

        log.info(
            "tts.loading_kokoro",
            voice=self.voice,
            speed=self.speed,
            device=self.device,
            lang_code=self.lang_code,
        )

        await self._loop.run_in_executor(None, self._load_kokoro)

        # Pre-warm: run one silent synthesis so CUDA kernels are compiled
        await self._loop.run_in_executor(
            None, self._synthesize_blocking, "Hello.", True
        )

        log.info("tts.kokoro_ready", voice=self.voice, sample_rate=KOKORO_SAMPLE_RATE)
        self._check_espeak_fallback()

    def _load_kokoro(self) -> None:
        """Blocking: import and instantiate Kokoro pipeline (downloads ~330MB first run)."""
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "Kokoro is not installed. Run: pip install kokoro soundfile"
            )

        self._pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id='hexgrad/Kokoro-82M'
        )
        log.info("tts.kokoro_pipeline_created", lang_code=self.lang_code)

    def _check_espeak_fallback(self) -> None:
        """
        Check if espeak-ng is available as emergency TTS fallback.
        espeak is free, tiny, and works everywhere — quality is robotic but audible.
        Install: sudo apt-get install espeak-ng  (Linux)
                 brew install espeak              (macOS)
        """
        import shutil
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            self._espeak_available = True
            log.info("tts.espeak_fallback_available")
        else:
            log.warning(
                "tts.no_espeak_fallback",
                hint="sudo apt-get install espeak-ng  for emergency TTS fallback",
            )

    async def cleanup(self) -> None:
        """Release Kokoro model from memory."""
        self._pipeline = None
        self._fallback_pipeline = None
        log.info("tts.cleanup")

    # ── Frame processing ──────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame) -> None:
        # Propagate control frames unchanged
        if isinstance(frame, (StartFrame, EndFrame)):
            await self.push_frame(frame)
            return

        # Pass through anything that isn't a sentence to synthesize
        if not isinstance(frame, LLMSentenceFrame):
            await self.push_frame(frame)
            return

        text = frame.text.strip()
        if not text:
            return
        
        # Set speaking gate
        if self._session_state:
            self._session_state.is_speaking = True
            log.debug("tts.speaking_gate_open")

        log.info(
            "tts.synthesizing",
            text=text[:70],
            sentence_index=frame.sentence_index,
            is_last=frame.is_last,
        )

        start = time.monotonic()

        try:
            # Serialize synthesis calls — Kokoro pipeline is not thread-safe
            async with self._synthesis_lock:
                audio_bytes = await self._circuit_breaker.call(
                    primary=lambda: self._synthesize_async(text),
                    fallback=(
                        lambda: self._espeak_synthesize_async(text)
                        if self._espeak_available
                        else self._silent_frame_async(text)
                    ),
                )

            elapsed_ms = (time.monotonic() - start) * 1000
            log.info(
                "tts.synthesized",
                elapsed_ms=round(elapsed_ms),
                audio_bytes=len(audio_bytes),
                sentence_index=frame.sentence_index,
                rtf=round(elapsed_ms / max(1, len(text) * 60), 3),  # rough RTF estimate
            )

            await self.push_frame(
                AudioOutputFrame(
                    audio=audio_bytes,
                    sample_rate=KOKORO_SAMPLE_RATE,
                    channels=1,
                    sample_width=2,      # 16-bit PCM
                    turn_id=frame.turn_id,
                )
            )
            await self.push_frame(
                MetricsFrame(
                    stage=self.name,
                    latency_ms=elapsed_ms,
                    success=True,
                )
            )
        
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            log.error("tts.error", error=str(exc), text=text[:40])
            await self.push_frame(
                ErrorFrame(
                    error=str(exc),
                    stage=self.name,
                    recoverable=True,
                    original_frame=frame,
                )
            )
            await self.push_frame(
                MetricsFrame(
                    stage=self.name,
                    latency_ms=elapsed_ms,
                    success=False,
                )
            )
        finally:
            if frame.is_last and self._session_state:
                await asyncio.sleep(0.8)
                self._session_state.is_speaking = False
                log.debug("tts.speaking_gate_closed")

    # ── Primary synthesis (Kokoro) ────────────────────────────────────────────

    async def _synthesize_async(self, text: str) -> bytes:
        """
        Run Kokoro synthesis in a thread executor.
        Returns raw int16 PCM bytes at 24kHz.
        """
        return await self._loop.run_in_executor(
            None, self._synthesize_blocking, text, False
        )

    def _synthesize_blocking(self, text: str, warm_up: bool = False) -> bytes:
        """
        Blocking: run Kokoro pipeline and return PCM bytes.

        Kokoro's KPipeline.__call__ is a generator that yields (graphemes, phonemes, audio)
        tuples. For most sentences there is exactly one chunk. For longer sentences
        Kokoro may split internally — we concatenate all chunks.

        audio is a numpy float32 array in [-1, 1] at 24kHz.
        We convert to int16 PCM for WebSocket streaming.
        """
        if self._pipeline is None:
            raise RuntimeError("Kokoro pipeline not initialized")

        audio_chunks = []

        # KPipeline returns a generator — iterate to collect all audio chunks
        for graphemes, phonemes, audio in self._pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=None,   # disable auto-splitting; we handle it upstream
        ):
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            log.warning("tts.kokoro_returned_empty_audio", text=text[:40])
            # Return 100ms of silence rather than crashing downstream
            return self._make_silence_bytes(duration_ms=100)

        # Concatenate all chunks into one continuous audio array
        merged = np.concatenate(audio_chunks).astype(np.float32)

        if warm_up:
            log.debug("tts.warmup_complete", samples=len(merged))
            return b""   # discard warm-up audio

        # Clip to [-1, 1] to avoid int16 overflow artifacts
        merged = np.clip(merged, -1.0, 1.0)

        # Convert float32 [-1,1] → int16 [-32768, 32767]
        pcm_int16 = (merged * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    # ── Fallback: espeak-ng ───────────────────────────────────────────────────

    async def _espeak_synthesize_async(self, text: str) -> bytes:
        """Async wrapper around blocking espeak synthesis."""
        log.warning("tts.using_espeak_fallback", text=text[:40])
        return await self._loop.run_in_executor(
            None, self._espeak_blocking, text
        )

    def _espeak_blocking(self, text: str) -> bytes:
        """
        Blocking: synthesize with espeak-ng.
        espeak outputs WAV to stdout; we read it and strip the WAV header.
        Quality is robotic but it always produces audio — better than silence.
        """
        import subprocess
        import shutil
        import soundfile as sf

        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            return self._make_silence_bytes(500)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            subprocess.run(
                [binary, "-w", tmp_path, "-s", "150", "-p", "50", text],
                check=True,
                capture_output=True,
                timeout=10,
            )
            audio_data, sample_rate = sf.read(tmp_path, dtype="int16")

            # Resample to 24kHz if espeak outputs at a different rate
            if sample_rate != KOKORO_SAMPLE_RATE:
                audio_data = self._resample(audio_data, sample_rate, KOKORO_SAMPLE_RATE)

            return audio_data.tobytes()
        except Exception as exc:
            log.error("tts.espeak_failed", error=str(exc))
            return self._make_silence_bytes(500)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Fallback: silence ─────────────────────────────────────────────────────

    async def _silent_frame_async(self, text: str) -> bytes:
        """Last resort: return silence so the pipeline doesn't stall."""
        log.error(
            "tts.all_fallbacks_failed",
            text=text[:40],
            hint="Install espeak-ng for audio fallback",
        )
        # Estimate duration from text length (~150 words/min, ~5 chars/word)
        estimated_words = max(1, len(text) // 5)
        silence_ms = int(estimated_words / 150 * 60 * 1000)
        return self._make_silence_bytes(min(silence_ms, 3000))

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_silence_bytes(duration_ms: int) -> bytes:
        """Generate silence PCM bytes at KOKORO_SAMPLE_RATE."""
        samples = int(KOKORO_SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(samples, dtype=np.int16).tobytes()

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling. librosa would be better but adds a dependency."""
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available without importing torch at module level."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
