"""
services/asr_service.py — Automatic Speech Recognition using OpenAI Whisper.

Whisper is fully open-source (MIT), runs locally. No API key required.
Models range from tiny.en (~39MB) to large-v3 (~1.5GB).

Pipeline role:
    Receives VADFrame (END_OF_SPEECH) → runs Whisper → emits TranscriptionFrame.
    Also has a fallback mode: lower-quality but faster tiny.en model.
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Optional

import numpy as np
import structlog

from config import get_settings
from core.circuit_breaker import CircuitBreaker
from core.frames import (
    EndFrame,
    Frame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    VADFrame,
    VADState,
)
from core.pipeline import FrameProcessor

log = structlog.get_logger(__name__)

HALLUCINATIONS = {
    "", ".", "..", "...", ". . .", ". . . .", ". . . . .",
    ". . . . . . . . .", "you", "bye", "thanks", "thank you",
    "subscribe", "subtitles by", "music",
}

WHISPER_PROMPT = (
    "The following is a clear conversation in Indian English with an AI voice assistant. "
    "Topics include cricket, sports, Indian culture, history, science, technology, and general knowledge. " 
    "Indian names and places are spoken clearly."
)


class WhisperASRProcessor(FrameProcessor):
    """
    Transcribes speech using Whisper.

    Input:  VADFrame with state=END_OF_SPEECH (contains buffered audio)
    Output: TranscriptionFrame
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        language: str = "en",
        use_faster_whisper: bool = False,  # set True if faster-whisper installed
        preloaded_model=None
    ):
        super().__init__(name="WhisperASR")
        cfg = get_settings()
        self.model_name = model_name or cfg.whisper_model
        self.device = device or cfg.whisper_device
        self.language = language
        self.use_faster_whisper = use_faster_whisper
        self._preloaded_model = preloaded_model

        self._model = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Fallback: use a smaller model if primary times out
        self._fallback_model_name = "tiny.en"
        self._fallback_model = None

        self._circuit_breaker = CircuitBreaker(
            name="whisper_asr",
            threshold=3,
            reset_s=30,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._loop = asyncio.get_event_loop()       
        if self._preloaded_model is not None:
            self._model = self._preloaded_model
            log.info("asr.using_preloaded_model", model=self.model_name)
            return
        
        log.info("asr.loading_whisper", model=self.model_name, device=self.device)
        if self.use_faster_whisper:
            await self._loop.run_in_executor(None, self._load_faster_whisper)
        else:
            await self._loop.run_in_executor(None, self._load_whisper)

        log.info("asr.model_loaded", model=self.model_name)

    def _load_whisper(self) -> None:
        """Blocking: loads original openai-whisper."""
        import whisper
        self._model = whisper.load_model(self.model_name, device=self.device)
        # Pre-load fallback
        if self.model_name != self._fallback_model_name:
            self._fallback_model = whisper.load_model(
                self._fallback_model_name, device=self.device
            )

    def _load_faster_whisper(self) -> None:
        """Blocking: loads faster-whisper (CTranslate2 backend, 4x faster)."""
        from faster_whisper import WhisperModel
        compute_type = "float16" if self.device == "cuda" else "int8"
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=compute_type
        )

    async def cleanup(self) -> None:
        self._model = None
        self._fallback_model = None
        log.info("asr.cleanup")

    # ── Processing ────────────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame) -> None:
        if isinstance(frame, (StartFrame, EndFrame)):
            await self.push_frame(frame)
            return

        if not isinstance(frame, VADFrame):
            await self.push_frame(frame)
            return

        if frame.state != VADState.END_OF_SPEECH:
            return  # Only transcribe complete utterances

        audio = frame.audio
        sample_rate = frame.sample_rate

        log.info(
            "asr.transcribing",
            duration_ms=round(len(audio) / sample_rate * 1000),
        )

        start = time.monotonic()

        try:
            result = await self._circuit_breaker.call(
                primary=lambda: self._transcribe_async(audio, sample_rate, model=self._model),
                fallback=(
                    lambda: self._transcribe_async(audio, sample_rate, model=self._fallback_model)
                    if self._fallback_model else None
                ),
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            text = result.get("text", "").strip()

            if self._is_hallucination(text):
                log.debug("asr.filtered_hallucination", text=text)
                return

            log.info(
                "asr.result",
                text=text[:80],
                elapsed_ms=round(elapsed_ms),
            )

            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    is_final=True,
                    confidence=result.get("confidence", 1.0),
                    language=result.get("language", self.language),
                    asr_latency_ms=elapsed_ms,
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
            log.error("asr.failed", error=str(exc))
            await self.push_frame(
                MetricsFrame(stage=self.name, latency_ms=elapsed_ms, success=False)
            )

    async def _transcribe_async(
        self, audio: np.ndarray, sample_rate: int, model=None
    ) -> dict:
        """Run Whisper transcription in a thread (blocks CPU)."""
        if model is None:
            raise RuntimeError("ASR model not loaded")

        if self.use_faster_whisper:
            return await self._loop.run_in_executor(
                None, self._run_faster_whisper, audio, sample_rate, model
            )
        else:
            return await self._loop.run_in_executor(
                None, self._run_whisper, audio, sample_rate, model
            )

    def _run_whisper(self, audio: np.ndarray, sample_rate: int, model) -> dict:
        """Original Whisper inference (blocking)."""
        import whisper

        # Whisper expects float32 at 16kHz
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        audio = audio.astype(np.float32)

        result = model.transcribe(
            audio,
            language=self.language,
            fp16=self.device == "cuda", 
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,  
            initial_prompt=WHISPER_PROMPT,
        )

        # Convert avg_logprob to a 0–1 confidence approximation
        segments = result.get("segments", [])
        avg_logprob = (
            sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            if segments else -1.0
        )
        confidence = max(0.0, min(1.0, 1.0 + avg_logprob))

        return {
            "text": result.get("text", ""),
            "language": result.get("language", self.language),
            "confidence": confidence,
        }

    def _run_faster_whisper(self, audio: np.ndarray, sample_rate: int, model) -> dict:
        """Faster-Whisper inference (blocking) — 4x faster via CTranslate2."""
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        segments, info = model.transcribe(
            audio,
            language=self.language,
            beam_size=1,      
            vad_filter=False,  
        )

        text_parts = [seg.text for seg in segments]
        return {
            "text": " ".join(text_parts).strip(),
            "language": info.language,
            "confidence": info.language_probability,
        }
    
    def _is_hallucination(self, text: str) -> bool:
        """Return True if whisper output looks like a hallucination."""
        cleaned = text.strip().lower()
        # Excat match against known hallucinations
        if cleaned in HALLUCINATIONS:
            return True
        # Only dots and spaces
        if all(c in '. ' for c in cleaned):
            return True
        # Too short to be a real question
        if len(cleaned) < 3:
            return True
    
