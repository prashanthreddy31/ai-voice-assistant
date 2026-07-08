"""
services/vad_service.py — Voice Activity Detection using Silero VAD.
 
Silero VAD is fully open-source (MIT), runs locally, requires no API key.
It's a tiny LSTM that classifies 30ms audio chunks as speech or silence.
 
Pipeline role:
    Receives AudioRawFrame → emits VADFrame with state + buffered speech audio.
    Accumulates audio chunks while speech is detected, then emits the complete
    utterance when silence exceeds vad_silence_ms. This is the "turn detector".
 
Key parameters:
    threshold:      0.0–1.0 speech probability (0.5 is a good starting point)
    silence_ms:     how many ms of silence before declaring end-of-speech
    min_speech_ms:  discard utterances shorter than this (prevents noise bursts)
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Optional, List

import numpy as np
import torch
import structlog

from config import get_settings
from core.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    StartFrame,
    VADFrame,
    VADState,
    MetricsFrame
)
from core.pipeline import FrameProcessor

log = structlog.get_logger(__name__)

SILERO_REPO = "snakers4/silero-vad"
SILERO_MODEL = "silero_vad"

ENERGY_GATE_RMS = 0.008
 
# Number of 512-sample chunks to use for noise floor calibration
CALIBRATION_CHUNKS = 10   # ~320ms at 16kHz

class SileroVADProcessor(FrameProcessor):
    """
    Wraps Silero VAD in a FrameProcessor.
 
    Input:  AudioRawFrame (any length, 16kHz float32)
    Output: VADFrame (SPEECH / SILENCE / END_OF_SPEECH)
            AudioRawFrame with buffered speech on END_OF_SPEECH

    Key design:
        - Buffers 128-sample chunks into 512-sample windows for Silero
        - Energy gate filters background noise before Silero runs
        - Auto-calibrates threshold above measured noise floor
        - Forces END_OF_SPEECH after 15s to prevent infinite SPEECH state
    """

    def __init__(
            self,
            threshold: Optional[float] = None,
            silence_ms: Optional[int] = None,
            min_speech_ms: int = 200,
            sample_rate: int = 16000,
            preloaded_model = None,
            session_state = None,
    ):
        super().__init__(name="SileroVAD")
        cfg = get_settings()
        self.threshold = threshold or cfg.vad_threshold
        self.silence_ms = silence_ms or cfg.vad_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        self._preloaded_model = preloaded_model
        self._session_state = session_state

        self._model = None

        # State machine
        self._state: VADState = VADState.SILENCE
        self._speech_buffer: Deque[np.ndarray] = deque()
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None

        self._chunk_buffer: np.ndarray = np.array([], dtype=np.float32)

        # ── Noise floor calibration ────────────────────────────────────────
        self._noise_calibrated: bool   = False
        self._noise_samples:    List[float] = []
        self._noise_floor:      float  = 0.0


    async def initialize(self) -> None:
        log.info("vad.loading_silero_model")
        if self._preloaded_model is not None:
            self._model = self._preloaded_model
            log.info("vad.using_preloaded_model")
        else:
            log.info("vad.loading_silero_model")
            loop = asyncio.get_event_loop()
            self._model, _ = await loop.run_in_executor(None, self._load_model)
            log.info("vad.model_loaded")
 
        log.info(
            "vad.ready",
            threshold=self.threshold,
            silence_ms=self.silence_ms,
            energy_gate_rms=ENERGY_GATE_RMS,
            calibration_chunks=CALIBRATION_CHUNKS,
        )

    def _load_model(self):
        """Blocking: downloads and caches Silero VAD from torch hub.""" 
        model, utils = torch.hub.load(
            repo_or_dir=SILERO_REPO,
            model=SILERO_MODEL,
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        model.eval()
        return model, utils
    
    async def cleanup(self) -> None:
        self._speech_buffer.clear()
        self._chunk_buffer = np.array([], dtype=np.float32)
        log.info("vad.cleanup")

    # Frame Processing

    async def process_frame(self, frame: Frame) -> None:
        if isinstance(frame, (StartFrame, EndFrame)):
            await self.push_frame(frame)
            return
        
        if not isinstance(frame, AudioRawFrame):
            await self.push_frame(frame)
            return
        
        # Buffer incoming chunks until we have enough for Silero (512 samples)
        self._chunk_buffer = np.concatenate([self._chunk_buffer, frame.audio])

        start = time.monotonic()
        processed_any = False

        while len(self._chunk_buffer) >= 512:
            processed_any = True
            chunk = self._chunk_buffer[:512]
            self._chunk_buffer = self._chunk_buffer[512:]
            
            await self._process_chunk(chunk, frame.sample_rate)

        if processed_any:
            elapsed_ms = (time.monotonic() - start) * 1000
            await self.push_frame(
                MetricsFrame(stage=self.name, latency_ms=elapsed_ms, success=True)
            )

    async def _process_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Process one 512-sample chunk through the full VAD pipeline."""
        # TTS speaking gate
        if self._session_state and self._session_state.is_speaking:
            if self._state == VADState.SPEECH:
                self._reset_state()
            return
 
        # ── Step 1: Energy gate ───────────────────────────────────────────
        # RMS energy check — much cheaper than running Silero.
        # Background noise has RMS < 0.008; speech has RMS > 0.02.
        rms = float(np.sqrt(np.mean(chunk ** 2)))
 
        if rms < ENERGY_GATE_RMS:
            # Definitely not speech — skip Silero entirely
            # print(f"[VAD] energy_gate blocked  rms={rms:.5f} < {ENERGY_GATE_RMS}",flush=True,)
            await self._update_state(False, chunk, sample_rate)
            return
 
        # ── Step 2: Noise floor calibration ──────────────────────────────
        # Use the first few non-silent windows to measure room noise, but still
        # process speech with the configured threshold during calibration.
        if not self._noise_calibrated:
            speech_prob = self._run_vad(chunk)
            self._noise_samples.append(speech_prob)
 
            log.debug(
                "vad.calibrating",
                sample=len(self._noise_samples),
                total=CALIBRATION_CHUNKS,
                rms=round(rms, 4),
                probability=round(speech_prob, 4),
            )
 
            if len(self._noise_samples) >= CALIBRATION_CHUNKS:
                self._noise_floor = max(self._noise_samples)
                # Set threshold 0.15 above noise floor, minimum 0.3, maximum 0.7
                self.threshold = min(0.7, max(0.3, self._noise_floor + 0.15))
                self._noise_calibrated = True
                log.info(
                    "vad.noise_calibrated",
                    noise_floor=round(self._noise_floor, 3),
                    new_threshold=round(self.threshold, 3),
                )
                print(
                    f"[VAD] Calibration complete — noise_floor={self._noise_floor:.3f} "
                    f"new_threshold={self.threshold:.3f}",
                    flush=True,
                )
            speech_detected = speech_prob >= self.threshold
            await self._update_state(speech_detected, chunk, sample_rate)
            return
 
        # ── Step 3: Silero inference ──────────────────────────────────────
        speech_prob    = self._run_vad(chunk)
        speech_detected = speech_prob >= self.threshold
 
        log.debug(
            "vad.inference",
            rms=round(rms, 4),
            probability=round(speech_prob, 4),
            threshold=round(self.threshold, 2),
            speech=speech_detected,
        )
 
        await self._update_state(speech_detected, chunk, sample_rate)

    # ── Silero inference ────────────────────────────────────────────────────── 

    def _run_vad(self, chunk: np.ndarray) -> float:
        """Run Silero on a single 512-sample chunk. Returns speech probability."""
        tensor = torch.from_numpy(chunk).unsqueeze(0)   # (1, 512)
        with torch.no_grad():
            speech_prob = self._model(tensor, self.sample_rate)
        return float(speech_prob)
    
    async def _update_state(
            self, is_speech: bool, audio: np.ndarray, sample_rate:int
    ) -> None:
        now = time.monotonic()

        if is_speech:
            self._silence_start = None
            self._speech_buffer.append(audio)

            if self._state == VADState.SILENCE:
                self._state = VADState.SPEECH
                self._speech_start = now
                log.debug("vad.speech_started")

            await self.push_frame(
                VADFrame(
                    state=VADState.SPEECH,
                    probability=1.0,
                    audio=audio,
                    sample_rate=sample_rate
                )
            )

        else:
            if self._state == VADState.SPEECH:
                if self._silence_start is None:
                    self._silence_start = now

                silence_elapsed_ms = (now - self._silence_start) * 1000
                speech_total_ms    = (now - (self._speech_start or now)) * 1000

                # ── Safety net: force END_OF_SPEECH after 15s ─────────────
                if speech_total_ms >=15000:
                    log.warning(
                        "vad.force_end_of_speech",
                        reason="max_duration_exceeded",
                        duration_ms=round(speech_total_ms),
                    )
                    await self._emit_end_of_speech(sample_rate)
                    return

                if silence_elapsed_ms >= self.silence_ms:
                    # Merge all buffered speech into one utterance
                    speech_duration_ms = speech_total_ms
 
                    if speech_duration_ms >= self.min_speech_ms:
                        log.info(
                            "vad.end_of_speech",
                            duration_ms=round(speech_duration_ms),
                            samples=sum(len(c) for c in self._speech_buffer),
                        )
                        await self._emit_end_of_speech(sample_rate)
                    else:
                        log.debug(
                            "vad.discarding_short_utterance",
                            duration_ms=round(speech_duration_ms),
                        )
                        self._reset_state()
                
            else:
                await self.push_frame(
                    VADFrame(
                        state=VADState.SILENCE,
                        probability=0.0,
                        audio=audio,
                        sample_rate=sample_rate,
                    )
                )

    async def _emit_end_of_speech(self, sample_rate: int) -> None:
        """Merge buffered speech and push END_OF_SPEECH frame downstream."""
        if self._speech_buffer:
            merged_audio = np.concatenate(list(self._speech_buffer))
            await self.push_frame(
                VADFrame(
                    state=VADState.END_OF_SPEECH,
                    probability=0.0,
                    audio=merged_audio,
                    sample_rate=sample_rate,
                )
            )
        self._reset_state()
 
    def _reset_state(self) -> None:
        """Reset all state machine variables after an utterance completes."""
        self._state         = VADState.SILENCE
        self._speech_buffer.clear()
        self._speech_start  = None
        self._silence_start = None
 

            
