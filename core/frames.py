"""
core/frames.py — All frame types that flow through the pipeline.
 
Frames are the atomic data units. Every pipeline stage consumes frames
from an upstream queue and produces frames into a downstream queue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np



# Frame base

@dataclass
class Frame:
    """Base class for all pipeline frames."""
    timestamp: float = field(default_factory= time.monotonic)
    turn_id: str = "" # UUID linking all frames in one conversational turn


# ─── Audio frames ─────────────────────────────────────────────────────────────

@dataclass
class AudioRawFrame(Frame):
    """
    Raw PCM audio from the microphone or WebSocket client.
    
    - audio:        numpy array, dtype=float32, shape=(samples,)
    - sample_rate:  typically 16000 Hz for ASR compatibility
    - duration_ms:  computed from len(audio) / sample_rate * 1000
    """
    audio: np.ndarray = field(default_factory= lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return len(self.audio) / self.sample_rate * 1000
    
@dataclass
class AudioOutputFrame(Frame):
    """Synthesized audio ready to stream to the WebSocket client."""
    audio: bytes = b""          # raw PCM bytes
    sample_rate: int = 22050
    channels: int = 1
    sample_width: int = 2       # bytes per sample (16-bit)


# ─── VAD frames ───────────────────────────────────────────────────────────────

class VADState(Enum):
    SILENCE = auto()
    SPEECH = auto()
    END_OF_SPEECH = auto()


@dataclass
class VADFrame(Frame):
    """Output from the Voice Activity Detector."""
    state: VADState = VADState.SILENCE
    probability: float = 0.0      # raw Silero speech probability
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000


# ─── Transcription frames ─────────────────────────────────────────────────────

@dataclass
class TranscriptionFrame(Frame):
    """
    Text output from Whisper ASR.
    
    - text:        transcribed text (may be empty)
    - is_final:    True = end of utterance, False = interim partial
    - confidence:  0.0–1.0 (Whisper avg log-prob converted to probability)
    - language:    detected language code (e.g. "en")
    """
    text: str = ""
    is_final: bool = False
    confidence: float = 1.0
    language: str = "en"
    asr_latency_ms: float = 0.0

# ─── LLM frames ───────────────────────────────────────────────────────────────

@dataclass
class LLMRequestFrame(Frame):
    """Assembled prompt sent to the LLM."""
    messages: list = field(default_factory=list)
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class LLMTokenFrame(Frame):
    """A single streaming token from the LLM."""
    token: str = ""
    is_final: bool = False


@dataclass
class LLMSentenceFrame(Frame):
    """
    A complete sentence aggregated from LLMTokenFrames.
    This is the unit sent to TTS — never wait for the full response.
    """
    text: str = ""
    sentence_index: int = 0      # 0-based order within the response
    is_last: bool = False        # True on the final sentence of a turn
    llm_latency_ms: float = 0.0


# ─── TTS frames ───────────────────────────────────────────────────────────────

@dataclass
class TTSRequestFrame(Frame):
    """Text to synthesize."""
    text: str = ""
    sentence_index: int = 0


# ─── Control frames ───────────────────────────────────────────────────────────

@dataclass
class ErrorFrame(Frame):
    """
    Signals a stage failure. Triggers fallback routing.
    Always carries enough context to decide which fallback to activate.
    """
    error: str = ""
    stage: str = ""           # which stage produced this error
    recoverable: bool = True
    original_frame: Optional[Frame] = None


@dataclass
class StartFrame(Frame):
    """Signals the pipeline to initialize. Sent once on connection."""
    session_id: str = ""


@dataclass
class EndFrame(Frame):
    """
    Signals end of a conversational turn (not end of session).
    Flushes all buffers. VAD resets to SILENCE state.
    """
    reason: str = "turn_complete"


@dataclass
class InterruptFrame(Frame):
    """
    User interrupted the assistant mid-response (barge-in).
    Stops TTS playback and clears the LLM output queue.
    """
    pass


@dataclass
class MetricsFrame(Frame):
    """Carries per-turn latency telemetry downstream to the metrics sink."""
    stage: str = ""
    latency_ms: float = 0.0
    success: bool = True
    fallback_used: bool = False
