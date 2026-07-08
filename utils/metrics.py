"""
utils/metrics.py — Prometheus metrics for the voice assistant pipeline.

Exposes per-stage latency histograms, timeout counters, fallback rates,
and circuit breaker states. Access the Grafana-compatible endpoint at:
    http://localhost:9090/metrics

Import `record_stage_latency` as a context manager in each processor to
automatically record timing without coupling the business logic to metrics.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Optional

import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

from config import get_settings
from core.frames import AudioOutputFrame, Frame, MetricsFrame, TranscriptionFrame
from core.pipeline import FrameProcessor

log = structlog.get_logger(__name__)

# ─── Metric definitions ───────────────────────────────────────────────────────

STAGE_LATENCY = Histogram(
    "voice_pipeline_stage_latency_ms",
    "Latency per pipeline stage in milliseconds",
    labelnames=["stage", "status"],  # status: success | timeout | error | fallback
    buckets=[10, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000, 10000],
)

TURN_LATENCY = Histogram(
    "voice_pipeline_turn_latency_ms",
    "End-to-end latency per conversational turn (VAD end → first audio byte)",
    buckets=[100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000],
)

TIMEOUT_COUNTER = Counter(
    "voice_pipeline_timeouts_total",
    "Number of stage timeouts",
    labelnames=["stage"],
)

FALLBACK_COUNTER = Counter(
    "voice_pipeline_fallbacks_total",
    "Number of times a fallback was activated",
    labelnames=["stage", "reason"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "voice_pipeline_circuit_breaker_open",
    "1 if circuit is OPEN, 0 if CLOSED",
    labelnames=["circuit"],
)

SESSION_COUNTER = Counter(
    "voice_pipeline_sessions_total",
    "Total WebSocket sessions started",
)

ACTIVE_SESSIONS = Gauge(
    "voice_pipeline_active_sessions",
    "Currently active WebSocket sessions",
)

ASR_CONFIDENCE = Histogram(
    "voice_pipeline_asr_confidence",
    "Whisper transcription confidence score",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

TTS_BYTES = Histogram(
    "voice_pipeline_tts_audio_bytes",
    "Size of TTS audio output per sentence",
    buckets=[1000, 5000, 10000, 25000, 50000, 100000, 250000],
)

PIPELINE_INFO = Info(
    "voice_pipeline",
    "Static metadata about the pipeline configuration",
)

# ─── Helper functions ─────────────────────────────────────────────────────────


def record_stage(
    stage: str,
    latency_ms: float,
    success: bool = True,
    fallback_used: bool = False,
    timeout: bool = False,
) -> None:
    """Record a single stage observation to the relevant metrics."""
    status = "success"
    if timeout:
        status = "timeout"
        TIMEOUT_COUNTER.labels(stage=stage).inc()
    elif fallback_used:
        status = "fallback"
        FALLBACK_COUNTER.labels(stage=stage, reason="timeout").inc()
    elif not success:
        status = "error"

    STAGE_LATENCY.labels(stage=stage, status=status).observe(latency_ms)


@contextlib.contextmanager
def time_stage(stage: str, budget_ms: Optional[float] = None):
    """
    Context manager: times a block and records it to Prometheus.

    Usage:
        with time_stage("WhisperASR", budget_ms=3000) as t:
            result = model.transcribe(audio)
        print(f"Took {t.elapsed_ms}ms")
    """

    class _Timer:
        elapsed_ms: float = 0.0

    t = _Timer()
    start = time.monotonic()
    timed_out = False
    try:
        yield t
    except TimeoutError:
        timed_out = True
        raise
    finally:
        t.elapsed_ms = (time.monotonic() - start) * 1000
        success = not timed_out
        record_stage(
            stage=stage,
            latency_ms=t.elapsed_ms,
            success=success,
            timeout=timed_out,
        )
        if budget_ms and t.elapsed_ms > budget_ms:
            log.warning(
                "metrics.budget_exceeded",
                stage=stage,
                budget_ms=budget_ms,
                elapsed_ms=round(t.elapsed_ms),
                overage_ms=round(t.elapsed_ms - budget_ms),
            )


def update_circuit_breaker_state(circuit_name: str, is_open: bool) -> None:
    """Call this when a circuit breaker changes state."""
    CIRCUIT_BREAKER_STATE.labels(circuit=circuit_name).set(1 if is_open else 0)


def record_turn_complete(total_latency_ms: float) -> None:
    """Record the full end-to-end latency for one conversational turn."""
    TURN_LATENCY.observe(total_latency_ms)


class MetricsSinkProcessor(FrameProcessor):
    """Records telemetry frames and forwards user-visible frames downstream."""

    def __init__(self):
        super().__init__(name="MetricsSink")

    async def process_frame(self, frame: Frame) -> None:
        if isinstance(frame, MetricsFrame):
            record_stage(
                stage=frame.stage,
                latency_ms=frame.latency_ms,
                success=frame.success,
                fallback_used=frame.fallback_used,
            )
            return

        if isinstance(frame, TranscriptionFrame):
            ASR_CONFIDENCE.observe(frame.confidence)
        elif isinstance(frame, AudioOutputFrame):
            TTS_BYTES.observe(len(frame.audio))

        await self.push_frame(frame)


# ─── Metrics server ───────────────────────────────────────────────────────────


def start_metrics_server() -> None:
    """Start Prometheus HTTP scrape endpoint in a background thread."""
    cfg = get_settings()
    if not cfg.enable_metrics:
        log.info("metrics.disabled")
        return

    # Set static pipeline info
    PIPELINE_INFO.info(
        {
            "asr_model": cfg.whisper_model,
            "llm_model": cfg.groq_model,
            "tts_model": cfg.kokoro_voice,
            "vad_threshold": str(cfg.vad_threshold),
        }
    )

    start_http_server(cfg.metrics_port)
    log.info(
        "metrics.server_started",
        port=cfg.metrics_port,
        url=f"http://localhost:{cfg.metrics_port}/metrics",
    )
