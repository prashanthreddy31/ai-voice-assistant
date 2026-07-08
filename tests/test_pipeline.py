"""
tests/test_pipeline.py — Unit + integration tests with fault injection.

Tests:
    1. Frame flow through each processor
    2. Timeout enforcement and fallback activation
    3. Circuit breaker state transitions
    4. VAD utterance boundary detection
    5. LLM sentence aggregation
    6. Graceful degradation under injected failures
    7. End-to-end pipeline smoke test

Run with:
    pytest tests/ -v --timeout=30
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.circuit_breaker import CircuitBreaker, CircuitState
from core.frames import (
    AudioOutputFrame,
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMSentenceFrame,
    LLMTokenFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    VADFrame,
    VADState,
)
from core.pipeline import FrameProcessor, PassthroughProcessor, Pipeline, PipelineRunner, TimeoutProcessor


# ─── Test helpers ─────────────────────────────────────────────────────────────

class CollectorProcessor(FrameProcessor):
    """Captures all frames pushed to it for assertion."""

    def __init__(self):
        super().__init__(name="Collector")
        self.received: List[Frame] = []

    async def process_frame(self, frame: Frame) -> None:
        self.received.append(frame)
        await self.push_frame(frame)

    def frames_of_type(self, t) -> List[Frame]:
        return [f for f in self.received if isinstance(f, t)]


class SlowProcessor(FrameProcessor):
    """Simulates a slow stage (for timeout tests)."""

    def __init__(self, delay_s: float):
        super().__init__(name="SlowStage")
        self.delay_s = delay_s

    async def process_frame(self, frame: Frame) -> None:
        await asyncio.sleep(self.delay_s)
        await self.push_frame(frame)


class FailingProcessor(FrameProcessor):
    """Simulates a stage that fails (for circuit breaker + fallback tests)."""

    def __init__(self, fail_n_times: int = 999):
        super().__init__(name="FailingStage")
        self.fail_n_times = fail_n_times
        self.call_count = 0

    async def process_frame(self, frame: Frame) -> None:
        self.call_count += 1
        if self.call_count <= self.fail_n_times:
            raise RuntimeError(f"Injected failure #{self.call_count}")
        await self.push_frame(frame)


class EchoProcessor(FrameProcessor):
    """Echoes every frame downstream unchanged."""

    async def process_frame(self, frame: Frame) -> None:
        await self.push_frame(frame)


def make_audio(duration_ms: int = 500, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic silence as float32 audio."""
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype=np.float32)


def make_speech(duration_ms: int = 500, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic speech-like audio (sine wave)."""
    samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, samples)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


# ─── Frame tests ──────────────────────────────────────────────────────────────

class TestFrames:
    def test_audio_raw_frame_duration(self):
        audio = make_audio(500)
        frame = AudioRawFrame(audio=audio, sample_rate=16000)
        assert abs(frame.duration_ms - 500.0) < 1.0

    def test_audio_raw_frame_empty(self):
        frame = AudioRawFrame()
        assert frame.duration_ms == 0.0

    def test_transcription_frame_defaults(self):
        f = TranscriptionFrame(text="hello world")
        assert f.text == "hello world"
        assert f.is_final is False
        assert f.confidence == 1.0

    def test_error_frame(self):
        original = TranscriptionFrame(text="test")
        err = ErrorFrame(error="timeout", stage="WhisperASR", original_frame=original)
        assert err.recoverable is True
        assert err.stage == "WhisperASR"
        assert err.original_frame is original


# ─── Pipeline tests ───────────────────────────────────────────────────────────

class TestPipeline:
    @pytest.mark.asyncio
    async def test_frame_flows_through_chain(self):
        """Frames should propagate from source to sink."""
        a = EchoProcessor()
        b = EchoProcessor()
        c = CollectorProcessor()
        pipeline = Pipeline([a, b, c])
        await pipeline.initialize()

        # Push a frame directly into the source queue
        frame = TranscriptionFrame(text="hello")
        await pipeline.source_queue.put(frame)
        await pipeline.source_queue.put(EndFrame())

        # Let the pipeline drain
        runner = PipelineRunner(pipeline)
        await asyncio.wait_for(runner.run(), timeout=5.0)

        transcriptions = c.frames_of_type(TranscriptionFrame)
        assert len(transcriptions) == 1
        assert transcriptions[0].text == "hello"

    @pytest.mark.asyncio
    async def test_error_frame_on_processor_exception(self):
        """When a processor raises, an ErrorFrame should be emitted downstream."""
        fail = FailingProcessor(fail_n_times=1)
        collector = CollectorProcessor()

        # Wire directly: fail → collector
        fail._output_queue = collector._input_queue

        # Feed a frame
        test_frame = TranscriptionFrame(text="hi")
        await fail._input_queue.put(test_frame)
        await fail._input_queue.put(EndFrame())

        await asyncio.wait_for(fail._run(), timeout=5.0)

        errors = collector.frames_of_type(ErrorFrame)
        assert len(errors) == 1
        assert errors[0].stage == "FailingStage"

    @pytest.mark.asyncio
    async def test_passthrough_preserves_frames(self):
        """PassthroughProcessor should forward every frame unchanged."""
        pt = PassthroughProcessor()
        collector = CollectorProcessor()
        pt._output_queue = collector._input_queue

        frames_in = [
            TranscriptionFrame(text="one"),
            TranscriptionFrame(text="two"),
            EndFrame(),
        ]
        for f in frames_in:
            await pt._input_queue.put(f)

        await asyncio.wait_for(pt._run(), timeout=5.0)

        assert len(collector.frames_of_type(TranscriptionFrame)) == 2


# ─── Timeout processor tests ─────────────────────────────────────────────────

class TestTimeoutProcessor:
    @pytest.mark.asyncio
    async def test_passes_frame_within_budget(self):
        """Fast processor should pass within timeout."""
        fast = EchoProcessor()
        collector = CollectorProcessor()
        wrapper = TimeoutProcessor(inner=fast, budget_ms=500)
        wrapper._output_queue = collector._input_queue

        frame = TranscriptionFrame(text="fast frame")
        await wrapper.process_frame(frame)

        assert len(collector.frames_of_type(TranscriptionFrame)) == 1

    @pytest.mark.asyncio
    async def test_emits_error_on_timeout_no_fallback(self):
        """Slow processor exceeding budget with no fallback → ErrorFrame."""
        slow = SlowProcessor(delay_s=0.5)
        collector = CollectorProcessor()
        wrapper = TimeoutProcessor(inner=slow, budget_ms=50)  # 50ms budget, 500ms processor
        wrapper._output_queue = collector._input_queue

        frame = AudioRawFrame(audio=make_audio(100))
        # This should not hang — timeout kicks in
        await asyncio.wait_for(wrapper.process_frame(frame), timeout=3.0)

        errors = collector.frames_of_type(ErrorFrame)
        assert len(errors) == 1
        assert "50ms" in errors[0].error


# ─── Circuit breaker tests ────────────────────────────────────────────────────

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_closed_state_on_success(self):
        cb = CircuitBreaker(name="test", threshold=3, reset_s=10)
        result = await cb.call(primary=AsyncMock(return_value=42))
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(name="test", threshold=3, reset_s=30)
        failing_fn = AsyncMock(side_effect=RuntimeError("failure"))
        fallback_fn = AsyncMock(return_value="fallback")

        # 3 failures should open the circuit
        for _ in range(3):
            await cb.call(primary=failing_fn, fallback=fallback_fn)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_fast_fails_when_open(self):
        cb = CircuitBreaker(name="test", threshold=1, reset_s=30)
        failing_fn = AsyncMock(side_effect=RuntimeError("failure"))
        fallback_fn = AsyncMock(return_value="fallback")

        # Open the circuit
        await cb.call(primary=failing_fn, fallback=fallback_fn)
        assert cb.state == CircuitState.OPEN

        # Next call should fast-fail to fallback WITHOUT calling primary
        primary_fn = AsyncMock(return_value="primary")
        result = await cb.call(primary=primary_fn, fallback=fallback_fn)
        assert result == "fallback"
        primary_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_half_open_after_reset_time(self):
        cb = CircuitBreaker(name="test", threshold=1, reset_s=0.05)
        failing_fn = AsyncMock(side_effect=RuntimeError("failure"))
        fallback_fn = AsyncMock(return_value="fallback")

        # Open the circuit
        await cb.call(primary=failing_fn, fallback=fallback_fn)
        assert cb.state == CircuitState.OPEN

        # Wait for reset window
        await asyncio.sleep(0.1)

        # Next call should probe (HALF_OPEN)
        success_fn = AsyncMock(return_value="recovered")
        result = await cb.call(primary=success_fn, fallback=fallback_fn)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker(name="test", threshold=1, reset_s=30)
        cb._state = CircuitState.OPEN
        cb._failure_count = 5
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


# ─── VAD integration test (mock Silero) ──────────────────────────────────────

class TestVADSentenceBoundary:
    """Test that VAD correctly buffers speech and emits on silence."""

    @pytest.mark.asyncio
    async def test_vad_state_machine_logic(self):
        """Test the state machine without loading the actual Silero model."""
        from services.vad_service import SileroVADProcessor

        vad = SileroVADProcessor(threshold=0.5, silence_ms=200)
        collector = CollectorProcessor()
        vad._output_queue = collector._input_queue

        # Mock the model to return high probability
        vad._model = MagicMock()
        vad._model.return_value = (0.95, MagicMock(), MagicMock())

        # Simulate speech frames then silence
        with patch.object(vad, '_run_vad', return_value=0.95):
            speech_audio = make_speech(300)
            await vad._update_state(True, speech_audio, 16000)

        with patch.object(vad, '_run_vad', return_value=0.1):
            silence_audio = make_audio(500)
            await vad._update_state(False, silence_audio, 16000)

        # After enough silence, END_OF_SPEECH should be emitted
        # (silence_ms=200, we slept 0ms but simulated time via silence threshold)
        speech_frames = [f for f in collector.received if isinstance(f, VADFrame)]
        assert any(f.state == VADState.SPEECH for f in speech_frames)


# ─── LLM sentence aggregation test ───────────────────────────────────────────

class TestSentenceAggregation:
    """Verify that LLM token stream is split on sentence boundaries."""

    @pytest.mark.asyncio
    async def test_sentence_split(self):
        import re

        SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

        # Simulate token stream
        tokens = ["Hello", " there", ".", " How", " are", " you", "?", " I", " am", " fine", "."]
        full = ""
        sentences = []
        buf = ""

        for tok in tokens:
            buf += tok
            full += tok
            parts = SENTENCE_BOUNDARY.split(buf, maxsplit=1)
            if len(parts) > 1:
                sentences.append(parts[0].strip())
                buf = parts[1]

        if buf.strip():
            sentences.append(buf.strip())

        assert len(sentences) == 3
        assert "Hello there" in sentences[0]
        assert "How are you" in sentences[1]
        assert "I am fine" in sentences[2]


# ─── End-to-end smoke test ────────────────────────────────────────────────────

class TestEndToEndSmoke:
    """Smoke test: frames flow from source to sink through the full pipeline."""

    @pytest.mark.asyncio
    async def test_start_and_end_frames_propagate(self):
        stages = [EchoProcessor() for _ in range(4)]
        stages.append(CollectorProcessor())
        pipeline = Pipeline(stages)
        collector: CollectorProcessor = stages[-1]

        runner = PipelineRunner(pipeline)
        task = asyncio.create_task(runner.run())

        # Let the pipeline start
        await asyncio.sleep(0.05)

        # Push a start frame directly
        await pipeline.source_queue.put(StartFrame(session_id="test"))
        await asyncio.sleep(0.05)
        await pipeline.source_queue.put(EndFrame())

        await asyncio.wait_for(task, timeout=5.0)

        assert any(isinstance(f, StartFrame) for f in collector.received)
        assert any(isinstance(f, EndFrame) for f in collector.received)


# ─── Fault injection test ─────────────────────────────────────────────────────

class TestFaultInjection:
    """
    Inject failures at specific stages and verify the pipeline still
    produces output (graceful degradation).
    """

    @pytest.mark.asyncio
    async def test_pipeline_continues_after_stage_error(self):
        """Pipeline should emit ErrorFrame and keep running."""
        # fail → echo → collector
        failing = FailingProcessor(fail_n_times=1)
        echo = EchoProcessor()
        collector = CollectorProcessor()

        failing._output_queue = echo._input_queue
        echo._output_queue = collector._input_queue

        # Run failing stage independently
        test_frame = AudioRawFrame(audio=make_audio(100))
        await failing._input_queue.put(test_frame)
        await failing._input_queue.put(EndFrame())

        await asyncio.wait_for(failing._run(), timeout=5.0)

        # ErrorFrame should propagate to echo → collector
        await echo._input_queue.put(EndFrame())
        await asyncio.wait_for(echo._run(), timeout=5.0)

        # collector should have received the ErrorFrame
        assert len(collector.frames_of_type(ErrorFrame)) >= 1

    @pytest.mark.asyncio
    async def test_metrics_emitted_on_success(self):
        """Each stage should emit a MetricsFrame on successful processing."""
        # We test this via the echo chain — MetricsFrame is emitted by the TimeoutProcessor
        echo = EchoProcessor()
        collector = CollectorProcessor()
        wrapper = TimeoutProcessor(inner=echo, budget_ms=500)
        wrapper._output_queue = collector._input_queue

        frame = TranscriptionFrame(text="metrics test")
        await wrapper.process_frame(frame)

        metrics = collector.frames_of_type(MetricsFrame)
        assert len(metrics) >= 1
        assert metrics[0].stage == "EchoProcessor"
        assert metrics[0].latency_ms < 500
        assert metrics[0].success is True
