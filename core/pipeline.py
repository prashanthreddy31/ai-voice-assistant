"""
pipeline.py — Base processor, async queue pipeline, and runner.
 
Architecture:
    Each FrameProcessor is an async coroutine that:
      1. Pulls frames from its input queue (asyncio.Queue)
      2. Processes them (possibly async I/O — model inference, API call)
      3. Pushes result frames to its output queue
 
    The Pipeline chains processors: output[n] == input[n+1]
    The PipelineRunner drives the whole thing and handles graceful shutdown.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

import structlog

from core.frames import Frame, ErrorFrame, EndFrame, StartFrame, MetricsFrame

log = structlog.get_logger(__name__)

QUEUE_SIZE = 512

class FrameProcessor(ABC):
    """
    Abstract base for every pipeline stage.
 
    Subclasses override process_frame() and optionally initialize() / cleanup().
    They call self.push_frame() to emit frames downstream.
    """

    def __init__(self, name: str = ""):
        self.name: str = name or self.__class__.__name__
        self._input_queue: asyncio.Queue = asyncio.Queue(maxsize= QUEUE_SIZE)
        self._output_queue: Optional[asyncio.Task] = None
        self._running: bool = False

    # Public API

    async def initialize(self) -> None:
        """Load models, open connections. Called once before pipeline starts."""
        pass

    async def cleanup(self) -> None:
        """Release resources. Called once after pipeline stops."""
        pass

    @abstractmethod
    async def process_frame(self, frame: Frame) -> None:
        """
        Process one frame. Call await self.push_frame(result) to emit output.
        Raise exceptions freely — the runner will catch and wrap them in ErrorFrame.
        """
        ...

    async def push_frame(self, frame: Frame) -> None:
        """Emit a frame to the next stage."""
        if self._output_queue is not None:
            await self._output_queue.put(frame)

    # Internal runner loop

    async def _run(self) -> None:
        self._running = True
        log.info("processor started", stage=self.name)
        try:
            while self._running:
                frame = await self._input_queue.get()
                try:
                    await self.process_frame(frame)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error(
                        "processor.error",
                        stage=self.name,
                        error=str(exc),
                        frame_type=type(frame).__name__,
                    )
                    await self.push_frame(
                        ErrorFrame(
                            error=str(exc),
                            stage=self.name,
                            recoverable=True,
                            original_frame=frame,
                        )
                    )
                finally:
                    self._input_queue.task_done()

                # Propogate Endframe signal to stop the loop
                if isinstance(frame, EndFrame):
                    self._running = False
        except asyncio.CancelledError:
            pass
        finally:
            log.info("processor.stopped", stage=self.name)


class PassthroughProcessor(FrameProcessor):
    """Forwards every frame unchanged. Useful for inserting metrics or logging."""

    async def process_frame(self, frame) -> None:
        await self.push_frame(frame)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Chains a list of FrameProcessors together.
 
    pipeline = Pipeline([vad, asr, context, llm, aggregator, tts, transport])
    The output queue of stage[n] is the input queue of stage[n+1].
    """

    def __init__(self, processors: List[FrameProcessor]):
        if not processors:
            raise ValueError("Pipeline requires at leasr one processor")
        
        self.processors = processors

        # wire up: processor[n].output → processor[n+1].input
        for i in range(len(processors) - 1):
            processors[i]._output_queue = processors[i + 1]._input_queue

    @property
    def source_queue(self) -> asyncio.Queue:
        """Push raw frames (e.g. from WebSocket) into this queue."""
        return self.processors[0]._input_queue
    
    async def initialize(self) -> None:
        for p in self.processors:
            log.info("pipeline.initializing", stage=p.name)
            await p.initialize()
    
    async def cleanup(self) -> None:
        for p in reversed(self.processors):
            await p.cleanup()

# ─── Pipeline Runner ──────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Starts all processor coroutines and manages the session lifecycle.
 
    Usage:
        runner = PipelineRunner(pipeline)
        await runner.run(session_id="abc-123")
    """ 

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._tasks: list[asyncio.Task] = []

    async def run(self, session_id: str = "") -> None:
        session_id = session_id or str(uuid.uuid4())
        log.info("runner.starting", session_id=session_id)

        await self.pipeline.initialize()

        # Kick off all processor loops concurrently
        self._tasks = [
            asyncio.create_task(p._run(), name=f"stage-{p.name}")
            for p in self.pipeline.processors
        ]

        # send the start signal
        await self.pipeline.source_queue.put(StartFrame(turn_id=session_id))

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.pipeline.cleanup()
            log.info("runner.stopped", session_id=session_id)

    async def push_audio(self, frame:Frame) -> None:
        """Feed an audio frame into the pipeline source."""
        await self.pipeline.source_queue.put(frame)

    async def stop(self) -> None:
        """Gracefully drain and stop the pipeline."""
        try:
            self.pipeline.source_queue.put_nowait(EndFrame(reason="session_end"))
        except asyncio.QueueFull:
            log.warning("runner.stop_source_queue_full")
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


# ─── Timeout wrapper ──────────────────────────────────────────────────────────
 
class TimeoutProcessor(FrameProcessor):
    """
    Wraps another FrameProcessor and enforces a per-frame latency budget.
 
    If `inner.process_frame()` exceeds `budget_ms`, an ErrorFrame is pushed
    and (if provided) `fallback` takes over for that frame.
    """
 
    def __init__(
        self,
        inner: FrameProcessor,
        budget_ms: int,
        fallback: Optional[FrameProcessor] = None,
    ):
        super().__init__(name=f"Timeout[{inner.name}]")
        self.inner = inner
        self.budget_ms = budget_ms
        self.fallback = fallback
 
        # Route: inner output → our output
        self.inner._output_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
 
    async def initialize(self) -> None:
        await self.inner.initialize()
        if self.fallback:
            await self.fallback.initialize()
 
    async def cleanup(self) -> None:
        await self.inner.cleanup()
        if self.fallback:
            await self.fallback.cleanup()
 
    async def process_frame(self, frame: Frame) -> None:
        start = time.monotonic()
 
        # Forward frame to inner processor
        await self.inner._input_queue.put(frame)
 
        try:
            result = await asyncio.wait_for(
                self.inner._input_queue.join(),
                timeout=self.budget_ms / 1000,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
 
            # Drain inner output to our output
            while not self.inner._output_queue.empty():
                out = self.inner._output_queue.get_nowait()
                await self.push_frame(out)
 
            await self.push_frame(
                MetricsFrame(
                    stage=self.inner.name,
                    latency_ms=elapsed_ms,
                    success=True,
                )
            )
 
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            log.warning(
                "timeout.budget_exceeded",
                stage=self.inner.name,
                budget_ms=self.budget_ms,
                elapsed_ms=round(elapsed_ms),
            )
            await self.push_frame(
                MetricsFrame(
                    stage=self.inner.name,
                    latency_ms=elapsed_ms,
                    success=False,
                    fallback_used=self.fallback is not None,
                )
            )
            if self.fallback:
                log.info("timeout.using_fallback", stage=self.fallback.name)
                await self.fallback._input_queue.put(frame)
                out = await self.fallback._input_queue.join()
                while not self.fallback._output_queue.empty():
                    out = self.fallback._output_queue.get_nowait()
                    await self.push_frame(out)
            else:
                await self.push_frame(
                    ErrorFrame(
                        error=f"Stage '{self.inner.name}' timed out after {self.budget_ms}ms",
                        stage=self.inner.name,
                        recoverable=False,
                        original_frame=frame,
                    )
                )