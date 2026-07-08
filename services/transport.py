"""
services/transport.py — WebSocket audio transport (input + output stages).

Handles bidirectional streaming between the browser and the pipeline:
  - Input:  receives raw PCM audio bytes from the browser over WebSocket
            → wraps in AudioRawFrame → pushes into pipeline
  - Output: receives AudioOutputFrame from TTS → sends PCM bytes to browser

Protocol (over WebSocket):
    Client → Server:  binary frames = raw PCM audio (16kHz, 16-bit, mono)
    Server → Client:  binary frames = synthesized PCM audio (22050Hz, 16-bit, mono)
    Server → Client:  text frames   = JSON control messages
                      e.g. {"type": "transcript", "text": "..."}
                           {"type": "error", "message": "..."}
                           {"type": "turn_end"}

The transport uses two separate asyncio tasks:
  1. receive_task: reads from WebSocket, decodes bytes, pushes AudioRawFrame
  2. send_task:    reads from output_queue, encodes audio, writes to WebSocket
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Optional

import numpy as np
import structlog
from fastapi import WebSocket, WebSocketDisconnect

from core.frames import (
    AudioOutputFrame,
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from core.pipeline import FrameProcessor

log = structlog.get_logger(__name__)


class WebSocketTransportInput(FrameProcessor):
    """
    Reads audio from a WebSocket connection and pushes AudioRawFrame into the pipeline.

    Expects raw 16-bit PCM bytes (int16, little-endian) at 16kHz mono.
    Converts to float32 (VAD and Whisper both expect float32 in [-1, 1]).
    """

    def __init__(self, websocket: WebSocket, sample_rate: int = 16000):
        super().__init__(name="WebSocketInput")
        self.websocket = websocket
        self.sample_rate = sample_rate
        self._running = True
        self._receive_task: Optional[asyncio.Task] = None
        self._dropped_audio_frames = 0

    async def initialize(self) -> None:
        log.info("transport.input.ready", sample_rate=self.sample_rate)

    async def process_frame(self, frame: Frame) -> None:
        """This processor SOURCES frames rather than transforming them."""
        await self.push_frame(frame)

    async def start_receiving(self, target_queue: asyncio.Queue = None) -> None:
        """
        Begin reading from WebSocket. Call this from the FastAPI endpoint handler.
        Runs until WebSocket disconnects or pipeline ends.
        """
        queue = target_queue or self._input_queue
        chunk_count = 0
        try:
            while self._running:
                message = await self.websocket.receive()

                if "bytes" in message:
                    raw_bytes = message["bytes"]
                    audio = self._decode_pcm(raw_bytes)
                    chunk_count += 1
                    if chunk_count % 1000 == 0:
                        log.debug("transport.input.audio_chunks", count=chunk_count)

                    frame = AudioRawFrame(audio=audio, sample_rate=self.sample_rate)

                    if queue is not None:
                        try:
                            queue.put_nowait(frame)
                        except asyncio.QueueFull:
                            self._dropped_audio_frames += 1
                            if self._dropped_audio_frames % 100 == 1:
                                log.warning(
                                    "transport.input.audio_dropped",
                                    dropped=self._dropped_audio_frames,
                                    queue_size=queue.qsize(),
                                )
                    else:
                        log.error("transport.input.queue_missing")
            

                elif "text" in message:
                    # Control message from client (e.g. {"type": "stop"})
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        log.info("transport.input.stop_requested")
                        if queue:
                            await queue.put(EndFrame(reason="client_stop"))
                        break

        except WebSocketDisconnect:
            log.info("transport.input.disconnected")
            if queue:
                await queue.put(EndFrame(reason="websocket_disconnect"))
        except Exception as exc:
            log.error("transport.input.error", error=str(exc))
            if queue:
                await queue.put(EndFrame(reason="transport_error"))

    def _decode_pcm(self, raw: bytes) -> np.ndarray:
        """Convert raw int16 PCM bytes to float32 array in [-1, 1]."""
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def stop(self) -> None:
        self._running = False


class WebSocketTransportOutput(FrameProcessor):
    """
    Reads AudioOutputFrame from the pipeline and sends PCM bytes to the WebSocket.

    Also handles TranscriptionFrame → sends transcript as JSON text message.
    Also handles ErrorFrame → sends error as JSON text message.
    """

    def __init__(self, websocket: WebSocket):
        super().__init__(name="WebSocketOutput")
        self.websocket = websocket
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._sender_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        # Start a dedicated sender coroutine to avoid blocking the pipeline
        self._sender_task = asyncio.create_task(self._sender_loop())
        log.info("transport.output.ready")

    async def cleanup(self) -> None:
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass

    async def process_frame(self, frame: Frame) -> None:
        if isinstance(frame, AudioOutputFrame):
            await self._send_queue.put(("audio", frame.audio))

        elif isinstance(frame, TranscriptionFrame):
            msg = json.dumps({
                "type": "transcript",
                "text": frame.text,
                "is_final": frame.is_final,
                "confidence": frame.confidence,
            })
            await self._send_queue.put(("text", msg))

        elif isinstance(frame, ErrorFrame):
            msg = json.dumps({
                "type": "error",
                "message": frame.error,
                "stage": frame.stage,
                "recoverable": frame.recoverable,
            })
            await self._send_queue.put(("text", msg))

        elif isinstance(frame, EndFrame):
            msg = json.dumps({"type": "turn_end", "reason": frame.reason})
            await self._send_queue.put(("text", msg))

        # Always propagate downstream (nothing after output, but good practice)
        await self.push_frame(frame)

    async def _sender_loop(self) -> None:
        """
        Dedicated coroutine that drains the send queue and writes to WebSocket.
        Separating this from process_frame prevents pipeline backpressure when
        the WebSocket send is slow.
        """
        try:
            while True:
                msg_type, payload = await self._send_queue.get()
                try:
                    if msg_type == "audio":
                        await self.websocket.send_bytes(payload)
                    else:
                        await self.websocket.send_text(payload)
                except Exception as exc:
                    log.error("transport.output.send_error", error=str(exc))
                finally:
                    self._send_queue.task_done()
        except asyncio.CancelledError:
            pass
