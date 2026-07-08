"""
services/llm_service.py — LLM inference via Groq API.

Pipeline role:
    Receives TranscriptionFrame
        → builds message list from rolling conversation history
        → streams tokens from Groq via SSE
        → detects sentence boundaries in the token stream
        → emits LLMSentenceFrame on each boundary (TTS starts immediately)
        → emits final LLMSentenceFrame for remaining buffer after stream ends
 
Key latency design: sentence boundary detection fires TTS on the FIRST
complete sentence while Groq is still generating the second. This overlaps
LLM generation with TTS synthesis, hiding most of the LLM latency from
the user's perspective.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import AsyncIterator, Deque, List, Optional


import structlog
from groq import AsyncGroq, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

from config import get_settings
from core.circuit_breaker import CircuitBreaker
from core.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMRequestFrame,
    LLMSentenceFrame,
    LLMTokenFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
)
from core.pipeline import FrameProcessor

log = structlog.get_logger(__name__)

# Sentence boundary: period/!/? followed by space or end-of-string
# Handles "Dr. Smith" and ellipsis "..." by requiring a space after.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

SYSTEM_PROMPT = """You are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible
Your responses will be converted to speech, so follow these rules strictly:
- Keep responses to 3–4 short sentences maximum.
- Never use markdown, bullet points, numbered lists, or headers.
- Never use special characters like *, #, `, or -.
- Spell out numbers and abbreviations (say "three" not "3", "Doctor" not "Dr.").
- Use natural spoken language. Be warm and conversational.
- If you don't know something, say so briefly and move on."""


class ContextManager:
    """
    Maintains a rolling conversation history for multi-turn dialogue.
    Keeps the last `max_turns` user+assistant exchanges. Older turns are
    automatically evicted from the left of the deque, keeping the context
    window bounded regardless of session length.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 8):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: Deque[dict] = deque(maxlen=max_turns * 2)  # user+assistant pairs

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})

    def build_messages(self) -> List[dict]:
        """Build the full message list including system prompt."""
        return [
            {"role": "system", "content": self.system_prompt},
            *list(self._history),
        ]

    def clear(self) -> None:
        """Reset on new session (StartFrame)."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2


class GroqLLMProcessor(FrameProcessor):
    """
    Streams completions from Groq API.

    Input:  TranscriptionFrame
    Output: LLMTokenFrame (one per token), LLMSentenceFrame (on sentence boundary)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
    ):
        super().__init__(name="GroqLLM")
        cfg = get_settings()

        self.model = model or cfg.groq_model
        self.api_key = api_key or cfg.groq_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self._context = ContextManager(system_prompt=system_prompt)
        self._client: Optional[AsyncGroq] = None
        self._circuit_breaker = CircuitBreaker(
            name="groq_llm",
            threshold=3,
            reset_s=30)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create Groq async client. Live validation belongs in setup.py."""
        if not self.api_key:
            raise ValueError(
                "Groq_api_key is not set"
            )
        self._client = AsyncGroq(api_key=self.api_key)
        log.info("llm.groq_ready", model=self.model)
        return


    async def cleanup(self) -> None:
        """Close the Groq async client gracefully."""
        if self._client:
            await self._client.close()
            self._client = None
        log.info("llm.cleanup")

    # ── Frame Processing ────────────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame) -> None:
        # New session: reset conversation history
        if isinstance(frame, StartFrame):
            self._context.clear()
            await self.push_frame(frame)
            return
        
        # Session end: propagate and stop
        if isinstance(frame, EndFrame):
            await self.push_frame(frame)
            return
        
        # Pass non-transcription frames straight through
        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame)
            return
        
        # Skip empty transcriptions (Whisper sometimes emits these on silence)
        user_text = frame.text.strip()
        if not user_text:
            log.debug("llm.skipping_empty_transcription")
            return
        
        # Add user turn to context and build message list
        self._context.add_user(user_text)
        messages = self._context.build_messages()

        log.info("llm.generating", model=self.model, input=user_text[:80], turn=self._context.turn_count,)
        start = time.monotonic()

        full_response = ""
        sentence_index = 0
        token_buffer = ""
        first_token_logged = False

        try:
            async def _stream_and_emit() -> None:
                """Inner coroutine: streams tokens, detects sentence boundaries."""
                nonlocal full_response, sentence_index, token_buffer, first_token_logged
 
                async for token in self._stream_tokens(messages):
                    # Log time to first token (key latency metric)
                    if not first_token_logged:
                        ttft_ms = (time.monotonic() - start) * 1000
                        log.info("llm.first_token", ttft_ms=round(ttft_ms))
                        first_token_logged = True
 
                    token_buffer += token
                    full_response += token
 
                    # Emit raw token frame (used by UI to show live typing)
                    await self.push_frame(
                        LLMTokenFrame(
                            token=token,
                            is_final=False,
                            turn_id=frame.turn_id,
                        )
                    )
 
                    # ── Sentence boundary detection ────────────────────────
                    # Split on the first boundary found in the buffer.
                    parts = SENTENCE_BOUNDARY.split(token_buffer, maxsplit=1)
                    if len(parts) > 1:
                        sentence_text = parts[0].strip()
                        token_buffer = parts[1]
 
                        if sentence_text:
                            sentence_latency_ms = (time.monotonic() - start) * 1000
                            log.debug(
                                "llm.sentence_boundary_detected",
                                sentence_index=sentence_index,
                                sentence=sentence_text[:60],
                                latency_ms=round(sentence_latency_ms),
                            )
                            await self.push_frame(
                                LLMSentenceFrame(
                                    text=sentence_text,
                                    sentence_index=sentence_index,
                                    is_last=False,
                                    llm_latency_ms=sentence_latency_ms,
                                    turn_id=frame.turn_id,
                                )
                            )
                            sentence_index += 1
 
            # Run through circuit breaker — falls back to canned response if open
            await self._circuit_breaker.call(
                primary=_stream_and_emit,
                fallback=lambda: self._emit_fallback_response(frame, start),
            )
 
            # ── Flush remaining buffer ─────────────────────────────────────
            # After the stream ends, whatever is left in token_buffer is the
            # final (possibly incomplete) sentence. Emit it as is_last=True.
            remaining = token_buffer.strip()
            if remaining:
                final_latency_ms = (time.monotonic() - start) * 1000
                log.debug(
                    "llm.flushing_final_buffer",
                    text=remaining[:60],
                    latency_ms=round(final_latency_ms),
                )
                await self.push_frame(
                    LLMSentenceFrame(
                        text=remaining,
                        sentence_index=sentence_index,
                        is_last=True,
                        llm_latency_ms=final_latency_ms,
                        turn_id=frame.turn_id,
                    )
                )
 
            # Store assistant response in context for next turn
            if full_response.strip():
                self._context.add_assistant(full_response.strip())
 
            # Emit final metrics
            total_ms = (time.monotonic() - start) * 1000
            log.info(
                "llm.turn_complete",
                total_ms=round(total_ms),
                sentences=sentence_index + 1,
                chars=len(full_response),
                model=self.model,
            )
            await self.push_frame(
                MetricsFrame(
                    stage=self.name,
                    latency_ms=total_ms,
                    success=True,
                )
            )
 
        except Exception as exc:
            total_ms = (time.monotonic() - start) * 1000
            log.error("llm.error", error=str(exc), model=self.model)
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
                    latency_ms=total_ms,
                    success=False,
                )
            )
 
    # ── Groq streaming ────────────────────────────────────────────────────────
 
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _stream_tokens(self, messages: List[dict]) -> AsyncIterator[str]:
        """
        Opens a streaming chat completion with Groq and yields tokens.
 
        The @retry decorator handles 429 rate-limit errors automatically
        with exponential backoff (1s → 2s → 4s → raises after 3 attempts).
 
        Groq's SSE stream delivers chunks with this structure:
            chunk.choices[0].delta.content  → token string or None
            chunk.choices[0].finish_reason  → None until done, then "stop"
        """
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            # stop sequences that work well for voice output
            stop=None,
        )
 
        async for chunk in stream:
            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason
 
            if delta.content:
                yield delta.content
 
            if finish == "stop":
                break
 
    # ── Fallback response ─────────────────────────────────────────────────────
 
    async def _emit_fallback_response(
        self, frame: TranscriptionFrame, start: float
    ) -> None:
        """
        Emits a canned spoken response when Groq is unreachable.
        Called by circuit breaker when the circuit is OPEN.
        TTS will still synthesize this — the user hears something instead of silence.
        """
        canned = (
            "I'm having trouble connecting right now. "
            "Please try again in a moment."
        )
        log.warning("llm.circuit_open_using_fallback")
        await self.push_frame(
            LLMSentenceFrame(
                text=canned,
                sentence_index=0,
                is_last=True,
                llm_latency_ms=(time.monotonic() - start) * 1000,
                turn_id=frame.turn_id,
            )
        )
