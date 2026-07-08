"""
circuit_breaker.py — Circuit breaker pattern for external service calls.

States:
    CLOSED   → normal operation, all calls go through
    OPEN     → fast-fail after threshold failures, calls routed to fallback
    HALF_OPEN→ probe: one call allowed; if it succeeds → CLOSED, else → OPEN

This prevents cascading failures when a downstream service (e.g. Groq,
Kokora TTS) is slow or unavailable, keeping the pipeline responsive.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import Awaitable, Callable, Optional, TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """
    Async circuit breaker.

    Usage:
        cb = CircuitBreaker(name="Groq_llm", threshold=3, reset_s=30)

        result = await cb.call(
            primary=lambda: my_llm_call(prompt),
            fallback=lambda: simple_cached_response(prompt),
        )
    """

    def __init__(
        self,
        name: str,
        threshold: int = 3,
        reset_s: float = 30.0,
    ):
        self.name = name
        self.threshold = threshold
        self.reset_s = reset_s

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def call(
        self,
        primary: Callable[[], Awaitable[T]],
        fallback: Optional[Callable[[], Awaitable[T]]] = None,
    ) -> T:
        """
        Execute `primary`. If the circuit is OPEN, execute `fallback` instead
        (or raise if no fallback provided). Records success/failure.
        """
        async with self._lock:
            state = self._evaluate_state()

        if state == CircuitState.OPEN:
            log.warning("circuit_breaker.open_fast_fail", name=self.name)
            if fallback:
                return await fallback()
            raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = await primary()
            await self._on_success()
            return result
        except Exception as exc:
            await self._on_failure(exc)
            if fallback:
                log.info("circuit_breaker.using_fallback", name=self.name)
                return await fallback()
            raise

    def reset(self) -> None:
        """Manually reset to CLOSED (useful in tests)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        log.info("circuit_breaker.manual_reset", name=self.name)

    # ── Internal state machine ────────────────────────────────────────────────

    def _evaluate_state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._opened_at and (time.monotonic() - self._opened_at) > self.reset_s:
                self._state = CircuitState.HALF_OPEN
                log.info("circuit_breaker.half_open", name=self.name)
        return self._state

    async def _on_success(self) -> None:
        async with self._lock:
            prev = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None
            if prev != CircuitState.CLOSED:
                log.info("circuit_breaker.recovered", name=self.name)

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failure_count += 1
            log.warning(
                "circuit_breaker.failure",
                name=self.name,
                failures=self._failure_count,
                threshold=self.threshold,
                error=str(exc),
            )
            if self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                log.error(
                    "circuit_breaker.opened",
                    name=self.name,
                    reset_in_s=self.reset_s,
                )