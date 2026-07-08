"""
main.py — FastAPI server: WebSocket endpoint + pipeline orchestration.

Endpoints:
    GET  /         → serves the browser UI
    GET  /health   → health check (liveness + model status)
    GET  /metrics  → Prometheus metrics (human-readable summary)
    WS   /ws       → WebSocket audio streaming endpoint

Pipeline per session:
    WebSocketInput → VAD → ASR → LLM → TTS → WebSocketOutput

Each WebSocket connection gets its own isolated pipeline with independent
state, context history, circuit breakers, and metrics labels.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict
from dataclasses import dataclass, field

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from core.pipeline import Pipeline, PipelineRunner
from services.asr_service import WhisperASRProcessor
from services.llm_service import GroqLLMProcessor
from services.transport import WebSocketTransportInput, WebSocketTransportOutput
from services.tts_service import KokoroTTSProcessor
from services.vad_service import SileroVADProcessor
from utils.metrics import (
    ACTIVE_SESSIONS,
    MetricsSinkProcessor,
    SESSION_COUNTER,
    record_turn_complete,
    start_metrics_server,
)

log = structlog.get_logger(__name__)
cfg = get_settings()

# Track active sessions for health endpoint
active_sessions: Dict[str, dict] = {}


# ─── App lifecycle ────────────────────────────────────────────────────────────

# Global pre-loaded model references
_whisper_model = None
_kokoro_pipeline = None
_silero_model = None


def resolve_whisper_device() -> str:
    """Choose a usable Whisper device from config, falling back cleanly."""
    if cfg.whisper_device != "auto":
        return cfg.whisper_device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up shared resources. Shutdown: clean up."""
    global _whisper_model, _kokoro_pipeline, _silero_model

    log.info("server.starting", host=cfg.server_host, port=cfg.server_port)

    # Start Prometheus metrics scrape endpoint on a separate port
    start_metrics_server()
    
    # Pre-load all models ONCE at startup, not per WebSocket connection
    whisper_device = resolve_whisper_device()
    log.info("startup.loading_whisper", model=cfg.whisper_model, device=whisper_device)
    import whisper
    loop = asyncio.get_event_loop()
    _whisper_model = await loop.run_in_executor(
        None, lambda: whisper.load_model(cfg.whisper_model, device=whisper_device)
    )
    log.info("startup.whisper_ready")

    log.info("startup.loading_silero")
    import torch
    _silero_model, _ = await loop.run_in_executor(
        None, lambda: torch.hub.load("snakers4/silero-vad", "silero_vad", verbose=False)
    )
    log.info("startup.silero_ready")

    log.info("startup.loading_kokoro")
    from kokoro import KPipeline
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    _kokoro_pipeline = await loop.run_in_executor(
        None, lambda: KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    )
    log.info("startup.kokoro_ready")

    log.info("server.ready", asr=cfg.whisper_model, llm=cfg.groq_model, tts=cfg.kokoro_voice)
    yield
    log.info("server.shutdown")


app = FastAPI(
    title="AI Voice Assistant",
    version="1.0.0",
    description="Real-time voice pipeline: Silero VAD → Whisper ASR → Groq LLM → Kokoro TTS",
    lifespan=lifespan,
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the browser client."""
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    """Liveness + readiness check."""
    return JSONResponse({
        "status": "ok",
        "active_sessions": len(active_sessions),
        "models": {
            "asr": cfg.whisper_model,
            "llm": cfg.groq_model,
            "tts": cfg.kokoro_voice,
            "vad": "silero-vad",
        },
        "config": {
            "budget_asr_ms": cfg.budget_asr_ms,
            "budget_llm_ms": cfg.budget_llm_ms,
            "budget_tts_ms": cfg.budget_tts_ms,
        },
    })


@app.get("/metrics/summary")
async def metrics_summary():
    """Human-readable summary of pipeline metrics."""
    return JSONResponse({
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "id": sid[:8],
                "started_at": s["started_at"],
                "turns": s["turns"],
            }
            for sid, s in active_sessions.items()
        ],
        "prometheus": f"http://localhost:{cfg.metrics_port}/metrics",
    })


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket handler.

    Protocol:
        1. Accept connection
        2. Build pipeline (VAD → ASR → LLM → TTS)
        3. Run input receiver and pipeline concurrently
        4. On disconnect: drain pipeline, clean up
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()

    SESSION_COUNTER.inc()
    ACTIVE_SESSIONS.inc()
    active_sessions[session_id] = {
        "started_at": time.time(),
        "turns": 0,
    }
    log.info("session.started", session_id=session_id[:8])

    # Send session handshake to client
    await websocket.send_text(json.dumps({
        "type": "connected",
        "session_id": session_id,
        "models": {
            "asr": cfg.whisper_model,
            "llm": cfg.groq_model,
            "tts": cfg.kokoro_voice,
        },
    }))

    # Build pipeline stages
    transport_in = WebSocketTransportInput(websocket, sample_rate=16000)
    vad = SileroVADProcessor(
        threshold=cfg.vad_threshold,
        silence_ms=cfg.vad_silence_ms,
        preloaded_model=_silero_model,
    )
    asr = WhisperASRProcessor(
        model_name=cfg.whisper_model,
        device=resolve_whisper_device(),
        preloaded_model=_whisper_model,
    )
    llm = GroqLLMProcessor(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
    )
    tts = KokoroTTSProcessor(
        voice=cfg.kokoro_voice,
        speed=cfg.kokoro_speed,
        preloaded_pipeline=_kokoro_pipeline,
    )
    metrics = MetricsSinkProcessor()
    transport_out = WebSocketTransportOutput(websocket)

    # Wire pipeline: input → VAD → ASR → LLM → TTS → output
    pipeline = Pipeline([transport_in, vad, asr, llm, tts, metrics, transport_out])
    runner = PipelineRunner(pipeline)

    turn_start_time: float = 0.0

    runner_task = None
    receiver_task = None

    try:
        runner_task = asyncio.create_task(runner.run(session_id=session_id))
        receiver_task = asyncio.create_task(transport_in.start_receiving())

        done, pending = await asyncio.wait(
            {runner_task, receiver_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            task.result()
        await receiver_task
        await runner_task

    except WebSocketDisconnect:
        log.info("session.disconnect", session_id=session_id[:8])
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log.error("session.error", session_id=session_id[:8], error=str(exc))
        try:
            await websocket.send_text(json.dumps({
                "type": "fatal_error",
                "message": str(exc),
            }))
        except Exception:
            pass
    finally:
        transport_in.stop()
        if receiver_task and not receiver_task.done():
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
        await runner.stop()
        ACTIVE_SESSIONS.dec()
        active_sessions.pop(session_id, None)
        log.info("session.ended", session_id=session_id[:8])


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    uvicorn.run(
        "main:app",
        host=cfg.server_host,
        port=cfg.server_port,
        log_level=cfg.log_level,
        reload=True,          
        ws_ping_interval=20,  
        ws_ping_timeout=30,
    )
