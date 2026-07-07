<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Production_Ready-7C3AED?style=for-the-badge"/>

# 🎙️ Real-Time Voice Assistant

### A production-grade, fully open-source voice pipeline with sub-500ms perceived latency

**Silero VAD → Whisper ASR → Groq LLM → Kokoro TTS**  
*No paid APIs for models · No vendor lock-in · Runs on your hardware*

[Getting Started](#-quick-start) · [Architecture](#-architecture) · [Latency Budget](#-latency-budget) · [Configuration](#-configuration) · [API Reference](#-api-reference)

---

</div>

## ✨ What Makes This Different

Most voice assistants are black boxes. This one exposes every millisecond.

| Feature | This Project | Typical Voice Assistant |
|---|---|---|
| Model transparency | Full source, local inference | Closed API |
| Latency visibility | Per-stage Prometheus metrics | None |
| Graceful degradation | 3-tier fallback per stage | Silent failure |
| Sentence streaming | TTS starts before LLM finishes | Wait for full response |
| Cost | $0 (Groq free tier) | Per-request billing |
| Privacy | Audio never leaves your machine* | Cloud processed |

*Groq processes text only — raw audio stays local.

---

## 🧠 Open-Source Model Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     Browser (WebSocket)                         │
│              AudioWorklet @ 16kHz  ←→  PCM @ 24kHz             │
└────────────────────────┬───────────────────────┬────────────────┘
                         │ audio bytes            │ audio bytes
                ▼                                 ▲
┌──────────────────────────────────────────────────────────────┐
│                      Python Pipeline                          │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Silero VAD  │───▶│ Whisper ASR │───▶│   Groq LLM      │  │
│  │   (local)   │    │   (local)   │    │  llama-3.3-70b  │  │
│  │  MIT lic.   │    │  MIT lic.   │    │  (cloud, free)  │  │
│  └─────────────┘    └─────────────┘    └────────┬────────┘  │
│                                                  │           │
│                                        ┌─────────▼────────┐  │
│                                        │   Kokoro TTS     │  │
│                                        │  82M params      │  │
│                                        │  Apache 2.0      │  │
│                                        └──────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

| Component | Model | License | VRAM | Latency |
|---|---|---|---|---|
| **VAD** | Silero VAD v4 | MIT | ~50MB | ~5ms |
| **ASR** | Whisper `base.en` / `small.en` | MIT | ~500MB / ~1GB | ~80–150ms |
| **LLM** | Llama 3.3 70B via Groq LPU | Llama 3 Community | Cloud | ~200–400ms TTFT |
| **TTS** | Kokoro 82M | Apache 2.0 | ~400MB | ~80–150ms |
| **Total** | | | **~1GB** | **~1.3s end-to-end** |

---

## ⚡ Latency Budget

> **Target:** First audio byte reaches the user within **1.3 seconds** of finishing their sentence.

```
You stop speaking
│
└── 600ms silence  (VAD_SILENCE_MS)
        │
        ▼  t = 0ms
   Whisper transcribes
        │   ~800ms on GPU (base.en)
        ▼  t = 800ms
   Groq receives text → streams tokens
        │   ~200ms to first token (LPU hardware)
        ▼  t = 1000ms
   Sentence boundary detected → Kokoro starts
        │   ~150ms synthesis
        ▼  t = 1150ms
   🔊 User hears first word of response

   [Meanwhile, Groq is still generating sentence 2]
   [Kokoro synthesizes sentence 2 while sentence 1 plays]
```

The key optimization is **sentence-boundary streaming** — TTS begins on sentence 1 while the LLM generates sentence 2, overlapping the two slowest stages.

---

## 🏗️ Architecture

### Frame-based Pipeline

Every stage is an async `FrameProcessor`. Data flows as typed frames through `asyncio.Queue` connections. No stage ever calls another directly.

```
AudioRawFrame (128 samples, 8ms)
    ↓ [accumulated into 512-sample windows]
VADFrame (SPEECH / SILENCE / END_OF_SPEECH)
    ↓ [only END_OF_SPEECH passes through]
TranscriptionFrame (text, confidence, latency_ms)
    ↓
LLMTokenFrame (per token) + LLMSentenceFrame (per sentence boundary)
    ↓ [sentence-by-sentence, not waiting for full response]
AudioOutputFrame (PCM bytes, 24kHz)
    ↓
Browser speaker
```

### Graceful Degradation

Each stage has a 3-tier fallback:

```
Stage fails once     → Retry with backoff
Stage fails 3x       → Circuit opens → Fallback service
All fallbacks fail   → Canned response + silence bytes
Pipeline never drops a user turn.
```

### Key Design Decisions

**Why sentence chunking?**  
Waiting for the full LLM response before starting TTS adds 400–800ms of silence. By detecting sentence boundaries (`.`, `!`, `?`) in the token stream and feeding each sentence to Kokoro immediately, the user hears the first word of the response ~500ms earlier.

**Why `asyncio` over threads?**  
All I/O (WebSocket, Groq API) is async. CPU-bound work (Whisper inference, Kokoro synthesis) runs in `run_in_executor()` thread pools. This keeps the event loop responsive under concurrent sessions.

**Why per-stage circuit breakers?**  
A slow Groq response shouldn't block Kokoro from finishing a previous sentence. Independent circuit breakers per stage ensure one failure doesn't cascade.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python3 --version

```

### 1. Clone and Install

```bash
git clone https://github.com/prashanthreddy31/ai-voice-assistant
cd voice-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=base.en
WHISPER_DEVICE=cuda          # or cpu
KOKORO_VOICE=af_heart
```

### 3. Run Setup Checker

```bash
python setup.py
```

This downloads all models (~1GB total), verifies your Groq key, checks VRAM budget, and confirms the full pipeline imports cleanly. Takes 2–5 minutes on first run.

### 4. Start the Server

```bash
python main.py
```

```
INFO  startup.whisper_ready
INFO  startup.silero_ready  
INFO  startup.kokoro_ready        ← ~45s first run, ~5s after
INFO  server.ready  asr=base.en  llm=llama-3.3-70b-versatile  tts=af_heart
INFO  Uvicorn running on http://0.0.0.0:8000
```

### 5. Open the Browser

```
http://localhost:8000
```

Click the microphone orb and start talking.

---

## 📁 Project Structure

```
voice-assistant/
│
├── main.py                    # FastAPI server, WebSocket endpoint, model preloading
├── config.py                  # Pydantic settings (all env var driven)
├── setup.py                   # First-run checker: downloads models, validates config
├── requirements.txt
├── .env.example
│
├── core/                      # Framework layer — no AI model code
│   ├── frames.py              # 16 typed frame classes (the data language)
│   ├── pipeline.py            # FrameProcessor, Pipeline, PipelineRunner, TimeoutProcessor
│   └── circuit_breaker.py     # CLOSED → OPEN → HALF_OPEN state machine
│
├── services/                  # AI model integrations (each is a pipeline stage)
│   ├── vad_service.py         # Silero VAD with noise calibration + energy gate
│   ├── asr_service.py         # Whisper with hallucination filter + fallback model
│   ├── llm_service.py         # Groq streaming with sentence boundary detection
│   ├── tts_service.py         # Kokoro with speaking gate + espeak fallback
│   └── transport.py           # WebSocket AudioWorklet input + PCM output
│
├── utils/
│   └── metrics.py             # Prometheus histograms, circuit breaker gauges
│
├── tests/
│   └── test_pipeline.py       # Unit tests + fault injection (15 test cases)
│
└── static/
    └── index.html             # Browser client (AudioWorklet, WebSocket, audio playback)
```

---

## ⚙️ Configuration

All settings are environment variables, readable from `.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-supported model |
| `WHISPER_MODEL` | `base.en` | `tiny.en` / `base.en` / `small.en` / `medium.en` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KOKORO_VOICE` | `af_heart` | See voice list below |
| `KOKORO_SPEED` | `1.0` | Playback speed (0.5–2.0) |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold (0.0–1.0) |
| `VAD_SILENCE_MS` | `600` | Silence duration before end-of-speech fires |
| `SERVER_PORT` | `8000` | HTTP + WebSocket port |
| `BUDGET_LLM_MS` | `2000` | Per-stage timeout before fallback activates |

### Available Voices (Kokoro)

| Voice | Gender | Accent | Character |
|---|---|---|---|
| `af_heart` | Female | American | Warm, natural *(default)* |
| `af_bella` | Female | American | Bright, energetic |
| `af_sarah` | Female | American | Clear, professional |
| `am_adam` | Male | American | Deep, authoritative |
| `am_michael` | Male | American | Neutral, balanced |
| `bf_emma` | Female | British | Refined, articulate |
| `bm_george` | Male | British | Distinguished |

### Groq Model Options

| Model | Speed | Quality | Best For |
|---|---|---|---|
| `llama-3.1-8b-instant` | Fastest (~150ms) | Good | Low-latency use cases |
| `llama-3.3-70b-versatile` | Fast (~300ms) | Excellent | General use *(recommended)* |
| `mixtral-8x7b-32768` | Fast (~250ms) | Very good | Long context |
| `gemma2-9b-it` | Fast (~200ms) | Good | Efficient alternative |

---

## 📡 API Reference

### WebSocket `/ws`

**Client → Server** (binary): Raw PCM audio, 16-bit int16, 16kHz, mono  
**Client → Server** (text): `{"type": "stop"}` — end session  

**Server → Client** (binary): Synthesized PCM audio, 16-bit int16, 24kHz, mono  
**Server → Client** (text JSON):

```json
{"type": "connected", "session_id": "...", "models": {...}}
{"type": "transcript", "text": "...", "is_final": true, "confidence": 0.97}
{"type": "error", "message": "...", "stage": "GroqLLM", "recoverable": true}
{"type": "turn_end", "reason": "turn_complete"}
```

### REST Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Browser client (HTML) |
| `/health` | GET | Liveness + model status |
| `/metrics/summary` | GET | Active sessions, turn counts |
| `/metrics` | GET | Prometheus scrape endpoint (port 9090) |

### Health Check Response

```json
{
  "status": "ok",
  "active_sessions": 2,
  "models": {
    "asr": "base.en",
    "llm": "llama-3.3-70b-versatile",
    "tts": "af_heart",
    "vad": "silero-vad"
  }
}
```

---

## 📊 Observability

Prometheus metrics are exposed at `http://localhost:9090/metrics`.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `voice_pipeline_stage_latency_ms` | Histogram | `stage`, `status` | P50/P99 per stage |
| `voice_pipeline_turn_latency_ms` | Histogram | — | End-to-end per turn |
| `voice_pipeline_timeouts_total` | Counter | `stage` | Budget breaches |
| `voice_pipeline_fallbacks_total` | Counter | `stage`, `reason` | Fallback activations |
| `voice_pipeline_circuit_breaker_open` | Gauge | `circuit` | 1=open, 0=closed |
| `voice_pipeline_active_sessions` | Gauge | — | Live WebSocket connections |

Import `grafana_dashboard.json` into Grafana for a live latency breakdown dashboard.

---

## 🧪 Testing

```bash
# Full test suite
pytest tests/ -v --timeout=30

# Specific categories
pytest tests/ -v -k "circuit"       # Circuit breaker state transitions
pytest tests/ -v -k "timeout"       # Latency budget enforcement
pytest tests/ -v -k "fault"         # Fault injection scenarios
pytest tests/ -v -k "vad"           # VAD sentence boundary detection
```

### Fault Injection

The test suite injects failures at each stage and verifies:
- `ErrorFrame` propagates correctly
- Fallback activates within budget
- Pipeline continues serving subsequent turns
- Circuit breaker opens and recovers correctly

---

## 🔧 GPU Setup (Recommended)

Running Whisper and Kokoro on GPU reduces total latency from ~3s to ~1.3s.

```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Update .env
WHISPER_DEVICE=cuda
```

### VRAM Budget (4GB GPU)

```
Whisper base.en    ~500MB  ✅
Whisper small.en   ~1.0GB  ✅  (recommended upgrade)
Kokoro TTS         ~400MB  ✅
─────────────────────────────
Total (small.en)   ~1.4GB  ✅  Well within 4GB
```

---

## 🛠️ Troubleshooting

<details>
<summary><b>VAD not detecting speech</b></summary>

1. Check microphone permissions in browser (lock icon → allow)
2. Lower threshold: `VAD_THRESHOLD=0.2` in `.env`
3. Check amplitude in terminal logs — `rms` values should be `>0.02` when speaking
4. Ensure correct input device in Windows Sound Settings

</details>


<details>
<summary><b>Kokoro loading slowly</b></summary>

First load (~45s) compiles CUDA kernels and downloads the model. Subsequent loads take ~5s. The model is cached at `~/.cache/huggingface/hub/`. This is a one-time cost per GPU.

</details>

<details>
<summary><b>WebSocket disconnects before pipeline loads</b></summary>

Models are pre-loaded at server startup. Ensure the `lifespan` function in `main.py` loads all three models before the server accepts connections. Look for `server.ready` in logs before opening the browser.

</details>


---

## 🗺️ Roadmap

- [ ] Multi-speaker support (VCTK voices via Kokoro)
- [ ] Barge-in interruption (user speaks while assistant is talking)
- [ ] Conversation memory across sessions (vector DB)
- [ ] Tool calling (weather, search, calendar)
- [ ] Mobile browser support (iOS Safari AudioWorklet)
- [ ] Docker deployment with GPU passthrough
- [ ] Faster-Whisper backend (4× speed on same hardware)

---

## 📄 License

MIT License — use freely. Bundled models retain their own licenses:

| Model | License |
|---|---|
| Silero VAD | MIT |
| OpenAI Whisper | MIT |
| Llama 3.3 (via Groq) | Llama 3 Community License |
| Kokoro TTS | Apache 2.0 |

---

<div align="center">

Built with Python · FastAPI · Silero · Whisper · Groq · Kokoro

*If this project helped you, consider starring the repo ⭐*

</div>
