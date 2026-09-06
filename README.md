# Voice Engine

A local voice intelligence pipeline: speech in, structured/cleaned text out.
It runs entirely on your own machine — no audio or transcript leaves your
network — using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for speech-to-text and a local [Ollama](https://ollama.com) model (Gemma 4B)
for language processing.

## Pipeline

Two stages, run sequentially — Stage 1's output is fed directly into Stage 2:

1. **Stage 1 — Whisper STT** (`app/pipeline/stt.py`): transcribes audio with
   `faster-whisper` on GPU (CUDA, float16).
2. **Stage 2 — Gemma LLM** (`app/pipeline/llm.py`): takes the raw transcript
   and runs it through one of several preset modes (`app/pipeline/modes.py`)
   — clean up, summarize, extract action items, etc. — via a local Ollama
   model. Each mode uses its own tightly constrained prompt (instruction
   repeated at both ends, explicit "do NOT" rules, a worked example) since
   small models like Gemma 4B drift off-format with looser instructions.

The two stages are exposed through a FastAPI server (`app/main.py`).

## Requirements

- Python 3.11
- An NVIDIA GPU with CUDA, for Whisper
- [Ollama](https://ollama.com) installed and running locally, with a Gemma
  model pulled (e.g. `ollama pull gemma3:4b` — set `OLLAMA_MODEL` to whatever
  tag you have installed)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # adjust values as needed
```

### CUDA DLLs on Windows

The pip-installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels drop their
DLLs inside site-packages instead of on `PATH`, so Windows can't find them at
load time. `app/dll_fix.py` resolves those package locations dynamically via
`importlib` and calls `os.add_dll_directory` on their `bin/` folders before
`faster_whisper` is imported — no manual PATH setup required.

## Running

```bash
python -m uvicorn app.main:app --reload
```

The Whisper model loads once at startup (not per-request).

## API

- `GET /health` — liveness check.
- `GET /modes` — lists available `/process` modes and what each one does.
- `POST /transcribe` — Stage 1 only. Upload an audio file (`file`), get back
  the raw transcript, detected language, and per-segment timestamps.
- `POST /process` — full pipeline. Upload an audio file (`file`), optionally
  pick a `mode` query param, and optionally pass an `instruction` form field
  to override just the core task text within that mode's output-format
  rules. Returns the raw transcript plus the LLM-processed output.

  Modes:
  - `clean` (default) — fix punctuation/grammar, remove filler words.
  - `summary` — concise 2-3 sentence summary of what was discussed.
  - `action_items` — bullet list of action items (or `None`).
  - `decisions` — bullet list of decisions that were made (or `None`).
  - `key_quotes` — notable quotes, copied verbatim (or `None`).

Example:

```bash
curl -X POST "http://127.0.0.1:8000/process?mode=action_items" \
  -F "file=@Recording.m4a"
```

## Configuration

All settings are read from environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `medium` | faster-whisper model size |
| `WHISPER_DEVICE` | `cuda` | inference device |
| `WHISPER_COMPUTE_TYPE` | `float16` | ctranslate2 compute type |
| `WHISPER_BEAM_SIZE` | `5` | beam search width |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:e4b` | Ollama model tag to use for Stage 2 |
| `LLM_TEMPERATURE` | `0.1` | sampling temperature for Stage 2 |
| `UPLOAD_DIR` | `uploads` | scratch dir for uploaded audio (cleaned up per-request) |

## Project layout

```
app/
  main.py              # FastAPI app and endpoints
  config.py            # settings loaded from environment / .env
  dll_fix.py           # CUDA DLL directory fix for Windows
  pipeline/
    stt.py             # Stage 1: Whisper STT
    llm.py             # Stage 2: Gemma LLM processing
    modes.py           # /process mode presets (prompt + format rules + example per mode)
    orchestrator.py     # wires Stage 1 -> Stage 2 sequentially
```
