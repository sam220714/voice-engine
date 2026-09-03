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
   and cleans it up (punctuation, grammar, filler-word removal) — or performs
   whatever instruction is passed in — via a local Ollama model.

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
- `POST /transcribe` — Stage 1 only. Upload an audio file (`file`), get back
  the raw transcript, detected language, and per-segment timestamps.
- `POST /process` — full pipeline. Upload an audio file (`file`) and an
  optional `instruction` form field to override what Stage 2 does with the
  transcript (default: clean up punctuation/grammar and remove filler words).
  Returns the raw transcript plus the LLM-processed output.

Example:

```bash
curl -X POST http://127.0.0.1:8000/process \
  -F "file=@Recording.m4a" \
  -F "instruction=Summarize this transcript in one sentence."
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
    orchestrator.py     # wires Stage 1 -> Stage 2 sequentially
```
