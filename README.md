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

## Dictation tool

`app/dictation` is a separate, standalone background tool — a WisprFlow-style
hotkey dictation app. It is **not** part of the FastAPI server and runs as
its own process. It is batch only — there is no live/streaming
transcription; it records the whole clip, then transcribes it once.

```bash
python -m app.dictation.app
```

Press the hotkey (`ctrl+alt+space` by default) to start recording — a small
red bar appears at the bottom of the screen. Press it again to stop; the
bar turns amber while Whisper transcribes and Gemma (`clean` mode) tidies
the result, then the cleaned text is automatically pasted into whatever
text field currently has focus, in any application.

Because cleanup runs through Gemma, **Ollama must be running** with the
configured model pulled, same as the FastAPI server. Expect ~9-12 seconds
of LLM latency on top of Whisper's transcription time (roughly 1-2 seconds
for a short dictation clip, using the `small` model by default).

Known limitations:
- Per Windows UIPI, the synthetic paste can't reach a window running
  elevated/as-Administrator unless this tool is also run elevated.
- A hotkey press while a previous clip is still transcribing is ignored,
  not queued — wait for the indicator to disappear before starting again.

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
| `DICTATION_HOTKEY` | `ctrl+alt+space` | global hotkey that toggles start/stop recording in `app/dictation` |
| `DICTATION_WHISPER_MODEL_SIZE` | `small` | Whisper model size used by the dictation tool (independent of `WHISPER_MODEL_SIZE`) |
| `DICTATION_SAMPLE_RATE` | `16000` | mic recording sample rate (Hz) for the dictation tool |
| `DICTATION_TEMP_DIR` | `uploads/dictation` | scratch dir for dictation clips (deleted after each transcription) |

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
  dictation/            # standalone hotkey dictation tool (see "Dictation tool" above)
    audio.py            # mic recording (sounddevice) to a temp WAV
    hotkey.py            # global hotkey registration + synthetic paste (keyboard)
    clipboard.py         # clipboard text (win32clipboard)
    indicator.py          # floating recording/transcribing indicator (tkinter)
    controller.py         # state machine wiring hotkey -> recorder -> Whisper/Gemma -> paste
    app.py                # entry point: python -m app.dictation.app
```
