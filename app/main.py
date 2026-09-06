import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.pipeline.modes import DEFAULT_MODE, MODES
from app.pipeline.orchestrator import Pipeline

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    pipeline.warm_up()  # load the Whisper model once at startup, not per-request
    yield


app = FastAPI(title="Voice Engine", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".m4a"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Stage 1 only: audio in, raw transcript out."""
    audio_path = _save_upload(file)
    try:
        result = pipeline.stt.transcribe(audio_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        audio_path.unlink(missing_ok=True)

    return {
        "transcript": result.text,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "elapsed": result.elapsed,
        "segments": result.segments,
    }


@app.get("/modes")
def list_modes():
    """All available `/process` modes and what each one does."""
    return {name: mode.description for name, mode in MODES.items()}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    mode: str = Query(DEFAULT_MODE, description=f"One of: {', '.join(MODES)}"),
    instruction: Optional[str] = Form(None),
):
    """Full pipeline: audio -> Whisper transcript -> Gemma processing.

    `mode` selects a preset, tightly-constrained prompt (see GET /modes).
    `instruction` optionally overrides just the core task text within that
    mode's output-format rules.
    """
    if mode not in MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Valid modes: {', '.join(MODES)}",
        )

    audio_path = _save_upload(file)
    try:
        result = pipeline.run(audio_path, mode, instruction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        audio_path.unlink(missing_ok=True)

    return JSONResponse(result)
