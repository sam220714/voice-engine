import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
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


@app.post("/process")
async def process(file: UploadFile = File(...), instruction: Optional[str] = Form(None)):
    """Full pipeline: audio -> Whisper transcript -> Gemma processing."""
    audio_path = _save_upload(file)
    try:
        result = pipeline.run(audio_path, instruction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        audio_path.unlink(missing_ok=True)

    return JSONResponse(result)
