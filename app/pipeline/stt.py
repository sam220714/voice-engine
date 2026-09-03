import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# Must run before faster_whisper is imported anywhere in the process.
from app.dll_fix import ensure_cuda_dlls

ensure_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

from app.config import settings  # noqa: E402


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    elapsed: float
    segments: list


class WhisperStage:
    """Stage 1: speech-to-text via faster-whisper (CUDA)."""

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        if self._model is None:
            self.load()

        start = time.time()
        segments, info = self._model.transcribe(
            str(audio_path), beam_size=settings.whisper_beam_size
        )

        collected = []
        full_text = []
        for segment in segments:
            collected.append(
                {"start": segment.start, "end": segment.end, "text": segment.text}
            )
            full_text.append(segment.text)
        elapsed = time.time() - start

        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            elapsed=elapsed,
            segments=collected,
        )
