import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from app.config import settings


class Recorder:
    """Records mic audio to a WAV file, from an explicit start() to stop()."""

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._chunks: List[np.ndarray] = []

    def _callback(self, indata, frames, time_info, status) -> None:
        self._chunks.append(indata.copy())

    def start(self) -> None:
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=settings.dictation_sample_rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> Path:
        self._stream.stop()
        self._stream.close()
        self._stream = None

        audio = np.concatenate(self._chunks, axis=0) if self._chunks else np.zeros((0, 1), dtype="int16")
        settings.dictation_temp_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.dictation_temp_dir / f"{uuid.uuid4().hex}.wav"
        wavfile.write(dest, settings.dictation_sample_rate, audio)
        return dest
