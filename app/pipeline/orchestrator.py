from pathlib import Path
from typing import Optional, Union

from app.pipeline.llm import GemmaStage
from app.pipeline.stt import WhisperStage


class Pipeline:
    """Sequential two-stage pipeline: Whisper STT -> Gemma LLM."""

    def __init__(self) -> None:
        self.stt = WhisperStage()
        self.llm = GemmaStage()

    def warm_up(self) -> None:
        self.stt.load()

    def run(self, audio_path: Union[str, Path], instruction: Optional[str] = None) -> dict:
        transcription = self.stt.transcribe(audio_path)
        llm_output = self.llm.process(transcription.text, instruction)

        return {
            "transcript": transcription.text,
            "language": transcription.language,
            "language_probability": transcription.language_probability,
            "duration": transcription.duration,
            "stt_elapsed": transcription.elapsed,
            "segments": transcription.segments,
            "llm_output": llm_output,
        }
