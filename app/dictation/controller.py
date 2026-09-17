import queue
import threading
from pathlib import Path

from app.config import settings
from app.dictation import clipboard, hotkey
from app.dictation.audio import Recorder
from app.dictation.indicator import FloatingIndicator
from app.pipeline.llm import GemmaStage
from app.pipeline.stt import WhisperStage

IDLE = "idle"
RECORDING = "recording"
TRANSCRIBING = "transcribing"


class DictationController:
    """State machine wiring the hotkey, recorder, Whisper/Gemma stages,
    indicator, and paste -- all cross-thread hand-off goes through one
    queue drained on the main (Tk) thread.
    """

    def __init__(self) -> None:
        self._state = IDLE
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._recorder = Recorder()
        self._stt = WhisperStage(model_size=settings.dictation_whisper_model_size)
        self._llm = GemmaStage()
        self.indicator = FloatingIndicator()

    def warm_up(self) -> None:
        self._stt.load()

    def on_hotkey(self) -> None:
        if self._state == IDLE:
            self._state = RECORDING
            self._recorder.start()
            self._queue.put(("show_recording",))
        elif self._state == RECORDING:
            self._state = TRANSCRIBING
            wav_path = self._recorder.stop()
            self._queue.put(("show_transcribing",))
            threading.Thread(
                target=self._transcribe_and_paste, args=(wav_path,), daemon=True
            ).start()
        # else: TRANSCRIBING -- ignore, a previous clip is still processing

    def _transcribe_and_paste(self, wav_path: Path) -> None:
        try:
            transcription = self._stt.transcribe(wav_path)
            cleaned = self._llm.process(transcription.text, mode="clean_fast")
        finally:
            wav_path.unlink(missing_ok=True)
        self._queue.put(("result", cleaned))

    def poll(self, root) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass
        root.after(50, self.poll, root)

    def _handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "show_recording":
            self.indicator.show_recording()
        elif kind == "show_transcribing":
            self.indicator.show_transcribing()
        elif kind == "result":
            text = message[1]
            clipboard.set_text(text)
            hotkey.simulate_paste()
            self.indicator.hide()
            self._state = IDLE
