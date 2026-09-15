import tkinter as tk

import win32con
import win32gui

_WIDTH = 120
_HEIGHT = 28
_RECORDING_COLOR = "#e5484d"
_TRANSCRIBING_COLOR = "#f5a623"


class FloatingIndicator:
    """A small frameless always-on-top bar showing recording/transcribing state.

    Applies WS_EX_NOACTIVATE so showing/raising it never steals Windows
    keyboard focus away from the app the user is dictating into -- the
    whole "paste into the focused window" mechanism depends on focus
    staying put.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - _WIDTH) // 2
        y = screen_h - _HEIGHT - 80
        self.root.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")

        self._bar = tk.Frame(self.root, bg=_RECORDING_COLOR)
        self._bar.pack(fill="both", expand=True)

        self.root.withdraw()
        self.root.update_idletasks()
        self._apply_no_activate()

    def _apply_no_activate(self) -> None:
        hwnd = self.root.winfo_id()
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd, win32con.GWL_EXSTYLE, style | win32con.WS_EX_NOACTIVATE
        )

    def show_recording(self) -> None:
        self._bar.configure(bg=_RECORDING_COLOR)
        self.root.deiconify()

    def show_transcribing(self) -> None:
        self._bar.configure(bg=_TRANSCRIBING_COLOR)

    def hide(self) -> None:
        self.root.withdraw()
