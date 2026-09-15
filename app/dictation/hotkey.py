"""Global hotkey registration and synthetic paste, via the `keyboard` library.

`keyboard` installs a low-level Windows keyboard hook, so it can both detect
the toggle hotkey while another app has focus and send a synthetic Ctrl+V
into that same focused app. One known, unsolved limitation: per Windows
UIPI, a non-elevated process cannot send synthetic keys into a window
running elevated/as-Administrator. If the focused app is running as admin,
the paste will silently do nothing.
"""

from typing import Callable

import keyboard


def register_toggle_hotkey(combo: str, callback: Callable[[], None]) -> None:
    keyboard.add_hotkey(combo, callback)


def simulate_paste() -> None:
    keyboard.send("ctrl+v")
