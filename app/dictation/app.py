from app.config import settings
from app.dictation import hotkey
from app.dictation.controller import DictationController


def main() -> None:
    controller = DictationController()
    controller.warm_up()
    hotkey.register_toggle_hotkey(settings.dictation_hotkey, controller.on_hotkey)

    root = controller.indicator.root
    root.after(50, controller.poll, root)

    print(f"Dictation ready. Press {settings.dictation_hotkey} to start/stop recording.")
    root.mainloop()


if __name__ == "__main__":
    main()
