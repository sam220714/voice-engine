"""Windows CUDA DLL resolution for faster-whisper / ctranslate2.

faster-whisper's CUDA backend (ctranslate2) links against cuBLAS and cuDNN
but the pip-installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels drop
their DLLs inside site-packages rather than on PATH, so Windows can't find
them at load time. `os.add_dll_directory` fixes that, but it must run
*before* `faster_whisper` (or ctranslate2) is imported anywhere in the
process, and it must happen exactly once.

Paths are resolved dynamically via importlib instead of being hardcoded,
so this works regardless of which interpreter / venv / user site-packages
the packages are installed into.
"""
import importlib.util
import os
from pathlib import Path

_done = False


def ensure_cuda_dlls() -> None:
    global _done
    if _done or os.name != "nt":
        return

    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            bin_dir = Path(location) / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))

    _done = True
