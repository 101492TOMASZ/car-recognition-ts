import os
import sys
import tempfile
from pathlib import Path

# Centralized path helpers for portable/frozen (PyInstaller) builds.


def is_frozen() -> bool:
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


import os
import sys
import tempfile
from pathlib import Path

# Centralized path helpers for portable/frozen (PyInstaller) builds.


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False)) and hasattr(sys, '_MEIPASS')


def app_root() -> Path:
    """Return directory for reading bundled assets.
    - In frozen onefile, use PyInstaller's temporary extract dir (_MEIPASS).
    - In dev, use the project root (this file's parent).
    """
    if is_frozen() and hasattr(sys, '_MEIPASS'):
        return Path(getattr(sys, '_MEIPASS')).resolve()
    return Path(__file__).resolve().parent


def cache_dir() -> Path:
    """Per-user cache for app data. Prefer %LOCALAPPDATA%/AutoDentifier or temp.
    """
    base = None
    for env in ('LOCALAPPDATA', 'APPDATA'):
        val = os.environ.get(env)
        if val:
            base = Path(val)
            break
    if not base:
        base = Path(tempfile.gettempdir())
    d = base / 'AutoDentifier'
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return cache_dir() / 'car_history.db'


def image_dir() -> Path:
    d = cache_dir() / 'db_images'
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_file() -> Path:
    logs = cache_dir() / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    return logs / 'app.log'


def runs_dir() -> Path:
    return app_root() / 'runs'


def yolo_weights_path(default_name: str = 'yolov8s.pt') -> Path:
    # Check next to exe/script first, then CWD.
    for c in (app_root() / default_name, Path.cwd() / default_name):
        if c.is_file():
            return c
    return app_root() / default_name


def resolve_asset_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)


def samples_dirs() -> list[Path]:
    return [resolve_asset_path('samples'), resolve_asset_path('tools', 'samples')]
    candidates = [app_root() / default_name, Path.cwd() / default_name]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def resolve_asset_path(*parts: str) -> Path:
    """Resolve an asset relative to app root (works in frozen and dev)."""
    return app_root().joinpath(*parts)


def samples_dirs() -> list[Path]:
    return [resolve_asset_path('samples'), resolve_asset_path('tools', 'samples')]
