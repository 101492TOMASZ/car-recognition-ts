import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

def setup_logger(level: str = "INFO", logfile: str = "logs/app.log") -> logging.Logger:
    global _LOGGER
    if _LOGGER:
        return _LOGGER
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger("car_app")
    logger.setLevel(level.upper())
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(logfile, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Przekierowanie logów YOLO / ultralytics do tych samych handlerów
    try:
        ul = logging.getLogger("ultralytics")
        ul.setLevel(level.upper())
        # Dodaj tylko brakujące handler'y aby nie duplikować wpisów
        existing = {type(h) for h in ul.handlers}
        for h in [fh, ch]:
            if type(h) not in existing:
                ul.addHandler(h)
        ul.propagate = False
    except Exception:
        pass
    logger.propagate = False
    _LOGGER = logger
    return logger

def get_logger() -> logging.Logger:
    return _LOGGER if _LOGGER else setup_logger()