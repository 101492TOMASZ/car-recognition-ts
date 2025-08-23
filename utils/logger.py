import logging
import sys


def get_logger(name: str = __name__, level: int = logging.INFO):
    """Create or return a configured logger that writes to stdout.

    Keeps handlers idempotent so repeated imports don't duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
