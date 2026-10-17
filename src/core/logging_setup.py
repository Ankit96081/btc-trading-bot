"""Logging configuration for long-running traders."""

import logging
import logging.handlers
import time
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(log_file: Path | None, *, level: int = logging.INFO) -> None:
    """Log to the console and, if given, to a size-rotated file."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
            )
        )
    formatter = logging.Formatter(_FORMAT)
    formatter.converter = time.gmtime  # timestamps in UTC
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=True)
