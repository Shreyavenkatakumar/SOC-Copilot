"""
Centralized logging configuration for SOC Copilot Phase I.

Every module obtains its logger via ``get_logger(__name__)`` so that log
records share a consistent format and destination (console + rotating
file handler).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from utils.config import LOG_DATE_FORMAT, LOG_FILE, LOG_FORMAT, LOG_LEVEL

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Attach console and rotating-file handlers to the root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    file_handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured with the shared format,
    console output, and rotating file output.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_root_logger()
    return logging.getLogger(name)
