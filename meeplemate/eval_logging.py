"""Structured logging for the eval CLI.

Both logging stacks are routed through stdlib logging::

    structlog loggers ─┐                                    ┌→ JSON file (eval.log)
                       ├→ stdlib logging → ProcessorFormatter
    stdlib loggers ────┘                                    └→ colored console (stderr)

This is a bit convoluted, but going through stdlib is what makes records from
langchain, httpx and friends come out in the same shape as our own, instead of a
second, differently-formatted stream.

:mod:`meeplemate.logging_config` is the API-side counterpart, using the same
bridging pattern. The two are deliberately kept separate: that one renders a single
JSON line to stdout for Cloud Run, this one renders a file *plus* a human-readable
console.

This module lives here rather than under ``meeplemate.eval`` on purpose. Its callers
import it and call :func:`configure_logging` before anything else, so that records
emitted while the rest of the package is still being imported are captured — and
importing ``meeplemate.eval`` would pull in langchain, deepeval and the tracing stack
first, defeating that. ``meeplemate/__init__.py`` is empty, so importing this costs
nothing.
"""

import logging
import sys
from pathlib import Path

import structlog


def _order_keys(*args):
    event_dict = args[2]
    key_order = ["event", "level", "logger", "timestamp"]
    other_keys = sorted(k for k in event_dict if k not in key_order)
    return {k: event_dict[k] for k in key_order + other_keys if k in event_dict}


# Processors applied to all log records before final rendering.
# Used both by structlog (as its processor chain) and by ProcessorFormatter
# as the foreign_pre_chain for standard logging records.
_shared_processors: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
    structlog.processors.UnicodeDecoder(),
    _order_keys,
]


def configure_logging(log_file: str | Path = "eval.log", level: int = logging.INFO) -> None:
    """Configure structlog and standard logging.

    Both structlog and standard loggers route through stdlib logging, which uses
    structlog's ProcessorFormatter to produce:
      - JSON output written to `log_file`
      - Colorful console output written to stderr
    """
    structlog.configure(
        processors=_shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=_shared_processors,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=_shared_processors,
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(level)
