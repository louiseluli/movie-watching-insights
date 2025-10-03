from __future__ import annotations

import atexit
import sys
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

# Global console for pretty messages
console = Console()


def _default_sink_format(record):
    # Loguru formatting with time, level, message, and source location (line:path)
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>\n"
    )


def setup_logging(level: str = "INFO", to_file: Optional[str] = None) -> None:
    """
    Configure Loguru once per session. Safe to call multiple times.
    """
    # Remove any default handlers
    logger.remove()

    # Add stdout sink
    logger.add(
        sys.stdout,
        colorize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        level=level.upper(),
        format=_default_sink_format,
    )

    # Optional: also log to a file
    if to_file:
        logger.add(
            to_file,
            rotation="20 MB",
            retention="10 days",
            compression="zip",
            level=level.upper(),
            format=_default_sink_format,
            enqueue=True,
        )

    logger.debug("Logging initialized. Level={}", level)


@dataclass
class Timer(ContextDecorator):
    """
    Simple timing context/decorator.

    Usage:
        with Timer("load_imdb_basics"):
            ... code ...
        # or
        @Timer("fuse_master").decorator
        def run(...):
            ...
    """
    label: str
    on_exit: Optional[Callable[[str, float], None]] = None

    def __post_init__(self):
        self.start_ns: Optional[int] = None
        self.elapsed_s: Optional[float] = None

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        logger.debug("⏱️  [{}] start", self.label)
        return self

    def __exit__(self, exc_type, exc, tb):
        end_ns = time.perf_counter_ns()
        self.elapsed_s = (end_ns - (self.start_ns or end_ns)) / 1e9
        if exc is None:
            logger.success("✅ [{}] done in {:.3f}s", self.label, self.elapsed_s)
        else:
            logger.exception("❌ [{}] failed after {:.3f}s", self.label, self.elapsed_s)
        if self.on_exit:
            try:
                self.on_exit(self.label, self.elapsed_s or 0.0)
            except Exception:
                logger.exception("Timer on_exit callback raised.")
        # Do not suppress exceptions
        return False

    @property
    def seconds(self) -> float:
        return self.elapsed_s or 0.0

    def decorator(self, func: Callable):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def progress_bar() -> Progress:
    """
    Rich progress bar with consistent columns.
    Use:
        with progress_bar() as prog:
            task = prog.add_task("Fetching OMDb", total=100)
            prog.update(task, advance=1)
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def stamp_utc() -> str:
    """UTC ISO timestamp for filenames & logs."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def print_kv_table(title: str, mapping: dict[str, object]) -> None:
    """
    Pretty print a small key–value table to the console.
    """
    table = Table(title=title, show_header=False, expand=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for k, v in mapping.items():
        table.add_row(str(k), str(v))
    console.print(table)


# Initialize a default logger on import (can be overridden by setup_logging)
setup_logging(level="INFO")

# Ensure a clean newline at process exit (avoids prompt gluing)
atexit.register(lambda: sys.stdout.write("\n"))
