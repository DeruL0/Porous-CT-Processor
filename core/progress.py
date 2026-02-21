"""
Progress event bus and observer utilities.

Core pipeline logic emits progress events through this module, while host
environments (GUI/CLI) subscribe with their own renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence
import sys
import time


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable progress payload."""

    percent: int
    message: str
    stage: Optional[str] = None
    channel: str = "stage"  # "stage" | "dag" | custom
    timestamp: float = field(default_factory=time.time)


class ProgressObserver(Protocol):
    """Observer protocol for progress events."""

    def on_progress(self, event: ProgressEvent) -> None:
        ...


class ProgressBus:
    """
    Simple observer-style event bus for progress propagation.
    """

    def __init__(self) -> None:
        self._observers: list[Callable[[ProgressEvent], None] | ProgressObserver] = []

    def subscribe(self, observer: Callable[[ProgressEvent], None] | ProgressObserver) -> "ProgressBus":
        self._observers.append(observer)
        return self

    def unsubscribe(self, observer: Callable[[ProgressEvent], None] | ProgressObserver) -> "ProgressBus":
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
        return self

    def emit(self, event: ProgressEvent) -> None:
        for observer in tuple(self._observers):
            if hasattr(observer, "on_progress"):
                observer.on_progress(event)  # type: ignore[attr-defined]
            else:
                observer(event)  # type: ignore[misc]

    def stage_callback(self, stage: str) -> Callable[[int, str], None]:
        def callback(percent: int, message: str) -> None:
            p = max(0, min(100, int(percent)))
            self.emit(ProgressEvent(percent=p, message=message, stage=stage, channel="stage"))

        return callback

    def dag_callback(self) -> Callable[[int, str], None]:
        def callback(percent: int, message: str) -> None:
            p = max(0, min(100, int(percent)))
            self.emit(ProgressEvent(percent=p, message=message, stage=None, channel="dag"))

        return callback


class StageProgressMapper:
    """
    Map per-stage local percentage [0..100] into pipeline-global [0..100].
    """

    def __init__(self, stages: Sequence[str]) -> None:
        stage_list = list(stages)
        self._count = max(len(stage_list), 1)
        self._index = {name: idx for idx, name in enumerate(stage_list)}

    def map(self, stage: Optional[str], local_percent: int) -> int:
        if not stage or stage not in self._index:
            return max(0, min(100, int(local_percent)))

        idx = self._index[stage]
        base = int(100 * idx / self._count)
        span = max(int(100 / self._count), 1)
        local = max(0, min(100, int(local_percent)))
        return min(100, base + int(local * span / 100))


class CancelFlagObserver:
    """
    Raises InterruptedError when cancellation flag is active.
    """

    def __init__(self, is_cancelled: Callable[[], bool], message: str = "Operation cancelled by user.") -> None:
        self._is_cancelled = is_cancelled
        self._message = message

    def on_progress(self, _event: ProgressEvent) -> None:
        if self._is_cancelled():
            raise InterruptedError(self._message)


class TerminalProgressObserver:
    """
    Text renderer for CLI usage.
    """

    def __init__(self, bar_width: int = 30, stream=None) -> None:
        self.bar_width = bar_width
        self.stream = stream or sys.stdout

    def on_progress(self, event: ProgressEvent) -> None:
        if event.channel == "dag":
            self.stream.write(f"  [DAG {event.percent:3d}%] {event.message}\n")
            self.stream.flush()
            return

        stage = event.stage or "task"
        filled = int(self.bar_width * event.percent / 100)
        bar = "#" * filled + "." * (self.bar_width - filled)
        self.stream.write(f"\r  [{stage}] [{bar}] {event.percent:3d}%  {event.message:<48}")
        if event.percent >= 100:
            self.stream.write("\n")
        self.stream.flush()


__all__ = [
    "ProgressEvent",
    "ProgressObserver",
    "ProgressBus",
    "StageProgressMapper",
    "CancelFlagObserver",
    "TerminalProgressObserver",
]
