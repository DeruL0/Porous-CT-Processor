from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtWidgets import QProgressDialog

from core.progress import ProgressEvent, ProgressBus, StageProgressMapper


class ProgressDialogObserver:
    """
    Render progress events to QProgressDialog and process UI events.
    """

    def __init__(
        self,
        progress_dialog: QProgressDialog,
        process_events: Callable[[], None],
        *,
        stage_prefix: bool = False,
    ) -> None:
        self._progress = progress_dialog
        self._process_events = process_events
        self._stage_prefix = stage_prefix

    def on_progress(self, event: ProgressEvent) -> None:
        message = event.message
        if self._stage_prefix and event.stage:
            message = f"[{event.stage}] {message}"

        self._progress.setValue(event.percent)
        self._progress.setLabelText(message)
        self._process_events()

        if self._progress.wasCanceled():
            raise InterruptedError("Operation cancelled by user.")


class QtSignalProgressObserver:
    """
    Emit progress events via Qt signals.
    """

    def __init__(
        self,
        emit_fn: Callable[[int, str], None],
        *,
        mapper: Optional[StageProgressMapper] = None,
        stage_prefix: bool = True,
    ) -> None:
        self._emit = emit_fn
        self._mapper = mapper
        self._stage_prefix = stage_prefix

    def on_progress(self, event: ProgressEvent) -> None:
        percent = event.percent
        if self._mapper is not None and event.channel == "stage":
            percent = self._mapper.map(event.stage, event.percent)

        message = event.message
        if self._stage_prefix and event.stage and event.channel == "stage":
            message = f"[{event.stage}] {message}"
        elif event.channel == "dag":
            message = f"[DAG] {message}"

        self._emit(percent, message)


def make_progress_dialog_callback(
    progress_dialog: QProgressDialog,
    process_events: Callable[[], None],
) -> Callable[[int, str], None]:
    """
    Build a legacy (percent, message) callback backed by observer bus.
    """
    bus = ProgressBus().subscribe(
        ProgressDialogObserver(
            progress_dialog=progress_dialog,
            process_events=process_events,
            stage_prefix=False,
        )
    )
    return bus.stage_callback("task")
