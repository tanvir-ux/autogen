import asyncio
import threading
from asyncio import Future
from typing import Any, Callable, List


def _cancel_linked_future(future: Future[Any]) -> None:
    """Cancel *future* on the event loop that owns it.

    ``Future.cancel()`` schedules completion callbacks with that loop. Calling it
    from another thread raises ``RuntimeError`` when the loop is in debug mode and
    leaves those callbacks unscheduled.
    """
    if future.done():
        return

    loop = future.get_loop()
    try:
        on_owning_loop = asyncio.get_running_loop() is loop
    except RuntimeError:
        on_owning_loop = False

    if on_owning_loop:
        future.cancel()
        return

    try:
        loop.call_soon_threadsafe(future.cancel)
    except RuntimeError:
        # The owning loop is closed, so callbacks can no longer be scheduled.
        if not future.done():
            try:
                future.cancel()
            except RuntimeError:
                return


class CancellationToken:
    """A token used to cancel pending async calls"""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        """Cancel pending async calls linked to this cancellation token."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                for callback in self._callbacks:
                    callback()

    def is_cancelled(self) -> bool:
        """Check if the CancellationToken has been used"""
        with self._lock:
            return self._cancelled

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Attach a callback that will be called when cancel is invoked"""
        with self._lock:
            if self._cancelled:
                callback()
            else:
                self._callbacks.append(callback)

    def link_future(self, future: Future[Any]) -> Future[Any]:
        """Link a pending async call to a token to allow its cancellation.

        The future is cancelled on its owning event loop, including when this
        token is cancelled from another thread.
        """

        def _cancel() -> None:
            _cancel_linked_future(future)

        with self._lock:
            if self._cancelled:
                _cancel()
            else:
                self._callbacks.append(_cancel)
        return future
