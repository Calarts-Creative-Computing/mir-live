"""Optional PyQtGraph display, running separately from audio and analysis."""

from collections import deque
import importlib.util
import multiprocessing as mp
from queue import Empty, Full
import signal

import numpy as np


class RMSPlot:
    """Send timestamped RMS results to a separate GUI process without waiting."""

    def __init__(self):
        if any(importlib.util.find_spec(name) is None for name in ("pyqtgraph", "PySide6")):
            raise RuntimeError("--plot needs PyQtGraph and PySide6; run python -m pip install -r requirements.txt")

        # Spawn gives Qt its own main thread, including on macOS.
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=32)
        self._status = context.Queue(maxsize=1)
        self._stop = context.Event()
        self._closed = False
        self.dropped_updates = 0
        self._process = context.Process(
            target=_run_plot, args=(self._updates, self._status, self._stop), daemon=True,
        )
        try:
            self._process.start()
            state, message = self._status.get(timeout=15)
            if state != "ready":
                raise RuntimeError(f"Could not open RMS plot: {message}")
        except Empty:
            self.close()
            raise RuntimeError("Could not start the Qt plot window; check that a graphical desktop is available") from None
        except BaseException:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        return not self._closed and self._process.is_alive()

    def update(self, times, values, current_time: float) -> None:
        """Queue a batch of results. If the GUI falls behind, replace old updates."""
        if not self.is_open:
            return
        batch = (times, values, current_time)
        try:
            self._updates.put_nowait(batch)
        except Full:
            # Keep audio moving instead of waiting for the GUI or building a backlog.
            self.dropped_updates += 1
            try:
                self._updates.get_nowait()
                self._updates.put_nowait(batch)
            except (Empty, Full):
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        # Do not wait for unread queue data after the GUI has exited.
        for queue in (self._updates, self._status):
            queue.cancel_join_thread()
            queue.close()


class _RMSPlotView:
    """Own the Qt widgets and ten seconds of plot history in the GUI process."""

    def __init__(self):
        import pyqtgraph as pg

        self._times = deque()
        self._values = deque()
        self._current_time = 0.0
        self._history_seconds = 10.0
        self._upper_limit = 1.0
        self._dirty = False
        self.widget = pg.PlotWidget(background="w", title="RMS")
        self.widget.setWindowTitle("Live RMS — PyQtGraph")
        self.widget.resize(1000, 450)
        self.widget.setLabel("bottom", "Audio time", units="s")
        self.widget.setLabel("left", "RMS (linear amplitude)")
        self.widget.showGrid(x=True, y=True, alpha=0.2)
        self.widget.setXRange(0, self._history_seconds, padding=0)
        self.widget.setYRange(0, self._upper_limit, padding=0)
        self.curve = self.widget.plot(pen=pg.mkPen("#2878b5", width=1), connect="finite")

    def append(self, times, values, current_time: float) -> None:
        self._times.extend(times)
        self._values.extend(values)
        self._current_time = current_time
        while self._times and self._times[0] < current_time - self._history_seconds:
            self._times.popleft()
            self._values.popleft()
        self._dirty = True

    def refresh(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        times = np.asarray(self._times)
        values = np.asarray(self._values)
        self.curve.setData(times, values)
        right = max(self._history_seconds, self._current_time)
        self.widget.setXRange(right - self._history_seconds, right, padding=0)
        finite_values = values[np.isfinite(values)]
        peak = float(finite_values.max()) if finite_values.size else 0.0
        # Fit the visible history, with 10% headroom and a nonzero range for silence.
        upper_limit = max(peak * 1.1, 1e-6)
        if upper_limit != self._upper_limit:
            self._upper_limit = upper_limit
            self.widget.setYRange(0, self._upper_limit, padding=0)


def _run_plot(updates, status, stop) -> None:
    """Qt's event loop draws on a timer; audio never calls GUI drawing methods."""
    # The audio process handles Ctrl+C and requests a clean GUI shutdown.
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # Choose the Qt binding installed by this project's requirements.
        from PySide6 import QtCore, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        view = _RMSPlotView()
        view.widget.show()

        def refresh():
            if stop.is_set():
                app.quit()
                return
            # Bound work per tick so window events remain responsive.
            for _ in range(32):
                try:
                    view.append(*updates.get_nowait())
                except Empty:
                    break
            view.refresh()

        timer = QtCore.QTimer()
        timer.timeout.connect(refresh)
        timer.start(17)  # About 60 display updates per second.
        app.processEvents()
        status.put(("ready", None))
        app.exec()
        timer.stop()
        view.widget.close()
    except Exception as error:
        try:
            status.put_nowait(("error", str(error)))
        except Full:
            pass
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
