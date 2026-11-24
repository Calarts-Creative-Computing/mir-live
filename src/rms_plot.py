"""Optional PyQtGraph display, running separately from audio and analysis."""

from collections import deque
import importlib.util
import multiprocessing as mp
from queue import Empty, Full
import signal

import numpy as np


class RMSPlot:
    """Send timestamped RMS results to a separate GUI process without waiting."""

    #def __init__(self):
    def __init__(self, show_centroid: bool = False):
        if any(importlib.util.find_spec(name) is None for name in ("pyqtgraph", "PySide6")):
            raise RuntimeError("--plot needs PyQtGraph and PySide6; run python -m pip install -r requirements.txt")
        self.show_centroid = show_centroid
        # Spawn gives Qt its own main thread, including on macOS.
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=32)
        self._status = context.Queue(maxsize=1)
        self._stop = context.Event()
        self._closed = False
        self.dropped_updates = 0
        self._process = context.Process(
            target=_run_plot, args=(self._updates, self._status, self._stop, show_centroid), daemon=True,
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

    def update(self, times, values, current_time: float, centroid_values=None) -> None:
        """Queue a batch of results. If the GUI falls behind, replace old updates."""
        if not self.is_open:
            return
        batch = (times, values, current_time, centroid_values)
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

    def __init__(self, show_centroid: bool = False):
        import pyqtgraph as pg

        self._show_centroid = show_centroid
        self._times = deque()
        self._values = deque()
        self._centroids = deque()  # Same length as _times; NaN where unknown.
        self._current_time = 0.0
        self._history_seconds = 10.0
        self._upper_limit = 1.0
        self._centroid_upper_limit = 1.0
        self._dirty = False
        # self.widget = pg.PlotWidget(background="w", title="RMS")
        self.widget = pg.GraphicsLayoutWidget()
        self.widget.setBackground("w")
        self.widget.setWindowTitle("Live RMS — PyQtGraph")
        #self.widget.resize(1000, 450)
        self.widget.resize(1000, 700 if show_centroid else 450)
        #self.widget.setLabel("bottom", "Audio time", units="s")
        #self.widget.setLabel("left", "RMS (linear amplitude)")
        #self.widget.showGrid(x=True, y=True, alpha=0.2)
        #self.widget.setXRange(0, self._history_seconds, padding=0)
        #self.widget.setYRange(0, self._upper_limit, padding=0)
        #self.curve = self.widget.plot(pen=pg.mkPen("#2878b5", width=1), connect="finite")

        self.rms_plot = self.widget.addPlot(row=0, col=0, title="RMS")
        self.rms_plot.setLabel("left", "RMS (linear amplitude)")
        self.rms_plot.getAxis("left").enableAutoSIPrefix(False)
        self.rms_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rms_plot.setXRange(0, self._history_seconds, padding=0)
        self.rms_plot.setYRange(0, self._upper_limit, padding=0)
        self.rms_curve = self.rms_plot.plot(pen=pg.mkPen("#2878b5", width=1), connect="finite")

        # Optional bottom plot: spectral centroid, sharing the time axis, so
        # setting the x range on the RMS plot moves both.
        self.centroid_plot = None
        self.centroid_curve = None
        if show_centroid:
            self.centroid_plot = self.widget.addPlot(row=1, col=0, title="Spectral centroid")
            self.centroid_plot.setLabel("left", "Centroid", units="Hz")
            self.centroid_plot.showGrid(x=True, y=True, alpha=0.2)
            self.centroid_plot.setXLink(self.rms_plot)
            self.centroid_plot.setYRange(0, self._centroid_upper_limit, padding=0)
            self.centroid_curve = self.centroid_plot.plot(pen=pg.mkPen("#d9534f", width=1), connect="finite")
            self.centroid_plot.setLabel("bottom", "Audio time", units="s")
        else:
            self.rms_plot.setLabel("bottom", "Audio time", units="s")

    def append(self, times, values, current_time: float, centroid_values=None) -> None:
        self._times.extend(times)
        self._values.extend(values)
        if centroid_values is None:
            self._centroids.extend([float("nan")] * len(times))
        else:
            self._centroids.extend(centroid_values)
        self._current_time = current_time
        while self._times and self._times[0] < current_time - self._history_seconds:
            self._times.popleft()
            self._values.popleft()
            self._centroids.popleft()
        self._dirty = True

    def refresh(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        times = np.asarray(self._times)
        values = np.asarray(self._values)
        #self.curve.setData(times, values)
        self.rms_curve.setData(times, values)
        right = max(self._history_seconds, self._current_time)
        #self.widget.setXRange(right - self._history_seconds, right, padding=0)
        self.rms_plot.setXRange(right - self._history_seconds, right, padding=0)
        finite_values = values[np.isfinite(values)]
        peak = float(finite_values.max()) if finite_values.size else 0.0
        # Fit the visible history, with 10% headroom and a nonzero range for silence.
        upper_limit = max(peak * 1.1, 1e-6)
        if upper_limit != self._upper_limit:
            self._upper_limit = upper_limit
            #self.widget.setYRange(0, self._upper_limit, padding=0)
            self.rms_plot.setYRange(0, self._upper_limit, padding=0)

        if self.centroid_plot is not None:
            centroids = np.asarray(self._centroids)
            self.centroid_curve.setData(times, centroids)
            finite_centroids = centroids[np.isfinite(centroids)]
            centroid_peak = float(finite_centroids.max()) if finite_centroids.size else 0.0
            centroid_upper = max(centroid_peak * 1.1, 1.0)
            if centroid_upper != self._centroid_upper_limit:
                self._centroid_upper_limit = centroid_upper
                self.centroid_plot.setYRange(0, self._centroid_upper_limit, padding=0)


def _run_plot(updates, status, stop, show_centroid=False) -> None:
    """Qt's event loop draws on a timer; audio never calls GUI drawing methods."""
    # The audio process handles Ctrl+C and requests a clean GUI shutdown.
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # Choose the Qt binding installed by this project's requirements.
        from PySide6 import QtCore, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        view = _RMSPlotView(show_centroid)
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
