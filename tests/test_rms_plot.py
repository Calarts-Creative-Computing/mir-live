"""Check PyQtGraph rendering and the separate GUI process without a desktop."""

import os
from queue import Queue
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
import numpy as np

from src.rms_plot import RMSPlot, _RMSPlotView, _run_plot


class RMSPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_view(self):
        view = _RMSPlotView()
        self.addCleanup(view.widget.close)
        return view

    def test_graph_keeps_all_values_and_marks_gaps(self):
        view = self.make_view()
        view.append([0.1, 0.2, 0.3, 0.4], [0.2, 0.5, np.nan, 0.1], 0.4)
        view.refresh()
        times, values = view.curve.getData()
        np.testing.assert_allclose(times, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(values, [0.2, 0.5, np.nan, 0.1])
        self.assertEqual(view.curve.opts["connect"], "finite")

    def test_history_is_bounded_and_large_amplitudes_stay_visible(self):
        view = self.make_view()
        view.append([0.0, 5.0, 11.0], [0.1, 2.0, 0.5], 11.0)
        view.refresh()
        np.testing.assert_allclose(view.curve.getData()[0], [5.0, 11.0])
        np.testing.assert_allclose(view.widget.viewRange()[0], [1.0, 11.0])
        self.assertGreater(view.widget.viewRange()[1][1], 2.0)

        # Once the loud samples leave the window, quiet signals fill the graph.
        view.append([22.0], [0.01], 22.0)
        view.refresh()
        np.testing.assert_allclose(view.widget.viewRange()[1], [0.0, 0.011])

        # Silence still needs a valid axis range.
        view.append([33.0], [0.0], 33.0)
        view.refresh()
        self.assertGreater(view.widget.viewRange()[1][1], 0.0)

    def test_full_queue_replaces_old_updates_without_waiting(self):
        plot = RMSPlot.__new__(RMSPlot)
        plot._closed = False
        plot._process = Mock()
        plot._process.is_alive.return_value = True
        plot._updates = Queue(maxsize=1)
        plot._updates.put(([0.1], [0.2], 0.1))
        plot.dropped_updates = 0
        plot.update([0.2], [0.3], 0.2)
        self.assertEqual(plot._updates.get_nowait(), ([0.2], [0.3], 0.2))
        self.assertEqual(plot.dropped_updates, 1)

    def test_plot_process_starts_and_exits_with_pending_updates(self):
        plot = RMSPlot()
        self.addCleanup(plot.close)
        self.assertTrue(plot.is_open)
        for i in range(200):
            plot.update([i / 100], np.array([0.25]), i / 100)
        plot.close()
        plot.close()  # Repeated cleanup should also be safe.
        self.assertFalse(plot.is_open)
        self.assertFalse(plot._process.is_alive())

    def test_missing_dependencies_give_install_instructions(self):
        with patch("src.rms_plot.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                RMSPlot()

    def test_gui_startup_error_is_reported_to_parent(self):
        status = Queue(maxsize=1)
        with patch("src.rms_plot._RMSPlotView", side_effect=RuntimeError("window failed")):
            _run_plot(Queue(), status, threading.Event())
        self.assertEqual(status.get_nowait(), ("error", "window failed"))
