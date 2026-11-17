"""Exercise file playback and plotting without requiring audio hardware."""

import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.audio_input import AudioBlock
from src.main import main, set_terminal_printing


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "stereo.wav"
        self.audio = np.tile([0.25, 0.75], (1030, 1)).astype(np.float32)
        sf.write(self.path, self.audio, 44100, subtype="FLOAT")
        self.args = ["--file", str(self.path), "--block-size", "256", "--frame-length", "256", "--hop-length", "128"]
        set_terminal_printing(True)

    def tearDown(self):
        set_terminal_printing(True)
        self.directory.cleanup()

    def run_main(self, args):
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["monitor", *args]), redirect_stdout(out), redirect_stderr(err):
            code = main()
        return code, out.getvalue(), err.getvalue()

    def test_playback_sends_all_mono_samples_without_extra_sleep(self):
        with patch("src.main.sd.OutputStream") as output, patch("src.audio_input.time.sleep") as sleep:
            stream = output.return_value.__enter__.return_value
            stream.write.return_value = False
            code, out, err = self.run_main([*self.args, "--output-device", "3"])
        self.assertEqual(code, 0, err)
        self.assertEqual(output.call_args.kwargs["samplerate"], 44100)
        self.assertEqual(output.call_args.kwargs["device"], 3)
        self.assertEqual(output.call_args.kwargs["channels"], 1)
        samples = np.concatenate([call.args[0] for call in stream.write.call_args_list])
        np.testing.assert_allclose(samples, np.full(1030, 0.5))
        self.assertEqual(len(stream.write.call_args_list[-1].args[0]), 6)
        self.assertIn("RMS=0.50000", out)
        sleep.assert_not_called()
        output.return_value.__exit__.assert_called_once()

    def test_silent_analysis_keeps_pacing_without_opening_output(self):
        with patch("src.main.sd.OutputStream") as output, patch("src.audio_input.time.sleep") as sleep:
            code, out, err = self.run_main([*self.args, "--no-playback"])
        self.assertEqual(code, 0, err)
        self.assertIn("RMS=0.50000", out)
        output.assert_not_called()
        self.assertTrue(sleep.called)

    def test_plotting_is_off_by_default_and_can_be_explicitly_disabled(self):
        for options in ([], ["--no-plot"], ["--plot", "--no-plot"]):
            with self.subTest(options=options), patch("src.rms_plot.RMSPlot") as plot, patch("src.audio_input.time.sleep"):
                code, _, err = self.run_main([*self.args, "--no-playback", *options])
            self.assertEqual(code, 0, err)
            plot.assert_not_called()

    def test_microphone_is_never_played_back(self):
        blocks = (block for block in [AudioBlock(np.zeros(2048, dtype=np.float32), 48000, 0)])
        with patch("src.main.sd.OutputStream") as output, patch("src.main.MicrophoneInput.blocks", return_value=blocks):
            code, _, err = self.run_main([])
        self.assertEqual(code, 0, err)
        output.assert_not_called()

    def test_plot_gets_every_frame_at_its_end_time_even_when_quiet(self):
        with patch("src.main.sd.OutputStream") as output, patch("src.rms_plot.RMSPlot") as plot:
            output.return_value.__enter__.return_value.write.return_value = False
            plot.return_value.is_open = True
            code, out, err = self.run_main([*self.args, "--plot", "--quiet"])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "")
        times = [time for call in plot.return_value.update.call_args_list for time in call.args[0]]
        values = [value for call in plot.return_value.update.call_args_list for value in call.args[1]]
        np.testing.assert_allclose(times, np.arange(256, 1031, 128) / 44100)
        np.testing.assert_allclose(values, np.full(7, 0.5))
        plot.return_value.close.assert_called_once()

    def test_closing_plot_stops_playback(self):
        with patch("src.main.sd.OutputStream") as output, patch("src.rms_plot.RMSPlot") as plot:
            stream = output.return_value.__enter__.return_value
            stream.write.return_value = False
            plot.return_value.is_open = False
            code, _, err = self.run_main([*self.args, "--plot"])
        self.assertEqual(code, 0, err)
        stream.write.assert_called_once()
        output.return_value.__exit__.assert_called_once()
        plot.return_value.close.assert_called_once()

    def test_output_error_closes_plot_and_stream(self):
        with patch("src.main.sd.OutputStream") as output, patch("src.rms_plot.RMSPlot") as plot:
            output.return_value.__enter__.return_value.write.side_effect = RuntimeError("output failed")
            code, _, err = self.run_main([*self.args, "--plot"])
        self.assertEqual(code, 1)
        self.assertIn("output failed", err)
        output.return_value.__exit__.assert_called_once()
        plot.return_value.close.assert_called_once()

    def test_duration_stops_playback_at_a_block_boundary(self):
        with patch("src.main.sd.OutputStream") as output:
            output.return_value.__enter__.return_value.write.return_value = False
            code, out, err = self.run_main([*self.args, "--duration", str(512 / 44100)])
        self.assertEqual(code, 0, err)
        self.assertEqual(output.return_value.__enter__.return_value.write.call_count, 2)
        self.assertIn("Received 512 mono samples", out)

    def test_playback_underflow_is_reported(self):
        with patch("src.main.sd.OutputStream") as output:
            output.return_value.__enter__.return_value.write.return_value = True
            code, out, err = self.run_main(self.args)
        self.assertEqual(code, 0)
        self.assertEqual(err.count("Warning: playback underflow"), 1)
        self.assertIn("Playback underflows=5", out)
