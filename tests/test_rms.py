"""Check streaming RMS against known signals and independent frame calculations."""

import unittest

import numpy as np

from src.feature.rms import RMSExtractor


class RMSExtractorTests(unittest.TestCase):
    def test_constant_and_silence(self):
        extractor = RMSExtractor(frame_length=64, hop_length=64)
        samples = np.concatenate((np.full(64, -0.5), np.zeros(64)))
        np.testing.assert_allclose(extractor.process(samples), [0.5, 0.0], atol=1e-6)

    def test_sine_amplitude(self):
        samples = 0.8 * np.sin(2 * np.pi * np.arange(2048) / 64)
        values = RMSExtractor().process(samples)
        np.testing.assert_allclose(values, [0.8 / np.sqrt(2)], rtol=1e-5)

    def test_overlap_is_independent_of_input_chunk_size(self):
        samples = np.random.default_rng(42).uniform(-1, 1, 3271).astype(np.float32)
        frame_length, hop_length = 256, 73
        expected = [
            np.sqrt(np.mean(samples[start:start + frame_length] ** 2))
            for start in range(0, len(samples) - frame_length + 1, hop_length)
        ]
        for chunk_size in (1, 73, 128, 512, 1000, len(samples)):
            with self.subTest(chunk_size=chunk_size):
                extractor = RMSExtractor(frame_length, hop_length)
                actual = np.concatenate([
                    extractor.process(samples[start:start + chunk_size])
                    for start in range(0, len(samples), chunk_size)
                ])
                np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_waits_for_a_complete_frame_without_padding(self):
        extractor = RMSExtractor(frame_length=8, hop_length=4)
        self.assertEqual(extractor.process(np.ones(3)).size, 0)
        self.assertEqual(extractor.process(np.ones(4)).size, 0)
        np.testing.assert_allclose(extractor.process(np.ones(1)), [1.0])
        self.assertEqual(extractor.process(np.empty(0)).size, 0)

    def test_reset_prevents_frames_spanning_a_gap(self):
        extractor = RMSExtractor(frame_length=8, hop_length=4)
        extractor.process(np.ones(8))
        extractor.reset()
        self.assertEqual(extractor.process(np.zeros(4)).size, 0)
        np.testing.assert_allclose(extractor.process(np.zeros(4)), [0.0])

    def test_invalid_frame_and_hop_lengths(self):
        for frame_length, hop_length in ((0, 1), (8, 0), (8, -1), (8, 9)):
            with self.subTest(frame_length=frame_length, hop_length=hop_length):
                with self.assertRaises(ValueError):
                    RMSExtractor(frame_length, hop_length)


if __name__ == "__main__":
    unittest.main()
