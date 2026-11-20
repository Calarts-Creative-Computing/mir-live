# Mel Spectrogram

import audioflux as af
import numpy as np
from audioflux.type import WindowType


class MelSpectrogram:

    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        # To start, we would require a nonempty frame and a hop that leaves no gaps between frames.
        if frame_length <= 0 or not 0 < hop_length <= frame_length:
            raise ValueError("require frame_length > 0 and 0 < hop_length <= frame_length")
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Diff: audioFlux calls the hop "slide_length".
        # Create its analysis object once and reuse it; our _pending buffer below handles the streaming history.
        # Temporal defaults to a Hann window, which tapers frame edges.
        self._temporal = af.Temporal(
            frame_length=frame_length,
            slide_length=hop_length,
            window_type=WindowType.RECT,
        )
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples after a gap or before a new stream."""
        # Dropped audio leaves a gap. Clearing saved samples prevents an RMS
        # frame from joining audio from before and after that gap.
        # Call before a new stream or when block.discontinuity is True in main.py.
        # Keep the buffer between normal blocks, even while waiting for a full frame.
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return one RMS value per new complete frame, or an empty array.

        Retain overlap and any incomplete frame for the next call. Call reset()
        when input is discontinuous so frames never span missing audio.
        """
        # 1. Represent incoming audio as a one-dimensional array of 32-bit
        # floats for audioFlux. Stereo must be mixed to mono before this step.
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")

        # 2. Append new samples to the history retained from the previous call.
        # Wait for a full frame instead of adding zeros that would alter its RMS.
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self.frame_length:
            return np.empty(0, dtype=np.float32)

        # 3. Analyze all complete frames in the buffer. temporal() returns a
        # dictionary; has_rms=True requests its "rms_arr" entry, with one value
        # per frame in time order. One input block may yield zero, one, or many
        # values, depending on the block size and the buffered history.
        rms_values = self._temporal.temporal(self._pending, has_rms=True)["rms_arr"]

        # 4. Advance to the next frame start, keeping overlap and any new tail.
        # For frame_length=8 and hop_length=4, a buffer of 10 samples yields one
        # frame (indices 0..7). Drop only indices 0..3: the next frame starts at
        # index 4, so indices 4..9 must remain until more samples arrive.
        # copy() keeps just those samples without holding the full old buffer.
        consumed = len(rms_values) * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return rms_values
