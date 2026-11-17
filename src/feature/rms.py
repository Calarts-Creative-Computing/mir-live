"""Streaming, time-domain RMS extraction using audioFlux.

RMS (root mean square) summarizes the amplitude of a short piece of audio:

    rms = sqrt(mean(samples ** 2))

Square the samples so positive and negative values do not cancel, average the
squared values, then take the square root to return to amplitude units. For
example, a frame containing only +0.5 and -0.5 has an RMS of 0.5; silence is 0.
RMS is a useful measure of signal level, but is not a model of perceived loudness.
Here it stays in linear amplitude units; we do not convert it to decibels (dB).

This is a time-domain feature: audioFlux works directly on waveform samples,
without an FFT or spectrogram. Its Temporal API can also return energy and
zero-crossing rate, but this extractor requests only RMS.

audioFlux reference: https://audioflux.top/feature/temporal.html
"""

import audioflux as af
import numpy as np
from audioflux.type import WindowType


class RMSExtractor:
    """Extract linear RMS from complete frames across incoming mono blocks.

    A block is the chunk delivered by the audio input; a frame is the group of
    samples used for one RMS calculation. These sizes need not be the same.

    frame_length is the number of samples measured together. hop_length is the
    distance between successive frame starts. With the defaults (2048 and 512),
    adjacent frames overlap by 1536 samples. At a sample rate of sr, the frame
    spans frame_length / sr seconds, and results are spaced hop_length / sr apart.

    A rectangular window leaves sample amplitudes unchanged. Frames start at the
    beginning of the stream, with no padding at either end; an incomplete final
    frame produces no result.
    """

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
