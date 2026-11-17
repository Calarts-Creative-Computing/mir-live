"""Streaming, time-domain RMS extraction using audioFlux."""

import audioflux as af
import numpy as np
from audioflux.type import WindowType


class RMSExtractor:
    """Extract linear RMS from complete frames across incoming mono blocks.

    Frame and hop lengths are in samples. A rectangular window preserves the
    signal's amplitude. Frames are left-aligned, with no padding at either end.
    """

    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        if frame_length <= 0 or not 0 < hop_length <= frame_length:
            raise ValueError("require frame_length > 0 and 0 < hop_length <= frame_length")
        self.frame_length = frame_length
        self.hop_length = hop_length
        self._temporal = af.Temporal(
            frame_length=frame_length,
            slide_length=hop_length,
            window_type=WindowType.RECT,
        )
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples after a gap or before a new stream."""
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return one RMS value per new complete frame, or an empty array.

        Retain overlap and any incomplete frame for the next call. Call reset()
        when input is discontinuous so frames never span missing audio.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self.frame_length:
            return np.empty(0, dtype=np.float32)

        rms_values = self._temporal.temporal(self._pending, has_rms=True)["rms_arr"]
        consumed = len(rms_values) * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return rms_values
