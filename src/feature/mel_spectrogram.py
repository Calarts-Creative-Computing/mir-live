#mel spectrogram extraction w audioFlux:
#https://audioflux.top/transforms/bft.html

import audioflux as af
import numpy as np
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType, WindowType


class MelExtractor:

    def __init__(self, sample_rate: int, frame_length: int = 2048, hop_length: int = 512, n_mels: int = 128):
        #need a power-of-two frame and a hop that leaves no gaps
        if frame_length <= 0 or frame_length & (frame_length - 1) or not 0 < hop_length <= frame_length:
            raise ValueError("require frame_length to be a power of two and 0 < hop_length <= frame_length")
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self._bft = af.BFT(
            num=n_mels,
            radix2_exp=frame_length.bit_length() - 1,
            samplate=sample_rate,
            slide_length=hop_length,
            window_type=WindowType.HANN,
            scale_type=SpectralFilterBankScaleType.MEL,
            data_type=SpectralDataType.POWER,
        )
        #centre frequency in Hz of each mel band lo to hi
        self.band_frequencies = self._bft.get_fre_band_arr()
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples after a gap or before a new stream."""
        # Dropped audio leaves a gap. Clearing saved samples prevents a mel
        # frame from joining audio from before and after that gap.
        # Call before a new stream or when block.discontinuity is True in main.py.
        # Keep the buffer between normal blocks, even while waiting for a full frame.
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        # 1. Represent incoming audio as a one-dimensional array of 32-bit
        # floats for audioFlux. Stereo must be mixed to mono before this step.
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")

        # 2. Append new samples to the history retained from the previous call.
        # Wait for a full frame instead of adding zeros that would alter its spectrum.
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self.frame_length:
            return np.empty((0, self.n_mels), dtype=np.float32)

        # 3. Analyze all complete frames in the buffer.
        mel_values = self._bft.bft(self._pending, result_type=1).T

        # 4. Advance to the next frame start, keeping overlap and any new tail.
        # For frame_length=8 and hop_length=4, a buffer of 10 samples yields one
        # frame (indices 0..7). Drop only indices 0..3: the next frame starts at
        # index 4, so indices 4..9 must remain until more samples arrive.
        # copy() keeps just those samples without holding the full old buffer.
        consumed = len(mel_values) * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return mel_values
