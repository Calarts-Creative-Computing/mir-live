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


class OnsetDetector:
    """Detect onsets frame-by-frame from streaming mono audio using audioFlux."""

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 48000,
        num_mels: int = 128,
        window_frames: int = 32,
        lookahead_frames: int = 1,
        novelty_type: NoveltyType = NoveltyType.FLUX,
        novelty_param: af.NoveltyParam | None = None,
    ):
        # audioFlux BFT requires a power-of-2 frame length (2 ** radix2_exp)
        # and a hop that leaves no gaps between frames.
        if frame_length <= 0 or (frame_length & (frame_length - 1)) != 0:
            raise ValueError("frame_length must be a positive power of 2 (e.g. 1024, 2048)")
        if not 0 < hop_length <= frame_length:
            raise ValueError("require 0 < hop_length <= frame_length")
        if window_frames < 4 or not 0 <= lookahead_frames < window_frames - 1:
            raise ValueError("require window_frames >= 4 and 0 <= lookahead_frames < window_frames - 1")

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.radix2_exp = frame_length.bit_length() - 1
        self.num_mels = num_mels
        self.window_frames = window_frames
        self.eval_idx = window_frames - 1 - lookahead_frames
        self.novelty_type = novelty_type
        self.novelty_param = novelty_param or af.NoveltyParam(1, 2, 0, 1, 0, 0, 0, 1)

        self._sample_rate: int | None = None
        self._bft: af.BFT | None = None
        self._onset: af.Onset | None = None

        self._init_audioflux(sample_rate)
        self.reset()

    def _init_audioflux(self, sample_rate: int) -> None:
        """Create reusable audioFlux BFT and Onset objects for a given sample rate."""
        self._sample_rate = int(sample_rate)
        # audioFlux calls the hop "slide_length".
        # is_padding=False avoids zero-padding short buffers.
        self._bft = af.BFT(
            num=self.num_mels,
            samplate=self._sample_rate,
            radix2_exp=self.radix2_exp,
            slide_length=self.hop_length,
            scale_type=SpectralFilterBankScaleType.MEL,
            data_type=SpectralDataType.POWER,
            is_padding=False,
        )
        self._onset = af.Onset(
            time_length=self.window_frames,
            fre_length=self.num_mels,
            slide_length=self.hop_length,
            samplate=self._sample_rate,
            novelty_type=self.novelty_type,
        )

    def reset(self) -> None:
        """Discard buffered samples and spectrogram history after a gap or before a new stream."""
        # Dropped audio leaves a gap. Clearing saved samples and the rolling
        # spectrogram prevents a frame or onset peak from spanning missing audio.
        # Call before a new stream or when block.discontinuity is True in main.py.
        self._pending = np.empty(0, dtype=np.float32)
        self._spec_window: np.ndarray | None = None

    def process(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (onset_strength, onset_flags) for each new complete frame.

        - onset_strength: float32 array of audioFlux evn_arr values per frame.
        - onset_flags: bool array (True where audioFlux picked an onset peak).

        Retains overlap and any incomplete frame for the next call. Call reset()
        when input is discontinuous so frames never span missing audio.
        """
        # Rebuild audioFlux objects automatically if the stream's sample rate differs.
        if sample_rate is not None and sample_rate != self._sample_rate:
            self._init_audioflux(sample_rate)
            self._spec_window = None

        # 1. Represent incoming audio as a one-dimensional array of 32-bit
        # floats for audioFlux. Stereo must be mixed to mono before this step.
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")

        # 2. Append new samples to the history retained from the previous call.
        # Wait for a full frame instead of adding zeros that would alter the spectrum.
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self.frame_length:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

        # 3. Compute Mel power spectrogram columns (in dB) for all complete frames
        # currently in _pending.
        spec_arr = self._bft.bft(np.ascontiguousarray(self._pending, dtype=np.float32))
        spec_db = af.utils.power_to_db(np.abs(spec_arr)).astype(np.float32)
        n_frames = spec_db.shape[1]

        env_values = np.empty(n_frames, dtype=np.float32)
        onset_flags = np.empty(n_frames, dtype=bool)

        # 4. Slide each new spectrogram column into the rolling window and run
        # audioFlux Onset, reading the result at eval_idx (1 hop lookahead by default).
        for i in range(n_frames):
            col = spec_db[:, i]
            if self._spec_window is None:
                # Repeat the first frame across the window to prevent a false startup spike.
                self._spec_window = np.repeat(col[:, None], self.window_frames, axis=1)
            else:
                self._spec_window[:, :-1] = self._spec_window[:, 1:]
                self._spec_window[:, -1] = col

            point_arr, evn_arr, _time_arr, _value_arr = self._onset.onset(
                np.ascontiguousarray(self._spec_window, dtype=np.float32),
                novelty_param=self.novelty_param,
            )
            env_values[i] = float(evn_arr[self.eval_idx])
            onset_flags[i] = bool(np.any(point_arr == self.eval_idx))

        # 5. Advance to the next frame start, keeping overlap and any new tail.
        # copy() keeps just those samples without holding the full old buffer.
        consumed = n_frames * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return env_values, onset_flags
