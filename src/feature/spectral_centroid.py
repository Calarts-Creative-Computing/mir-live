"""Streaming spectral centroid extraction using audioFlux.

The spectral centroid is the "centre of mass" of a frame's spectrum:

    centroid = sum(f_k * s_k) / sum(s_k)

f_k is the frequency of FFT bin k in Hz and s_k is that bin's magnitude. A
frame dominated by low frequencies has a low centroid; a bright or noisy frame
has a high one. A pure 1000 Hz tone gives about 1000 Hz, and silence (where the
sum of s_k is zero) gives 0. The result is in Hz, not a normalized 0..1 value.
It is often described as the "brightness" of a sound.

Unlike RMS, this is a frequency-domain feature. audioFlux's Spectral.centroid
does not take waveform samples: it takes a spectrogram with shape
(frequency bins, time frames). So each call does two steps:

    1. Frame the buffered samples, apply a window, and take a real FFT
       (done here with NumPy, so any frame_length works, not only powers of 2).
    2. Pass the magnitude spectrogram to audioFlux's Spectral.centroid.

audioFlux reference:
https://audioflux.top/feature/spectral.html#audioflux.Spectral.centroid
"""

import audioflux as af
import numpy as np


class SpectralCentroidExtractor:
    """Extract spectral centroid (Hz) from complete frames across mono blocks.

    It follows the same streaming pattern as RMSExtractor: create it once,
    call process() with each block, and call reset() after a discontinuity.
    With the same frame_length and hop_length, it returns one value for each
    value RMSExtractor returns, so the two results line up frame by frame.

    Unlike RMSExtractor, it needs the sample_rate, because the frequency of
    each FFT bin depends on it. Create it after the source's rate is known.

    window="hann" tapers frame edges, which is the usual choice before an FFT
    because it reduces spectral leakage. window="rect" leaves samples unchanged.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        window: str = "hann",
    ):
        # Same frame/hop rule as RMSExtractor, so both extractors stay in step.
        if frame_length <= 0 or not 0 < hop_length <= frame_length:
            raise ValueError("require frame_length > 0 and 0 < hop_length <= frame_length")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window not in ("hann", "rect"):
            raise ValueError("window must be 'hann' or 'rect'")
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Window values applied to each frame before the FFT. np.hanning(N + 1)[:-1]
        # is the periodic Hann window, the usual choice for spectral analysis.
        if window == "hann":
            self._window = np.hanning(frame_length + 1)[:-1].astype(np.float32)
        else:
            self._window = np.ones(frame_length, dtype=np.float32)

        # A real FFT of frame_length samples gives frame_length // 2 + 1 bins.
        # Bin k sits at k * sample_rate / frame_length Hz (0 Hz up to Nyquist).
        # audioFlux needs these frequencies to weight each bin in the centroid.
        fre_band_arr = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate).astype(np.float32)
        self._num_bins = len(fre_band_arr)

        # Create the audioFlux object once and reuse it. Its time length changes
        # from call to call, so it is set again inside process().
        self._spectral = af.Spectral(num=self._num_bins, fre_band_arr=fre_band_arr)
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples after a gap or before a new stream."""
        # Clearing saved samples prevents a frame from joining audio from
        # before and after a gap. Call when block.discontinuity is True.
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return one centroid (Hz) per new complete frame, or an empty array.

        Retain overlap and any incomplete frame for the next call. Call reset()
        when input is discontinuous so frames never span missing audio.
        """
        # 1. Accept one-dimensional mono float32 samples, as RMSExtractor does.
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")

        # 2. Append to the retained history; wait for a full first frame rather
        # than zero-padding, which would change the spectrum.
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self.frame_length:
            return np.empty(0, dtype=np.float32)

        # 3. Cut the buffer into overlapping frames. A frame starts every
        # hop_length samples, so n_frames = (len - frame_length) // hop_length + 1.
        # Shape is (n_frames, frame_length). This is a view, not a copy.
        frames = np.lib.stride_tricks.sliding_window_view(
            self._pending, self.frame_length
        )[:: self.hop_length]
        n_frames = len(frames)

        # 4. Window each frame, take the real FFT along each frame, and keep the
        # magnitude. audioFlux expects shape (frequency bins, time frames), so
        # transpose, and make it contiguous float32 for the C library.
        spectrum = np.abs(np.fft.rfft(frames * self._window, axis=1))
        magnitude = np.ascontiguousarray(spectrum.T, dtype=np.float32)

        # 5. Tell audioFlux how many time frames this call holds, then compute
        # one centroid per frame (shape: (n_frames,)).
        self._spectral.set_time_length(n_frames)
        centroid_values = self._spectral.centroid(magnitude)

        # 6. Advance to the next frame start, exactly as RMSExtractor does:
        # drop only the samples consumed by the frames just analysed and keep
        # the overlap plus any incomplete tail. copy() releases the old buffer.
        consumed = n_frames * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return np.asarray(centroid_values, dtype=np.float32)