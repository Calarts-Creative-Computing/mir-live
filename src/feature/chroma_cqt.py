#chroma cqt extraction with audioFLUX:
#https://audioflux.top/transforms/cqt.html

import math

import audioflux as af
import numpy as np
from audioflux.utils import note_to_hz


class ChromaExtractor:
    #have to enforce a frame length unlike rms/mel, otherwise don't get lowest bands

    #useful pitch class names. probably helpful for the visual team?
    PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    def __init__(self, sample_rate: int, hop_length: int = 512):
        #power-of-two hops from 64 up
        if hop_length < 64 or hop_length & (hop_length - 1):
            raise ValueError("require hop_length to be a power of two, at least 64")
        self.hop_length = hop_length

        # Lowest band's window: Q = 1 / (2 ** (1/12) - 1), about 16.8 cycles of C1,
        # rounded up to a power of two.
        #this is a lil complicated and required a bit of help from my friend Claude
        q = 1 / (2 ** (1 / 12) - 1)
        self.frame_length = 2 ** math.ceil(math.log2(q * sample_rate / note_to_hz("C1")))

        # Frame t is centered on t * hop_length. The first frame with no padding
        # is _first; the buffer needs _needed samples to finish it.
        self._first = math.ceil(self.frame_length / 2 / hop_length)
        self._needed = self._first * hop_length + self.frame_length // 2

        self._cqt = af.CQT(num=84, samplate=sample_rate, slide_length=hop_length)
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples after a gap or before a new stream."""
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        # 1. Represent incoming audio as a one-dimensional array of 32-bit
        # floats for audioFlux. Stereo must be mixed to mono before this step.
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array")

        # 2. append to the history and wait until a complete frame
        self._pending = np.concatenate((self._pending, samples))
        if len(self._pending) < self._needed:
            return np.empty((0, 12), dtype=np.float32)

        # 3. Analyze the buffer.
        #pass the whole cqt to chroma but only keep the complete frames and transpose
        frames = (len(self._pending) - self._needed) // self.hop_length + 1
        cqt_values = self._cqt.cqt(self._pending)
        chroma_values = self._cqt.chroma(cqt_values)[:, self._first:self._first + frames].T

        # 4. Drop one hop per frame, keep overlap for next frame
        consumed = len(chroma_values) * self.hop_length
        self._pending = self._pending[consumed:].copy()
        return chroma_values