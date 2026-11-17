"""Microphone and paced file sources yielding the same mono audio blocks."""

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import time
from typing import Iterator

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "sample_audio"


@dataclass(frozen=True)
class AudioBlock:
    """Samples and their position in the source; time is relative to its start.

    A discontinuity means samples were lost before/in this block. Downstream
    analysis should reset overlap and temporal history when it is True.
    Microphone hardware overflows can lose an unknown number of samples, so
    start_sample is a delivered-sample position, not a hardware clock.
    """

    samples: np.ndarray  # One-dimensional, contiguous float32; not windowed.
    sample_rate: int
    start_sample: int
    discontinuity: bool = False

    @property
    def time_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return (self.start_sample + len(self.samples)) / self.sample_rate


def resolve_audio_file(filename: str | Path) -> Path:
    """Resolve bare names inside sample_audio, independently of working directory.

    Explicit absolute paths and project-root paths such as sample_audio/foo.wav
    are also accepted.
    """
    path = Path(filename).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0] == "sample_audio":
            path = SAMPLE_AUDIO_DIR.parent / path
        else:
            path = SAMPLE_AUDIO_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


class FileInput:
    """Read a file incrementally, downmixing channels and pacing by sample count.

    Uses the file's native sample rate. The last block may be shorter than
    block_size; no padding is added. This source does not play through speakers.
    """

    def __init__(self, filename: str | Path, block_size: int = 512, *, realtime: bool = True):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.path = resolve_audio_file(filename)
        self.block_size = block_size
        # Disable sleep-based pacing when an audio output stream supplies the clock.
        self.realtime = realtime

    def blocks(self) -> Iterator[AudioBlock]:
        with sf.SoundFile(self.path) as audio:
            sample_rate = audio.samplerate
            start_sample = 0
            started_at = time.monotonic()
            while True:
                data = audio.read(self.block_size, dtype="float32", always_2d=True)
                if not len(data):
                    return
                samples = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
                block = AudioBlock(samples, sample_rate, start_sample)
                # Release a block when its last sample would have arrived live.
                if self.realtime:
                    delay = started_at + block.end_seconds - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield block
                start_sample += len(samples)


class MicrophoneInput:
    """Capture independently of the consumer, keeping at most queue_blocks.

    If analysis falls behind, drop the oldest queued block to retain recent
    audio. Python's callback/queue approach is for soft real-time analysis;
    it does not guarantee hard real-time audio deadlines.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int | None = None,
        block_size: int = 512,
        channels: int = 1,
        queue_blocks: int = 8,
    ):
        if block_size <= 0 or channels <= 0 or queue_blocks <= 0:
            raise ValueError("block_size, channels, and queue_blocks must be positive")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.queue_blocks = queue_blocks
        self.dropped_blocks = 0
        self.input_overflows = 0

    def blocks(self) -> Iterator[AudioBlock]:
        info = sd.query_devices(self.device, "input")
        sample_rate = self.sample_rate or int(info["default_samplerate"])
        sd.check_input_settings(
            device=self.device, channels=self.channels,
            dtype="float32", samplerate=sample_rate,
        )
        pending = Queue(maxsize=self.queue_blocks)
        start_sample = 0
        expected_sample = 0
        self.dropped_blocks = 0
        self.input_overflows = 0

        def capture(indata, frames, timing, status):
            nonlocal start_sample
            if status.input_overflow:
                self.input_overflows += 1
            # Own the buffer: PortAudio reuses indata after the callback returns.
            item = (indata.copy(), start_sample, bool(status))
            start_sample += frames
            try:
                pending.put_nowait(item)
            except Full:
                try:
                    pending.get_nowait()
                    self.dropped_blocks += 1
                except Empty:
                    pass  # The consumer freed a slot in the meantime.
                pending.put_nowait(item)

        with sd.InputStream(
            device=self.device, samplerate=sample_rate, channels=self.channels,
            blocksize=self.block_size, dtype="float32", latency="low",
            callback=capture,
        ) as stream:
            while True:
                try:
                    data, offset, status = pending.get(timeout=0.25)
                except Empty:
                    if not stream.active:
                        raise RuntimeError("Microphone stream stopped unexpectedly")
                    continue
                samples = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
                discontinuity = status or offset != expected_sample
                expected_sample = offset + len(samples)
                yield AudioBlock(samples, sample_rate, offset, discontinuity)
