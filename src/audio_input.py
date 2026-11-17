"""Read audio from a file or microphone, one block at a time.

Both sources yield AudioBlock objects, so main.py can process them the same way.
By default, a block contains 512 samples per channel. This is a delivery size,
not the RMS frame size: the extractor collects samples into its own frames.
"""

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
    """One chunk of mono audio, plus information about when it occurred.

    discontinuity flags a gap or input warning. main.py then resets the RMS
    buffer so an analysis frame does not join audio across a possible gap.
    If the audio system loses samples, their count may be unknown; our sample
    positions cannot account for that missing time exactly.
    """

    samples: np.ndarray  # One-dimensional, contiguous float32; not windowed.
    sample_rate: int  # Samples per second, e.g. 48000.
    start_sample: int  # Index of this block's first sample, counted from zero.
    discontinuity: bool = False

    @property
    def time_seconds(self) -> float:
        # Convert sample position to seconds: sample 48000 at 48000 Hz is 1 s.
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        # Time just after the last sample: a block starting at sample 0 with
        # 512 samples ends at 512 / sample_rate seconds.
        return (self.start_sample + len(self.samples)) / self.sample_rate


def resolve_audio_file(filename: str | Path) -> Path:
    """Resolve bare names inside sample_audio, independently of working directory.

    Examples:
        "foo.wav"              -> <project>/sample_audio/foo.wav
        "sample_audio/foo.wav" -> <project>/sample_audio/foo.wav
        "/any/folder/foo.wav"  -> use that location directly

    Other relative paths are also resolved inside sample_audio, not the folder
    from which the command was run. An absolute path can point anywhere.
    """
    path = Path(filename).expanduser()  # Expand ~ to the user's home folder.

    # Use this project's sample_audio folder, even when run from another folder.
    if not path.is_absolute():
        if path.parts and path.parts[0] == "sample_audio":
            path = SAMPLE_AUDIO_DIR.parent / path
        else:
            path = SAMPLE_AUDIO_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


class FileInput:
    """Read a file in small chunks and average its channels into mono.

    Uses the file's native sample rate. The last block may be shorter than
    block_size; no padding is added. This source does not play through speakers.

    Reading follows main.py's requests: read A -> process A -> read B.
    While A is being processed, B stays on disk. There is no waiting queue to
    fill up, so slow processing does not make this reader drop file samples.
    Speaker playback can still run out of audio and have a gap; main.py handles
    playback separately and reports those gaps as playback underflows.
    """

    def __init__(self, filename: str | Path, block_size: int = 512, *, realtime: bool = True):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.path = resolve_audio_file(filename)
        self.block_size = block_size
        # realtime=True adds timed waits to imitate live input. main.py uses
        # this for silent analysis; with playback, output writes set the pace.
        self.realtime = realtime

    # Iterator[AudioBlock]: yield one AudioBlock at a time for a loop to consume.
    def blocks(self) -> Iterator[AudioBlock]:
        """Yield one block per request, closing the file when iteration ends."""

        # 1. Open the file and keep its original sample rate; no resampling.
        with sf.SoundFile(self.path) as audio:
            sample_rate = audio.samplerate
            start_sample = 0

            # A monotonic clock measures elapsed time without wall-clock jumps.
            started_at = time.monotonic()

            while True:
                # 2. Read the next block, normally 512 samples per channel.
                # Shape is (samples, channels), even for mono files; an empty
                # result means we reached the end. The last block can be shorter.
                data = audio.read(self.block_size, dtype="float32", always_2d=True)
                if not len(data):
                    return

                # 3. Average channels at each sample: [0.2, 0.6] becomes 0.4.
                # axis=1 means average across channels, not across time.
                # Store the mono result as consecutive 32-bit floats in memory.
                samples = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
                block = AudioBlock(samples, sample_rate, start_sample)

                # 4. Pace delivery like live audio instead of reading as fast as
                # possible. Wait only if ahead of schedule; never skip samples.
                # main.py disables this wait when speaker playback sets the pace.
                # Target time = stream start + this block's end time. At 48000 Hz,
                # the first 512 samples are due about 10.7 ms after starting.
                # Using the original start avoids adding processing time to
                # every wait. If already late, deliver immediately without sleep.
                if self.realtime:
                    delay = started_at + block.end_seconds - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                # 5. yield pauses this function while main.py handles the block.
                # Its next request resumes here, then reads the next block.
                yield block

                # Advance by the actual length, including a shorter final block.
                start_sample += len(samples)


class MicrophoneInput:
    """Microphone -> waiting queue -> main.py -> feature extractor.

    The microphone keeps recording while main.py processes earlier blocks.
    The queue holds up to 8 blocks of 512 samples by default, but main.py takes
    them one at a time; it does not wait for all 8 or combine them into a frame.
    If the queue fills, we drop its oldest block to keep the analysis recent.
    This design handles brief delays, but cannot guarantee uninterrupted audio.
    The queue is separate from RMS's buffer: RMS initially collects 4 blocks
    for a 2048-sample frame, then reuses overlap with each new 512-sample block.
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
        # Track two kinds of loss: blocks we discard, and overflows reported
        # by the audio system before our code can receive all the samples.
        self.dropped_blocks = 0
        self.input_overflows = 0

    # Iterator[AudioBlock]: yield one AudioBlock at a time for a loop to consume.
    def blocks(self) -> Iterator[AudioBlock]:
        """Capture in a callback and yield queued blocks to the processing loop."""

        # 1. Choose and validate the input settings before starting capture.
        # Use the selected device's default rate unless the caller specifies one.
        info = sd.query_devices(self.device, "input")
        sample_rate = self.sample_rate or int(info["default_samplerate"])
        sd.check_input_settings(
            device=self.device, channels=self.channels,
            dtype="float32", samplerate=sample_rate,
        )

        # 2. Set up a queue shared by the capture callback and the loop below.
        # At 48000 Hz, 8 x 512 samples hold about 85 ms of waiting audio.
        # That is capacity, not a required wait: blocks can be taken immediately.
        pending = Queue(maxsize=self.queue_blocks)
        start_sample = 0  # Next block position assigned by the capture callback.
        expected_sample = 0  # Next block position the processing loop expects.
        self.dropped_blocks = 0
        self.input_overflows = 0

        # 3. sounddevice calls this whenever a new microphone block arrives.
        # In simple terms :) save the audio here; calculate features later in main.py.
        def capture(indata, frames, timing, status):

            # Update the counter in blocks(), rather than create a local one.
            nonlocal start_sample

            # Overflow means the audio system lost input before we received it.
            # This is separate from our own full-queue drop policy below.
            if status.input_overflow:
                self.input_overflows += 1

            # Copy because the audio system reuses indata after this callback.
            # Keep each block's original position, even if we later discard it.
            # Here "frames" means samples per channel in this input block,
            # normally 512; it is not the RMS analysis frame length of 2048.
            item = (indata.copy(), start_sample, bool(status))
            start_sample += frames
            try:
                # Do not wait for analysis to make room: capture must keep up
                # with incoming audio. A full queue raises Full immediately.
                pending.put_nowait(item)
            except Full:
                # Our deliberate drop policy: remove the oldest waiting block
                # to make room for the new one. Example with a 3-block queue:
                # [A B C] + D -> [B C D]. Only a full queue triggers this.
                # "Oldest" means oldest still waiting here; audio already sent
                # to RMS is separate and cannot be removed by this queue.
                try:
                    pending.get_nowait()
                    self.dropped_blocks += 1
                except Empty:
                    pass  # The consumer freed a slot in the meantime.
                pending.put_nowait(item)

        # 4. Entering the stream starts the callbacks and calls capture() whenever a new audio block arrives;
        # leaving this with-block stops capture and closes the device, including when an error occurs.
        with sd.InputStream(
            device=self.device, samplerate=sample_rate, channels=self.channels,
            blocksize=self.block_size, dtype="float32", latency="low",
            callback=capture,
        ) as stream:
            while True:
                try:
                    # 5. Take one block for processing. If none is ready, wait
                    # up to 0.25 s before checking whether capture has stopped.
                    # An empty queue is normal; it does not itself mean lost audio.
                    data, offset, status = pending.get(timeout=0.25)
                except Empty:
                    if not stream.active:
                        raise RuntimeError("Microphone stream stopped unexpectedly")
                    continue
                # Average input channels into mono, just as FileInput does.
                samples = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)

                # A jump in sample position reveals a block dropped from our
                # queue. Input status also flags possible loss in the audio
                # system. Pass this warning to main.py so it can reset RMS.
                # Example: after 512 samples, we expect offset 512. If the next
                # block starts at 1024, samples 512..1023 were skipped. Resetting
                # prevents saved RMS samples from being joined across that gap.
                discontinuity = status or offset != expected_sample
                expected_sample = offset + len(samples)

                # Pause this loop while main.py processes the block. Microphone
                # callbacks keep running and can add more blocks to the queue.
                yield AudioBlock(samples, sample_rate, offset, discontinuity)
