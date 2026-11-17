# Streaming audio input

Run these commands from the repository root, with the project environment active:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main
```

The default is the system input device, one channel, 512 samples per block, and
the device's native sample rate. Stop with **Ctrl+C**. On macOS, allow microphone
access for the terminal or PyCharm when prompted. Select a physical microphone
if the system default input is a virtual audio device.

## Check the microphone

```sh
python -m src.main --list-devices
python -m src.main --duration 5
python -m src.main --device 1 --sample-rate 48000 --duration 5
```

Replace `1` with an input-capable device ID from the list (or supply a quoted
device name). The selected device must support the requested rate and channel
count. The terminal prints the selected input at startup, the latest RMS about
every half second once a complete frame is available, and a summary when
streaming stops. Ideally discontinuities, queue drops, and hardware overflows
stay at zero.

## RMS extraction

`src/feature/rms.py` uses audioFlux's `Temporal.temporal(..., has_rms=True)`.
It calculates linear, time-domain RMS with a rectangular window (no taper),
using 2048-sample frames and a 512-sample hop by default. These settings are
independent of the input block size:

```sh
python -m src.main --frame-length 2048 --hop-length 512 --block-size 512
```

The extractor buffers incoming blocks, processes every complete frame, and
retains overlap for the next call. It waits for a full first frame and leaves
any incomplete final frame unprocessed; no zero padding is added. Hop length
must be positive and no greater than frame length. On an input discontinuity,
`main.py` resets the buffer before processing the new block.

RMS is calculated on every analysis frame, including in `--quiet` mode.
Only terminal updates are limited to about twice per second; each displays the
latest result. The extractor's `process(samples)` method returns all new RMS
values as a NumPy array for use by downstream code.

## Toggle terminal printing

Routine monitoring output is on by default. Disable it from the command line:

```sh
python -m src.main --quiet
```

Warnings and errors remain visible, and `--list-devices` still prints its list.
For a runtime switch within `main.py`, use the small helpers defined there:

```python
set_terminal_printing(False)  # Off
terminal_print("Input started")  # Hidden
set_terminal_printing(True)   # On again
terminal_print("Input started")  # Printed
```

Use `terminal_print(...)` for output you want to toggle; it accepts the same
arguments as `print(...)`, including `flush`, `sep`, and `end`.

## Stream a file

Put a WAV, FLAC, AIFF, or other libsndfile-supported file in `sample_audio/`:

```sh
python -m src.main --file example.wav
python -m src.main --file sample_audio/example.wav --duration 5
```

Bare file names resolve inside `sample_audio/`, regardless of the working
directory. Absolute paths are accepted too. Quote paths containing spaces.
File blocks arrive at real-time speed, using the **file's native sample rate**;
all channels are averaged to mono. Reading is incremental, and the stream stops
at EOF. The last block can be shorter than 512 samples. `--duration` stops at a
block boundary. Files now play through the system's default output device while
RMS is calculated. Playback uses the same mono signal as analysis, so stereo
files are heard as mono. No sample files are bundled here.

To choose speakers or headphones, select an output-capable device from
`--list-devices`. For silent analysis, use `--no-playback`:

```sh
python -m src.main --file example.wav --output-device 2
python -m src.main --file example.wav --no-playback
```

The output device must support the file's sample rate. Playback writes set the
pace; silent file analysis uses timed reads instead. The playback section is
commented in `main.py`. Microphone input is never played through the speakers.

Use `--block-size 1024` to change block size. Input blocks are transport chunks,
not analysis frames: the RMS extractor accumulates and overlaps them as needed.
`--sample-rate`, `--device`, and `--channels` apply only to microphones;
file resampling is not performed.

## Plot RMS live

Plotting is off by default. Add `--plot` to open a PyQtGraph window showing the
last ten seconds of RMS, or use `--no-plot` to explicitly turn it off:

```sh
python -m src.main --file example.wav --plot
python -m src.main --file example.wav --no-playback --plot
python -m src.main --device 1 --plot
python -m src.main --file example.wav --no-plot
```

Each point is one analysis frame's RMS, placed at the time that frame ends.
The y-axis follows the maximum RMS in the visible ten-second history, with 10%
headroom. It scales back down when louder values leave the window; the values
themselves remain linear RMS amplitudes.
The window runs in its own process and refreshes about 60 times per second.
`main.py` sends results through a small queue, so drawing does not run in the
audio loop. If the window falls behind and the queue fills, older plot updates
are discarded instead of making analysis wait. RMS extraction still processes
every analysis frame. `--quiet --plot` still shows the graph. A gap in microphone
input breaks the line and resets the extractor.

Close the plot or press Ctrl+C to stop. The plot closes when the stream ends.
Install the plotting dependencies with `python -m pip install -r requirements.txt`
and run from a local terminal with a graphical desktop. The graph follows the
audio source timeline; speaker buffering adds a small delay to what you hear.
Separating the window from audio reduces interference, but does not guarantee
exact synchronization with the speakers. Playback gaps are reported as
playback underflows.

## Add your own feature extractor

The pattern in `main.py` is: read an audio block, pass its samples to an
extractor, then use the returned feature values. Here is a shorter RMS example:

```python
from contextlib import closing
from src.audio_input import MicrophoneInput
from src.feature.rms import RMSExtractor

source = MicrophoneInput()
extractor = RMSExtractor(frame_length=2048, hop_length=512)

with closing(source.blocks()) as blocks:
    for block in blocks:
        if block.discontinuity:
            extractor.reset()  # Don't combine samples from before and after a gap.

        values = extractor.process(block.samples)
        if values.size:
            print(values[-1])  # Use the latest RMS value.
```

Create the extractor once, before the loop, so it can remember samples between
blocks. `process()` may return an empty array while it waits for enough samples,
or several values if the block completes several frames. The example prints
the latest value each time results are available; `main.py` prints less often.

To try another feature:

1. Add a file under `src/feature/`, such as `zcr.py` for zero-crossing rate.
2. Use `rms.py` as a starting point. Keep the frame buffering, `process()`, and
   `reset()` pattern, and replace the RMS calculation with your feature.
3. Import and create your extractor in `main.py`, then pass each block's samples
   to it inside the loop. Reset it when `block.discontinuity` is true.
4. Check the values with a known sound before using them to control a visual.

The extractor calculates feature values; `main.py` decides how to display or
send them. Spectral features and OSC output have not been added yet.
