"""Stream audio and extract RMS. Defaults to the system's input device."""

import argparse
from contextlib import ExitStack
import math
import sys

import sounddevice as sd
from .audio_input import FileInput, MicrophoneInput
from .feature.rms import RMSExtractor
from .feature.spectral_centroid import SpectralCentroidExtractor
from .feature.mel_spectrogram import MelExtractor
from .feature.chroma_cqt import ChromaExtractor

_printing_enabled = True


def set_terminal_printing(enabled: bool) -> None:
    """Enable or disable calls to terminal_print (enabled by default)."""
    global _printing_enabled
    _printing_enabled = enabled


def terminal_print(*args, **kwargs) -> None:
    """Print when enabled, accepting the same arguments as built-in print."""
    if _printing_enabled:
        print(*args, **kwargs)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def device_id(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def main() -> int:
    # Read command-line options: choose a file or microphone as the audio source.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="File name under sample_audio (otherwise use microphone)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=device_id, help="Microphone device ID or name")

    # Optional file playback and live RMS display.
    parser.add_argument("--output-device", type=device_id, help="File playback output device ID or name (default: system output)")
    parser.add_argument("--no-playback", action="store_true", help="Analyze a file silently instead of playing it")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False,
                        help="Show a PyQtGraph RMS window (default: off; --no-plot disables it)")

    # Input settings control how audio arrives; frame and hop sizes control analysis.
    parser.add_argument("--sample-rate", type=positive_int, help="Microphone rate in Hz (default: device native rate)")
    parser.add_argument("--channels", type=positive_int, default=1, help="Microphone channels to average to mono (default: 1)")
    parser.add_argument("--block-size", type=positive_int, default=512, help="Samples per input block (default: 512)")

    parser.add_argument("--frame-length", type=positive_int, default=2048, help="Samples per RMS analysis frame (default: 2048)")
    parser.add_argument("--hop-length", type=positive_int, default=512, help="Samples between RMS frames (default: 512)")

    # Run length and terminal output settings.
    parser.add_argument("--duration", type=positive_seconds, help="Stop after this many source seconds, at a block boundary")
    parser.add_argument("--quiet", action="store_true", help="Hide routine monitoring output (warnings and errors remain visible)")
    args = parser.parse_args()

    # Check for incompatible settings before opening audio devices.
    if args.hop_length > args.frame_length:
        parser.error("--hop-length must not exceed --frame-length")
    #check for frame length
    if args.frame_length & (args.frame_length - 1):
        parser.error("--frame-length must be power of 2 for melspec")
    if args.quiet:
        set_terminal_printing(False)
    if args.file and (args.device is not None or args.sample_rate is not None or args.channels != 1):
        parser.error("--device, --sample-rate and --channels are microphone options; files use their native rate and downmix to mono")
    if not args.file and (args.no_playback or args.output_device is not None):
        parser.error("--no-playback and --output-device are file options; use --file")
    if args.no_playback and args.output_device is not None:
        parser.error("--output-device cannot be combined with --no-playback")

    # Printed summary when streaming stops
    source = None
    received = 0
    discontinuities = 0
    playback_underflows = 0

    try:
        # Device listing is a separate command; no audio stream is started.
        if args.list_devices:
            devices = sd.query_devices()
            if not devices:
                print("No audio devices found. Check audio permissions and connected devices.", file=sys.stderr)
                return 1
            print(devices)
            return 0

        # Select the source. Playback supplies the file's clock when enabled.
        play_file = bool(args.file) and not args.no_playback
        source = FileInput(args.file, args.block_size, realtime=not play_file) if args.file else MicrophoneInput(
            device=args.device, sample_rate=args.sample_rate,
            block_size=args.block_size, channels=args.channels,
        )

        # Close the audio streams and plot automatically on exit or error.
        with ExitStack() as resources:
            # Open the optional plot before audio starts.
            rms_plot = None
            if args.plot:
                from .rms_plot import RMSPlot

                rms_plot = RMSPlot()
                resources.callback(rms_plot.close)

            label = str(source.path) if isinstance(source, FileInput) else "microphone"
            terminal_print(f"Input: {label}. Press Ctrl+C to stop.", flush=True)

            # Create the extractor once so it retains overlap between input blocks.
            rms_extractor = RMSExtractor(args.frame_length, args.hop_length)
            # The centroid needs the sample rate, which is known from the first block.
            centroid_extractor = None

            #create mel_extractor/chroma_extractor only as placeholders since they needs sample rate, after first block
            mel_extractor = None
            chroma_extractor = None

            next_frame_start = 0
            next_report_seconds = 0.0
            playback = None

            # Read successive blocks from the selected audio source.
            blocks = source.blocks()
            resources.callback(blocks.close)

            for block in blocks:
                # File playback only: write the same mono samples that RMS analyzes.
                # Blocking output writes supply the clock, so FileInput does not sleep.
                # Microphone input is never routed to the speakers.
                if play_file:
                    if playback is None:
                        playback = resources.enter_context(sd.OutputStream(
                            samplerate=block.sample_rate, channels=1, dtype="float32",
                            device=args.output_device, blocksize=args.block_size, latency="high",
                        ))
                    if playback.write(block.samples):
                        playback_underflows += 1
                        if playback_underflows == 1:
                            print("Warning: playback underflow; audio output may have a gap.", file=sys.stderr)

                # Create the centroid extractor once the source's sample rate is known.
                if args.centroid and centroid_extractor is None:
                    centroid_extractor = SpectralCentroidExtractor(
                        block.sample_rate, args.frame_length, args.hop_length)

                # Count received samples and reset analysis history if audio was lost.
                received += len(block.samples)
                discontinuities += int(block.discontinuity)
                if block.discontinuity:
                    rms_extractor.reset()
                    mel_extractor.reset()
                    chroma_extractor.reset()

                    centroid_extractor.reset()
                    next_frame_start = block.start_sample
                    if rms_plot is not None:
                        #rms_plot.update([block.time_seconds], [float("nan")], block.time_seconds)
                        rms_plot.update(
                            [block.time_seconds], [float("nan")], block.time_seconds,
                            [float("nan")] if args.centroid else None,
                        )
                    print("Warning: input discontinuity; RMS analysis history reset.", file=sys.stderr)

                # Extract RMS for every new complete frame, keeping any overlap.
                rms_values = rms_extractor.process(block.samples)
                #extract mel values
                mel_values = mel_extractor.process(block.samples)
                #extract chroma values
                chroma_values = chroma_extractor.process(block.samples)

                # Send frame-end times and results to the plotting
                # Spectral centroid uses the same frame/hop, so it returns one value
                # per RMS value and both arrays stay aligned.
                centroid_values = (
                    centroid_extractor.process(block.samples) if centroid_extractor is not None else None
                )

                # Send frame-end times and results to the separate plotting process.
                if rms_plot is not None:
                    frame_times = [
                        (next_frame_start + args.frame_length + i * args.hop_length) / block.sample_rate
                        for i in range(len(rms_values))
                    ]
                    rms_plot.update(frame_times, rms_values, block.end_seconds, centroid_values)
                    if not rms_plot.is_open:
                        break
                next_frame_start += len(rms_values) * args.hop_length

                # Limit terminal updates to about twice per second.
                if rms_values.size and block.end_seconds >= next_report_seconds:
                    terminal_print(f"RMS={rms_values[-1]:.5f}", flush=True)
                    terminal_print(f"Mel peak={mel_extractor.band_frequencies[mel_values[-1].argmax()]:.0f} Hz",
                                   flush=True)
                    #chroma_values needs enough frames to have a valid frame since it uses larger frames sizes than the other extrators
                    if chroma_values.size:
                        terminal_print(f"Chroma peak={chroma_extractor.PITCH_CLASSES[chroma_values[-1].argmax()]}",
                                   flush=True)

                    terminal_print(f"Centroid={centroid_values[-1]:.1f} Hz", flush=True)
                    next_report_seconds = block.end_seconds + 0.5

                # Stop at a block boundar
                # y when the requested duration is reached.
                if args.duration is not None and block.end_seconds >= args.duration:
                    break

    except KeyboardInterrupt:
        terminal_print("\nStopped.")
    except (OSError, ValueError, RuntimeError, sd.PortAudioError) as error:
        print(f"Audio/visual stream error: {error}", file=sys.stderr)
        if not args.file:
            print("Try --list-devices and --device ID; on macOS allow microphone access for your terminal or IDE.", file=sys.stderr)
        elif not args.no_playback:
            print("Check the output device and its supported sample rates, or use --no-playback for silent analysis.", file=sys.stderr)
        return 1

    # Report sample totals and any capture or playback problems.
    if received:
        terminal_print(f"Received {received} mono samples; discontinuities={discontinuities}.")
    if isinstance(source, MicrophoneInput):
        terminal_print(f"Queue drops={source.dropped_blocks}; hardware input overflows={source.input_overflows}.")
    if args.file and not args.no_playback:
        terminal_print(f"Playback underflows={playback_underflows}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
