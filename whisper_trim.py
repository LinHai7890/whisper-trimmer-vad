#!/usr/bin/env python3
"""
Remove non-speech sections from a media file or recursive folder of media.

The script detects spoken ranges with Whisper, expands each range by a small
padding, merges nearby ranges, and asks FFmpeg to render only those ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {
    ".3g2",
    ".3gp",
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

AUDIO_EXTENSIONS = {
    ".aac", ".ac3", ".aif", ".aiff", ".alac", ".amr", ".ape", ".au",
    ".flac", ".m4a", ".mka", ".mp3", ".oga", ".ogg", ".opus", ".ra",
    ".wav", ".wma", ".wv",
}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Whisper speech timestamps to remove non-speech parts of audio/video."
    )
    parser.add_argument("input", type=Path, help="Input media file or folder (scanned recursively)")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="Output file for a file input; optional output folder for a folder input",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing outputs in folder mode (otherwise they are skipped).",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name, such as tiny, base, small, medium, or large-v3",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "faster-whisper", "openai-whisper"),
        default="auto",
        help="Whisper package to use. auto tries faster-whisper first.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, such as en or es. Omit for auto-detection.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for Whisper, usually auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type, such as default, int8, float16, or float32.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Whisper beam size. Higher can be more accurate but slower.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Seconds to keep before and after detected speech.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.45,
        help="Merge speech ranges separated by this many seconds or less.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.08,
        help="Discard detected speech ranges shorter than this many seconds before padding.",
    )
    parser.add_argument(
        "--segment-timestamps",
        action="store_true",
        help="Use Whisper segment timestamps instead of word timestamps.",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Use faster-whisper's VAD filter when the faster-whisper backend is active.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Video quality for H.264 output. Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="H.264 encoder preset for video output.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate for compressed audio/video outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected ranges without rendering an output file.",
    )
    return parser.parse_args()


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"{name} was not found. Install FFmpeg and make sure {name} is on your PATH."
        )


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def ffprobe(input_path: Path) -> dict:
    return run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ]
    )


def media_duration(probe: dict) -> float:
    duration = probe.get("format", {}).get("duration")
    if duration is not None:
        return float(duration)

    stream_durations = [
        float(stream["duration"])
        for stream in probe.get("streams", [])
        if stream.get("duration") is not None
    ]
    if stream_durations:
        return max(stream_durations)

    raise SystemExit("Could not determine the input file duration.")


def has_stream(probe: dict, codec_type: str) -> bool:
    return any(stream.get("codec_type") == codec_type for stream in probe.get("streams", []))


def load_faster_whisper(args: argparse.Namespace):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    return WhisperModel(args.model, device=args.device, compute_type=args.compute_type)


def transcribe_with_faster_whisper(
    args: argparse.Namespace, model=None,
) -> list[tuple[float, float]]:
    if model is None:
        model = load_faster_whisper(args)
    segments, _info = model.transcribe(
        str(args.input),
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=not args.segment_timestamps,
        vad_filter=args.vad_filter,
    )

    intervals: list[tuple[float, float]] = []
    for segment in segments:
        if not args.segment_timestamps and segment.words:
            for word in segment.words:
                if word.start is not None and word.end is not None:
                    intervals.append((float(word.start), float(word.end)))
        elif segment.start is not None and segment.end is not None:
            intervals.append((float(segment.start), float(segment.end)))

    return intervals


def load_openai_whisper(args: argparse.Namespace):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper is not installed") from exc

    device = None if args.device == "auto" else args.device
    return whisper.load_model(args.model, device=device)


def transcribe_with_openai_whisper(
    args: argparse.Namespace, model=None,
) -> list[tuple[float, float]]:
    if model is None:
        model = load_openai_whisper(args)
    result = model.transcribe(
        str(args.input),
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=not args.segment_timestamps,
        verbose=False,
    )

    intervals: list[tuple[float, float]] = []
    for segment in result.get("segments", []):
        words = segment.get("words") or []
        if not args.segment_timestamps and words:
            for word in words:
                if word.get("start") is not None and word.get("end") is not None:
                    intervals.append((float(word["start"]), float(word["end"])))
        elif segment.get("start") is not None and segment.get("end") is not None:
            intervals.append((float(segment["start"]), float(segment["end"])))

    return intervals


class WhisperTranscriber:
    """Lazily load one backend/model and reuse it for a sequential batch."""

    def __init__(self) -> None:
        self.model = None
        self.backend = None
        self.load_error = None

    def __call__(self, args: argparse.Namespace) -> list[tuple[float, float]]:
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        if self.model is None:
            try:
                self._load(args)
            except Exception as exc:
                self.load_error = f"Could not load Whisper: {exc}"
                raise RuntimeError(self.load_error) from exc

        print(f"Transcribing with {self.backend} model '{args.model}'...")
        if self.backend == "faster-whisper":
            return transcribe_with_faster_whisper(args, self.model)
        return transcribe_with_openai_whisper(args, self.model)

    def _load(self, args: argparse.Namespace) -> None:
        if args.backend in ("auto", "faster-whisper"):
            try:
                print(f"Loading faster-whisper model '{args.model}'...")
                self.model = load_faster_whisper(args)
                self.backend = "faster-whisper"
                return
            except (ImportError, RuntimeError) as exc:
                if args.backend == "faster-whisper":
                    raise
                print(f"faster-whisper could not load: {exc}; trying openai-whisper...")

        print(f"Loading openai-whisper model '{args.model}'...")
        self.model = load_openai_whisper(args)
        self.backend = "openai-whisper"


def transcribe(
    args: argparse.Namespace, transcriber: WhisperTranscriber | None = None,
) -> list[tuple[float, float]]:
    # Keep transcribe(args) available to whisper_trim_all_dependant.py.
    if transcriber is None:
        transcriber = WhisperTranscriber()
    return transcriber(args)


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    duration: float,
    padding: float,
    merge_gap: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if end - start < min_duration:
            continue
        padded_start = max(0.0, start - padding)
        padded_end = min(duration, end + padding)
        cleaned.append((padded_start, padded_end))

    cleaned.sort(key=lambda item: item[0])
    if not cleaned:
        return []

    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= merge_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def print_ranges(intervals: list[tuple[float, float]]) -> None:
    total = sum(end - start for start, end in intervals)
    print(f"Keeping {len(intervals)} speech range(s), {total:.2f} seconds total:")
    for start, end in intervals:
        print(f"  {start:9.3f} -> {end:9.3f}  ({end - start:.3f}s)")


def label(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def build_filter_script(
    intervals: list[tuple[float, float]],
    include_video: bool,
    include_audio: bool,
) -> str:
    lines: list[str] = []
    concat_inputs: list[str] = []

    for index, (start, end) in enumerate(intervals):
        if include_video:
            v_label = label("v", index)
            lines.append(
                f"[0:v:0]trim=start={start:.6f}:end={end:.6f},"
                f"setpts=PTS-STARTPTS[{v_label}];"
            )
            concat_inputs.append(f"[{v_label}]")
        if include_audio:
            a_label = label("a", index)
            lines.append(
                f"[0:a:0]atrim=start={start:.6f}:end={end:.6f},"
                f"asetpts=PTS-STARTPTS[{a_label}];"
            )
            concat_inputs.append(f"[{a_label}]")

    video_count = 1 if include_video else 0
    audio_count = 1 if include_audio else 0
    outputs = ("[outv]" if include_video else "") + ("[outa]" if include_audio else "")
    lines.append(
        "".join(concat_inputs)
        + f"concat=n={len(intervals)}:v={video_count}:a={audio_count}{outputs}"
    )
    return "\n".join(lines)


def audio_codec_args(output_path: Path, bitrate: str) -> list[str]:
    suffix = output_path.suffix.lower()
    if suffix == ".wav":
        return ["-c:a", "pcm_s16le"]
    if suffix == ".flac":
        return ["-c:a", "flac"]
    if suffix == ".mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    return ["-c:a", "aac", "-b:a", bitrate]


def render_output(
    input_path: Path,
    output_path: Path,
    intervals: list[tuple[float, float]],
    include_video: bool,
    include_audio: bool,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ffmpeg", delete=False) as filter_file:
        filter_file.write(build_filter_script(intervals, include_video, include_audio))
        filter_path = Path(filter_file.name)

    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex_script",
            str(filter_path),
        ]
        if include_video:
            command += [
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-preset",
                args.preset,
                "-crf",
                str(args.crf),
                "-pix_fmt",
                "yuv420p",
                *audio_codec_args(output_path, args.audio_bitrate),
                "-movflags",
                "+faststart",
            ]
        else:
            command += ["-map", "[outa]", *audio_codec_args(output_path, args.audio_bitrate)]

        command.append(str(output_path))
        print("Rendering trimmed file with FFmpeg...")
        subprocess.run(command, check=True)
    finally:
        filter_path.unlink(missing_ok=True)


def trim_file(args: argparse.Namespace, transcriber: WhisperTranscriber) -> bool:
    """Return False when there is no speech; otherwise render or preview."""
    probe = ffprobe(args.input)
    duration = media_duration(probe)
    input_has_audio = has_stream(probe, "audio")
    input_has_video = has_stream(probe, "video")
    output_wants_video = args.output.suffix.lower() in VIDEO_EXTENSIONS
    include_video = input_has_video and output_wants_video

    if not input_has_audio:
        raise SystemExit("The input file has no audio stream to analyze for speech.")
    if output_wants_video and not input_has_video:
        raise SystemExit("The output extension looks like video, but the input has no video stream.")

    raw_intervals = transcribe(args, transcriber)
    intervals = merge_intervals(
        raw_intervals,
        duration=duration,
        padding=args.padding,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    if not intervals:
        print("No speech was detected, so no output file was created.")
        return False

    print_ranges(intervals)
    if args.dry_run:
        return True

    render_output(
        input_path=args.input,
        output_path=args.output,
        intervals=intervals,
        include_video=include_video,
        include_audio=True,
        args=args,
    )
    print(f"Done: {args.output}")
    return True


def find_media_files(folder: Path, output_folder: Path | None = None) -> list[Path]:
    """Snapshot sources, pruning a nested output folder and skipping generated files."""
    files: list[Path] = []

    def scan_error(error: OSError) -> None:
        raise error

    for root, directories, names in os.walk(folder, onerror=scan_error):
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".")
            and not (Path(root) / name).is_symlink()
            and (output_folder is None or (Path(root) / name).resolve() != output_folder)
        )
        for name in sorted(names, key=str.lower):
            path = Path(root) / name
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            if path.stem.lower().endswith("_trimmed"):
                print(f"Skipping generated output: {path}")
                continue
            if path.stat().st_size == 0:
                print(f"Skipping empty file: {path}")
                continue
            files.append(path)
    return files


def process_folder(args: argparse.Namespace, transcriber: WhisperTranscriber) -> int:
    files = find_media_files(args.input, args.output)
    print(f"Found {len(files)} source media file(s) in {args.input}")
    completed = skipped = failed = 0
    for index, input_path in enumerate(files, 1):
        relative = input_path.relative_to(args.input)
        destination = args.output / relative if args.output else input_path
        output_path = destination.with_name(f"{destination.stem}_trimmed{destination.suffix}")
        print(f"\n[{index}/{len(files)}] {input_path} -> {output_path}")
        if output_path.exists() and not args.overwrite:
            print("Skipping: output already exists (use --overwrite to replace it).")
            skipped += 1
            continue
        if output_path.is_symlink():
            print("Failed: refusing to write through an output symlink.")
            failed += 1
            continue
        file_args = argparse.Namespace(**vars(args))
        file_args.input = input_path
        file_args.output = output_path
        try:
            if trim_file(file_args, transcriber):
                completed += 1
            else:
                skipped += 1
        except (Exception, SystemExit) as exc:
            print(f"Failed: {input_path}: {exc}", file=sys.stderr)
            failed += 1
    action = "Previewed" if args.dry_run else "Completed"
    print(f"\nFinished. {action}: {completed}. Skipped: {skipped}. Failed: {failed}.")
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")
    if args.padding < 0 or args.merge_gap < 0 or args.min_duration < 0:
        raise SystemExit("--padding, --merge-gap, and --min-duration must be non-negative.")

    if args.input.is_dir():
        if args.output is not None and args.output.exists() and not args.output.is_dir():
            raise SystemExit("A folder input requires an output folder, not an existing file.")
        if args.output == args.input:
            args.output = None
    else:
        if args.output is None:
            raise SystemExit("An output file path is required for a single-file input.")
        if args.output == args.input or (args.output.exists() and args.input.samefile(args.output)):
            raise SystemExit("Input and output must be different files.")
        if args.output.is_dir():
            raise SystemExit("A single-file input requires an output file, not a folder.")

    require_tool("ffmpeg")
    require_tool("ffprobe")
    transcriber = WhisperTranscriber()
    if args.input.is_dir():
        return process_folder(args, transcriber)
    return 0 if trim_file(args, transcriber) else 1


if __name__ == "__main__":
    raise SystemExit(main())
