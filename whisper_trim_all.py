#!/usr/bin/env python3
"""
Trim every media file in the same folder as this script.

Outputs are written beside each original file using a suffix such as:
  interview.mp4 -> interview_trimmed.mp4
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
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
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".au",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


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


def transcribe_with_faster_whisper(args: argparse.Namespace) -> list[tuple[float, float]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
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


def transcribe_with_openai_whisper(args: argparse.Namespace) -> list[tuple[float, float]]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper is not installed") from exc

    device = None if args.device == "auto" else args.device
    model = whisper.load_model(args.model, device=device)
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


def transcribe(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.backend in ("auto", "faster-whisper"):
        try:
            print(f"Transcribing with faster-whisper model '{args.model}'...")
            return transcribe_with_faster_whisper(args)
        except RuntimeError as exc:
            if args.backend == "faster-whisper":
                raise SystemExit(str(exc)) from exc
            print("faster-whisper is not available; trying openai-whisper...")

    try:
        print(f"Transcribing with openai-whisper model '{args.model}'...")
        return transcribe_with_openai_whisper(args)
    except RuntimeError as exc:
        raise SystemExit(
            "No supported Whisper backend is installed. Install dependencies with:\n"
            "  python -m pip install faster-whisper\n"
            "or install openai-whisper separately."
        ) from exc


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Whisper speech timestamps to trim every media file beside this script."
    )
    parser.add_argument(
        "--output-suffix",
        default="_trimmed",
        help="Suffix added before each output extension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing batch output files.",
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
        help="Print detected ranges without rendering output files.",
    )
    return parser.parse_args()


def output_path_for(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def find_media_files(folder: Path, output_suffix: str) -> list[Path]:
    files: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if output_suffix and path.stem.endswith(output_suffix):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.name.lower())


def trim_file(input_path: Path, args: argparse.Namespace) -> bool:
    output_path = output_path_for(input_path, args.output_suffix)
    if output_path == input_path:
        print(f"Skipping {input_path.name}: output path would replace the original.")
        return False
    if output_path.exists() and not args.overwrite and not args.dry_run:
        print(f"Skipping {input_path.name}: {output_path.name} already exists.")
        return True

    file_args = copy.copy(args)
    file_args.input = input_path.resolve()
    file_args.output = output_path.resolve()

    print(f"\n==> {input_path.name} -> {output_path.name}")
    probe = ffprobe(file_args.input)
    duration = media_duration(probe)
    input_has_audio = has_stream(probe, "audio")
    input_has_video = has_stream(probe, "video")
    output_wants_video = file_args.output.suffix.lower() in VIDEO_EXTENSIONS
    include_video = input_has_video and output_wants_video

    if not input_has_audio:
        print("Skipping: no audio stream to analyze for speech.")
        return True
    if output_wants_video and not input_has_video:
        print("Skipping: file extension looks like video, but no video stream was found.")
        return True

    raw_intervals = transcribe(file_args)
    intervals = merge_intervals(
        raw_intervals,
        duration=duration,
        padding=file_args.padding,
        merge_gap=file_args.merge_gap,
        min_duration=file_args.min_duration,
    )

    if not intervals:
        print("Skipping: no speech was detected.")
        return True

    print_ranges(intervals)
    if file_args.dry_run:
        return True

    render_output(
        input_path=file_args.input,
        output_path=file_args.output,
        intervals=intervals,
        include_video=include_video,
        include_audio=True,
        args=file_args,
    )
    print(f"Done: {file_args.output}")
    return True


def main() -> int:
    args = parse_args()
    if not args.output_suffix:
        raise SystemExit("--output-suffix must not be empty.")
    if args.padding < 0 or args.merge_gap < 0 or args.min_duration < 0:
        raise SystemExit("--padding, --merge-gap, and --min-duration must be non-negative.")

    require_tool("ffmpeg")
    require_tool("ffprobe")

    folder = Path(__file__).resolve().parent
    files = find_media_files(folder, args.output_suffix)
    if not files:
        print(f"No media files found in {folder}")
        return 0

    print(f"Found {len(files)} media file(s) in {folder}")
    succeeded = 0
    failed = 0
    for input_path in files:
        try:
            if trim_file(input_path, args):
                succeeded += 1
            else:
                failed += 1
        except (OSError, RuntimeError, subprocess.CalledProcessError, SystemExit) as exc:
            failed += 1
            print(f"Failed: {input_path.name}: {exc}")

    print(f"\nFinished. Successful or skipped: {succeeded}. Failed: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
