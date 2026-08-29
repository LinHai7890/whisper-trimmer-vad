#!/usr/bin/env python3
"""
Remove non-speech sections from an audio or video file using Whisper timestamps.

The script detects spoken ranges with Whisper, expands each range by a small
padding, merges nearby ranges, and asks FFmpeg to render only those ranges.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Whisper speech timestamps to remove non-speech parts of audio/video."
    )
    parser.add_argument("input", type=Path, help="Input audio or video file")
    parser.add_argument("output", type=Path, help="Trimmed output file")
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
            "  python -m pip install -r requirements.txt\n"
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


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")
    if args.padding < 0 or args.merge_gap < 0 or args.min_duration < 0:
        raise SystemExit("--padding, --merge-gap, and --min-duration must be non-negative.")

    require_tool("ffmpeg")
    require_tool("ffprobe")

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

    raw_intervals = transcribe(args)
    intervals = merge_intervals(
        raw_intervals,
        duration=duration,
        padding=args.padding,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    if not intervals:
        raise SystemExit("No speech was detected, so no output file was created.")

    print_ranges(intervals)
    if args.dry_run:
        return 0

    render_output(
        input_path=args.input,
        output_path=args.output,
        intervals=intervals,
        include_video=include_video,
        include_audio=True,
        args=args,
    )
    print(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
