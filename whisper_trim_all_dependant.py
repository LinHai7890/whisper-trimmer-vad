#!/usr/bin/env python3
"""
Trim every media file in the same folder as this script.

Outputs are written beside each original file using a suffix such as:
  interview.mp4 -> interview_trimmed.mp4
"""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

from whisper_trim import (
    VIDEO_EXTENSIONS,
    ffprobe,
    has_stream,
    media_duration,
    merge_intervals,
    print_ranges,
    render_output,
    require_tool,
    transcribe,
)


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
