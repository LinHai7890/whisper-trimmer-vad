#!/usr/bin/env python3
"""
Remove non-speech sections from a media file or recursive folder of media.

The script finds spoken ranges, expands each range by a small padding, merges
nearby ranges, and asks FFmpeg to render only those ranges.

The ranges are applied in a single pass, by selecting the frames and samples
whose timestamps fall inside them, which keeps the cost flat as the number of
ranges grows. On an episode that turns into three hundred of them, the older
trim and concat approach needed one filter branch per range and slowed down
sharply.

Two detectors are available. Silero VAD is the default: it is a voice activity
model, so it keeps shouting, singing and overlapping dialogue that a speech
recognizer drops, and it never invents speech over music or noise. Whisper is
still available with ``--detector whisper`` when you also want transcript-based
refinement.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Silero VAD only runs on this sample rate.
VAD_SAMPLE_RATE = 16000

# Hardware encoders are tried in this order when --encoder auto is used.
ENCODER_PREFERENCE = ("h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_vaapi", "libx264")
# Past this many characters the select expression gets unwieldy and the trim and
# concat graph is used instead.
SELECT_EXPRESSION_LIMIT = 60_000
ENCODER_CHOICES = (
    "auto",
    "copy",
    "libx264",
    "libx265",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_qsv",
    "h264_vaapi",
    "hevc_vaapi",
)
ENCODER_ALIASES = {
    "x264": "libx264",
    "x265": "libx265",
    "nvenc": "h264_nvenc",
    "qsv": "h264_qsv",
    "vaapi": "h264_vaapi",
}


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
        "--detector",
        choices=("vad", "whisper"),
        default="vad",
        help="How speech ranges are found. vad uses Silero voice activity "
        "detection and is the default; whisper uses speech recognition "
        "timestamps",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name, such as tiny, base, small, medium, or large-v3. "
        "Only used with --detector whisper",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "faster-whisper", "openai-whisper"),
        default="auto",
        help="Whisper package to use. auto tries faster-whisper first. "
        "Only used with --detector whisper",
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
        help="Seconds to keep before and after detected speech. "
        "Only used with --detector whisper; the VAD detector uses --vad-pad-ms.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.45,
        help="Merge speech ranges separated by this many seconds or less. "
        "Only used with --detector whisper; the VAD detector uses --vad-min-silence-ms.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.08,
        help="Discard detected speech ranges shorter than this many seconds before padding. "
        "Only used with --detector whisper; the VAD detector uses --vad-min-speech-ms.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech probability threshold from 0 to 1. Lower keeps "
        "quieter speech; raise it to stop music and noise from being kept.",
    )
    parser.add_argument(
        "--vad-pad-ms",
        type=int,
        default=200,
        help="Milliseconds of audio to keep before and after each VAD range.",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="Discard VAD ranges shorter than this many milliseconds.",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=350,
        help="Join VAD ranges separated by this many milliseconds of silence or less.",
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
        "--audio-stream",
        type=int,
        default=0,
        help="Which audio track to detect speech on and to keep, counting from 0.",
    )
    parser.add_argument(
        "--video-stream",
        type=int,
        default=0,
        help="Which video track to keep, counting from 0.",
    )
    parser.add_argument(
        "--encoder",
        default="auto",
        help="How the output is produced. auto picks a hardware video encoder "
        "when FFmpeg has one and falls back to libx264. copy cuts without "
        "re-encoding, which is much faster but snaps cuts to key frames. "
        "Otherwise use one of: " + ", ".join(ENCODER_CHOICES),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Video quality for libx264 and libx265 output. Lower is higher quality.",
    )
    parser.add_argument(
        "--cq",
        type=int,
        default=19,
        help="Video quality for hardware encoder output. Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="Encoder preset for video output. libx264 and libx265 use x264-style "
        "names such as veryfast or medium; hardware encoders use their own.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate for compressed audio/video outputs.",
    )
    parser.add_argument(
        "--list-dir",
        type=Path,
        default=None,
        help="Also write the cut list for every source here: an .ffconcat file "
        "that FFmpeg and editors can open, plus a .csv with the original "
        "timecodes of each kept range.",
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
    )
    # FFmpeg writes UTF-8 regardless of the console code page, so decoding it
    # with the locale encoding breaks on non-ASCII file names.
    return json.loads(completed.stdout.decode("utf-8", errors="replace"))


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


def count_streams(probe: dict, codec_type: str) -> int:
    return sum(1 for stream in probe.get("streams", []) if stream.get("codec_type") == codec_type)


def decode_audio_mono(input_path: Path, stream_index: int):
    """Decode one audio track to mono float32 at the rate Silero expects."""
    import numpy as np

    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            f"0:a:{stream_index}",
            "-ac",
            "1",
            "-ar",
            str(VAD_SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


class SileroVadDetector:
    """Find voice activity with Silero, without transcribing any of it.

    Unlike a speech recognizer this keeps shouting, singing and overlapping
    dialogue, and it never invents speech over music, ambience or effects.
    """

    def __init__(self) -> None:
        self.model = None
        self.torch = None
        self.load_error = None

    def __call__(self, args: argparse.Namespace) -> list[tuple[float, float]]:
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        if self.model is None:
            self._load()

        audio = decode_audio_mono(args.input, args.audio_stream)
        if audio.size == 0:
            return []
        print(
            f"Scanning {audio.size / VAD_SAMPLE_RATE:.1f}s of audio with Silero VAD "
            f"(threshold {args.vad_threshold})..."
        )
        from silero_vad import get_speech_timestamps

        timestamps = get_speech_timestamps(
            self.torch.from_numpy(audio),
            self.model,
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_pad_ms,
            return_seconds=True,
        )
        return [(float(stamp["start"]), float(stamp["end"])) for stamp in timestamps]

    def _load(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad

            self.torch = torch
            self.model = load_silero_vad()
        except Exception as exc:
            self.load_error = (
                f"Could not load Silero VAD: {exc}. "
                "Install it with: pip install silero-vad"
            )
            raise RuntimeError(self.load_error) from exc
        print("Loaded Silero VAD model.")


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


def build_detector(args: argparse.Namespace):
    """Return the detector named by --detector. Both are called as detector(args)."""
    if getattr(args, "detector", "vad") == "whisper":
        return WhisperTranscriber()
    return SileroVadDetector()


def detector_padding(args: argparse.Namespace) -> tuple[float, float, float]:
    """Padding, merge gap and minimum length for the configured detector.

    Silero already applies its own padding and gap merging, so the VAD detector
    only passes zero here, which turns merge_intervals into a sort and clamp.
    """
    if getattr(args, "detector", "vad") == "vad":
        return 0.0, 0.0, 0.0
    return args.padding, args.merge_gap, args.min_duration


def transcribe(
    args: argparse.Namespace, transcriber=None,
) -> list[tuple[float, float]]:
    # Keep transcribe(args) available to whisper_trim_all_dependant.py.
    if transcriber is None:
        transcriber = build_detector(args)
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


def timecode(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def ffconcat_escape(path: Path) -> str:
    """Escape a path for the FFmpeg concat demuxer.

    Current FFmpeg reads a Windows path with forward slashes straight from the
    quoted file line, so only the separator and a single quote need handling.
    """
    return str(path).replace("\\", "/").replace("'", r"'\''")


def video_keyframes(input_path: Path, stream_index: int) -> list[float]:
    """Timestamps of every key frame in a video track.

    Read from packet flags rather than by decoding, which keeps this cheap on
    a feature length file.
    """
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"v:{stream_index}",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    timestamps: list[float] = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        columns = line.split(",")
        if len(columns) < 2 or "K" not in columns[1]:
            continue
        try:
            timestamps.append(float(columns[0]))
        except ValueError:
            continue
    return sorted(set(timestamps))


def align_to_keyframes(
    intervals: list[tuple[float, float]],
    keyframes: list[float],
    duration: float,
) -> list[tuple[float, float]]:
    """Grow each range outwards to whole key frames.

    A copied H.264 range is only decodable when it starts on an IDR, so every
    range has to begin at a key frame and end where the next one begins.
    Anything in between is kept even when it holds no speech, which is the
    price of not re-encoding.
    """
    if not keyframes:
        raise SystemExit(
            "Could not find any key frames to align to. "
            "Re-encode instead of using --encoder copy."
        )

    aligned: list[tuple[float, float]] = []
    for start, end in intervals:
        start_index = bisect.bisect_right(keyframes, start + 1e-3) - 1
        if start_index < 0:
            start_index = 0
        aligned_start = keyframes[start_index]

        end_index = bisect.bisect_left(keyframes, end - 1e-3)
        if end_index >= len(keyframes):
            aligned_end = duration
        else:
            aligned_end = keyframes[end_index]

        if aligned_end > aligned_start:
            aligned.append((aligned_start, aligned_end))

    return merge_intervals(aligned, duration=duration, padding=0.0, merge_gap=0.0, min_duration=0.0)


def build_ffconcat(input_path: Path, intervals: list[tuple[float, float]]) -> str:
    lines = ["ffconcat version 1.0"]
    for start, end in intervals:
        lines.append(f"file '{ffconcat_escape(input_path)}'")
        lines.append(f"inpoint {start:.6f}")
        lines.append(f"outpoint {end:.6f}")
    return "\n".join(lines) + "\n"


def write_cut_list(
    list_dir: Path,
    input_path: Path,
    relative: Path,
    intervals: list[tuple[float, float]],
) -> tuple[Path, Path]:
    """Write an .ffconcat cut list and a .csv of the original timecodes."""
    destination = list_dir / relative.with_suffix("")
    concat_path = destination.with_suffix(".ffconcat")
    csv_path = destination.with_suffix(".csv")
    concat_path.parent.mkdir(parents=True, exist_ok=True)

    concat_path.write_text(build_ffconcat(input_path, intervals), encoding="utf-8")

    rows = ["index,start_seconds,end_seconds,duration_seconds,start_timecode,end_timecode"]
    for index, (start, end) in enumerate(intervals, 1):
        rows.append(
            f"{index},{start:.3f},{end:.3f},{end - start:.3f},"
            f"{timecode(start)},{timecode(end)}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return concat_path, csv_path


def select_expression(intervals: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{start:.6f},{end:.6f})" for start, end in intervals)


def build_select_script(
    intervals: list[tuple[float, float]],
    include_video: bool,
    include_audio: bool,
    video_stream: int = 0,
    audio_stream: int = 0,
) -> str:
    """Keep the frames and samples that fall inside the ranges, in one pass.

    Selecting against a timestamp expression decodes the source once and leaves
    the timeline compressed, which stays fast when there are hundreds of
    ranges. The trim and concat equivalent needs one filter branch per range
    and slows down sharply as they pile up.
    """
    expression = select_expression(intervals)
    chains: list[str] = []

    if include_video:
        chains.append(
            f"[0:v:{video_stream}]select='{expression}',setpts=N/FRAME_RATE/TB[outv]"
        )
    if include_audio:
        chains.append(f"[0:a:{audio_stream}]aselect='{expression}',asetpts=N/SR/TB[outa]")

    return ";\n".join(chains) + ";\n"


def build_filter_script(
    intervals: list[tuple[float, float]],
    include_video: bool,
    include_audio: bool,
    video_stream: int = 0,
    audio_stream: int = 0,
) -> str:
    lines: list[str] = []
    concat_inputs: list[str] = []

    for index, (start, end) in enumerate(intervals):
        if include_video:
            v_label = label("v", index)
            lines.append(
                f"[0:v:{video_stream}]trim=start={start:.6f}:end={end:.6f},"
                f"setpts=PTS-STARTPTS[{v_label}];"
            )
            concat_inputs.append(f"[{v_label}]")
        if include_audio:
            a_label = label("a", index)
            lines.append(
                f"[0:a:{audio_stream}]atrim=start={start:.6f}:end={end:.6f},"
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


_available_encoders: set[str] | None = None


def available_encoders() -> set[str]:
    """Encoder names this FFmpeg build supports, probed once per run."""
    global _available_encoders
    if _available_encoders is None:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        text = completed.stdout.decode("utf-8", errors="replace")
        names: set[str] = set()
        for line in text.splitlines():
            columns = line.split()
            # Video encoder lines look like " V....D h264_nvenc   NVIDIA ...".
            if len(columns) >= 2 and columns[0].startswith("V") and len(columns[0]) > 1:
                names.add(columns[1])
        _available_encoders = names
    return _available_encoders


def resolve_encoder(requested: str) -> str:
    if requested != "auto":
        name = ENCODER_ALIASES.get(requested, requested)
        if name not in ENCODER_CHOICES or name == "auto":
            raise SystemExit(f"Unknown encoder '{requested}'. Choose one of: {', '.join(ENCODER_CHOICES)}")
        if name == "copy":
            # Stream copy is a concat demuxer mode, not an FFmpeg encoder.
            return name
        if name not in available_encoders():
            raise SystemExit(
                f"This FFmpeg build has no '{name}' encoder. "
                "Use --encoder auto to pick one automatically."
            )
        return name

    supported = available_encoders()
    for encoder in ENCODER_PREFERENCE:
        if encoder in supported:
            return encoder
    return "libx264"


def video_encoder_args(encoder: str, args: argparse.Namespace) -> list[str]:
    quality = getattr(args, "cq", 19)
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        return [
            "-c:v", encoder,
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(quality),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", encoder,
            "-preset", getattr(args, "preset", "medium"),
            "-global_quality", str(quality),
            "-look_ahead", "0",
            "-pix_fmt", "yuv420p",
        ]
    if encoder in ("h264_vaapi", "hevc_vaapi"):
        return [
            "-c:v", encoder,
            "-vaapi_device", "/dev/dri/renderD128",
            "-rc_mode", "CQP",
            "-qp", str(quality),
            "-pix_fmt", "nv12",
        ]
    # libx264 and libx265.
    return [
        "-c:v", encoder,
        "-preset", getattr(args, "preset", "medium"),
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
    ]


def render_copy(
    input_path: Path,
    output_path: Path,
    intervals: list[tuple[float, float]],
    duration: float,
    video_stream: int = 0,
) -> list[tuple[float, float]]:
    """Cut without re-encoding, after growing each range to whole key frames.

    Far faster than re-encoding and lossless, but a copied H.264 range is only
    decodable when it starts on an IDR, so every range is widened to the key
    frame before it and the key frame after it. Whatever sits in between is
    kept even when it holds no speech.
    """
    keyframes = video_keyframes(input_path, video_stream)
    aligned = align_to_keyframes(intervals, keyframes, duration)

    kept = sum(end - start for start, end in aligned)
    print(
        f"Aligned {len(intervals)} range(s) to {len(keyframes)} key frame(s): "
        f"{kept:.2f}s kept instead of "
        f"{sum(end - start for start, end in intervals):.2f}s "
        f"(+{kept - sum(end - start for start, end in intervals):.2f}s that holds no speech)."
    )

    with tempfile.TemporaryDirectory() as workspace:
        list_path = Path(workspace) / "cut.ffconcat"
        list_path.write_text(build_ffconcat(input_path, aligned), encoding="utf-8")
        print("Rendering trimmed file with FFmpeg (stream copy)...")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    return aligned


def render_output(
    input_path: Path,
    output_path: Path,
    intervals: list[tuple[float, float]],
    include_video: bool,
    include_audio: bool,
    args: argparse.Namespace,
    duration: float | None = None,
) -> list[tuple[float, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = resolve_encoder(getattr(args, "encoder", "auto"))
    if encoder == "copy":
        if include_video and duration is not None:
            return render_copy(
                input_path, output_path, intervals, duration,
                getattr(args, "video_stream", 0),
            )
        # Stream copy cannot transcode, and an audio-only output still needs the
        # speech ranges applied as filters.
        print("Audio-only output cannot be stream copied; re-encoding instead.")
        encoder = resolve_encoder("auto")

    video_stream = getattr(args, "video_stream", 0)
    audio_stream = getattr(args, "audio_stream", 0)

    # Prefer the single pass select graph. It stays fast with hundreds of
    # ranges, where one trim branch per range does not.
    expression_length = len(select_expression(intervals))
    if expression_length <= SELECT_EXPRESSION_LIMIT:
        filter_graph = build_select_script(
            intervals, include_video, include_audio, video_stream, audio_stream
        )
    else:
        print(
            f"{expression_length} expression characters is past the "
            f"{SELECT_EXPRESSION_LIMIT} limit; falling back to trim and concat."
        )
        filter_graph = build_filter_script(
            intervals, include_video, include_audio, video_stream, audio_stream
        )

    with tempfile.NamedTemporaryFile("w", suffix=".ffmpeg", delete=False) as filter_file:
        filter_file.write(filter_graph)
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
                *video_encoder_args(encoder, args),
                *audio_codec_args(output_path, args.audio_bitrate),
                "-movflags",
                "+faststart",
            ]
        else:
            command += ["-map", "[outa]", *audio_codec_args(output_path, args.audio_bitrate)]

        command.append(str(output_path))
        if include_video:
            print(f"Rendering trimmed file with FFmpeg ({encoder})...")
        else:
            print("Rendering trimmed file with FFmpeg...")
        subprocess.run(command, check=True)
    finally:
        filter_path.unlink(missing_ok=True)

    return intervals


def trim_file(args: argparse.Namespace, transcriber=None) -> bool:
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

    audio_total = count_streams(probe, "audio")
    if args.audio_stream >= audio_total:
        raise SystemExit(
            f"--audio-stream {args.audio_stream} was requested but the input has "
            f"{audio_total} audio track(s)."
        )
    if include_video and args.video_stream >= count_streams(probe, "video"):
        raise SystemExit(
            f"--video-stream {args.video_stream} was requested but the input has "
            f"{count_streams(probe, 'video')} video track(s)."
        )

    raw_intervals = transcribe(args, transcriber)
    padding, merge_gap, min_duration = detector_padding(args)
    intervals = merge_intervals(
        raw_intervals,
        duration=duration,
        padding=padding,
        merge_gap=merge_gap,
        min_duration=min_duration,
    )

    if not intervals:
        print("No speech was detected, so no output file was created.")
        return False

    print_ranges(intervals)
    if args.list_dir is not None:
        source_root = getattr(args, "source_root", None)
        if source_root is not None:
            relative = args.input.relative_to(source_root)
        else:
            # Single-file mode has no folder to mirror, and an absolute path
            # would make the join discard list_dir entirely.
            relative = Path(args.input.name)
        concat_path, csv_path = write_cut_list(args.list_dir, args.input, relative, intervals)
        print(f"Cut list: {concat_path}")
        print(f"Timecodes: {csv_path}")
    if args.dry_run:
        return True

    rendered = render_output(
        input_path=args.input,
        output_path=args.output,
        intervals=intervals,
        include_video=include_video,
        include_audio=True,
        args=args,
        duration=duration,
    )

    # Stream copying widens every range to a key frame, so the cut list is
    # written afterwards to record what was actually rendered.
    if args.list_dir is not None and rendered != intervals:
        source_root = getattr(args, "source_root", None)
        if source_root is not None:
            relative = args.input.relative_to(source_root)
        else:
            relative = Path(args.input.name)
        concat_path, csv_path = write_cut_list(args.list_dir, args.input, relative, rendered)
        print(f"Cut list: {concat_path}")
        print(f"Timecodes: {csv_path}")

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


def process_folder(args: argparse.Namespace, transcriber=None) -> int:
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
        file_args.source_root = args.input
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
    # Source names are often non-ASCII; never die because the console code page
    # cannot represent one.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = parse_args()
    args.input = args.input.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if args.list_dir is not None:
        args.list_dir = args.list_dir.expanduser().resolve()
    args.source_root = None
    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")
    if args.padding < 0 or args.merge_gap < 0 or args.min_duration < 0:
        raise SystemExit("--padding, --merge-gap, and --min-duration must be non-negative.")
    if not 0 < args.vad_threshold < 1:
        raise SystemExit("--vad-threshold must be between 0 and 1.")
    if min(args.vad_pad_ms, args.vad_min_speech_ms, args.vad_min_silence_ms) < 0:
        raise SystemExit("The --vad-* millisecond options must be non-negative.")
    if min(args.audio_stream, args.video_stream) < 0:
        raise SystemExit("--audio-stream and --video-stream must be non-negative.")

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
    detector = build_detector(args)
    if args.input.is_dir():
        return process_folder(args, detector)
    return 0 if trim_file(args, detector) else 1


if __name__ == "__main__":
    raise SystemExit(main())
