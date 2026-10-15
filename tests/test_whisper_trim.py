import argparse
import shutil
import subprocess

import pytest

import whisper_trim
from whisper_trim import (
    SileroVadDetector,
    build_detector,
    build_filter_script,
    detector_padding,
    ffconcat_escape,
    merge_intervals,
    resolve_encoder,
    timecode,
    video_encoder_args,
    write_cut_list,
)
from pathlib import Path


def whisper_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        detector="whisper",
        padding=0.25,
        merge_gap=0.45,
        min_duration=0.08,
        vad_threshold=0.5,
        vad_pad_ms=200,
        vad_min_speech_ms=250,
        vad_min_silence_ms=350,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestMergeIntervals:
    def test_drops_empty_ranges(self):
        assert merge_intervals([(1.0, 1.0), (2.0, 3.0)], 10.0, 0.0, 0.0, 0.0) == [(2.0, 3.0)]

    def test_pads_and_clamps_to_duration(self):
        merged = merge_intervals([(1.0, 2.0)], duration=2.5, padding=0.5, merge_gap=0.0, min_duration=0.0)
        assert merged == [(0.5, 2.5)]

    def test_never_starts_before_zero(self):
        assert merge_intervals([(0.1, 1.0)], 10.0, 1.0, 0.0, 0.0) == [(0.0, 2.0)]

    def test_merges_ranges_within_gap(self):
        merged = merge_intervals(
            [(1.0, 2.0), (2.4, 3.0)], duration=10.0, padding=0.0, merge_gap=0.5, min_duration=0.0
        )
        assert merged == [(1.0, 3.0)]

    def test_keeps_ranges_beyond_gap(self):
        merged = merge_intervals(
            [(1.0, 2.0), (5.0, 6.0)], duration=10.0, padding=0.0, merge_gap=0.5, min_duration=0.0
        )
        assert merged == [(1.0, 2.0), (5.0, 6.0)]

    def test_min_duration_applies_before_padding(self):
        merged = merge_intervals(
            [(1.0, 1.05), (5.0, 6.0)], duration=10.0, padding=0.0, merge_gap=0.0, min_duration=0.08
        )
        assert merged == [(5.0, 6.0)]

    def test_sorts_before_merging(self):
        merged = merge_intervals(
            [(5.0, 6.0), (1.0, 2.0)], duration=10.0, padding=0.0, merge_gap=0.0, min_duration=0.0
        )
        assert merged == [(1.0, 2.0), (5.0, 6.0)]

    def test_zero_padding_still_joins_overlaps(self):
        merged = merge_intervals(
            [(1.0, 2.0), (1.5, 3.0)], duration=10.0, padding=0.0, merge_gap=0.0, min_duration=0.0
        )
        assert merged == [(1.0, 3.0)]


class TestDetectorSelection:
    def test_vad_is_the_default(self):
        assert isinstance(build_detector(whisper_args(detector="vad")), SileroVadDetector)

    def test_whisper_detector_is_selectable(self):
        assert isinstance(build_detector(whisper_args(detector="whisper")), whisper_trim.WhisperTranscriber)

    def test_vad_owns_its_own_padding(self):
        assert detector_padding(whisper_args(detector="vad")) == (0.0, 0.0, 0.0)

    def test_whisper_uses_the_padded_options(self):
        assert detector_padding(whisper_args(detector="whisper")) == (0.25, 0.45, 0.08)


class TestFilterScript:
    def test_uses_the_requested_tracks(self):
        script = build_filter_script([(1.0, 2.0)], True, True, video_stream=1, audio_stream=2)
        assert "[0:v:1]trim=start=1.000000:end=2.000000" in script
        assert "[0:a:2]atrim=start=1.000000:end=2.000000" in script

    def test_audio_only_output_has_no_video_trim(self):
        script = build_filter_script([(1.0, 2.0)], False, True)
        assert "[0:v:0]" not in script
        assert "concat=n=1:v=0:a=1[outa]" in script


class TestEncoders:
    def test_nvenc_is_preferred_when_available(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", {"libx264", "h264_nvenc"})
        assert resolve_encoder("auto") == "h264_nvenc"

    def test_falls_back_to_libx264(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", {"libx264"})
        assert resolve_encoder("auto") == "libx264"

    def test_explicit_missing_encoder_is_an_error(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", {"libx264"})
        with pytest.raises(SystemExit):
            resolve_encoder("h264_nvenc")

    def test_short_names_are_accepted(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", {"libx264", "h264_nvenc", "h264_qsv"})
        assert resolve_encoder("nvenc") == "h264_nvenc"
        assert resolve_encoder("x264") == "libx264"
        assert resolve_encoder("qsv") == "h264_qsv"

    def test_copy_is_not_probed_as_an_encoder(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", set())
        assert resolve_encoder("copy") == "copy"

    def test_auto_never_picks_stream_copy(self, monkeypatch):
        monkeypatch.setattr(whisper_trim, "_available_encoders", {"libx264", "h264_nvenc"})
        assert resolve_encoder("auto") == "h264_nvenc"

    def test_nvenc_uses_constant_quality(self):
        args = argparse.Namespace(crf=18, cq=19, preset="medium")
        flags = video_encoder_args("h264_nvenc", args)
        assert flags[flags.index("-cq") + 1] == "19"
        assert "-crf" not in flags

    def test_libx264_uses_crf(self):
        args = argparse.Namespace(crf=20, cq=19, preset="veryfast")
        flags = video_encoder_args("libx264", args)
        assert flags[flags.index("-crf") + 1] == "20"
        assert flags[flags.index("-preset") + 1] == "veryfast"


class TestCutList:
    def test_timecode_formatting(self):
        assert timecode(0) == "00:00:00.000"
        assert timecode(3661.5) == "01:01:01.500"

    def test_windows_path_uses_forward_slashes(self):
        assert ffconcat_escape(Path(r"D:\AI WORKSPACE\clip.mkv")) == "D:/AI WORKSPACE/clip.mkv"

    def test_single_quote_is_escaped(self):
        assert ffconcat_escape(Path("/tmp/it's.mkv")) == r"/tmp/it'\''s.mkv"

    def test_relative_path_is_required(self, tmp_path):
        """An absolute 'relative' path would make the join drop list_dir."""
        source = tmp_path / "media" / "ep10.mkv"
        source.parent.mkdir()
        source.write_bytes(b"")
        concat_path, _ = write_cut_list(
            tmp_path / "lists", source, Path("ep10.mkv"), [(1.0, 2.0)]
        )
        assert concat_path.parent == tmp_path / "lists"

    def test_writes_ffconcat_and_csv(self, tmp_path):
        concat_path, csv_path = write_cut_list(
            tmp_dir := tmp_path / "lists", Path("/media/ep10.mkv"), Path("s1/ep10.mkv"), [(1.0, 2.5), (10.0, 12.0)]
        )
        assert concat_path == tmp_dir / "s1" / "ep10.ffconcat"
        assert csv_path == tmp_dir / "s1" / "ep10.csv"

        concat = concat_path.read_text(encoding="utf-8").splitlines()
        assert concat[0] == "ffconcat version 1.0"
        assert concat[1:4] == ["file '/media/ep10.mkv'", "inpoint 1.000000", "outpoint 2.500000"]

        rows = csv_path.read_text(encoding="utf-8").splitlines()
        assert rows[0].startswith("index,start_seconds")
        assert rows[1] == "1,1.000,2.500,1.500,00:00:01.000,00:00:02.500"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_ffconcat_cut_list_is_playable(tmp_path):
    """The cut list has to survive a real concat run, paths and all."""
    source = tmp_path / "a b's clip.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
            str(source),
        ],
        check=True,
    )
    concat_path, _ = write_cut_list(tmp_path / "lists", source, Path(source.name), [(1.0, 3.0)])
    assert concat_path.exists()

    rendered = tmp_path / "rendered.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-c", "copy", str(rendered),
        ],
        check=True,
    )
    assert rendered.stat().st_size > 0
    assert 1.5 < duration_of(rendered) < 2.5


def duration_of(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        stdout=subprocess.PIPE,
    )
    return float(completed.stdout.decode("utf-8", errors="replace").strip())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_silence_is_not_reported_as_speech(tmp_path):
    pytest.importorskip("silero_vad")
    source = tmp_path / "silence.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=8",
            str(source),
        ],
        check=True,
    )
    args = whisper_args(
        input=source,
        audio_stream=0,
        detector="vad",
        vad_threshold=0.5,
        vad_pad_ms=200,
        vad_min_speech_ms=250,
        vad_min_silence_ms=350,
    )
    assert SileroVadDetector()(args) == []
