import argparse
import shutil
import subprocess

import pytest

import whisper_trim
from whisper_trim import (
    SileroVadDetector,
    align_to_keyframes,
    bitrate_for_target_size,
    build_detector,
    build_filter_script,
    build_select_script,
    detector_padding,
    ffconcat_escape,
    ffprobe,
    media_duration,
    merge_intervals,
    parse_bitrate,
    parse_size,
    passes_needed,
    render_output,
    resolve_encoder,
    resolve_rate_control,
    select_expression,
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


class TestKeyframeAlignment:
    KEYFRAMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_grows_each_range_outwards(self):
        aligned = align_to_keyframes([(1.4, 2.6)], self.KEYFRAMES, duration=6.0)
        assert aligned == [(1.0, 3.0)]

    def test_range_starting_on_a_keyframe_does_not_move(self):
        assert align_to_keyframes([(2.0, 2.5)], self.KEYFRAMES, duration=6.0) == [(2.0, 3.0)]

    def test_range_after_the_last_keyframe_runs_to_the_end(self):
        assert align_to_keyframes([(4.2, 5.5)], self.KEYFRAMES, duration=6.0) == [(4.0, 6.0)]

    def test_first_range_snaps_to_the_first_keyframe(self):
        assert align_to_keyframes([(0.1, 0.4)], self.KEYFRAMES, duration=6.0) == [(0.0, 1.0)]

    def test_overlapping_grown_ranges_are_merged(self):
        # 1.4-2.0 grows to 1.0-2.0 and 2.2-2.6 grows to 2.0-3.0.
        aligned = align_to_keyframes([(1.4, 2.0), (2.2, 2.6)], self.KEYFRAMES, duration=6.0)
        assert aligned == [(1.0, 3.0)]

    def test_no_keyframes_is_an_error(self):
        with pytest.raises(SystemExit):
            align_to_keyframes([(1.0, 2.0)], [], duration=6.0)


class TestSelectScript:
    def test_expression_covers_every_range(self):
        expression = select_expression([(1.0, 2.0), (3.0, 4.5)])
        assert expression == "between(t,1.000000,2.000000)+between(t,3.000000,4.500000)"

    def test_graph_labels_both_outputs(self):
        script = build_select_script([(1.0, 2.0)], include_video=True, include_audio=True)
        assert "[0:v:0]select=" in script
        assert "setpts=N/FRAME_RATE/TB[outv]" in script
        assert "[0:a:0]aselect=" in script
        assert "asetpts=N/SR/TB[outa]" in script

    def test_audio_only_graph_has_no_video_chain(self):
        script = build_select_script([(1.0, 2.0)], include_video=False, include_audio=True)
        assert "[0:v:0]" not in script
        assert "asetpts=N/SR/TB[outa]" in script

    def test_graph_uses_the_requested_tracks(self):
        script = build_select_script([(1.0, 2.0)], True, True, video_stream=1, audio_stream=2)
        assert "[0:v:1]select=" in script
        assert "[0:a:2]aselect=" in script


def video_frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return int(completed.stdout.decode("utf-8", errors="replace").strip())


def video_duration(path: Path) -> float:
    """Duration of the video stream alone; the container also holds audio."""
    for stream in ffprobe(path)["streams"]:
        if stream.get("codec_type") == "video" and stream.get("duration"):
            return float(stream["duration"])
    return media_duration(ffprobe(path))


def assert_frame_rate_intact(path: Path, rate: float = 25.0) -> None:
    """Every frame has to be a real frame, at the source rate.

    A cut that starts mid GOP leaves the decoder unable to emit new pictures
    until the next IDR, so it repeats stale frames instead. That shows up as
    far more frames than the duration can hold, and as a stalled picture: the
    clip measured 41.8fps against a 30fps source before the ranges were grown
    to key frames.

    Two per cent of slack absorbs the frame or two each cut boundary picks up
    from inclusive in and out points, and still leaves plenty of room between
    that and the 39 per cent a stalled cut adds.
    """
    frames = video_frame_count(path)
    seconds = video_duration(path)
    expected = seconds * rate
    assert abs(frames - expected) <= max(2, expected * 0.02), (
        f"{frames} frames in {seconds:.3f}s is {frames / seconds:.2f}fps, "
        f"expected {rate}fps ({expected:.1f} frames)"
    )


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """Six seconds of 25fps video with a key frame every second."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    source = tmp_path_factory.mktemp("media") / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
            "-c:v", "libx264", "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
    )
    return source


def render_args(source: Path, output: Path, **overrides) -> argparse.Namespace:
    values = dict(
        input=source,
        output=output,
        audio_stream=0,
        video_stream=0,
        encoder="libx264",
        rate_control="cq",
        bitrate=None,
        target_size=None,
        video_bitrate=None,
        cq=19,
        crf=23,
        preset="veryfast",
        audio_bitrate="128k",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestRateControl:
    def test_bitrate_suffixes_match_ffmpeg(self):
        assert parse_bitrate("1500k") == 1_500_000
        assert parse_bitrate("2M") == 2_000_000
        assert parse_bitrate("192k") == 192_000
        assert parse_bitrate("2500000") == 2_500_000

    def test_size_suffixes_are_decimal(self):
        assert parse_size("700M") == 700_000_000
        assert parse_size("1.5G") == 1_500_000_000
        assert parse_size("500000") == 500_000

    def test_unreadable_value_is_an_error(self):
        with pytest.raises(SystemExit):
            parse_bitrate("fast")

    def test_target_size_leaves_room_for_audio(self):
        # 700 MB is 5.6 Gbit; 192 kbps of audio over 1000 seconds takes
        # 192 Mbit of it, leaving 5.408 Mbps for video.
        video = bitrate_for_target_size(700_000_000, 1000.0, 192_000)
        assert abs(video - 5_408_000) < 1_000

    def test_target_size_smaller_than_the_audio_is_an_error(self):
        with pytest.raises(SystemExit):
            bitrate_for_target_size(1_000_000, 1000.0, 192_000)

    def test_two_pass_uses_nvenc_vbr_and_a_buffer(self):
        args = argparse.Namespace(cq=19, crf=18, preset="medium", rate_control="2pass",
                                  video_bitrate=1_500_000)
        flags = video_encoder_args("h264_nvenc", args)
        assert flags[flags.index("-rc") + 1] == "vbr_hq"
        assert flags[flags.index("-multipass") + 1] == "fullres"
        assert flags[flags.index("-b:v") + 1] == "1500000"
        assert flags[flags.index("-bufsize") + 1] == "3000000"
        assert "-cq" not in flags

    def test_single_pass_vbr_disables_multipass(self):
        args = argparse.Namespace(cq=19, crf=18, preset="medium", rate_control="vbr",
                                  video_bitrate=1_500_000)
        flags = video_encoder_args("h264_nvenc", args)
        assert flags[flags.index("-multipass") + 1] == "disabled"

    def test_libx264_two_pass_uses_a_bitrate_not_a_crf(self):
        args = argparse.Namespace(cq=19, crf=18, preset="medium", rate_control="2pass",
                                  video_bitrate=1_500_000)
        flags = video_encoder_args("libx264", args)
        assert flags[flags.index("-b:v") + 1] == "1500000"
        assert "-crf" not in flags

    def test_quality_mode_is_unchanged_without_a_bitrate(self):
        args = argparse.Namespace(cq=19, crf=18, preset="medium", rate_control="cq",
                                  video_bitrate=None)
        flags = video_encoder_args("h264_nvenc", args)
        assert flags[flags.index("-cq") + 1] == "19"
        assert flags[flags.index("-b:v") + 1] == "0"

    def test_two_pass_runs_twice(self):
        assert passes_needed(argparse.Namespace(rate_control="2pass", encoder="auto")) == 2
        assert passes_needed(argparse.Namespace(rate_control="vbr", encoder="auto")) == 1
        assert passes_needed(argparse.Namespace(rate_control="cq", encoder="auto")) == 1

    def test_stream_copy_is_never_two_pass(self):
        assert passes_needed(argparse.Namespace(rate_control="2pass", encoder="copy")) == 1


class TestResolveRateControl:
    def base(self, **overrides):
        values = dict(
            encoder="auto", rate_control=None, bitrate=None, target_size=None,
            audio_bitrate="192k", video_bitrate=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_bitrate_alone_selects_single_pass_vbr(self):
        args = self.base(bitrate="1500k")
        resolve_rate_control(args, [(0.0, 100.0)])
        assert args.rate_control == "vbr"
        assert args.video_bitrate == 1_500_000

    def test_target_size_alone_selects_two_pass(self, capsys):
        args = self.base(target_size="700M")
        resolve_rate_control(args, [(0.0, 1000.0)])
        assert args.rate_control == "2pass"
        assert args.video_bitrate > 0
        assert "700M" in capsys.readouterr().out

    def test_no_options_falls_back_to_quality(self):
        args = self.base()
        resolve_rate_control(args, [(0.0, 100.0)])
        assert args.rate_control == "cq"
        assert args.video_bitrate is None

    def test_bitrate_is_required_for_two_pass(self):
        with pytest.raises(SystemExit):
            resolve_rate_control(self.base(rate_control="2pass"), [(0.0, 100.0)])

    def test_stream_copy_ignores_rate_control(self):
        args = self.base(encoder="copy", bitrate="1500k")
        resolve_rate_control(args, [(0.0, 100.0)])
        assert args.rate_control is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_two_pass_render_lands_on_the_target_bitrate(synthetic_clip, tmp_path):
    """Two passes have to actually spend the bits they were given."""
    output = tmp_path / "twopass.mkv"
    args = render_args(
        synthetic_clip, output,
        encoder="libx264", rate_control="2pass", video_bitrate=400_000,
        crf=23, preset="ultrafast",
    )
    render_output(
        input_path=synthetic_clip,
        output_path=output,
        intervals=[(1.1, 4.9)],
        include_video=True,
        include_audio=True,
        args=args,
        duration=6.0,
    )
    assert_frame_rate_intact(output)

    # Matroska does not always report a per stream bit rate, so measure it.
    seconds = video_duration(output)
    audio = next(s for s in ffprobe(output)["streams"] if s["codec_type"] == "audio")
    audio_bytes = int(audio.get("bit_rate", 128_000))
    file_size = output.stat().st_size
    measured = (file_size * 8 - audio_bytes * seconds) / seconds
    assert 0.7 * 400_000 < measured < 1.3 * 400_000, f"measured {measured:.0f} bps"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_rendering_keeps_the_frame_rate(synthetic_clip, tmp_path):
    """The rendered cut must hold exactly the frames the range asked for.

    A copied cut that starts mid GOP leaves the decoder unable to produce new
    frames, which shows up as duplicated frames and a stalled picture.
    """
    output = tmp_path / "out.mp4"
    render_output(
        input_path=synthetic_clip,
        output_path=output,
        intervals=[(1.1, 2.9)],
        include_video=True,
        include_audio=True,
        args=render_args(synthetic_clip, output),
        duration=6.0,
    )
    # 1.1s to 2.9s at 25fps is 45 frames.
    assert abs(video_frame_count(output) - 45) <= 1
    assert_frame_rate_intact(output)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_stream_copy_keeps_the_frame_rate(synthetic_clip, tmp_path):
    output = tmp_path / "copied.mp4"
    aligned = render_output(
        input_path=synthetic_clip,
        output_path=output,
        intervals=[(1.1, 2.9)],
        include_video=True,
        include_audio=True,
        args=render_args(synthetic_clip, output, encoder="copy"),
        duration=6.0,
    )
    # Grown to 1.0s to 3.0s, which is two whole seconds at 25fps.
    assert aligned == [(1.0, 3.0)]
    assert_frame_rate_intact(output)


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
