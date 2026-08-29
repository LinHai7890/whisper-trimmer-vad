import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import whisper_trim


class MediaMetadataTests(unittest.TestCase):
    def test_media_duration_prefers_format_duration(self) -> None:
        probe = {
            "format": {"duration": "12.5"},
            "streams": [{"duration": "11.0"}],
        }

        self.assertEqual(whisper_trim.media_duration(probe), 12.5)

    def test_media_duration_uses_longest_stream(self) -> None:
        probe = {"streams": [{"duration": "3.0"}, {"duration": "4.5"}]}

        self.assertEqual(whisper_trim.media_duration(probe), 4.5)

    def test_has_stream_detects_requested_type(self) -> None:
        probe = {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]}

        self.assertTrue(whisper_trim.has_stream(probe, "audio"))
        self.assertFalse(whisper_trim.has_stream(probe, "subtitle"))


class IntervalTests(unittest.TestCase):
    def test_merge_intervals_filters_pads_clamps_and_merges(self) -> None:
        result = whisper_trim.merge_intervals(
            [(-0.1, 0.02), (0.2, 0.6), (0.8, 1.1), (4.8, 5.2)],
            duration=5.0,
            padding=0.1,
            merge_gap=0.2,
            min_duration=0.08,
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0][0], 0.0)
        self.assertAlmostEqual(result[0][1], 1.2)
        self.assertAlmostEqual(result[1][0], 4.7)
        self.assertAlmostEqual(result[1][1], 5.0)

    def test_merge_intervals_returns_empty_for_invalid_ranges(self) -> None:
        result = whisper_trim.merge_intervals(
            [(1.0, 1.0), (2.0, 1.0)],
            duration=5.0,
            padding=0.25,
            merge_gap=0.45,
            min_duration=0.08,
        )

        self.assertEqual(result, [])


class FfmpegArgumentTests(unittest.TestCase):
    def test_filter_script_contains_video_and_audio_outputs(self) -> None:
        script = whisper_trim.build_filter_script(
            [(1.0, 2.5), (4.0, 5.0)], include_video=True, include_audio=True
        )

        self.assertIn("[0:v:0]trim=start=1.000000:end=2.500000", script)
        self.assertIn("[0:a:0]atrim=start=4.000000:end=5.000000", script)
        self.assertIn("concat=n=2:v=1:a=1[outv][outa]", script)

    def test_audio_codec_matches_output_extension(self) -> None:
        self.assertEqual(
            whisper_trim.audio_codec_args(Path("output.wav"), "192k"),
            ["-c:a", "pcm_s16le"],
        )
        self.assertEqual(
            whisper_trim.audio_codec_args(Path("output.m4a"), "128k"),
            ["-c:a", "aac", "-b:a", "128k"],
        )


class CommandSafetyTests(unittest.TestCase):
    def test_main_rejects_same_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            media_path = Path(temporary_directory) / "recording.wav"
            media_path.touch()
            arguments = argparse.Namespace(
                input=media_path,
                output=media_path,
                padding=0.25,
                merge_gap=0.45,
                min_duration=0.08,
            )

            with patch.object(whisper_trim, "parse_args", return_value=arguments):
                with self.assertRaisesRegex(SystemExit, "must be different"):
                    whisper_trim.main()


if __name__ == "__main__":
    unittest.main()
