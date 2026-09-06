# Whisper Trimmer Script Guide

These scripts use Whisper timestamps to identify speech in audio or video, then use FFmpeg to remove the non-speech portions. Folder modes write separate outputs and skip existing outputs unless `--overwrite` is supplied. Single-file mode in `whisper_trim.py` replaces an existing output by default and rejects using the input file as the output.

## At a glance

| File | Purpose | Input location | Output location |
| --- | --- | --- | --- |
| `whisper_trim.py` | Trims one file or recursively processes a folder, reusing one model per run | A file or folder supplied on the command line | Explicit file path, or `_trimmed` files beside sources / in a supplied output folder |
| `whisper_trim_all.py` | Trims every supported media file beside the script; fully self-contained | The script's own folder | Beside each original, with `_trimmed` added |
| `whisper_trim_all_dependant.py` | Trims every supported media file beside the script; reuses `whisper_trim.py` | The script's own folder | Beside each original, with `_trimmed` added |
| `trim_parent_folder.bat` | Sends every supported media file in the scripts' parent folder through `whisper_trim.py` | The folder one level above the batch file | A `trimmed` subfolder in that parent folder |
| `trim_parent_folder_medium.bat` | Runs `trim_parent_folder.bat` with a preset for more accurate, less aggressive trimming | Same as `trim_parent_folder.bat` | Same as `trim_parent_folder.bat` |

## `whisper_trim.py`

This is the core trimmer, supporting both single files and recursive folder processing. It also supplies shared functions used by some of the other scripts.

It:

1. Checks the input with `ffprobe` to determine its duration and whether it contains audio and video.
2. Lazily loads a Whisper model on the first transcription and reuses it for subsequent files in the same folder run. With `--backend auto`, it tries `faster-whisper` and falls back to `openai-whisper` if the first backend cannot be imported or raises a runtime error during loading. Transcription errors are reported for the affected file without switching models.
3. Collects word timestamps by default, or broader segment timestamps with `--segment-timestamps`.
4. Discards extremely short detections, adds padding around speech, and joins ranges separated by short pauses.
5. Uses FFmpeg to extract and concatenate the retained ranges.
6. Encodes video as H.264 and chooses an appropriate audio codec based on the output extension.

Basic use:

```powershell
python whisper_trim.py "input.mp4" "output.mp4"
```

Single-file mode requires both an input and an output path. It can also produce audio-only output, perform a dry run, select the Whisper backend/model/language/device, and adjust detection or encoding settings.

For recursive folder processing, supply a folder instead of a file:

```powershell
python whisper_trim.py "D:\Media"
python whisper_trim.py "D:\Media" "D:\Trimmed Media" --model medium
```

With no output folder, each output is saved beside its original with `_trimmed`
added before the extension. With an output folder, the source's relative folder
structure is preserved: `D:\Media\Day 1\clip.mp4` becomes
`D:\Trimmed Media\Day 1\clip_trimmed.mp4`.

The recursive scan takes a snapshot before processing and:

- Skips any filename stem ending in `_trimmed`, case-insensitively, so default
  outputs from this script and the older Python batch scripts are not reprocessed.
- Excludes the destination folder when it is nested inside the source folder.
- Skips hidden names, symbolic links, empty files, and unrecognized extensions.

Files generated under other names or older custom output suffixes are not
automatically identified as outputs. Move those outside the source tree if needed.

Existing destinations are skipped unless `--overwrite` is supplied. All existing
transcription and encoding options apply to every file. Processing is sequential,
so a single model is reused without concurrent calls into it; there is no
`--workers` option. No model is loaded if every file is skipped before transcription.

```powershell
python whisper_trim.py "D:\Media" --dry-run
python whisper_trim.py "D:\Media" --overwrite
```

A dry run prints ranges and creates no output media or destination folders. It
still skips existing outputs unless combined with `--overwrite`. Files with no
detected speech are counted as skipped. Other per-file errors are counted as
failures, and processing continues. The final summary reports completed (or
previewed), skipped, and failed totals. Exit status is 1 if any file failed,
otherwise 0, including when no eligible source files are found.

Folder mode retains source extensions; it does not add new encoders. Video is
encoded once as H.264; audio uses PCM for `.wav`, FLAC for `.flac`, MP3 for `.mp3`,
and AAC otherwise. Some recognized containers, such as WebM, cannot hold these
codecs and may fail. For those files, use single-file mode with a compatible
output extension such as `.mp4` or `.wav`.

## `whisper_trim_all.py`

This is a self-contained batch version. Place it in a folder containing media and run it without input or output filenames:

```powershell
python whisper_trim_all.py
```

It scans only that folder, not its subfolders. For every supported audio or video file, it creates a sibling output such as:

```text
interview.mp4 -> interview_trimmed.mp4
```

Files whose names already end in the configured output suffix are ignored, preventing the script from repeatedly trimming its own results. Existing outputs are skipped unless `--overwrite` is supplied. A different suffix can be selected with `--output-suffix`.

This file contains its own copies of the probing, Whisper, interval-merging, and FFmpeg-rendering functions. It therefore does not need `whisper_trim.py` to be present.

## `whisper_trim_all_dependant.py`

This performs the same folder-wide job and supports the same options as `whisper_trim_all.py`, but imports its core functions from `whisper_trim.py` instead of duplicating them.

Both files must remain together:

```text
whisper_trim.py
whisper_trim_all_dependant.py
```

Run it with:

```powershell
python whisper_trim_all_dependant.py
```

This version is easier to maintain because shared trimming behavior lives in one place, but it will fail to start if `whisper_trim.py` is absent or cannot be imported. The filename uses the spelling `dependant`; it means this script is dependent on `whisper_trim.py`.

The two older `whisper_trim_all` launchers still scan only their own folder and
load a model per transcription. Use the core script's folder mode for recursion
and model reuse; its one-argument `transcribe(args)` interface remains compatible
with the dependent launcher.

## `trim_parent_folder.bat`

This Windows launcher processes media located one folder above the folder containing the batch file. For example, with this layout:

```text
Media Folder/
|-- recording.mp4
`-- Whisper Trimmer/
    |-- trim_parent_folder.bat
    `-- whisper_trim.py
```

running the batch file creates:

```text
Media Folder/
|-- recording.mp4
`-- trimmed/
    `-- recording_trimmed.mp4
```

It checks that `whisper_trim.py` exists beside it and that Python is available, creates the `trimmed` output folder, processes recognized media extensions, skips outputs that already exist, and prints found/done/skipped/failed totals. Extra command-line options are passed directly to `whisper_trim.py`.

Example:

```powershell
.\trim_parent_folder.bat --model small --dry-run
```

Unlike the Python batch scripts, it calls `whisper_trim.py` once per file and supports a somewhat narrower hard-coded list of extensions.

## `trim_parent_folder_medium.bat`

This is a convenience preset. It calls `trim_parent_folder.bat` with:

```text
--model medium --padding 0.35 --merge-gap 0.7
```

Compared with the defaults, it uses the larger `medium` Whisper model, retains 0.35 seconds around detected speech, and preserves pauses of up to 0.7 seconds between speech ranges. It is likely to be slower and require more memory, but it may detect speech more accurately and make less abrupt cuts.

## Shared requirements

All five entry points ultimately require:

- Python
- FFmpeg, including both `ffmpeg` and `ffprobe`, available on `PATH`
- A supported Whisper package (`faster-whisper`, or `openai-whisper` when selected or used as a fallback)

The Python dependencies listed by this project can be installed with:

```powershell
python -m pip install -r requirements.txt
```

The Whisper model is downloaded the first time it is used if it is not already cached.
