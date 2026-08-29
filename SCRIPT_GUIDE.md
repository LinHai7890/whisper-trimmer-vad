# Whisper Trimmer Script Guide

These scripts use Whisper timestamps to identify speech in audio or video, then use FFmpeg to remove the non-speech portions. The original media is not modified unless an existing output is deliberately replaced with the `--overwrite` option supported by the batch Python scripts.

## At a glance

| File | Purpose | Input location | Output location |
| --- | --- | --- | --- |
| `whisper_trim.py` | Trims one explicitly named media file | Any path supplied on the command line | Any path supplied on the command line |
| `whisper_trim_all.py` | Trims every supported media file beside the script; fully self-contained | The script's own folder | Beside each original, with `_trimmed` added |
| `whisper_trim_all_dependant.py` | Trims every supported media file beside the script; reuses `whisper_trim.py` | The script's own folder | Beside each original, with `_trimmed` added |
| `trim_parent_folder.bat` | Sends every supported media file in the scripts' parent folder through `whisper_trim.py` | The folder one level above the batch file | A `trimmed` subfolder in that parent folder |
| `trim_parent_folder_medium.bat` | Runs `trim_parent_folder.bat` with a preset for more accurate, less aggressive trimming | Same as `trim_parent_folder.bat` | Same as `trim_parent_folder.bat` |

## `whisper_trim.py`

This is the core single-file trimmer and the shared implementation used by some of the other scripts.

It:

1. Checks the input with `ffprobe` to determine its duration and whether it contains audio and video.
2. Uses `faster-whisper` when available, falling back to `openai-whisper` when the backend is set to `auto`.
3. Collects word timestamps by default, or broader segment timestamps with `--segment-timestamps`.
4. Discards extremely short detections, adds padding around speech, and joins ranges separated by short pauses.
5. Uses FFmpeg to extract and concatenate the retained ranges.
6. Encodes video as H.264 and chooses an appropriate audio codec based on the output extension.

Basic use:

```powershell
python whisper_trim.py "input.mp4" "output.mp4"
```

It requires both an input and an output path. It can also produce audio-only output, perform a dry run, select the Whisper backend/model/language/device, and adjust detection or encoding settings.

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
