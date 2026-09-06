# Whisper Trimmer

Trim an audio or video file, or a whole folder and its subfolders, down to the parts that contain speech.

The script uses Whisper timestamps to find speech, then uses FFmpeg to remove
the non-speech sections.

## Setup

1. Install FFmpeg and make sure `ffmpeg` and `ffprobe` are available in your terminal.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Usage

```powershell
python whisper_trim.py "input.mp4" "output.mp4"
```

Audio output also works:

```powershell
python whisper_trim.py "input.wav" "speech_only.mp3"
```

### Recursive folder processing

To process a folder and all its subfolders:

```powershell
python whisper_trim.py "D:\Media"
```

Outputs are written beside each source, for example `interview.mp4` becomes
`interview_trimmed.mp4`. To write to a separate folder while preserving the
source folder structure:

```powershell
python whisper_trim.py "D:\Media" "D:\Trimmed Media"
```

- Files whose stems end in `_trimmed` (case-insensitive) are explicitly skipped,
  including outputs created by the older batch scripts with their default suffix.
- An output folder inside the input folder is excluded from the recursive scan.
- Existing outputs are skipped. Use `--overwrite` to replace them.
- Hidden names, symbolic links, empty files, and unrecognized media extensions are skipped.
- Files are processed sequentially. The selected Whisper model is loaded on the
  first transcription and reused for the rest of that run.
- A failed file does not stop the batch. The final summary reports completed,
  skipped, and failed files; any failure produces exit status 1.
- `--dry-run` previews speech ranges without writing media or creating output
  folders. Existing outputs are still skipped unless `--overwrite` is also supplied.

Folder mode keeps each source's extension. The existing renderer uses H.264 for
video and WAV/FLAC/MP3-specific encoders or AAC for audio; not every recognized
input container supports those output codecs (for example, WebM). Such files
may fail during rendering. Use single-file mode with a compatible output such
as `.mp4` for video or `.wav` for audio. Renamed generated files and files made
with custom suffixes by older scripts cannot be recognized as generated outputs.

Single-file mode still requires an explicit output filename and replaces an
existing output by default. Input and output must be different files.

### Tuning

Useful options:

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --model medium --padding 0.35 --merge-gap 0.7
```

- `--model`: Whisper model to use. `small` is the default. Larger models are slower but often more accurate.
- `--padding`: seconds to keep before and after each detected speech range.
- `--merge-gap`: keep short pauses instead of cutting them out.
- `--dry-run`: show the ranges that would be kept without creating an output file.
- `--segment-timestamps`: use broader Whisper segment timestamps instead of word timestamps.
- `--overwrite`: in folder mode, replace existing trimmed outputs.

All transcription and encoding options also apply to recursive folder runs:

```powershell
python whisper_trim.py "D:\Media" --model medium --padding 0.35 --merge-gap 0.7 --dry-run
```

If you already use OpenAI's `openai-whisper` package, this script can use it too:

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --backend openai-whisper
```

## Trim every media file beside the script

This older launcher scans only its own folder. Use `whisper_trim.py "folder"`
for recursive scanning and model reuse across files.

Copy `whisper_trim_all.py` into the folder with the media files, then run:

```powershell
python whisper_trim_all.py
```

Each output is written beside the original with `_trimmed` added to the name,
such as `input.mp4` -> `input_trimmed.mp4`. Existing `_trimmed` files are
ignored so the script does not trim its own outputs again.

The same tuning options are available:

```powershell
python whisper_trim_all.py --model medium --padding 0.35 --merge-gap 0.7
```

Use `--overwrite` to replace existing trimmed outputs.
