# Whisper Trimmer

Trim an audio or video file down to the parts that contain speech.

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

Useful options:

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --model medium --padding 0.35 --merge-gap 0.7
```

- `--model`: Whisper model to use. `small` is the default. Larger models are slower but often more accurate.
- `--padding`: seconds to keep before and after each detected speech range.
- `--merge-gap`: keep short pauses instead of cutting them out.
- `--dry-run`: show the ranges that would be kept without creating an output file.
- `--segment-timestamps`: use broader Whisper segment timestamps instead of word timestamps.

If you already use OpenAI's `openai-whisper` package, this script can use it too:

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --backend openai-whisper
```

## Trim every media file beside the script

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
