# Whisper Trimmer

Trim an audio or video file, or a whole folder and its subfolders, down to the parts
that contain speech.

Two detectors are available:

| Detector | Default | What it uses | Best for |
| --- | --- | --- | --- |
| `vad` | yes | Silero voice activity detection | Keeping dialogue. Keeps shouting, singing and overlapping speech, never invents speech over music or noise, runs fully offline |
| `whisper` | no | Whisper speech recognition timestamps | When you also want transcript-based refinement, or to keep the original word-level tightness |

Pick the VAD detector unless you have a reason not to. A speech recognizer is built
to produce text, not to report whether a voice is present, so it tends to miss
shouted or overlapping lines and to hallucinate words over music, ambience and sound
effects.

## Setup

1. Install FFmpeg and make sure `ffmpeg` and `ffprobe` are available in your terminal.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Python 3.10 to 3.13 works well. If `ctranslate2` or `torch` refuse to install, you are
most likely on a brand new Python; create a virtual environment with 3.12 instead.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```powershell
python whisper_trim.py "input.mp4" "output.mp4"
```

Audio output also works:

```powershell
python whisper_trim.py "input.wav" "speech_only.mp3"
```

Check what would be kept before rendering anything:

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --dry-run
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
- Files are processed sequentially. The loaded model is reused for the rest of that run.
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

### Keeping a record of what was cut

`--list-dir` writes two files per source, mirroring the input folder structure:

- `<name>.ffconcat`, which `ffmpeg -f concat -i` and most editors can open directly
- `<name>.csv`, with the original timecode of every kept range

Because the output is a new condensed timeline, the timecodes in the output no longer
match the source. The CSV is how you find a scene again.

```powershell
python whisper_trim.py "D:\Media" --list-dir "D:\Cut Lists"
```

## Tuning

### Detector

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --detector vad --vad-threshold 0.4
```

- `--detector`: `vad` (default) or `whisper`.
- `--vad-threshold`: speech probability from 0 to 1, default `0.5`. Lower keeps
  quieter speech and more of a noisy scene; raise it to stop music and effects from
  being kept.
- `--vad-pad-ms`: milliseconds kept before and after each range, default `200`.
- `--vad-min-speech-ms`: ranges shorter than this are dropped, default `250`.
- `--vad-min-silence-ms`: ranges separated by this much silence are joined,
  default `350`.

Silero applies all four itself, so the Whisper-era `--padding`, `--merge-gap` and
`--min-duration` do not apply in VAD mode.

Singing is speech as far as a voice activity model is concerned, so theme songs and
background vocals are kept. That is usually what you want from a dialogue edit.

### Whisper detector

```powershell
python whisper_trim.py "input.mp4" "output.mp4" --detector whisper --model medium --padding 0.35 --merge-gap 0.7
```

- `--model`: Whisper model to use. `small` is the default. Larger models are slower but often more accurate.
- `--backend`: `auto` (default), `faster-whisper` or `openai-whisper`.
- `--padding`: seconds to keep before and after each detected speech range.
- `--merge-gap`: keep short pauses instead of cutting them out.
- `--min-duration`: discard ranges shorter than this before padding.
- `--segment-timestamps`: use broader Whisper segment timestamps instead of word timestamps.
- `--language`, `--device`, `--compute-type`, `--beam-size`: passed to Whisper.
  `--vad-filter` additionally applies faster-whisper's own VAD before recognition.
- `--dry-run`: show the ranges that would be kept without creating an output file.

Whisper models are downloaded from Hugging Face the first time they are used. If that
fails behind a proxy, set `HTTPS_PROXY` before running.

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
python whisper_trim.py "input.mp4" "output.mp4" --detector whisper
```

`--device cuda` also needs the CUDA 12 runtime libraries on the path, which PyTorch
does not always provide on Windows. If you get `cublas64_12.dll is not found`, either
install the NVIDIA runtime or fall back to `--device cpu --compute-type int8`. The VAD
detector has neither problem: it needs no download and runs on the CPU.

### Encoding

`--encoder auto` (the default) asks FFmpeg which video encoders it has and uses the
first hardware encoder it finds, falling back to `libx264`. Rendering a season on a
GPU instead of the CPU is the single biggest difference in total runtime.

```powershell
python whisper_trim.py "D:\Media" --encoder auto
```

- `--encoder`: `auto`, `copy`, `libx264`, `libx265`, `h264_nvenc`, `hevc_nvenc`,
  `h264_qsv`, `h264_vaapi`, `hevc_vaapi`. Short names `nvenc`, `qsv`, `vaapi`, `x264`
  and `x265` also work.
- `--crf`: quality for `libx264` and `libx265`, default `18`.
- `--cq`: quality for hardware encoders, default `19`. NVENC needs a higher number
  than x264 for a similar file size; `21` to `23` is a good starting point for
  re-encoding material that was already compressed.
- `--preset`: `medium` by default. For `libx264` and `libx265` this is an x264-style
  name such as `veryfast`; hardware encoders use their own presets.
- `--audio-bitrate`: default `192k`.

#### Cutting without re-encoding

`--encoder copy` hands the ranges to FFmpeg's concat demuxer instead of re-encoding.
On a five minute test clip it took 0.9 seconds against 18.7 seconds for NVENC, and
nothing is lost, because no frame is decoded and encoded again.

```powershell
python whisper_trim.py "D:\Media" --encoder copy
```

The trade-off is accuracy: every range snaps to the nearest key frame before it, so a
cut can start up to one GOP early, the output is slightly longer than the re-encoded
one, and audio and video are not frame accurate against each other. Re-encode when
the cuts have to land exactly, stream copy when you want the result quickly.

The output container has to be able to hold the source codecs. Copying H.264 and AAC
into `.mp4` or `.mkv` is fine. Audio-only output is always re-encoded, because stream
copy cannot apply the ranges as filters.

### Streams and dry runs

- `--audio-stream`, `--video-stream`: which track to work on, counting from `0`.
  Useful for sources with several audio tracks.
- `--dry-run`: print the ranges that would be kept without writing media.

All of these also apply to recursive folder runs:

```powershell
python whisper_trim.py "D:\Media" --vad-threshold 0.45 --vad-min-silence-ms 500 --encoder auto --list-dir "D:\Cut Lists" --dry-run
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
