# COMMAND LINE FCPXL AND SRT GENERATOR

This repository is a Python script (`generate_srt.py`) that automates subtitle creation for Final Cut Pro.

It does the following in order:
* Downloads or ingests media (local file or YouTube URL)
* Transcribes audio with OpenAI Whisper
* Generates SRT subtitles
* Converts SRT → FCPXML via `srt2subtitles`, a node package

It then generates the following output:
* A working an importable FCPXML file (note: you will have to load the file, and then drop the subtitles onto your clips)
* An SRT file
* A framerate file

---

## Features

* Local files **or** YouTube URLs
* Use batch mode to do multiple files at a time (just put all your files in the /input folder and run --batch)
* Cadence‑aware subtitle chunking
* FPS detection with ffprobe
* Post‑generation FCPXML fixes
* Optional XML line wrapping (`&#10;`)
* Font, size, position, and **line spacing** control

---

## Requirements

### System Dependencies

You must have the following installed:

* **Python 3.9+**
* **ffmpeg** (includes `ffprobe`)
* **Node.js** (for `srt2subtitles`)

#### macOS (recommended)

```bash
brew install ffmpeg node
```

---

### Python Packages

Create and activate a virtual environment:
(Within the folder)

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -U torch openai-whisper yt-dlp
```

---

### Node Dependency (SRT → FCPXML)

Install the CLI globally:

```bash
npm install -g srt2fcpxml-cli
```

This provides the command:

```bash
srt2subtitles
```

---

## Basic Usage

### Single File

```bash
python generate_srt.py video.mp4
```

Item outputs to:

```
output/001/
```

### YouTube URL

```bash
python generate_srt.py https://youtube.com/watch?v=XXXX
```

Item outputs to:

```
output/001/
```

### Batch Mode

Put media files in `input/` and run:

For example
```
input/
├── video.mp4
├── video2.mp4
```

```bash
python generate_srt.py --batch
```

Each item outputs to:

```
output/001/
output/002/
...
```

---

## Output Structure

Each output folder contains:

```text
output/XXX/
├── video.mp4
├── subtitles.srt
├── subtitles.fcpxml
└── framerate.txt
```

`framerate.txt` contains:

```
29.97
100/2997
```

* **Line 1:** decimal FPS (used by Node tools)
* **Line 2:** rational FPS (used to fix Final Cut frameDuration)

---

## Command‑Line Flags

### Styling

| Flag               | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `--font`           | Font family name (e.g. `Helvetica Neue`)                  |
| `--fontsize`       | Font size in pixels                                       |
| `--position "X Y"` | Subtitle position in FCP coordinates                      |
| `--line-spacing`   | Adds `lineSpacing="N"` to text elements (can be negative) |

Example:

```bash
python generate_srt.py clip.mp4 \
  --font "Helvetica Neue" \
  --fontsize 48 \
  --line-spacing -6
```

---

### Cadence / Readability

| Flag             | Description                               |
| ---------------- | ----------------------------------------- |
| `--cadence`      | Enable cadence‑aware splitting            |
| `--max-words`    | Max words per subtitle (default: 7)       |
| `--max-chars`    | Max characters per subtitle (default: 42) |
| `--max-duration` | Max seconds per subtitle (default: 2.6)   |

Example:

```bash
python generate_srt.py clip.mp4 --cadence
```

---

### Line Breaks (XML‑Safe)

| Flag         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `--lb_chars` | Insert `&#10;` line breaks when text exceeds N chars |

This avoids Final Cut layout drift caused by literal newlines.

---

## Batch MP3 + SRT (`batch_mp3_srt.py`)

A lighter, second command for a different job: take a pile of long video files,
spit out an MP3 for each (small enough to upload/share easily) and an SRT
transcript for each, and optionally search every transcript for a key phrase.

Unlike `generate_srt.py`, this does **not** produce FCPXML — it's for
transcription/upload prep, not Final Cut captioning.

### Usage

```bash
python batch_mp3_srt.py video1.mp4 video2.mov video3.mkv
```

Or process a whole folder:

```bash
python batch_mp3_srt.py --input-dir input
```

Search every transcript for a phrase and get timestamped matches back:

```bash
python batch_mp3_srt.py *.mp4 --keyphrase "sign up now"
```

### Output Structure

```text
output/<video-name>/
├── <video-name>.mp3
├── <video-name>.srt
└── keyphrase_matches.txt   # only written if --keyphrase matched something in this file

output/keyphrase_matches.txt  # combined report across all files, only if --keyphrase given
```

### Flags

| Flag                | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `--input-dir DIR`    | Process every media file in a directory (in addition to any files passed directly) |
| `--output-dir DIR`   | Where per-video folders are created (default: `output`)            |
| `--quality`          | `auto` \| `high` \| `medium` \| `low` (default: `auto`)             |
| `--style`            | `sentence` \| `cadence` (default: `sentence`)                      |
| `--bitrate`          | MP3 bitrate (default: `192k`)                                      |
| `--keyphrase TEXT`   | Search every transcript for this phrase, report timestamped matches |
| `--case-sensitive`   | Make `--keyphrase` matching case-sensitive (default: off)          |

It reuses the same Whisper transcription + subtitle-chunking engine as
`generate_srt.py`, so wording/timing/chunking behave identically between the
two tools.

---

## Known Limitations and Author Notes

* Whisper timing is segment‑based (not word‑level unless extended)
* Captions export to FCPXL. Double click the file and it loads the captions as a seperate project in Final Cut. Copy paste these into your project. It is the closest I've been able to get to a one click solution
* The captions will be slighty off or may overlap due to an issue with the framerate in the node package. Frame rates are fractions, and the node packages forces you to use an estimation which is a roudned number.
* Assumes Final Cut Pro XML 1.10+ compatibility
* Requires `srt2fcpxml-cli` (Node)

---

## License

MIT — go for it.
JOE BEGLEY


---

Final Cut can natively generate SRT, but they are non-editable. This is my AI version of a workaround.
