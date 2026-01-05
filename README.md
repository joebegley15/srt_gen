# generate_srt.py — SRT → Final Cut Pro XML Subtitle Pipeline

This repository is a Python script (`generate_srt.py`) that automates subtitle creation for Final Cut Pro.

It can:
* Download or ingest media (local file or YouTube URL)
* Transcribe audio with OpenAI Whisper
* Generate SRT subtitles
* Convert SRT → FCPXML via `srt2subtitles`, a node package

The output is a **drop‑in Final Cut Pro subtitle generator** designed for clips

---

## Features

* ✅ Local files **or** YouTube URLs
* ✅ Use batch mode to do multiple files at a time (just put all your files in the /input folder and run --batch)
* ✅ Cadence‑aware subtitle chunking
* ✅ FPS detection with ffprobe
* ✅ Post‑generation FCPXML fixes
* ✅ Optional XML line wrapping (`&#10;`)
* ✅ Font, size, position, and **line spacing** control

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

## Environment Variables (Optional)

If you prefer an `.env` file (not required by default):

```bash
OPENAI_API_KEY=your_key_here
```

Source it manually if needed:

```bash
source .env
```

> Note: Whisper runs **locally** — no API key is required unless you extend the script.

---

## Basic Usage

### Single File

```bash
python generate_srt.py video.mp4
```

### YouTube URL

```bash
python generate_srt.py https://youtube.com/watch?v=XXXX
```

### Batch Mode

Put media files in `input/` and run:

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

## Why This Script Exists

Final Cut Pro is **extremely sensitive** to:

* FPS rounding (23.976 vs 24)
* Frame duration mismatches
* Subtitle overlap from fractional drift

This pipeline:

* Detects real FPS
* Preserves NTSC precision
* Fixes `frameDuration` post‑generation
* Keeps pasted subtitles frame‑accurate

Result: **no overlap, no drift, no manual fixes**.

---

## Known Limitations and Author Notes

* Whisper timing is segment‑based (not word‑level unless extended)
* Captions export to FCPXL. Double click the file and it loads the captions as a seperate project in Final Cut. Copy paste these into your project. It is the closest I've been able to get to a one click solution
* The captions will be slighty off or may overlap due to an issue with the framerate in the node package.
* Assumes Final Cut Pro XML 1.10+ compatibility
* Requires `srt2fcpxml-cli` (Node)

---

## License

MIT — go for it.
JOE BEGLEY


---

Final Cut can natively generate SRT, but they are non-editable. This is my AI version of a workaround.
