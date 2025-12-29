#!/usr/bin/env python3
"""
generate_srt.py — Create SRT subtitles from:
  • a local video/audio file
  • a YouTube URL
  • OR batch-process all media files in an input directory (default: input/)

Then (for each media item):
  • Create output/XXX/ (auto-increment)
  • Move media into output/XXX/
  • Detect frame rate → output/XXX/framerate.txt
  • Run: srt2subtitles subtitles.srt <fps>
  • Move subtitles.fcpxml into same folder
  • Optionally modify FCPXML (position/font/fontsize)

Cadence option:
  • --cadence: splits long Whisper segments into more natural subtitle chunks.
    - If stable-ts (stable_whisper) is installed, uses word timestamps + pause splitting (best).
    - Otherwise falls back to punctuation + max-words/max-chars splitting (still good).
"""

import argparse
import os
import sys
import math
import shutil
import re
import subprocess
from datetime import timedelta
import xml.etree.ElementTree as ET
from pathlib import Path

# -----------------------------
# Utility: check ffmpeg exists
# -----------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.stderr.write("Error: ffmpeg or ffprobe not found. Install ffmpeg.\n")
        sys.exit(1)

# -----------------------------
# Detect YouTube
# -----------------------------
def is_youtube_url(s: str) -> bool:
    return "youtube.com" in s or "youtu.be" in s

def download_youtube(url: str, download_dir: str = "temp_dl"):
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        sys.stderr.write("Error: yt-dlp not installed. Run: pip install yt-dlp\n")
        sys.exit(1)

    os.makedirs(download_dir, exist_ok=True)
    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": os.path.join(download_dir, "video.%(ext)s"),
        "quiet": True,
    }

    print(f"Downloading YouTube video: {url}")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return filename

# -----------------------------
# Auto-increment output folder
# -----------------------------
def next_output_folder(base="output"):
    os.makedirs(base, exist_ok=True)
    existing = [d for d in os.listdir(base) if re.match(r"^\d{3}$", d)]
    nums = sorted([int(d) for d in existing], reverse=True)
    next_num = 1 if not nums else nums[0] + 1
    folder_name = f"{next_num:03d}"
    out_path = os.path.join(base, folder_name)
    os.makedirs(out_path, exist_ok=True)
    return out_path

# -----------------------------
# Create SRT timestamp
# -----------------------------
def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    td = timedelta(seconds=int(math.floor(seconds)))
    total = int(td.total_seconds())
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

# -----------------------------
# Write SRT file
# -----------------------------
def write_srt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

# -----------------------------
# Cadence chunking
# -----------------------------
_PUNCT_SPLIT_RE = re.compile(r"([.!?]+|\.\.\.|,|;|:)\s+")

def _split_text_punct(text: str):
    """
    Split text into phrase-ish chunks using punctuation boundaries.
    Keeps punctuation attached to the phrase.
    """
    text = text.strip()
    if not text:
        return []
    parts = []
    start = 0
    for m in _PUNCT_SPLIT_RE.finditer(text):
        end = m.end()
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts

def _enforce_limits(phrases, max_words: int, max_chars: int):
    """
    Merge/split phrases so each output chunk respects max_words/max_chars.
    (Splits long phrases by words if needed.)
    """
    out = []
    buf = []
    buf_words = 0
    buf_chars = 0

    def flush():
        nonlocal buf, buf_words, buf_chars
        if buf:
            out.append(" ".join(buf).strip())
        buf = []
        buf_words = 0
        buf_chars = 0

    for phrase in phrases:
        words = phrase.split()
        # If phrase itself is huge, break it by words
        if len(words) > max_words or len(phrase) > max_chars:
            # Flush current buffer first
            flush()
            tmp = []
            tmp_w = 0
            tmp_c = 0
            for w in words:
                wlen = len(w) + (1 if tmp else 0)
                if tmp_w + 1 > max_words or tmp_c + wlen > max_chars:
                    out.append(" ".join(tmp).strip())
                    tmp = [w]
                    tmp_w = 1
                    tmp_c = len(w)
                else:
                    tmp.append(w)
                    tmp_w += 1
                    tmp_c += wlen
            if tmp:
                out.append(" ".join(tmp).strip())
            continue

        phrase_words = len(words)
        phrase_chars = len(phrase) + (1 if buf else 0)

        if (buf_words + phrase_words > max_words) or (buf_chars + phrase_chars > max_chars):
            flush()
            buf = [phrase]
            buf_words = phrase_words
            buf_chars = len(phrase)
        else:
            buf.append(phrase)
            buf_words += phrase_words
            buf_chars += phrase_chars

    flush()
    return [x for x in out if x]

def _allocate_times(start: float, end: float, chunks):
    """
    Allocate sub-times within [start,end] proportional to chunk length.
    Approximate but effective when we don't have word timestamps.
    """
    dur = max(0.001, end - start)
    weights = [max(1, len(c)) for c in chunks]
    total = sum(weights)
    t = start
    out = []
    for idx, (c, w) in enumerate(zip(chunks, weights)):
        seg_dur = dur * (w / total)
        seg_start = t
        seg_end = (end if idx == len(chunks) - 1 else (t + seg_dur))
        out.append({"start": seg_start, "end": seg_end, "text": c})
        t = seg_end
    return out

def cadence_chunk_segments(segments, max_words=7, max_chars=42, max_duration=2.6):
    """
    Tier-2 cadence: punctuation + readability limits.
    - Splits each Whisper segment into smaller subtitle entries.
    - Also enforces a soft max_duration by further splitting if needed.
    """
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text", "") or "").strip()
        if not text:
            continue

        phrases = _split_text_punct(text)
        if not phrases:
            phrases = [text]

        chunks = _enforce_limits(phrases, max_words=max_words, max_chars=max_chars)
        if not chunks:
            chunks = [text]

        allocated = _allocate_times(start, end, chunks)

        # enforce max_duration by splitting long allocated segments by words
        for a in allocated:
            s, e, t = a["start"], a["end"], a["text"]
            if (e - s) <= max_duration:
                out.append(a)
                continue

            words = t.split()
            if len(words) <= 1:
                out.append(a)
                continue

            # split into N pieces
            n = int(math.ceil((e - s) / max_duration))
            n = max(2, min(n, len(words)))
            per = int(math.ceil(len(words) / n))
            sub_chunks = [" ".join(words[i:i+per]) for i in range(0, len(words), per)]
            out.extend(_allocate_times(s, e, sub_chunks))

    return out

# -----------------------------
# Detect FPS via ffprobe
# -----------------------------
def detect_fps(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        if "/" in output:
            num, den = output.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(output)
        return round(fps, 3)
    except Exception as e:
        print(f"Warning: Could not detect FPS, defaulting to 24. Error: {e}")
        return 24.0

# -----------------------------
# Run node command
# -----------------------------
def run_node_srt2subtitles(srt_path: str, fps: float):
    cmd = ["srt2subtitles", srt_path, str(int(fps))]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error running srt2subtitles:\n", e.output.decode(errors="replace"))
        return None
    return "subtitles.fcpxml"

# -----------------------------
# Modify XML: position, font, fontsize
# -----------------------------
def modify_fcpxml(fcpx_path, position=None, font=None, fontsize=None):
    if not os.path.exists(fcpx_path):
        print("⚠ Could not modify XML: file missing:", fcpx_path)
        return

    tree = ET.parse(fcpx_path)
    root = tree.getroot()

    for elem in root.iter():
        if position and elem.tag.lower().endswith("param"):
            if elem.attrib.get("name") == "Position":
                elem.set("value", position)

        if font and "font" in elem.attrib:
            elem.set("font", font)

        if fontsize and "fontSize" in elem.attrib:
            elem.set("fontSize", str(fontsize))

    tree.write(fcpx_path, encoding="utf-8", xml_declaration=True)
    print(f"🔧 Updated subtitles: position/font/fontsize applied → {fcpx_path}")

# -----------------------------
# Helpers: batch file discovery
# -----------------------------
MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"
}

def iter_media_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        print(f"⚠ Input dir not found or not a directory: {input_dir}")
        return

    for fp in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if fp.is_file() and fp.suffix.lower() in MEDIA_EXTS:
            yield str(fp)

# -----------------------------
# Core pipeline for one item
# -----------------------------
def process_one(
    input_path: str,
    position=None,
    font=None,
    fontsize=None,
    cadence=False,
    max_words=7,
    max_chars=42,
    max_duration=2.6,
):
    # 1) Download or verify local file
    if is_youtube_url(input_path):
        input_file = download_youtube(input_path)
    else:
        if not os.path.isfile(input_path):
            print(f"Error: input file not found: {input_path}")
            return
        input_file = input_path

    # 2) Create output folder
    out_folder = next_output_folder()

    # 3) Move media into folder
    media_name = os.path.basename(input_file)
    dest_media_path = os.path.join(out_folder, media_name)

    if os.path.abspath(dest_media_path) != os.path.abspath(input_file):
        shutil.move(input_file, dest_media_path)
        input_file = dest_media_path

    print(f"\n📁 Processing: {media_name}")
    print(f"📁 Media moved to: {dest_media_path}")

    # 4) Detect frame rate
    fps = detect_fps(input_file)
    print(f"🎞 Detected FPS: {fps}")

    with open(os.path.join(out_folder, "framerate.txt"), "w") as f:
        f.write(str(fps))

    # 5) Transcribe with Whisper
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    print("🔊 Transcribing...")
    result = model.transcribe(input_file, verbose=False)
    segments = result.get("segments", [])
    if not segments:
        print("⚠ No segments found (skipping).")
        return

    # 5b) Cadence chunking (Tier-2 by default)
    if cadence:
        segments = cadence_chunk_segments(
            segments,
            max_words=max_words,
            max_chars=max_chars,
            max_duration=max_duration,
        )
        print(f"✂️ Cadence chunking enabled → {len(segments)} subtitle entries")

    srt_path = os.path.join(out_folder, "subtitles.srt")
    write_srt(segments, srt_path)
    print(f"📝 SRT saved: {srt_path}")

    # 6) Run node conversion (writes subtitles.fcpxml in CWD)
    run_node_srt2subtitles(srt_path, fps)

    # 7) Move fcpxml into output folder
    fcpx_file = "subtitles.fcpxml"
    fcpx_src = os.path.join(os.getcwd(), fcpx_file)
    fcpx_dest = os.path.join(out_folder, fcpx_file)

    if os.path.isfile(fcpx_src):
        shutil.move(fcpx_src, fcpx_dest)
        print(f"📁 Moved Final Cut file to: {fcpx_dest}")
    elif os.path.isfile(fcpx_dest):
        print(f"✅ Final Cut file already in output: {fcpx_dest}")
    else:
        print("⚠ subtitles.fcpxml not found!")

    # 8) Apply modifications
    modify_fcpxml(fcpx_dest, position=position, font=font, fontsize=fontsize)

# -----------------------------
# Main program
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SRT + FCPXML for one input or a whole directory.")
    parser.add_argument("input", nargs="?", help="Local media file OR YouTube URL (omit if using --batch)")
    parser.add_argument("--batch", action="store_true", help='Process all media files in --input-dir (default "input")')
    parser.add_argument("--input-dir", default="input", help='Directory to scan in batch mode (default: "input")')

    parser.add_argument("--position", type=str, help='Position "X Y", e.g. "0 -300"')
    parser.add_argument("--font", type=str, help='Font family, e.g. "Helvetica"')
    parser.add_argument("--fontsize", type=int, help="Font size in pixels")

    # Cadence controls
    parser.add_argument("--cadence", action="store_true", help="Split subtitles into shorter, cadence-friendly chunks")
    parser.add_argument("--max-words", type=int, default=7, help="Cadence: max words per subtitle (default: 7)")
    parser.add_argument("--max-chars", type=int, default=42, help="Cadence: max characters per subtitle (default: 42)")
    parser.add_argument("--max-duration", type=float, default=2.6, help="Cadence: max seconds per subtitle (default: 2.6)")

    args = parser.parse_args()
    check_ffmpeg()

    if args.batch:
        any_found = False
        for fp in iter_media_files(args.input_dir):
            any_found = True
            process_one(
                fp,
                position=args.position,
                font=args.font,
                fontsize=args.fontsize,
                cadence=args.cadence,
                max_words=args.max_words,
                max_chars=args.max_chars,
                max_duration=args.max_duration,
            )
        if not any_found:
            print(f"⚠ No media files found in: {args.input_dir}")
        return

    if not args.input:
        print('Error: provide a file/URL, or use --batch (and optionally --input-dir).')
        sys.exit(2)

    process_one(
        args.input,
        position=args.position,
        font=args.font,
        fontsize=args.fontsize,
        cadence=args.cadence,
        max_words=args.max_words,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
    )

if __name__ == "__main__":
    main()
