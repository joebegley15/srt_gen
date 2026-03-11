#!/usr/bin/env python3
"""
generate_srt.py

Create SRT subtitles from:
  • a local video/audio file
  • a YouTube URL
  • OR batch-process all media files in an input directory (default: input/)

Then (for each media item):
  • Create output/XXX/ (auto-increment)
  • Move media into output/XXX/
  • Detect frame rate → output/XXX/framerate.txt
  • Run: srt2subtitles subtitles.srt <fps>
  • Move subtitles.fcpxml into same folder
  • Optionally modify FCPXML (position/font/fontsize/lineSpacing)
  • Optional FCPXML line breaks: if --line-break-chars is provided, wrap long text
    nodes by inserting XML line-break entity '&#10;' at word boundaries.

Subtitle styles:
  • sentence
      - smoother, more natural subtitle flow
      - breaks primarily at sentence endings using Whisper word timestamps
  • cadence
      - punchier, shorter caption chunks for social/short-form usage

YouTube options:
  • --ytmp3
      - For YouTube URLs, download audio-only as MP3 instead of video.
  • --timestamp "HH:MM:SS-HH:MM:SS"
      - For YouTube URLs, attempt to download only that segment.
        If yt-dlp/ffmpeg fails, falls back to downloading and trimming locally.
      - For local media, trims with ffmpeg before transcription.
  • --cookies <path>
      - Pass a Netscape cookies.txt file to yt-dlp.
  • --cookies-from-browser "chrome[:PROFILE]" or "firefox[:PROFILE]"
      - Let yt-dlp read cookies directly from your browser profile.
  • --remote-ejs
      - Enables yt-dlp remote EJS challenge solver distribution (ejs:github).
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

from youtube_download import is_youtube_url, download_youtube

# -----------------------------
# Utility: check ffmpeg exists
# -----------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.stderr.write("Error: ffmpeg or ffprobe not found. Install ffmpeg.\n")
        sys.exit(1)

# -----------------------------
# Timestamp parsing
# -----------------------------
_HMS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}$")

def _hms_to_seconds(hms: str) -> int:
    if not _HMS_RE.match(hms):
        raise ValueError(f"Invalid timecode: {hms}")
    hh, mm, ss = hms.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def parse_timestamp_range(ts: str):
    """
    Parse "HH:MM:SS-HH:MM:SS"
    Returns (start_hms, end_hms, start_seconds, end_seconds) or None.
    """
    if not ts:
        return None
    ts = ts.strip()
    if not _TS_RE.match(ts):
        raise ValueError('Invalid --timestamp format. Expected "HH:MM:SS-HH:MM:SS".')
    start_hms, end_hms = ts.split("-", 1)
    start_s = _hms_to_seconds(start_hms)
    end_s = _hms_to_seconds(end_hms)
    if end_s <= start_s:
        raise ValueError("--timestamp end must be after start.")
    return start_hms, end_hms, float(start_s), float(end_s)

# -----------------------------
# Local trim helpers
# -----------------------------
def trim_media_ffmpeg(in_path: str, out_path: str, start_hms: str, end_hms: str) -> bool:
    """
    Trim media using ffmpeg.
    Try stream copy first; if it fails, fall back to re-encode.
    """
    cmd_copy = [
        "ffmpeg", "-y",
        "-ss", start_hms,
        "-to", end_hms,
        "-i", in_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path
    ]
    try:
        subprocess.check_output(cmd_copy, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        cmd_re = [
            "ffmpeg", "-y",
            "-ss", start_hms,
            "-to", end_hms,
            "-i", in_path,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            out_path
        ]
        try:
            subprocess.check_output(cmd_re, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e2:
            print("Error trimming media with ffmpeg:\n", e2.output.decode(errors="replace"))
            return False

def trim_audio_ffmpeg(in_path: str, out_path: str, start_hms: str, end_hms: str) -> bool:
    """
    Trim audio to mp3 using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", start_hms,
        "-to", end_hms,
        "-i", in_path,
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        out_path
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print("Error trimming audio with ffmpeg:\n", e.output.decode(errors="replace"))
        return False

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
    whole = math.floor(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0

    td = timedelta(seconds=int(whole))
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
            if not text:
                continue
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

# -----------------------------
# Sentence-flow chunking
# -----------------------------
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?$')
_SOFT_BREAK_RE = re.compile(r'[,;:]["\']?$')

def _word_text(word_obj) -> str:
    return word_obj.get("word", "")

def _join_whisper_words(words) -> str:
    return "".join(_word_text(w) for w in words).strip()

def _count_real_words(words) -> int:
    count = 0
    for w in words:
        token = _word_text(w).strip()
        if token:
            count += 1
    return count

def sentence_chunk_segments(
    segments,
    max_words=12,
    max_chars=70,
    max_duration=4.0,
):
    """
    Build subtitle chunks using Whisper word timestamps.
    Prefer breaking on sentence endings (. ! ?), then commas/pauses,
    then hard length limits.
    """
    out = []

    for seg in segments:
        words = seg.get("words") or []
        if not words:
            text = (seg.get("text") or "").strip()
            if text:
                out.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": text,
                })
            continue

        cur_words = []
        cur_start = None

        def flush():
            nonlocal cur_words, cur_start
            if not cur_words:
                return

            text = _join_whisper_words(cur_words)
            start = cur_start if cur_start is not None else float(cur_words[0].get("start", seg.get("start", 0.0)))
            end = float(cur_words[-1].get("end", seg.get("end", 0.0)))

            if text:
                out.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })

            cur_words = []
            cur_start = None

        for w in words:
            token = _word_text(w)
            w_start = float(w.get("start", seg.get("start", 0.0)))
            w_end = float(w.get("end", seg.get("end", 0.0)))

            if cur_start is None:
                cur_start = w_start

            cur_words.append(w)

            text_now = _join_whisper_words(cur_words)
            word_count = _count_real_words(cur_words)
            duration = max(0.0, w_end - cur_start)

            stripped = token.strip()
            is_sentence_end = bool(_SENTENCE_END_RE.search(stripped))
            is_soft_break = bool(_SOFT_BREAK_RE.search(stripped))

            too_long = (
                word_count >= max_words
                or len(text_now) >= max_chars
                or duration >= max_duration
            )

            if is_sentence_end:
                flush()
            elif is_soft_break and too_long:
                flush()
            elif too_long:
                flush()

        flush()

    return out

# -----------------------------
# Cadence chunking
# -----------------------------
_PUNCT_SPLIT_RE = re.compile(r"([.!?]+|\.\.\.|,|;|:)\s+")

def _split_text_punct(text: str):
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

        if len(words) > max_words or len(phrase) > max_chars:
            flush()
            tmp = []
            tmp_w = 0
            tmp_c = 0
            for w in words:
                wlen = len(w) + (1 if tmp else 0)
                if tmp_w + 1 > max_words or tmp_c + wlen > max_chars:
                    if tmp:
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
    dur = max(0.001, end - start)
    weights = [max(1, len(c)) for c in chunks]
    total = sum(weights)
    t = start
    out = []
    for idx, (c, w) in enumerate(zip(chunks, weights)):
        seg_dur = dur * (w / total)
        seg_start = t
        seg_end = end if idx == len(chunks) - 1 else (t + seg_dur)
        out.append({"start": seg_start, "end": seg_end, "text": c})
        t = seg_end
    return out

def cadence_chunk_segments(segments, max_words=7, max_chars=42, max_duration=2.6):
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text", "") or "").strip()
        if not text:
            continue

        phrases = _split_text_punct(text) or [text]
        chunks = _enforce_limits(phrases, max_words=max_words, max_chars=max_chars) or [text]
        allocated = _allocate_times(start, end, chunks)

        for a in allocated:
            s, e, t = a["start"], a["end"], a["text"]
            if (e - s) <= max_duration:
                out.append(a)
                continue

            words = t.split()
            if len(words) <= 1:
                out.append(a)
                continue

            n = int(math.ceil((e - s) / max_duration))
            n = max(2, min(n, len(words)))
            per = int(math.ceil(len(words) / n))
            sub_chunks = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
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
# Run srt2subtitles
# -----------------------------
def run_node_srt2subtitles(srt_path: str, fps: float):
    cmd = ["srt2subtitles", srt_path, str(round(fps))]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error running srt2subtitles:\n", e.output.decode(errors="replace"))
        return None
    return "subtitles.fcpxml"

# -----------------------------
# FCPXML line-break helper
# -----------------------------
_LB_TOKEN = "__FCPXML_LINEBREAK__"

def _wrap_text_word_boundary_to_token(s: str, max_chars: int) -> str:
    if not s or max_chars is None or max_chars <= 0:
        return s

    lines = s.splitlines()
    out_lines = []

    for line in lines:
        line = line.strip()
        if len(line) <= max_chars:
            out_lines.append(line)
            continue

        words = line.split()
        if not words:
            out_lines.append(line)
            continue

        cur = words[0]
        for w in words[1:]:
            candidate = cur + " " + w
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                out_lines.append(cur)
                cur = w
                while len(cur) > max_chars:
                    out_lines.append(cur[:max_chars])
                    cur = cur[max_chars:]
        out_lines.append(cur)

    return _LB_TOKEN.join([x for x in out_lines if x != ""])

def apply_fcpxml_line_breaks_entity(fcpx_path: str, line_break_chars: int) -> bool:
    if not os.path.exists(fcpx_path):
        print("Could not apply line breaks, file missing:", fcpx_path)
        return False

    try:
        tree = ET.parse(fcpx_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Could not parse FCPXML for line breaks: {e}")
        return False

    changed = False
    for elem in root.iter():
        if elem.text and isinstance(elem.text, str):
            raw = elem.text
            if len(raw.strip()) > line_break_chars and " " in raw:
                wrapped = _wrap_text_word_boundary_to_token(raw, line_break_chars)
                if wrapped != raw:
                    elem.text = wrapped
                    changed = True

    if not changed:
        return False

    tree.write(fcpx_path, encoding="utf-8", xml_declaration=True)

    try:
        xml = Path(fcpx_path).read_text(encoding="utf-8")
        if _LB_TOKEN in xml:
            xml = xml.replace(_LB_TOKEN, "&#10;")
            Path(fcpx_path).write_text(xml, encoding="utf-8")
        print(f"Inserted line breaks as '&#10;' for lines over {line_break_chars} chars → {fcpx_path}")
    except Exception as e:
        print(f"Failed to post-process '&#10;' replacement: {e}")
        return False

    return True

# -----------------------------
# Modify XML
# -----------------------------
def modify_fcpxml(
    fcpx_path,
    position=None,
    font=None,
    fontsize=None,
    line_spacing=None,
    line_break_chars=None,
):
    if not os.path.exists(fcpx_path):
        print("Could not modify XML, file missing:", fcpx_path)
        return

    tree = ET.parse(fcpx_path)
    root = tree.getroot()

    for elem in root.iter():
        if position and elem.tag.lower().endswith("param"):
            if elem.attrib.get("name") == "Position":
                elem.set("value", position)

        if font and "font" in elem.attrib:
            elem.set("font", font)
            if line_spacing is not None:
                elem.set("lineSpacing", str(int(line_spacing)))

        if fontsize and "fontSize" in elem.attrib:
            elem.set("fontSize", str(fontsize))

    tree.write(fcpx_path, encoding="utf-8", xml_declaration=True)

    if line_break_chars is not None:
        apply_fcpxml_line_breaks_entity(fcpx_path, int(line_break_chars))

    print(f"Updated subtitles settings → {fcpx_path}")

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
        print(f"Input dir not found or not a directory: {input_dir}")
        return

    for fp in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if fp.is_file() and fp.suffix.lower() in MEDIA_EXTS:
            yield str(fp)

# -----------------------------
# Style helpers
# -----------------------------
def resolve_style_defaults(style: str, max_words, max_chars, max_duration):
    """
    If the user did not explicitly override values, give good defaults by style.
    """
    if style == "cadence":
        if max_words is None:
            max_words = 7
        if max_chars is None:
            max_chars = 42
        if max_duration is None:
            max_duration = 2.6
    else:
        if max_words is None:
            max_words = 12
        if max_chars is None:
            max_chars = 70
        if max_duration is None:
            max_duration = 4.0
    return max_words, max_chars, max_duration

# -----------------------------
# Core pipeline for one item
# -----------------------------
def process_one(
    input_path: str,
    position=None,
    font=None,
    fontsize=None,
    line_spacing=None,
    style="sentence",
    max_words=None,
    max_chars=None,
    max_duration=None,
    line_break_chars=None,
    ytmp3=False,
    timestamp=None,
    cookies_file=None,
    cookies_from_browser=None,
    remote_ejs=False,
):
    ts = parse_timestamp_range(timestamp) if timestamp else None
    max_words, max_chars, max_duration = resolve_style_defaults(
        style, max_words, max_chars, max_duration
    )

    if is_youtube_url(input_path):
        dl = download_youtube(
            input_path,
            ytmp3=ytmp3,
            timestamp_range=ts,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
            remote_ejs=remote_ejs,
        )
        input_file = dl["path"]

        if ts and not dl.get("was_ranged"):
            start_hms, end_hms = ts[0], ts[1]
            if ytmp3:
                clipped = os.path.join(os.getcwd(), "__clip__audio.mp3")
                ok = trim_audio_ffmpeg(input_file, clipped, start_hms, end_hms)
            else:
                clipped = os.path.join(os.getcwd(), "__clip__video.mp4")
                ok = trim_media_ffmpeg(input_file, clipped, start_hms, end_hms)

            if ok:
                input_file = clipped
            else:
                print("Local trim failed, continuing with full download.")
    else:
        if not os.path.isfile(input_path):
            print(f"Error: input file not found: {input_path}")
            return
        input_file = input_path

        if ts:
            start_hms, end_hms = ts[0], ts[1]
            base = os.path.basename(input_path)
            tmp_trim = os.path.join(os.getcwd(), f"__clip__{base}")
            ok = trim_media_ffmpeg(input_path, tmp_trim, start_hms, end_hms)
            if not ok:
                print("Trim failed, skipping.")
                return
            input_file = tmp_trim

    out_folder = next_output_folder()

    media_name = os.path.basename(input_file)
    dest_media_path = os.path.join(out_folder, media_name)

    if os.path.abspath(dest_media_path) != os.path.abspath(input_file):
        shutil.move(input_file, dest_media_path)
        input_file = dest_media_path

    print(f"\nProcessing: {media_name}")
    print(f"Media moved to: {dest_media_path}")

    fps = detect_fps(input_file)
    print(f"Detected FPS: {fps}")

    with open(os.path.join(out_folder, "framerate.txt"), "w", encoding="utf-8") as f:
        f.write(str(fps))

    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    print("Transcribing...")
    result = model.transcribe(
        input_file,
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    raw_segments = result.get("segments", [])
    if not raw_segments:
        print("No segments found, skipping.")
        return

    if style == "cadence":
        segments = cadence_chunk_segments(
            raw_segments,
            max_words=max_words,
            max_chars=max_chars,
            max_duration=max_duration,
        )
        print(f"Cadence chunking enabled → {len(segments)} subtitle entries")
    else:
        segments = sentence_chunk_segments(
            raw_segments,
            max_words=max_words,
            max_chars=max_chars,
            max_duration=max_duration,
        )
        print(f"Sentence-flow chunking enabled → {len(segments)} subtitle entries")

    srt_path = os.path.join(out_folder, "subtitles.srt")
    write_srt(segments, srt_path)
    print(f"SRT saved: {srt_path}")

    run_node_srt2subtitles(srt_path, fps)

    fcpx_file = "subtitles.fcpxml"
    fcpx_src = os.path.join(os.getcwd(), fcpx_file)
    fcpx_dest = os.path.join(out_folder, fcpx_file)

    if os.path.isfile(fcpx_src):
        shutil.move(fcpx_src, fcpx_dest)
        print(f"Moved Final Cut file to: {fcpx_dest}")
    elif os.path.isfile(fcpx_dest):
        print(f"Final Cut file already in output: {fcpx_dest}")
    else:
        print("subtitles.fcpxml not found")

    modify_fcpxml(
        fcpx_dest,
        position=position,
        font=font,
        fontsize=fontsize,
        line_spacing=line_spacing,
        line_break_chars=line_break_chars,
    )

# -----------------------------
# Main program
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate SRT + FCPXML for one input or a whole directory."
    )

    parser.add_argument("input", nargs="?", help="Local media file OR YouTube URL")
    parser.add_argument("--batch", action="store_true", help='Process all media files in --input-dir')
    parser.add_argument("--input-dir", default="input", help='Directory to scan in batch mode')

    parser.add_argument("--position", type=str, help='Position "X Y", for example "0 -300"')
    parser.add_argument("--font", type=str, help='Font family, for example "Helvetica"')
    parser.add_argument("--fontsize", type=int, help="Font size in pixels")
    parser.add_argument(
        "--line-spacing",
        dest="line_spacing",
        type=int,
        default=None,
        help='Add lineSpacing="N" to FCPXML elements where font is set',
    )
    parser.add_argument(
        "--line-break-chars",
        dest="line_break_chars",
        type=int,
        default=None,
        help="Insert '&#10;' in FCPXML text nodes when text exceeds this many characters",
    )

    parser.add_argument(
        "--style",
        choices=["sentence", "cadence"],
        default="sentence",
        help="Subtitle style. sentence = smoother/natural, cadence = punchier/short-form",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Maximum words per subtitle chunk. Defaults depend on --style",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum characters per subtitle chunk. Defaults depend on --style",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum seconds per subtitle chunk. Defaults depend on --style",
    )

    parser.add_argument(
        "--ytmp3",
        action="store_true",
        help="For YouTube URLs, download audio-only as MP3 instead of video",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help='Restrict processing to this time window. Format: "HH:MM:SS-HH:MM:SS"',
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to a Netscape cookies.txt file for yt-dlp",
    )
    parser.add_argument(
        "--cookies-from-browser",
        dest="cookies_from_browser",
        type=str,
        default=None,
        help='Read cookies from browser. Example: "chrome" or "chrome:Profile 1"',
    )
    parser.add_argument(
        "--remote-ejs",
        action="store_true",
        help="Enable yt-dlp remote EJS challenge solver distribution",
    )

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
                line_spacing=args.line_spacing,
                style=args.style,
                max_words=args.max_words,
                max_chars=args.max_chars,
                max_duration=args.max_duration,
                line_break_chars=args.line_break_chars,
                ytmp3=False,
                timestamp=args.timestamp,
                cookies_file=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                remote_ejs=args.remote_ejs,
            )

        if not any_found:
            print(f"No media files found in: {args.input_dir}")
        return

    if not args.input:
        print('Error: provide a file/URL, or use --batch')
        sys.exit(2)

    process_one(
        args.input,
        position=args.position,
        font=args.font,
        fontsize=args.fontsize,
        line_spacing=args.line_spacing,
        style=args.style,
        max_words=args.max_words,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        line_break_chars=args.line_break_chars,
        ytmp3=args.ytmp3,
        timestamp=args.timestamp,
        cookies_file=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        remote_ejs=args.remote_ejs,
    )

if __name__ == "__main__":
    main()