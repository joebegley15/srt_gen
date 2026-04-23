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
  • sentence  — breaks at sentence endings and natural pauses using
                Whisper word timestamps. No external API required.
  • cadence   — punchier, shorter chunks for social/short-form usage.

Quality presets (--quality):
  • auto      — high for videos under 10 min, medium above (default)
  • high      — Whisper large, beam_size=5
  • medium    — Whisper medium, beam_size=3
  • low       — Whisper small, greedy decode

YouTube options:
  • --ytmp3                          Download audio-only as MP3
  • --timestamp "HH:MM:SS-HH:MM:SS"  Process only this segment
  • --cookies <path>                 Netscape cookies.txt for yt-dlp
  • --cookies-from-browser "chrome[:PROFILE]"
  • --remote-ejs                     Enable yt-dlp remote EJS solver
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import whisper

try:
    import tomllib          # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib   # pip install tomli
    except ImportError:
        tomllib = None

from youtube_download import is_youtube_url, download_youtube


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILENAME = "config.toml"

def load_config(path: str = CONFIG_FILENAME) -> dict:
    if tomllib is None:
        return {}
    for p in (Path(path), Path(__file__).parent / path):
        if p.exists():
            with open(p, "rb") as f:
                cfg = tomllib.load(f)
            print(f"Loaded config: {p}")
            return cfg
    return {}

def _cfg(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SubtitleConfig:
    style: str            = "sentence"
    quality: str          = "auto"     # auto | high | medium | low
    min_words: int        = None       # sentence mode: min words before a soft break fires
    max_words: int        = None
    max_chars: int        = None
    max_duration: float   = None
    font: str             = None
    fontsize: int         = None
    position: str         = None
    line_spacing: int     = None
    line_break_chars: int = None

    def resolve_defaults(self):
        """Fill in style-specific defaults for any unset limit fields."""
        if self.style == "cadence":
            self.max_words    = self.max_words    or 7
            self.max_chars    = self.max_chars    or 42
            self.max_duration = self.max_duration or 2.6
        else:
            self.max_words    = self.max_words    or 18
            self.max_chars    = self.max_chars    or 120
            self.max_duration = self.max_duration or 8.0
            self.min_words    = self.min_words    or 4


@dataclass
class YoutubeConfig:
    ytmp3: bool                         = False
    timestamp: Optional[str]            = None
    cookies_file: Optional[str]         = None
    cookies_from_browser: Optional[str] = None
    remote_ejs: bool                    = False


@dataclass
class WhisperPreset:
    model: str
    beam_size: int
    best_of: int
    condition_on_previous_text: bool

# high   — large model, full beam search. Best accuracy, slowest.
# medium — medium model, reduced beam. Good accuracy, ~2-3x faster.
# low    — small model, greedy decode. Fastest, noticeably lower accuracy.
QUALITY_PRESETS: dict[str, WhisperPreset] = {
    "high":   WhisperPreset(model="large",  beam_size=5, best_of=5, condition_on_previous_text=True),
    "medium": WhisperPreset(model="medium", beam_size=3, best_of=3, condition_on_previous_text=False),
    "low":    WhisperPreset(model="small",  beam_size=1, best_of=1, condition_on_previous_text=False),
}

AUTO_QUALITY_THRESHOLD_SECS = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
_HMS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_TS_RE  = re.compile(r"^\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}$")

def _hms_to_seconds(hms: str) -> int:
    if not _HMS_RE.match(hms):
        raise ValueError(f"Invalid timecode: {hms}")
    hh, mm, ss = hms.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def parse_timestamp_range(ts: str):
    """Parse 'HH:MM:SS-HH:MM:SS' → (start_hms, end_hms, start_s, end_s) or None."""
    if not ts:
        return None
    ts = ts.strip()
    if not _TS_RE.match(ts):
        raise ValueError('Invalid --timestamp format. Expected "HH:MM:SS-HH:MM:SS".')
    start_hms, end_hms = ts.split("-", 1)
    start_s = _hms_to_seconds(start_hms)
    end_s   = _hms_to_seconds(end_hms)
    if end_s <= start_s:
        raise ValueError("--timestamp end must be after start.")
    return start_hms, end_hms, float(start_s), float(end_s)


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.stderr.write("Error: ffmpeg or ffprobe not found. Install ffmpeg.\n")
        sys.exit(1)

def _run_ffmpeg(cmd: list, loud: bool = False) -> bool:
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        if loud:
            print("Error running ffmpeg:\n", e.output.decode(errors="replace"))
        return False

def trim_with_ffmpeg(
    in_path: str, out_path: str,
    start_hms: str, end_hms: str,
    audio_only: bool = False,
) -> bool:
    """
    Trim media with ffmpeg.
    audio_only=True  → re-encode to mp3 directly.
    audio_only=False → try stream copy first, fall back to h264/aac re-encode.
    """
    base = ["ffmpeg", "-y", "-ss", start_hms, "-to", end_hms, "-i", in_path]
    if audio_only:
        return _run_ffmpeg(base + ["-vn", "-c:a", "libmp3lame", "-b:a", "192k", out_path], loud=True)
    if _run_ffmpeg(base + ["-c", "copy", "-avoid_negative_ts", "make_zero", out_path]):
        return True
    return _run_ffmpeg(base + [
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path,
    ], loud=True)

def detect_fps(video_path: str) -> float:
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
            video_path,
        ]).decode().strip()
        num, _, den = raw.partition("/")
        return round(float(num) / float(den) if den else float(num), 3)
    except Exception as e:
        print(f"Warning: Could not detect FPS, defaulting to 24. Error: {e}")
        return 24.0

def get_duration(media_path: str) -> float:
    """Return media duration in seconds, or 0.0 on failure."""
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "0", "-of", "csv=p=0",
            "-show_entries", "format=duration",
            media_path,
        ]).decode().strip()
        return float(raw)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------
def next_output_folder(base: str = "output") -> str:
    os.makedirs(base, exist_ok=True)
    nums   = [int(d) for d in os.listdir(base) if re.match(r"^\d{3}$", d)]
    folder = os.path.join(base, f"{(max(nums) + 1) if nums else 1:03d}")
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# SRT formatting
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole   = int(seconds)
    ms      = round((seconds - whole) * 1000)
    if ms == 1000:
        whole += 1
        ms = 0
    hh, rem = divmod(whole, 3600)
    mm, ss  = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def write_srt(all_words: list, out_path: str):
    """Write one SRT entry per word using Whisper word-level timestamps."""
    with open(out_path, "w", encoding="utf-8") as f:
        i = 1
        for w in all_words:
            text = _word_text(w).strip()
            if not text:
                continue
            start = float(w.get("start", 0.0))
            end   = float(w.get("end",   0.0))
            if end <= start:
                end = start + 0.1
            f.write(
                f"{i}\n"
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{text}\n\n"
            )
            i += 1


def write_sentences_srt(segments: list, out_path: str):
    """Write a sentence-level SRT for srt2subtitles to consume."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            text = seg.get("text", "").strip()
            if not text:
                continue
            f.write(
                f"{i}\n"
                f"{format_timestamp(seg.get('start', 0.0))} --> {format_timestamp(seg.get('end', 0.0))}\n"
                f"{text}\n\n"
            )


# ---------------------------------------------------------------------------
# Shared word helpers
# ---------------------------------------------------------------------------
def _word_text(w: dict) -> str:
    return w.get("word", "")

def _join_words(words: list) -> str:
    return "".join(_word_text(w) for w in words).strip()

def _count_real_words(words: list) -> int:
    return sum(1 for w in words if _word_text(w).strip())

def _flatten_words(segments: list) -> list:
    """Extract all word-level dicts from Whisper segments into a flat list."""
    all_words = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            all_words.extend(words)
        else:
            text = (seg.get("text") or "").strip()
            if text:
                all_words.append({
                    "word":  " " + text,
                    "start": float(seg.get("start", 0.0)),
                    "end":   float(seg.get("end",   0.0)),
                })
    return all_words



# ---------------------------------------------------------------------------
# Punctuation restoration (local, no API)
# ---------------------------------------------------------------------------
def _restore_punctuation(all_words: list) -> list:
    """
    Run deepmultilingualpunctuation on the flat word list to restore
    commas, periods, question marks etc. Returns a new word list with
    the same timestamps but punctuated text.

    Falls back to the original word list if the model isn't installed.
    """
    try:
        from deepmultilingualpunctuation import PunctuationModel
    except ImportError:
        print("Note: install deepmultilingualpunctuation for better chunking: pip install deepmultilingualpunctuation")
        return all_words

    raw_text = " ".join(_word_text(w).strip() for w in all_words if _word_text(w).strip())
    if not raw_text:
        return all_words

    print("Restoring punctuation...")
    try:
        model  = PunctuationModel()
        result = model.restore_punctuation(raw_text)
    except TypeError:
        # Older deepmultilingualpunctuation uses grouped_entities which was removed
        # in newer transformers. Patch it in place.
        from transformers import pipeline as hf_pipeline
        import deepmultilingualpunctuation.punctuationmodel as _pm
        _pm.pipeline = lambda task, model, **kw: hf_pipeline(task, model, aggregation_strategy="none")
        model  = PunctuationModel()
        result = model.restore_punctuation(raw_text)

    # model returns a string — split back into tokens and re-attach timestamps
    punct_tokens = result.split()
    orig_tokens  = [_word_text(w).strip() for w in all_words if _word_text(w).strip()]

    # zip by position — lengths should match since punctuation model only
    # adds punctuation characters, doesn't insert or delete words
    if len(punct_tokens) != len(orig_tokens):
        print(f"Warning: punctuation token count mismatch ({len(punct_tokens)} vs {len(orig_tokens)}), skipping.")
        return all_words

    # rebuild word list with punctuated tokens, preserving original timestamps
    real_idx  = 0
    new_words = []
    for w in all_words:
        token = _word_text(w).strip()
        if not token:
            new_words.append(w)
            continue
        new_w = dict(w)
        # preserve leading space Whisper puts on words
        leading = " " if _word_text(w).startswith(" ") else ""
        new_w["word"] = leading + punct_tokens[real_idx]
        new_words.append(new_w)
        real_idx += 1

    return new_words


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?$')
_SOFT_BREAK_RE   = re.compile(r'[,;:]["\']?$')

def sentence_chunk_segments(
    segments_or_words,
    max_words=9,
    max_chars=60,
    max_duration=4.0,
    min_words=6,
):
    """
    Chunk a transcript into subtitle lines by breaking at punctuation boundaries.
    Accepts either raw Whisper segments or a pre-flattened word list (e.g. after
    punctuation restoration). Sentence endings (.!?) always trigger a break;
    soft pauses (,;:) only break after min_words have accumulated.
    """
    # Accept either a flat word list or raw Whisper segments
    if segments_or_words and isinstance(segments_or_words[0], dict) and "word" in segments_or_words[0]:
        all_words = segments_or_words  # already flattened
    else:
        all_words = _flatten_words(segments_or_words)

    out: list       = []
    cur_words: list = []
    cur_start: Optional[float] = None

    def flush():
        nonlocal cur_words, cur_start
        if not cur_words:
            return
        text  = _join_words(cur_words)
        start = cur_start if cur_start is not None else float(cur_words[0].get("start", 0.0))
        end   = float(cur_words[-1].get("end", 0.0))
        if text:
            out.append({"start": start, "end": end, "text": text})
        cur_words = []
        cur_start = None

    for w in all_words:
        token   = _word_text(w)
        w_start = float(w.get("start", 0.0))
        w_end   = float(w.get("end",   0.0))

        if cur_start is None:
            cur_start = w_start
        cur_words.append(w)

        text_now   = _join_words(cur_words)
        word_count = _count_real_words(cur_words)
        duration   = max(0.0, w_end - cur_start)
        stripped   = token.strip()

        over_limit = (
            word_count >= max_words
            or len(text_now) >= max_chars
            or duration >= max_duration
        )

        if over_limit:
            # Hard limit — always flush
            flush()
        elif _SENTENCE_END_RE.search(stripped):
            # Sentence ending (.!?) — always flush, even on short phrases
            flush()
        elif _SOFT_BREAK_RE.search(stripped) and word_count >= min_words:
            # Soft pause (,;:) — only flush once enough words have built up
            flush()

    flush()
    return out


# ---------------------------------------------------------------------------
# Cadence chunking
# ---------------------------------------------------------------------------
_PUNCT_SPLIT_RE = re.compile(r"([.!?]+|\.\.\.|,|;|:)\s+")

def _split_text_punct(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    parts, start = [], 0
    for m in _PUNCT_SPLIT_RE.finditer(text):
        chunk = text[start:m.end()].strip()
        if chunk:
            parts.append(chunk)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts

def _enforce_limits(phrases: list, max_words: int, max_chars: int) -> list:
    out, buf, buf_words, buf_chars = [], [], 0, 0

    def flush():
        nonlocal buf, buf_words, buf_chars
        if buf:
            out.append(" ".join(buf).strip())
        buf, buf_words, buf_chars = [], 0, 0

    for phrase in phrases:
        words = phrase.split()
        if len(words) > max_words or len(phrase) > max_chars:
            flush()
            tmp, tmp_w, tmp_c = [], 0, 0
            for w in words:
                wlen = len(w) + (1 if tmp else 0)
                if tmp_w + 1 > max_words or tmp_c + wlen > max_chars:
                    if tmp:
                        out.append(" ".join(tmp).strip())
                    tmp, tmp_w, tmp_c = [w], 1, len(w)
                else:
                    tmp.append(w)
                    tmp_w += 1
                    tmp_c += wlen
            if tmp:
                out.append(" ".join(tmp).strip())
            continue

        phrase_words = len(words)
        phrase_chars = len(phrase) + (1 if buf else 0)
        if buf_words + phrase_words > max_words or buf_chars + phrase_chars > max_chars:
            flush()
            buf, buf_words, buf_chars = [phrase], phrase_words, len(phrase)
        else:
            buf.append(phrase)
            buf_words += phrase_words
            buf_chars += phrase_chars

    flush()
    return [x for x in out if x]

def _allocate_times(start: float, end: float, chunks: list) -> list:
    dur     = max(0.001, end - start)
    weights = [max(1, len(c)) for c in chunks]
    total   = sum(weights)
    t, out  = start, []
    for idx, (c, w) in enumerate(zip(chunks, weights)):
        seg_end = end if idx == len(chunks) - 1 else t + dur * (w / total)
        out.append({"start": t, "end": seg_end, "text": c})
        t = seg_end
    return out

def cadence_chunk_segments(segments, max_words=7, max_chars=42, max_duration=2.6):
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end",   0.0))
        text  = (seg.get("text", "") or "").strip()
        if not text:
            continue

        phrases   = _split_text_punct(text) or [text]
        chunks    = _enforce_limits(phrases, max_words, max_chars) or [text]
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
            n          = max(2, min(int(math.ceil((e - s) / max_duration)), len(words)))
            per        = int(math.ceil(len(words) / n))
            sub_chunks = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            out.extend(_allocate_times(s, e, sub_chunks))

    return out


# ---------------------------------------------------------------------------
# FCPXML helpers
# ---------------------------------------------------------------------------
_LB_TOKEN = "__FCPXML_LINEBREAK__"

def _wrap_to_token(s: str, max_chars: int) -> str:
    if not s or max_chars is None or max_chars <= 0:
        return s
    out_lines = []
    for line in s.splitlines():
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
            candidate = f"{cur} {w}"
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                out_lines.append(cur)
                cur = w
                while len(cur) > max_chars:
                    out_lines.append(cur[:max_chars])
                    cur = cur[max_chars:]
        out_lines.append(cur)
    return _LB_TOKEN.join(x for x in out_lines if x)

def apply_fcpxml_line_breaks(fcpx_path: str, line_break_chars: int) -> bool:
    if not os.path.exists(fcpx_path):
        print("Could not apply line breaks, file missing:", fcpx_path)
        return False
    try:
        tree = ET.parse(fcpx_path)
    except Exception as e:
        print(f"Could not parse FCPXML for line breaks: {e}")
        return False

    changed = False
    for elem in tree.getroot().iter():
        raw = elem.text
        if isinstance(raw, str) and len(raw.strip()) > line_break_chars and " " in raw:
            wrapped = _wrap_to_token(raw, line_break_chars)
            if wrapped != raw:
                elem.text = wrapped
                changed = True

    if not changed:
        return False

    tree.write(fcpx_path, encoding="utf-8", xml_declaration=True)
    try:
        xml_text = Path(fcpx_path).read_text(encoding="utf-8")
        if _LB_TOKEN in xml_text:
            Path(fcpx_path).write_text(xml_text.replace(_LB_TOKEN, "&#10;"), encoding="utf-8")
        print(f"Inserted line breaks as '&#10;' for lines over {line_break_chars} chars → {fcpx_path}")
    except Exception as e:
        print(f"Failed to post-process '&#10;' replacement: {e}")
        return False
    return True

def modify_fcpxml(fcpx_path: str, sub: SubtitleConfig):
    if not os.path.exists(fcpx_path):
        print("Could not modify XML, file missing:", fcpx_path)
        return
    tree = ET.parse(fcpx_path)
    root = tree.getroot()

    for elem in root.iter():
        if sub.position and elem.tag.lower().endswith("param") and elem.attrib.get("name") == "Position":
            elem.set("value", sub.position)
        if sub.font and "font" in elem.attrib:
            elem.set("font", sub.font)
            if sub.line_spacing is not None:
                elem.set("lineSpacing", str(int(sub.line_spacing)))
        if sub.fontsize and "fontSize" in elem.attrib:
            elem.set("fontSize", str(sub.fontsize))

    tree.write(fcpx_path, encoding="utf-8", xml_declaration=True)
    if sub.line_break_chars is not None:
        apply_fcpxml_line_breaks(fcpx_path, int(sub.line_break_chars))
    print(f"Updated subtitle settings → {fcpx_path}")


# ---------------------------------------------------------------------------
# srt2subtitles runner
# ---------------------------------------------------------------------------
def run_srt2subtitles(srt_path: str, fps: float):
    cmd = ["srt2subtitles", srt_path, str(round(fps))]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error running srt2subtitles:\n", e.output.decode(errors="replace"))


# ---------------------------------------------------------------------------
# Batch file discovery
# ---------------------------------------------------------------------------
MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi",
              ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

def iter_media_files(input_dir: str):
    p = Path(input_dir)
    if not p.is_dir():
        print(f"Input dir not found or not a directory: {input_dir}")
        return
    for fp in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if fp.is_file() and fp.suffix.lower() in MEDIA_EXTS:
            yield str(fp)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def process_one(input_path: str, sub: SubtitleConfig, yt: YoutubeConfig):
    sub.resolve_defaults()
    ts = parse_timestamp_range(yt.timestamp) if yt.timestamp else None

    # --- Acquire input file ---
    if is_youtube_url(input_path):
        dl = download_youtube(
            input_path,
            ytmp3=yt.ytmp3,
            timestamp_range=ts,
            cookies_file=yt.cookies_file,
            cookies_from_browser=yt.cookies_from_browser,
            remote_ejs=yt.remote_ejs,
        )
        input_file = dl["path"]
        if ts and not dl.get("was_ranged"):
            clip_ext  = ".mp3" if yt.ytmp3 else ".mp4"
            clip_name = "audio" if yt.ytmp3 else "video"
            clip_path = os.path.join(os.getcwd(), f"__clip__{clip_name}{clip_ext}")
            if trim_with_ffmpeg(input_file, clip_path, ts[0], ts[1], audio_only=yt.ytmp3):
                input_file = clip_path
            else:
                print("Local trim failed, continuing with full download.")
    else:
        if not os.path.isfile(input_path):
            print(f"Error: input file not found: {input_path}")
            return
        input_file = input_path
        if ts:
            tmp = os.path.join(os.getcwd(), f"__clip__{os.path.basename(input_path)}")
            if trim_with_ffmpeg(input_path, tmp, ts[0], ts[1]):
                input_file = tmp
            else:
                print("Trim failed, skipping.")
                return

    # --- Move media to output folder ---
    out_folder      = next_output_folder()
    media_name      = os.path.basename(input_file)
    dest_media_path = os.path.join(out_folder, media_name)

    if os.path.abspath(dest_media_path) != os.path.abspath(input_file):
        shutil.move(input_file, dest_media_path)
        input_file = dest_media_path

    print(f"\nProcessing: {media_name}")
    print(f"Media moved to: {dest_media_path}")

    fps = detect_fps(input_file)
    print(f"Detected FPS: {fps}")
    Path(out_folder, "framerate.txt").write_text(str(fps), encoding="utf-8")

    # --- Select Whisper preset ---
    if sub.quality == "auto":
        duration = get_duration(input_file)
        resolved_quality = "high" if duration < AUTO_QUALITY_THRESHOLD_SECS else "medium"
        print(f"Auto quality: {duration:.0f}s → using '{resolved_quality}' preset")
    else:
        resolved_quality = sub.quality

    preset = QUALITY_PRESETS[resolved_quality]
    print(f"Quality preset '{resolved_quality}': model={preset.model}, beam_size={preset.beam_size}")

    # --- Transcribe ---
    # MPS doesn't support float64 which Whisper's word-timestamp alignment requires.
    # Always use CPU for transcription to ensure word timestamps work correctly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = whisper.load_model(preset.model, device=device)
    print(f"Transcribing with Whisper {preset.model}...")
    result = model.transcribe(
        input_file,
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=preset.condition_on_previous_text,
        temperature=0,
        beam_size=preset.beam_size,
        best_of=preset.best_of,
    )

    raw_segments = result.get("segments", [])
    if not raw_segments:
        print("No segments found, skipping.")
        return

    # --- Chunk ---
    all_words = _flatten_words(raw_segments)
    all_words = _restore_punctuation(all_words)

    if sub.style == "cadence":
        segments = cadence_chunk_segments(
            raw_segments,
            max_words=sub.max_words,
            max_chars=sub.max_chars,
            max_duration=sub.max_duration,
        )
        print(f"Cadence chunking → {len(segments)} subtitle entries")
    else:
        segments  = sentence_chunk_segments(
            all_words,
            max_words=sub.max_words,
            max_chars=sub.max_chars,
            max_duration=sub.max_duration,
            min_words=sub.min_words,
        )
        print(f"Sentence chunking → {len(segments)} subtitle entries")

    # --- Write SRT + FCPXML ---
    # Word-level SRT — one entry per word with precise timestamps
    words_srt_path = os.path.join(out_folder, "subtitles_words.srt")
    write_srt(all_words, words_srt_path)
    print(f"Word-level SRT saved: {words_srt_path}")

    # Sentence-level SRT — named subtitles.srt so srt2subtitles produces subtitles.fcpxml
    srt_path = os.path.join(out_folder, "subtitles.srt")
    write_sentences_srt(segments, srt_path)
    print(f"Sentence SRT saved: {srt_path}")

    run_srt2subtitles(srt_path, fps)

    fcpx_src  = os.path.join(os.getcwd(), "subtitles.fcpxml")
    fcpx_dest = os.path.join(out_folder,  "subtitles.fcpxml")
    if os.path.isfile(fcpx_src):
        shutil.move(fcpx_src, fcpx_dest)
        print(f"Moved Final Cut file to: {fcpx_dest}")
    elif not os.path.isfile(fcpx_dest):
        print("subtitles.fcpxml not found")

    modify_fcpxml(fcpx_dest, sub)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Generate SRT + FCPXML for one input or a whole directory."
    )

    # Positional / batch
    parser.add_argument("input", nargs="?", help="Local media file OR YouTube URL")
    parser.add_argument("--batch",     action="store_true", help="Process all media in --input-dir")
    parser.add_argument("--input-dir", default=_cfg(cfg, "batch", "input_dir", default="input"),
                        help="Directory to scan in batch mode")

    # Subtitle style
    parser.add_argument("--style",        choices=["sentence", "cadence"],
                        default=_cfg(cfg, "style", "default", default="sentence"))
    parser.add_argument("--quality",      choices=["auto", "high", "medium", "low"],
                        default=_cfg(cfg, "style", "quality", default="auto"),
                        help="Whisper quality preset. auto = high under 10 min, medium above (default: auto)")
    parser.add_argument("--min-words",    type=int, default=_cfg(cfg, "style", "min_words"),
                        dest="min_words",
                        help="Fallback chunker: min words before a soft break fires (default: 6)")
    parser.add_argument("--max-words",    type=int,   default=_cfg(cfg, "style", "max_words"))
    parser.add_argument("--max-chars",    type=int,   default=_cfg(cfg, "style", "max_chars"))
    parser.add_argument("--max-duration", type=float, default=_cfg(cfg, "style", "max_duration"))

    # FCPXML appearance
    parser.add_argument("--position",         type=str, default=_cfg(cfg, "fcpxml", "position"))
    parser.add_argument("--font",             type=str, default=_cfg(cfg, "fcpxml", "font"))
    parser.add_argument("--fontsize",         type=int, default=_cfg(cfg, "fcpxml", "fontsize"))
    parser.add_argument("--line-spacing",     type=int, default=_cfg(cfg, "fcpxml", "line_spacing"),
                        dest="line_spacing")
    parser.add_argument("--line-break-chars", type=int, default=_cfg(cfg, "fcpxml", "line_break_chars"),
                        dest="line_break_chars")

    # YouTube
    parser.add_argument("--ytmp3",                action="store_true",
                        default=_cfg(cfg, "youtube", "ytmp3", default=False))
    parser.add_argument("--timestamp",            type=str, default=None)
    parser.add_argument("--cookies",              type=str, default=_cfg(cfg, "youtube", "cookies"))
    parser.add_argument("--cookies-from-browser", type=str, dest="cookies_from_browser",
                        default=_cfg(cfg, "youtube", "cookies_from_browser"))
    parser.add_argument("--remote-ejs",           action="store_true")

    args = parser.parse_args()
    check_ffmpeg()

    sub = SubtitleConfig(
        style=args.style,
        quality=args.quality,
        min_words=args.min_words,
        max_words=args.max_words,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        font=args.font,
        fontsize=args.fontsize,
        position=args.position,
        line_spacing=args.line_spacing,
        line_break_chars=args.line_break_chars,
    )
    yt = YoutubeConfig(
        ytmp3=args.ytmp3,
        timestamp=args.timestamp,
        cookies_file=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        remote_ejs=args.remote_ejs,
    )

    if args.batch:
        found = False
        for fp in iter_media_files(args.input_dir):
            found = True
            process_one(fp, sub, yt)
        if not found:
            print(f"No media files found in: {args.input_dir}")
        return

    if not args.input:
        print("Error: provide a file/URL, or use --batch")
        sys.exit(2)

    process_one(args.input, sub, yt)


if __name__ == "__main__":
    main()