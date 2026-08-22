#!/usr/bin/env python3
"""
batch_mp3_srt.py

Batch-convert multiple long video files into:
  • an MP3  — <name>.mp3   (audio-only, small enough for easy uploads)
  • an SRT  — <name>.srt   (full transcript, same chunking engine as generate_srt.py)

Optionally search every transcript for a key phrase and get back a timestamped
list of every match, across every file, without re-watching anything.

Usage:
  python batch_mp3_srt.py video1.mp4 video2.mov video3.mkv
  python batch_mp3_srt.py --input-dir input          # process a whole folder
  python batch_mp3_srt.py *.mp4 --keyphrase "sign up now"

Each video gets its own folder:
  output/<video-name>/<video-name>.mp3
  output/<video-name>/<video-name>.srt
  output/<video-name>/keyphrase_matches.txt   (only if --keyphrase matched something)

A combined report across all files is also written to:
  output/keyphrase_matches.txt                (only if --keyphrase given)

Options:
  --output-dir DIR         Where per-video folders are created (default: output)
  --quality {auto,high,medium,low}   Whisper quality preset (default: auto)
  --style {sentence,cadence}         Subtitle chunking style (default: sentence)
  --bitrate RATE            MP3 bitrate, e.g. 192k (default: 192k)
  --keyphrase TEXT           Search every transcript for this phrase
  --case-sensitive           Make --keyphrase matching case-sensitive (default: off)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import whisper

from generate_srt import (
    check_ffmpeg,
    get_duration,
    format_timestamp,
    write_srt,
    sentence_chunk_segments,
    subdivide_segments,
    cadence_chunk_segments,
    _flatten_words,
    _restore_punctuation,
    QUALITY_PRESETS,
    AUTO_QUALITY_THRESHOLD_SECS,
    SubtitleConfig,
    iter_media_files,
    load_config,
    _cfg,
)


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_mp3(input_path: str, out_path: str, bitrate: str = "192k") -> bool:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, out_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print("Error extracting MP3:\n", e.output.decode(errors="replace"))
        return False


# ---------------------------------------------------------------------------
# Key phrase search
# ---------------------------------------------------------------------------
def find_keyphrase_matches(chunks: list, phrase: str, case_sensitive: bool = False) -> list:
    """Search final SRT chunks for a phrase. Returns matching {start, end, text} dicts."""
    needle = phrase if case_sensitive else phrase.lower()
    matches = []
    for c in chunks:
        haystack = c["text"] if case_sensitive else c["text"].lower()
        if needle in haystack:
            matches.append(c)
    return matches


def format_matches(video_name: str, phrase: str, matches: list) -> str:
    if not matches:
        return f"{video_name} — no matches\n"
    lines = [f"{video_name} — {len(matches)} match(es)"]
    for m in matches:
        ts = f"{format_timestamp(m['start'])} --> {format_timestamp(m['end'])}"
        lines.append(f"  [{ts}] {m['text']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Model cache — avoid reloading the same Whisper model for every file
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

def get_model(name: str, device: str):
    if name not in _MODEL_CACHE:
        print(f"Loading Whisper model '{name}'...")
        _MODEL_CACHE[name] = whisper.load_model(name, device=device)
    return _MODEL_CACHE[name]


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------
def process_video(input_path: str, output_dir: str, sub: SubtitleConfig,
                   bitrate: str, keyphrase: str | None, case_sensitive: bool) -> dict:
    if not os.path.isfile(input_path):
        print(f"Error: file not found, skipping: {input_path}")
        return {}

    stem       = Path(input_path).stem
    out_folder = os.path.join(output_dir, stem)
    os.makedirs(out_folder, exist_ok=True)

    print(f"\n=== {os.path.basename(input_path)} ===")

    # --- Extract MP3 ---
    mp3_path = os.path.join(out_folder, f"{stem}.mp3")
    print(f"Extracting MP3 → {mp3_path}")
    if not extract_mp3(input_path, mp3_path, bitrate=bitrate):
        print(f"Skipping {input_path}: MP3 extraction failed.")
        return {}

    # --- Pick quality preset ---
    if sub.quality == "auto":
        duration = get_duration(mp3_path)
        resolved_quality = "high" if duration < AUTO_QUALITY_THRESHOLD_SECS else "medium"
        print(f"Auto quality: {duration:.0f}s → using '{resolved_quality}' preset")
    else:
        resolved_quality = sub.quality
    preset = QUALITY_PRESETS[resolved_quality]

    # --- Transcribe (from the MP3 we just made — smaller/faster, same content) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = get_model(preset.model, device)
    print(f"Transcribing with Whisper {preset.model}...")
    result = model.transcribe(
        mp3_path,
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=preset.condition_on_previous_text,
        temperature=0,
        beam_size=preset.beam_size,
        best_of=preset.best_of,
    )

    raw_segments = result.get("segments", [])
    if not raw_segments:
        print("No segments found, skipping SRT for this file.")
        return {"mp3_path": mp3_path}

    all_words = _flatten_words(raw_segments)
    all_words = _restore_punctuation(all_words)

    if sub.style == "cadence":
        chunks = cadence_chunk_segments(
            raw_segments,
            max_words=sub.max_words,
            max_chars=sub.max_chars,
            max_duration=sub.max_duration,
        )
    else:
        sentences = sentence_chunk_segments(
            all_words,
            max_words=sub.max_words,
            max_chars=sub.max_chars,
            max_duration=sub.max_duration,
            min_words=sub.min_words,
        )
        chunks = subdivide_segments(sentences, max_chars=sub.max_chars)

    srt_path = os.path.join(out_folder, f"{stem}.srt")
    write_srt(chunks, srt_path)
    print(f"SRT saved: {srt_path}")

    matches = []
    if keyphrase:
        matches = find_keyphrase_matches(chunks, keyphrase, case_sensitive)
        report  = format_matches(os.path.basename(input_path), keyphrase, matches)
        print(report)
        if matches:
            matches_path = os.path.join(out_folder, "keyphrase_matches.txt")
            Path(matches_path).write_text(report, encoding="utf-8")
            print(f"Key phrase matches saved: {matches_path}")

    return {
        "name": os.path.basename(input_path),
        "mp3_path": mp3_path,
        "srt_path": srt_path,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Batch-convert long video files into MP3 + SRT, with optional key-phrase search."
    )
    parser.add_argument("inputs", nargs="*", help="Video files to process")
    parser.add_argument("--input-dir", default=None,
                        help="Process every media file in this directory instead of / in addition to positional args")
    parser.add_argument("--output-dir", default="output",
                        help="Where per-video output folders are created (default: output)")

    parser.add_argument("--quality", choices=["auto", "high", "medium", "low"],
                        default=_cfg(cfg, "style", "quality", default="auto"))
    parser.add_argument("--style", choices=["sentence", "cadence"],
                        default=_cfg(cfg, "style", "default", default="sentence"))
    parser.add_argument("--min-words",    type=int,   default=_cfg(cfg, "style", "min_words"))
    parser.add_argument("--max-words",    type=int,   default=_cfg(cfg, "style", "max_words"))
    parser.add_argument("--max-chars",    type=int,   default=_cfg(cfg, "style", "max_chars"))
    parser.add_argument("--max-duration", type=float, default=_cfg(cfg, "style", "max_duration"))

    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k)")

    parser.add_argument("--keyphrase", type=str, default=None,
                        help="Search every transcript for this phrase and report timestamped matches")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Make --keyphrase matching case-sensitive (default: off)")

    args = parser.parse_args()
    check_ffmpeg()

    inputs = list(args.inputs)
    if args.input_dir:
        inputs.extend(iter_media_files(args.input_dir))
    if not inputs:
        print("Error: provide one or more video files, or use --input-dir")
        sys.exit(2)

    sub = SubtitleConfig(
        style=args.style,
        quality=args.quality,
        min_words=args.min_words,
        max_words=args.max_words,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
    )
    sub.resolve_defaults()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for input_path in inputs:
        res = process_video(
            input_path, args.output_dir, sub,
            bitrate=args.bitrate,
            keyphrase=args.keyphrase,
            case_sensitive=args.case_sensitive,
        )
        if res:
            results.append(res)

    if args.keyphrase:
        combined = [f'Key phrase: "{args.keyphrase}" ({"case-sensitive" if args.case_sensitive else "case-insensitive"})\n']
        total = 0
        for res in results:
            matches = res.get("matches", [])
            total += len(matches)
            combined.append(format_matches(res["name"], args.keyphrase, matches))
        summary_path = os.path.join(args.output_dir, "keyphrase_matches.txt")
        Path(summary_path).write_text("\n".join(combined), encoding="utf-8")
        print(f"\nTotal matches across {len(results)} file(s): {total}")
        print(f"Combined report saved: {summary_path}")

    print(f"\nDone. Processed {len(results)}/{len(inputs)} file(s).")


if __name__ == "__main__":
    main()
