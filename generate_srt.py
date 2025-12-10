#!/usr/bin/env python3
"""
generate_srt.py — Create SRT subtitles from:
  • a local video/audio file
  • a YouTube URL

Then:
  • Move media into output/XXX/
  • Detect frame rate → output/XXX/framerate.txt
  • Run: srt2subtitles subtitles.srt <fps>
  • Save converted subtitles into same folder
"""

import argparse
import os
import sys
import math
import shutil
import re
import subprocess
from datetime import timedelta
from pathlib import Path

# -----------------------------
# Utility: check ffmpeg exists
# -----------------------------
def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.stderr.write(
            "Error: ffmpeg or ffprobe not found. Install ffmpeg.\n"
        )
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
def run_node_srt2subtitles(srt_path: str, fps: float, out_folder: str):
    cmd = ["srt2subtitles", srt_path, str(int(fps))]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print("Error running srt2subtitles:", e.output.decode())
        return None

    # The tool always generates subtitles.fcpxml
    return "subtitles.fcpxml"

# -----------------------------
# Main program
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SRT, detect FPS, run node conversion.")
    parser.add_argument("input", help="Local video/audio file OR YouTube URL")
    args = parser.parse_args()

    check_ffmpeg()

    input_path = args.input

    # 1. Download or verify file
    if is_youtube_url(input_path):
        input_file = download_youtube(input_path)
    else:
        if not os.path.isfile(input_path):
            sys.stderr.write(f"Error: input file not found: {input_path}\n")
            sys.exit(1)
        input_file = input_path

    # 2. Create output folder
    out_folder = next_output_folder()

    # 3. Move video into folder
    media_name = os.path.basename(input_file)
    dest_media_path = os.path.join(out_folder, media_name)

    if os.path.abspath(dest_media_path) != os.path.abspath(input_file):
        shutil.move(input_file, dest_media_path)
        input_file = dest_media_path

    print(f"📁 Media moved to {dest_media_path}")

    # 4. Detect frame rate
    fps = detect_fps(input_file)
    print(f"🎞 Detected FPS: {fps}")

    # Write framerate.txt
    with open(os.path.join(out_folder, "framerate.txt"), "w") as f:
        f.write(str(fps))

    # 5. TRANSCRIBE using Whisper
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    print("🔊 Transcribing...")
    result = model.transcribe(input_file, verbose=False)
    segments = result.get("segments", [])
    if not segments:
        print("No segments found!")
        sys.exit(1)

    # Save SRT
    srt_path = os.path.join(out_folder, "subtitles.srt")
    write_srt(segments, srt_path)
    print(f"📝 SRT saved: {srt_path}")

    # 6. Run node conversion
    converted = run_node_srt2subtitles(srt_path, fps, out_folder)

    # Explicit expected file
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

if __name__ == "__main__":
    main()
