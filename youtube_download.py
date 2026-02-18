"""
youtube_download.py

All YouTube / yt-dlp related logic for generate_srt.py lives here.

Responsibilities:
- Identify and normalize YouTube URLs (strip surrounding quotes, unescape common shell escapes)
- Configure yt-dlp (cookies, cookies-from-browser, remote EJS, JS runtimes)
- Download full media or attempt ranged download (download_ranges) when a timestamp window is provided

Important design choice:
- If ranged download fails, we return the full download and let the caller trim locally.
  This keeps this module independent from the caller's ffmpeg trim helpers.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def is_youtube_url(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    return ("youtube.com" in s) or ("youtu.be" in s)


def normalize_youtube_url(url: str) -> str:
    """
    Normalize a YouTube URL that may be pasted with surrounding quotes or common shell escapes.

    Examples that become valid:
      "https://www.youtube.com/watch\?v\=abc"  -> https://www.youtube.com/watch?v=abc
      'https://youtu.be/abc\?t\=12'            -> https://youtu.be/abc?t=12
    """
    if not url:
        return url
    url = str(url).strip()

    # Strip surrounding single or double quotes
    if (url.startswith('"') and url.endswith('"')) or (url.startswith("'") and url.endswith("'")):
        url = url[1:-1].strip()

    # Unescape the common sequences people paste in shells
    url = (
        url.replace("\\?", "?")
           .replace("\\&", "&")
           .replace("\\=", "=")
    )
    return url


def _auto_js_runtimes_dict():
    """
    yt-dlp may need a JS runtime to extract some formats.
    Newer yt-dlp expects js_runtimes as: {runtime: {config}}
    """
    runtimes = []
    for rt in ("node", "deno", "bun", "quickjs"):
        if shutil.which(rt):
            runtimes.append(rt)
    return {rt: {} for rt in runtimes} if runtimes else None


def _newest_in_dir(download_dir: str, exts):
    p = Path(download_dir)
    files = []
    for ext in exts:
        files.extend(p.glob(f"*{ext}"))
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def _parse_cookies_from_browser_arg(s: str):
    """
    Parse --cookies-from-browser into the tuple expected by yt-dlp's Python API.

    CLI syntax (yt-dlp):
      BROWSER[+KEYRING][:PROFILE][::CONTAINER]

    Returns:
      (browser_name, profile, keyring, container)
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None

    container = None
    if "::" in s:
        s, container = s.split("::", 1)
        container = container.strip() or None

    profile = None
    if ":" in s:
        s, profile = s.split(":", 1)
        profile = profile.strip() or None

    keyring = None
    browser = s.strip()
    if "+" in browser:
        browser, keyring = browser.split("+", 1)
        keyring = keyring.strip() or None
        if keyring:
            keyring = keyring.upper()

    browser = browser.strip().lower()
    if not browser:
        return None

    return (browser, profile, keyring, container)


def download_youtube(
    url: str,
    download_dir: str = "temp_dl",
    ytmp3: bool = False,
    timestamp_range=None,  # (start_hms, end_hms, start_s, end_s) or None
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    remote_ejs: bool = False,
) -> Dict[str, Any]:
    """
    Download a YouTube URL using yt-dlp.

    Returns:
      {"path": <downloaded file path>, "was_ranged": <bool>}

    Behavior:
      - If timestamp_range is provided, we try ranged download first via download_ranges.
      - If ranged download fails, we fall back to full download and return was_ranged=False.
        The caller can then trim locally to guarantee timestamps work.
    """
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import download_range_func
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    os.makedirs(download_dir, exist_ok=True)

    url = normalize_youtube_url(url)

    def _ydl_opts(for_range: bool):
        opts = {
            "quiet": True,
            "noplaylist": True,
            "retries": 10,
            "fragment_retries": 10,
            "outtmpl": os.path.join(download_dir, ("audio.%(ext)s" if ytmp3 else "video.%(ext)s")),
            "hls_prefer_native": True,
            "concurrent_fragment_downloads": 1,
        }

        js_runtimes = _auto_js_runtimes_dict()
        if js_runtimes:
            opts["js_runtimes"] = js_runtimes

        if remote_ejs:
            # matches CLI: --remote-components ejs:github
            opts["remote_components"] = ["ejs:github"]

        if cookies_file:
            opts["cookiefile"] = cookies_file

        if cookies_from_browser:
            cfb = _parse_cookies_from_browser_arg(cookies_from_browser)
            if cfb:
                opts["cookiesfrombrowser"] = cfb

        # Prefer web-ish clients. 'web_safari' is often more stable than plain 'web'.
        opts["extractor_args"] = {"youtube": {"player_client": ["web_safari", "tv", "web"]}}

        if for_range and timestamp_range:
            _, _, start_s, end_s = timestamp_range
            opts["download_ranges"] = download_range_func(None, [(start_s, end_s)])

        if ytmp3:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        else:
            opts["format"] = "bestvideo*+bestaudio/best"

        return opts

    print(f"Downloading YouTube {'audio (mp3)' if ytmp3 else 'video'}: {url}")

    # 1) Try ranged download first (if requested)
    if timestamp_range:
        try:
            with YoutubeDL(_ydl_opts(for_range=True)) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)

            if ytmp3:
                mp3 = _newest_in_dir(download_dir, [".mp3"])
                if mp3:
                    return {"path": mp3, "was_ranged": True}
            return {"path": filename, "was_ranged": True}

        except Exception as e:
            print(f"⚠ Ranged YouTube download failed ({e}). Falling back to full download.")

    # 2) Fallback full download
    with YoutubeDL(_ydl_opts(for_range=False)) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    if ytmp3:
        mp3 = _newest_in_dir(download_dir, [".mp3"])
        if not mp3:
            mp3 = _newest_in_dir(download_dir, [".m4a", ".webm", ".aac", ".opus"])
        if mp3:
            return {"path": mp3, "was_ranged": False}
        return {"path": filename, "was_ranged": False}

    src = filename
    if not os.path.isfile(src):
        src = _newest_in_dir(download_dir, [".mp4", ".mkv", ".webm"]) or filename

    return {"path": src, "was_ranged": False}
