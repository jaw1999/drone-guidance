#!/usr/bin/env python3
"""Download sample maritime videos for testing.

This script downloads free stock footage of boats/ships from Pexels
for testing the terminal guidance system.

Usage:
    python scripts/download_sample_video.py

The videos will be saved to the 'videos/' directory.
"""

import os
import sys
import urllib.request
import urllib.error

# Free maritime video URLs from Pexels (CC0 license)
# These are direct download links for stock footage
SAMPLE_VIDEOS = [
    {
        "name": "sailing_boat.mp4",
        "url": "https://www.pexels.com/download/video/857251/",
        "description": "Sailing boat on ocean - good for testing",
    },
    {
        "name": "yacht_aerial.mp4",
        "url": "https://www.pexels.com/download/video/2491284/",
        "description": "Aerial view of yacht - drone perspective",
    },
    {
        "name": "boats_harbor.mp4",
        "url": "https://www.pexels.com/download/video/3571264/",
        "description": "Multiple boats in harbor",
    },
]


def download_video(url: str, output_path: str) -> bool:
    """Download a video file with progress indication."""
    try:
        print(f"Downloading: {output_path}")
        print(f"  From: {url}")

        # Create request with browser-like headers
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
                print(f"  Size: {total_size / 1024 / 1024:.1f} MB")

            # Download in chunks
            chunk_size = 8192
            downloaded = 0

            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

            print(f"\n  Saved to: {output_path}")
            return True

    except urllib.error.HTTPError as e:
        print(f"  Error: HTTP {e.code} - {e.reason}")
        print("  Note: Pexels may require browser access. Try downloading manually.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    # Create videos directory
    videos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print("=" * 60)
    print("  Sample Maritime Video Downloader")
    print("=" * 60)
    print()
    print("This will download free stock footage for testing.")
    print(f"Videos will be saved to: {videos_dir}")
    print()

    # Check if yt-dlp is available for YouTube downloads
    try:
        import subprocess
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        has_ytdlp = result.returncode == 0
    except:
        has_ytdlp = False

    if has_ytdlp:
        print("yt-dlp detected! You can download YouTube videos directly:")
        print()
        print("  # Download a specific video:")
        print('  yt-dlp -o "videos/boats.mp4" "https://youtube.com/watch?v=VIDEO_ID"')
        print()
        print("  # Search and download maritime footage:")
        print('  yt-dlp -o "videos/%(title)s.%(ext)s" "ytsearch:drone boat ocean footage"')
        print()
    else:
        print("Tip: Install yt-dlp for easy YouTube downloads:")
        print("  brew install yt-dlp   # macOS")
        print("  pip install yt-dlp    # Any platform")
        print()

    print("Alternative: Download manually from these free sources:")
    print("  - https://www.pexels.com/search/videos/boat%20ocean/")
    print("  - https://pixabay.com/videos/search/boat/")
    print("  - https://www.videvo.net/free-video-footage/boat/")
    print()

    # Try downloading from Pexels
    print("Attempting to download sample videos from Pexels...")
    print("(Note: Direct downloads may be blocked - manual download recommended)")
    print()

    success_count = 0
    for video in SAMPLE_VIDEOS:
        output_path = os.path.join(videos_dir, video["name"])

        if os.path.exists(output_path):
            print(f"Already exists: {video['name']}")
            success_count += 1
            continue

        if download_video(video["url"], output_path):
            success_count += 1
        print()

    print("=" * 60)
    if success_count > 0:
        print(f"Downloaded {success_count} video(s)")
        print()
        print("Usage:")
        print(f'  python scripts/fake_camera.py --video videos/sailing_boat.mp4')
    else:
        print("No videos downloaded (Pexels may require browser access)")
        print()
        print("Please download manually from Pexels/Pixabay and save to videos/")
        print()
        print("Then run:")
        print('  python scripts/fake_camera.py --video videos/your_video.mp4')
    print("=" * 60)


if __name__ == "__main__":
    main()
