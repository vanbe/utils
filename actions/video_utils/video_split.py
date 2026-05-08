#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def split_video(video_path, start_time, end_time=None):
    # Check if ffmpeg is installed
    if not subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode == 0:
        print("Error: ffmpeg is not installed.")
        sys.exit(1)

    # Generate output filename
    base_name = os.path.splitext(video_path)[0]
    ext = os.path.splitext(video_path)[1]
    output_video = f"{base_name}_split{ext}"

    # Build ffmpeg command
    cmd = ['ffmpeg', '-i', video_path, '-ss', start_time]
    if end_time:
        cmd.extend(['-to', end_time])
    cmd.extend(['-c', 'copy', output_video])

    # Run ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: ffmpeg failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    return output_video

def main():
    parser = argparse.ArgumentParser(description="Split video using ffmpeg")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("--start", required=True, help="Start time in HH:MM:SS")
    parser.add_argument("--end", help="End time in HH:MM:SS (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.video_file):
        print(f"Error: {args.video_file} does not exist")
        sys.exit(1)

    print("Starting video split...")
    output_file = split_video(args.video_file, args.start, args.end)

    print("Video split complete!")
    print(f"Split video: {output_file}")

if __name__ == "__main__":
    main()