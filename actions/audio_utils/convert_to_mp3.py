#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess

def convert_to_mp3(input_file, quality='medium'):
    print(f"Starting MP3 conversion for file: {input_file} with quality: {quality}")
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)

    # Determine bitrate based on quality
    if quality == 'low':
        bitrate = '128k'
    elif quality == 'high':
        bitrate = '320k'
    else:  # medium
        bitrate = '192k'

    print(f"Using bitrate: {bitrate}")

    # Generate output file name
    base, _ = os.path.splitext(input_file)
    output_file = f"{base}.mp3"
    print(f"Output file: {output_file}")

    # FFmpeg command
    command = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_file,
        "-b:a", bitrate,
        "-vn",  # No video
        output_file
    ]

    print(f"Running command: {' '.join(command)}")
    try:
        # Run ffmpeg, capturing stderr to stdout for progress
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print(f"Audio converted to MP3: {output_file} (quality: {quality}, bitrate: {bitrate})")
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        print("Ensure 'ffmpeg' is installed.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found.")
        print("Please install ffmpeg (e.g., 'sudo apt install ffmpeg' on Ubuntu).")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio file to MP3")
    parser.add_argument("input_file", nargs='+', help="Path to the audio file")
    parser.add_argument("--quality", choices=['low', 'medium', 'high'], default='medium', help="MP3 quality (default: medium)")
    args = parser.parse_args()

    input_file = ' '.join(args.input_file)
    convert_to_mp3(input_file, args.quality)