#!/bin/bash

# Script to extract audio from a video file
# Usage: ./extract_audio.sh <video_file>

if [ $# -eq 0 ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

VIDEO_FILE="$*"

if [ ! -f "$VIDEO_FILE" ]; then
  echo "File not found: $VIDEO_FILE"
  exit 1
fi

# Get directory and base name without extension
DIR=$(dirname "$VIDEO_FILE")
BASE=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')

OUTPUT_FILE="$DIR/${BASE}_audio.wav"

# Extract audio using ffmpeg to lossless WAV
ffmpeg -i "$VIDEO_FILE" -vn -acodec pcm_s16le "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
  echo "Audio extracted to: $OUTPUT_FILE"
else
  echo "Error extracting audio"
  exit 1
fi