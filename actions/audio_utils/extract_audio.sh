#!/bin/bash

# Extract audio from video file using ffmpeg
# Usage: ./extract_audio.sh <video_file>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

input_file="$*"

if [ ! -f "$input_file" ]; then
    echo "Error: File $input_file does not exist"
    exit 1
fi

# Generate output file name (replace extension with .wav)
output_file="${input_file%.*}.wav"

echo "Extracting audio from: $input_file"
echo "Output: $output_file"

# Use ffmpeg to extract audio as lossless WAV
/usr/bin/ffmpeg -y -i "$input_file" -vn -acodec pcm_s16le "$output_file"

if [ $? -eq 0 ]; then
    echo "Audio extracted successfully: $output_file"
else
    echo "Error extracting audio"
    exit 1
fi