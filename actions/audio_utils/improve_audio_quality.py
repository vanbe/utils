#!/usr/bin/env python3

import sys
from audio_utils_common import improve_audio_quality as improve_func

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improve audio quality")
    parser.add_argument("input_file", nargs='+', help="Path to the audio file")
    args = parser.parse_args()

    input_file = ' '.join(args.input_file)
    try:
        improve_func(input_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)