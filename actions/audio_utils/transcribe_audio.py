#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
import json
import warnings
from dotenv import load_dotenv

# Set PyTorch memory allocation config to reduce fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Project root is two levels up from this script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'vendor', 'models')

# Load environment variables from root .env
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, '.env'))

# Point HuggingFace Hub cache to the shared models directory (covers pyannote + any HF downloads)
os.environ.setdefault('HF_HOME', os.path.join(_MODELS_DIR, 'huggingface'))

import torch

# Note: TF32 is intentionally left at default (disabled by pyannote for better diarization accuracy).
# Enabling it here has no effect as pyannote.audio resets it before running.

# Suppress pyannote internal warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=r"TensorFloat-32 \(TF32\) has been disabled")

# Limit CPU threads to leave cores free (configured via OMP_NUM_THREADS in .env)
os.environ.setdefault('OMP_NUM_THREADS', '8')

from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Try importing the helper, fallback if not found
try:
    from audio_utils_common import improve_audio_quality
except ImportError:
    def improve_audio_quality(input_file, output_file):
        import shutil
        shutil.copy(input_file, output_file)
        return output_file

# Map CLI model names to faster-whisper model IDs.
# large-v3 is the highest quality model; same weights as openai-whisper large-v3.
# turbo = large-v3-turbo: slightly faster, minimal quality difference.
MODEL_NAME_MAP = {
    'tiny':   'tiny',
    'base':   'base',
    'small':  'small',
    'medium': 'medium',
    'large':  'large-v3',
    'turbo':  'large-v3-turbo',
}

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def preprocess_audio(input_file):
    """
    Converts input to 16kHz Mono WAV AND applies audio improvements.
    """
    print("Preprocessing audio (Improving quality & converting to 16kHz Mono WAV)...")

    base, _ = os.path.splitext(input_file)
    temp_file = f"{base}_quality_improved.wav"

    # First, improve the audio
    improved_temp = improve_audio_quality(input_file, f"{base}_temp_improved.wav")

    # Then, convert to 16kHz mono
    filter_chain = "aresample=16000,pan=mono|c0=c0"
    command = [
        "ffmpeg",
        "-y",
        "-i", improved_temp,
        "-af", filter_chain,
        "-c:a", "pcm_s16le",
        temp_file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.exists(improved_temp):
            os.remove(improved_temp)
        return temp_file
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found.")
        sys.exit(1)

def convert_to_wav(input_file):
    """
    Convert audio to 16kHz Mono WAV without improvement.
    """
    print("Converting to 16kHz Mono WAV...")
    base, _ = os.path.splitext(input_file)
    output_file = f"{base}_converted.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_file
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_file

def get_speaker_for_segment(w_start, w_end, diarization_segments, speaker_map):
    """
    Determines the speaker by calculating which speaker has the most
    temporal overlap with the Whisper segment.
    """
    if not diarization_segments:
        return None

    max_overlap = 0
    best_speaker = "Unknown"

    for d_seg in diarization_segments:
        start = max(w_start, d_seg["start"])
        end = min(w_end, d_seg["end"])
        overlap = end - start

        if overlap > 0 and overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker_map.get(d_seg["speaker"], "Unknown")

    # If no overlap found (e.g. slight timing mismatch), fall back to midpoint check
    if max_overlap == 0:
        w_mid = w_start + (w_end - w_start) / 2
        for d_seg in diarization_segments:
            if d_seg["start"] <= w_mid <= d_seg["end"]:
                return speaker_map.get(d_seg["speaker"], "Unknown")

    return best_speaker

def transcribe_audio(input_file, language='en', no_speaker=False, output_format='md',
                     model_size='large', no_improve=False, max_speakers=0):
    import contextlib, io, time

    _SEP = '─' * 56

    def _elapsed(t0: float) -> str:
        e = time.time() - t0
        return f"{int(e // 60)}m {int(e % 60):02d}s" if e >= 60 else f"{e:.1f}s"

    def _step(n: int, label: str) -> float:
        """Inline step — result printed on same line by _ok()."""
        print(f"  [{n}]  {label:<30}", end='', flush=True)
        return time.time()

    def _ok(t0: float, detail: str = '') -> None:
        d = f"   {detail}" if detail else ''
        print(f"  ✓  {_elapsed(t0)}{d}")

    def _step_block(n: int, label: str) -> float:
        """Block step — produces its own output (progress bars etc)."""
        print(f"  [{n}]  {label}")
        return time.time()

    def _ok_block(t0: float, detail: str = '') -> None:
        d = f"   {detail}" if detail else ''
        print(f"       ✓  {_elapsed(t0)}{d}")

    t_total = time.time()
    step_n  = 0

    # ── hardware ────────────────────────────────────────────────────────────
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    n_threads    = int(os.environ.get('OMP_NUM_THREADS', '4'))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))

    print(_SEP)
    print(f"  {os.path.basename(input_file)}")
    print(f"  device: {device}  ·  compute: {compute_type}  ·  threads: {n_threads}")
    print(_SEP)

    # ── validation ──────────────────────────────────────────────────────────
    if not os.path.exists(input_file):
        print(f"\n  Error: file not found: {input_file}")
        sys.exit(1)

    hf_token = os.getenv('AUDIO_UTILS_HF_TOKEN')
    if not hf_token:
        print("\n  Error: AUDIO_UTILS_HF_TOKEN not set in .env.")
        sys.exit(1)
    os.environ["HF_TOKEN"] = hf_token

    base, _ = os.path.splitext(input_file)
    if "_quality_improved" in base or "_converted" in base:
        base = base.replace("_quality_improved", "").replace("_converted", "")
    processing_file   = f"{base}_quality_improved.wav" if not no_improve else f"{base}_converted.wav"
    whisper_cache     = f"{base}_whisper_{model_size}_segments.json"
    diarization_cache = f"{base}_diarization_segments.json"
    output_file       = f"{base}_transcription.{output_format}"

    if os.path.exists(output_file):
        print(f"\n  Already exists: {os.path.basename(output_file)}")
        return

    # ── [1] Pre-process audio ───────────────────────────────────────────────
    step_n += 1
    t = _step(step_n, "Pre-process audio")
    if os.path.exists(processing_file):
        _ok(t, "cached")
    elif "_quality_improved" in input_file:
        processing_file = input_file
        _ok(t, "input already processed")
    elif no_improve:
        with contextlib.redirect_stdout(io.StringIO()):
            processing_file = convert_to_wav(input_file)
        _ok(t, "converted to 16kHz mono")
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            processing_file = preprocess_audio(input_file)
        _ok(t, "quality improved + 16kHz mono")

    # ── [2] Load Whisper ────────────────────────────────────────────────────
    step_n += 1
    t = _step(step_n, f"Load Whisper  ({model_size})")
    if device == "cuda":
        torch.cuda.empty_cache()

    models_to_try = (['large', 'turbo', 'medium', 'small', 'base', 'tiny'] if language == 'en'
                     else ['large', 'medium', 'small', 'base', 'tiny'])
    if model_size in models_to_try:
        models_to_try.remove(model_size)
        models_to_try.insert(0, model_size)

    model, used_model = None, None
    for try_model in models_to_try:
        mapped_name = MODEL_NAME_MAP.get(try_model, try_model)
        try:
            model = WhisperModel(mapped_name, device=device, compute_type=compute_type,
                                 download_root=os.path.join(_MODELS_DIR, 'whisper'))
            used_model = try_model
            break
        except torch.cuda.OutOfMemoryError:
            print(f"\n     OOM for {try_model}, trying smaller…", end='', flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"\n     OOM for {try_model}, trying smaller…", end='', flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"\n  Error: {e}")
                sys.exit(1)

    if model is None:
        print("\n  Error: no Whisper model could be loaded.")
        sys.exit(1)

    detail = f"{used_model}" if used_model == model_size else f"{used_model} (fallback)"
    _ok(t, f"{detail} · {MODEL_NAME_MAP.get(used_model, used_model)} · {compute_type}")

    # ── [3] Transcribe ──────────────────────────────────────────────────────
    step_n += 1
    t = _step(step_n, "Transcribe")
    if os.path.exists(whisper_cache):
        with open(whisper_cache) as f:
            whisper_segments = json.load(f)
        _ok(t, f"{len(whisper_segments)} segments  (cached)")
    else:
        segments_iter, _ = model.transcribe(
            processing_file,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True,
        )
        whisper_segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments_iter
        ]
        with open(whisper_cache, 'w') as f:
            json.dump(whisper_segments, f)
        _ok(t, f"{len(whisper_segments)} segments")

    # Free Whisper VRAM before loading pyannote
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── [4 & 5] Speaker diarization (optional) ──────────────────────────────
    if no_speaker:
        diarization_segments = []
    else:
        # [4] Load diarization model
        step_n += 1
        t = _step(step_n, "Load diarization model")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=hf_token)
            pipeline.to(torch.device(device))
            seg_batch = 64 if device == "cuda" else 8
            emb_batch = 32 if device == "cuda" else 8  # 64 OOMs on 6GB with pyannote 4.x fbank
            for attr, val in [("segmentation_batch_size", seg_batch),
                               ("embedding_batch_size",    emb_batch)]:
                if hasattr(pipeline, attr):
                    setattr(pipeline, attr, val)
            _ok(t, f"seg_batch={seg_batch}  emb_batch={emb_batch}")
        except Exception as e:
            print(f"\n  Error: {e}")
            sys.exit(1)

        # [5] Diarize
        step_n += 1
        if os.path.exists(diarization_cache):
            t = _step(step_n, "Diarize")
            with open(diarization_cache) as f:
                diarization_segments = json.load(f)
            unique_speakers = len(set(s["speaker"] for s in diarization_segments))
            _ok(t, f"{unique_speakers} speakers  (cached)")
        else:
            t = _step_block(step_n, "Diarize")

            # Pre-load audio tensor — avoids pyannote re-reading the file each sweep
            # torchaudio.load is broken in 2.11 (requires torchcodec); use soundfile.
            audio_input = processing_file
            try:
                import soundfile as _sf
                _data, _sr = _sf.read(processing_file, dtype="float32", always_2d=True)
                audio_input = {"waveform": torch.from_numpy(_data.T), "sample_rate": _sr}
            except Exception:
                pass  # fall back to file path on any error

            diarize_kwargs = {}
            if max_speakers > 0:
                diarize_kwargs["max_speakers"] = max_speakers

            try:
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                with ProgressHook() as hook:
                    diarization_result = pipeline(audio_input, hook=hook, **diarize_kwargs)
            except (ImportError, TypeError):
                diarization_result = pipeline(audio_input, **diarize_kwargs)

            if hasattr(diarization_result, "speaker_diarization"):
                diarization = diarization_result.speaker_diarization
            else:
                diarization = diarization_result

            diarization_segments = sorted(
                [{"start": seg.start, "end": seg.end, "speaker": spk}
                 for seg, _, spk in diarization.itertracks(yield_label=True)],
                key=lambda x: x["start"]
            )
            with open(diarization_cache, 'w') as f:
                json.dump(diarization_segments, f)

            unique_speakers = len(set(s["speaker"] for s in diarization_segments))
            _ok_block(t, f"{unique_speakers} speakers")

    # ── [N] Save ─────────────────────────────────────────────────────────────
    step_n += 1
    t = _step(step_n, "Save transcription")

    speaker_map = {}
    if diarization_segments:
        speaker_map = {spk: f"Speaker {i+1}"
                       for i, spk in enumerate(sorted(set(s["speaker"]
                                                          for s in diarization_segments)))}

    if output_format == 'srt':
        lines = []
        for i, w_seg in enumerate(whisper_segments, 1):
            w_start, w_end = w_seg['start'], w_seg['end']
            text = w_seg['text'].strip()
            if diarization_segments:
                spk  = get_speaker_for_segment(w_start, w_end, diarization_segments, speaker_map)
                text = f"{spk}: {text}"
            lines.append(f"{i}\n{format_srt_time(w_start)} --> {format_srt_time(w_end)}\n{text}\n")
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
    else:
        grouped, current = [], {"speaker": None, "text": []}
        if whisper_segments:
            first = whisper_segments[0]
            current["speaker"] = (
                get_speaker_for_segment(first['start'], first['end'],
                                        diarization_segments, speaker_map)
                if diarization_segments else "Text"
            )
        for w_seg in whisper_segments:
            w_start, w_end = w_seg['start'], w_seg['end']
            spk = (get_speaker_for_segment(w_start, w_end, diarization_segments, speaker_map)
                   if diarization_segments else "Text")
            if spk == current["speaker"] or (spk == "Unknown"
                                              and current["speaker"] != "Unknown"):
                current["text"].append(w_seg['text'].strip())
            else:
                if current["text"]:
                    grouped.append(current)
                current = {"speaker": spk, "text": [w_seg['text'].strip()]}
        if current["text"]:
            grouped.append(current)

        lines = [
            f"**{g['speaker']}**: {' '.join(g['text'])}" if g["speaker"]
            else ' '.join(g['text'])
            for g in grouped
        ]
        with open(output_file, 'w') as f:
            f.write("# Transcription\n\n" + '\n\n'.join(lines))

    # Remove cache files quietly
    for cache_file in [whisper_cache, diarization_cache]:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    _ok(t)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print(f"  Total:  {_elapsed(t_total)}")
    print(f"  Saved:  {os.path.basename(output_file)}")
    print(_SEP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("input_file", nargs='+', help="Path to the audio file")
    parser.add_argument("--language", default="en", help="Language code (e.g., en, fr)")
    parser.add_argument("--no-speaker", action="store_true", help="Skip speaker recognition")
    parser.add_argument("--output-format", choices=['md', 'srt'], default='md', help="Output format: md or srt")
    parser.add_argument("--model", choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'], default='large',
                        help="Whisper model size (default: large = large-v3, ~2.7 GB VRAM; turbo = large-v3-turbo, faster)")
    parser.add_argument("--no-improve", action="store_true", help="Skip audio quality improvement for faster processing")
    parser.add_argument("--max-speakers", type=int, default=0, metavar='N',
                        help="Maximum number of speakers expected (0 = auto). "
                             "Setting this speeds up clustering significantly — "
                             "use 2 for an interview, 4-6 for a meeting.")
    args = parser.parse_args()

    input_file = ' '.join(args.input_file)
    transcribe_audio(input_file, args.language, args.no_speaker, args.output_format, args.model, args.no_improve, args.max_speakers)
