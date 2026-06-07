import os
import subprocess

# Chaîne ffmpeg de réhaussement pour audio DÉGRADÉ (voix captée de loin, salle
# bruyante). Contrairement à loudnorm (qui ne change pas le SNR), elle agit sur
# la clarté : highpass=80 (coupe rumble/HVAC) + afftdn (débruitage FFT) +
# dynaudnorm (remonte dynamiquement les passages faibles, mieux que loudnorm
# pour une voix qui varie en distance). 100 % filtres ffmpeg intégrés (aucune
# dépendance). Mutualisée entre transcribe_audio.py et transcribe_channels.py.
SPEECH_ENHANCE_FILTERS = "highpass=f=80,afftdn=nr=10:nf=-25,dynaudnorm=f=200:g=15"


def improve_audio_quality(input_file, output_file=None):
    """
    Improve audio quality using ffmpeg loudnorm filter.
    If output_file is None, generates {base}_quality_improved{ext}
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist")

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_quality_improved{ext}"

    print(f"Improving audio quality for: {input_file}")
    # Use ffmpeg to apply loudnorm (normalization)
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        output_file
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"Improved audio saved to: {output_file}")
    return output_file