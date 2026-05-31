# Reusable file-transformation runtime for the `utils` project.
#
# Scope: media + document conversions (image/video thumbnails, PDF/DOCX/ODT/PPT,
# audio remux). The heavy AI stack (torch, faster-whisper, kokoro, mineru) is
# intentionally EXCLUDED to keep the image small; add a separate extended image
# if those actions are needed.
#
# The project CODE is NOT copied in — bind-mount the repo at runtime so it can
# evolve without rebuilding:
#   docker run --rm -v /root/code/utils:/opt/utils:ro -v <data>:/data \
#       utils:latest python /opt/utils/utils_run.py <args>
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        pandoc \
        libreoffice \
        libimage-exiftool-perl \
        poppler-utils \
        ocrmypdf \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra \
        build-essential cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "Pillow>=10.0.0" \
        rawpy==0.25.1 \
        imageio==2.31.2 \
        PyPDF2 \
        python-pptx \
        reportlab \
        PyMuPDF \
        pypandoc \
        pydub \
        soundfile \
        numpy \
        python-dotenv

WORKDIR /opt/utils
CMD ["python", "-c", "print('utils runtime image — bind-mount the repo at /opt/utils and invoke a script, e.g. python /opt/utils/actions/picture_utils/thumbnailing.py --help')"]
