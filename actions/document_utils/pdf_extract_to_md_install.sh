#!/bin/bash

# Install Python dependencies
pip install --break-system-packages PyMuPDF python-dotenv

# Install OCRmyPDF
sudo apt update
sudo apt install -y ocrmypdf

echo "Installation complete. PDF Extract to MD is ready."