#!/bin/bash
# Create virtual environment
python3 -m venv traffic-analysis-env
source  traffic-analysis-env/bin/activate
# Installing dependencies
pip3 install -r requirements.txt

# Install tesseract for OCR
sudo add-apt-repository ppa:alex-p/tesseract-ocr
sudo apt-get update
sudo apt install tesseract-ocr libtesseract-dev
