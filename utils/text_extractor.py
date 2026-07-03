"""
Text Extractor using OCR
"""

import pytesseract
from PIL import Image


def extract_text(image_path):
    """
    Extract text from an image.
    """

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()

    except Exception as e:
        return f"Error: {e}"