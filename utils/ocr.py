"""Extract plate from image using OCR"""
from typing import Tuple

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def ocr_with_coordinates(image: np.ndarray, coordinates: Tuple) -> str:
    """Get the string from coordinates in an image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array
        coordinates (Tuple): coordinate tuple (x, y, width, height)

    Returns:
        str: ocr result
    """
    x, y, width, height = coordinates

    # Crop the image
    cropped_image = image[y : y + height, x : x + width]

    

    # Extract text
    extracted_text = pytesseract.image_to_string(
        cropped_image,
        lang="eng",
        config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    
    return extracted_text
