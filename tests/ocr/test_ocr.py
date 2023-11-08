"""
This file runs pytests against the ocr function

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

from typing import Tuple
import cv2
import pytest
from licence_system.utils.ocr import ocr_image


# Define test cases
@pytest.mark.parametrize(
    "image_path, coordinates, expected_text",
    # (path_to_img, (x, y, width, height), expected_text)
    [
        # ("./tests/ocr/2.jpg", (165, 295, 57, 35), "BE33TA"),
        # ("./tests/ocr/3.jpg", (195, 235, 56, 28), "CNK06N"),
        # ("./tests/ocr/4.jpg", (340, 280, 57, 50), "BF14QG"),
        # ("./tests/ocr/5.jpg", (97, 305, 45, 28), "719SDR"),
        ("./tests/ocr/1.jpg", (30, 10, 252, 97), "SEB384"),
        ("./tests/ocr/1.jpg", (30, 100, 252, 97), "600063"),
        ("./tests/ocr/1.jpg", (35, 190, 252, 97), "NOA770"),
        ("./tests/ocr/1.jpg", (35, 190, 252, 97), "NOA710"),
        ("./tests/ocr/7.png", (1, 1, 293, 197), "KLA674"),
        ("./tests/ocr/8.png", (1, 1, 293, 197), "AA56QH"),
        ("./tests/ocr/9.png", (1, 1, 293, 197), "CI59VP"),
        ("./tests/ocr/9.png", (1, 1, 293, 197), "C159VP"),
        ("./tests/ocr/11.png", (1, 1, 293, 197), "OZY450"),
        ("./tests/ocr/10.png", (1, 1, 639, 476), "OZY450"),
    ],
)
def test_ocr_image(image_path: str, coordinates: Tuple, expected_text: str):
    """Test OCR against image with coordinates

    Args:
        image_path (str): Image path
        coordinates (tuple): coordinate tuple (x, y, width, height)
        expected_text (str): expected result from OCR
    """
    # Read the image from the file
    image = cv2.imread(image_path)

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # Call ocr function
    extracted_text = ocr_image(image, coordinates)
    print(extracted_text)

    # Assert
    assert extracted_text.strip() == expected_text.strip()
