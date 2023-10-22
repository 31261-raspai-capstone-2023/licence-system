"""Pytest for the OCR function"""

import cv2
import pytest
from licence_system.utils.ocr import ocr_with_coordinates


# Define test cases
@pytest.mark.parametrize(
    "image_path, coordinates, expected_text",
    # (path_to_img, (x, y, width, height), expected_text)
    [
        ("./tests/ocr/1.jpg", (30, 10, 252, 97), "SEB384"),
        ("./tests/ocr/1.jpg", (30, 100, 252, 97), "600063"),
        ("./tests/ocr/1.jpg", (35, 190, 252, 97), "NOA770"),
        # ("./tests/ocr/1.jpg", (30, 190, 252, 97), "NOA770"),
        ("./tests/ocr/2.jpg", (165, 295, 57, 35), "BE33TA"),
    ],
)
def test_ocr_with_coordinates(image_path, coordinates, expected_text):
    # Read the image from the file
    image = cv2.imread(image_path)

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # Call ocr function
    extracted_text = ocr_with_coordinates(image, coordinates)
    print(extracted_text)

    # Assert
    assert extracted_text.strip() == expected_text.strip()
