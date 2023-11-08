"""
This file defines the functions for extracting the plate text from image using OCR

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
from typing import Tuple

import cv2
import numpy as np
import pytesseract
import os
import re

import cv2
import numpy as np


def deskew_image(image):
    # Use thresholding to get a binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate the moments to find the centroid
    moments = cv2.moments(binary)

    # Calculate the skew based on the moments
    if moments["mu02"] != 0:
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, -0.5 * image.shape[0] * skew], [0, 1, 0]])
        # Warp the image to correct the skew
        img = cv2.warpAffine(
            binary,
            M,
            (image.shape[1], image.shape[0]),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        )
        return img

    # Return the original if no skew was detected
    return image


# Retrieve the TESSERACT_CMD environment variable
tesseract_cmd = os.getenv("TESSERACT_CMD")

# If the environment variable is not set, raise an exception
if tesseract_cmd is None:
    raise EnvironmentError("The TESSERACT_CMD environment variable is not set.")

# Set the tesseract command for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def ensure_grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Image is already in grayscale, so no conversion is necessary
        return image
    else:
        # Convert the image to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, padding=50) -> np.ndarray:
    """
    Reduces the size of the image by a padding amount and then finds the largest rectangle.

    Args:
        image (np.ndarray): Image loaded in as numpy nd array
        padding (int, optional): Padding to reduce the image by. Defaults to 50.

    Returns:
        np.ndarray: cropped image
    """
    # Read the image
    original_height, original_width = image.shape[:2]

    # Calculate new image dimensions
    new_width = original_width - 2 * padding
    new_height = original_height - 2 * padding

    # Ensure new dimensions are positive
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    # Define the new image area (with padding applied)
    new_area = (padding, padding, new_width, new_height)

    # Crop the image to the new area
    cropped_image = image[
        new_area[1] : new_area[1] + new_area[3], new_area[0] : new_area[0] + new_area[2]
    ]

    return cropped_image


def get_coords_largest_rectangle(image) -> Tuple[int, int, int, int]:
    """Get the coordiates of the largest rectangle in the image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        Tuple[int, int, int, int]: x, y, w, h coordaintes
    """
    # Set a threshold to identify black (or very dark) regions in the image
    _, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the largest rectangle in the contours
    largest_area = 0
    largest_rectangle = None

    for cnt in contours:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Choose the largest area which is more likely to be our target rectangle
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)

    if largest_rectangle is not None:
        x, y, w, h = largest_rectangle

        return largest_rectangle
    else:
        return None


def increase_contrast(image):
    """Increases the contrast of the image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        image: image with increased contrast
    """
    # Convert to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

    # Convert back to BGR
    output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    return output


def preprocess_image(image):
    # Convert the image to grayscale
    binary_image = ensure_grayscale(image)

    # Apply Non-local Means Denoising
    blurred_image = cv2.fastNlMeansDenoising(binary_image, None, 5, 7, 21)

    # Adaptive thresholding
    # this considers small regions of the image to adaptively change the threshold and
    # might work better with varying lighting conditions
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((1, 1), np.uint8)

    # Dilation and erosion to close gaps between letters
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)

    # Morphological closing to close small holes in the foreground
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("Image", binary_image)
    # cv2.waitKey(0)
    coordinates = get_coords_largest_rectangle(binary_image)
    if coordinates is not None:
        x, y, w, h = coordinates
        cropped_image = binary_image[y : y + h, x : x + w]

    while True:
        # new_image = resize_image(cropped_image, 8)
        new_image = resize_image(cropped_image, 10)
        # cv2.imshow("Image", new_image)
        # cv2.waitKey(0)
        coordinates = get_coords_largest_rectangle(new_image)

        if coordinates is None:
            break

        x, y, w, h = coordinates

        aspect_ratio = float(w) / h
        min_aspect_ratio = 2
        max_aspect_ratio = 6

        # Check if aspect ratio is within expected range
        if not (min_aspect_ratio < aspect_ratio < max_aspect_ratio):
            break

        cropped_image = new_image[y : y + h, x : x + w]
        # cv2.imshow("Image", cropped_image)
        # cv2.waitKey(0)

    # cv2.imshow("STOPPED", cropped_image)
    # cv2.waitKey(0)
    return cropped_image


def extract_license_plate_text(preprocessed_image) -> str:
    """Inputs an image into tesseract to OCR it

    Args:
        preprocessed_image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        str: text of the licence plate
    """
    # Use Tesseract to do OCR on the preprocessed image
    custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(
        preprocessed_image,
        config=custom_config,
        lang="eng",
    )

    # Clean up the text
    # This might break it below
    for item in re.split(r"\s", text):
        if len(item) == 6:
            text = item
            break

    text = "".join(e for e in text if e.isalnum())

    return text


def extract_license_plate_text_all_ocr_modes(preprocessed_image) -> str:
    """Inputs an image into tesseract to OCR it. Runs against all PSM modes (Testing function)

    Args:
        preprocessed_image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        str: text of the licence plate
    """
    texts = []
    for i in range(0, 13):
        try:
            custom_config = f"--oem 3 --psm {i} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # You might need to adjust the psm value based on your images
            text = pytesseract.image_to_string(
                preprocessed_image,
                config=custom_config,
                lang="eng",
            )
            texts.append(f"{i}: {text}")
        except:
            pass

    return texts


def ocr_image(image: np.ndarray, coordinates: Tuple = None) -> str:
    """Get the string from coordinates in an image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array
        coordinates (Tuple): coordinate tuple (x, y, width, height)

    Returns:
        str: ocr result
    """
    # Expand the bounding box coordinates if provided and crop image
    if coordinates:
        x, y, width, height = coordinates

        # Crop the image based on the bounding box coordinates
        cropped_image = image[y : y + height, x : x + width]
    else:
        cropped_image = image

    # Preprocess the image
    preprocessed_image = preprocess_image(cropped_image)

    # Extract text
    license_plate_text = extract_license_plate_text(preprocessed_image)
    return license_plate_text
