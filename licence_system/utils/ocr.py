"""
This module defines defines the functions for extracting the plate text from image using OCR

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
from typing import Tuple
from io import BytesIO
import os
import re
import json

import cv2
import numpy as np
import pytesseract

import requests


# Retrieve the TESSERACT_CMD environment variable
tesseract_cmd = os.getenv("TESSERACT_CMD")

# If the environment variable is not set, raise an exception
if tesseract_cmd is None:
    raise EnvironmentError("The TESSERACT_CMD environment variable is not set.")

# Set the tesseract command for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Retrieve the OCR_API_KEY environment variable
api_key_ocr = os.getenv("OCR_API_KEY")

# If the environment variable is not set, raise an exception
if api_key_ocr is None:
    raise EnvironmentError("The OCR_API_KEY environment variable is not set.")


def ocr_api(image):
    """
    Sends an image to the OCR.space API for optical character recognition
    and returns the extracted text.

    Args:
        image (np.ndarray): The image to be sent for OCR.

    Returns:
        str: The cleaned text extracted from the image by the OCR API. If the text
        is not found or an error occurs, None is returned.

    Raises:
        HTTPError: If the request to the OCR.space API fails.
    """
    # Set the API endpoint
    url = "https://api.ocr.space/parse/image"

    # Encode the image as a JPEG in memory
    _, buffer = cv2.imencode(".jpg", image)  # pylint: disable=no-member
    io_buf = BytesIO(buffer)

    # Prepare the headers for the HTTP request
    headers = {
        "apikey": api_key_ocr,
    }

    # Prepare the payload for the POST request
    payload = {
        "isOverlayRequired": True,
        "language": "eng",
        "detectOrientation": True,
        "isCreateSearchablePdf": False,
        "isSearchablePdfHideTextLayer": False,
        "scale": True,
        "isTable": False,
        "OCREngine": 2,
    }

    files = {
        "file": ("image.jpg", io_buf, "image/jpeg"),
    }

    response = requests.post(
        url, headers=headers, data=payload, files=files, timeout=20
    )

    # Check the response
    if response.status_code != 200:
        print("Request failed with status code: ", response.status_code)

    # Extract the plate results
    api_result = json.loads(response.text)
    parsed_text = (
        api_result["ParsedResults"][0]["ParsedText"]
        if api_result["ParsedResults"]
        else None
    )

    if parsed_text:
        # Clean up the text
        parsed_text = re.sub(r"[^a-zA-Z0-9\n]", "", parsed_text)

        for item in re.split(r"\s", parsed_text):
            if len(item) == 6:
                parsed_text = item
                break

        parsed_text = "".join(e for e in parsed_text if e.isalnum())
    return parsed_text


def ensure_grayscale(image):
    """
    Ensures that an image is in grayscale format.

    Args:
        image (np.ndarray): The input image that needs to be ensured as grayscale.

    Returns:
        np.ndarray: The image in grayscale format.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Image is already in grayscale, so no conversion is necessary
        return image

    # Convert the image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member


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
    _, thresh = cv2.threshold(  # pylint: disable=no-member
        image, 15, 255, cv2.THRESH_BINARY_INV  # pylint: disable=no-member
    )

    # Find contours
    contours, _ = cv2.findContours(  # pylint: disable=no-member
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # pylint: disable=no-member
    )

    # Look for the largest rectangle in the contours
    largest_area = 0
    largest_rectangle = None

    for cnt in contours:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)  # pylint: disable=no-member
        area = w * h

        # Choose the largest area which is more likely to be our target rectangle
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)

    if largest_rectangle is not None:
        x, y, w, h = largest_rectangle
        return largest_rectangle
    return None


def increase_contrast(image):
    """Increases the contrast of the image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        image: image with increased contrast
    """
    # Convert to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # pylint: disable=no-member

    # Apply histogram equalization
    image_yuv[:, :, 0] = cv2.equalizeHist(  # pylint: disable=no-member
        image_yuv[:, :, 0]
    )

    # Convert back to BGR
    output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)  # pylint: disable=no-member

    return output


def preprocess_image(image):
    """
    Performs preprocessing on an image to prepare it for OCR.

    Args:
        image (np.ndarray): The image to be preprocessed.

    Returns:
        np.ndarray: The cropped and preprocessed image ready for OCR.
    """
    # Convert the image to grayscale
    binary_image = ensure_grayscale(image)

    # Apply Non-local Means Denoising
    blurred_image = cv2.fastNlMeansDenoising(  # pylint: disable=no-member
        binary_image, None, 5, 7, 21
    )

    # Adaptive thresholding
    # this considers small regions of the image to adaptively change the threshold and
    # might work better with varying lighting conditions
    binary_image = cv2.adaptiveThreshold(  # pylint: disable=no-member
        blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # pylint: disable=no-member
        cv2.THRESH_BINARY,  # pylint: disable=no-member
        11,
        2,
    )

    kernel = np.ones((1, 1), np.uint8)

    # Dilation and erosion to close gaps between letters
    binary_image = cv2.dilate(  # pylint: disable=no-member
        binary_image, kernel, iterations=1
    )
    binary_image = cv2.erode(  # pylint: disable=no-member
        binary_image, kernel, iterations=1
    )

    # Morphological closing to close small holes in the foreground
    binary_image = cv2.morphologyEx(  # pylint: disable=no-member
        binary_image, cv2.MORPH_CLOSE, kernel  # pylint: disable=no-member
    )

    # cv2.imshow("Image", binary_image)   # pylint: disable=no-member
    # cv2.waitKey(0)   # pylint: disable=no-member
    coordinates = get_coords_largest_rectangle(binary_image)
    if coordinates is not None:
        x, y, w, h = coordinates
        cropped_image = binary_image[y : y + h, x : x + w]

    while True:
        # new_image = resize_image(cropped_image, 8)
        new_image = resize_image(cropped_image, 10)
        # cv2.imshow("Image", new_image)   # pylint: disable=no-member
        # cv2.waitKey(0)   # pylint: disable=no-member
        coordinates = get_coords_largest_rectangle(new_image)

        if coordinates is None:
            break

        x, y, w, h = coordinates

        aspect_ratio = float(w) / h
        min_aspect_ratio = 2
        max_aspect_ratio = 6

        # Check if aspect ratio is within expected range
        if not min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            break

        # Check if the image is to small to be reagit pu
        if h < 30 or w < 60:
            break

        cropped_image = new_image[y : y + h, x : x + w]
        # cv2.imshow("Image", cropped_image)   # pylint: disable=no-member
        # cv2.waitKey(0)   # pylint: disable=no-member

    # cv2.imshow("STOPPED", cropped_image)   # pylint: disable=no-member
    # cv2.waitKey(0)   # pylint: disable=no-member
    return cropped_image


def extract_license_plate_text(preprocessed_image) -> str:
    """Inputs an image into tesseract to OCR it

    Args:
        preprocessed_image (np.ndarray): Image loaded in as numpy nd array

    Returns:
        str: text of the licence plate
    """
    # Use Tesseract to do OCR on the preprocessed image
    custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # pylint: disable=line-too-long
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
            custom_config = f"--oem 3 --psm {i} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # pylint: disable=line-too-long
            text = pytesseract.image_to_string(
                preprocessed_image,
                config=custom_config,
                lang="eng",
            )
            texts.append(f"{i}: {text}")
        except:  # pylint: disable=bare-except
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
    # preprocessed_image = preprocess_image(cropped_image)

    # Extract text
    # license_plate_text = extract_license_plate_text(preprocessed_image)

    # Extract text from API
    license_plate_text = ocr_api(cropped_image)
    return license_plate_text
