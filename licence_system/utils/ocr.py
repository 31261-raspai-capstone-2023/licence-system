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

import imutils

# Retrieve the TESSERACT_CMD environment variable
tesseract_cmd = os.getenv("TESSERACT_CMD")
# tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# If the environment variable is not set, raise an exception
if tesseract_cmd is None:
    raise EnvironmentError("The TESSERACT_CMD environment variable is not set.")

# Set the tesseract command for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# def deskew_image(image):
#     # Convert the image to grayscale and then apply edge detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#     # Use Hough transform to detect lines in the image
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

#     # Calculate the average angle of the lines
#     angles = []
#     for rho, theta in lines[:, 0]:
#         angles.append(theta)

#     # Compute the angle of rotation needed to deskew the image
#     average_angle = np.mean(angles)
#     angle_degrees = (average_angle * 180 / np.pi) - 90
#     angle_degrees = -angle_degrees if angle_degrees < 0 else 180 - angle_degrees

#     # Rotate the image around its center
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
#     rotated = cv2.warpAffine(
#         image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#     )

#     return rotated


def preprocess_image(image):
    # Convert the image to grayscale
    # deskewed_image = deskew_image(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Use GaussianBlur to reduce noise and improve OCR accuracy
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # # Apply Non-local Means Denoising
    # blurred_image = cv2.fastNlMeansDenoising(gray_image, None, 5, 7, 21)

    # Instead of GaussianBlur, let's use a median blur which preserves edges better
    blurred_image = cv2.medianBlur(gray_image, 3)

    # Adaptive thresholding - this considers small regions of the image to adaptively
    # change the threshold and might work better with varying lighting conditions
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Resize for better accuracy
    width = int(binary_image.shape[1] * 2)
    height = int(binary_image.shape[0] * 2)
    dimensions = (width, height)

    binary_image = cv2.resize(binary_image, dimensions, interpolation=cv2.INTER_CUBIC)

    # # Thresholding to get a binary image
    # _, binary_image = cv2.threshold(
    #     blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # You can experiment with dilation and erosion to close gaps between letters
    kernel = np.ones((1, 1), np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=10)

    cv2.imshow("Image", binary_image)
    cv2.waitKey(0)

    return binary_image

    # # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply Non-local Means Denoising
    # denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # # Apply CLAHE to improve the contrast after denoising
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # contrast_image = clahe.apply(denoised_image)

    # # Adaptive thresholding after improving contrast
    # binary_image = cv2.adaptiveThreshold(
    #     contrast_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )

    # # Morphological closing (dilation followed by erosion) to close small holes in the foreground
    # kernel = np.ones((3, 3), np.uint8)
    # closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("Image", closing_image)
    # cv2.waitKey(0)

    # return closing_image


def extract_license_plate_text(preprocessed_image):
    # Use Tesseract to do OCR on the preprocessed image
    custom_config = r"--oem 3 --psm 8"  # You might need to adjust the psm value based on your images
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

    # Clean up the text
    text = "".join(e for e in text if e.isalnum())

    return text


def ocr_with_coordinates(image: np.ndarray, coordinates: Tuple) -> str:
    """Get the string from coordinates in an image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array
        coordinates (Tuple): coordinate tuple (x, y, width, height)

    Returns:
        str: ocr result
    """
    # x, y, width, height = coordinates

    # # Crop the image
    # cropped_image = image[y : y + height, x : x + width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)
    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=30)

    # # # Convert to greyscale

    # # # Binarization
    # _, binary = cv2.threshold(
    #     cropped_image, 0, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # # # Resize for better accuracy
    # binary = cv2.resize(binary, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)

    # # # binary = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(binary)

    # # cv2.imshow("Image", binary)
    # # cv2.imshow("Image", cropped_image)
    # # cv2.waitKey(0)

    # # Extract text
    # # psm=8, 6
    # extracted_text = pytesseract.image_to_string(
    #     cropped_image,
    #     lang="eng",
    #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # )

    # return extracted_text

    # x, y, width, height = coordinates

    # # Crop the image
    # cropped_image = image[y : y + height, x : x + width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)

    # # Sharpening
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpened = cv2.filter2D(cropped_image, -1, kernel)

    # # Binarization
    # _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Resize for better accuracy
    # # binary = cv2.resize(binary, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
    # # binary = cv2.resize(binary, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("Image", binary)
    # cv2.waitKey(0)

    # # Extract text
    # extracted_text = pytesseract.image_to_string(
    #     binary,
    #     lang="eng",
    #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # )

    # return extracted_text.strip()

    # x, y, width, height = coordinates

    # # Crop the image
    # cropped_image = image[y : y + height, x : x + width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # # Mild Sharpening
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # sharpened = cv2.filter2D(cropped_image, -1, kernel)

    # # Binarization using Adaptive Thresholding
    # binary = cv2.adaptiveThreshold(
    #     sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )

    # # binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("Image", binary)
    # cv2.waitKey(0)

    # # Extract text
    # extracted_text = pytesseract.image_to_string(
    #     binary,
    #     lang="eng",
    #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # )

    # return extracted_text.strip()

    # x, y, width, height = coordinates

    # # Crop the image
    # cropped_image = image[y:y+height, x:x+width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.6, beta=20)

    # # Adaptive Thresholding
    # binary = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow("Image", binary)
    # cv2.waitKey(0)

    # # Extract text
    # extracted_text = pytesseract.image_to_string(
    #     binary,
    #     lang="eng",
    #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # )

    # return extracted_text.strip()

    x, y, width, height = coordinates

    # Crop the image
    cropped_image = image[y : y + height, x : x + width]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=0.5, beta=30)
    cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)
    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=30)

    # https://yashlahoti.medium.com/number-plate-recognition-in-python-using-tesseract-ocr-cc15853aca36
    # https://github.com/kangsunghyun111/Car-license-plate-recognition-using-tesseract-OCR/blob/master/main.cpp
    # https://stackoverflow.com/questions/66935787/why-does-tesseract-not-recognize-the-text-on-the-licence-plate-while-easyocr-doe
    # https://stackoverflow.com/questions/55349307/how-to-tune-tesseract-for-identifying-number-plate-of-a-car-more-accurately
    # https://stackoverflow.com/questions/72381645/python-tesseract-license-plate-recognition
    # https://www.google.com/search?q=teseract+ocr+on+blurry+licence+plate+image
    cropped_image = cv2.bilateralFilter(cropped_image, 100, 5, 5)
    # cropped_image = cv2.Canny(cropped_image, 10, 5) # no
    # contours = cv2.findContours(cropped_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    # # Sharpening
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # cropped_image = cv2.filter2D(cropped_image, -1, kernel)

    # # Convert to greyscale

    # # Binarization
    # _, binary = cv2.threshold(
    #     cropped_image, 0, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    #

    # cv2.imshow("Image", binary)
    cv2.imshow("Image", cropped_image)
    cv2.waitKey(0)

    # Extract text
    # psm=8, 6
    extracted_text = pytesseract.image_to_string(
        cropped_image,
        lang="eng",
        config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        # config="--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    cropped_image = image[y : y + height, x : x + width]

    preprocessed_image = preprocess_image(cropped_image)

    # # Extract text
    license_plate_text = extract_license_plate_text(preprocessed_image)
    return license_plate_text

    return extracted_text
