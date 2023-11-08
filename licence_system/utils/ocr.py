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


def find_and_crop_square(image):
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

        # Crop the image according to the found rectangle
        cropped_image = image[y : y + h, x : x + w]

        # Save or return the cropped image
        return cropped_image
    else:
        raise Exception("S")


def increase_contrast(image):
    # Convert to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

    # Convert back to BGR
    output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    return output


def binarize_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary_image


def remove_noise(image):
    # Convert to grayscale
    gray_image = image

    # Apply non-local means denoising
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    return denoised_image


def deskew_image(image):
    gray = image
    # Use thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Calculate the moments to find the centroid
    moments = cv2.moments(binary)
    # Calculate the skew based on the moments
    if moments["mu02"] != 0:
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, -0.5 * gray.shape[0] * skew], [0, 1, 0]])
        # Warp the image to correct the skew
        img = cv2.warpAffine(
            binary,
            M,
            (gray.shape[1], gray.shape[0]),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        )
        return img
    return image  # Return the original if no skew was detected


# Retrieve the TESSERACT_CMD environment variable
tesseract_cmd = os.getenv("TESSERACT_CMD")

# If the environment variable is not set, raise an exception
if tesseract_cmd is None:
    raise EnvironmentError("The TESSERACT_CMD environment variable is not set.")

# Set the tesseract command for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def preprocess_image(image):
    # deskewed_image = deskew_image(image)
    # Convert the image to grayscale
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary_image = increase_contrast(image)

    # # Resize for better accuracy
    # width = int(binary_image.shape[1] * 2)
    # height = int(binary_image.shape[0] * 2)
    # dimensions = (width, height)

    # binary_image = cv2.resize(binary_image, dimensions, interpolation=cv2.INTER_CUBIC)
    # binary_image = binarize_image(binary_image)
    # binary_image = remove_noise(binary_image)
    # binary_image = deskew_image(binary_image)

    # # # Use GaussianBlur to reduce noise and improve OCR accuracy
    # blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # # # Apply Non-local Means Denoising
    blurred_image = cv2.fastNlMeansDenoising(binary_image, None, 5, 7, 21)

    # Instead of GaussianBlur, let's use a median blur which preserves edges better
    # blurred_image = cv2.medianBlur(binary_image, 3)

    # Adaptive thresholding - this considers small regions of the image to adaptively
    # change the threshold and might work better with varying lighting conditions
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # # # Thresholding to get a binary image
    # # _, binary_image = cv2.threshold(
    # #     blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # # )

    # # You can experiment with dilation and erosion to close gaps between letters
    # kernel = np.ones((1, 1), np.uint8)
    # binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    # binary_image = cv2.erode(binary_image, kernel, iterations=10)

    cv2.imshow("Image", binary_image)
    cv2.waitKey(0)
    cropped_image = find_and_crop_square(binary_image)
    cv2.imshow("Image", cropped_image)
    cv2.waitKey(0)

    return cropped_image

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
    # # Use Tesseract to do OCR on the preprocessed image
    # custom_config = r"--oem 3 --psm 8"  # You might need to adjust the psm value based on your images
    # text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

    # # Clean up the text
    # text = "".join(e for e in text if e.isalnum())
    texts = []
    for i in range(0, 13):
        try:
            custom_config = f"--oem 3 --psm {i} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # You might need to adjust the psm value based on your images
            text = pytesseract.image_to_string(
                preprocessed_image,
                config=custom_config,
                lang="eng",
            )
            texts.append(text)
        except:
            pass

    return texts


def ocr_with_coordinates(image: np.ndarray, coordinates: Tuple) -> str:
    """Get the string from coordinates in an image

    Args:
        image (np.ndarray): Image loaded in as numpy nd array
        coordinates (Tuple): coordinate tuple (x, y, width, height)

    Returns:
        str: ocr result
    """
    # x, y, width, height = coordinates

    # # # Crop the image
    # # cropped_image = image[y : y + height, x : x + width]
    # # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)
    # # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=30)

    # # # # Convert to greyscale

    # # # # Binarization
    # # _, binary = cv2.threshold(
    # #     cropped_image, 0, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # # )

    # # # # Resize for better accuracy
    # # binary = cv2.resize(binary, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)

    # # # # binary = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(binary)

    # # # cv2.imshow("Image", binary)
    # # # cv2.imshow("Image", cropped_image)
    # # # cv2.waitKey(0)

    # # # Extract text
    # # # psm=8, 6
    # # extracted_text = pytesseract.image_to_string(
    # #     cropped_image,
    # #     lang="eng",
    # #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # # )

    # # return extracted_text

    # # x, y, width, height = coordinates

    # # # Crop the image
    # # cropped_image = image[y : y + height, x : x + width]
    # # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)

    # # # Sharpening
    # # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # # sharpened = cv2.filter2D(cropped_image, -1, kernel)

    # # # Binarization
    # # _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # # Resize for better accuracy
    # # # binary = cv2.resize(binary, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
    # # # binary = cv2.resize(binary, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

    # # cv2.imshow("Image", binary)
    # # cv2.waitKey(0)

    # # # Extract text
    # # extracted_text = pytesseract.image_to_string(
    # #     binary,
    # #     lang="eng",
    # #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # # )

    # # return extracted_text.strip()

    # # x, y, width, height = coordinates

    # # # Crop the image
    # # cropped_image = image[y : y + height, x : x + width]
    # # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # # # Mild Sharpening
    # # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # # sharpened = cv2.filter2D(cropped_image, -1, kernel)

    # # # Binarization using Adaptive Thresholding
    # # binary = cv2.adaptiveThreshold(
    # #     sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # # )

    # # # binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # # cv2.imshow("Image", binary)
    # # cv2.waitKey(0)

    # # # Extract text
    # # extracted_text = pytesseract.image_to_string(
    # #     binary,
    # #     lang="eng",
    # #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # # )

    # # return extracted_text.strip()

    # # x, y, width, height = coordinates

    # # # Crop the image
    # # cropped_image = image[y:y+height, x:x+width]
    # # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.6, beta=20)

    # # # Adaptive Thresholding
    # # binary = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # #                                cv2.THRESH_BINARY, 11, 2)

    # # cv2.imshow("Image", binary)
    # # cv2.waitKey(0)

    # # # Extract text
    # # extracted_text = pytesseract.image_to_string(
    # #     binary,
    # #     lang="eng",
    # #     config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # # )

    # # return extracted_text.strip()

    # x, y, width, height = coordinates

    # # Crop the image
    # cropped_image = image[y : y + height, x : x + width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=0.5, beta=30)
    # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=30)
    # # cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=30)

    # # https://yashlahoti.medium.com/number-plate-recognition-in-python-using-tesseract-ocr-cc15853aca36
    # # https://github.com/kangsunghyun111/Car-license-plate-recognition-using-tesseract-OCR/blob/master/main.cpp
    # # https://stackoverflow.com/questions/66935787/why-does-tesseract-not-recognize-the-text-on-the-licence-plate-while-easyocr-doe
    # # https://stackoverflow.com/questions/55349307/how-to-tune-tesseract-for-identifying-number-plate-of-a-car-more-accurately
    # # https://stackoverflow.com/questions/72381645/python-tesseract-license-plate-recognition
    # # https://www.google.com/search?q=teseract+ocr+on+blurry+licence+plate+image
    # cropped_image = cv2.bilateralFilter(cropped_image, 100, 5, 5)
    # # cropped_image = cv2.Canny(cropped_image, 10, 5) # no
    # # contours = cv2.findContours(cropped_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # contours = imutils.grab_contours(contours)
    # # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    # # # Sharpening
    # # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # # cropped_image = cv2.filter2D(cropped_image, -1, kernel)

    # # # Convert to greyscale

    # # # Binarization
    # # _, binary = cv2.threshold(
    # #     cropped_image, 0, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # # )

    # #

    # # # cv2.imshow("Image", binary)
    # # cv2.imshow("Image", cropped_image)
    # # cv2.waitKey(0)

    # # # Extract text
    # # # psm=8, 6
    # # extracted_text = pytesseract.image_to_string(
    # #     cropped_image,
    # #     lang="eng",
    # #     config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # #     # config="--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # # )
    # return extracted_text

    x, y, width, height = coordinates

    cropped_image = image[y : y + height, x : x + width]

    preprocessed_image = preprocess_image(cropped_image)

    # Extract text
    license_plate_text = extract_license_plate_text(preprocessed_image)
    return license_plate_text
