"""
This file defines the runner for inferencing the model

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import faulthandler
import os
import random
import numpy as np
from licence_system.utils.model_class import LPRInference
from licence_system.utils.ocr import ocr_image

faulthandler.enable()

inference_class = LPRInference(
    model_path="licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth",
    display_output=True,
)

IMAGE_PATH = "inference-images"
# read all images for inference
print("Loading Images...")
images = []
for file in os.listdir(IMAGE_PATH):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        images.append(os.path.join(IMAGE_PATH, file))

print("Successfully loaded images!")

img_index = random.randrange(0, len(images))
selected_img = images[img_index]
image = inference_class.get_bounding_box_from_img(selected_img)
image = np.asarray(image)

# Run OCR
print(ocr_image(image))
