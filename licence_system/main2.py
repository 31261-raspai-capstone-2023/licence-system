"""
This file defines the runner for inferencing the model

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import os
import random
import faulthandler
import numpy as np
import torch
from PIL import Image
from licence_system.utils.data_loader import show_imgs
from licence_system.utils.model_class import LPLocalNet

PATH = "licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc78.84.pth"
IMAGE_PATH = "inference-images"

faulthandler.enable()

print(f"Loading Model: {PATH}")
model = LPLocalNet()
map_location = torch.device("cpu")
state_dict = torch.load(PATH, map_location=map_location)
model.load_state_dict(state_dict)
model.eval()
print("Successfully loaded model!")

# read all images for inference
print("Loading Images...")
images = []
for file in os.listdir(IMAGE_PATH):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        images.append(os.path.join(IMAGE_PATH, file))

print("Successfully loaded images!")

img_index = random.randrange(0, len(images))
selected_img = images[img_index]
print(f"Opening image {selected_img}...")
with Image.open(selected_img).convert("L") as img:
    resized_img = img.resize((416, 416))
    numpy_data = np.array(resized_img)
    print("Preprocessing image...")
    # print(numpy_data)
    X = torch.Tensor(numpy_data)
    X = X / 255.0
    print("Passing image into model...")
    model_in = X.view(-1, 1, 416, 416)
    print(model_in)
    net_out = model(model_in)
    print("Input Shape:", model_in.shape)
    out1 = model.conv1(model_in)
    print("After conv1:", out1.shape)
    # print(f"Estimate bounding box: {net_out}")
    # show_imgs([[selected_img, img, (0, 0, 0, 0), (0, 0, 0, 0)]])
