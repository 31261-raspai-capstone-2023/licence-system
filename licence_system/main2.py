import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from licence_system.utils.data_loader import show_imgs
from licence_system.utils.model_class import LPLocalNet
from PIL import Image

PATH = "licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc78.84.pth"
IMAGE_PATH = "inference-images"

print(f"Loading Model: {PATH}")
model = LPLocalNet()
model.load_state_dict(torch.load(PATH, map_location="cpu"))
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
    print(f"Estimate bounding box: {net_out}")
    show_imgs([[selected_img, img, (0, 0, 0, 0), (0, 0, 0, 0)]])
