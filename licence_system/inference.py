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
from PIL import Image, ImageDraw
from licence_system.utils.data_loader import show_imgs
from licence_system.utils.model_class import LPLocalNet

PATH = "licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth"
IMAGE_PATH = "inference-images"

faulthandler.enable()

print(f"Loading Model: {PATH}")
model = LPLocalNet()
state_dict = torch.load(PATH, map_location='cpu')
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
selected_img = images[2]
print(f"Opening image {selected_img}...")
DISPLAY_OUTPUT = True
BBOX_BUFFER = 60
with Image.open(selected_img).convert("L") as img:
    resized_img = img.copy().resize((416, 416))
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
    print(f"Estimate bounding box: {net_out.detach().cpu()}")
    
    #displaying output
    print("displaying output")
    if DISPLAY_OUTPUT is True:
        bbox=net_out.detach().cpu()[0]
        #bbox = torch.round(bbox).int()
        print("bounding box: ", bbox)
        imgcp = img.copy()
        
        # resize output bounding to fix og image
        original_width, original_height = imgcp.size  # Replace with your original image dimensions
        scaling_factor_width = original_width / 416
        scaling_factor_height = original_height / 416

        x1 = bbox[0].item() * scaling_factor_width - BBOX_BUFFER
        y1 = bbox[1].item() * scaling_factor_height - BBOX_BUFFER
        x2 = bbox[2].item() * scaling_factor_width + BBOX_BUFFER
        y2 = bbox[3].item() * scaling_factor_height + BBOX_BUFFER
        
        # draw bounding box onto img
        #imgcp_draw = ImageDraw.Draw(imgcp)
        #imgcp_draw.rectangle([x1,y1,x2,y2], fill = None, outline = "white", width=7)
        cropped_img = imgcp.crop((x1, y1, x2, y2))
        
        # cropped output gets fed into OCR code
        cropped_img.show()
