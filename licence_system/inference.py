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

faulthandler.enable()

class Inference():
    PATH = "licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth" #path of ML model
    DISPLAY_OUTPUT = False # when true, shows a window with the output of the original image cropped to the detected bounding box
    BBOX_BUFFER = 60 # size to add around the detected bounding box for safety, should be relative to the original image size
    #IMAGE_PATH = "inference-images"

    def __init__(self):
        print(f"Loading Model: {PATH}")
        self.model = LPLocalNet()
        state_dict = torch.load(PATH, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Successfully loaded model!")

    def Infer(self, selected_img):
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
            net_out = self.model(model_in).detach().cpu()[0]
            print("Input Shape:", model_in.shape)
            print(f"Estimate bounding box: {net_out}")
            
            bbox = net_out
            imgcp = img.copy()
            
            # resize output bounding to fix og image
            original_width, original_height = imgcp.size  # Replace with your original image dimensions
            scaling_factor_width = original_width / 416
            scaling_factor_height = original_height / 416

            x1 = bbox[0].item() * scaling_factor_width - self.BBOX_BUFFER
            y1 = bbox[1].item() * scaling_factor_height - self.BBOX_BUFFER
            x2 = bbox[2].item() * scaling_factor_width + self.BBOX_BUFFER
            y2 = bbox[3].item() * scaling_factor_height + self.BBOX_BUFFER
            
            # draw bounding box onto img
            #imgcp_draw = ImageDraw.Draw(imgcp)
            #imgcp_draw.rectangle([x1,y1,x2,y2], fill = None, outline = "white", width=7)
            cropped_img = imgcp.crop((x1, y1, x2, y2))
            
            #displaying output
            print("displaying output")
            if self.DISPLAY_OUTPUT is True:
                # cropped output gets fed into OCR code
                cropped_img.show()
                
            return cropped_img
