"""
This file defines the runner for inferencing the model

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import faulthandler
from licence_system.utils.model_class import LPR_Inference

faulthandler.enable()

inference_class = LPR_Inference(
    model_path="licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth",
    display_output=True,
    bbox_buffer=60,
)

inference_class.get_bounding_box_from_img(daddy_deco_goes_here)
