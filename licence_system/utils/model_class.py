"""
This file defines the model classes

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from typing import Tuple


class LPR_Training_Dataset_Processed:
    def __init__(
        self, image_path: str, annotation_path: str, testing_images_size: float
    ):
        self.IMAGES_FOLDER: str = image_path
        self.ANNOTATIONS_FOLDER: str = annotation_path
        self.TESTING_IMAGES_SIZE: float = testing_images_size

        self.img_list = os.listdir(self.IMAGES_FOLDER)

        self.training_data: list = []
        self.testing_data: list = []

        self.train_X: torch.Tensor = None
        self.train_Y: torch.Tensor = None
        self.test_X: torch.Tensor = None
        self.test_Y: torch.Tensor = None

        self.neural_network: nn.Module = None

    def create_training_data(self):
        """Create training data function"""
        testing_size = len(self.img_list) * self.TESTING_IMAGES_SIZE

        i = 0
        for a in tqdm(range(max(2000, len(self.img_list)))):
            img_data = self.img_list[a]
            img_path = os.path.join(self.IMAGES_FOLDER, img_data)
            ANNOTATIONS_FOLDER = os.path.join(
                self.ANNOTATIONS_FOLDER, img_data.replace(".jpg", ".xml")
            )  # get required image annotations
            with Image.open(img_path).convert("L") as img:
                with open(ANNOTATIONS_FOLDER) as source:
                    root = ET.parse(source).getroot()

                    # Iterate through the XML and extract bounding box coordinates
                    for obj in root.findall(".//object"):
                        bndbox = obj.find("bndbox")
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)

                    bounding_box_coordinates = (
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    )  # resize bounding box to fit resized image

                    if i < testing_size:
                        # self.testing_data.append([torch.Tensor(np.asarray(img)).view(-1, 416, 416) / 255, torch.Tensor(bounding_box_coordinates)])
                        self.testing_data.append(
                            [np.asarray(img), bounding_box_coordinates]
                        )
                    else:
                        self.training_data.append(
                            [np.asarray(img), bounding_box_coordinates]
                        )
                        # self.training_data.append([torch.Tensor(np.asarray(img)).view(-1, 416, 416) / 255, torch.Tensor(bounding_box_coordinates)])
            i += 1

        np.random.shuffle(self.training_data)
        print(f"Training Images: {len(self.training_data)}")
        print(f"Testing Images: {len(self.testing_data)}")


class LPR_Inference:
    def __init__(
        self, model_path: str, display_output: bool = False, bbox_buffer: int = 60
    ):
        self.PATH = model_path
        self.MODEL = LPLocalNet()
        self.DISPLAY_OUTPUT = display_output
        self.BBOX_BUFFER = bbox_buffer
        
        self.__load()

    def __load(self):
        print(f"Loading Model: {self.PATH}")
        self.state_dict = torch.load(self.PATH, map_location="cpu")
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        print("Successfully loaded model!")
        
    def __image_preprocessing(self, image):
        X = torch.Tensor(image)
        X = X / 255.0
        return X
    
    def __resize_bounding_box_from_image(self, image, bbox) -> Tuple[int, int, int, int]:
        (original_width, original_height) = image.size
        
        scaling_factor_width = original_width / 416
        scaling_factor_height = original_height / 416

        x1 = bbox[0].item() * scaling_factor_width - self.BBOX_BUFFER
        y1 = bbox[1].item() * scaling_factor_height - self.BBOX_BUFFER
        x2 = bbox[2].item() * scaling_factor_width + self.BBOX_BUFFER
        y2 = bbox[3].item() * scaling_factor_height + self.BBOX_BUFFER

        return (x1, y1, x2, y2)
        # draw bounding box onto img
        # imgcp_draw = ImageDraw.Draw(imgcp)
        # imgcp_draw.rectangle([x1,y1,x2,y2], fill = None, outline = "white", width=7)
        
    def get_bounding_box_from_img(self, image):
        with Image.open(image).convert("L") as img:
            # Resize the image to put it into the model
            resized_img = img.copy().resize((416, 416))
            numpy_data = np.array(resized_img)
            # print(numpy_data)
            
            # Preprocess the image
            print("Preprocessing image...")
            X = self.__image_preprocessing(numpy_data)
            
            # Pass image into model
            print("Passing image into model...")
            model_in = X.view(-1, 1, 416, 416)
            # print(model_in)
            net_out = self.model(model_in).detach().cpu()[0]
            print("Input Shape:", model_in.shape)
            print(f"Estimate bounding box: {net_out}")

            # Resize the bounding box to orginal dimensions
            image_copy = img.copy()
            bounding_box_coordinates = self.__resize_bounding_box_from_image(image_copy, net_out):

            # Crop the original image to the bounding box
            cropped_img = image_copy.crop(bounding_box_coordinates)

            if self.DISPLAY_OUTPUT is True:
                print("Displaying Output")
                cropped_img.show()
                # cropped output to get fed into OCR code

            return cropped_img


class LPLocalNet(nn.Module):
    """Neural network model class

    Args:
        nn (class): inherited base class
    """

    def __init__(self):
        super(LPLocalNet, self).__init__()

        # CNNs for grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)

        # Connecting CNN outputs with Fully Connected layers for bounding box
        self.box_fc1 = nn.Linear(in_features=12288, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_out = nn.Linear(in_features=120, out_features=4)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv5(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=4, stride=2)

        t = torch.flatten(t, start_dim=1)

        box_t = self.box_fc1(t)
        box_t = F.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)

        box_t = self.box_out(box_t)

        return box_t
