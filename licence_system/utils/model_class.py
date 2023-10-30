import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class LPR_Training_Dataset_Processed:
    def __init__(
        self, image_path: str, annotation_path: str, testing_images_size: float
    ):
        self.IMAGES_FOLDER: str = image_path
        self.ANNOTATIONS_FOLDER: str = annotation_path
        self.TESTING_IMAGES_SIZE: float = testing_images_size

        self.img_list = os.listdir(self.IMAGES_FOLDER)

        # self.training_data: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        self.training_data: list = []
        # self.testing_data: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        self.testing_data: list = []

        # self.train_X: List[np.ndarray] = []
        self.train_X: torch.Tensor = None
        # self.train_Y: List[Tuple[int, int, int, int]] = []
        self.train_Y: torch.Tensor = None
        # self.test_X: List[np.ndarray] = []
        self.test_X: torch.Tensor = None
        # self.test_Y: List[Tuple[int, int, int, int]] = []
        self.test_Y: torch.Tensor = None

        self.neural_network: nn.Module = None

    def create_training_data(self):
        """_summary_"""
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


class LPLocalNet(nn.Module):
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
