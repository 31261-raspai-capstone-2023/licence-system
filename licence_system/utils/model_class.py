"""
This module defines model classes for license plate detection and recognition.

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

import os
import xml.etree.ElementTree as ET
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class LPRTrainingDatasetProcessed:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """
    A class to handle the processed training dataset for license plate recognition.
    """

    def __init__(
        self, image_path: str, annotation_path: str, testing_images_size: float
    ):
        self.IMAGES_FOLDER = image_path  # pylint: disable=invalid-name
        self.ANNOTATIONS_FOLDER = annotation_path  # pylint: disable=invalid-name
        self.TESTING_IMAGES_SIZE = testing_images_size  # pylint: disable=invalid-name

        self.img_list = os.listdir(self.IMAGES_FOLDER)

        self.training_data = []
        self.testing_data = []

        self.train_X: torch.Tensor = None  # pylint: disable=invalid-name
        self.train_Y: torch.Tensor = None  # pylint: disable=invalid-name
        self.test_X: torch.Tensor = None  # pylint: disable=invalid-name
        self.test_Y: torch.Tensor = None  # pylint: disable=invalid-name

        self.neural_network: nn.Module = None

    def create_training_data(self):  # pylint: disable=too-many-locals
        """
        Creates and processes training data from images and annotations.
        """
        testing_size = len(self.img_list) * self.TESTING_IMAGES_SIZE

        i = 0
        for a in tqdm(range(max(2000, len(self.img_list)))):
            img_data = self.img_list[a]
            img_path = os.path.join(self.IMAGES_FOLDER, img_data)
            ANNOTATIONS_FOLDER = os.path.join(  # pylint: disable=invalid-name
                self.ANNOTATIONS_FOLDER, img_data.replace(".jpg", ".xml")
            )  # get required image annotations
            with Image.open(img_path).convert("L") as img:
                with open(  # pylint: disable=unspecified-encoding
                    ANNOTATIONS_FOLDER
                ) as source:
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
                        # self.testing_data.append([torch.Tensor(np.asarray(img)).view(-1, 416, 416) / 255, torch.Tensor(bounding_box_coordinates)]) # pylint: disable=line-too-long
                        self.testing_data.append(
                            [np.asarray(img), bounding_box_coordinates]
                        )
                    else:
                        self.training_data.append(
                            [np.asarray(img), bounding_box_coordinates]
                        )
                        # self.training_data.append([torch.Tensor(np.asarray(img)).view(-1, 416, 416) / 255, torch.Tensor(bounding_box_coordinates)])# pylint: disable=line-too-long
            i += 1

        np.random.shuffle(self.training_data)
        print(f"Training Images: {len(self.training_data)}")
        print(f"Testing Images: {len(self.testing_data)}")


class LPRInference:
    """
    A class to handle the inference process for license plate detection.

    Attributes:
        model_path (str): Path to the trained model file.
        display_output (bool): Whether to display the output image.
        bbox_buffer (int): Buffer size to adjust the bounding box around the detected license plate.
    """

    def __init__(
        self, model_path: str, display_output: bool = False, bbox_buffer: int = 15
    ):
        self.PATH = model_path  # pylint: disable=invalid-name
        self.MODEL = LPLocalNet()  # pylint: disable=invalid-name
        self.DISPLAY_OUTPUT = display_output  # pylint: disable=invalid-name
        self.BBOX_BUFFER = bbox_buffer  # pylint: disable=invalid-name

        self.__load()

    def __load(self):
        """
        Loads the model from the specified path.
        """
        print(f"Loading Model: {self.PATH}")
        self.state_dict = torch.load(self.PATH, map_location="cpu")
        self.MODEL.load_state_dict(self.state_dict)
        self.MODEL.eval()
        print("Successfully loaded model!")

    def __image_preprocessing(self, image):
        """
        Perform preprocessing
        """
        X = torch.Tensor(image)  # pylint: disable=invalid-name
        X = X / 255.0  # pylint: disable=invalid-name
        return X

    def __resize_bounding_box_from_image(
        self, image, bbox
    ) -> Tuple[int, int, int, int]:
        """
        Resize bounding box from image
        """
        (original_width, original_height) = image.size

        scaling_factor_width = original_width / 416
        scaling_factor_height = original_height / 416

        x1 = (bbox[0].item() - self.BBOX_BUFFER) * scaling_factor_width
        y1 = (bbox[1].item() - self.BBOX_BUFFER) * scaling_factor_height
        x2 = (bbox[2].item() + self.BBOX_BUFFER) * scaling_factor_width
        y2 = (bbox[3].item() + self.BBOX_BUFFER) * scaling_factor_height

        return (x1, y1, x2, y2)
        # draw bounding box onto img
        # imgcp_draw = ImageDraw.Draw(imgcp)
        # imgcp_draw.rectangle([x1,y1,x2,y2], fill = None, outline = "white", width=7)

    def get_img_from_tensor(self, pred: torch.Tensor):
        """
        Converts a PyTorch tensor into a PIL image.

        Args:
            pred (torch.Tensor): A 4D tensor representing the predicted image,
                                with dimensions corresponding to
                                [batch_size, channels, height, width].

        Returns:
            Image: A PIL image obtained from the input tensor.
        """
        pred = pred.data.cpu().numpy()
        pred = pred[0].transpose((1, 2, 0)) * 255.0

        pred = pred.astype(np.uint8)
        return Image.fromarray(pred)

    def get_bb_from_tensor(self, tensor):
        """
        Processes an input tensor to extract the bounding box
        and crops the image to this bounding box.

        Args:
            tensor (torch.Tensor): A 4D tensor representing the input image
                                    to the model, with dimensions corresponding to
                                    [batch_size, channels, height, width].

        Returns:
            Image: A PIL image that is cropped to the predicted bounding box coordinates.
        """
        X = tensor.detach().clone()  # pylint: disable=invalid-name
        # Pass image into model
        print("Passing image into model...")
        model_in = X.view(-1, 1, 416, 416)
        # print(model_in)
        net_out = self.MODEL(model_in).detach().cpu()[0]
        print("Input Shape:", model_in.shape)
        print(f"Estimate bounding box: {net_out}")

        # Resize the bounding box to orginal dimensions
        image_copy = self.get_img_from_tensor(tensor)
        bounding_box_coordinates = self.__resize_bounding_box_from_image(
            image_copy, net_out
        )

        # Crop the original image to the bounding box
        cropped_img = image_copy.crop(bounding_box_coordinates)

        if self.DISPLAY_OUTPUT is True:
            print("Displaying Output")
            cropped_img.show()
            # cropped output to get fed into OCR code

        return cropped_img

    def get_bounding_box_from_img(self, image):
        """
        Obtains the bounding box from an image and returns the cropped image at the bounding box.

        Args:
            image (Union[str, Image.Image]): The input image which can be a file path
                                                or a PIL Image object.

        Returns:
            Image: A PIL image cropped to the bounding box predicted by the model.
                    If 'DISPLAY_OUTPUT' is True, it also displays the cropped image.

        Raises:
            ValueError: If the provided image input is neither a file path nor an Image object.
        """

        if isinstance(image, str):
            # Open the image as a grayscale image
            img = Image.open(image).convert("L")
        elif isinstance(image, Image.Image):
            # If image is already an Image object, ensure it's in grayscale
            if image.mode != "L":
                img = image.convert("L")
            else:
                img = image
        else:
            raise ValueError(
                "The provided image input is neither a file path nor an Image object."
            )

        with img:
            # Resize the image to put it into the model
            resized_img = img.copy().resize((416, 416))
            numpy_data = np.array(resized_img)
            # print(numpy_data)

            # Preprocess the image
            print("Preprocessing image...")
            X = self.__image_preprocessing(numpy_data)  # pylint: disable=invalid-name

            # Pass image into model
            print("Passing image into model...")
            model_in = X.view(-1, 1, 416, 416)
            # print(model_in)
            net_out = self.MODEL(model_in).detach().cpu()[0]
            print("Input Shape:", model_in.shape)
            print(f"Estimate bounding box: {net_out}")

            # Resize the bounding box to orginal dimensions
            image_copy = img.copy()
            bounding_box_coordinates = self.__resize_bounding_box_from_image(
                image_copy, net_out
            )

            # Crop the original image to the bounding box
            cropped_img = image_copy.crop(bounding_box_coordinates)

            if self.DISPLAY_OUTPUT is True:
                print("Displaying Output")
                cropped_img.show()
                # cropped output to get fed into OCR code

            return cropped_img


class LPLocalNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    A convolutional neural network module for localizing license plates in images.

    Inherits from nn.Module.
    """

    def __init__(self):
        super().__init__()

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
        """
        Defines the forward pass of the neural network.

        Args:
            t (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The tensor containing the predicted bounding box coordinates.
        """
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
        t = F.avg_pool2d(t, kernel_size=4, stride=2)  # pylint: disable=not-callable

        t = torch.flatten(t, start_dim=1)

        box_t = self.box_fc1(t)
        box_t = F.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)

        box_t = self.box_out(box_t)

        return box_t
