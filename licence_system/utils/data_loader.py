"""
This file defines all the utility functions for data loading.

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import os
import shutil

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from licence_system.utils.model_class import LPRTrainingDatasetProcessed
from licence_system.utils.logger import logger


def separate_images_and_annotations(
    source_folder: str, images_folder: str, annotations_folder: str
):
    """
    Separate image and annotation files into distinct folders.

    Args:
        source_folder (str): Path to the source folder containing both images and annotations.
        images_folder (str): Path to the destination folder for images.
        annotations_folder (str): Path to the destination folder for annotations.
    """

    # Download from Roboflow
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="")
    # project = rf.workspace("augmented-startups").project("vehicle-registration-plates-trudk")
    # dataset = project.version(1).download("voc")

    # Ensure the destination folders exist; create them if they don't
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # Loop through files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a JPG image
        if filename.lower().endswith(".jpg"):
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(images_folder, filename)
            # Move the JPG file to the images folder
            shutil.move(source_file_path, destination_file_path)
        # Check if the file is an XML annotation file
        elif filename.lower().endswith(".xml"):
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(annotations_folder, filename)
            # Move the XML file to the annotations folder
            shutil.move(source_file_path, destination_file_path)

    logger.info("Separation completed.")


def show_imgs(data: list):
    """
    Display multiple images with bounding boxes.

    Args:
        data: A list of lists. Each inner list contains: [title, image, original_bbox, predicted_bbox]
    """

    num_imgs = len(data)
    fig, axes = plt.subplots(1, num_imgs, figsize=(15, 5 * num_imgs))

    # If there's only one image, axes won't be a list, so we wrap it in a list for consistency
    if num_imgs == 1:
        axes = [axes]

    for ax, (title, image, orig_bbox, pred_bbox) in zip(axes, data):
        # Handle torch.Tensor
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                image = image.cpu()  # Move tensor to CPU if it's on CUDA

            if image.dim() == 4:  # If a batch of images
                image = image[0]  # Take the first image

            img = image.detach().permute(1, 2, 0).numpy()  # permute to HWC layout
        # Handle PIL.Image
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        # If the image values are between 0 and 1, scale them to [0, 255]
        # If the image values are between 0 and 1, scale them to [0, 255]
        if isinstance(img, np.ndarray) and img.max() <= 1.0:
            img = img * 255

        ax.imshow(
            img.astype(int),
            cmap="gray" if len(img.shape) == 2 or img.shape[2] == 1 else None,
        )

        # Draw the original bounding box
        x, y, x1, y1 = orig_bbox
        if isinstance(orig_bbox, torch.Tensor):
            x, y, x1, y1 = orig_bbox.cpu().numpy()
        rect_orig = patches.Rectangle(
            (x, y), x1 - x, y1 - y, linewidth=2, edgecolor="b", facecolor="none"
        )
        ax.add_patch(rect_orig)

        # Draw the predicted bounding box
        x, y, x1, y1 = pred_bbox
        if isinstance(pred_bbox, torch.Tensor):
            x, y, x1, y1 = pred_bbox.cpu().numpy()
        rect_pred = patches.Rectangle(
            (x, y), x1 - x, y1 - y, linewidth=2, edgecolor="y", facecolor="none"
        )
        ax.add_patch(rect_pred)

        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def split_data(
    training_dataset: LPRTrainingDatasetProcessed, test_to_train_percent: float
):
    """Split the training dataset into training and test sets.

    Args:
        training_dataset (LPRTrainingDatasetProcessed): training dataset object
        test_to_train_percent (float): float percentage of test to train data
    """
    resized_images = [Image.fromarray(i[0]) for i in training_dataset.training_data]
    numpy_data = np.array([np.array(img.resize((416, 416))) for img in resized_images])
    X = torch.Tensor(numpy_data)

    X = X / 255.0

    numpy_bbox = np.array(
        [
            (
                [
                    416 / resized_images[i].width,
                    416 / resized_images[i].height,
                    416 / resized_images[i].width,
                    416 / resized_images[i].height,
                ]
                * np.array(data[1])
            )
            for i, data in enumerate(training_dataset.training_data)
        ]
    )
    y = torch.Tensor(numpy_bbox)

    val_size = int(len(X) * test_to_train_percent)

    # create test and training splits
    training_dataset.train_X = X[:-val_size]
    training_dataset.train_Y = y[:-val_size]

    training_dataset.test_X = X[-val_size:]
    training_dataset.test_y = y[-val_size:]

    logger.info(len(training_dataset.train_X))
    logger.info(len(training_dataset.test_X))

    demo_arr = []
    for i in range(5):
        demo_arr.append(
            [
                f"Image #{i}",
                Image.fromarray(training_dataset.training_data[i][0]),
                np.array(training_dataset.training_data[i][1]),
                [0.2, 0.4, 0.2, 0.4] * np.array(training_dataset.training_data[i][1]),
            ]
        )
    show_imgs(demo_arr)
    demo_arr = []
    for i in range(5):
        orig_img = Image.fromarray(training_dataset.training_data[i][0])
        demo_arr.append([f"Image #{i}", numpy_data[i], numpy_bbox[i], [10, 20, 30, 40]])
    show_imgs(demo_arr)
