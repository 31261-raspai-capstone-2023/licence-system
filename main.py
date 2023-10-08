import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.centernet import SimpleCenterNet
from torch.utils.data import DataLoader, Dataset, random_split
from utils.data_loader import (
    adjust_bounding_boxes,
    load_and_preprocess_image,
    parse_annotations,
)

# Configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
IMAGE_DIRECTORY = "./data/kaggle-dataset-433/train/images"
ANNOTATION_DIRECTORY = "./data/kaggle-dataset-433/train/annotations"

TEST_IMAGE_DIRECTORY = "./data/kaggle-dataset-433/test/images"
TEST_ANNOTATION_DIRECTORY = "./data/kaggle-dataset-433/test/annotations"

# Check for CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = SimpleCenterNet().to(device)

# Loss and optimizer
criterion_center = nn.MSELoss()
criterion_reg = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Dataset and DataLoader
class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.annotations = parse_annotations(annotation_dir)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_data = self.annotations[idx]
        image_path = os.path.join(self.image_dir, image_data["filename"])
        image = load_and_preprocess_image(image_path)
        bbox = adjust_bounding_boxes(
            (image_data["width"], image_data["height"]),
            (256, 256),
            image_data["objects"][0],
        )  # Assuming one plate per image

        sample = {"image": image, "bbox": bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_heatmap(width, height, center, sigma=10):
    """
    Generate a heatmap with a Gaussian distribution around the provided center.

    Parameters:
    - width, height: Dimensions of the heatmap.
    - center: Center coordinates as [x, y].
    - sigma: Standard deviation for the Gaussian distribution.

    Returns:
    - heatmap: A heatmap with a Gaussian distribution around the center.
    """
    x, y = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))
    heatmap = torch.exp(
        -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2)
    )
    return heatmap


# Split dataset into train and validation sets
dataset = LicensePlateDataset(IMAGE_DIRECTORY, ANNOTATION_DIRECTORY)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        print(batch["bbox"])

        for image in batch["image"]:
            image_np = image.permute(1, 2, 0).cpu().numpy()
            cv2.imshow("Image", image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        images, bboxes = batch["image"].to(device), batch["bbox"].to(device)
        optimizer.zero_grad()
        center_preds, regression_preds = model(images)
        loss_center = criterion_center(center_preds, center_ground_truth)
        loss_reg = criterion_reg(regression_preds, bbox_ground_truth)
        loss = loss_center + loss_reg
        loss.backward()
        optimizer.step()

    # Validation loop (pseudo code)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, bboxes = batch["image"].to(device), batch["bbox"].to(device)
            center_preds, regression_preds = model(images)
            val_loss_center = criterion_center(center_preds, ...)
            val_loss_reg = criterion_reg(regression_preds, bboxes)
            val_loss = val_loss_center + val_loss_reg

# Save the trained model
torch.save(model.state_dict(), "models/saved_models/centernet.pth")


# Visualization
def visualize_results(image, bounding_box):
    """
    Visualize bounding boxes on the image.
    """
    color = (0, 255, 0)  # Green color for bounding box
    thickness = 2
    cv2.rectangle(
        image,
        (bounding_box["xmin"], bounding_box["ymin"]),
        (bounding_box["xmax"], bounding_box["ymax"]),
        color,
        thickness,
    )
    cv2.imshow("Result Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decode_regression_pred(center_pred, regression_pred):
    """
    Decode regression predictions into bounding box coordinates.

    Parameters:
    - center_pred: Predicted object center as [x, y].
    - regression_pred: Predicted regression outputs as [offset_xmin, offset_ymin, offset_xmax, offset_ymax].

    Returns:
    - bbox: Bounding box coordinates as {"xmin": value, "ymin": value, "xmax": value, "ymax": value}
    """
    center_x, center_y = center_pred
    offset_xmin, offset_ymin, offset_xmax, offset_ymax = regression_pred
    bbox = {
        "xmin": center_x - offset_xmin,
        "ymin": center_y - offset_ymin,
        "xmax": center_x + offset_xmax,
        "ymax": center_y + offset_ymax,
    }
    return bbox


# Testing and visualization
test_dataset = LicensePlateDataset(TEST_IMAGE_DIRECTORY, TEST_ANNOTATION_DIRECTORY)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()
with torch.no_grad():
    for batch in test_loader:
        image = batch["image"].to(device)
        center_pred, regression_pred = model(image)
        bbox = decode_regression_pred(regression_pred)
        visualize_results(image.cpu().numpy(), bbox)
