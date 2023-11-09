## show imgs
print("init show images function")
import os
import cv2
import pdb
import pathlib
import torch
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.nn.functional as F

TESTING_IMAGES_SIZE = 0.1


def show_imgs(data):
    """
    Display multiple images with bounding boxes.

    Parameters:
    - data: A list of lists. Each inner list contains:
        [title, image, original_bbox, predicted_bbox]
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


## model definition
print("init LPLocalNet model")


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
        print("conv 1: ", t)
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


## evaulation
print("load model")
net = LPLocalNet().to("cpu")
net.load_state_dict(
    torch.load(
        "licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth",
        map_location="cpu",
    )
)
net.eval()
print("Model loaded", net)


def close_enough(num1, num2):
    a = num1
    b = num2
    return abs(a - b) < ACCEPTABLE_DISTANCE


demo_arr = []

print("loading images")
images = []
IMAGE_PATH = "inference-images"
for file in os.listdir(IMAGE_PATH):
    if file.endswith(".png") or file.endswith(".jpeg") or file.endswith("jpg"):
        images.append(os.path.join(IMAGE_PATH, file))
selected_img = images[2]
with Image.open(selected_img).convert("L") as img:
    width, height = img.size  # Get dimensions
    # img.show()
    left = width / 1.2
    top = height / 1.2
    img = img.crop((left / 4, top / 4, left, top))
    resized_img = img.resize((416, 416))
    numpy_data = np.array(resized_img)
    X = torch.Tensor(numpy_data)
    X = X / 255.0
    with torch.no_grad():
        # db.set_trace()
        net_out = net(X.to("cpu").view(-1, 1, 416, 416))[0]
        predicted_bbox = net_out
        demo_arr.append(
            [
                "Image #{}".format(0),
                X.to("cpu").view(-1, 1, 416, 416),
                (0, 0, 0, 0),
                predicted_bbox,
            ]
        )

    # displaying output
    print("displaying output")
    if len(demo_arr) > 0:
        bbox = predicted_bbox.detach()
        print("bounding box: ", bbox)
        imgcp = resized_img.copy()
        imgcp_draw = ImageDraw.Draw(imgcp)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        imgcp_draw.rectangle([x1, y1, x2, y2], fill=None, outline="white", width=2)
        imgcp.show()

# show_imgs(demo_arr)

torch.cuda.empty_cache()
