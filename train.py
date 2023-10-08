import os
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from torchvision import transforms

class CustomALPRDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(self.annotation_dir, image_name.replace('.png', '.xml'))

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Parse the XML annotation file to extract bounding box coordinates
        # Load the XML annotation file
        root = ET.parse(annotation_path).getroot()
        
        # Iterate through the XML and extract bounding box coordinates
        for obj in root.findall('.//object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
        bounding_box_coordinates = (xmin, ymin, xmax, ymax)
    
        # Preprocess and transform the image (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        # Create a dictionary containing image and target information
        target = {
            "image": torch.tensor(image, dtype=torch.float32),
            "bbox": torch.tensor(bounding_box_coordinates, dtype=torch.float32),  # Replace with actual bounding box coordinates
            "labels": torch.tensor([1], dtype=torch.int64),  # Assuming there is only one class (license plate)
        }

        return target

# Define the transform for image preprocessing (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Create an instance of the custom dataset
dataset = CustomALPRDataset(image_dir='data/kaggle-dataset-433/train/images', annotation_dir='data/kaggle-dataset-433/train/annotations', transform=transform)

for sample in range(0, len(dataset)):
    print(dataset[sample])

# DataLoader can be used for batching and shuffling the dataset during training
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


# TRAINING
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
#from engine import train_one_epoch, evaluate
import utils

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define paths to your training and validation datasets
train_image_dir = 'path/to/train/images'
train_annotation_dir = 'path/to/train/annotations'
val_image_dir = 'path/to/val/images'
val_annotation_dir = 'path/to/val/annotations'

# Define the transform for image preprocessing
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create custom datasets for training and validation
train_dataset = CustomALPRDataset(train_image_dir, train_annotation_dir, transform)
val_dataset = CustomALPRDataset(val_image_dir, val_annotation_dir, transform)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

# Create a Faster R-CNN model with a ResNet-50 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # Background and license plate
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the GPU (if available)
model.to(device)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    # Training
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

    # Validation
    evaluate(model, val_loader, device=device)

# Save the trained model
torch.save(model.state_dict(), 'alpr_model.pth')
