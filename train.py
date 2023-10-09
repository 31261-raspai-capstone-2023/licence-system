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

print("Image Data:", dataset[0].get("bbox"))

# DataLoader can be used for batching and shuffling the dataset during training
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)