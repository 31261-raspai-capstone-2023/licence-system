import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def parse_annotations(directory):
    """
    Parse XML annotations in the given directory.
    Return structured data with filenames and associated bounding boxes.
    """
    annotations = []
    
    for xml_file in os.listdir(directory):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(directory, xml_file))
            root = tree.getroot()

            image_data = {
                "filename": root.find("filename").text,
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "objects": []
            }

            for obj in root.findall("object"):
                obj_data = {
                    "name": obj.find("name").text,
                    "xmin": int(obj.find("bndbox/xmin").text),
                    "ymin": int(obj.find("bndbox/ymin").text),
                    "xmax": int(obj.find("bndbox/xmax").text),
                    "ymax": int(obj.find("bndbox/ymax").text)
                }
                image_data["objects"].append(obj_data)
                
            annotations.append(image_data)
    
    return annotations

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the image.
    Resize to 256x256 and normalize pixel values.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 256x256
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize the pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    return img_normalized

def adjust_bounding_boxes(original_dims, target_dims, bounding_box):
    """
    Adjust bounding box coordinates based on image resizing.
    """
    original_width, original_height = original_dims
    target_width, target_height = target_dims
    
    # Calculate scaling factors for width and height
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    
    # Adjust bounding box coordinates
    adjusted_box = {
        "xmin": int(bounding_box["xmin"] * width_scale),
        "ymin": int(bounding_box["ymin"] * height_scale),
        "xmax": int(bounding_box["xmax"] * width_scale),
        "ymax": int(bounding_box["ymax"] * height_scale)
    }
    
    return adjusted_box

from torch.utils.data import Dataset, DataLoader

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
        bbox = adjust_bounding_boxes((image_data["width"], image_data["height"]), 
                                     (256, 256), 
                                     image_data["objects"][0]) # Assuming one plate per image

        sample = {'image': image, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample
