# Image Preparation Functions
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import numpy as np

class ImgProcSuite():
    def __init__(self):
        pass

    # Define the transform for preprocessing
    preprocess_transform = transforms.Compose([
        transforms.Grayscale(), # Convert image to grayscale
        transforms.Resize((416, 416)), # Resize image to 416x416
        transforms.ToTensor(), # Convert image to PyTorch tensor
    ])

    def read_image(self, file_path):
        '''Reads an image from file path and converts it to grayscale.'''
        img = Image.open(file_path).convert('L')
        return img

    def process_image_to_tensor(self, image, mean=None, std=None):
        '''Resizes and normalizes image using the given mean and std.'''
        
        # Contrast increase
        img_enhanced = self.enhance_contrast(image)

        img_tensor = self.preprocess_transform(img_enhanced)
        
        img_tensor = self.min_max_scale_tensor(img_tensor)

        img_tensor = self.quantize_tensor(img_tensor, 5)

        # Normalise by mean and std if provided
        # if mean is not None and std is not None:
        #     img_tensor = TF.normalize(img_tensor, [mean], [std])

        return img_tensor

    def enhance_contrast(self, image):
        return ImageOps.equalize(image)

    # def tensor_from_path(image_path, mean=None, std=None):
        #     '''Prepares an image for the model input.'''
        #     img = read_image(image_path)
        #     img = enhance_contrast(img)  # Enhance contrast after reading the image
        #     img_tensor = resize_and_normalize_image(img, mean, std)
        #     return img_tensor

    def quantize_tensor(self, tensor, levels=5):
        # Assuming the input tensor is already normalized to [0, 1]
        # If not, uncomment the following line:
        # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Calculate the step size for each quantization level
        step = 1.0 / (levels - 1)
        
        # Quantize the tensor
        quantized_tensor = torch.round(tensor / step) * step
        
        # Clamp the values to ensure they are within [0, 1]
        quantized_tensor = torch.clamp(quantized_tensor, 0, 1)
        
        return quantized_tensor

    def min_max_scale_tensor(self, img_tensor):
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        scaled_tensor = (img_tensor - min_val) / (max_val - min_val)
        return scaled_tensor

    def normalize_tensor(self, img_tensor, mean, std):
        # if mean is not None and std is not None:
        #     img_tensor = TF.normalize(img_tensor, [mean], [std])

        return img_tensor

    def compute_mean_std(self, image_tensors_list):
        '''Computes mean and std for a list of image tensors.'''
        all_images_tensor = torch.stack(image_tensors_list)
        mean = all_images_tensor.mean()
        std = all_images_tensor.std()
        return mean.item(), std.item()

    def resize_bbox(self, original_bbox, original_size, new_size=(416, 416)):
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        xmin, ymin, xmax, ymax = original_bbox
        new_bbox = (xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y)
        return new_bbox
