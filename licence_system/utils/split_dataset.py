"""
This file defines the functions for splitting the dataset

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import argparse
import os
import random
import shutil


def split_dataset(data_dir: str, train_ratio: float = 0.8):
    """Function for splitting the dataset

    Args:
        data_dir (str): directory containing the training data
        train_ratio (float, optional): ratio between training and inference split. Defaults to 0.8.
    """
    image_dir = os.path.join(data_dir, "images")
    annotation_dir = os.path.join(data_dir, "annotations")

    if not os.path.exists(image_dir) or not os.path.exists(annotation_dir):
        print(
            f"Error: 'images/' or 'annotations/' directory not found inside '{data_dir}'."
        )
        return

    image_files = [
        f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]

    random.shuffle(image_files)

    train_size = int(len(image_files) * train_ratio)
    train_images = image_files[:train_size]
    test_images = image_files[train_size:]

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "annotations"), exist_ok=True)

    for img in train_images:
        shutil.move(
            os.path.join(image_dir, img),
            os.path.join(os.path.join(train_dir, "images"), img),
        )
        xml_file = (
            img.replace(".png", ".xml").replace(".jpg", ".xml").replace(".jpeg", ".xml")
        )
        if xml_file in xml_files:
            shutil.move(
                os.path.join(annotation_dir, xml_file),
                os.path.join(os.path.join(train_dir, "annotations"), xml_file),
            )

    for img in test_images:
        shutil.move(
            os.path.join(image_dir, img),
            os.path.join(os.path.join(test_dir, "images"), img),
        )
        xml_file = (
            img.replace(".png", ".xml").replace(".jpg", ".xml").replace(".jpeg", ".xml")
        )
        if xml_file in xml_files:
            shutil.move(
                os.path.join(annotation_dir, xml_file),
                os.path.join(os.path.join(test_dir, "annotations"), xml_file),
            )

    # Remove original 'images/' and 'annotations/' directories if empty
    for dir_name in ["images", "annotations"]:
        original_dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(original_dir_path) and not os.listdir(original_dir_path):
            os.rmdir(original_dir_path)

    print(f"Split completed. testing={len(train_images)} training={len(test_images)}")


def main():
    """Main function
    """
    parser = argparse.ArgumentParser(
        description="Split a dataset into training and testing sets."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default is 0.8 or 80%)",
    )
    parser.add_argument(
        "data_dir", type=str, help="Directory containing images and annotations"
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    train_ratio = args.train_ratio

    if not os.path.exists(data_dir):
        print(f"Error: The specified directory '{data_dir}' does not exist.")
        return

    split_dataset(data_dir, train_ratio)


if __name__ == "__main__":
    main()
