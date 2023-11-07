"""
This file defines the runner for training the model

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
from licence_system.config import (
    SOURCE_FOLDER,
    IMAGES_FOLDER,
    ANNOTATIONS_FOLDER,
    TESTING_IMAGES_SIZE,
    TEST_TRAIN_SPLIT_PERCENT,
    BATCH_SIZE,
    LEARNING_RATE,
)
from licence_system.utils.data_loader import separate_images_and_annotations, split_data
from licence_system.utils.logger import logger
from licence_system.utils.model_class import LPR_Training_Dataset_Processed
from licence_system.utils.trainer import get_accuracy, save_model, train

logger.info("Starting")
separate_images_and_annotations(SOURCE_FOLDER, IMAGES_FOLDER, ANNOTATIONS_FOLDER)

training_dataset = LPR_Training_Dataset_Processed(
    IMAGES_FOLDER, ANNOTATIONS_FOLDER, TESTING_IMAGES_SIZE
)
training_dataset.create_training_data()

split_data(training_dataset, TEST_TRAIN_SPLIT_PERCENT)
final_epoch = train(training_dataset)
accuracy = get_accuracy(training_dataset)

save_model(training_dataset, final_epoch + 1, BATCH_SIZE, LEARNING_RATE, accuracy)
logger.info("Finished")
