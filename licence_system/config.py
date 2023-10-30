# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Other settings
IMAGE_SIZE = (256, 256)
ANNOTATION_DIRECTORY = "./data/kaggle-dataset-433/annotations"
IMAGE_DIRECTORY = "./data/kaggle-dataset-433/images"

TESTING_IMAGES_SIZE = 0.1

# Define the source folder containing JPG and XML files
SOURCE_FOLDER = 'data/roboflow/train'
# Define the destination folders
IMAGES_FOLDER = 'data/roboflow/images'
ANNOTATIONS_FOLDER = 'data/roboflow/annotations'

# percent of data we want to use for testing vs training
TEST_TRAIN_SPLIT_PERCENT = 0.1