from pathlib import Path
import os

MODEL_NAME = "Universal Resnet50 23_05_31 #1"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameters
NUM_BATCHES = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # foreground + background

TRAIN_DIR = Path("./data/train/")
DEV_DIR = Path("./data/dev/")
TEST_DIR = Path("./data/test/")
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
