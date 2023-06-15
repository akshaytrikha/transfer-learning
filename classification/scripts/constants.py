from pathlib import Path
import os

MODEL_NAME = "HiFiSCDS Classification 23_06_14"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameters
NUM_BATCHES = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 9 

TRAIN_DIR = Path("./data/train/")
DEV_DIR = Path("./data/dev/")
TEST_DIR = Path("./data/test/")
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
