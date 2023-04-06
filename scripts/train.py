import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchmetrics
from torchinfo import summary
from pathlib import Path
import os
import pandas as pd
import data, model, engine, utils
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


MODEL_NAME = "Universal Resnet50 23_03_18 #1"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameterse
NUM_BATCHES = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
NUM_CLASSES = 1  # background + foreground

TRAIN_DIR = Path("./data/train/")
DEV_DIR = Path("./data/dev/")
TEST_DIR = Path("./data/test/")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

# ------------------ Data ------------------
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

(
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    class_names,
) = data.create_dataloaders(
    train_dir=TRAIN_DIR,
    dev_dir=DEV_DIR,
    test_dir=TEST_DIR,
    batch_size=NUM_BATCHES,
    device=device,
    image_transform=image_transform,
    mask_transform=mask_transform,
)
NUM_CLASSES = len(class_names)

# ------------------ Model ------------------
# instantiate DeepLabV3 model pretrained with resnet50 weights
weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device)

# modify classifier layer for desired number of classes
model.classifier = DeepLabHead(2048, NUM_CLASSES)

model.to(device)
torch.manual_seed = RANDOM_SEED

# ------------------ Training ------------------
# define loss, optimizer, accuracy
loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
# accuracy_fn = torchmetrics.Accuracy(task="binary", num_classes=NUM_CLASSES - 1)
accuracy_fn = torchmetrics.JaccardIndex(task="binary", num_classes=1)

# train model
training_results = engine.train(
    model,
    train_dataloader,
    dev_dataloader,
    loss_fn,
    optimizer,
    accuracy_fn,
    NUM_EPOCHS,
    device,
)


# save model
utils.save_model(model, MODEL_NAME)

# save training results
pd.DataFrame(training_results).to_csv(
    Path(f"./models/{MODEL_NAME}/{MODEL_NAME}_training.csv"), index_label="epoch"
)
