import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchmetrics
from torchsummary import summary
from pathlib import Path
import os
import pandas as pd
import data, model, engine, utils


MODEL_NAME = "Universal Resnet18 23_03_18 #1"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameterse
NUM_BATCHES = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_DIR = Path("./data/train/images")
DEV_DIR = Path("./data/dev/images")
TEST_DIR = Path("./data/test/images")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

# ------------------ Data ------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    transform=transform,
)
NUM_CLASSES = len(class_names)

# ------------------ Model ------------------
# instantiate pretrained resnet18 model
model = model.resnet_model(n_resnet_layers="resnet18", pretrained=True, device=device)

# modify classifier layer for desired number of classes
old_num_features = model.fc.in_features
model.fc = nn.Linear(old_num_features, NUM_CLASSES)

model.to(device)

breakpoint()

torch.manual_seed = RANDOM_SEED


# ------------------ Training ------------------
# define loss, optimizer, accuracy
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)

# # train model
# training_results = engine.train(
#     model,
#     train_dataloader,
#     dev_dataloader,
#     loss_fn,
#     optimizer,
#     accuracy_fn,
#     NUM_EPOCHS,
#     device,
# )

# # save model
# utils.save_model(model, MODEL_NAME)

# # save training results
# pd.DataFrame(training_results).to_csv(
#     Path(f"./models/{MODEL_NAME}/{MODEL_NAME}_training.csv"), index_label="epoch"
# )
