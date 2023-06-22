import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchmetrics
from pathlib import Path
import wandb

import data, utils, engine
from constants import *

run = wandb.init(
    project="Defect Classification",
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
    },
)

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # Apple Silicon
    device = "mps"
else:
    device = "cpu"

# ------------------ Data ------------------
image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ]
)

train_dataloader, dev_dataloader, test_dataloader = data.create_dataloaders(
    data_dir=Path("./data"),
    batch_size=BATCH_SIZE,
    device=device,
    image_transform=image_transform,
)

# ------------------ Model ------------------
# instantiate pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

torch.manual_seed = RANDOM_SEED

# modify classifier layer for number of classes
model.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=CLASSIFIER_IN_FEATURES, out_features=NUM_CLASSES),
).to(device)

# ------------------ Training ------------------
# define loss, optimizer, accuracy
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)

Path(f"./models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)
Path(f"./models/{MODEL_NAME}/test_predictions").mkdir(parents=True, exist_ok=True)

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

# run test loop
test_loss, test_acc = engine.test_step(
    model, test_dataloader, loss_fn, accuracy_fn, device
)
