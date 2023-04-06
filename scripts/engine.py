import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from typing import Tuple


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Accuracy,
    device: torch.device,
) -> Tuple[float, float]:
    """performs one training iteration with a batch
    Args:
        model (nn.Module): model to be trained
        dataloader (DataLoader) training dataloader
        loss_fn (nn.Module) loss function to minimize
        optimizer (Optimizer) optimization function
        device (torch.device): "cpu" or "cuda"

    Returns:
        Tuple[train_loss (float), train_accuracy (float)]
    """
    train_loss, train_accuracy = 0, 0

    for batch, (X_train, y_train) in enumerate(dataloader):
        # training mode
        model.train()

        # forward pass
        outputs = model(X_train.to(device))["out"]

        breakpoint()

        # calculate loss & accuracy
        loss = loss_fn(outputs, y_train)
        train_loss += loss.cpu().detach().numpy()
        train_accuracy += accuracy_fn(outputs, y_train).detach().numpy()

        # clear optimizer accumulation
        optimizer.zero_grad()

        # calculate gradients via backpropagation
        loss.backward()

        # update parameters
        optimizer.step()

    # average loss & accuracy across batch
    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    print(
        f"Train loss: {train_loss:.5f}\
            Train Accuracy: {train_accuracy:.5f}"
    )

    return train_loss, train_accuracy


def dev_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Accuracy,
    device: torch.device,
) -> Tuple[float, float]:
    """performs one validation step with a batch
    Args:
        model (nn.Module): model to be trained
        dataloader (DataLoader) validation dataloader
        loss_fn (nn.Module) loss function
        device (torch.device): "cpu" or "cuda"

    Returns:
        Tuple[train_loss (float), train_accuracy (float)]
    """
    dev_loss, dev_accuracy = 0, 0

    for batch, (X_dev, y_dev) in enumerate(dataloader):
        # faster inferences, no autograd
        with torch.inference_mode():
            # forward pass
            outputs = model(X_dev.to(device))["out"]

            # calculate loss & accuracy
            loss = loss_fn(outputs, y_dev)
            dev_loss += loss.cpu().detach().numpy()
            dev_accuracy += accuracy_fn(outputs, y_dev).detach().numpy()

    # average loss & accuracy across batch
    dev_loss /= len(dataloader)
    dev_accuracy /= len(dataloader)

    print(
        f"Dev loss: {dev_loss:.5f}\
            Dev Accuracy: {dev_accuracy:.5f}"
    )

    return dev_loss, dev_accuracy


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accuracy_fn: Accuracy,
    epochs: int,
    device: torch.device,
):
    # for tracking learning
    results = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}

    # accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    for epoch in tqdm(range(epochs)):
        print(f"-------Epoch: {epoch}-------")
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, accuracy_fn, device
        )

        dev_loss, dev_acc = dev_step(
            model, dev_dataloader, loss_fn, accuracy_fn, device
        )

        # update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["dev_loss"].append(dev_loss)
        results["dev_acc"].append(dev_acc)

    return results
