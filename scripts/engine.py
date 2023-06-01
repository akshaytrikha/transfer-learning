import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from typing import Tuple
from einops import rearrange
from constants import *


def step_shape_helper(
    outputs: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
):
    """formats tensors in correct shape and calculates loss & accuracy"""
    pass


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

    for batch_i, (X_train, y_train) in enumerate(dataloader):
        # training mode
        model.train()

        # forward pass
        # outputs are in the format [logit(false),logit(true)] for each sample
        # logit = log(unnormalized probability)
        outputs = model(X_train.to(device))["out"]

        # modify outputs to be in format [logit(true)] for each sample
        # and match same dims as y_train
        # new shape: [32, 1, height, width]
        outputs = rearrange(
            outputs,
            "bat cla height width -> bat cla height width",
            bat=len(X_train),
            cla=NUM_CLASSES,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

        y_train = rearrange(
            y_train,
            "bat cla height width -> bat cla height width",
            bat=len(X_train),
            cla=1,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

        # calculate pytorch loss
        loss = loss_fn(
            outputs,
            rearrange(y_train, "bat cla height width -> (bat cla) height width"),
        )

        # calculate accuracy
        # binarize predictions by taking softmax
        outputs = nn.functional.softmax(outputs, dim=1)[:, 1, :, :]
        outputs = rearrange(
            outputs,
            "bat height width -> bat 1 height width",
            bat=len(X_train),
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

        y_pred = (outputs > 0.5).type(torch.int32)
        # new shape: [32, 1, height, width]

        # calculate loss & accuracy
        train_loss += loss.cpu().detach().numpy()
        train_accuracy += (
            accuracy_fn(y_pred.to("cpu"), y_train.to("cpu")).detach().numpy()
        )

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
