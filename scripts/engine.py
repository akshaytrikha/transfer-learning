import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from typing import Tuple
from einops import rearrange
from constants import *
import pandas as pd
import wandb


def step_shape_helper(
    outputs: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    loss_fn: nn.Module,
    accuracy_fn: Accuracy,
):
    """formats tensors in correct shape and calculates loss & accuracy
    Args:
        outputs (torch.Tensor): model outputs with shape (batch, classes, height, width)
        target (torch.Tensor): ground truth training data with shape (batch, 1, height, width)
        batch_size (int): length of current batch
        loss_fn (nn.Module): function used to calculate loss
        accuracy_fn (Accuracy): function used to calculate accuracy

    Returns:
        loss (torch.Tensor): calulcated loss
        accuracy (torch.Tensor): calculated accuracy
    """
    # modify outputs to be in format [logit(true)] for each sample
    # and match same dims as y_train
    # new shape: [32, 1, height, width]
    outputs = rearrange(
        outputs,
        "bat cla height width -> bat cla height width",
        bat=batch_size,
        cla=NUM_CLASSES,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
    )

    target = rearrange(
        target,
        "bat cla height width -> bat cla height width",
        bat=batch_size,
        cla=1,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
    )

    # calculate pytorch loss
    loss = loss_fn(
        outputs,
        rearrange(target, "bat cla height width -> (bat cla) height width"),
    )

    # calculate accuracy
    # binarize predictions by taking softmax
    outputs = nn.functional.softmax(outputs, dim=1)[:, 1, :, :]
    outputs = rearrange(
        outputs,
        "bat height width -> bat 1 height width",
        bat=batch_size,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
    )

    y_pred = (outputs > 0.5).type(torch.int32)
    # new shape: [32, 1, height, width]

    return (loss, accuracy_fn(y_pred.to("cpu"), target.to("cpu")))


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

        # calculate loss & accuracy
        loss, accuracy = step_shape_helper(
            outputs=outputs,
            target=y_train,
            batch_size=len(X_train),
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
        )

        train_loss += loss.detach().cpu().numpy()
        train_accuracy += accuracy.detach().numpy()

        # clear optimizer accumulation
        optimizer.zero_grad()

        # calculate gradients via backpropagation
        loss.backward()

        # update parameters
        optimizer.step()

    # average loss & accuracy across batch
    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    print(f"Train loss: {train_loss:.5f} Train Accuracy: {train_accuracy:.5f}")

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

    for batch_i, (X_dev, y_dev) in enumerate(dataloader):
        # faster inferences, no autograd
        with torch.inference_mode():
            # forward pass
            # outputs are in the format [logit(false),logit(true)] for each sample
            # logit = log(unnormalized probability)
            outputs = model(X_dev.to(device))["out"]

            # calculate loss & accuracy
            loss, accuracy = step_shape_helper(
                outputs=outputs,
                target=y_dev,
                batch_size=len(X_dev),
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
            )

            dev_loss += loss.detach().cpu().numpy()
            dev_accuracy += accuracy.detach().numpy()

    # average loss & accuracy across batch
    dev_loss /= len(dataloader)
    dev_accuracy /= len(dataloader)

    print(f"Dev loss: {dev_loss:.5f} Dev Accuracy: {dev_accuracy:.5f}")

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

    for epoch in tqdm(range(epochs)):
        print(f"-------Epoch: {epoch}-------")
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, accuracy_fn, device
        )

        dev_loss, dev_acc = dev_step(
            model, dev_dataloader, loss_fn, accuracy_fn, device
        )

        # update & save results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["dev_loss"].append(dev_loss)
        results["dev_acc"].append(dev_acc)

        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )

        # save training results
        pd.DataFrame(results).to_csv(
            Path(f"./models/{MODEL_NAME}/{MODEL_NAME}_training.csv"),
            index_label="epoch",
        )

    return results
