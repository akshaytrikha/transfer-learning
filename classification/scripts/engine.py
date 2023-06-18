import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from typing import Tuple
from constants import *
import pandas as pd
import wandb

from utils import EarlyStopper


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
        dataloader (DataLoader): training dataloader
        loss_fn (nn.Module): loss function to minimize
        optimizer (Optimizer): optimization function
        accuracy_fn (fn): accuracy function
        device (torch.device): "cpu", "cuda", or "mps"

    Returns:
        Tuple[train_loss (float), train_accuracy (float)]
    """
    train_loss, train_accuracy = 0, 0

    for batch_i, (X_train, y_train, filenames) in enumerate(dataloader):
        # training mode
        model.train()

        # forward pass
        # outputs are in the format [logit(true)] for each sample
        # logit = log(unnormalized probability)
        outputs = model(X_train.to(device))

        # calculate loss & accuracy
        loss = loss_fn(outputs, y_train)
        accuracy = accuracy_fn(outputs, y_train)
        
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
        dataloader (DataLoader): validation dataloader
        loss_fn (nn.Module): loss function
        accuracy_fn (fn): accuracy function
        device (torch.device): "cpu", "cuda", or "mps"

    Returns:
        Tuple[train_loss (float), train_accuracy (float)]
    """
    dev_loss, dev_accuracy = 0, 0

    for batch_i, (X_dev, y_dev, filenames) in enumerate(dataloader):
        # faster inferences, no autograd
        with torch.inference_mode():
            # forward pass
            # outputs are in the format [logit(true)] for each sample
            # logit = log(unnormalized probability)
            outputs = model(X_dev.to(device))

            # calculate loss & accuracy
            loss = loss_fn(outputs, y_dev)
            accuracy = accuracy_fn(outputs, y_dev)

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
    results = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}
    early_stopper = EarlyStopper(patience=3, min_delta=0.001)

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

        # early stopping
        if early_stopper.early_stop(dev_loss):
            break
    return results


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Accuracy,
    device: torch.device,
):
    test_loss, test_acc = 0, 0

    results_df = pd.DataFrame(columns=["filename", "ground_truth", "prediction"])

    for batch_i, (X_test, y_test, filenames) in enumerate(tqdm(dataloader)):
        # faster inferences, no autograd
        with torch.inference_mode():
            # forward pass
            # outputs are in the format [logit(true)] for each sample
            # logit = log(unnormalized probability)
            outputs = model(X_test.to(device))
            predictions = nn.Sigmoid(outputs) > 0.5

            # calculate loss & accuracy
            loss = loss_fn(outputs, y_test)
            accuracy = accuracy_fn(outputs, y_test)

            test_loss += loss.detach().cpu().numpy()
            test_acc += accuracy.detach().numpy()

            # save predictions as .csv
            results_df["filename"] += filenames
            results_df["ground_truth"] += [dataloader.dataset.idx_to_class[x] for x in X_test]
            results_df["prediction"] += [dataloader.dataset.idx_to_class[x] for x in outputs]

    # average loss & accuracy across batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    # save results
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )

    print(f"Test loss: {test_loss:.5f} Test Accuracy: {test_acc:.5f}")

    results_df.to_csv(Path(f"./models/{MODEL_NAME}/{MODEL_NAME}_testing.csv"),)

    return test_loss, test_acc
