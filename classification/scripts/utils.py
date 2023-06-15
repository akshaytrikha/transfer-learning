from pathlib import Path
import torch
from torch import nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import numpy as np
from typing import Union, OrderedDict
from einops import rearrange
import cv2
from tqdm import tqdm
import wandb

from constants import *


def save_model(model: nn.Module, model_name: str):
    """saves a model to ./models/{model_name}/{model_name}.pth"""
    # PyTorch expected file extension
    model_path = Path(f"./models/{model_name}/{model_name}.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)


def my_load_state_dict(model: nn.Module, input_state_dict: OrderedDict):
    """loads input_state_dict into given model in place

    Args:
        model (nn.Module): model who's state_dict to update
        input_state_dict (OrderedDict): will update model's curret state_dict
    """
    own_state = model.state_dict()
    for name, param in input_state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """loads a model with state_dict from ./models/{model_name}/{model_name}.pth"""
    # ------------------ Model ------------------
    # instantiate DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_resnet50().to(device)

    # modify classifier layer for desired number of classes
    model.classifier = DeepLabHead(in_channels=2048, num_classes=NUM_CLASSES)

    new_state_dict_path = Path(f"../models/{model_name}/{model_name}.pth")
    new_state_dict = torch.load(new_state_dict_path, map_location=device)
    my_load_state_dict(model, new_state_dict)

    return model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: Union[torch.Tensor, np.ndarray]):
        """stops training if validation loss starts monotonically increasing patience number of times

        Args:
            validation_loss (torch.Tensor | np.ndarray) current validation loss

        Returns:
            stop (bool): True if training should stop, false otherwise
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


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
        y_pred (torch.Tensor): binarized predictions
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

    return (loss, accuracy_fn(y_pred.to("cpu"), target.to("cpu")), y_pred)


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Accuracy,
    device: torch.device,
):
    test_loss, test_acc = 0, 0

    for batch_i, (X_test, y_test, filenames) in enumerate(tqdm(dataloader)):
        # faster inferences, no autograd
        with torch.inference_mode():
            # forward pass
            # outputs are in the format [logit(false),logit(true)] for each sample
            # logit = log(unnormalized probability)
            outputs = model(X_test.to(device))["out"]

            # calculate loss & accuracy
            loss, accuracy, y_pred = step_shape_helper(
                outputs=outputs,
                target=y_test,
                batch_size=len(X_test),
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
            )

            test_loss += loss.detach().cpu().numpy()
            test_acc += accuracy.detach().numpy()

            # save predictions as .png
            for i, prediction in enumerate(y_pred):
                cv2.imwrite(
                    f"./models/{MODEL_NAME}/test_predictions/{filenames[i]}",
                    prediction.squeeze().cpu().numpy() * 100,
                )

    # average loss & accuracy across batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    # # save results
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )

    print(f"Test loss: {test_loss:.5f} Test Accuracy: {test_acc:.5f}")

    return test_loss, test_acc
