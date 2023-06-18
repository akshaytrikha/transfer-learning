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