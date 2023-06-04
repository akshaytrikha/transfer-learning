import sys
import os
from pathlib import Path
import torch
from torch import nn
import numpy as np
from typing import Union

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch


def load_url(url, model_dir="./pretrained", map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split("/")[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def save_model(model: nn.Module, model_name: str):
    """saves a model to ./models/{model_name}/{model_name}.pth"""
    # PyTorch expected file extension
    model_path = Path(f"./models/{model_name}/{model_name}.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)


def load_model(model_name: str) -> nn.Module:
    """loads a model from ./models/{model_name}/{model_name}.pth"""
    # PyTorch expected file extension
    model_path = Path(f"./models/{model_name}/{model_name}.pth")

    return torch.load(f=model_path)


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
