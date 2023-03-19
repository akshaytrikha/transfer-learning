import sys
import os
from pathlib import Path
import torch
from torch import nn

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
    # PyTorch expected file extension
    model_path = Path(f"./models/{model_name}/{model_name}.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)


def load_model(model_name: str) -> nn.Module:
    # PyTorch expected file extension
    model_path = Path(f"./models/{model_name}/{model_name}.pth")

    return torch.load(f=model_path)
