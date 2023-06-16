import re
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


IMAGE_FILE_REGEX = re.compile("[-\w]+\.(?:png|jpg|jpeg|tif|tiff|bmp)")
"""allowed image file extensions"""


def pil_loader(path: str, mode: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


def find_classes(class_map_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """Loads classes from class_map_path which contains data in the format:
    {"0": "class_1",
     "1": "class_2"
     .
     "n": "class_n"
    }

    Returns:
        classes List[str]: list containing class names
        idx_to_class Dict[int, str]: pixel value to class name map
        class_to_idx Dict[str, idx]: class name to pixel value map
    """
    with open(class_map_path, "rb") as open_file:
        idx_to_class = json.load(open_file)
        idx_to_class = {
            int(k): v for k, v in idx_to_class.items()
        }  # convert keys to int

    classes = idx_to_class.values()

    return classes, idx_to_class, {v: k for k, v in idx_to_class.items()}


class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        class_map_path: Path,
        image_transform: transforms.Compose,
        device: torch.device,
    ):
        self.classes, self.idx_to_class, self.class_to_idx = find_classes(
            class_map_path
        )

        self.image_paths = []
        self.labels = []
        self.classes = []

        for filename in os.listdir(data_dir):
            # folder names are classes
            if os.path.isdir(data_dir / filename):
                # get paths to images
                split_filepaths = os.listdir(data_dir / filename / split)

                # append paths to images, corresponding labels
                self.image_paths += split_filepaths
                self.labels += [self.class_to_idx[filename]] * len(split_filepaths)

        self.image_transform = image_transform
        self.device = device

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image, mask & filename into memory
        image = pil_loader(self.image_paths[idx], mode="RGB")

        # apply transformations to image & mask
        if self.image_transform is not None:
            image = self.image_transform(image)

        return (image.to(self.device), self.image_paths[idx].name)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    device: torch.device,
    image_transform: transforms.Compose,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """create dataloaders from corresponding directories

    Args:
        data_dir (Path): directory containing all data
        transform (torchvision.transforms.Compose) to perform on training and testing data.
        batch_size (int) number of samples per batch in each of the DataLoaders.
        num_workers (int) for number of workers per DataLoader.

    Returns:
        Tuple[train_dataloader, dev_dataloader, test_dataloader]
    """
    class_map_path = data_dir / "defect_map.json"
    # read data into create datasets
    train_data = ClassificationDataset(
        data_dir, "train", class_map_path, image_transform, device
    )
    dev_data = ClassificationDataset(
        data_dir, "dev", class_map_path, image_transform, device
    )
    test_data = ClassificationDataset(
        data_dir, "test", class_map_path, image_transform, device
    )

    # convert images into dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    dev_dataloader = DataLoader(
        dev_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )

    return train_dataloader, dev_dataloader, test_dataloader
