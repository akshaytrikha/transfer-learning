from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import os
import re
from PIL import Image
import json


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
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}  # convert keys to int

    classes = idx_to_class.values()

    return classes, idx_to_class, {v: k for k, v in idx_to_class.items()}


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dir_path: Path,
        class_map_path: Path,
        image_transform: transforms.Compose,
        mask_transform: transforms.Compose,
        device: torch.device,
    ):
        # store the image & mask filepaths, augmentation transforms
        self.image_paths = [
            dir_path / "Images" / x
            for x in os.listdir(dir_path / "Images")
            if IMAGE_FILE_REGEX.match(x)
        ]
        self.mask_paths = [
            dir_path / "Segmentations" / x
            for x in os.listdir(dir_path / "Segmentations")
            if IMAGE_FILE_REGEX.match(x)
        ]
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.classes, self.idx_to_class, self.class_to_idx = find_classes(class_map_path)
        self.device = device

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image, mask & filename into memory
        image = pil_loader(self.image_paths[idx], mode="RGB")
        mask = pil_loader(self.mask_paths[idx], mode="L")

        # apply transformations to image & mask
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            # ToTensor divides by 256
            mask *= 256
            mask = mask.long()

        return (image.to(self.device), mask.to(self.device), self.image_paths[idx].name)

    def get_images(self, idx):
        image, mask = self[idx]
        T = transforms.ToPILImage()
        return T(image), T(mask)


def create_dataloaders(
    train_dir: Path,
    dev_dir: Path,
    test_dir: Path,
    batch_size: int,
    device: torch.device,
    image_transform: transforms.Compose,
    mask_transform: transforms.Compose,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """create dataloaders from corresponding directories

    Args:
        train_dir (Path): training directory
        dev_dir (Path): validation directory
        test_dir (Path) test directory
        batch_size (int) number of samples per batch in each of the DataLoaders
        device (torch.device): "cpu", "cuda", or "mps"
        image_transform (torchvision.transforms.Compose) to perform on training and testing input data
        mask_transform (torchvision.transforms.Compose) to perform on training and testing targets
        num_workers (int) number of workers per DataLoader
    Returns:
        Tuple[train_dataloader, dev_dataloader, test_dataloader]
    """
    # read data into create datasets
    class_map_path = train_dir.parent / "defect_map.json"
    train_data = SegmentationDataset(
        train_dir,
        class_map_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
        device=device,
    )
    dev_data = SegmentationDataset(
        dev_dir,
        class_map_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
        device=device,
    )
    test_data = SegmentationDataset(
        test_dir,
        class_map_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
        device=device,
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
